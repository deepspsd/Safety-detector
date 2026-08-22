"""
services/chew_monitor.py — Chewing & Clean-Shave Detection via MediaPipe
=========================================================================
Two best-effort heuristics using MediaPipe FaceMesh (468 facial landmarks).

REQ-017  Chewing detection
──────────────────────────
  Algorithm: track vertical distance between upper lip (#13) and lower lip
  (#14) over time per (camera_id, track_id). If the jaw opens and closes
  >= CHEW_CYCLE_COUNT times within CHEW_WINDOW_SEC → "chewing_detected".
  Rate limit: 1 check per CHEW_CHECK_INTERVAL_SEC per person.
  Alert status: "pending_review" (medium confidence — lighting dependent).

REQ-015  Clean-shave detection (best-effort)
────────────────────────────────────────────
  Algorithm: crop the lower-jaw region from FaceMesh bounding box.
  Compute the fraction of dark pixels (HSV V < threshold) in that region.
  If fraction > BEARD_PIXEL_THRESHOLD → possible facial hair.
  Rate limit: 1 check per SHAVE_CHECK_INTERVAL_SEC per person.
  Alert status: always "pending_review" (low confidence — skin-tone/lighting
  variability). Admin must manually confirm before action is taken.

Performance
───────────
  Both checks share one MediaPipe FaceMesh instance per camera (lazy-init).
  Checks are skipped if MEDIAPIPE_ENABLED=false in config or if mediapipe
  is not installed (degrades gracefully with a one-time warning log).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings

log = logging.getLogger("chew_monitor")

# ── Tuning constants ──────────────────────────────────────────────────────────
# Chewing detection
CHEW_CHECK_INTERVAL_SEC  = 10     # check at most once per 10s per person
CHEW_WINDOW_SEC          = 5.0   # look back this many seconds for cycles
CHEW_CYCLE_COUNT         = 3     # jaw must open+close ≥ this many times
CHEW_OPEN_THRESHOLD_PX   = 6     # minimum jaw-gap (px in normalised frame) to count as "open"

# Clean-shave detection
SHAVE_CHECK_INTERVAL_SEC = 60    # check at most once per 60s per person
BEARD_PIXEL_THRESHOLD    = 0.18  # dark pixel fraction that suggests facial hair

# MediaPipe landmark indices (FaceMesh 468-point model)
_UPPER_LIP = 13
_LOWER_LIP = 14
_JAW_LANDMARKS = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]

# ── Module-level state ────────────────────────────────────────────────────────
# Per-person jaw history: (camera_id, track_id) → list[(timestamp, gap_px)]
_jaw_history: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
# Per-person last-check timestamps
_chew_last_check:  Dict[Tuple[int, int], float] = {}
_shave_last_check: Dict[Tuple[int, int], float] = {}

# MediaPipe face-mesh instances per camera_id (lazy-created)
_face_meshes: Dict[int, object] = {}

_mediapipe_unavailable = False   # set to True on first import failure


# ── MediaPipe helper ──────────────────────────────────────────────────────────

def _get_face_mesh(camera_id: int):
    """Return (or lazily create) a FaceMesh instance for this camera."""
    global _mediapipe_unavailable
    if _mediapipe_unavailable or not settings.MEDIAPIPE_ENABLED:
        return None
    if camera_id in _face_meshes:
        return _face_meshes[camera_id]
    try:
        import mediapipe as mp
        fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _face_meshes[camera_id] = fm
        return fm
    except ImportError:
        _mediapipe_unavailable = True
        log.warning(
            "[ChewMonitor] mediapipe not installed — chewing/cleanshave checks disabled. "
            "Install with: pip install mediapipe>=0.10.0"
        )
        return None
    except Exception as exc:
        _mediapipe_unavailable = True
        log.warning("[ChewMonitor] MediaPipe init failed: %s", exc)
        return None


def _crop_person(frame: np.ndarray, bbox: List) -> Optional[np.ndarray]:
    """Safely crop person bounding box from frame."""
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 20 or y2 - y1 < 40:
        return None
    return frame[y1:y2, x1:x2]


# ── Chewing detection ─────────────────────────────────────────────────────────

def _count_jaw_cycles(history: List[Tuple[float, float]], now: float) -> int:
    """Count open→close transitions in the recent jaw-gap history."""
    recent = [(t, g) for t, g in history if now - t <= CHEW_WINDOW_SEC]
    if len(recent) < 4:
        return 0
    cycles = 0
    was_open = False
    for _, gap in recent:
        is_open = gap >= CHEW_OPEN_THRESHOLD_PX
        if was_open and not is_open:
            cycles += 1
        was_open = is_open
    return cycles


def _check_chewing(
    face_mesh,
    crop: np.ndarray,
    key: Tuple[int, int],
    now: float,
) -> bool:
    """Return True if chewing pattern detected."""
    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return False
        lm = results.multi_face_landmarks[0].landmark
        h, w = crop.shape[:2]
        upper_y = lm[_UPPER_LIP].y * h
        lower_y = lm[_LOWER_LIP].y * h
        gap = abs(lower_y - upper_y)

        history = _jaw_history.setdefault(key, [])
        history.append((now, gap))
        # Prune old samples (keep last 20s)
        _jaw_history[key] = [(t, g) for t, g in history if now - t <= 20.0]

        cycles = _count_jaw_cycles(_jaw_history[key], now)
        return cycles >= CHEW_CYCLE_COUNT
    except Exception as exc:
        log.debug("[ChewMonitor] chewing check error: %s", exc)
        return False


# ── Clean-shave detection ─────────────────────────────────────────────────────

def _check_facial_hair(
    face_mesh,
    crop: np.ndarray,
) -> bool:
    """
    Return True if the lower-jaw region has an unusually high dark-pixel
    fraction (possible beard / stubble). Low-confidence — pending_review only.
    """
    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return False
        lm = results.multi_face_landmarks[0].landmark
        h, w = crop.shape[:2]
        # Build jaw polygon from landmarks
        pts = np.array(
            [[int(lm[i].x * w), int(lm[i].y * h)] for i in _JAW_LANDMARKS],
            dtype=np.int32,
        )
        # Create mask for jaw region
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        # Restrict to lower half of face (avoid chin-cup false positives)
        mask[:h // 2, :] = 0

        jaw_pixels = crop[mask == 255]
        if len(jaw_pixels) < 100:
            return False
        # Convert to HSV; dark pixels = low Value channel
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        dark_in_jaw = np.sum((v_channel[mask == 255]) < 80)
        dark_fraction = dark_in_jaw / len(jaw_pixels)
        return dark_fraction > BEARD_PIXEL_THRESHOLD
    except Exception as exc:
        log.debug("[ChewMonitor] facial-hair check error: %s", exc)
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def process_frame(
    db,
    camera_id: int,
    frame: np.ndarray,
    persons: List[dict],
) -> None:
    """
    Run chewing and clean-shave checks for all tracked persons in the frame.
    Call from camera_manager._detection_loop() after person tracking.

    persons: list of dicts with at least {"bbox": [...], "track_id": int}
    """
    if not settings.MEDIAPIPE_ENABLED:
        return

    face_mesh = _get_face_mesh(camera_id)
    if face_mesh is None:
        return

    now = time.monotonic()

    for person in persons:
        track_id = person.get("track_id")
        bbox     = person.get("bbox")
        if track_id is None or not bbox:
            continue

        key = (camera_id, int(track_id))
        crop = _crop_person(frame, bbox)
        if crop is None:
            continue

        # ── Chewing check ──────────────────────────────────────────────────
        if now - _chew_last_check.get(key, 0) >= CHEW_CHECK_INTERVAL_SEC:
            _chew_last_check[key] = now
            if _check_chewing(face_mesh, crop, key, now):
                _fire_alert(
                    db=db,
                    camera_id=camera_id,
                    track_id=int(track_id),
                    detected_issue="Eating / chewing detected",
                    message=(
                        f"🍬 Chewing/eating detected on camera {camera_id} "
                        f"(track {track_id}). Hygiene violation — admin review required."
                    ),
                    confidence_tier="low",
                )

        # ── Clean-shave check ──────────────────────────────────────────────
        if now - _shave_last_check.get(key, 0) >= SHAVE_CHECK_INTERVAL_SEC:
            _shave_last_check[key] = now
            # Only check upper face crop (head area) for better accuracy
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            head_h = (y2 - y1) // 2
            head_crop = frame[y1: y1 + head_h, max(0, x1): min(frame.shape[1], x2)]
            if head_crop is not None and head_crop.size > 0:
                if _check_facial_hair(face_mesh, head_crop):
                    _fire_alert(
                        db=db,
                        camera_id=camera_id,
                        track_id=int(track_id),
                        detected_issue="Facial hair detected",
                        message=(
                            f"🪒 Possible facial hair detected on camera {camera_id} "
                            f"(track {track_id}). Admin visual review required "
                            f"(low-confidence heuristic)."
                        ),
                        confidence_tier="low",
                    )


def cleanup_camera(camera_id: int) -> None:
    """Remove all state for a stopped camera. Call from camera_manager.stop()."""
    keys = [k for k in _jaw_history if k[0] == camera_id]
    for k in keys:
        _jaw_history.pop(k, None)
        _chew_last_check.pop(k, None)
        _shave_last_check.pop(k, None)
    if camera_id in _face_meshes:
        try:
            _face_meshes.pop(camera_id).close()
        except Exception:
            pass


# ── Internal ──────────────────────────────────────────────────────────────────

def _fire_alert(
    db,
    camera_id: int,
    track_id: int,
    detected_issue: str,
    message: str,
    confidence_tier: str,
) -> None:
    try:
        from services.alert_service import save_alert
        from services.rule_engine import _get_rule_engine_user_id
        save_alert(
            db=db,
            user_id=_get_rule_engine_user_id(db),
            message=message,
            role="Safety Monitor",
            severity="medium",
            detected_issue=detected_issue,
            camera_id=camera_id,
            confidence_tier=confidence_tier,
        )
        log.info("[ChewMonitor] Alert fired: %s cam=%d track=%d", detected_issue, camera_id, track_id)
    except Exception as exc:
        log.error("[ChewMonitor] Alert save failed: %s", exc)
