"""
services/hand_motion.py — MediaPipe Hand Landmark Motion Tracker
================================================================
Uses the `hand_landmarks/hand_landmarker.task` MediaPipe model to detect
21 3D landmarks per hand per frame, then tracks their velocity to determine
whether a worker's hands are in motion.

Used by packing_monitor.py as the most accurate hand-movement signal.

Landmark IDs (relevant ones):
  0  — WRIST
  4  — THUMB_TIP
  8  — INDEX_FINGER_TIP
  12 — MIDDLE_FINGER_TIP
  16 — RING_FINGER_TIP
  20 — PINKY_TIP

Algorithm:
  • Extract (x, y) for 6 key landmarks per hand, per frame
  • Compute mean Euclidean displacement vs. previous frame
  • Displacement > _MOTION_THRESH_PX → hands moving
  • Track per person bbox: match detected hand to nearest person crop

Complexity: runs full frame through MediaPipe at ~30 FPS (CPU).
  Throttle to every 2–3 frames in the detection loop (called at 3 fps
  from camera_manager anyway, so no throttling needed).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("hand_motion")

# ── Key landmark indices to track ─────────────────────────────────────────────
_KEY_LM = [0, 4, 8, 12, 16, 20]  # wrist + 5 fingertips

# Motion threshold: mean per-landmark pixel displacement between frames.
# Below this → hands stationary.
_MOTION_THRESH_PX = 6.0

# ── Model path ────────────────────────────────────────────────────────────────
_TASK_FILE = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "portable_models_package", "hand_landmarks",
    "hand_landmarker.task",
)

# ── Singleton state ───────────────────────────────────────────────────────────
_detector = None
_model_loaded: bool = False

# Per-camera landmark history: { camera_id: { hand_idx: [(x,y), ...] } }
_prev_landmarks: Dict[int, Dict[int, List[Tuple[float, float]]]] = {}


def _load():
    """Lazy-load MediaPipe HandLandmarker. Thread-safe via Python GIL for flag."""
    global _detector, _model_loaded

    if _model_loaded:
        return _detector is not None

    _model_loaded = True  # don't retry

    task_path = os.path.abspath(_TASK_FILE)
    if not os.path.isfile(task_path):
        log.error("[HandMotion] hand_landmarker.task not found: %s", task_path)
        return False

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_opts = mp_python.BaseOptions(model_asset_path=task_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=4,          # up to 4 hands per frame (2 workers visible)
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        _detector = mp_vision.HandLandmarker.create_from_options(options)
        log.info("[HandMotion] MediaPipe HandLandmarker loaded OK")
        return True
    except Exception as exc:
        log.error("[HandMotion] Failed to load HandLandmarker: %s", exc)
        _detector = None
        return False


def detect_hand_motion(
    frame: np.ndarray,
    camera_id: int,
    person_bboxes: Optional[List[List[int]]] = None,
) -> List[Dict]:
    """
    Detect hand landmarks in `frame` and return per-hand motion status.

    Parameters
    ----------
    frame        : BGR numpy frame (full camera frame)
    camera_id    : used to maintain per-camera landmark history
    person_bboxes: optional list of [x1,y1,x2,y2] person boxes.
                   If provided, each hand result is tagged with the
                   nearest person bbox index.

    Returns
    -------
    List of dicts, one per detected hand:
    {
        "hand_index": int,
        "landmarks": [(x, y), ...],   # pixel coords, 21 points
        "displacement_px": float,     # mean displacement vs prev frame
        "is_moving": bool,
        "person_idx": int | None,     # index into person_bboxes, or None
    }
    Returns [] on model failure or no hands detected.
    """
    if not _load() or _detector is None:
        return []

    try:
        import mediapipe as mp
        from mediapipe.tasks.python.vision import RunningMode

        h, w = frame.shape[:2]
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _detector.detect(mp_image)
    except Exception as exc:
        log.debug("[HandMotion] detect error: %s", exc)
        return []

    if not result.hand_landmarks:
        # No hands — clear previous landmarks for this camera
        _prev_landmarks.pop(camera_id, None)
        return []

    cam_prev = _prev_landmarks.setdefault(camera_id, {})
    output = []

    for hand_idx, lm_list in enumerate(result.hand_landmarks):
        # Convert normalised → pixel
        px_lms: List[Tuple[float, float]] = [
            (lm.x * w, lm.y * h) for lm in lm_list
        ]

        # Key landmarks only for displacement
        key_pts = [px_lms[i] for i in _KEY_LM]

        prev_key = cam_prev.get(hand_idx)
        displacement = 0.0
        if prev_key is not None and len(prev_key) == len(key_pts):
            diffs = [
                ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
                for a, b in zip(key_pts, prev_key)
            ]
            displacement = float(np.mean(diffs))

        cam_prev[hand_idx] = key_pts
        is_moving = displacement >= _MOTION_THRESH_PX

        # Match to nearest person bbox (by wrist proximity)
        person_idx: Optional[int] = None
        if person_bboxes:
            wrist_x, wrist_y = px_lms[0]
            best_dist = float("inf")
            for pi, bbox in enumerate(person_bboxes):
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                d = ((wrist_x - cx)**2 + (wrist_y - cy)**2)**0.5
                if d < best_dist:
                    best_dist = d
                    person_idx = pi

        output.append({
            "hand_index": hand_idx,
            "landmarks": px_lms,
            "displacement_px": displacement,
            "is_moving": is_moving,
            "person_idx": person_idx,
        })

    # Clean up stale hand slots (fewer hands than previous frame)
    stale = [k for k in cam_prev if k >= len(result.hand_landmarks)]
    for k in stale:
        del cam_prev[k]

    return output


def hands_moving_for_person(
    frame: np.ndarray,
    camera_id: int,
    person_bboxes: List[List[int]],
) -> Dict[int, bool]:
    """
    Convenience wrapper: returns { person_bbox_index: is_moving } for each
    detected person.  Persons with no detected hand default to None (unknown).

    Used by packing_monitor to get per-worker hand motion status.
    """
    results = detect_hand_motion(frame, camera_id, person_bboxes)
    motion: Dict[int, bool] = {}
    for r in results:
        pi = r["person_idx"]
        if pi is not None:
            # If any hand for this person is moving → count as moving
            motion[pi] = motion.get(pi, False) or r["is_moving"]
    return motion


def cleanup_camera(camera_id: int) -> None:
    """Remove per-camera landmark history (call on camera stop)."""
    _prev_landmarks.pop(camera_id, None)


def annotate_hands(frame: np.ndarray, hand_results: List[Dict]) -> np.ndarray:
    """Draw hand landmarks and motion status on frame for debug/stream."""
    import cv2
    out = frame.copy()
    for r in hand_results:
        lms = r["landmarks"]
        color = (0, 220, 60) if r["is_moving"] else (0, 80, 220)
        for x, y in lms:
            cv2.circle(out, (int(x), int(y)), 3, color, -1)
        # Draw wrist label
        wx, wy = int(lms[0][0]), int(lms[0][1])
        label = f"{'MOVING' if r['is_moving'] else 'IDLE'} {r['displacement_px']:.1f}px"
        cv2.putText(out, label, (wx, wy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
