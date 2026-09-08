"""
headcap_monitor.py -- Per-person Bakery Head-Cap compliance state machine
=========================================================================
v3.2 — Crop-based inference + temporal smoothing

Root cause of false NO_HEAD_CAP: YOLO was running on the FULL FRAME.
Small/distant persons produce tiny head regions where the cap occupies
only a few pixels — YOLO cannot classify what it cannot see.

Fix: extract the head crop from the person bbox, run YOLO on THAT crop,
then apply temporal smoothing before changing state.

Pipeline per tracked person (once per second):
    1. Derive head crop: top 30% of person bbox + 10% padding
    2. Validate crop: must be >= 32x32 px
    3. Submit crop to YOLO (shared inference pool)
    4. Map Bakery-Head-Cap confidence to:
         conf >= HEADCAP_CONF_THRESHOLD  → HEAD_CAP
         conf <= HEADCAP_UNCERTAIN_LOW   → NO_HEAD_CAP
         between                          → UNCERTAIN
    5. Append raw prediction to per-track sliding window (last N)
    6. Majority vote on window → smoothed prediction
    7. Only transition state if smoothed prediction sustained >= HEADCAP_MISSING_SECONDS

State machine per (camera_id, track_id):
    COMPLIANT
        |  smoothed NO_HEAD_CAP (sustained >= HEADCAP_MISSING_SECONDS)
        v
    PENDING_MISSING
        |  HEAD_CAP seen again    → COMPLIANT  (timer reset)
        |  sustained >= threshold
        v
    ALERT_TRIGGERED
        |  HEAD_CAP seen again    → COMPLIANT
        |  (cooldown)             → no new alert

Public API (unchanged):
    update_camera(camera_id, persons, raw_dets, db, frame) → list[PersonCapStatus]
    get_debug_info(camera_id)                              → list[dict]
    reset_camera(camera_id)                                → None
    get_diagnostics(camera_id)                             → list[HeadCapDiagnostic]
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings

log = logging.getLogger("headcap_monitor")

# ── State constants ────────────────────────────────────────────────────────────
STATE_COMPLIANT = "COMPLIANT"
STATE_PENDING   = "PENDING_MISSING"
STATE_ALERT     = "ALERT_TRIGGERED"

# ── Prediction labels ──────────────────────────────────────────────────────────
PRED_HEAD_CAP    = "HEAD_CAP"
PRED_NO_HEAD_CAP = "NO_HEAD_CAP"
PRED_UNCERTAIN   = "UNCERTAIN"

RULE_MSG = "Not Wearing Head Cap"

# ── Tuning constants (override via config/settings as needed) ──────────────────
_HEAD_CROP_RATIO    = 0.40   # top 40% of person bbox = head region (was 0.30)
_HEAD_CROP_PADDING  = 0.15   # 15% horizontal padding on crop (was 0.10)
_MIN_CROP_PX        = 24     # crops smaller than this → LOW_RESOLUTION flag (was 32)
_WINDOW_SIZE        = 10     # sliding window for temporal smoothing
_INFER_INTERVAL     = 1.0    # seconds between inferences per track
_CONF_THRESHOLD     = getattr(settings, "HEADCAP_CONF_THRESHOLD", 0.35)  # lowered from 0.45
_UNCERTAIN_LOW      = 0.20   # below this = NO_HEAD_CAP (no ambiguity)
_MISSING_SECONDS    = getattr(settings, "HEADCAP_MISSING_SECONDS", 5.0)  # raised from 3.0
_ALERT_COOLDOWN     = getattr(settings, "HEADCAP_ALERT_COOLDOWN", 60.0)  # raised from 30.0

# ── YOLO class name to look for in crop ───────────────────────────────────────
_CAP_CLASS_LABEL = "Bakery-Head-Cap"

# ── Diagnostic flag values ─────────────────────────────────────────────────────
DIAG_OK               = "OK"
DIAG_LOW_RESOLUTION   = "LOW_RESOLUTION"    # crop too small for reliable inference
DIAG_CROP_FAILURE     = "CROP_FAILURE"      # could not extract crop from frame
DIAG_LOW_CONFIDENCE   = "LOW_CONFIDENCE"    # prediction in UNCERTAIN zone
DIAG_INFER_ERROR      = "INFER_ERROR"       # YOLO call threw exception


# ─────────────────────────────────────────────────────────────────────────────
# Head crop geometry
# ─────────────────────────────────────────────────────────────────────────────

def _derive_head_crop_box(person_bbox: List[int], frame_shape: Tuple[int, int]) -> List[int]:
    """
    Return [x1, y1, x2, y2] for the head crop region.
    - Top 30% of person height.
    - 10% horizontal expansion for occlusion tolerance.
    - Clamped to frame bounds.
    """
    x1, y1, x2, y2 = person_bbox
    w = x2 - x1
    h = y2 - y1
    margin_x = int(w * _HEAD_CROP_PADDING)
    head_h = int(h * _HEAD_CROP_RATIO)
    fh, fw = frame_shape[:2]
    cx1 = max(0, x1 - margin_x)
    cy1 = max(0, y1)
    cx2 = min(fw, x2 + margin_x)
    cy2 = min(fh, y1 + head_h)
    return [cx1, cy1, cx2, cy2]


def _extract_crop(frame: np.ndarray, crop_box: List[int]) -> Optional[np.ndarray]:
    """Extract and return a crop from frame. Returns None on failure."""
    x1, y1, x2, y2 = crop_box
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


# ─────────────────────────────────────────────────────────────────────────────
# Crop-based YOLO inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_on_crop(crop: np.ndarray) -> Tuple[float, str]:
    """
    Run hairnet_glove_detection YOLO model on the head crop.
    Returns (cap_confidence: float, diagnostic_flag: str).
      - Hair_Cover / Hairnet / Bakery-Head-Cap detected: returns (best_conf, DIAG_OK)
      - Hair / NO-Bakery-Head-Cap detected (bare hair): returns (0.0, DIAG_OK)
      - Neither detected: returns (0.0, DIAG_LOW_CONFIDENCE)
    """
    try:
        from services.model_manager import ModelManager
        mm = ModelManager()
        if mm.is_loaded("hairnet_glove_detection") or mm.registry.is_model_enabled("hairnet_glove_detection"):
            # Use conf=0.15 — lower than global default (0.25) to catch weaker
            # Hair / Hair_Cover signals on small top-down CCTV head crops
            dets = mm.infer("hairnet_glove_detection", crop, conf=0.15)
            # ModelManager returns raw model names, but other model paths can
            # return normalized aliases.  Normalize before classification so a
            # valid cap is not lost because it arrived as Hairnet/hair_cover_ok.
            from services.multi_model_detector import normalize_class_label

            cap_confs = []
            hair_confs = []
            for det in dets:
                raw_name = str(getattr(det, "class_name", "") or "")
                normalized, _det_type = normalize_class_label(raw_name)
                name = normalized or raw_name
                if name in ("Bakery-Head-Cap", "Hairnet", "hairnet", "Hair_Cover", "hair_cover_ok"):
                    cap_confs.append(float(det.confidence))
                elif name in ("NO-Bakery-Head-Cap", "Hair", "hair", "NO-Hairnet", "no_hairnet", "no_hair_cover"):
                    hair_confs.append(float(det.confidence))

            if cap_confs:
                return float(max(cap_confs)), DIAG_OK
            elif hair_confs:
                return 0.0, DIAG_OK
            else:
                return 0.0, DIAG_LOW_CONFIDENCE
    except Exception as exc:
        log.debug("[HeadCap] model_manager crop inference error: %s", exc)

    # Fallback to inference pool if hairnet model is unavailable
    try:
        from services.inference_pool import inference_pool
        crop_key = f"_headcap_crop_{threading.get_ident()}"
        inference_pool.put_frame(crop_key, crop)

        result = None
        for _ in range(10):
            result = inference_pool.get_result(crop_key)
            if result is not None:
                break
            time.sleep(0.005)

        inference_pool.clear_result(crop_key)

        if result is None:
            return 0.0, DIAG_INFER_ERROR

        detections = result.get("detections", [])
        best_conf = 0.0
        from services.multi_model_detector import normalize_class_label
        for det in detections:
            lbl = det.get("label", "")
            conf = float(det.get("confidence", 0.0))
            normalized, _det_type = normalize_class_label(lbl)
            if (normalized == _CAP_CLASS_LABEL or lbl in ("Hairnet", "hairnet", "Hair_Cover", "hair_cover_ok")) and conf > best_conf:
                best_conf = conf

        return best_conf, DIAG_OK

    except Exception as exc:
        log.debug("[HeadCap] crop inference error: %s", exc)
        return 0.0, DIAG_INFER_ERROR


def _map_confidence_to_prediction(conf: float, diag_flag: str = DIAG_OK) -> str:
    """Map raw confidence to 3-state prediction.

    LOW_RESOLUTION / INFER_ERROR → UNCERTAIN (benefit of doubt — don't alert
    for workers who are too far or whose crop is too small to classify).
    """
    if conf >= _CONF_THRESHOLD:
        return PRED_HEAD_CAP
    if diag_flag in (DIAG_LOW_CONFIDENCE, DIAG_LOW_RESOLUTION, DIAG_INFER_ERROR):
        return PRED_UNCERTAIN  # can't see clearly enough → don't flag
    if conf <= _UNCERTAIN_LOW:
        return PRED_NO_HEAD_CAP
    return PRED_UNCERTAIN


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat: full-frame association (fallback when no frame given)
# ─────────────────────────────────────────────────────────────────────────────

def _head_region(person_box: List[int]) -> List[int]:
    """Return [x1, y1, x2, y2] for the top 35% of the person bbox."""
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    w = x2 - x1
    margin_x = int(w * 0.10)
    head_y2 = y1 + int(h * 0.35)
    return [max(0, x1 - margin_x), y1, x2 + margin_x, head_y2]


def _iou(a: List[int], b: List[int]) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    areaB = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(areaA + areaB - inter)


def _center_in_box(small_box: List[int], large_box: List[int]) -> bool:
    cx = (small_box[0] + small_box[2]) / 2
    cy = (small_box[1] + small_box[3]) / 2
    return (large_box[0] <= cx <= large_box[2] and
            large_box[1] <= cy <= large_box[3])


def _headcap_belongs_to_person(
    person_box: List[int],
    cap_box: List[int],
    cap_conf: float,
) -> bool:
    if cap_conf < _UNCERTAIN_LOW:
        return False
    head_rgn = _head_region(person_box)
    if _center_in_box(cap_box, head_rgn):
        return True
    if _iou(head_rgn, cap_box) >= 0.05:
        return True
    x1, y1, x2, y2 = person_box
    cap_cy = (cap_box[1] + cap_box[3]) / 2
    cap_cx = (cap_box[0] + cap_box[2]) / 2
    person_mid_y = (y1 + y2) / 2
    if x1 <= cap_cx <= x2 and y1 <= cap_cy <= person_mid_y:
        return True
    return False


def associate_headcaps_to_persons(
    persons: List[dict],
    raw_dets: List[dict],
) -> List[Tuple[dict, bool, float, Optional[List[int]]]]:
    """
    Fallback: associate caps from full-frame detections.
    Used when frame is not available for crop-based inference.
    """
    cap_dets = [
        d for d in raw_dets
        if (d.get("label") in (_CAP_CLASS_LABEL, "Hairnet", "hairnet")
            or d.get("raw_label") in ("hairnet", "Hairnet"))
        and d.get("confidence", 0) >= _UNCERTAIN_LOW
    ]

    results: List[Tuple[dict, bool, float, Optional[List[int]]]] = []
    claimed: set = set()

    for person in persons:
        p_box = person.get("bbox", [])
        if not p_box or len(p_box) != 4:
            results.append((person, False, 0.0, None))
            continue

        best_score = -1.0
        best_cap = None
        best_idx = -1

        for idx, cap in enumerate(cap_dets):
            if idx in claimed:
                continue
            c_box = cap.get("bbox", [])
            c_conf = cap.get("confidence", 0.0)
            if not _headcap_belongs_to_person(p_box, c_box, c_conf):
                continue
            head_rgn = _head_region(p_box)
            score = c_conf + _iou(head_rgn, c_box)
            if score > best_score:
                best_score = score
                best_cap = cap
                best_idx = idx

        if best_cap is not None:
            claimed.add(best_idx)
            results.append((person, True, best_cap["confidence"], best_cap["bbox"]))
        else:
            results.append((person, False, 0.0, None))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _PersonState:
    track_id: str
    state: str = STATE_COMPLIANT
    missing_since: Optional[float] = None
    last_alert_at: Optional[float] = None
    # Current observation
    has_headcap: bool = False
    cap_confidence: float = 0.0
    raw_prediction: str = PRED_UNCERTAIN
    smoothed_prediction: str = PRED_UNCERTAIN
    # Geometry
    person_bbox: List[int] = field(default_factory=list)
    head_region: List[int] = field(default_factory=list)
    crop_bbox: List[int] = field(default_factory=list)
    cap_bbox: Optional[List[int]] = None
    # Temporal smoothing
    prediction_window: Deque[str] = field(
        default_factory=lambda: deque(maxlen=_WINDOW_SIZE)
    )
    # Throttle: only run crop inference every _INFER_INTERVAL seconds
    last_infer_at: float = 0.0
    # Diagnostics
    crop_width: int = 0
    crop_height: int = 0
    diagnostic_flag: str = DIAG_OK


@dataclass
class PersonCapStatus:
    camera_id: int
    track_id: str
    has_headcap: bool
    state: str
    missing_seconds: float
    alert_fired: bool
    cap_confidence: float
    raw_prediction: str
    smoothed_prediction: str
    cap_bbox: Optional[List[int]]
    person_bbox: List[int]
    head_region: List[int]
    crop_bbox: List[int]
    crop_width: int
    crop_height: int
    diagnostic_flag: str


@dataclass
class HeadCapDiagnostic:
    """Full diagnostic snapshot per track — for /diagnostics endpoint and overlays."""
    camera_id: int
    track_id: str
    original_bbox: List[int]
    crop_bbox: List[int]
    crop_width: int
    crop_height: int
    raw_prediction: str
    confidence: float
    smoothed_prediction: str
    window: List[str]
    diagnostic_flag: str
    state: str
    missing_seconds: float
    timestamp: float


# ─────────────────────────────────────────────────────────────────────────────
# Monitor
# ─────────────────────────────────────────────────────────────────────────────

class HeadCapMonitor:
    """
    Thread-safe singleton. One instance across all cameras.
    v3.2: crop-based YOLO inference + temporal smoothing.
    """

    def __init__(self) -> None:
        self._state: Dict[Tuple[int, str], _PersonState] = {}
        self._lock = threading.Lock()

    def update_camera(
        self,
        camera_id: int,
        persons: List[dict],
        raw_dets: List[dict],
        db: Any,
        frame: Optional[np.ndarray] = None,
    ) -> List[PersonCapStatus]:
        """
        Main entry point: call once per detection-loop tick.
        Returns list of PersonCapStatus for debug overlay.

        v3.2 logic:
          - If frame is available: run crop-based YOLO inference (throttled to 1/s)
          - Otherwise: fall back to full-frame association from raw_dets
        """
        now = time.monotonic()
        wall_now = time.time()
        statuses: List[PersonCapStatus] = []
        pending_alerts: List[PersonCapStatus] = []
        active_keys: set = set()

        for person in persons:
            track_id = str(person.get("track_id", ""))
            if not track_id or track_id in ("-1", ""):
                continue

            key = (camera_id, track_id)
            active_keys.add(key)
            p_box = person.get("bbox", [])
            if not p_box or len(p_box) != 4:
                continue

            h_rgn = _head_region(p_box)

            with self._lock:
                ps = self._state.get(key)
                if ps is None:
                    ps = _PersonState(track_id=track_id)
                    self._state[key] = ps

            # ── Choose inference method ────────────────────────────────────
            # Priority 1: Check full-frame detections from raw_dets (native YOLO imgsz=960)
            matching_full_frame = [
                d for d in raw_dets
                if (d.get("label") in ("Bakery-Head-Cap", "NO-Bakery-Head-Cap", "Hairnet", "NO-Hairnet")
                    or d.get("raw_label") in ("hairnet", "no_hairnet"))
                and _headcap_belongs_to_person(p_box, d.get("bbox", []), d.get("confidence", 0.0))
            ]

            if matching_full_frame:
                best_ff = max(matching_full_frame, key=lambda d: d.get("confidence", 0.0))
                raw_n = best_ff.get("raw_label") or best_ff.get("label")
                ff_conf = float(best_ff.get("confidence", 0.0))
                ff_box = best_ff.get("bbox")
                if raw_n in ("Bakery-Head-Cap", "Hairnet", "hairnet"):
                    raw_pred = PRED_HEAD_CAP
                    cap_conf = ff_conf
                else:
                    raw_pred = PRED_NO_HEAD_CAP
                    cap_conf = 0.0
                diag_flag = DIAG_OK

                with self._lock:
                    ps.last_infer_at = wall_now
                    ps.cap_bbox = ff_box
                    ps.cap_confidence = cap_conf
                    ps.raw_prediction = raw_pred
                    ps.diagnostic_flag = diag_flag
                    if raw_pred != PRED_UNCERTAIN:
                        ps.prediction_window.append(raw_pred)

            # Priority 2: Crop-based fallback only if no full-frame detection and throttle allows
            elif frame is not None and (wall_now - ps.last_infer_at) >= _INFER_INTERVAL:
                crop_box = _derive_head_crop_box(p_box, frame.shape)
                crop = _extract_crop(frame, crop_box)
                cw = crop_box[2] - crop_box[0]
                ch = crop_box[3] - crop_box[1]

                if crop is None or cw < _MIN_CROP_PX or ch < _MIN_CROP_PX:
                    # Crop too small — skip inference this tick, mark diagnostic
                    raw_pred = PRED_UNCERTAIN
                    cap_conf = 0.0
                    diag_flag = DIAG_LOW_RESOLUTION
                else:
                    cap_conf, diag_flag = _infer_on_crop(crop)
                    raw_pred = _map_confidence_to_prediction(cap_conf, diag_flag)
                    if raw_pred == PRED_UNCERTAIN and diag_flag == DIAG_OK:
                        diag_flag = DIAG_LOW_CONFIDENCE

                with self._lock:
                    ps.last_infer_at = wall_now
                    ps.crop_bbox = crop_box
                    ps.crop_width = cw
                    ps.crop_height = ch
                    ps.cap_confidence = cap_conf
                    ps.raw_prediction = raw_pred
                    ps.diagnostic_flag = diag_flag
                    # Add to temporal window (skip UNCERTAIN to avoid noise)
                    if raw_pred != PRED_UNCERTAIN:
                        ps.prediction_window.append(raw_pred)

            else:
                # Use cached values from last inference tick
                with self._lock:
                    cap_conf = ps.cap_confidence
                    raw_pred = ps.raw_prediction
                    diag_flag = ps.diagnostic_flag

            # ── Temporal smoothing (majority vote) ─────────────────────────
            with self._lock:
                window_list = list(ps.prediction_window)
            if window_list:
                head_cap_votes = window_list.count(PRED_HEAD_CAP)
                no_cap_votes   = window_list.count(PRED_NO_HEAD_CAP)
                if head_cap_votes > no_cap_votes:
                    smoothed = PRED_HEAD_CAP
                elif no_cap_votes > head_cap_votes:
                    smoothed = PRED_NO_HEAD_CAP
                else:
                    smoothed = PRED_UNCERTAIN  # tie → uncertain = safe
            else:
                smoothed = PRED_UNCERTAIN

            has_headcap = smoothed == PRED_HEAD_CAP

            # ── State machine ──────────────────────────────────────────────
            alert_fired = False
            with self._lock:
                ps.has_headcap       = has_headcap
                ps.smoothed_prediction = smoothed
                ps.person_bbox       = p_box
                ps.head_region       = h_rgn

                if has_headcap or smoothed == PRED_UNCERTAIN:
                    # Cap seen (or uncertain — benefit of doubt)
                    if ps.state != STATE_COMPLIANT:
                        log.debug(
                            "[HeadCap] cam=%d track=%s → COMPLIANT (conf=%.2f smoothed=%s)",
                            camera_id, track_id, cap_conf, smoothed,
                        )
                    ps.state = STATE_COMPLIANT
                    ps.missing_since = None
                else:
                    # smoothed = NO_HEAD_CAP
                    if ps.missing_since is None:
                        ps.missing_since = now
                        ps.state = STATE_PENDING
                        log.debug("[HeadCap] cam=%d track=%s → PENDING_MISSING", camera_id, track_id)

                    missing_secs = now - ps.missing_since

                    if missing_secs >= _MISSING_SECONDS:
                        ps.state = STATE_ALERT
                        since_last = (
                            (now - ps.last_alert_at)
                            if ps.last_alert_at is not None
                            else float("inf")
                        )
                        # One alert per continuous violation.  Repeating after a
                        # cooldown made a person standing all day generate
                        # hundreds/thousands of identical alerts.  A fresh alert
                        # is allowed only after cap compliance resets state.
                        if ps.state != STATE_ALERT and since_last >= _ALERT_COOLDOWN:
                            ps.last_alert_at = now
                            alert_fired = True
                            log.warning(
                                "[HeadCap] ALERT cam=%d track=%s missing=%.1fs",
                                camera_id, track_id, missing_secs,
                            )

                missing_seconds = (now - ps.missing_since) if ps.missing_since else 0.0
                state_snap = ps.state
                crop_bbox_snap = list(ps.crop_bbox)
                cw_snap = ps.crop_width
                ch_snap = ps.crop_height
                diag_snap = ps.diagnostic_flag

            st = PersonCapStatus(
                camera_id=camera_id,
                track_id=track_id,
                has_headcap=has_headcap,
                state=state_snap,
                missing_seconds=missing_seconds,
                alert_fired=alert_fired,
                cap_confidence=cap_conf,
                raw_prediction=raw_pred,
                smoothed_prediction=smoothed,
                cap_bbox=None,  # no longer relevant in crop mode
                person_bbox=p_box,
                head_region=h_rgn,
                crop_bbox=crop_bbox_snap,
                crop_width=cw_snap,
                crop_height=ch_snap,
                diagnostic_flag=diag_snap,
            )
            statuses.append(st)
            if alert_fired:
                pending_alerts.append(st)

        # Fire alerts outside the lock (avoids holding it during DB I/O)
        for st in pending_alerts:
            self._fire_alert(
                camera_id=camera_id,
                track_id=st.track_id,
                missing_seconds=st.missing_seconds,
                db=db,
                frame=frame,
            )

        self._evict_stale(camera_id, active_keys)
        return statuses

    def get_debug_info(self, camera_id: int) -> List[dict]:
        """Snapshot of per-person state for overlay rendering."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "track_id": ps.track_id,
                    "state": ps.state,
                    "has_headcap": ps.has_headcap,
                    "cap_confidence": round(ps.cap_confidence, 3),
                    "raw_prediction": ps.raw_prediction,
                    "smoothed_prediction": ps.smoothed_prediction,
                    "missing_seconds": (
                        round(now - ps.missing_since, 2)
                        if ps.missing_since else 0.0
                    ),
                    "person_bbox": ps.person_bbox,
                    "head_region": ps.head_region,
                    "crop_bbox": ps.crop_bbox,
                    "crop_width": ps.crop_width,
                    "crop_height": ps.crop_height,
                    "diagnostic_flag": ps.diagnostic_flag,
                    "window": list(ps.prediction_window),
                }
                for (cam_id, _), ps in self._state.items()
                if cam_id == camera_id
            ]

    def get_diagnostics(self, camera_id: int) -> List[HeadCapDiagnostic]:
        """Full diagnostic snapshots — for /diagnostics endpoint."""
        now_mono = time.monotonic()
        now_wall = time.time()
        with self._lock:
            result = []
            for (cam_id, _), ps in self._state.items():
                if cam_id != camera_id:
                    continue
                result.append(HeadCapDiagnostic(
                    camera_id=camera_id,
                    track_id=ps.track_id,
                    original_bbox=list(ps.person_bbox),
                    crop_bbox=list(ps.crop_bbox),
                    crop_width=ps.crop_width,
                    crop_height=ps.crop_height,
                    raw_prediction=ps.raw_prediction,
                    confidence=round(ps.cap_confidence, 4),
                    smoothed_prediction=ps.smoothed_prediction,
                    window=list(ps.prediction_window),
                    diagnostic_flag=ps.diagnostic_flag,
                    state=ps.state,
                    missing_seconds=(
                        round(now_mono - ps.missing_since, 2)
                        if ps.missing_since else 0.0
                    ),
                    timestamp=now_wall,
                ))
            return result

    def reset_camera(self, camera_id: int) -> None:
        with self._lock:
            for key in [k for k in self._state if k[0] == camera_id]:
                del self._state[key]
        log.info("[HeadCap] State reset cam=%d", camera_id)

    def _evict_stale(self, camera_id: int, active_keys: set) -> None:
        with self._lock:
            for k in [k for k in self._state
                      if k[0] == camera_id and k not in active_keys]:
                del self._state[k]

    @staticmethod
    def _fire_alert(
        camera_id: int,
        track_id: str,
        missing_seconds: float,
        db: Any,
        frame: Optional[np.ndarray],
    ) -> None:
        try:
            import base64
            from services.alert_service import save_alert
            from services.rule_engine import _get_rule_engine_user_id
            from services.face_service import get_worker_identity

            uid = _get_rule_engine_user_id(db)

            # Resolve floor from camera record
            floor_name = None
            try:
                from database import Camera
                cam = db.query(Camera).filter(Camera.id == camera_id).first()
                if cam:
                    floor_name = cam.floor
            except Exception:
                pass

            # Resolve worker name from face recognition cache
            identity = get_worker_identity(camera_id, track_id)
            worker_name = identity["name"] if identity else None
            emp_id = identity["employee_id"] if identity else None

            snapshot_b64 = None
            if frame is not None:
                try:
                    _, buf = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    snapshot_b64 = (
                        "data:image/jpeg;base64,"
                        + base64.b64encode(buf).decode("utf-8")
                    )
                except Exception as snap_err:
                    log.debug("[HeadCap] snapshot err: %s", snap_err)

            msg = (
                f"{worker_name} has not worn a Bakery Head Cap (missing {missing_seconds:.1f}s, camera {camera_id})"
                if worker_name
                else f"Person #{track_id} — Not Wearing a Bakery Head Cap (missing {missing_seconds:.1f}s, camera {camera_id})"
            )
            issue_title = f"{worker_name} — No Head Cap" if worker_name else RULE_MSG
            save_alert(
                db=db,
                user_id=uid,
                message=msg,
                role="Bakery Worker",
                severity="critical",
                detected_issue=issue_title,
                confidence=None,
                snapshot_b64=snapshot_b64,
                camera_id=camera_id,
                floor=floor_name,
                employee_id=emp_id,
                confidence_tier="high",
                worker_name=worker_name,
            )
            log.info(
                "[HeadCap] Alert saved cam=%d track=%s worker=%s missing=%.1fs floor=%s",
                camera_id, track_id, worker_name or "Unknown", missing_seconds, floor_name,
            )
        except Exception as exc:
            log.error(
                "[HeadCap] Alert save error cam=%d track=%s: %s",
                camera_id, track_id, exc,
            )



# ── Module-level singleton ────────────────────────────────────────────────────
headcap_monitor = HeadCapMonitor()
