"""
services/packing_monitor.py — Packing-line activity monitoring
==============================================================
Monitors the packing zone for worker activity.

v2 (active): Intel OpenVINO person-detection-action-recognition-0006 used as
  primary signal.  If model reports raising_hand / writing → hands active.
  If model is unavailable → falls back to frame-difference pixel variance
  heuristic (v1) on person bounding box crops.

Called from camera_manager detection loop for cameras whose zone_type
contains "packing".
"""

import datetime
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("packing_monitor")

# ─────────────────────────────────────────────────────────────────────────────
# Per-track state: { (camera_id, track_id): {"idle_since": float, "prev_crop_mean": float} }
# ─────────────────────────────────────────────────────────────────────────────
_packing_state: Dict[Tuple[int, int], dict] = {}
_last_alert: Dict[Tuple[int, int], float] = {}
_IDLE_THRESHOLD_SEC = 90  # alert if hands stationary > 90s in packing zone
_MOTION_VAR_THRESH = 8.0  # pixel variance threshold — below = no hand movement
_COOLDOWN_SEC = 120


def _centroid(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _in_zone(bbox: List[int], polygon: List) -> bool:
    if not polygon or len(polygon) < 3:
        return False
    cx, cy = _centroid(bbox)
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > cy) != (yj > cy)) and (
            cx < (xj - xi) * (cy - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def _crop_bbox(frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
    """Extract person bounding box crop, focus on lower half (hands area)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    # Only look at lower 60% of person (hands/torso area)
    mid_y = y1 + int((y2 - y1) * 0.4)
    crop = frame[mid_y:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _iou(b1, b2) -> float:
    xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2]); yb = min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def process_packing_frame(
    db,
    camera_id: int,
    frame: np.ndarray,
    persons: List[Dict],
    packing_polygon: Optional[List] = None,
) -> None:
    """
    Process one camera frame for packing zone activity.

    Signal priority (highest → lowest accuracy):
      1. MediaPipe HandLandmarker  — wrist + fingertip pixel velocity
         (hand_motion.py, uses hand_landmarker.task)
      2. Intel OpenVINO action-recognition model
         raising_hand/writing → active; sitting/listening → idle
      3. Frame-diff pixel variance fallback
         (no ML, but works when above models unavailable)

    Parameters
    ----------
    camera_id       : DB cameras.id
    frame           : BGR numpy frame
    persons         : list of dicts with keys: bbox, track_id
    packing_polygon : [[x,y], ...] pixel coords of packing zone.
                      If None, monitoring covers the whole frame.
    """
    now = time.time()

    # ── Priority 1: MediaPipe hand landmark motion ──────────────────────────
    # { person_bbox_index: is_moving }  — populated if hand model runs OK
    hand_motion_map: Dict[int, bool] = {}
    person_bboxes = [p.get("bbox", []) for p in persons if len(p.get("bbox", [])) == 4]
    try:
        from services.hand_motion import hands_moving_for_person
        hand_motion_map = hands_moving_for_person(frame, camera_id, person_bboxes)
    except Exception as _hme:
        log.debug("[Packing] hand_motion unavailable: %s", _hme)

    # ── Attempt action recognition (v2) ─────────────────────────────────────
    action_map: Dict[int, str] = {}  # track_id -> action label
    try:
        from services import action_recognition as _ar
        action_results = _ar.process_frame(frame)
        # Match action detections to tracked persons by IoU
        for act in action_results:
            best_tid = -1
            best_iou = 0.25
            for p in persons:
                pb = p.get("bbox", [])
                if len(pb) < 4:
                    continue
                iou = _iou(act["bbox"], pb)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = int(p.get("track_id", -1))
            if best_tid != -1:
                action_map[best_tid] = act["action"]
    except Exception as _ae:
        log.debug("[Packing] action_recognition unavailable: %s", _ae)

    # ── Frame-diff fallback (v1) ─────────────────────────────────────────────
    gray = None
    try:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return

    active_keys = set()

    for person in persons:
        tid = int(person.get("track_id", -1))
        if tid == -1:
            continue
        bbox = person.get("bbox", [])
        if len(bbox) < 4:
            continue

        if packing_polygon is not None and not _in_zone(bbox, packing_polygon):
            continue

        key = (camera_id, tid)
        active_keys.add(key)

        # ── Determine if hands are moving (priority 1→2→3) ──────────────────
        hands_moving: Optional[bool] = None

        # Priority 1: MediaPipe hand landmark velocity
        # Match by person bbox index (same order as person_bboxes list above)
        try:
            bbox_idx = person_bboxes.index(bbox)
            if bbox_idx in hand_motion_map:
                hands_moving = hand_motion_map[bbox_idx]
        except (ValueError, KeyError):
            pass

        # Priority 2: OpenVINO action recognition
        if hands_moving is None and tid in action_map:
            act_label = action_map[tid]
            from services.action_recognition import WORKING_ACTIONS, IDLE_ACTIONS
            if act_label in WORKING_ACTIONS:
                hands_moving = True
            elif act_label in IDLE_ACTIONS:
                hands_moving = False
            # standing/side_talks → ambiguous, fall through to frame-diff

        # Priority 3: Frame-diff fallback
        if hands_moving is None:
            crop = _crop_bbox(gray, bbox)
            if crop is not None:
                var = float(np.var(crop))
                hands_moving = var >= _MOTION_VAR_THRESH

        if hands_moving is None:
            continue

        # ── Update state and fire alert if idle too long ──────────────────────
        if key not in _packing_state:
            _packing_state[key] = {
                "idle_since": now if not hands_moving else None,
            }
        else:
            st = _packing_state[key]
            if not hands_moving:
                if st["idle_since"] is None:
                    st["idle_since"] = now
                else:
                    elapsed = now - st["idle_since"]
                    if elapsed > _IDLE_THRESHOLD_SEC:
                        last = _last_alert.get(key, 0)
                        if now - last > _COOLDOWN_SEC:
                            _last_alert[key] = now
                            _fire_packing_idle_alert(db, camera_id, tid, elapsed)
            else:
                st["idle_since"] = None  # reset — movement detected

    # Clean up tracks that have left the zone
    departed = {k for k in _packing_state if k[0] == camera_id} - active_keys
    for key in departed:
        _packing_state.pop(key, None)


def _fire_packing_idle_alert(db, camera_id: int, track_id: int, elapsed: float):
    try:
        from services.alert_service import save_alert
        from services.rule_engine import _get_rule_engine_user_id

        uid = _get_rule_engine_user_id(db)
        save_alert(
            db=db,
            user_id=uid,
            message=(
                f"[PACKING ZONE] Camera {camera_id} — track #{track_id} "
                f"appears idle in packing zone for {elapsed:.0f}s. "
                "Worker may have stopped packing. Supervisor review required."
            ),
            role="System",
            severity="medium",
            detected_issue="Packing zone idle",
            camera_id=camera_id,
        )
        log.warning(
            f"[Packing] Idle alert — cam={camera_id} track={track_id} elapsed={elapsed:.0f}s"
        )
    except Exception as exc:
        log.error(f"[Packing] _fire_packing_idle_alert failed: {exc}")


def get_packing_summary(db, camera_id: Optional[int] = None) -> dict:
    """
    Return summary of packing activity alerts for the workflow page.
    Since packing events are stored as alerts (detected_issue='Packing zone idle'),
    query the alerts table.
    """
    try:
        from database import Alert

        today = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        q = db.query(Alert).filter(
            Alert.detected_issue == "Packing zone idle",
            Alert.timestamp >= today,
        )
        if camera_id:
            q = q.filter(Alert.camera_id == camera_id)
        rows = q.order_by(Alert.timestamp.desc()).limit(50).all()
        return {
            "today_count": len(rows),
            "events": [
                {
                    "id": r.id,
                    "camera_id": r.camera_id,
                    "message": r.message,
                    "timestamp": r.timestamp.isoformat(),
                    "severity": r.severity,
                }
                for r in rows
            ],
        }
    except Exception as exc:
        log.error(f"[Packing] get_packing_summary failed: {exc}")
        return {"today_count": 0, "events": []}
