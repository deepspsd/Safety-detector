"""
services/packing_monitor.py — Packing-line activity monitoring
==============================================================
Monitors the packing zone for worker activity.

v1 approach: frame-difference heuristic on person bounding box crops.
  - If a person is in the packing zone and their bbox crop shows low
    pixel variance across consecutive frames → hands are stationary → alert.

v2 upgrade path: replace frame-diff with YOLOv8-Pose wrist-movement check.

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
_IDLE_THRESHOLD_SEC  = 90    # alert if hands stationary > 90s in packing zone
_MOTION_VAR_THRESH   = 8.0   # pixel variance threshold — below = no hand movement
_COOLDOWN_SEC        = 120


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
        if ((yi > cy) != (yj > cy)) and (cx < (xj - xi) * (cy - yi) / (yj - yi + 1e-9) + xi):
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


def process_packing_frame(
    db,
    camera_id:       int,
    frame:           np.ndarray,
    persons:         List[Dict],
    packing_polygon: Optional[List] = None,
) -> None:
    """
    Process one camera frame for packing zone activity.

    Parameters
    ----------
    camera_id       : DB cameras.id
    frame           : BGR numpy frame
    persons         : list of dicts with keys: bbox, track_id
    packing_polygon : [[x,y], ...] pixel coords of packing zone.
                      If None, monitoring is skipped.
    """
    if packing_polygon is None or frame is None:
        return

    now = time.time()
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

        if not _in_zone(bbox, packing_polygon):
            continue

        key = (camera_id, tid)
        active_keys.add(key)

        # Compute variance in the lower half of person crop
        crop = _crop_bbox(gray, bbox)
        if crop is None:
            continue
        var = float(np.var(crop))

        if key not in _packing_state:
            _packing_state[key] = {"idle_since": now if var < _MOTION_VAR_THRESH else None, "last_var": var}
        else:
            st = _packing_state[key]
            if var < _MOTION_VAR_THRESH:
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
            st["last_var"] = var

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
            db=db, user_id=uid,
            message=(
                f"[PACKING ZONE] Camera {camera_id} — track #{track_id} "
                f"appears idle in packing zone for {elapsed:.0f}s. "
                "Worker may have stopped packing. Supervisor review required."
            ),
            role="System", severity="medium",
            detected_issue="Packing zone idle",
            camera_id=camera_id,
        )
        log.warning(f"[Packing] Idle alert — cam={camera_id} track={track_id} elapsed={elapsed:.0f}s")
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
        today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
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
                    "id":        r.id,
                    "camera_id": r.camera_id,
                    "message":   r.message,
                    "timestamp": r.timestamp.isoformat(),
                    "severity":  r.severity,
                }
                for r in rows
            ],
        }
    except Exception as exc:
        log.error(f"[Packing] get_packing_summary failed: {exc}")
        return {"today_count": 0, "events": []}
