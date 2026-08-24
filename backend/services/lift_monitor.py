"""
services/lift_monitor.py — Lift zone entry/exit tracking
=========================================================
Uses zone-polygon logic (no new ML model required).

When a tracked person enters the lift zone → logs a LiftEvent(entry).
When they exit → logs LiftEvent(exit) with floor_from / duration_sec.
If no exit is detected after lift_idle_limit_sec → fires an idle alert.

Called from camera_manager detection loop on every frame for cameras
whose zone_type contains "lift".
"""

import datetime
import logging
import time
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("lift_monitor")

# ─────────────────────────────────────────────────────────────────────────────
# Per-camera in-memory state (cleared on server restart — acceptable for v1)
# ─────────────────────────────────────────────────────────────────────────────
# { (camera_id, track_id): {"entered_at": float, "floor": str, "db_id": int} }
_lift_state: Dict[Tuple[int, int], dict] = {}
_LIFT_IDLE_LIMIT_SEC = 300  # 5 min — alert if person stays in lift zone

# Cooldown for idle alerts
_last_idle_alert: Dict[Tuple[int, int], float] = {}
_COOLDOWN_SEC = 180


def _centroid(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _in_zone(bbox: List[int], polygon: List) -> bool:
    """Ray-casting centroid-in-polygon check."""
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


def _save_lift_event(
    db,
    camera_id: int,
    track_id: int,
    event_type: str,
    floor_from: Optional[str] = None,
    floor_to: Optional[str] = None,
    duration_sec: Optional[float] = None,
    employee_id: Optional[int] = None,
) -> Optional[int]:
    """Insert a LiftEvent row and return its id."""
    try:
        from database import LiftEvent

        row = LiftEvent(
            camera_id=camera_id,
            track_id=track_id,
            event_type=event_type,
            floor_from=floor_from,
            floor_to=floor_to,
            duration_sec=duration_sec,
            employee_id=employee_id,
            timestamp=datetime.datetime.utcnow(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        log.info(
            f"[Lift] event={event_type} cam={camera_id} track={track_id} "
            f"from={floor_from} to={floor_to} dur={duration_sec}s"
        )
        return row.id
    except Exception as exc:
        log.error(f"[Lift] _save_lift_event failed: {exc}")
        try:
            db.rollback()
        except Exception:
            pass
        return None


def process_lift_frame(
    db,
    camera_id: int,
    floor: str,
    persons: List[Dict],
    lift_polygon: Optional[List] = None,
) -> None:
    """
    Process one camera frame for lift zone tracking.

    Parameters
    ----------
    camera_id    : DB cameras.id
    floor        : 'ground' | 'first' | 'second' | 'shop'
    persons      : list of dicts with keys: bbox, track_id, employee_id (optional)
    lift_polygon : [[x,y], ...] pixel coords of the lift zone polygon.
                   If None, no tracking is performed.
    """
    if lift_polygon is None:
        return

    now = time.time()
    active_keys = set()

    for person in persons:
        tid = int(person.get("track_id", -1))
        if tid == -1:
            continue
        bbox = person.get("bbox", [])
        if not bbox:
            continue
        employee_id = person.get("employee_id")

        key = (camera_id, tid)
        in_zone = _in_zone(bbox, lift_polygon)

        if in_zone:
            active_keys.add(key)
            if key not in _lift_state:
                # New entry
                db_id = _save_lift_event(
                    db,
                    camera_id,
                    tid,
                    "entry",
                    floor_from=floor,
                    employee_id=employee_id,
                )
                _lift_state[key] = {
                    "entered_at": now,
                    "floor": floor,
                    "db_id": db_id,
                }

            else:
                # Check idle inside lift
                elapsed = now - _lift_state[key]["entered_at"]
                if elapsed > _LIFT_IDLE_LIMIT_SEC:
                    last = _last_idle_alert.get(key, 0)
                    if now - last > _COOLDOWN_SEC:
                        _last_idle_alert[key] = now
                        try:
                            from services.alert_service import save_alert
                            from services.rule_engine import \
                                _get_rule_engine_user_id

                            uid = _get_rule_engine_user_id(db)
                            save_alert(
                                db=db,
                                user_id=uid,
                                message=(
                                    f"[LIFT IDLE] Camera {camera_id} — track #{tid} "
                                    f"has been in the lift zone for {elapsed:.0f}s on "
                                    f"{floor} floor."
                                ),
                                role="System",
                                severity="medium",
                                detected_issue="Lift zone idle",
                                camera_id=camera_id,
                            )
                        except Exception as exc:
                            log.error(f"[Lift] idle alert failed: {exc}")

    # Handle exits — any tracked key no longer in zone
    exited = set(_lift_state.keys()) - active_keys
    exited_cam = {k for k in exited if k[0] == camera_id}
    for key in exited_cam:
        state = _lift_state.pop(key)
        duration = now - state["entered_at"]
        _, tid = key
        _save_lift_event(
            db,
            camera_id,
            tid,
            "exit",
            floor_from=state["floor"],
            duration_sec=round(duration, 1),
        )


def get_recent_events(
    db, camera_id: Optional[int] = None, limit: int = 100
) -> List[dict]:
    """Return recent lift events for the workflow page."""
    try:
        from database import LiftEvent

        q = db.query(LiftEvent)
        if camera_id:
            q = q.filter(LiftEvent.camera_id == camera_id)
        rows = q.order_by(LiftEvent.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "camera_id": r.camera_id,
                "track_id": r.track_id,
                "event_type": r.event_type,
                "floor_from": r.floor_from,
                "floor_to": r.floor_to,
                "duration_sec": r.duration_sec,
                "employee_id": r.employee_id,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in rows
        ]
    except Exception as exc:
        log.error(f"[Lift] get_recent_events failed: {exc}")
        return []
