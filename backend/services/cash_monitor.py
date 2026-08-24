"""
services/cash_monitor.py — Cash-zone & Stock-zone monitoring
=============================================================
Monitors:
  • CashMonitor  — alerts when a Person is detected inside the cashbox zone
                   during unauthorized hours (no cashier present)
  • StockMonitor — alerts when goods are left unattended (Exposed-Item class,
                   or Person loitering in stock zone without authorization)

Both monitors are stateless per-call (thread-safe) and write alerts via
alert_service.save_alert(). They are called from the camera_manager
detection loop after YOLO inference.

Authorization model (v1):
  Cashbox zone allowed hours: configurable via SystemSettings key
  "cashbox_allowed_hours" (default: "08:00-20:00").
  Outside those hours, any Person detection inside the cashbox zone fires alert.
"""

import datetime
import logging
import time
from typing import Dict, List, Optional

log = logging.getLogger("cash_monitor")

# ─────────────────────────────────────────────────────────────────────────────
# Cooldown registry — prevent alert spam
# ─────────────────────────────────────────────────────────────────────────────
_last_alert_ts: Dict[str, float] = {}
_COOLDOWN_SEC = 120  # 2 min between identical alerts per camera


def _can_alert(key: str) -> bool:
    now = time.time()
    last = _last_alert_ts.get(key, 0)
    if now - last < _COOLDOWN_SEC:
        return False
    _last_alert_ts[key] = now
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_allowed_hours(db) -> tuple:
    """
    Return (start_h, end_h) from SystemSettings 'cashbox_allowed_hours'.
    Default: 08:00–20:00.
    """
    try:
        from database import SystemSettings

        row = (
            db.query(SystemSettings)
            .filter(SystemSettings.key == "cashbox_allowed_hours")
            .first()
        )
        if row:
            parts = row.value.split("-")
            start_h = int(parts[0].split(":")[0])
            end_h = int(parts[1].split(":")[0])
            return start_h, end_h
    except Exception:
        pass
    return 8, 20


def _person_in_zone(person_bbox: List[int], zone_polygon: Optional[List]) -> bool:
    """
    Simple centroid-in-polygon check.
    zone_polygon: list of [x, y] points (already loaded from ZoneConfig).
    Returns True if person centroid is inside the polygon.
    """
    if not zone_polygon or len(zone_polygon) < 3:
        return False
    cx = (person_bbox[0] + person_bbox[2]) / 2
    cy = (person_bbox[1] + person_bbox[3]) / 2
    # Ray-casting algorithm
    n = len(zone_polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = zone_polygon[i]
        xj, yj = zone_polygon[j]
        if ((yi > cy) != (yj > cy)) and (
            cx < (xj - xi) * (cy - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def _fire_alert(db, camera_id: int, message: str, severity: str, issue: str):
    """Save an alert via alert_service using the system user."""
    try:
        from services.alert_service import save_alert
        from services.rule_engine import _get_rule_engine_user_id

        user_id = _get_rule_engine_user_id(db)
        save_alert(
            db=db,
            user_id=user_id,
            message=message,
            role="System",
            severity=severity,
            detected_issue=issue,
            confidence=None,
            snapshot_b64=None,
            camera_id=camera_id,
        )
    except Exception as exc:
        log.error(f"[CashMonitor] _fire_alert failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CashMonitor
# ─────────────────────────────────────────────────────────────────────────────


def check_cash_zone(
    db,
    camera_id: int,
    detections: List[Dict],
    cashbox_polygon: Optional[List] = None,
) -> None:
    """
    Call this every N frames from the detection loop for cameras that cover
    the shop counter / cashbox zone.

    Parameters
    ----------
    detections      : list of dicts with keys 'label', 'bbox', 'confidence'
    cashbox_polygon : [[x,y], ...] pixel coords of the cashbox zone polygon.
                      If None, no check is performed.
    """
    if cashbox_polygon is None:
        return

    now_h = datetime.datetime.utcnow().hour
    start_h, end_h = _get_allowed_hours(db)
    unauthorized_hour = not (start_h <= now_h < end_h)

    persons_in_zone = [
        d
        for d in detections
        if d.get("label") == "Person"
        and _person_in_zone(d.get("bbox", []), cashbox_polygon)
    ]

    if persons_in_zone and unauthorized_hour:
        key = f"cash_unauthorized_{camera_id}"
        if _can_alert(key):
            msg = (
                f"[CASH ZONE] Camera {camera_id} — {len(persons_in_zone)} person(s) "
                f"detected near cashbox at {datetime.datetime.utcnow().strftime('%H:%M')} "
                f"(outside authorized hours {start_h:02d}:00–{end_h:02d}:00)."
            )
            log.warning(msg)
            _fire_alert(db, camera_id, msg, "high", "Unauthorized cashbox access")


# ─────────────────────────────────────────────────────────────────────────────
# StockMonitor
# ─────────────────────────────────────────────────────────────────────────────


def check_stock_zone(
    db,
    camera_id: int,
    detections: List[Dict],
    stock_polygon: Optional[List] = None,
) -> None:
    """
    Alert when Exposed-Item class is detected OR when Person loiters
    in stock zone without moving (handled by idle_service separately).

    This function only handles the Exposed-Item YOLO class check.
    """
    exposed_items = [d for d in detections if d.get("label") == "Exposed-Item"]

    if not exposed_items:
        return

    # If zone polygon is defined, filter to only items inside the zone
    if stock_polygon:
        exposed_items = [
            d
            for d in exposed_items
            if _person_in_zone(d.get("bbox", []), stock_polygon)
        ]

    if exposed_items:
        key = f"stock_exposed_{camera_id}"
        if _can_alert(key):
            msg = (
                f"[STOCK ZONE] Camera {camera_id} — {len(exposed_items)} exposed "
                f"item(s) detected in open stock area at "
                f"{datetime.datetime.utcnow().strftime('%H:%M')}. "
                "Cover or secure goods immediately."
            )
            log.warning(msg)
            _fire_alert(db, camera_id, msg, "medium", "Exposed stock detected")
