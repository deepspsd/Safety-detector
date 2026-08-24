"""
zone_service.py — Zone polygon helpers for per-camera spatial gating
=====================================================================
Provides:
  • point_in_zone(cx, cy, polygon)    — cv2.pointPolygonTest wrapper
  • bbox_center(bbox)                 — [x1,y1,x2,y2] → (cx, cy)
  • load_zones_for_camera(cam_id, db) — cached dict of zone_name → polygon
  • invalidate_zone_cache(cam_id)     — call after zone create/delete

Design
──────
Zones are stored in `zone_configs` (ZoneConfig model).
polygon_json is a JSON string of [[x,y], ...] pixel coordinates in the
camera's frame space.

An in-process cache avoids hitting the DB on every ~5 fps detection tick.
Cache is busted by invalidate_zone_cache(), called from the zone CRUD
endpoints whenever a polygon is created or deleted.

Zone names → Pipeline functions (from req.md)
──────────────────────────────────────────────
ENTRANCES & MOVEMENT
  "entrance"          → OCR gate (inward invoices + face attendance)
  "entrance_outward"  → OCR gate (outward order forms + person snapshot)
  "glass_door"        → Face attendance (glassdoor) + lift item tracking
  "loading"           → Loading/unloading + OCR gate + stock check

GROUND FLOOR PRODUCTION
  "packing"           → packing_monitor (hand-motion idle alert, >5 min)
  "oven" / "stove" / "oven_stove"
                      → gas_idle_update (boil/oil idle >10 min alert)
  "dough_table" / "dough_mixing"
                      → check_machinery_zone (shift check + post-job idle)
  "cutting_machine" / "machine"
                      → check_machinery_zone (machine-on → worker must work;
                         after finish → must move to packing)

STOCK & GOODS (ALL FLOORS)
  "stock" / "stock_area" / "raw_material"
                      → check_stock_zone (exposure + dirty floor)
  "finished_goods"    → check_stock_zone (dispatch check)
  "cylinder_area"     → process_cylinder_detections (count + usage days)

WINDOWS (ALL FLOORS)
  "window" / "window_throw"
                      → check_stock_zone restricted to window polygon
                         (stealing / throwing goods alert — immediate)

LIFT (ALL FLOORS)
  "lift"              → lift_monitor (person + item tracking across floors)

SHOP FLOOR
  "cashbox" / "cash_counter"
                      → cash_monitor.check_cash_zone (pocket vs cashbox)
  "shop_counter"      → check_shop_absence (alert if no person >1 min)
  "vendor_desk" / "payment_desk"
                      → trigger_vendor_snapshot (payee photo)

ALL CAMERAS
  "camera_standing"   → check_camera_blocking (person blocks cam >1 min)
  "cleaning_area"     → check_dirty_floor (baseline zone)
  any zone polygon    → active_zones set — painting a polygon IS sufficient
                        to activate the matching rule without changing zone_type
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("zone_service")

# ─────────────────────────────────────────────────────────────────────────────
# In-process zone cache
# ─────────────────────────────────────────────────────────────────────────────
# Structure: { camera_id: { zone_name: [[x,y], ...] } }
_zone_cache: Dict[int, Dict[str, List[List[int]]]] = {}
_cache_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────


def bbox_center(bbox: List[int]) -> Tuple[float, float]:
    """Return (cx, cy) from [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_zone(
    cx: float,
    cy: float,
    polygon: List[List[int]],
) -> bool:
    """
    Return True if point (cx, cy) is inside the closed polygon.

    Uses cv2.pointPolygonTest with measureDist=False.
    Returns True for points on the boundary too (measureDist ≥ 0).

    polygon: list of [x, y] integer pixel coordinates, at least 3 points.
    """
    if not polygon or len(polygon) < 3:
        return False
    try:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        result = cv2.pointPolygonTest(pts, (float(cx), float(cy)), measureDist=False)
        return result >= 0  # 1.0 inside, 0.0 on boundary, -1.0 outside
    except Exception as exc:
        log.error(f"[zone_service] pointPolygonTest error: {exc}")
        return False


def bbox_in_zone(
    bbox: List[int],
    polygon: List[List[int]],
) -> bool:
    """
    Convenience: check if the *centre* of a bounding box is inside the zone.
    This is sufficient for person/object gating. Wrist keypoints are Phase 4.
    """
    cx, cy = bbox_center(bbox)
    return point_in_zone(cx, cy, polygon)


# ─────────────────────────────────────────────────────────────────────────────
# Zone loading (cached)
# ─────────────────────────────────────────────────────────────────────────────


def load_zones_for_camera(
    camera_id: int,
    db,
) -> Dict[str, List[List[int]]]:
    """
    Return { zone_name: [[x,y],...] } for all zones configured for camera_id.
    Results are cached in-process until invalidate_zone_cache() is called.
    If JSON parsing fails for a row, that zone is skipped (logged).
    """
    with _cache_lock:
        if camera_id in _zone_cache:
            return _zone_cache[camera_id]

    # Cache miss — query DB
    from database import ZoneConfig

    try:
        rows = db.query(ZoneConfig).filter(ZoneConfig.camera_id == camera_id).all()
    except Exception as exc:
        log.error(f"[zone_service] DB query failed for cam {camera_id}: {exc}")
        return {}

    zones: Dict[str, List[List[int]]] = {}
    for row in rows:
        try:
            polygon = json.loads(row.polygon_json)
            # Validate structure
            if not isinstance(polygon, list) or len(polygon) < 3:
                log.warning(
                    f"[zone_service] cam={camera_id} zone={row.zone_name!r}: "
                    f"polygon has < 3 points, skipped"
                )
                continue
            zones[row.zone_name] = [[int(p[0]), int(p[1])] for p in polygon]
        except Exception as exc:
            log.warning(
                f"[zone_service] cam={camera_id} zone={row.zone_name!r}: "
                f"parse error ({exc}), skipped"
            )

    with _cache_lock:
        _zone_cache[camera_id] = zones

    log.debug(f"[zone_service] Loaded zones for cam {camera_id}: {list(zones.keys())}")
    return zones


def invalidate_zone_cache(camera_id: int) -> None:
    """
    Bust the cache for camera_id.
    Call after creating or deleting a zone so the daemon picks up the change.
    """
    with _cache_lock:
        _zone_cache.pop(camera_id, None)
    log.info(f"[zone_service] Zone cache invalidated for cam {camera_id}")


def load_zones_without_db(camera_id: int) -> Optional[Dict[str, List[List[int]]]]:
    """
    Return cached zones if available, or None if cache miss (no DB available).
    Used by the detection daemon when no DB session is open yet.
    """
    with _cache_lock:
        return _zone_cache.get(camera_id)
