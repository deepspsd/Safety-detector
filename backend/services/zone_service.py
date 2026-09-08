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


_zone_details_cache: Dict[int, Dict[str, Dict[str, Any]]] = {}


def load_zone_details_for_camera(
    camera_id: int,
    db,
) -> Dict[str, Dict[str, Any]]:
    """
    Return full zone details { zone_name: { id, polygon, capabilities, zone_models, zone_type } }
    for all zones configured for camera_id. Cached until invalidate_zone_cache().
    """
    with _cache_lock:
        if camera_id in _zone_details_cache:
            return _zone_details_cache[camera_id]

    from database import ZoneConfig

    try:
        rows = db.query(ZoneConfig).filter(ZoneConfig.camera_id == camera_id).all()
    except Exception as exc:
        log.error(f"[zone_service] DB query failed for cam {camera_id}: {exc}")
        return {}

    details: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            polygon = json.loads(row.polygon_json)
            if not isinstance(polygon, list) or len(polygon) < 3:
                continue
            caps = []
            if getattr(row, "capabilities_json", None):
                try:
                    caps = json.loads(row.capabilities_json)
                except Exception:
                    caps = []
            z_models = []
            if getattr(row, "zone_models_json", None):
                try:
                    z_models = json.loads(row.zone_models_json)
                except Exception:
                    z_models = []

            details[row.zone_name] = {
                "id": row.id,
                "zone_name": row.zone_name,
                "zone_type": getattr(row, "zone_type", None) or row.zone_name,
                "polygon": [[int(p[0]), int(p[1])] for p in polygon],
                "capabilities": caps,
                "zone_models": z_models,
            }
        except Exception as exc:
            log.warning(f"[zone_service] parse error for zone {row.zone_name}: {exc}")

    with _cache_lock:
        _zone_details_cache[camera_id] = details

    return details


def load_zone_details_without_db(camera_id: int) -> Optional[Dict[str, Dict[str, Any]]]:
    """Return cached zone details if available without querying DB."""
    with _cache_lock:
        return _zone_details_cache.get(camera_id)


def invalidate_zone_cache(camera_id: int) -> None:
    """
    Bust the cache for camera_id.
    Call after creating or deleting a zone so the daemon picks up the change.
    """
    with _cache_lock:
        _zone_cache.pop(camera_id, None)
        _zone_details_cache.pop(camera_id, None)
    log.info(f"[zone_service] Zone cache invalidated for cam {camera_id}")


def load_zones_without_db(camera_id: int) -> Optional[Dict[str, List[List[int]]]]:
    """
    Return cached zones if available, or None if cache miss (no DB available).
    Used by the detection daemon when no DB session is open yet.
    """
    with _cache_lock:
        return _zone_cache.get(camera_id)


# ─────────────────────────────────────────────────────────────────────────────
# Zone transition tracker
# ─────────────────────────────────────────────────────────────────────────────

# Structure: { (camera_id, track_id): zone_name | None }
_zone_state: Dict[Tuple[int, str], Optional[str]] = {}
_zone_state_lock = threading.Lock()


def compute_zone_for_track(
    camera_id: int,
    track_id: str,
    cx: float,
    cy: float,
    zones: Dict[str, List[List[int]]],
) -> Optional[str]:
    """Return the zone name that contains point (cx, cy), or None."""
    for zone_name, polygon in zones.items():
        if point_in_zone(cx, cy, polygon):
            return zone_name
    return None


def update_track_zone(
    camera_id: int,
    track_id: str,
    cx: float,
    cy: float,
    zones: Dict[str, List[List[int]]],
) -> Optional[Dict]:
    """Update zone state for a track and emit ENTER_ZONE / EXIT_ZONE events.

    Returns a dict describing the transition, or None if no zone change occurred.

    Dict keys: event_type, track_id, camera_id, zone, previous_zone
    """
    current_zone = compute_zone_for_track(camera_id, track_id, cx, cy, zones)
    key = (camera_id, track_id)

    with _zone_state_lock:
        previous_zone = _zone_state.get(key)

        if current_zone == previous_zone:
            return None  # no change

        _zone_state[key] = current_zone

    # Determine event type
    if previous_zone is not None and current_zone is None:
        event_type = "EXIT_ZONE"
        zone = previous_zone
    elif previous_zone is None and current_zone is not None:
        event_type = "ENTER_ZONE"
        zone = current_zone
    else:
        event_type = "ZONE_CHANGE"
        zone = current_zone

    transition = {
        "event_type": event_type,
        "track_id": track_id,
        "camera_id": camera_id,
        "zone": zone,
        "previous_zone": previous_zone,
    }

    # Emit platform event (best-effort — never crashes the detection loop)
    try:
        from services.platform_events import emit
        emit(
            event_type,
            camera_id=camera_id,
            track_id=track_id,
            payload={"zone": zone, "previous_zone": previous_zone},
            source="zone-engine",
        )
    except Exception as exc:
        log.warning("[zone_service] Could not emit zone event: %s", exc)

    log.debug(
        "[zone_service] cam=%s track=%s %s → %s",
        camera_id, track_id, previous_zone, current_zone,
    )
    return transition


def reset_track_zone(camera_id: int, track_id: str) -> None:
    """Remove zone state when a track is lost."""
    with _zone_state_lock:
        _zone_state.pop((camera_id, track_id), None)


# ─────────────────────────────────────────────────────────────────────────────
# Virtual line crossing
# ─────────────────────────────────────────────────────────────────────────────

# A virtual line is defined by two points: [[x1,y1],[x2,y2]]
# Crossing direction is determined by which side of the line the center moves from.


def _side_of_line(
    px: float, py: float,
    lx1: float, ly1: float,
    lx2: float, ly2: float,
) -> float:
    """Return the signed cross-product indicating which side of the line the point is on."""
    return (lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1)


def check_line_crossing(
    camera_id: int,
    track_id: str,
    prev_center: Tuple[float, float],
    curr_center: Tuple[float, float],
    lines: Dict[str, List[List[int]]],
) -> Optional[Dict]:
    """Detect if a track has crossed any configured virtual line.

    Args:
        camera_id:   Camera identifier.
        track_id:    Track identifier.
        prev_center: Previous (cx, cy) of the track.
        curr_center: Current (cx, cy) of the track.
        lines:       { line_name: [[x1,y1],[x2,y2]] } from VirtualLine DB rows.

    Returns:
        Dict with keys (line_name, direction='INWARD'|'OUTWARD', camera_id, track_id),
        or None if no crossing was detected.
    """
    if prev_center is None or curr_center is None:
        return None

    px, py = prev_center
    cx, cy = curr_center

    for line_name, pts in lines.items():
        if len(pts) < 2:
            continue
        lx1, ly1 = pts[0][0], pts[0][1]
        lx2, ly2 = pts[1][0], pts[1][1]

        side_prev = _side_of_line(px, py, lx1, ly1, lx2, ly2)
        side_curr = _side_of_line(cx, cy, lx1, ly1, lx2, ly2)

        if side_prev == 0 or side_curr == 0:
            continue  # on the line — skip
        if (side_prev > 0) == (side_curr > 0):
            continue  # same side — no crossing

        # Crossing detected
        # Convention: moving from positive to negative side = INWARD
        direction = "INWARD" if side_prev > 0 else "OUTWARD"

        event_payload = {
            "line_name": line_name,
            "direction": direction,
            "camera_id": camera_id,
            "track_id": track_id,
        }

        try:
            from services.platform_events import emit
            emit(
                f"{direction}_MOVEMENT",
                camera_id=camera_id,
                track_id=track_id,
                payload=event_payload,
                source="zone-engine",
            )
        except Exception as exc:
            log.warning("[zone_service] Could not emit line-crossing event: %s", exc)

        log.debug(
            "[zone_service] cam=%s track=%s crossed line '%s' → %s",
            camera_id, track_id, line_name, direction,
        )
        return event_payload

    return None


def load_virtual_lines(camera_id: int, db) -> Dict[str, List[List[int]]]:
    """Load VirtualLine rows for *camera_id* from the database.

    Returns { line_name: [[x1,y1],[x2,y2]] }.
    Skips malformed rows silently.
    """
    try:
        from database import VirtualLine
        rows = db.query(VirtualLine).filter(VirtualLine.camera_id == camera_id).all()
    except Exception as exc:
        log.error("[zone_service] load_virtual_lines DB error cam=%s: %s", camera_id, exc)
        return {}

    lines: Dict[str, List[List[int]]] = {}
    for row in rows:
        try:
            pts = json.loads(row.points_json)
            if isinstance(pts, list) and len(pts) >= 2:
                lines[row.name] = [[int(p[0]), int(p[1])] for p in pts[:2]]
        except Exception as exc:
            log.warning("[zone_service] VirtualLine parse error row=%s: %s", row.id, exc)
    return lines
