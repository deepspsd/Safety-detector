"""
services/cash_monitor.py — Cash-zone & Stock-zone monitoring
=============================================================
v2: Multi-frame CashEventTracker state machine.

Monitors:
  • CashEventTracker  — tracks Cash objects (class 14) across frames,
                        classifies each as DEPOSITED (cashbox) or
                        SUSPICIOUS (moved toward employee body zone then
                        disappeared without reaching cashbox).
                        Fires theft alert only on SUSPICIOUS confirmation.

  • StockMonitor      — alerts when Exposed-Item class is detected
                        (unchanged from v1).

CashEventTracker state machine (per cash track_id):
  PENDING
    → DEPOSITED   if cash centroid enters cashbox polygon and stays ≥ 1 s
    → SUSPICIOUS  if cash centroid enters an employee body zone, then
                  disappears (bbox gone) for ≥ CONFIRM_SECS without
                  having reached the cashbox

Design notes
────────────
• Body-zone check is ONLY active for shop-floor cameras.
• "Employee body zone" = lower 60% of each tracked Person bbox,
  expanded ×1.15 horizontally — no extra polygon needed.
• One CashEventTracker singleton per camera_id, stored in _tracker_registry.
• All per-camera state is isolated; thread-safe via per-tracker lock.
• alert_service.save_alert() is called once per confirmed SUSPICIOUS event,
  with a 90 s cooldown per (camera_id, employee_track_id) pair.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("cash_monitor")

# ─────────────────────────────────────────────────────────────────────────────
# Tuneable constants
# ─────────────────────────────────────────────────────────────────────────────

# How many seconds cash must stay in cashbox zone to count as DEPOSITED
_DEPOSIT_CONFIRM_SECS: float = 1.0

# How many seconds cash must be gone (after entering body zone) to confirm SUSPICIOUS
_THEFT_CONFIRM_SECS: float = 2.0

# Seconds before a cash track is declared lost if bbox disappears
_TRACK_TIMEOUT_SECS: float = 5.0

# Body zone: lower this fraction of person bbox is the pocket/body region
_BODY_ZONE_LOWER_FRAC: float = 0.60

# Body zone horizontal expansion factor
_BODY_ZONE_X_EXPAND: float = 1.15

# Cooldown between alerts per (camera_id, employee_track_id)
_ALERT_COOLDOWN_SECS: float = 90.0

# Cash class label strings — includes exact model output + common lowercase variants
_CASH_LABELS = {"Cash", "cash", "banknote", "money", "note", "currency", "rupee", "bill"}
_CASH_LABELS.update({f"{v} bgn" for v in (5, 10, 20, 50, 100)})
_CASH_LABELS.update({f"{v} eur" for v in (5, 10, 20, 50, 100)})
_CASH_LABELS.update({f"{v} inr" for v in (10, 20, 50, 100, 200, 500, 2000)})


def _is_cash_label(label: str) -> bool:
    if not label:
        return False
    lbl = label.lower()
    if lbl in _CASH_LABELS:
        return True
    return any(lbl.endswith(suf) for suf in (" bgn", " eur", " inr", " usd", " gbp", " aed", " pkr", " bdt"))


# Minimum confidence to treat a detection as real Cash.
# The global DETECTION_CONF=0.30 is intentionally low for PPE.
# Cash false-positives (random boxes labelled Cash) are suppressed here.
_CASH_MIN_CONF: float = 0.65


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class CashState(Enum):
    PENDING    = "pending"
    DEPOSITED  = "deposited"
    SUSPICIOUS = "suspicious"


@dataclass
class CashTrack:
    """Per-cash-object tracking state."""
    track_id: str
    state: CashState = CashState.PENDING
    # Timestamps
    first_seen:           float = field(default_factory=time.monotonic)
    last_seen:            float = field(default_factory=time.monotonic)
    in_cashbox_since:     Optional[float] = None   # when it entered cashbox
    in_body_zone_since:   Optional[float] = None   # when it entered a body zone
    disappeared_at:       Optional[float] = None   # when bbox was last seen
    associated_person_id: Optional[str]  = None   # person track_id it was near
    last_bbox:            Optional[List]  = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _centroid(bbox: List) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_in_polygon(cx: float, cy: float, polygon: List[List]) -> bool:
    """Ray-casting point-in-polygon test."""
    if not polygon or len(polygon) < 3:
        return False
    try:
        import cv2
        import numpy as np
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0
    except Exception:
        # Fallback pure-Python ray casting
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > cy) != (yj > cy)) and (
                cx < (xj - xi) * (cy - yi) / ((yj - yi) or 1e-9) + xi
            ):
                inside = not inside
            j = i
        return inside


def _build_body_zone(person_bbox: List) -> List[List[int]]:
    """
    Derive the pocket/body zone from the lower 60% of a person's bounding box.
    Expanded horizontally by _BODY_ZONE_X_EXPAND to capture arm reach.
    Returns a 4-point polygon [[x,y], ...].
    """
    x1, y1, x2, y2 = [int(v) for v in person_bbox[:4]]
    h = y2 - y1
    body_y1 = y1 + int(h * (1.0 - _BODY_ZONE_LOWER_FRAC))
    body_y2 = y2

    # Expand horizontally
    cx = (x1 + x2) / 2.0
    half_w = (x2 - x1) / 2.0 * _BODY_ZONE_X_EXPAND
    bx1 = int(cx - half_w)
    bx2 = int(cx + half_w)

    return [
        [bx1, body_y1],
        [bx2, body_y1],
        [bx2, body_y2],
        [bx1, body_y2],
    ]


def _cash_in_body_zone(
    cash_bbox: List,
    persons: List[Dict],
) -> Optional[str]:
    """
    Return the track_id of the first employee whose body zone contains the
    cash centroid, or None.
    """
    cx, cy = _centroid(cash_bbox)
    for person in persons:
        pbbox = person.get("bbox", [])
        if not pbbox:
            continue
        body_zone = _build_body_zone(pbbox)
        if _point_in_polygon(cx, cy, body_zone):
            return str(person.get("track_id", "unknown"))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Alert cooldown registry
# ─────────────────────────────────────────────────────────────────────────────

_alert_cooldown_registry: Dict[str, float] = {}
_alert_cooldown_lock = threading.Lock()


def _can_alert(key: str) -> bool:
    now = time.monotonic()
    with _alert_cooldown_lock:
        last = _alert_cooldown_registry.get(key, 0.0)
        if now - last < _ALERT_COOLDOWN_SECS:
            return False
        _alert_cooldown_registry[key] = now
    return True


def _fire_alert(db, camera_id: int, message: str, severity: str, issue: str,
                snapshot_b64: Optional[str] = None) -> None:
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
            snapshot_b64=snapshot_b64,
            camera_id=camera_id,
        )
    except Exception as exc:
        log.error("[CashMonitor] _fire_alert failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# CashEventTracker — one per camera
# ─────────────────────────────────────────────────────────────────────────────

class CashEventTracker:
    """
    Tracks Cash objects across frames for a single camera.

    Call update() every frame (or every N frames) from the detection loop.
    It mutates internal state and fires alerts via _fire_alert when a
    SUSPICIOUS event is confirmed.
    """

    def __init__(self, camera_id: int) -> None:
        self.camera_id = camera_id
        self._tracks: Dict[str, CashTrack] = {}
        self._lock = threading.Lock()

    # ── public API ───────────────────────────────────────────────────────────

    def update(
        self,
        db,
        cash_detections: List[Dict],
        person_detections: List[Dict],
        cashbox_polygon: Optional[List],
        floor: str,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """
        Process one frame of detections.

        Parameters
        ----------
        cash_detections  : list of dicts with keys label, bbox, track_id, confidence
        person_detections: list of dicts with keys label, bbox, track_id, confidence
                           (persons must have track_id assigned by ByteTrack)
        cashbox_polygon  : [[x,y],...] polygon in pixel coords, or None
        floor            : camera floor string — body-zone check only on 'shop'
        frame            : optional BGR ndarray for snapshot on alert
        """
        now = time.monotonic()
        is_cash_floor = floor in ("shop", "bakery")

        with self._lock:
            active_ids: set = set()

            for det in cash_detections:
                label = str(det.get("label", ""))
                if not _is_cash_label(label):
                    continue

                # ── Confidence gate — suppress false positives ──────────────
                conf = float(det.get("confidence", 0.0))
                if conf < _CASH_MIN_CONF:
                    log.debug(
                        "[CashTracker] low-conf Cash dropped: %.2f < %.2f",
                        conf, _CASH_MIN_CONF,
                    )
                    continue

                # Assign a stable track id from the detection dict
                raw_tid = det.get("track_id")
                if raw_tid is None:
                    # Fall back to bbox-hash if no tracker id (happens when
                    # ByteTrack hasn't confirmed the object yet)
                    bbox = det.get("bbox", [])
                    raw_tid = f"bbox_{int(bbox[0])}_{int(bbox[1])}" if bbox else "unk"
                tid = str(raw_tid)
                active_ids.add(tid)

                bbox = det.get("bbox", [])
                if not bbox:
                    continue

                track = self._tracks.setdefault(tid, CashTrack(track_id=tid))
                track.last_seen = now
                track.last_bbox = bbox

                if track.state in (CashState.DEPOSITED, CashState.SUSPICIOUS):
                    continue  # terminal state — no more processing

                cx, cy = _centroid(bbox)

                # ── Check cashbox ──────────────────────────────────────────
                in_cashbox = (
                    cashbox_polygon is not None
                    and _point_in_polygon(cx, cy, cashbox_polygon)
                )
                if in_cashbox:
                    if track.in_cashbox_since is None:
                        track.in_cashbox_since = now
                    elif now - track.in_cashbox_since >= _DEPOSIT_CONFIRM_SECS:
                        track.state = CashState.DEPOSITED
                        log.info(
                            "[CashTracker] cam=%d cash_id=%s DEPOSITED in cashbox",
                            self.camera_id, tid,
                        )
                    # Cash is heading to cashbox — reset body-zone state
                    track.in_body_zone_since = None
                    track.disappeared_at = None
                    continue
                else:
                    track.in_cashbox_since = None

                # ── Check body zone (shop / bakery cameras only) ──────────────
                if not is_cash_floor:
                    continue

                person_tid = _cash_in_body_zone(bbox, person_detections)
                if person_tid is not None:
                    if track.in_body_zone_since is None:
                        track.in_body_zone_since = now
                        track.associated_person_id = person_tid
                        log.debug(
                            "[CashTracker] cam=%d cash_id=%s entered body zone "
                            "of person %s",
                            self.camera_id, tid, person_tid,
                        )
                else:
                    # Left body zone without entering cashbox
                    if track.in_body_zone_since is not None:
                        track.in_body_zone_since = None

            # ── Check for disappeared cash that was in body zone ──────────
            if is_cash_floor:
                for tid, track in list(self._tracks.items()):
                    if track.state in (CashState.DEPOSITED, CashState.SUSPICIOUS):
                        continue
                    if tid in active_ids:
                        track.disappeared_at = None
                        continue
                    # bbox not seen this frame
                    if track.in_body_zone_since is None:
                        # Not previously in body zone — just a normal track loss
                        if now - track.last_seen > _TRACK_TIMEOUT_SECS:
                            self._tracks.pop(tid, None)
                        continue
                    # Was in body zone — start disappearance timer
                    if track.disappeared_at is None:
                        track.disappeared_at = now
                    elif now - track.disappeared_at >= _THEFT_CONFIRM_SECS:
                        # Confirmed: entered body zone, then vanished → SUSPICIOUS
                        track.state = CashState.SUSPICIOUS
                        self._fire_theft_alert(db, track, frame)

            # ── Expire very old DEPOSITED / SUSPICIOUS tracks ─────────────
            expired = [
                tid for tid, t in self._tracks.items()
                if t.state != CashState.PENDING
                and now - t.last_seen > 30.0
            ]
            for tid in expired:
                self._tracks.pop(tid, None)

    # ── private ──────────────────────────────────────────────────────────────

    def _fire_theft_alert(
        self,
        db,
        track: CashTrack,
        frame: Optional[np.ndarray],
    ) -> None:
        person_id = track.associated_person_id or "unknown"
        alert_key = f"cash_theft_{self.camera_id}_{person_id}"

        if not _can_alert(alert_key):
            log.debug("[CashTracker] alert suppressed by cooldown: %s", alert_key)
            return

        snapshot_b64 = None
        if frame is not None:
            try:
                import base64
                import cv2 as _cv2
                _, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 75])
                snapshot_b64 = base64.b64encode(buf).decode()
            except Exception:
                pass

        utc_now = datetime.datetime.utcnow().strftime("%H:%M:%S")
        msg = (
            f"[CASH THEFT ALERT] Camera {self.camera_id} — "
            f"Cash object entered employee body zone and disappeared "
            f"without reaching the cashbox "
            f"(employee track_id={person_id}, detected at {utc_now} UTC). "
            f"Please review CCTV footage immediately."
        )
        log.warning(msg)
        _fire_alert(
            db=db,
            camera_id=self.camera_id,
            message=msg,
            severity="critical",
            issue="Cash theft suspected",
            snapshot_b64=snapshot_b64,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-camera tracker registry
# ─────────────────────────────────────────────────────────────────────────────

_tracker_registry: Dict[int, CashEventTracker] = {}
_registry_lock = threading.Lock()


def get_tracker(camera_id: int) -> CashEventTracker:
    """Return (or lazily create) the CashEventTracker for a camera."""
    with _registry_lock:
        if camera_id not in _tracker_registry:
            _tracker_registry[camera_id] = CashEventTracker(camera_id)
        return _tracker_registry[camera_id]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points called from camera_manager.py
# ─────────────────────────────────────────────────────────────────────────────

def check_cash_zone(
    db,
    camera_id: int,
    detections: List[Dict],
    persons: Optional[List[Dict]] = None,
    cashbox_polygon: Optional[List] = None,
    floor: str = "shop",
    frame: Optional[np.ndarray] = None,
) -> None:
    """
    Main entry point — call every detection frame for cameras covering
    the shop counter / cashbox zone.

    Parameters
    ----------
    db              : SQLAlchemy session
    camera_id       : camera row ID
    detections      : all YOLO detections this frame (any class)
    persons         : tracked person dicts (must include track_id from ByteTrack)
    cashbox_polygon : [[x,y], ...] cashbox zone polygon, or None
    floor           : 'shop' enables pocket/body-zone theft detection
    frame           : BGR ndarray for snapshot on alert (optional)
    """
    # Extract Cash detections (class label 'Cash' or lowercase variants)
    # Pre-filter by confidence here too so low-conf detections never reach the tracker
    cash_dets = [
        d for d in detections
        if _is_cash_label(str(d.get("label", "")))
        and float(d.get("confidence", 0.0)) >= _CASH_MIN_CONF
    ]

    # Also accept the numeric class id=14 if label not set (with same conf gate)
    cash_dets += [
        d for d in detections
        if d.get("class_id") == 14
        and d not in cash_dets
        and float(d.get("confidence", 0.0)) >= _CASH_MIN_CONF
    ]

    tracker = get_tracker(camera_id)
    tracker.update(
        db=db,
        cash_detections=cash_dets,
        person_detections=persons or [],
        cashbox_polygon=cashbox_polygon,
        floor=floor,
        frame=frame,
    )


# ─────────────────────────────────────────────────────────────────────────────
# StockMonitor (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

# Cooldown registry for stock alerts
_last_stock_alert_ts: Dict[str, float] = {}
_STOCK_COOLDOWN_SEC = 120


def _stock_can_alert(key: str) -> bool:
    now = time.time()
    last = _last_stock_alert_ts.get(key, 0)
    if now - last < _STOCK_COOLDOWN_SEC:
        return False
    _last_stock_alert_ts[key] = now
    return True


def check_stock_zone(
    db,
    camera_id: int,
    detections: List[Dict],
    stock_polygon: Optional[List] = None,
) -> None:
    """
    Alert when Exposed-Item class is detected inside the stock zone polygon.
    """
    exposed_items = [d for d in detections if d.get("label") == "Exposed-Item"]

    if not exposed_items:
        return

    if stock_polygon:
        def _centroid_in_zone(d: Dict) -> bool:
            bbox = d.get("bbox", [])
            if not bbox:
                return False
            cx, cy = _centroid(bbox)
            return _point_in_polygon(cx, cy, stock_polygon)

        exposed_items = [d for d in exposed_items if _centroid_in_zone(d)]

    if exposed_items:
        key = f"stock_exposed_{camera_id}"
        if _stock_can_alert(key):
            msg = (
                f"[STOCK ZONE] Camera {camera_id} — {len(exposed_items)} exposed "
                f"item(s) detected in open stock area at "
                f"{datetime.datetime.utcnow().strftime('%H:%M')}. "
                "Cover or secure goods immediately."
            )
            log.warning(msg)
            _fire_alert(db, camera_id, msg, "medium", "Exposed stock detected")
