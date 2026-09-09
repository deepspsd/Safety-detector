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
_CASH_MIN_CONF: float = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class CashState(Enum):
    PENDING        = "pending"
    DEPOSITED      = "deposited"
    SUSPICIOUS     = "suspicious"
    PAYEE_HANDOVER = "payee_handover"


@dataclass
class CashTrack:
    """Per-cash-object tracking state with trajectory history."""
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
    trajectory:           List[Tuple[float, float, float]] = field(default_factory=list)  # [(cx, cy, ts), ...]
    heading:              str = "unknown"  # "to_cashbox", "to_pocket", or "neutral"



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop_person(frame: Optional[np.ndarray], bbox: List) -> Optional[str]:
    """Crop head and upper torso of person for vendor payee photo capture."""
    if frame is None or not bbox or len(bbox) < 4:
        return None
    try:
        import base64
        import cv2
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        # Focus on head + upper torso (upper 65% of person bbox)
        crop_h = max(40, int((y2 - y1) * 0.65))
        cy2 = min(h, y1 + crop_h)
        cx1 = max(0, x1 - 15)
        cx2 = min(w, x2 + 15)
        cy1 = max(0, y1 - 15)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    except Exception as e:
        log.debug(f"[CashTracker] _crop_person error: {e}")
        return None


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


def _evaluate_trajectory_and_pocket(
    cx: float,
    cy: float,
    track: CashTrack,
    cashbox_polygon: Optional[List],
    persons: List[Dict],
    pose_observations: Optional[List[Dict]] = None,
) -> Tuple[Optional[str], str]:
    """
    Evaluate:
      1. Is cash centroid currently inside any person's pocket zone?
         (Uses YOLO-Pose hip/pocket zone if available, otherwise falls back to lower body bbox).
      2. Is cash trajectory heading towards cashbox or towards a person's pocket?

    Returns:
      (associated_person_id, heading) where heading is 'to_cashbox', 'to_pocket', or 'neutral'.
    """
    now = time.monotonic()
    track.trajectory.append((cx, cy, now))
    if len(track.trajectory) > 30:
        track.trajectory = track.trajectory[-30:]

    associated_person_id = None
    persons_pockets = []

    for idx, p in enumerate(persons):
        pbbox = p.get("bbox", [])
        if not pbbox:
            continue
        pid = str(p.get("track_id", f"person_{idx}"))

        # Look for matching pose observation
        pocket_poly = None
        if pose_observations:
            for obs in pose_observations:
                obbox = obs.get("bbox", [])
                if obbox and len(obbox) >= 4:
                    if abs(obbox[0] - pbbox[0]) < 80 and abs(obbox[1] - pbbox[1]) < 80:
                        pocket_poly = obs.get("pocket_zone")
                        break

        if not pocket_poly:
            pocket_poly = _build_body_zone(pbbox)

        persons_pockets.append((pid, pocket_poly))

        if _point_in_polygon(cx, cy, pocket_poly):
            associated_person_id = pid

    heading = "neutral"
    if len(track.trajectory) >= 3:
        start_cx, start_cy = track.trajectory[0][:2]
        latest_cx, latest_cy = track.trajectory[-1][:2]

        # Trajectory towards cashbox
        if cashbox_polygon and len(cashbox_polygon) >= 3:
            cb_x = float(np.mean([pt[0] for pt in cashbox_polygon]))
            cb_y = float(np.mean([pt[1] for pt in cashbox_polygon]))
            d_init_cb = (start_cx - cb_x) ** 2 + (start_cy - cb_y) ** 2
            d_curr_cb = (latest_cx - cb_x) ** 2 + (latest_cy - cb_y) ** 2
            if d_curr_cb < d_init_cb * 0.75:
                heading = "to_cashbox"

        # Trajectory towards person pocket
        for pid, pk_poly in persons_pockets:
            pk_x = float(np.mean([pt[0] for pt in pk_poly]))
            pk_y = float(np.mean([pt[1] for pt in pk_poly]))
            d_init_pk = (start_cx - pk_x) ** 2 + (start_cy - pk_y) ** 2
            d_curr_pk = (latest_cx - pk_x) ** 2 + (latest_cy - pk_y) ** 2
            if d_curr_pk < d_init_pk * 0.75:
                heading = "to_pocket"
                if not associated_person_id:
                    associated_person_id = pid
                break

    track.heading = heading
    return associated_person_id, heading


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

        # Guard: camera_id must refer to an existing row or the INSERT will
        # trigger a FK constraint failure that poisons the SQLAlchemy session
        # and breaks ALL subsequent DB operations on the same connection
        # (including the WebSocket annotation thread).  Use None when the
        # camera doesn't exist (e.g. sentinel id 9999 used by cash_monitor).
        _safe_cam_id: Optional[int] = camera_id
        if camera_id is not None:
            try:
                from database import Camera as _CameraModel
                _exists = db.query(_CameraModel.id).filter(
                    _CameraModel.id == camera_id
                ).first()
                if _exists is None:
                    _safe_cam_id = None
            except Exception:
                _safe_cam_id = None

        save_alert(
            db=db,
            user_id=user_id,
            message=message,
            role="System",
            severity=severity,
            detected_issue=issue,
            confidence=None,
            snapshot_b64=snapshot_b64,
            camera_id=_safe_cam_id,
        )
    except Exception as exc:
        log.error("[CashMonitor] _fire_alert failed: %s", exc)
        # CRITICAL: roll back the poisoned transaction so the session remains
        # usable for subsequent calls (avoids cascading WS session crash).
        try:
            db.rollback()
        except Exception:
            pass


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
        self.last_theft_alert: Optional[str] = None
        self.last_theft_time: float = 0.0
        self.last_payee_snapshot: Optional[str] = None
        self.last_payee_time: float = 0.0
        self.last_payee_id: Optional[str] = None

    # ── public API ───────────────────────────────────────────────────────────

    def update(
        self,
        db,
        cash_detections: List[Dict],
        person_detections: List[Dict],
        cashbox_polygon: Optional[List],
        floor: str,
        frame: Optional[np.ndarray] = None,
        vendor_polygon: Optional[List] = None,
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
        vendor_polygon   : [[x,y],...] vendor / customer zone polygon, or None
        """
        now = time.monotonic()
        is_cash_floor = floor in ("shop", "bakery")

        with self._lock:
            active_ids: set = set()

            # Optional pose estimation for hip/pocket keypoints & trajectories
            pose_obs = None
            if frame is not None and is_cash_floor and cash_detections:
                try:
                    from services.pose_layer import pose_adapter
                    pose_obs = pose_adapter.analyse(frame, [p.get("bbox") for p in person_detections if p.get("bbox")])
                except Exception as _pe:
                    log.debug(f"[CashTracker] pose analysis error: {_pe}")

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

                # ── Check body zone & trajectory (shop / bakery cameras only) ──────────────
                if not is_cash_floor:
                    continue

                person_tid, heading = _evaluate_trajectory_and_pocket(
                    cx, cy, track, cashbox_polygon, person_detections, pose_obs
                )
                if person_tid is not None:
                    if track.in_body_zone_since is None:
                        track.in_body_zone_since = now
                        track.associated_person_id = person_tid
                        log.debug(
                            "[CashTracker] cam=%d cash_id=%s entered pocket zone "
                            "of person %s (heading=%s)",
                            self.camera_id, tid, person_tid, heading,
                        )
                else:
                    # Left body zone without entering cashbox
                    if track.in_body_zone_since is not None and heading != "to_pocket":
                        track.in_body_zone_since = None


            # ── Check Vendor Payee Handover (REQ-SH-2) ────────────────────
            # Triggered when cash is active and a payee is present
            if frame is not None and person_detections and cash_detections:
                payee = self._identify_payee(person_detections, vendor_polygon, cashbox_polygon)
                if payee is not None:
                    # Verify cash is physically near the payee (handover interaction)
                    p_box = payee.get("bbox", [])
                    p_cx, p_cy = _centroid(p_box)
                    p_w = max(50, p_box[2] - p_box[0]) if len(p_box) >= 4 else 100
                    cash_near = any(
                        ((_centroid(c.get("bbox", [0, 0, 0, 0]))[0] - p_cx) ** 2 +
                         (_centroid(c.get("bbox", [0, 0, 0, 0]))[1] - p_cy) ** 2) ** 0.5 < p_w * 2.0
                        for c in cash_detections if c.get("bbox")
                    )
                    if cash_near:
                        payee_tid = str(payee.get("track_id", "payee"))
                        if now - self.last_payee_time >= 30.0:  # 30s cooldown
                            snap = _crop_person(frame, payee.get("bbox", []))
                            if snap:
                                self.last_payee_snapshot = snap
                                self.last_payee_time = now
                                self.last_payee_id = payee_tid
                                self._fire_payee_alert(db, payee_tid, snap)
                                for t in self._tracks.values():
                                    if t.state == CashState.PENDING:
                                        t.state = CashState.PAYEE_HANDOVER

            # ── Check for disappeared cash that was in body zone ──────────
            if is_cash_floor:
                for tid, track in list(self._tracks.items()):
                    if track.state in (CashState.DEPOSITED, CashState.SUSPICIOUS, CashState.PAYEE_HANDOVER):
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

    def _identify_payee(
        self,
        person_detections: List[Dict],
        vendor_polygon: Optional[List],
        cashbox_polygon: Optional[List],
    ) -> Optional[Dict]:
        """
        Identify the payee / vendor person in the shop counter scene.
        1. If vendor_polygon is configured, pick person whose center is in polygon.
        2. If 2+ persons are present, the payee is the person who is NOT the cashier.
        3. If 1 person is present in front of counter, that person is the payee.
        """
        if not person_detections:
            return None

        # 1. Direct zone check
        if vendor_polygon and len(vendor_polygon) >= 3:
            for p in person_detections:
                cx, cy = _centroid(p.get("bbox", []))
                if _point_in_polygon(cx, cy, vendor_polygon):
                    return p

        # 2. Distinguish Cashier vs Payee when multiple persons are present
        if len(person_detections) >= 2:
            # Check uniform
            cashier_cand = None
            for p in person_detections:
                if p.get("has_uniform") or p.get("uniform_prediction") in ("UNIFORM", "Uniform"):
                    cashier_cand = p
                    break
            if cashier_cand:
                for p in person_detections:
                    if p != cashier_cand:
                        return p

            # Proximity to cashbox: cashier is closest to cashbox, payee is the other
            if cashbox_polygon and len(cashbox_polygon) >= 1:
                cb_cx = float(np.mean([pt[0] for pt in cashbox_polygon]))
                cb_cy = float(np.mean([pt[1] for pt in cashbox_polygon]))
                sorted_by_dist = sorted(
                    person_detections,
                    key=lambda p: (
                        (_centroid(p["bbox"])[0] - cb_cx) ** 2 +
                        (_centroid(p["bbox"])[1] - cb_cy) ** 2
                    )
                )
                return sorted_by_dist[1] if len(sorted_by_dist) > 1 else sorted_by_dist[0]

            # Heuristic: person furthest from the camera/center
            return max(person_detections, key=lambda p: _centroid(p["bbox"])[0])

        # 3. Single person without a configured vendor_polygon cannot be a vendor payee handover
        return None

    def _fire_payee_alert(
        self,
        db,
        payee_id: str,
        snapshot_b64: str,
    ) -> None:
        alert_key = f"vendor_payee_{self.camera_id}_{payee_id}"
        if not _can_alert(alert_key):
            return

        utc_now = datetime.datetime.utcnow().strftime("%H:%M:%S")
        msg = (
            f"[VENDOR PAYMENT] Camera {self.camera_id} — "
            f"Payee photo captured during cash handover "
            f"(Payee #{payee_id} at {utc_now} UTC)."
        )
        log.info(msg)
        _fire_alert(
            db=db,
            camera_id=self.camera_id,
            message=msg,
            severity="low",
            issue="Vendor Payee Photo Captured",
            snapshot_b64=snapshot_b64,
        )

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
        traj_note = (
            "Trajectory analysis confirmed: Cash moved toward employee pocket and away from cashbox."
            if track.heading == "to_pocket"
            else "Cash object entered employee pocket zone."
        )
        msg = (
            f"[CASH THEFT ALERT] Camera {self.camera_id} — "
            f"{traj_note} Cash disappeared without reaching the cashbox "
            f"(employee track_id={person_id}, detected at {utc_now} UTC). "
            f"Please review CCTV footage immediately."
        )
        log.warning(msg)
        self.last_theft_alert = msg
        self.last_theft_time = time.monotonic()
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
# Public entry points called from camera_manager.py and cctv.py
# ─────────────────────────────────────────────────────────────────────────────

def check_cash_zone(
    db,
    camera_id: int,
    detections: List[Dict],
    persons: Optional[List[Dict]] = None,
    cashbox_polygon: Optional[List] = None,
    floor: str = "shop",
    frame: Optional[np.ndarray] = None,
    vendor_polygon: Optional[List] = None,
) -> Dict:
    """
    Main entry point — call every detection frame for cameras covering
    the shop counter / cashbox zone.

    Returns dict containing status, theft_alert, and payee snapshot.
    """
    # Extract Cash detections
    cash_dets = [
        d for d in detections
        if _is_cash_label(str(d.get("label", "")))
        and float(d.get("confidence", 0.0)) >= _CASH_MIN_CONF
    ]
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
        vendor_polygon=vendor_polygon,
    )

    now = time.monotonic()
    return {
        "cash_detected": bool(cash_dets),
        "theft_alert": tracker.last_theft_alert if (now - tracker.last_theft_time < 6.0) else None,
        "payee_detected": bool(tracker.last_payee_snapshot and (now - tracker.last_payee_time < 12.0)),
        "payee_snapshot_b64": tracker.last_payee_snapshot if (now - tracker.last_payee_time < 12.0) else None,
        "payee_id": tracker.last_payee_id,
    }


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
