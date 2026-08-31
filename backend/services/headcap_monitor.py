"
headcap_monitor.py -- Per-person Bakery Head-Cap compliance state machine
=========================================================================
Implements the GLOBAL head-cap compliance rule:

    EVERY TRACKED PERSON must wear a Bakery Head Cap.
    Zone, polygon, floor -- NONE of these affect the rule.

State machine per (camera_id, track_id):

    COMPLIANT
        |  headcap missing (frame-level)
        v
    PENDING_MISSING
        |  headcap seen again          -> COMPLIANT  (timer reset)
        |  missing >= HEADCAP_MISSING_SECONDS
        v
    ALERT_TRIGGERED
        |  headcap seen again          -> COMPLIANT
        |  (same violation, cooldown)  -> no new alert

Configurable via config.py (and optionally .env):
    HEADCAP_MISSING_SECONDS   -- persistence threshold (default 3.0 s)
    HEADCAP_ALERT_COOLDOWN    -- repeat-alert suppression (default 30.0 s)
    HEADCAP_CONF_THRESHOLD    -- min YOLO confidence for valid cap (default 0.25)

Integration:
    Called from camera_manager._detection_loop() after obtaining tracked
    persons and raw YOLO detections.

    from services.headcap_monitor import headcap_monitor
    headcap_monitor.update_camera(camera_id, persons, raw_dets, db, frame)

Public API:
    update_camera(camera_id, persons, raw_dets, db, frame) -> list[PersonCapStatus]
    get_debug_info(camera_id)                              -> list[dict]
    reset_camera(camera_id)                                -> None
"

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import settings

log = logging.getLogger(headcap_monitor)

# -- State constants -----------------------------------------------------------
STATE_COMPLIANT = COMPLIANT
STATE_PENDING   = PENDING_MISSING
STATE_ALERT     = ALERT_TRIGGERED

RULE_MSG = Not Wearing Head Cap

# What fraction of the person height is the head region (top N%)
_HEAD_REGION_RATIO = 0.35


# -- Head-region helpers -------------------------------------------------------

def _head_region(person_box: List[int]) -> List[int]:
    "Return [x1, y1, x2, y2] for the top 35% of the person bbox."
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    w = x2 - x1
    margin_x = int(w * 0.10)
    head_y2 = y1 + int(h * _HEAD_REGION_RATIO)
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
    "
    True if cap_box can be associated to person_box as a Bakery Head Cap.

    Association criteria (ANY of):
      1. cap centroid falls inside the person head region (top 35%)
      2. IoU(head_region, cap_box) >= 0.05
      3. cap centroid inside full person box AND in upper half vertically
    "
    if cap_conf < settings.HEADCAP_CONF_THRESHOLD:
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
    "
    For each tracked person: find the best Bakery-Head-Cap detection.
    Returns list of (person_dict, has_headcap, confidence, cap_bbox).

    - Uses head-region association (NOT full-body box).
    - Each cap is claimed by the best-scoring person only.
    - NO Mode-2 absence logic -- absence is handled by the timer in update_camera.
    "
    cap_dets = [
        d for d in raw_dets
        if d.get(label) == Bakery-Head-Cap
        and d.get(confidence, 0) >= settings.HEADCAP_CONF_THRESHOLD
    ]

    results: List[Tuple[dict, bool, float, Optional[List[int]]]] = []
    claimed: set = set()

    for person in persons:
        p_box = person.get(bbox, [])
        if not p_box or len(p_box) != 4:
            results.append((person, False, 0.0, None))
            continue

        best_score = -1.0
        best_cap = None
        best_idx = -1

        for idx, cap in enumerate(cap_dets):
            if idx in claimed:
                continue
            c_box = cap.get(bbox, [])
            c_conf = cap.get(confidence, 0.0)
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
            results.append((person, True, best_cap[confidence], best_cap[bbox]))
        else:
            results.append((person, False, 0.0, None))

    return results


# -- Per-person state ----------------------------------------------------------

@dataclass
class _PersonState:
    track_id: str
    state: str = STATE_COMPLIANT
    missing_since: Optional[float] = None
    last_alert_at: Optional[float] = None
    has_headcap: bool = False
    cap_confidence: float = 0.0
    person_bbox: List[int] = field(default_factory=list)
    head_region: List[int] = field(default_factory=list)
    cap_bbox: Optional[List[int]] = None


@dataclass
class PersonCapStatus:
    camera_id: int
    track_id: str
    has_headcap: bool
    state: str
    missing_seconds: float
    alert_fired: bool
    cap_confidence: float
    cap_bbox: Optional[List[int]]
    person_bbox: List[int]
    head_region: List[int]


# -- Monitor ------------------------------------------------------------------

class HeadCapMonitor:
    "
    Global, zone-independent head-cap compliance monitor.
    Thread-safe singleton, one instance shared across all cameras.
    "

    def __init__(self) -> None:
        self._state: Dict[Tuple[int, str], _PersonState] = {}
        self._lock = threading.Lock()

    def update_camera(
        self,
        camera_id: int,
        persons: List[dict],
        raw_dets: List[dict],
        db,
        frame: Optional[np.ndarray] = None,
    ) -> List[PersonCapStatus]:
        "
        Main entry point: call once per detection-loop tick.
        Returns list of PersonCapStatus for debug overlay.
        "
        now = time.monotonic()
        association_results = associate_headcaps_to_persons(persons, raw_dets)
        statuses: List[PersonCapStatus] = []
        pending_alerts: List[PersonCapStatus] = []
        active_keys: set = set()

        for person, has_headcap, cap_conf, cap_bbox in association_results:
            track_id = str(person.get(track_id, "))
 if not track_id or track_id == -1:
 continue

 key = (camera_id, track_id)
 active_keys.add(key)
 p_box = person.get(bbox, [])
 h_rgn = _head_region(p_box) if p_box else []

 with self._lock:
 ps = self._state.get(key)
 if ps is None:
 ps = _PersonState(track_id=track_id)
 self._state[key] = ps

 ps.has_headcap = has_headcap
 ps.cap_confidence = cap_conf
 ps.person_bbox = p_box
 ps.head_region = h_rgn
 ps.cap_bbox = cap_bbox
 alert_fired = False

 if has_headcap:
 if ps.state != STATE_COMPLIANT:
 log.debug(
 [HeadCap] cam=%d track=%s COMPLIANT (conf=%.2f),
 camera_id, track_id, cap_conf,
 )
 ps.state = STATE_COMPLIANT
 ps.missing_since = None
 else:
 if ps.missing_since is None:
 ps.missing_since = now
 ps.state = STATE_PENDING
 log.debug(
 [HeadCap] cam=%d track=%s PENDING_MISSING,
 camera_id, track_id,
 )

 missing_secs = now - ps.missing_since

 if missing_secs >= settings.HEADCAP_MISSING_SECONDS:
 ps.state = STATE_ALERT
 since_last = (
 (now - ps.last_alert_at)
 if ps.last_alert_at is not None
 else float(inf)
 )
 if since_last >= settings.HEADCAP_ALERT_COOLDOWN:
 ps.last_alert_at = now
 alert_fired = True
 log.warning(
 [HeadCap] ALERT cam=%d track=%s missing=%.1fs,
 camera_id, track_id, missing_secs,
 )

 st = PersonCapStatus(
 camera_id=camera_id,
 track_id=track_id,
 has_headcap=has_headcap,
 state=ps.state,
 missing_seconds=(now - ps.missing_since) if ps.missing_since else 0.0,
 alert_fired=alert_fired,
 cap_confidence=cap_conf,
 cap_bbox=cap_bbox,
 person_bbox=p_box,
 head_region=h_rgn,
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
 "Snapshot of per-person state for overlay rendering."
 now = time.monotonic()
 with self._lock:
 return [
 {
 track_id: ps.track_id,
 state: ps.state,
 has_headcap: ps.has_headcap,
 cap_confidence: round(ps.cap_confidence, 3),
 missing_seconds: (
 round(now - ps.missing_since, 2)
 if ps.missing_since else 0.0
 ),
 person_bbox: ps.person_bbox,
 head_region: ps.head_region,
 cap_bbox: ps.cap_bbox,
 }
 for (cam_id, _), ps in self._state.items()
 if cam_id == camera_id
 ]

 def reset_camera(self, camera_id: int) -> None:
 with self._lock:
 for key in [k for k in self._state if k[0] == camera_id]:
 del self._state[key]
 log.info([HeadCap] State reset cam=%d, camera_id)

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
 db,
 frame: Optional[np.ndarray],
 ) -> None:
 try:
 import base64
 import cv2
 from services.alert_service import save_alert
 from services.rule_engine import _get_rule_engine_user_id

 uid = _get_rule_engine_user_id(db)

 snapshot_b64 = None
 if frame is not None:
 try:
 _, buf = cv2.imencode(
 .jpg, frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
 )
 snapshot_b64 = (
 data:image/jpeg;base64,
 + base64.b64encode(buf).decode(utf-8)
 )
 except Exception as snap_err:
 log.debug([HeadCap] snapshot err: %s, snap_err)

 msg = (
 fPerson #{track_id} is NOT wearing a Bakery Head Cap 
 f(missing {missing_seconds:.1f}s, camera {camera_id})
 )
 save_alert(
 db=db,
 user_id=uid,
 message=msg,
 role=Bakery Worker,
 severity=critical,
 detected_issue=RULE_MSG,
 confidence=None,
 snapshot_b64=snapshot_b64,
 camera_id=camera_id,
 confidence_tier=high,
 )
 log.info(
 [HeadCap] Alert saved cam=%d track=%s missing=%.1fs,
 camera_id, track_id, missing_seconds,
 )
 except Exception as exc:
 log.error(
 [HeadCap] Alert save error cam=%d track=%s: %s,
 camera_id, track_id, exc,
 )


# -- Module-level singleton ---------------------------------------------------
headcap_monitor = HeadCapMonitor()
