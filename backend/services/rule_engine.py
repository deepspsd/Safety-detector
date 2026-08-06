"""
rule_engine.py — Non-ML, time/state-based factory compliance rules
===================================================================
All rules in this module are purely engineering logic — no YOLO or ML models
are called here.  The rules consume the output of yolo_service.py (detection
dicts) and camera_manager.py (raw frames) and decide whether to fire an alert.

Sections
────────
  A. Settings loader        — DB-backed threshold cache (replaces hardcoded constants)
  B. Shift-start checker    — did any person appear before shift_start time?
  C. Cylinder tracker       — log "detected" / "swapped" events, count usage days
  D. Dirty-floor detector   — frame-diff vs client-supplied clean baseline photo
  E. Gas / oven idle monitor — motion heuristic + person-supervision check (Phase 4)
  F. Shop absence checker   — fire alert if no person seen in shop camera > 60 s

Integration
───────────
  Call from camera_manager._detection_loop():

    rule_engine.check_shift_start(camera_id, floor, db)
    rule_engine.process_cylinder_detections(camera_id, raw_dets, db)
    rule_engine.check_dirty_floor(camera_id, frame, db)
    rule_engine.check_shop_absence(camera_id, floor, persons, db)
    rule_engine.gas_idle_update(camera_id, zone_id, frame, person_present, db)

Alert ownership
───────────────
  All rule-engine alerts use user_id=1 (first admin created at setup) as the
  notification target.  Factory-level alerts are not per-viewer — they go to
  the admin account.  Adjust RULE_ENGINE_USER_ID if a dedicated system user is
  added later.

Accuracy disclaimers (mandatory for client transparency)
────────────────────────────────────────────────────────
  • Dirty-floor: best-effort visual heuristic, ~70-80% for obvious debris.
    Misses subtle dirt.  NOT suitable as sole evidence for disciplinary action.
    Recommend human review of every flagged frame.
  • Gas/oven idle: motion heuristic.  Steam/vapor recall ≈ 40-60%.
    NOT a temperature or gas-flow measurement.  See section E for full note.
  • These limitations are explicitly documented in bakery_cv_plan.md §12.
"""

from __future__ import annotations

import datetime
import logging
import math
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("rule_engine")

# ── System user that receives rule-engine alerts ─────────────────────────────
# user_id=1 = first admin registered. Change via env RULE_ENGINE_USER_ID.
RULE_ENGINE_USER_ID: int = int(os.environ.get("RULE_ENGINE_USER_ID", "1"))

# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — DB-backed settings loader
# ═════════════════════════════════════════════════════════════════════════════
# In-process cache: { key -> (value_str, cached_at) }
_settings_cache:   Dict[str, Tuple[str, float]] = {}
_settings_lock     = threading.Lock()
_SETTINGS_TTL_SEC  = 60       # re-read from DB at most once per minute

# Default values — written to DB on first startup via seed_defaults()
SETTING_DEFAULTS: Dict[str, Tuple[str, str]] = {
    # (value, description)
    "idle_limit_default":          ("300",   "Idle alert threshold (seconds) — all floors"),
    "idle_limit_camera_standing":  ("60",    "Person blocking camera alert (seconds)"),
    "idle_limit_shop_counter":     ("60",    "Employee absent from shop alert (seconds)"),
    "idle_limit_gas_idle":         ("600",   "Stove/oil idle alert (seconds) — 2nd floor"),
    "shift_start_ground":          ("08:00", "Ground floor shift start time (HH:MM)"),
    "shift_start_first":           ("06:00", "First floor shift start time (HH:MM)"),
    "shift_start_second":          ("05:00", "Second floor shift start time (HH:MM)"),
    "shift_start_shop":            ("08:00", "Shop floor shift start time (HH:MM)"),
    "dirty_floor_threshold":       ("0.08",  "Dirty-floor: changed pixel fraction (0-1)"),
    "dirty_floor_consecutive_hits":("5",     "Consecutive dirty frames before alert"),
    "cylinder_bbox_min_delta":     ("1",     "Min bbox-count change to log cylinder swap"),
    "cylinder_log_rate_sec":       ("60",    "Minimum seconds between cylinder 'detected' logs"),
    "move_threshold_px":           ("15",    "Person centroid movement threshold (px, at 640w)"),
    "gas_activity_threshold":      ("5.0",   "Mean pixel change below which stove zone = idle"),
    "shop_absence_limit_sec":      ("60",    "Seconds with no person in shop before alert"),
}


def seed_defaults(db) -> None:
    """
    Insert default settings rows that don't already exist.
    Safe to call on every startup — uses INSERT OR IGNORE semantics.
    """
    from database import SystemSettings
    for key, (value, description) in SETTING_DEFAULTS.items():
        existing = db.query(SystemSettings).filter(SystemSettings.key == key).first()
        if not existing:
            db.add(SystemSettings(key=key, value=value, description=description))
    db.commit()
    log.info("[rule_engine] SystemSettings defaults seeded")


def _fetch_setting_from_db(key: str, db) -> Optional[str]:
    from database import SystemSettings
    row = db.query(SystemSettings).filter(SystemSettings.key == key).first()
    return row.value if row else None


def get_setting(key: str, db=None) -> Optional[str]:
    """
    Return the value for a settings key, using an in-process 60-second cache.
    Falls back to SETTING_DEFAULTS if the key is not in the DB.
    """
    now = time.monotonic()
    with _settings_lock:
        cached = _settings_cache.get(key)
        if cached and (now - cached[1]) < _SETTINGS_TTL_SEC:
            return cached[0]

    # Cache miss — query DB
    value = None
    if db is not None:
        try:
            value = _fetch_setting_from_db(key, db)
        except Exception as exc:
            log.warning(f"[rule_engine] settings DB read error for {key!r}: {exc}")

    if value is None:
        default_tuple = SETTING_DEFAULTS.get(key)
        value = default_tuple[0] if default_tuple else None

    if value is not None:
        with _settings_lock:
            _settings_cache[key] = (value, now)

    return value


def get_int(key: str, db=None) -> int:
    v = get_setting(key, db)
    try:
        return int(v) if v is not None else int(SETTING_DEFAULTS[key][0])
    except (ValueError, KeyError):
        return 0


def get_float(key: str, db=None) -> float:
    v = get_setting(key, db)
    try:
        return float(v) if v is not None else float(SETTING_DEFAULTS[key][0])
    except (ValueError, KeyError):
        return 0.0


def invalidate_settings_cache(key: Optional[str] = None) -> None:
    """Bust cache for one key, or all keys if key is None."""
    with _settings_lock:
        if key:
            _settings_cache.pop(key, None)
        else:
            _settings_cache.clear()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — Shift-start compliance checker
# ═════════════════════════════════════════════════════════════════════════════
# State: { floor -> date checked } — checked once per floor per calendar day.
_shift_checked:     Dict[str, datetime.date] = {}
_shift_first_seen:  Dict[str, Optional[datetime.datetime]] = {}   # floor -> first detection ts
_shift_lock         = threading.Lock()

# Map camera floor → settings key
_FLOOR_SETTING_KEY = {
    "ground":  "shift_start_ground",
    "first":   "shift_start_first",
    "second":  "shift_start_second",
    "shop":    "shift_start_shop",
}


def record_person_seen(floor: str) -> None:
    """
    Record that at least one person was detected on this floor right now.
    Call this from the detection loop for every frame that contains a person.
    The shift-start checker reads _shift_first_seen to determine if the floor
    was active before the shift_start time.
    """
    with _shift_lock:
        if floor not in _shift_first_seen:
            _shift_first_seen[floor] = datetime.datetime.now()


def check_shift_start(camera_id: int, floor: str, db) -> None:
    """
    Once per floor per calendar day: verify that at least one person was
    detected on this floor BEFORE the configured shift_start time.

    If no person was seen by shift_start + a 5-minute grace window, fire a
    "Late shift start" alert with severity=high.

    Call this from camera_manager._detection_loop() near the start of each tick.
    The check is a no-op on every tick except the first tick after shift_start
    on a new day for each floor.
    """
    if floor not in _FLOOR_SETTING_KEY:
        return  # unknown floor — skip

    now = datetime.datetime.now()
    today = now.date()

    with _shift_lock:
        last_checked = _shift_checked.get(floor)
        if last_checked == today:
            return  # already checked today for this floor

    # Only run the check after shift_start time (don't spam at 2 AM)
    setting_key    = _FLOOR_SETTING_KEY[floor]
    shift_start_str = get_setting(setting_key, db) or SETTING_DEFAULTS[setting_key][0]
    try:
        h, m = map(int, shift_start_str.split(":"))
        shift_start_time = datetime.time(h, m)
    except ValueError:
        log.error(f"[rule_engine] Invalid shift_start format for {floor!r}: {shift_start_str!r}")
        return

    # 5-minute grace window before we fire the alert
    grace_minutes = 5
    shift_dt = datetime.datetime.combine(today, shift_start_time)
    check_after = shift_dt + datetime.timedelta(minutes=grace_minutes)

    if now < check_after:
        return  # not yet time to check

    # Mark as checked for today so we don't re-run
    with _shift_lock:
        _shift_checked[floor] = today
        first_seen = _shift_first_seen.get(floor)

    # Determine if the floor was active on time
    was_on_time = (
        first_seen is not None
        and first_seen.date() == today
        and first_seen.time() <= shift_start_time
    )

    if not was_on_time:
        from services.alert_service import save_alert
        from database import SessionLocal

        late_str = (
            f"First activity at {first_seen.strftime('%H:%M')}"
            if first_seen and first_seen.date() == today
            else "No activity detected"
        )
        _db = None
        try:
            _db = db  # reuse caller's session if available
            save_alert(
                db=_db,
                user_id=RULE_ENGINE_USER_ID,
                message=(
                    f"⏰ Late shift start on {floor.capitalize()} Floor. "
                    f"Shift starts at {shift_start_str}. {late_str}."
                ),
                role="Factory Worker",
                severity="high",
                detected_issue="Late shift start",
                camera_id=camera_id,
                floor=floor,
            )
            log.warning(
                f"[rule_engine] Late shift start — floor={floor} "
                f"expected={shift_start_str} {late_str}"
            )
        except Exception as exc:
            log.error(f"[rule_engine] shift-start alert save failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — Cylinder usage tracker
# ═════════════════════════════════════════════════════════════════════════════
# State per camera: last_bbox_count, last_log_ts, last_day_increment_date, usage_day_count
_cylinder_state: Dict[int, dict] = {}
_cylinder_lock   = threading.Lock()


def process_cylinder_detections(
    camera_id: int,
    floor: str,
    raw_detections: List[dict],
    db,
) -> None:
    """
    Process Cylinder-class detections from one inference frame.

    Behaviour
    ---------
    1. Count the number of Cylinder bboxes in this frame.
    2. Rate-limited "detected" log: write to CylinderLog at most once per
       cylinder_log_rate_sec seconds per camera.
    3. "Swapped" detection: if the bbox count changed by ≥ cylinder_bbox_min_delta
       compared to the last logged count, write a "swapped" event.
    4. Daily usage increment: once per calendar day per camera, increment
       usage_day_count by 1 and update the last-swapped row.
    """
    cylinder_dets = [d for d in raw_detections if d.get("label") == "Cylinder"]
    if not cylinder_dets:
        return

    current_count = len(cylinder_dets)
    now           = datetime.datetime.now()
    today         = now.date()

    log_rate     = get_int("cylinder_log_rate_sec", db)
    min_delta    = get_int("cylinder_bbox_min_delta", db)

    from database import CylinderLog

    with _cylinder_lock:
        state = _cylinder_state.setdefault(camera_id, {
            "last_bbox_count":     current_count,
            "last_log_ts":         None,
            "last_day_date":       None,
            "usage_day_count":     0,
        })

        # ── Rate-limited "detected" log ────────────────────────────────
        last_log_ts = state.get("last_log_ts")
        seconds_since_last = (
            (now - last_log_ts).total_seconds() if last_log_ts else float("inf")
        )
        should_log_detected = seconds_since_last >= log_rate

        if should_log_detected:
            state["last_log_ts"] = now

        # ── Daily usage-day increment ───────────────────────────────────
        last_day = state.get("last_day_date")
        should_increment_day = (last_day is None or last_day < today)
        if should_increment_day:
            state["usage_day_count"] += 1
            state["last_day_date"]    = today

        # ── Swap detection ──────────────────────────────────────────────
        prev_count    = state.get("last_bbox_count", current_count)
        count_delta   = abs(current_count - prev_count)
        swapped       = count_delta >= min_delta
        state["last_bbox_count"] = current_count
        usage_day_count          = state["usage_day_count"]

    try:
        if should_log_detected:
            db.add(CylinderLog(
                camera_id       = camera_id,
                event_type      = "detected",
                timestamp       = now,
                usage_day_count = usage_day_count,
            ))

        if swapped:
            db.add(CylinderLog(
                camera_id       = camera_id,
                event_type      = "swapped",
                timestamp       = now,
                usage_day_count = usage_day_count,
            ))
            log.info(
                f"[rule_engine] Cylinder swap detected cam={camera_id} "
                f"prev={prev_count} curr={current_count} day={usage_day_count}"
            )

        db.commit()
    except Exception as exc:
        log.error(f"[rule_engine] CylinderLog write failed (cam={camera_id}): {exc}")
        try:
            db.rollback()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — Dirty-floor heuristic detector
# ═════════════════════════════════════════════════════════════════════════════
# ⚠️  ACCURACY NOTE (bakery_cv_plan.md §12):
# This is a best-effort visual heuristic comparing live frames to a client-
# supplied "clean" reference photo.  Expected accuracy: ~70-80% for obvious
# debris.  Misses subtle dirt.  NOT suitable as sole evidence for disciplinary
# action.  Recommend human review of every flagged frame.


class DirtyFloorDetector:
    """
    Compares live frames to a client-supplied clean-state baseline photo.
    One instance per camera.  The admin can update the baseline photo and
    call reload() — no server restart required.

    Algorithm (from bakery_cv_plan.md §9.6)
    ----------------------------------------
    1. Resize live frame to baseline resolution.
    2. Gaussian blur both (suppresses RTSP artefacts / minor lighting shifts).
    3. Absolute pixel difference.
    4. Threshold at 30 grey levels.
    5. Compute fraction of "changed" pixels.
    6. If fraction > threshold → dirty.
    Alert fires only after `consecutive_hits` consecutive dirty detections
    to avoid false positives from workers walking through the zone.
    """

    def __init__(
        self,
        camera_id: int,
        baseline_path: str,
        threshold: float = 0.08,
        consecutive_hits: int = 5,
    ):
        self.camera_id        = camera_id
        self.baseline_path    = baseline_path
        self.threshold        = threshold
        self.consecutive_hits = consecutive_hits
        self._consecutive_dirty = 0
        self._baseline: Optional[np.ndarray] = None
        self._load_baseline()

    def _load_baseline(self) -> None:
        if not os.path.exists(self.baseline_path):
            log.warning(
                f"[DirtyFloorDetector] cam={self.camera_id} "
                f"baseline not found: {self.baseline_path!r}"
            )
            self._baseline = None
            return
        raw = cv2.imread(self.baseline_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            log.error(f"[DirtyFloorDetector] cam={self.camera_id} could not read baseline image")
            self._baseline = None
            return
        self._baseline = cv2.GaussianBlur(raw, (21, 21), 0)
        log.info(f"[DirtyFloorDetector] cam={self.camera_id} baseline loaded from {self.baseline_path!r}")

    def reload(self) -> None:
        """Hot-reload baseline — call after admin uploads a new photo."""
        self._load_baseline()

    def check(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (alert_should_fire: bool, changed_fraction: float).

        alert_should_fire is True only after `consecutive_hits` consecutive
        dirty frames — single frames never trigger an alert alone.
        changed_fraction is always returned for logging/debugging.
        """
        if self._baseline is None:
            return False, 0.0

        bh, bw = self._baseline.shape[:2]
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray    = cv2.resize(gray, (bw, bh))
        gray    = cv2.GaussianBlur(gray, (21, 21), 0)
        diff    = cv2.absdiff(self._baseline, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        fraction  = float(thresh.sum()) / (255.0 * thresh.size)

        is_dirty = fraction > self.threshold
        if is_dirty:
            self._consecutive_dirty += 1
        else:
            self._consecutive_dirty = 0

        should_alert = (
            is_dirty and self._consecutive_dirty >= self.consecutive_hits
        )
        if should_alert:
            # Reset counter so we don't spam — next alert needs another N hits
            self._consecutive_dirty = 0

        return should_alert, round(fraction, 4)


# Registry of DirtyFloorDetector instances (one per camera_id)
_dirty_detectors:      Dict[int, DirtyFloorDetector] = {}
_dirty_detectors_lock  = threading.Lock()


def _get_or_create_detector(camera_id: int, db) -> Optional[DirtyFloorDetector]:
    """Return cached detector, or create one from the DB baseline record."""
    with _dirty_detectors_lock:
        if camera_id in _dirty_detectors:
            return _dirty_detectors[camera_id]

    # Look up baseline from DB
    from database import DirtyFloorBaseline
    try:
        row = (
            db.query(DirtyFloorBaseline)
            .filter(DirtyFloorBaseline.camera_id == camera_id)
            .first()
        )
    except Exception:
        return None

    if row is None:
        return None  # no baseline configured for this camera yet

    threshold        = get_float("dirty_floor_threshold", db)
    consecutive_hits = get_int("dirty_floor_consecutive_hits", db)
    detector = DirtyFloorDetector(
        camera_id=camera_id,
        baseline_path=row.image_path,
        threshold=threshold,
        consecutive_hits=consecutive_hits,
    )
    with _dirty_detectors_lock:
        _dirty_detectors[camera_id] = detector
    return detector


def reload_detector(camera_id: int) -> None:
    """Called when the admin uploads a new baseline photo for a camera."""
    with _dirty_detectors_lock:
        detector = _dirty_detectors.get(camera_id)
    if detector:
        detector.reload()
    else:
        # Force re-creation on next check_dirty_floor() call
        with _dirty_detectors_lock:
            _dirty_detectors.pop(camera_id, None)


def check_dirty_floor(
    camera_id: int,
    floor: str,
    frame: np.ndarray,
    db,
) -> None:
    """
    Run the dirty-floor heuristic for one camera frame.
    If the alert fires, saves a high-severity alert to the DB.

    Safe to call on every detection tick — the detector internally rate-limits
    alerts via the consecutive_hits counter.
    """
    detector = _get_or_create_detector(camera_id, db)
    if detector is None:
        return  # no baseline configured — skip silently

    try:
        should_alert, fraction = detector.check(frame)
    except Exception as exc:
        log.error(f"[rule_engine] dirty-floor check error cam={camera_id}: {exc}")
        return

    if should_alert:
        from services.alert_service import save_alert
        try:
            save_alert(
                db=db,
                user_id=RULE_ENGINE_USER_ID,
                message=(
                    f"🧹 Floor cleanliness alert on {floor.capitalize()} Floor — "
                    f"camera {camera_id}. {fraction*100:.1f}% of frame differs "
                    f"from clean baseline. Human review required."
                ),
                role="Cleanliness Monitor",
                severity="medium",
                detected_issue="Dirty floor detected",
                camera_id=camera_id,
                floor=floor,
                # Phase 4 heuristic — ~75-85% accuracy; requires admin review
                confidence_tier="low",
            )
            log.warning(
                f"[rule_engine] Dirty floor alert cam={camera_id} fraction={fraction:.3f}"
            )
        except Exception as exc:
            log.error(f"[rule_engine] dirty-floor alert save failed: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION E — Gas / oven idle monitor  (Phase 4 — visual heuristic)
# ═════════════════════════════════════════════════════════════════════════════
# ⚠️  ACCURACY NOTE (bakery_cv_plan.md §12):
# This is a VISUAL MOTION HEURISTIC — it does NOT measure temperature, gas
# flow, or actual flame.  Steam / vapor recall ≈ 40-60%.  For production
# reliability, pair with an IoT gas sensor (e.g. MQ-2 + MQTT → FastAPI) and
# use this module only as a secondary visual confirmation.
# This approach is explicitly documented as Phase 4 / R&D in bakery_cv_plan.md.


class GasIdleMonitor:
    """
    Tracks stove/oven zone activity via frame-difference (no ML model needed).

    Detection logic:
      1. Compute mean absolute pixel difference between consecutive frames
         inside the monitored zone polygon (or full frame if no polygon).
      2. If mean_diff < gas_activity_threshold for > idle_limit_sec seconds
         AND person_present=False → alert: stove left unattended.
      3. Resets if activity is detected (mean_diff > threshold) or person arrives.

    One instance per camera.
    """

    def __init__(self, camera_id: int, idle_limit_sec: int = 600):
        self.camera_id    = camera_id
        self.idle_limit   = idle_limit_sec
        self._prev_frame: Optional[np.ndarray] = None
        self._idle_since: Optional[float]       = None  # monotonic timestamp
        self._alert_fired                        = False

    def update(
        self,
        frame: np.ndarray,
        zone_polygon: Optional[List] = None,
        person_present: bool = False,
        db=None,
        floor: str = "second",
    ) -> bool:
        """
        Returns True if a new alert was fired this tick.

        zone_polygon: [[x,y],...] in frame pixel coords. If None, full frame used.
        person_present: True if any person detected in this frame.
        """
        # Extract zone ROI
        roi = self._extract_roi(frame, zone_polygon)

        if self._prev_frame is None:
            self._prev_frame = roi.copy()
            return False

        # Frame difference
        diff       = cv2.absdiff(self._prev_frame, roi)
        mean_diff  = float(diff.mean())
        self._prev_frame = roi.copy()

        activity_threshold = float(
            get_setting("gas_activity_threshold", db) or "5.0"
        )
        idle_limit = int(
            get_setting("idle_limit_gas_idle", db) or str(self.idle_limit)
        )

        zone_active = mean_diff >= activity_threshold

        if zone_active or person_present:
            # Activity detected — reset idle timer
            self._idle_since   = None
            self._alert_fired  = False
            return False

        # Zone appears idle
        now = time.monotonic()
        if self._idle_since is None:
            self._idle_since = now

        idle_for = now - self._idle_since

        if idle_for >= idle_limit and not self._alert_fired:
            self._alert_fired = True
            if db is not None:
                self._fire_alert(idle_for, floor, db)
            return True

        return False

    def _extract_roi(
        self,
        frame: np.ndarray,
        polygon: Optional[List],
    ) -> np.ndarray:
        """Crop to zone polygon bounding box for comparison, or use full frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if polygon is None or len(polygon) < 3:
            return gray
        pts = np.array(polygon, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        x1 = max(0, x); y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        return gray[y1:y2, x1:x2]

    def _fire_alert(self, idle_for: float, floor: str, db) -> None:
        from services.alert_service import save_alert
        mins = idle_for / 60
        try:
            save_alert(
                db=db,
                user_id=RULE_ENGINE_USER_ID,
                message=(
                    f"🔥 Stove/oven zone appears idle for {mins:.1f} minutes "
                    f"with no person supervising — camera {self.camera_id}. "
                    f"Visual heuristic only; verify physically."
                ),
                role="Safety Monitor",
                severity="high",
                detected_issue="Gas/oven idle — no supervision",
                camera_id=self.camera_id,
                floor=floor,
                # Phase 4 heuristic — ~40-60% steam recall; requires admin review
                confidence_tier="low",
            )
            log.warning(
                f"[rule_engine] Gas-idle alert cam={self.camera_id} "
                f"idle_for={idle_for:.0f}s"
            )
        except Exception as exc:
            log.error(f"[rule_engine] gas-idle alert save failed: {exc}")


# Registry (camera_id → GasIdleMonitor)
_gas_monitors:     Dict[int, GasIdleMonitor] = {}
_gas_monitor_lock  = threading.Lock()


def gas_idle_update(
    camera_id: int,
    floor: str,
    frame: np.ndarray,
    person_present: bool,
    zone_polygon: Optional[List] = None,
    db=None,
) -> bool:
    """
    Update the gas-idle monitor for one camera frame.
    Only meaningful for 2nd-floor stove cameras — call only when floor == "second".
    Returns True if an alert was fired.
    """
    with _gas_monitor_lock:
        if camera_id not in _gas_monitors:
            idle_limit = get_int("idle_limit_gas_idle", db) or 600
            _gas_monitors[camera_id] = GasIdleMonitor(camera_id, idle_limit)
        monitor = _gas_monitors[camera_id]

    return monitor.update(
        frame=frame,
        zone_polygon=zone_polygon,
        person_present=person_present,
        db=db,
        floor=floor,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION F — Shop absence checker
# ═════════════════════════════════════════════════════════════════════════════
# Rule: if a camera on the "shop" floor sees ZERO persons for > shop_absence_limit_sec
# seconds, fire an alert.
# Source: req.md → "If an employee is not there in the shop for more than 1 min,
#         an alert should come to the admin."
# Implementation: pure time/state — no ML needed.

_shop_absence_state: Dict[int, dict] = {}   # camera_id → {empty_since, alert_fired}
_shop_absence_lock   = threading.Lock()


def check_shop_absence(
    camera_id: int,
    floor: str,
    persons: List[dict],
    db,
) -> None:
    """
    Monitor whether the shop counter zone is staffed.
    Call from the detection loop for every camera on floor="shop".

    persons: list of person detection dicts from yolo_service output.
             An empty list means no one is visible to the camera.
    """
    if floor != "shop":
        return  # only applies to shop floor cameras

    limit = get_int("shop_absence_limit_sec", db) or 60
    now   = time.monotonic()

    with _shop_absence_lock:
        state = _shop_absence_state.setdefault(camera_id, {
            "empty_since":  None,
            "alert_fired":  False,
        })

        if persons:
            # Someone visible — reset
            state["empty_since"]  = None
            state["alert_fired"]  = False
            return

        # No persons visible
        if state["empty_since"] is None:
            state["empty_since"] = now
            return

        empty_for = now - state["empty_since"]

        if empty_for >= limit and not state["alert_fired"]:
            state["alert_fired"] = True
            _fire_shop_absence_alert(camera_id, empty_for, db)


def _fire_shop_absence_alert(camera_id: int, empty_for: float, db) -> None:
    from services.alert_service import save_alert
    mins = empty_for / 60
    try:
        save_alert(
            db=db,
            user_id=RULE_ENGINE_USER_ID,
            message=(
                f"🏪 Shop unattended for {mins:.1f} minutes — "
                f"no employee visible on camera {camera_id}. "
                f"Admin action required."
            ),
            role="Shop Monitor",
            severity="high",
            detected_issue="Shop unattended — employee absent",
            camera_id=camera_id,
            floor="shop",
        )
        log.warning(
            f"[rule_engine] Shop absence alert cam={camera_id} "
            f"empty_for={empty_for:.0f}s"
        )
    except Exception as exc:
        log.error(f"[rule_engine] shop-absence alert save failed: {exc}")
