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
from zoneinfo import ZoneInfo

import cv2
import numpy as np

log = logging.getLogger("rule_engine")

# IST timezone — all shift-time comparisons use this, never bare datetime.now()
_IST = ZoneInfo("Asia/Kolkata")

# ── System user that receives rule-engine alerts ─────────────────────────────
# user_id=1 = first admin registered. Change via env RULE_ENGINE_USER_ID.
RULE_ENGINE_USER_ID: int = int(os.environ.get("RULE_ENGINE_USER_ID", "1"))


def _get_rule_engine_user_id(db) -> int:
    """
    Return a valid user_id for system-generated rule-engine alerts.

    Tries RULE_ENGINE_USER_ID first (env-configured, default=1).
    If that user doesn't exist (e.g. fresh DB with no registrations yet),
    falls back to the lowest-id user to prevent FK constraint violations.
    Returns RULE_ENGINE_USER_ID as-is if no users exist at all — the caller's
    try/except will catch the FK error and log it without crashing the loop.
    """
    from database import User

    try:
        user = db.query(User).filter(User.id == RULE_ENGINE_USER_ID).first()
        if user:
            return user.id
        any_user = db.query(User).order_by(User.id).first()
        if any_user:
            log.warning(
                "[rule_engine] user_id=%d not found; falling back to user_id=%d",
                RULE_ENGINE_USER_ID,
                any_user.id,
            )
            return any_user.id
    except Exception as exc:
        log.error("[rule_engine] user lookup failed: %s", exc)
    return RULE_ENGINE_USER_ID


# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — DB-backed settings loader
# ═════════════════════════════════════════════════════════════════════════════
# In-process cache: { key -> (value_str, cached_at) }
_settings_cache: Dict[str, Tuple[str, float]] = {}
_settings_lock = threading.Lock()
_SETTINGS_TTL_SEC = 60  # re-read from DB at most once per minute

# Default values — written to DB on first startup via seed_defaults()
SETTING_DEFAULTS: Dict[str, Tuple[str, str]] = {
    # (value, description)
    "idle_limit_default": ("300", "Idle alert threshold (seconds) — all floors"),
    "idle_limit_camera_standing": ("60", "Person blocking camera alert (seconds)"),
    "idle_limit_shop_counter": ("60", "Employee absent from shop alert (seconds)"),
    "idle_limit_gas_idle": ("600", "Stove/oil idle alert (seconds) — 2nd floor"),
    "shift_start_ground": ("08:00", "Ground floor shift start time (HH:MM)"),
    "shift_start_first": ("06:00", "First floor shift start time (HH:MM)"),
    "shift_start_second": ("05:00", "Second floor shift start time (HH:MM)"),
    "shift_start_shop": ("08:00", "Shop floor shift start time (HH:MM)"),
    "dirty_floor_threshold": ("0.08", "Dirty-floor: changed pixel fraction (0-1)"),
    "dirty_floor_consecutive_hits": ("5", "Consecutive dirty frames before alert"),
    "cylinder_bbox_min_delta": ("1", "Min bbox-count change to log cylinder swap"),
    "cylinder_log_rate_sec": ("60", "Minimum seconds between cylinder 'detected' logs"),
    "move_threshold_px": ("15", "Person centroid movement threshold (px, at 640w)"),
    "gas_activity_threshold": (
        "5.0",
        "Mean pixel change below which stove zone = idle",
    ),
    "shop_absence_limit_sec": ("60", "Seconds with no person in shop before alert"),
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
_shift_checked: Dict[str, datetime.date] = {}
_shift_first_seen: Dict[str, Optional[datetime.datetime]] = (
    {}
)  # floor -> first detection ts
_shift_lock = threading.Lock()

# Map camera floor → settings key
_FLOOR_SETTING_KEY = {
    "ground": "shift_start_ground",
    "first": "shift_start_first",
    "second": "shift_start_second",
    "shop": "shift_start_shop",
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
            # Record the first-seen time in IST-aware datetime for correct comparison
            _shift_first_seen[floor] = datetime.datetime.now(_IST)


def check_shift_start(camera_id: int, floor: str, db) -> None:
    """
    Once per floor per calendar day: verify that at least one person was
    detected on this floor BEFORE the configured shift_start time (IST).

    Two checks are made:
      1. Pre-shift warning: fires 5 minutes BEFORE shift_start if no one seen yet.
      2. Post-shift alert: fires 5 minutes AFTER shift_start if still no one seen.

    Both use IST so 08:00 IST shift correctly matches client wall-clock time.
    Call this from camera_manager._detection_loop() near the start of each tick.
    """
    if floor not in _FLOOR_SETTING_KEY:
        return  # unknown floor — skip

    now_ist = datetime.datetime.now(_IST)
    today = now_ist.date()

    setting_key = _FLOOR_SETTING_KEY[floor]
    shift_start_str = get_setting(setting_key, db) or SETTING_DEFAULTS[setting_key][0]
    try:
        h, m = map(int, shift_start_str.split(":"))
        shift_start_time = datetime.time(h, m)
    except ValueError:
        log.error(
            f"[rule_engine] Invalid shift_start format for {floor!r}: {shift_start_str!r}"
        )
        return

    shift_dt = datetime.datetime.combine(today, shift_start_time, tzinfo=_IST)

    # ── Pre-shift warning: 5 minutes BEFORE shift start ──────────────────────────────
    pre_warn_key = f"_pre_warned_{floor}"
    warn_at = shift_dt - datetime.timedelta(minutes=5)
    warn_end = shift_dt  # stop warning once shift actually started

    if warn_at <= now_ist < warn_end:
        with _shift_lock:
            already_warned = _shift_checked.get(pre_warn_key)
            first_seen = _shift_first_seen.get(floor)

        if not already_warned and first_seen is None:
            with _shift_lock:
                _shift_checked[pre_warn_key] = today
            from services.alert_service import save_alert

            try:
                save_alert(
                    db=db,
                    user_id=_get_rule_engine_user_id(db),
                    message=(
                        f"⏰ {floor.capitalize()} Floor shift starts in 5 minutes "
                        f"({shift_start_str} IST) — no employee detected yet on camera {camera_id}."
                    ),
                    role="Factory Worker",
                    severity="high",
                    detected_issue="Pre-shift warning — no employee on floor",
                    camera_id=camera_id,
                    floor=floor,
                )
                log.warning(
                    f"[rule_engine] Pre-shift warning fired — floor={floor} shift={shift_start_str}"
                )
            except Exception as exc:
                log.error(f"[rule_engine] pre-shift alert save failed: {exc}")

    # ── Post-shift alert: 5 minutes AFTER shift start ──────────────────────────────
    grace_minutes = 5
    check_after = shift_dt + datetime.timedelta(minutes=grace_minutes)

    if now_ist < check_after:
        return  # not yet time for the post-shift check

    with _shift_lock:
        last_checked = _shift_checked.get(floor)
        if last_checked == today:
            return  # already checked today for this floor

    # Mark as checked for today so we don’t re-run
    with _shift_lock:
        _shift_checked[floor] = today
        first_seen = _shift_first_seen.get(floor)

    # Determine if the floor was active on time (IST comparison)
    was_on_time = (
        first_seen is not None
        and first_seen.date() == today
        and first_seen.timetz().replace(tzinfo=None) <= shift_start_time
    )

    if not was_on_time:
        from services.alert_service import save_alert

        if first_seen and first_seen.date() == today:
            late_str = f"First activity at {first_seen.strftime('%H:%M IST')}"
        else:
            late_str = "No activity detected"

        try:
            save_alert(
                db=db,
                user_id=_get_rule_engine_user_id(db),
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
_cylinder_lock = threading.Lock()


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
    now = datetime.datetime.now()
    today = now.date()

    log_rate = get_int("cylinder_log_rate_sec", db)
    min_delta = get_int("cylinder_bbox_min_delta", db)

    from database import CylinderLog

    with _cylinder_lock:
        state = _cylinder_state.setdefault(
            camera_id,
            {
                "last_bbox_count": current_count,
                "last_log_ts": None,
                "last_day_date": None,
                "usage_day_count": 0,
            },
        )

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
        should_increment_day = last_day is None or last_day < today
        if should_increment_day:
            state["usage_day_count"] += 1
            state["last_day_date"] = today

        # ── Swap detection ──────────────────────────────────────────────
        prev_count = state.get("last_bbox_count", current_count)
        count_delta = abs(current_count - prev_count)
        swapped = count_delta >= min_delta
        state["last_bbox_count"] = current_count
        usage_day_count = state["usage_day_count"]

    try:
        if should_log_detected:
            db.add(
                CylinderLog(
                    camera_id=camera_id,
                    event_type="detected",
                    timestamp=now,
                    usage_day_count=usage_day_count,
                )
            )

        if swapped:
            db.add(
                CylinderLog(
                    camera_id=camera_id,
                    event_type="swapped",
                    timestamp=now,
                    usage_day_count=usage_day_count,
                )
            )
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
        self.camera_id = camera_id
        self.baseline_path = baseline_path
        self.threshold = threshold
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
            log.error(
                f"[DirtyFloorDetector] cam={self.camera_id} could not read baseline image"
            )
            self._baseline = None
            return
        self._baseline = cv2.GaussianBlur(raw, (21, 21), 0)
        log.info(
            f"[DirtyFloorDetector] cam={self.camera_id} baseline loaded from {self.baseline_path!r}"
        )

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (bw, bh))
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(self._baseline, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        fraction = float(thresh.sum()) / (255.0 * thresh.size)

        is_dirty = fraction > self.threshold
        if is_dirty:
            self._consecutive_dirty += 1
        else:
            self._consecutive_dirty = 0

        should_alert = is_dirty and self._consecutive_dirty >= self.consecutive_hits
        if should_alert:
            # Reset counter so we don't spam — next alert needs another N hits
            self._consecutive_dirty = 0

        return should_alert, round(fraction, 4)


# Registry of DirtyFloorDetector instances (one per camera_id)
_dirty_detectors: Dict[int, DirtyFloorDetector] = {}
_dirty_detectors_lock = threading.Lock()


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

    threshold = get_float("dirty_floor_threshold", db)
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
                user_id=_get_rule_engine_user_id(db),
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
        self.camera_id = camera_id
        self.idle_limit = idle_limit_sec
        self._prev_frame: Optional[np.ndarray] = None
        self._idle_since: Optional[float] = None  # monotonic timestamp
        self._alert_fired = False

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
        diff = cv2.absdiff(self._prev_frame, roi)
        mean_diff = float(diff.mean())
        self._prev_frame = roi.copy()

        activity_threshold = float(get_setting("gas_activity_threshold", db) or "5.0")
        idle_limit = int(get_setting("idle_limit_gas_idle", db) or str(self.idle_limit))

        zone_active = mean_diff >= activity_threshold

        if zone_active or person_present:
            # Activity detected — reset idle timer
            self._idle_since = None
            self._alert_fired = False
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
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        return gray[y1:y2, x1:x2]

    def _fire_alert(self, idle_for: float, floor: str, db) -> None:
        from services.alert_service import save_alert

        mins = idle_for / 60
        try:
            save_alert(
                db=db,
                user_id=_get_rule_engine_user_id(db),
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
_gas_monitors: Dict[int, GasIdleMonitor] = {}
_gas_monitor_lock = threading.Lock()


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

_shop_absence_state: Dict[int, dict] = {}  # camera_id → {empty_since, alert_fired}
_shop_absence_lock = threading.Lock()


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
    now = time.monotonic()

    with _shop_absence_lock:
        state = _shop_absence_state.setdefault(
            camera_id,
            {
                "empty_since": None,
                "alert_fired": False,
            },
        )

        if persons:
            # Someone visible — reset
            state["empty_since"] = None
            state["alert_fired"] = False
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
            user_id=_get_rule_engine_user_id(db),
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


# -----------------------------------------------------------------------------
# SECTION G � Camera Blocking Detection (REQ-041)
# -----------------------------------------------------------------------------
_camera_blocking_state: Dict[int, dict] = {}
_camera_blocking_lock = threading.Lock()


def check_camera_blocking(
    camera_id: int, frame_shape: Tuple[int, int, int], persons: List[dict], db
) -> None:
    CAMERA_BLOCK_RATIO = 0.70
    limit = get_int("idle_limit_camera_standing", db) or 60
    now = time.monotonic()

    frame_h, frame_w = frame_shape[:2]
    frame_area = frame_h * frame_w

    is_blocked = False
    for p in persons:
        bbox = p.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area / frame_area > CAMERA_BLOCK_RATIO:
            is_blocked = True
            break

    with _camera_blocking_lock:
        state = _camera_blocking_state.setdefault(
            camera_id,
            {
                "blocked_since": None,
                "alert_fired": False,
            },
        )

        if not is_blocked:
            state["blocked_since"] = None
            state["alert_fired"] = False
            return

        if state["blocked_since"] is None:
            state["blocked_since"] = now
            return

        blocked_for = now - state["blocked_since"]
        if blocked_for >= limit and not state["alert_fired"]:
            state["alert_fired"] = True
            from services.alert_service import save_alert

            try:
                save_alert(
                    db=db,
                    user_id=_get_rule_engine_user_id(db),
                    message=(
                        f" Camera {camera_id} is blocked by a person "
                        f"standing too close for over {limit} seconds."
                    ),
                    role="Safety Monitor",
                    severity="high",
                    detected_issue="Camera blocked",
                    camera_id=camera_id,
                )
                log.warning(f"[rule_engine] Camera blocked alert cam={camera_id}")
            except Exception as exc:
                log.error(f"[rule_engine] camera-blocked alert save failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────────
# SECTION H — Stock Zone Monitor (REQ-011, REQ-014)
# ─────────────────────────────────────────────────────────────────────────────────
_stock_zone_state: Dict[int, float] = {}  # camera_id → last_alert_time
_stock_zone_lock = threading.Lock()


def check_stock_zone(
    camera_id: int,
    floor: str,
    raw_detections: List[dict],
    zones: dict,
    db,
) -> None:
    """Alert when an Exposed-Item detection falls inside a monitored zone."""
    if not zones:
        return

    exposed_items = [d for d in raw_detections if d.get("label") == "Exposed-Item"]
    if not exposed_items:
        return

    from services.zone_service import bbox_in_zone

    for zone_name in ("entrance", "glassdoor", "stock"):
        poly = zones.get(zone_name)
        if not poly:
            continue
        for item in exposed_items:
            if not bbox_in_zone(item["bbox"], poly):
                continue
            # Rate-limit: one alert per camera per 5 minutes
            now = time.monotonic()
            with _stock_zone_lock:
                if now - _stock_zone_state.get(camera_id, 0) < 300:
                    return
                _stock_zone_state[camera_id] = now
            from services.alert_service import save_alert

            try:
                save_alert(
                    db=db,
                    user_id=_get_rule_engine_user_id(db),
                    message=f"⚠️ Stock left openly in '{zone_name}' zone on camera {camera_id}.",
                    role="Factory Worker",
                    severity="high",
                    detected_issue="Stock Kept Openly",
                    camera_id=camera_id,
                    floor=floor,
                )
            except Exception as exc:
                log.error(f"[rule_engine] stock-zone alert save failed: {exc}")
            return  # one alert per tick maximum


# ─────────────────────────────────────────────────────────────────────────────────
# SECTION I — Machinery Zone Monitor (REQ-022, REQ-023)
# ─────────────────────────────────────────────────────────────────────────────────
_machinery_idle_state: Dict[int, dict] = {}
_machinery_idle_lock = threading.Lock()


def check_machinery_zone(
    camera_id: int,
    floor: str,
    raw_detections: List[dict],
    persons: List[dict],
    zones: dict,
    db,
) -> None:
    """Alert when a machine zone (dough, biscuit_cutting) is unattended for >60s."""
    if not zones:
        return

    from services.zone_service import bbox_in_zone

    now = time.monotonic()

    for zone_name in ("dough", "biscuit_cutting"):
        poly = zones.get(zone_name)
        if not poly:
            continue

        machines = [d for d in raw_detections if d.get("label") == "machinery"]
        machines_in_zone = any(bbox_in_zone(m["bbox"], poly) for m in machines)
        persons_in_zone = any(bbox_in_zone(p["bbox"], poly) for p in persons)

        state_key = f"{camera_id}_{zone_name}"
        with _machinery_idle_lock:
            state = _machinery_idle_state.setdefault(
                state_key,
                {
                    "unattended_since": None,
                    "alert_fired": False,
                },
            )

            if machines_in_zone and not persons_in_zone:
                if state["unattended_since"] is None:
                    state["unattended_since"] = now
                elif now - state["unattended_since"] >= 60 and not state["alert_fired"]:
                    state["alert_fired"] = True
                    from services.alert_service import save_alert

                    try:
                        save_alert(
                            db=db,
                            user_id=_get_rule_engine_user_id(db),
                            message=(
                                f"⚠️ Machinery in '{zone_name}' zone is unattended "
                                f"(no person present) on camera {camera_id}."
                            ),
                            role="Factory Worker",
                            severity="high",
                            detected_issue="Machine unattended",
                            camera_id=camera_id,
                            floor=floor,
                        )
                    except Exception as exc:
                        log.error(
                            f"[rule_engine] machinery-zone alert save failed: {exc}"
                        )
            else:
                state["unattended_since"] = None
                state["alert_fired"] = False


# ─────────────────────────────────────────────────────────────────────────────────
# SECTION J — Counter Activity Snapshot (REQ-041)
# Captures a snapshot when a person is detected at the shop counter/cashbox
# zone. Useful for vendor payment audits and access log.
# Rate-limited to once per 5 minutes per camera to avoid snapshot flooding.
# ─────────────────────────────────────────────────────────────────────────────────
_counter_snapshot_state: Dict[int, float] = {}  # camera_id → last_snapshot_time
_counter_snapshot_lock = threading.Lock()


def trigger_vendor_snapshot(
    camera_id: int,
    floor: str,
    persons: List[dict],
    frame,
    zones: dict,
    db,
) -> None:
    """Save a counter-activity snapshot when a person is at the counter/cashbox."""
    if floor != "shop" or not zones:
        return

    poly = zones.get("counter") or zones.get("cashbox")
    if not poly:
        return

    from services.zone_service import bbox_in_zone

    if not any(bbox_in_zone(p["bbox"], poly) for p in persons):
        return

    now = time.monotonic()
    with _counter_snapshot_lock:
        if now - _counter_snapshot_state.get(camera_id, 0) < 300:  # 5 min cooldown
            return
        _counter_snapshot_state[camera_id] = now

    from services.alert_service import save_alert
    from services.yolo_service import encode_frame

    try:
        save_alert(
            db=db,
            user_id=_get_rule_engine_user_id(db),
            message="Counter/cashbox activity detected. Snapshot saved for audit log.",
            role="Shop Monitor",
            severity="low",
            detected_issue="Counter Activity Snapshot",
            camera_id=camera_id,
            floor=floor,
            snapshot_b64=encode_frame(frame, quality=70),
        )
    except Exception as exc:
        log.error(f"[rule_engine] counter-snapshot save failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────────
# SECTION K — Window-Throw / Theft Detection (REQ-SF-07)
# ─────────────────────────────────────────────────────────────────────────────────
# Detects fast-moving objects near window zones that may indicate stealing,
# throwing, or passing goods through windows.
#
# Algorithm:
#   1. Track optical flow magnitude (pixel displacement per frame) for all
#      non-person detections inside the window zone polygon.
#   2. If any object's flow magnitude exceeds the throw_velocity_threshold
#      AND the object is moving outward (away from frame center toward edge),
#      fire an immediate high-severity alert.
#   3. Also fires if a person is detected inside the window zone for > 30s
#      without a corresponding machinery/stock-zone activity (loitering).
#
# Accuracy: ~70-85% for obvious throws; misses slow hand-offs.
# Not suitable as sole evidence — recommend human review.
# ─────────────────────────────────────────────────────────────────────────────────

_window_throw_state: Dict[str, dict] = {}
_window_throw_lock = threading.Lock()

_WINDOW_THROW_VELOCITY_PX = 40.0  # min pixel displacement per frame to flag
_WINDOW_LOITER_SEC = 30.0  # person near window without work = suspicious
_WINDOW_COOLDOWN_SEC = 120


def check_window_throw(
    camera_id: int,
    floor: str,
    raw_detections: List[dict],
    persons: List[dict],
    frame: np.ndarray,
    zones: dict,
    db,
) -> None:
    """
    Detect fast-moving objects near window zones (stealing/throwing).

    Uses dense optical flow (Farneback) to measure pixel displacement between
    consecutive frames inside the window zone polygon. High displacement of
    a non-person object moving toward the frame edge triggers an alert.

    Also monitors for loitering: a person standing in the window zone for
    longer than _WINDOW_LOITER_SEC without nearby machinery activity.
    """
    if not zones or frame is None:
        return

    window_zones = {k: v for k, v in zones.items() if "window" in k}
    if not window_zones:
        return

    import cv2

    now = time.monotonic()

    for zone_name, polygon in window_zones.items():
        state_key = f"{camera_id}_{zone_name}"

        # ── Part 1: Optical-flow velocity check on non-person objects ──────
        non_person_dets = [
            d
            for d in raw_detections
            if d.get("label")
            not in ("Person", "Hardhat", "Mask", "Safety Vest", "Bakery-Head-Cap")
        ]

        if non_person_dets:
            try:
                from services.zone_service import bbox_in_zone

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                with _window_throw_lock:
                    prev = _window_throw_state.get(state_key, {}).get("prev_gray")

                    if prev is not None and prev.shape == gray.shape:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev,
                            gray,
                            None,
                            pyr_scale=0.5,
                            levels=3,
                            winsize=15,
                            iterations=3,
                            poly_n=5,
                            poly_sigma=1.2,
                            flags=0,
                        )

                        for det in non_person_dets:
                            bbox = det.get("bbox", [])
                            if not bbox or not bbox_in_zone(bbox, polygon):
                                continue

                            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                            roi_flow = flow[max(0, y1) : y2, max(0, x1) : x2]
                            if roi_flow.size == 0:
                                continue

                            mag = cv2.magnitude(roi_flow[..., 0], roi_flow[..., 1])
                            mean_mag = float(mag.mean())
                            max_mag = float(mag.max())

                            # Check outward trajectory: object moving toward edge
                            fx = float(roi_flow[..., 0].mean())
                            frame_w = frame.shape[1]
                            cx_obj = (x1 + x2) / 2.0
                            moving_outward = (fx > 0 and cx_obj > frame_w * 0.6) or (
                                fx < 0 and cx_obj < frame_w * 0.4
                            )

                            if (
                                mean_mag > _WINDOW_THROW_VELOCITY_PX
                                or max_mag > _WINDOW_THROW_VELOCITY_PX * 2
                            ):
                                if moving_outward:
                                    _fire_window_throw_alert(
                                        db,
                                        camera_id,
                                        floor,
                                        zone_name,
                                        det.get("label", "unknown"),
                                        mean_mag,
                                        max_mag,
                                    )
                                    _window_throw_state[state_key] = {"prev_gray": gray}
                                    return

                    _window_throw_state.setdefault(state_key, {})["prev_gray"] = gray

            except Exception as exc:
                log.debug(
                    f"[rule_engine] window-throw flow error cam={camera_id}: {exc}"
                )

        # ── Part 2: Person loitering near window ───────────────────────────
        from services.zone_service import bbox_in_zone

        persons_in_window = [
            p for p in persons if bbox_in_zone(p.get("bbox", []), polygon)
        ]

        loiter_key = f"loiter_{camera_id}_{zone_name}"
        with _window_throw_lock:
            state = _window_throw_state.setdefault(
                loiter_key,
                {
                    "since": None,
                    "alert_fired": False,
                },
            )

            if persons_in_window:
                if state["since"] is None:
                    state["since"] = now
                elif (
                    now - state["since"] >= _WINDOW_LOITER_SEC
                    and not state["alert_fired"]
                ):
                    state["alert_fired"] = True
                    _fire_window_loiter_alert(
                        db,
                        camera_id,
                        floor,
                        zone_name,
                        len(persons_in_window),
                    )
            else:
                state["since"] = None
                state["alert_fired"] = False


def _fire_window_throw_alert(
    db,
    camera_id: int,
    floor: str,
    zone_name: str,
    obj_label: str,
    mean_mag: float,
    max_mag: float,
) -> None:
    from services.alert_service import save_alert

    try:
        save_alert(
            db=db,
            user_id=_get_rule_engine_user_id(db),
            message=(
                f"[WINDOW ALERT] Camera {camera_id} — possible theft/throwing "
                f"detected in '{zone_name}' zone. Object '{obj_label}' moving at "
                f"velocity {mean_mag:.1f}px/frame (max {max_mag:.1f}px). "
                f"Immediate review required."
            ),
            role="Security Monitor",
            severity="critical",
            detected_issue="Window throw/theft detected",
            camera_id=camera_id,
            floor=floor,
            confidence_tier="low",
        )
        log.warning(
            f"[rule_engine] Window-throw alert cam={camera_id} "
            f"zone={zone_name} vel={mean_mag:.1f}"
        )
    except Exception as exc:
        log.error(f"[rule_engine] window-throw alert save failed: {exc}")


def _fire_window_loiter_alert(
    db,
    camera_id: int,
    floor: str,
    zone_name: str,
    person_count: int,
) -> None:
    from services.alert_service import save_alert

    try:
        save_alert(
            db=db,
            user_id=_get_rule_engine_user_id(db),
            message=(
                f"[WINDOW ALERT] Camera {camera_id} — {person_count} person(s) "
                f"loitering in '{zone_name}' zone for >{_WINDOW_LOITER_SEC:.0f}s "
                f"without activity. Possible theft/passing goods."
            ),
            role="Security Monitor",
            severity="high",
            detected_issue="Window loitering",
            camera_id=camera_id,
            floor=floor,
            confidence_tier="low",
        )
        log.warning(
            f"[rule_engine] Window-loiter alert cam={camera_id} zone={zone_name}"
        )
    except Exception as exc:
        log.error(f"[rule_engine] window-loiter alert save failed: {exc}")


def cleanup_window_state(camera_id: int) -> None:
    """Remove all window-throw state for a stopped camera."""
    keys = [k for k in _window_throw_state if k.startswith(str(camera_id))]
    for k in keys:
        _window_throw_state.pop(k, None)


# ─────────────────────────────────────────────────────────────────────────────────
# SECTION L — Cash-in-pocket (replaced by CashEventTracker in cash_monitor.py)
# ─────────────────────────────────────────────────────────────────────────────────
# The previous gesture-heuristic implementation (centroid trajectory, downward+inward
# motion check) had ~50% precision and fired on ordinary movements like
# reaching for items.  It is replaced by the multi-frame CashEventTracker state
# machine in cash_monitor.py which:
#   • Requires a trained Cash detector (ppe_factory_v0_cash.pt, class ID 14)
#   • Tracks the *actual cash object* across frames via ByteTrack IDs
#   • Only fires SUSPICIOUS after cash enters body zone AND disappears without
#     reaching the cashbox — event-based, not gesture-based
#
# This function is kept as a no-op shim for API compatibility only.
# camera_manager.py no longer calls it (the comment block at line ~537 explains).
# ─────────────────────────────────────────────────────────────────────────────────

# Old module-level state — no longer used, preserved to avoid AttributeError
# if any other path references it.
_cash_pocket_state: Dict[int, dict] = {}
_cash_pocket_lock = threading.Lock()


def check_cash_in_pocket(
    camera_id: int,
    floor: str,
    persons: List[dict],
    frame: np.ndarray,
    zones: dict,
    db,
) -> None:
    """
    No-op shim — logic moved to cash_monitor.CashEventTracker.

    The CashEventTracker.update() call inside cash_monitor.check_cash_zone()
    handles body-zone detection and theft confirmation for shop cameras.
    This function is kept for API compatibility only.
    """
    return  # intentional no-op




# ─────────────────────────────────────────────────────────────────────────────────
# SECTION M — Finished Goods Dispatch Monitor (REQ-FF-11)
# ─────────────────────────────────────────────────────────────────────────────────
# Tracks items moving from lift zone to vehicle/loading zone.
# If an item appears in the finished_goods zone but no vehicle is detected
# in the loading zone within a configurable time window, fire an alert.
#
# This is a zone-transition tracker — no ML model required.
# Depends on: "finished_goods" zone polygon and "loading" / "vehicle" zone polygon.
# ─────────────────────────────────────────────────────────────────────────────

_fg_dispatch_state: Dict[str, dict] = {}
_fg_dispatch_lock = threading.Lock()
_FG_DISPATCH_TIMEOUT_SEC = 600  # 10 min — item in finished_goods but no vehicle


def check_finished_goods_dispatch(
    camera_id: int,
    floor: str,
    raw_detections: List[dict],
    persons: List[dict],
    zones: dict,
    db,
) -> None:
    """
    Monitor finished goods zone for items awaiting dispatch.

    When a non-Person detection (stock/item) appears in the 'finished_goods'
    zone, start a timer. If no vehicle detection appears in the 'loading' /
    'vehicle' zone within _FG_DISPATCH_TIMEOUT_SEC, fire an alert indicating
    goods are ready but not dispatched.
    """
    if not zones:
        return

    fg_poly = zones.get("finished_goods")
    if not fg_poly:
        return

    from services.zone_service import bbox_in_zone

    now = time.monotonic()
    state_key = str(camera_id)

    # Check for items in finished goods zone
    items_in_fg = [
        d
        for d in raw_detections
        if d.get("label")
        not in (
            "Person",
            "Hardhat",
            "Mask",
            "Safety Vest",
            "Bakery-Head-Cap",
            "vehicle",
        )
        and bbox_in_zone(d.get("bbox", []), fg_poly)
    ]

    # Check for vehicles in loading zone
    loading_poly = zones.get("loading") or zones.get("vehicle")
    vehicle_present = False
    if loading_poly:
        vehicle_present = any(
            d.get("label") == "vehicle"
            and bbox_in_zone(d.get("bbox", []), loading_poly)
            for d in raw_detections
        )

    with _fg_dispatch_lock:
        state = _fg_dispatch_state.setdefault(
            state_key,
            {
                "items_since": None,
                "alert_fired": False,
            },
        )

        if items_in_fg:
            if state["items_since"] is None:
                state["items_since"] = now
                state["alert_fired"] = False

            elapsed = now - state["items_since"]
            if elapsed >= _FG_DISPATCH_TIMEOUT_SEC and not state["alert_fired"]:
                if not vehicle_present:
                    state["alert_fired"] = True
                    _fire_fg_dispatch_alert(
                        db,
                        camera_id,
                        floor,
                        len(items_in_fg),
                        elapsed,
                    )
        else:
            state["items_since"] = None
            state["alert_fired"] = False


def _fire_fg_dispatch_alert(
    db,
    camera_id: int,
    floor: str,
    item_count: int,
    elapsed: float,
):
    from services.alert_service import save_alert

    mins = elapsed / 60
    try:
        save_alert(
            db=db,
            user_id=_get_rule_engine_user_id(db),
            message=(
                f"[DISPATCH] Camera {camera_id} — {item_count} item(s) in "
                f"finished goods zone for {mins:.1f} min with no vehicle detected. "
                f"Goods ready but not dispatched."
            ),
            role="Dispatch Monitor",
            severity="medium",
            detected_issue="Finished goods not dispatched",
            camera_id=camera_id,
            floor=floor,
        )
        log.warning(
            f"[rule_engine] FG dispatch alert cam={camera_id} "
            f"items={item_count} elapsed={elapsed:.0f}s"
        )
    except Exception as exc:
        log.error(f"[rule_engine] FG dispatch alert save failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────────
# SECTION N — Enhanced Machinery Workflow Enforcement (REQ-BHV-03, REQ-BHV-04)
# ─────────────────────────────────────────────────────────────────────────────────
# After a task completes in the dough/biscuit-cutting zone, the worker must
# move to the next zone (e.g. packing). If they remain idle in the same zone
# after machinery stops, fire an alert.
#
# This enhances Section I (check_machinery_zone) by adding a "post-task
# enforcement" mode: when machinery is detected but person was previously
# working, and machinery stops (no longer detected), track whether the person
# leaves the zone within a grace period.
# ─────────────────────────────────────────────────────────────────────────────

_workflow_enforce_state: Dict[str, dict] = {}
_workflow_enforce_lock = threading.Lock()
_WORKFLOW_GRACE_PERIOD_SEC = 120  # 2 min after machine stops to move
_WORKFLOW_COOLDOWN_SEC = 300


def check_workflow_enforcement(
    camera_id: int,
    floor: str,
    raw_detections: List[dict],
    persons: List[dict],
    zones: dict,
    db,
) -> None:
    """
    Enforce post-task zone transitions for dough/biscuit-cutting workers.

    State machine:
      IDLE → MACHINE_ON: machinery detected in zone, person working
      MACHINE_ON → MACHINE_OFF: machinery disappears (task finished)
      MACHINE_OFF → ALERT: person still in zone after grace period
    """
    if not zones:
        return

    from services.zone_service import bbox_in_zone

    now = time.monotonic()

    workflow_zones = {
        "dough": "packing",
        "dough_table": "packing",
        "dough_mixing": "packing",
        "biscuit_cutting": "packing",
        "cutting_machine": "packing",
    }

    for zone_name, next_zone in workflow_zones.items():
        poly = zones.get(zone_name)
        if not poly:
            continue

        machines = [
            d
            for d in raw_detections
            if d.get("label") == "machinery" and bbox_in_zone(d.get("bbox", []), poly)
        ]
        persons_here = [p for p in persons if bbox_in_zone(p.get("bbox", []), poly)]

        state_key = f"{camera_id}_{zone_name}"
        with _workflow_enforce_lock:
            state = _workflow_enforce_state.setdefault(
                state_key,
                {
                    "machine_was_on": False,
                    "machine_off_since": None,
                    "alert_fired": False,
                },
            )

            machine_on = bool(machines)

            if machine_on:
                state["machine_was_on"] = True
                state["machine_off_since"] = None
                state["alert_fired"] = False
                continue

            # Machine is off
            if state["machine_was_on"] and not machine_on:
                if state["machine_off_since"] is None:
                    state["machine_off_since"] = now

                if (
                    state["machine_off_since"] is not None
                    and persons_here
                    and not state["alert_fired"]
                ):
                    elapsed = now - state["machine_off_since"]
                    if elapsed >= _WORKFLOW_GRACE_PERIOD_SEC:
                        if (
                            now
                            - _workflow_enforce_state.get(f"_last_alert_{camera_id}", 0)
                            >= _WORKFLOW_COOLDOWN_SEC
                        ):
                            state["alert_fired"] = True
                            _workflow_enforce_state[f"_last_alert_{camera_id}"] = now
                            _fire_workflow_alert(
                                db,
                                camera_id,
                                floor,
                                zone_name,
                                next_zone,
                                elapsed,
                            )

            if not persons_here:
                state["machine_was_on"] = False
                state["machine_off_since"] = None
                state["alert_fired"] = False


def _fire_workflow_alert(
    db,
    camera_id: int,
    floor: str,
    from_zone: str,
    to_zone: str,
    elapsed: float,
):
    from services.alert_service import save_alert

    try:
        save_alert(
            db=db,
            user_id=_get_rule_engine_user_id(db),
            message=(
                f"[WORKFLOW] Camera {camera_id} — task in '{from_zone}' zone "
                f"appears finished but worker has not moved to '{to_zone}' "
                f"after {elapsed:.0f}s. Worker must transition to next station."
            ),
            role="Factory Worker",
            severity="medium",
            detected_issue="Workflow sequence violation",
            camera_id=camera_id,
            floor=floor,
        )
        log.warning(
            f"[rule_engine] Workflow alert cam={camera_id} "
            f"from={from_zone} to={to_zone}"
        )
    except Exception as exc:
        log.error(f"[rule_engine] workflow alert save failed: {exc}")
