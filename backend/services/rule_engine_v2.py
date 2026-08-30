"""Database-driven rule evaluator for the enterprise event stream.

Rule State Machines
───────────────────
IdleRuleStateMachine      — ACTIVE → IDLE_TIMER_RUNNING → ALERTED
AbsenceRuleStateMachine   — PERSON_PRESENT → ABSENT_TIMER_RUNNING → ALERTED
ShiftStartChecker         — checks if work has started by the shift schedule
CameraStandingChecker     — person blocks camera view > threshold

All thresholds are read from settings / rule config — nothing is hard-coded.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from services.platform_events import Event, bus, emit


# ─────────────────────────────────────────────────────────────────────────────
# Core rule engine — subscribes to all events and evaluates DB-configured rules
# ─────────────────────────────────────────────────────────────────────────────


class ConfigurableRuleEngine:
    def __init__(self) -> None:
        self._cooldowns: Dict[Tuple[int, str], float] = {}
        self._lock = threading.Lock()
        bus.subscribe("*", self.evaluate)

    def evaluate(self, event: Event) -> None:
        if event.event_type in {
            "RULE_MATCHED",
            "ALERT_CREATED",
            "NOTIFICATION_REQUESTED",
            "NOTIFICATION_DELIVERED",
            "NOTIFICATION_FAILED",
        }:
            return
        from database import RuleDefinition, RuleEvaluation, SessionLocal

        db = SessionLocal()
        try:
            rules = (
                db.query(RuleDefinition).filter(RuleDefinition.enabled.is_(True)).all()
            )
            for rule in rules:
                if not self._matches(rule, event):
                    continue
                key = (rule.id, self._scope_key(event))
                with self._lock:
                    last = self._cooldowns.get(key, 0.0)
                    if time.monotonic() - last < float(rule.cooldown_sec or 0):
                        continue
                    self._cooldowns[key] = time.monotonic()
                db.add(
                    RuleEvaluation(
                        rule_id=rule.id,
                        event_id=event.event_id,
                        camera_id=event.camera_id,
                        outcome="matched",
                        reason="event and configured conditions matched",
                        context_json=json.dumps(event.to_dict(), default=str),
                    )
                )
                db.commit()
                emit(
                    "RULE_MATCHED",
                    camera_id=event.camera_id,
                    zone_id=event.zone_id,
                    track_id=event.track_id,
                    employee_id=event.employee_id,
                    confidence=event.confidence,
                    correlation_id=event.correlation_id,
                    calibration_version=event.calibration_version,
                    source="rule-engine",
                    payload={
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "priority": rule.priority,
                        "notification_targets": self._json(
                            rule.notification_targets_json, []
                        ),
                        "escalation": self._json(rule.escalation_json, {}),
                        "event": event.to_dict(),
                    },
                )
        except Exception:
            db.rollback()
        finally:
            db.close()

    def _matches(self, rule, event: Event) -> bool:
        if rule.zone_id and rule.zone_id != event.zone_id:
            return False
        workflow_stage = (
            event.payload.get("workflow_stage")
            or event.payload.get("stage")
            or event.payload.get("context", {}).get("workflow_stage")
        )
        if rule.workflow_stage and workflow_stage != rule.workflow_stage:
            return False
        required = set(self._json(rule.required_events_json, []))
        forbidden = set(self._json(rule.forbidden_events_json, []))
        # A rule must explicitly declare at least one input event. This avoids
        # a partially configured rule matching every operational event.
        if (
            not required
            or event.event_type in forbidden
            or event.event_type not in required
        ):
            return False
        if (
            rule.confidence_threshold is not None
            and (event.confidence or 0) < rule.confidence_threshold
        ):
            return False
        conditions = self._json(rule.conditions_json, {})
        calibration_status = event.payload.get(
            "calibration_status"
        ) or event.payload.get("context", {}).get("calibration_status")
        if conditions.get("require_calibrated") and calibration_status != "calibrated":
            return False
        return True

    @staticmethod
    def _scope_key(event: Event) -> str:
        return f"{event.camera_id}:{event.zone_id}:{event.track_id or '-'}"

    @staticmethod
    def _json(value: str, default):
        try:
            return json.loads(value or "")
        except (TypeError, ValueError):
            return default


rule_engine_v2 = ConfigurableRuleEngine()


# ─────────────────────────────────────────────────────────────────────────────
# IDLE RULE STATE MACHINE
# State: ACTIVE → IDLE_TIMER_RUNNING → ALERTED
# ─────────────────────────────────────────────────────────────────────────────


class IdleRuleStateMachine:
    """Fires EMPLOYEE_IDLE when a tracked person stays stationary beyond threshold.

    Uses the track's movement_state from the tracker.  When pose/hand-motion
    signals are available they can be fed via mark_active() to suppress the timer
    even when the body position is stable (e.g. packing workers).

    Args:
        idle_threshold_sec: Seconds of continuous inactivity before alert.
        cooldown_sec:       Minimum seconds between repeated alerts per track.
    """

    def __init__(
        self,
        idle_threshold_sec: Optional[float] = None,
        cooldown_sec: float = 60.0,
    ) -> None:
        from config import settings
        self._threshold = idle_threshold_sec or settings.IDLE_LIMITS.get("default", 300)
        self._cooldown = cooldown_sec
        self._lock = threading.Lock()
        # { (camera_id, track_id): {"state": str, "timer_start": float, "last_alert": float} }
        self._states: Dict[Tuple[int, str], Dict] = {}

    def update(
        self,
        camera_id: int,
        track_id: str,
        movement_state: str,   # "moving" | "stationary"
        zone: Optional[str] = None,
    ) -> None:
        """Call once per tracker update tick for each active person."""
        from config import settings

        # Zone-specific threshold
        threshold = self._threshold
        if zone and zone in settings.IDLE_LIMITS:
            threshold = settings.IDLE_LIMITS[zone]

        key = (camera_id, track_id)
        now = time.monotonic()
        with self._lock:
            s = self._states.setdefault(key, {"state": "ACTIVE", "timer_start": now, "last_alert": 0.0})
            if movement_state == "moving":
                s["state"] = "ACTIVE"
                s["timer_start"] = now
                return
            # Stationary
            if s["state"] == "ACTIVE":
                s["state"] = "IDLE_TIMER_RUNNING"
                s["timer_start"] = now
            elapsed = now - s["timer_start"]
            if elapsed >= threshold and s["state"] == "IDLE_TIMER_RUNNING":
                since_last = now - s["last_alert"]
                if since_last >= self._cooldown:
                    s["state"] = "ALERTED"
                    s["last_alert"] = now
                    self._fire_alert(camera_id, track_id, int(elapsed), zone)

    def mark_active(self, camera_id: int, track_id: str) -> None:
        """Call from pose/motion module to suppress idle timer even while stationary."""
        key = (camera_id, track_id)
        with self._lock:
            s = self._states.get(key)
            if s:
                s["state"] = "ACTIVE"
                s["timer_start"] = time.monotonic()

    def get_state(self, camera_id: int, track_id: str) -> Optional[str]:
        """Return the current state machine state for a track ('ACTIVE', 'IDLE_TIMER_RUNNING', 'ALERTED')."""
        with self._lock:
            s = self._states.get((camera_id, track_id))
            return s["state"] if s else None

    def remove_track(self, camera_id: int, track_id: str) -> None:
        with self._lock:
            self._states.pop((camera_id, track_id), None)

    @staticmethod
    def _fire_alert(camera_id: int, track_id: str, idle_seconds: int, zone: Optional[str]) -> None:
        emit(
            "EMPLOYEE_IDLE",
            camera_id=camera_id,
            track_id=track_id,
            source="idle-rule-engine",
            payload={
                "idle_seconds": idle_seconds,
                "zone": zone,
                "message": f"Employee idle for {idle_seconds // 60}m {idle_seconds % 60}s",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# SHOP ABSENCE STATE MACHINE
# State: PERSON_PRESENT → ABSENT_TIMER_RUNNING → ALERTED
# ─────────────────────────────────────────────────────────────────────────────


class AbsenceRuleStateMachine:
    """Fires SHOP_ABSENCE when no person is in the shop zone beyond threshold.

    Args:
        absence_threshold_sec: Seconds without any detected person before alert.
        cooldown_sec:          Minimum seconds between repeated alerts per camera.
        target_zone:           Zone name to monitor (default 'shop_counter').
    """

    def __init__(
        self,
        absence_threshold_sec: Optional[float] = None,
        cooldown_sec: float = 120.0,
        target_zone: str = "shop_counter",
    ) -> None:
        from config import settings
        self._threshold = absence_threshold_sec or settings.SHOP_ABSENCE_THRESHOLD_SEC
        self._cooldown = cooldown_sec
        self._zone = target_zone
        self._lock = threading.Lock()
        # { camera_id: {"state": str, "timer_start": float, "last_alert": float} }
        self._states: Dict[int, Dict] = {}

    def update(self, camera_id: int, persons_in_shop: int) -> None:
        """Call after each detection tick with count of persons in the monitored zone."""
        now = time.monotonic()
        with self._lock:
            s = self._states.setdefault(
                camera_id,
                {"state": "PERSON_PRESENT", "timer_start": now, "last_alert": 0.0},
            )
            if persons_in_shop > 0:
                s["state"] = "PERSON_PRESENT"
                s["timer_start"] = now
                return
            # No persons
            if s["state"] == "PERSON_PRESENT":
                s["state"] = "ABSENT_TIMER_RUNNING"
                s["timer_start"] = now
            elapsed = now - s["timer_start"]
            if elapsed >= self._threshold and s["state"] == "ABSENT_TIMER_RUNNING":
                since_last = now - s["last_alert"]
                if since_last >= self._cooldown:
                    s["state"] = "ALERTED"
                    s["last_alert"] = now
                    self._fire_alert(camera_id, int(elapsed))

    def get_state(self, camera_id: int) -> Optional[str]:
        """Return current state ('PERSON_PRESENT', 'ABSENT_TIMER_RUNNING', 'ALERTED')."""
        with self._lock:
            s = self._states.get(camera_id)
            return s["state"] if s else None

    def reset(self, camera_id: int) -> None:
        with self._lock:
            self._states.pop(camera_id, None)

    @staticmethod
    def _fire_alert(camera_id: int, absent_seconds: int) -> None:
        emit(
            "SHOP_ABSENCE",
            camera_id=camera_id,
            source="absence-rule-engine",
            payload={
                "absent_seconds": absent_seconds,
                "zone": "shop_counter",
                "message": f"Shop unattended for {absent_seconds}s",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# SHIFT START CHECKER
# ─────────────────────────────────────────────────────────────────────────────


class ShiftStartChecker:
    """Fires SHIFT_NOT_STARTED if no person detected by the configured shift start time.

    Checks once per camera per day.  Schedules are read from settings.SHIFT_SCHEDULES
    and can be overridden per floor/zone.

    Call tick() from the camera detection loop or a periodic background thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # { (camera_id, date_str): bool }  — True means already fired today
        self._fired: Dict[Tuple[int, str], bool] = {}

    def tick(
        self,
        camera_id: int,
        floor: str,
        persons_detected: int,
    ) -> None:
        """Call once per camera detection cycle."""
        from config import settings

        schedule = settings.SHIFT_SCHEDULES.get(floor) or settings.SHIFT_SCHEDULES.get("ground", "08:00")
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        key = (camera_id, date_str)

        with self._lock:
            if self._fired.get(key):
                return  # already fired today

        try:
            required_hour, required_min = map(int, schedule.split(":"))
        except ValueError:
            return

        shift_time = now.replace(hour=required_hour, minute=required_min, second=0, microsecond=0)
        if now < shift_time:
            return  # shift hasn't started yet — nothing to check

        if persons_detected > 0:
            with self._lock:
                self._fired[key] = True  # work has started — suppress today's alert
            return

        # Past shift time, no person detected
        with self._lock:
            if self._fired.get(key):
                return
            self._fired[key] = True  # only fire once

        emit(
            "SHIFT_NOT_STARTED",
            camera_id=camera_id,
            source="shift-rule-engine",
            payload={
                "floor": floor,
                "required_time": schedule,
                "message": f"Shift on {floor} floor not started by {schedule}",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA STANDING CHECKER
# ─────────────────────────────────────────────────────────────────────────────


class CameraStandingChecker:
    """Fires CAMERA_STANDING when a person stays in the camera_standing zone too long.

    Uses track IDs (not raw detections) to avoid false positives from momentary
    detections of the same person.

    Args:
        threshold_sec: Seconds in zone before alert (from settings by default).
        cooldown_sec:  Minimum seconds between repeated alerts per track.
    """

    def __init__(
        self,
        threshold_sec: Optional[float] = None,
        cooldown_sec: float = 120.0,
    ) -> None:
        from config import settings
        self._threshold = threshold_sec or settings.CAMERA_STANDING_THRESHOLD_SEC
        self._cooldown = cooldown_sec
        self._lock = threading.Lock()
        # { (camera_id, track_id): {"entry_ts": float, "last_alert": float} }
        self._states: Dict[Tuple[int, str], Dict] = {}

    def update(
        self,
        camera_id: int,
        track_id: str,
        in_standing_zone: bool,
    ) -> None:
        """Call each detection tick for each tracked person."""
        key = (camera_id, track_id)
        now = time.monotonic()
        with self._lock:
            if not in_standing_zone:
                self._states.pop(key, None)
                return
            s = self._states.setdefault(key, {"entry_ts": now, "last_alert": 0.0})
            elapsed = now - s["entry_ts"]
            if elapsed >= self._threshold:
                since_last = now - s["last_alert"]
                if since_last >= self._cooldown:
                    s["last_alert"] = now
                    self._fire_alert(camera_id, track_id, int(elapsed))

    def remove_track(self, camera_id: int, track_id: str) -> None:
        with self._lock:
            self._states.pop((camera_id, track_id), None)

    @staticmethod
    def _fire_alert(camera_id: int, track_id: str, elapsed_sec: int) -> None:
        emit(
            "CAMERA_STANDING",
            camera_id=camera_id,
            track_id=track_id,
            source="camera-standing-rule-engine",
            payload={
                "elapsed_seconds": elapsed_sec,
                "zone": "camera_standing",
                "message": f"Person #{track_id} blocking camera for {elapsed_sec}s",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instances (imported by the detection loop)
# ─────────────────────────────────────────────────────────────────────────────

idle_rule = IdleRuleStateMachine()
absence_rule = AbsenceRuleStateMachine()
shift_checker = ShiftStartChecker()
camera_standing_rule = CameraStandingChecker()
