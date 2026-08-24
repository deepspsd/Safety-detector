"""Database-driven rule evaluator for the enterprise event stream."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple

from services.platform_events import Event, bus, emit


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
