"""Alert engine consumes RULE_MATCHED only; notification remains a separate concern."""
from __future__ import annotations

import json
from uuid import uuid4

from services.platform_events import Event, bus, emit


class AlertEngine:
    def __init__(self) -> None:
        bus.subscribe("RULE_MATCHED", self.create_alert)

    def create_alert(self, event: Event) -> None:
        from database import SessionLocal, AlertCase, User
        db = SessionLocal()
        try:
            payload = event.payload
            title = payload.get("rule_name", "Configured surveillance rule")
            case = AlertCase(public_id=uuid4().hex, rule_id=payload.get("rule_id"), camera_id=event.camera_id,
                             event_id=payload.get("event", {}).get("event_id"), severity=payload.get("priority", "warning"),
                             title=title, details_json=json.dumps(payload, default=str))
            db.add(case)
            db.commit()
            db.refresh(case)
            emit("ALERT_CREATED", camera_id=event.camera_id, zone_id=event.zone_id, track_id=event.track_id,
                 correlation_id=case.public_id, source="alert-engine",
                 payload={"alert_case_id": case.id, "public_id": case.public_id, "title": title,
                          "severity": case.severity, "targets": payload.get("notification_targets", []),
                          "event": payload.get("event", {})})
            emit("NOTIFICATION_REQUESTED", camera_id=event.camera_id, correlation_id=case.public_id,
                 source="alert-engine", payload={"alert_case_id": case.id, "targets": payload.get("notification_targets", []),
                                                  "title": title, "severity": case.severity})
        except Exception:
            db.rollback()
        finally:
            db.close()


alert_engine_v2 = AlertEngine()
