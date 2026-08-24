"""Notification delivery adapter for enterprise alert cases.

It consumes notification requests, records every delivery attempt, and keeps
provider-specific code outside the alert and rule engines. New channels only
need an adapter branch (or a future worker consuming the same event contract).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable

from services.platform_events import Event, bus, emit


class NotificationEngine:
    def __init__(self) -> None:
        bus.subscribe("NOTIFICATION_REQUESTED", self.deliver)

    def deliver(self, event: Event) -> None:
        from database import AlertCase, Camera, NotificationDelivery, SessionLocal

        payload = event.payload
        targets = payload.get("targets") or [{"channel": "dashboard"}]
        db = SessionLocal()
        outcomes = []
        try:
            case = (
                db.query(AlertCase)
                .filter(AlertCase.id == payload.get("alert_case_id"))
                .first()
            )
            camera = (
                db.query(Camera).filter(Camera.id == event.camera_id).first()
                if event.camera_id
                else None
            )
            for target in self._normalise_targets(targets):
                delivery = NotificationDelivery(
                    alert_case_id=case.id if case else None,
                    channel=target["channel"],
                    target=target.get("target"),
                    status="queued",
                    payload_json=json.dumps(payload, default=str),
                )
                db.add(delivery)
                db.flush()
                self._send(delivery, target, payload, camera)
                outcomes.append(
                    {
                        "delivery_id": delivery.id,
                        "channel": target["channel"],
                        "status": delivery.status,
                    }
                )
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()
        for outcome in outcomes:
            emit(
                (
                    "NOTIFICATION_DELIVERED"
                    if outcome["status"] == "delivered"
                    else "NOTIFICATION_FAILED"
                ),
                camera_id=event.camera_id,
                correlation_id=str(payload.get("alert_case_id") or ""),
                source="notification-engine",
                payload=outcome,
            )

    @staticmethod
    def _normalise_targets(targets: Iterable[Any]):
        for target in targets:
            if isinstance(target, str):
                yield {"channel": target.lower()}
            elif isinstance(target, dict) and target.get("channel"):
                yield {
                    "channel": str(target["channel"]).lower(),
                    "target": target.get("target"),
                }

    def _send(
        self, delivery, target: Dict[str, Any], payload: Dict[str, Any], camera
    ) -> None:
        channel = target["channel"]
        delivery.attempted_at = datetime.utcnow()
        if channel in {"dashboard", "push"}:
            # The dashboard and installed PWA read the durable AlertCase/
            # NotificationDelivery records. A websocket/push provider can be
            # added later without changing rules or alerts.
            delivery.status = "delivered"
            delivery.delivered_at = delivery.attempted_at
        elif channel == "telegram":
            from services.notification_service import send_telegram_alert

            sent = send_telegram_alert(
                message=payload.get("title", "Surveillance alert"),
                severity=payload.get("severity", "warning"),
                floor=getattr(camera, "floor", None),
                camera_name=getattr(camera, "name", None),
            )
            delivery.status = "delivered" if sent else "failed"
            delivery.delivered_at = delivery.attempted_at if sent else None
            delivery.error_message = (
                None
                if sent
                else "Telegram provider was unavailable or rejected delivery"
            )
        else:
            delivery.status = "queued"
            delivery.error_message = f"Channel '{channel}' has no configured provider"


notification_engine = NotificationEngine()
