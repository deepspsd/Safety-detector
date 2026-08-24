"""Durable, in-process event backbone for the enterprise surveillance path.

The deployment currently runs as one FastAPI process, so this module provides
the same publish/subscribe contract that can later be backed by Kafka, NATS or
Redis Streams without changing any producer/consumer interfaces.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

log = logging.getLogger("platform.events")


@dataclass(frozen=True)
class Event:
    event_type: str
    camera_id: Optional[int] = None
    zone_id: Optional[int] = None
    track_id: Optional[str] = None
    employee_id: Optional[int] = None
    confidence: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "platform"
    correlation_id: Optional[str] = None
    calibration_version: Optional[int] = None
    event_id: str = field(default_factory=lambda: uuid4().hex)
    occurred_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["occurred_at"] = self.occurred_at.isoformat()
        return data


Subscriber = Callable[[Event], None]


class EventBus:
    """Thread-safe event dispatcher with durable persistence before fan-out."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Subscriber]] = {}
        self._lock = threading.RLock()

    def subscribe(self, event_type: str, handler: Subscriber) -> None:
        with self._lock:
            handlers = self._subscribers.setdefault(event_type, [])
            if handler not in handlers:
                handlers.append(handler)

    def publish(self, event: Event, persist: bool = True) -> Event:
        if persist:
            self._persist(event)
        with self._lock:
            handlers = list(self._subscribers.get(event.event_type, []))
            handlers.extend(self._subscribers.get("*", []))
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                log.exception("Event subscriber failed for %s", event.event_type)
        return event

    @staticmethod
    def _persist(event: Event) -> None:
        try:
            from database import SessionLocal, SurveillanceEvent

            db = SessionLocal()
            try:
                db.add(
                    SurveillanceEvent(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        occurred_at=event.occurred_at,
                        camera_id=event.camera_id,
                        zone_id=event.zone_id,
                        track_id=event.track_id,
                        employee_id=event.employee_id,
                        calibration_version=event.calibration_version,
                        confidence=event.confidence,
                        correlation_id=event.correlation_id,
                        source=event.source,
                        payload_json=json.dumps(event.payload, default=str),
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception:
            # Events must never take the camera loop down.  A health record is
            # written by the caller/monitor where persistence is unavailable.
            log.exception("Could not persist event %s", event.event_type)


bus = EventBus()


def emit(event_type: str, **kwargs: Any) -> Event:
    """Convenience factory used by all independent platform services."""
    return bus.publish(Event(event_type=event_type, **kwargs))
