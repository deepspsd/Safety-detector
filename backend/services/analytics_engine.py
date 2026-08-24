"""Read-only analytics projection over durable platform records.

This service deliberately does not import alert, rule, notification, detection,
or workflow engines. It can later be moved to a reporting worker unchanged.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func


class AnalyticsEngine:
    def summary(self, db, days: int) -> dict:
        from database import AlertCase, Camera, ContextSnapshot, SurveillanceEvent

        since = datetime.utcnow() - timedelta(days=days)
        event_counts = (
            db.query(SurveillanceEvent.event_type, func.count(SurveillanceEvent.id))
            .filter(SurveillanceEvent.occurred_at >= since)
            .group_by(SurveillanceEvent.event_type)
            .all()
        )
        severity_counts = (
            db.query(AlertCase.severity, func.count(AlertCase.id))
            .filter(AlertCase.opened_at >= since)
            .group_by(AlertCase.severity)
            .all()
        )
        zone_visits = (
            db.query(ContextSnapshot.zone_id, func.count(ContextSnapshot.id))
            .filter(
                ContextSnapshot.occurred_at >= since,
                ContextSnapshot.zone_id.isnot(None),
            )
            .group_by(ContextSnapshot.zone_id)
            .all()
        )
        camera_health = [
            {
                "camera_id": c.id,
                "name": c.name,
                "status": c.health_status,
                "calibration_status": c.calibration_status,
                "heartbeat_at": c.heartbeat_at.isoformat() if c.heartbeat_at else None,
            }
            for c in db.query(Camera).order_by(Camera.floor, Camera.name).all()
        ]
        return {
            "from": since.isoformat(),
            "days": days,
            "events": dict(event_counts),
            "violations": dict(severity_counts),
            "zone_utilization": {str(k): v for k, v in zone_visits},
            "camera_health": camera_health,
        }


analytics_engine = AnalyticsEngine()
