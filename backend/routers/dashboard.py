"""
Dashboard summary endpoint.

GET /dashboard/summary  — returns the single top-level KPI block the
                          dashboard home page needs in one round-trip.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from auth_utils import get_current_user
from database import Alert, AlertCase, Camera, SessionLocal, User
from services import camera_manager
from services.tracking_layer import tracker as _tracker

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/summary")
def dashboard_summary(
    db: Session = Depends(_db),
    current_user: User = Depends(get_current_user),
):
    """
    Returns a real-time KPI snapshot:

    - total / online / offline cameras
    - employees currently detected (unique track IDs across all cameras)
    - active (unresolved) alerts count
    - alerts raised in the last 24 h
    - recent events (last 10 alert cases)
    - current mock_mode flag
    """
    from config import settings

    # ── Camera counts ────────────────────────────────────────────────────────
    cam_statuses = camera_manager.list_status()
    total_cams = len(cam_statuses)
    online_cams = sum(1 for c in cam_statuses if c.get("status") == "online")
    offline_cams = total_cams - online_cams

    # ── Live person count ────────────────────────────────────────────────────
    unique_track_ids: set = set()
    for (cam_id, tid) in _tracker.all_track_keys():
        obs = _tracker.get_track(cam_id, tid)
        if obs and (time.time() - obs.last_seen_at.timestamp()) < 10:
            unique_track_ids.add((cam_id, tid))
    employees_detected = len(unique_track_ids)

    # ── Alert counts ─────────────────────────────────────────────────────────
    cutoff_24h = datetime.utcnow() - timedelta(hours=24)
    try:
        active_alerts = (
            db.query(AlertCase)
            .filter(AlertCase.status.in_(["open", "active", "new"]))
            .count()
        )
        alerts_24h = (
            db.query(AlertCase)
            .filter(AlertCase.created_at >= cutoff_24h)
            .count()
        )
        recent_events = [
            {
                "id": a.id,
                "public_id": a.public_id,
                "title": a.title,
                "severity": a.severity,
                "camera_id": a.camera_id,
                "status": a.status,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in db.query(AlertCase)
            .order_by(AlertCase.id.desc())
            .limit(10)
            .all()
        ]
    except Exception:
        active_alerts = 0
        alerts_24h = 0
        recent_events = []

    return {
        "cameras": {
            "total": total_cams,
            "online": online_cams,
            "offline": offline_cams,
        },
        "employees_detected": employees_detected,
        "alerts": {
            "active": active_alerts,
            "last_24h": alerts_24h,
        },
        "recent_events": recent_events,
        "mock_mode": settings.MOCK_MODE,
        "generated_at": datetime.utcnow().isoformat(),
    }
