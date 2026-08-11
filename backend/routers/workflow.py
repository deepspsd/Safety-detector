"""
routers/workflow.py — Cash, Stock, Lift, and Packing workflow events API
========================================================================
Endpoints
---------
  GET /workflow/cash-events     Recent unauthorized cash zone access alerts
  GET /workflow/stock-events    Recent exposed-stock alerts
  GET /workflow/lift-events     Recent lift entry/exit events
  GET /workflow/packing-events  Recent packing zone idle alerts
  GET /workflow/summary         Combined summary for the workflow dashboard
"""

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database import get_db, Alert
from routers.auth import get_current_user

log = logging.getLogger("workflow_router")
router = APIRouter(prefix="/workflow", tags=["workflow"])


def _alert_to_dict(a: Alert) -> dict:
    return {
        "id":            a.id,
        "message":       a.message,
        "severity":      a.severity,
        "detected_issue": a.detected_issue,
        "camera_id":     getattr(a, "camera_id", None),
        "timestamp":     a.timestamp.isoformat(),
        "status":        getattr(a, "status", "confirmed"),
    }


# ── Cash zone ──────────────────────────────────────────────────────────────────

@router.get("/cash-events")
def get_cash_events(
    limit:        int     = Query(50, le=200),
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """Recent unauthorized cash zone access events."""
    rows = (
        db.query(Alert)
        .filter(Alert.detected_issue == "Unauthorized cashbox access")
        .order_by(Alert.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [_alert_to_dict(r) for r in rows]


# ── Stock zone ─────────────────────────────────────────────────────────────────

@router.get("/stock-events")
def get_stock_events(
    limit:        int     = Query(50, le=200),
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """Recent exposed-stock alerts."""
    rows = (
        db.query(Alert)
        .filter(Alert.detected_issue == "Exposed stock detected")
        .order_by(Alert.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [_alert_to_dict(r) for r in rows]


# ── Lift events ────────────────────────────────────────────────────────────────

@router.get("/lift-events")
def get_lift_events(
    camera_id:    Optional[int] = Query(None),
    limit:        int           = Query(100, le=500),
    db:           Session       = Depends(get_db),
    current_user                 = Depends(get_current_user),
):
    """Recent lift entry/exit events."""
    from services.lift_monitor import get_recent_events
    return get_recent_events(db, camera_id=camera_id, limit=limit)


# ── Packing zone ───────────────────────────────────────────────────────────────

@router.get("/packing-events")
def get_packing_events(
    camera_id:    Optional[int] = Query(None),
    db:           Session       = Depends(get_db),
    current_user                 = Depends(get_current_user),
):
    """Recent packing zone idle alerts."""
    from services.packing_monitor import get_packing_summary
    return get_packing_summary(db, camera_id=camera_id)


# ── Summary ────────────────────────────────────────────────────────────────────

@router.get("/summary")
def get_workflow_summary(
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """Combined workflow summary for the dashboard — all event types today."""
    today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    cash_count = db.query(Alert).filter(
        Alert.detected_issue == "Unauthorized cashbox access",
        Alert.timestamp >= today,
    ).count()

    stock_count = db.query(Alert).filter(
        Alert.detected_issue == "Exposed stock detected",
        Alert.timestamp >= today,
    ).count()

    lift_count = db.query(Alert).filter(
        Alert.detected_issue == "Lift zone idle",
        Alert.timestamp >= today,
    ).count()

    packing_count = db.query(Alert).filter(
        Alert.detected_issue == "Packing zone idle",
        Alert.timestamp >= today,
    ).count()

    return {
        "cash_events_today":    cash_count,
        "stock_events_today":   stock_count,
        "lift_events_today":    lift_count,
        "packing_events_today": packing_count,
        "total_today":          cash_count + stock_count + lift_count + packing_count,
    }
