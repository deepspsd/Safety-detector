"""
routers/workflow.py — Cash, Stock, Lift, Packing, Idle, Dress-code, Oven, Window, Eating, Dispatch, Cylinder workflow events API
==========================================================================================
Endpoints
---------
  GET /workflow/cash-events         Recent unauthorized cash zone access alerts
  GET /workflow/stock-events        Recent exposed-stock alerts
  GET /workflow/lift-events         Recent lift entry/exit events
  GET /workflow/packing-events      Recent packing zone idle alerts
  GET /workflow/idle-events         Recent idle alerts (all floors / by floor)
  GET /workflow/dress-code-events   Head-cap / uniform / dress-code violations
  GET /workflow/oven-events         Floor-2 oven/fire/gas-waste alerts
  GET /workflow/window-events       Window throwing/stealing alerts
  GET /workflow/eating-events       Eating from store alerts
  GET /workflow/dispatch-events     Finished goods dispatch alerts
  GET /workflow/cylinder-events     Cylinder monitoring events
  GET /workflow/summary             Combined summary for the workflow dashboard
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
        "floor":         getattr(a, "floor", None),
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


# ── Idle events (all floors) ───────────────────────────────────────────────────

@router.get("/idle-events")
def get_idle_events(
    floor:        Optional[str] = Query(None, description="ground | first | second | shop"),
    limit:        int           = Query(50, le=200),
    db:           Session       = Depends(get_db),
    current_user                 = Depends(get_current_user),
):
    """
    Recent idle alerts across all floors (or filtered by floor).
    Covers: 5-min idle rule, packing section idle, dough section idle,
    shop absent >1 min, standing in front of camera >1 min.
    """
    # Idle alerts come from multiple detected_issue strings
    _IDLE_ISSUES = [
        "Idle too long",
        "Employee idle",
        "Worker idle",
        "Packing zone idle",
        "Standing in front of camera",
        "Absent from shop",
        "Dough section idle",
    ]
    query = db.query(Alert).filter(
        Alert.detected_issue.in_(_IDLE_ISSUES)
    )
    if floor:
        query = query.filter(Alert.floor == floor)
    rows = query.order_by(Alert.timestamp.desc()).limit(limit).all()

    # Also catch partial matches (e.g. "Employee idle for 7 min")
    if len(rows) < 5:
        fuzzy_rows = (
            db.query(Alert)
            .filter(Alert.message.ilike("%idle%"))
            .order_by(Alert.timestamp.desc())
            .limit(limit)
            .all()
        )
        seen_ids = {r.id for r in rows}
        for r in fuzzy_rows:
            if r.id not in seen_ids:
                rows.append(r)
        rows = sorted(rows, key=lambda x: x.timestamp, reverse=True)[:limit]

    return [_alert_to_dict(r) for r in rows]


# ── Dress code events ──────────────────────────────────────────────────────────

@router.get("/dress-code-events")
def get_dress_code_events(
    floor:        Optional[str] = Query(None),
    limit:        int           = Query(50, le=200),
    db:           Session       = Depends(get_db),
    current_user                 = Depends(get_current_user),
):
    """
    Dress-code violation alerts: no head cap, no uniform, bangles detected,
    no mask, chewing/eating, hair issue.
    """
    _DRESS_ISSUES = [
        "No Head Cap",
        "No Hardhat",
        "No Mask",
        "Bangle detected",
        "Bangles not allowed",
        "Uniform violation",
        "No Uniform",
        "Dress code violation",
        "Chewing detected",
        "Eating at workstation",
    ]
    query = db.query(Alert).filter(
        Alert.detected_issue.in_(_DRESS_ISSUES)
    )
    if floor:
        query = query.filter(Alert.floor == floor)
    rows = query.order_by(Alert.timestamp.desc()).limit(limit).all()

    # Fuzzy fallback for partial matches ("No Hardhat detected")
    if len(rows) < 5:
        for keyword in ["head cap", "hardhat", "uniform", "bangle", "dress code"]:
            fuzzy = (
                db.query(Alert)
                .filter(Alert.detected_issue.ilike(f"%{keyword}%"))
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )
            seen_ids = {r.id for r in rows}
            for r in fuzzy:
                if r.id not in seen_ids:
                    rows.append(r)
        rows = sorted(rows, key=lambda x: x.timestamp, reverse=True)[:limit]

    return [_alert_to_dict(r) for r in rows]


# ── Oven / fire / gas-waste events (Floor 2) ───────────────────────────────────

@router.get("/oven-events")
def get_oven_events(
    limit:        int     = Query(50, le=200),
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """
    Floor-2 oven / fire / gas monitoring alerts:
    - No one present when fire is on
    - Water boiling >10 min unattended
    - Oil heated idle >10 min
    - Gas waste detected
    """
    _OVEN_ISSUES = [
        "Oven unattended",
        "Gas waste",
        "Fire unattended",
        "Oil idle",
        "Water boiling unattended",
        "No attendant at oven",
    ]
    query = db.query(Alert).filter(
        Alert.detected_issue.in_(_OVEN_ISSUES)
    )
    rows = query.order_by(Alert.timestamp.desc()).limit(limit).all()

    # Fuzzy fallback
    if len(rows) < 5:
        for keyword in ["oven", "gas", "fire", "boiling", "oil idle"]:
            fuzzy = (
                db.query(Alert)
                .filter(Alert.detected_issue.ilike(f"%{keyword}%"))
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )
            seen_ids = {r.id for r in rows}
            for r in fuzzy:
                if r.id not in seen_ids:
                    rows.append(r)
        rows = sorted(rows, key=lambda x: x.timestamp, reverse=True)[:limit]

    return [_alert_to_dict(r) for r in rows]


# ── Window throwing/stealing events ────────────────────────────────────────────

@router.get("/window-events")
def get_window_events(
    limit:        int     = Query(50, le=200),
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """Recent window throwing/stealing alerts — immediate alerts only."""
    _WINDOW_ISSUES = [
        "Window throwing detected",
        "Window theft detected",
        "Items thrown from window",
        "Stealing through window",
    ]
    query = db.query(Alert).filter(
        Alert.detected_issue.in_(_WINDOW_ISSUES)
    )
    rows = query.order_by(Alert.timestamp.desc()).limit(limit).all()

    # Fuzzy fallback
    if len(rows) < 5:
        for keyword in ["window", "throw", "steal"]:
            fuzzy = (
                db.query(Alert)
                .filter(Alert.detected_issue.ilike(f"%{keyword}%"))
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )
            seen_ids = {r.id for r in rows}
            for r in fuzzy:
                if r.id not in seen_ids:
                    rows.append(r)
        rows = sorted(rows, key=lambda x: x.timestamp, reverse=True)[:limit]

    return [_alert_to_dict(r) for r in rows]


# ── Eating from store events ───────────────────────────────────────────────────

@router.get("/eating-events")
def get_eating_events(
    limit:        int     = Query(50, le=200),
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """Recent eating-from-store alerts (MediaPipe pose wrist-to-nose)."""
    _EATING_ISSUES = [
        "Eating from store",
        "Eating detected",
        "Chewing detected",
    ]
    query = db.query(Alert).filter(
        Alert.detected_issue.in_(_EATING_ISSUES)
    )
    rows = query.order_by(Alert.timestamp.desc()).limit(limit).all()

    # Fuzzy fallback
    if len(rows) < 5:
        for keyword in ["eating", "chewing", "store"]:
            fuzzy = (
                db.query(Alert)
                .filter(Alert.detected_issue.ilike(f"%{keyword}%"))
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )
            seen_ids = {r.id for r in rows}
            for r in fuzzy:
                if r.id not in seen_ids:
                    rows.append(r)
        rows = sorted(rows, key=lambda x: x.timestamp, reverse=True)[:limit]

    return [_alert_to_dict(r) for r in rows]


# ── Finished goods dispatch events ─────────────────────────────────────────────

@router.get("/dispatch-events")
def get_dispatch_events(
    limit:        int     = Query(50, le=200),
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """Recent finished goods dispatch alerts — items moved to dispatch without vehicle."""
    _DISPATCH_ISSUES = [
        "Dispatch without vehicle",
        "Finished goods not loaded",
        "FG dispatch alert",
    ]
    query = db.query(Alert).filter(
        Alert.detected_issue.in_(_DISPATCH_ISSUES)
    )
    rows = query.order_by(Alert.timestamp.desc()).limit(limit).all()

    # Fuzzy fallback
    if len(rows) < 5:
        for keyword in ["dispatch", "finished goods", "vehicle"]:
            fuzzy = (
                db.query(Alert)
                .filter(Alert.detected_issue.ilike(f"%{keyword}%"))
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )
            seen_ids = {r.id for r in rows}
            for r in fuzzy:
                if r.id not in seen_ids:
                    rows.append(r)
        rows = sorted(rows, key=lambda x: x.timestamp, reverse=True)[:limit]

    return [_alert_to_dict(r) for r in rows]


# ── Cylinder monitoring events ─────────────────────────────────────────────────

@router.get("/cylinder-events")
def get_cylinder_events(
    limit:        int     = Query(50, le=200),
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """Recent cylinder monitoring events — count changes, usage-day alerts."""
    _CYLINDER_ISSUES = [
        "Cylinder count change",
        "Cylinder usage alert",
        "Cylinder low count",
    ]
    query = db.query(Alert).filter(
        Alert.detected_issue.in_(_CYLINDER_ISSUES)
    )
    rows = query.order_by(Alert.timestamp.desc()).limit(limit).all()

    # Fuzzy fallback
    if len(rows) < 5:
        for keyword in ["cylinder", "gas cylinder"]:
            fuzzy = (
                db.query(Alert)
                .filter(Alert.detected_issue.ilike(f"%{keyword}%"))
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )
            seen_ids = {r.id for r in rows}
            for r in fuzzy:
                if r.id not in seen_ids:
                    rows.append(r)
        rows = sorted(rows, key=lambda x: x.timestamp, reverse=True)[:limit]

    return [_alert_to_dict(r) for r in rows]


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

    idle_count = db.query(Alert).filter(
        Alert.message.ilike("%idle%"),
        Alert.timestamp >= today,
    ).count()

    dress_code_count = db.query(Alert).filter(
        Alert.detected_issue.in_(["No Head Cap", "No Hardhat", "Bangle detected", "Dress code violation", "No Uniform"]),
        Alert.timestamp >= today,
    ).count()

    oven_count = db.query(Alert).filter(
        Alert.detected_issue.in_(["Oven unattended", "Gas waste", "Fire unattended"]),
        Alert.timestamp >= today,
    ).count()

    window_count = db.query(Alert).filter(
        Alert.detected_issue.in_(["Window throwing detected", "Window theft detected", "Items thrown from window", "Stealing through window"]),
        Alert.timestamp >= today,
    ).count()

    eating_count = db.query(Alert).filter(
        Alert.detected_issue.in_(["Eating from store", "Eating detected", "Chewing detected"]),
        Alert.timestamp >= today,
    ).count()

    dispatch_count = db.query(Alert).filter(
        Alert.detected_issue.in_(["Dispatch without vehicle", "Finished goods not loaded", "FG dispatch alert"]),
        Alert.timestamp >= today,
    ).count()

    cylinder_count = db.query(Alert).filter(
        Alert.detected_issue.in_(["Cylinder count change", "Cylinder usage alert", "Cylinder low count"]),
        Alert.timestamp >= today,
    ).count()

    # Per-floor alert counts today
    floor_counts = {}
    for floor_name in ["ground", "first", "second", "shop"]:
        floor_counts[floor_name] = db.query(Alert).filter(
            Alert.floor == floor_name,
            Alert.timestamp >= today,
        ).count()

    return {
        "cash_events_today":       cash_count,
        "stock_events_today":      stock_count,
        "lift_events_today":       lift_count,
        "packing_events_today":    packing_count,
        "idle_events_today":       idle_count,
        "dress_code_events_today": dress_code_count,
        "oven_events_today":       oven_count,
        "window_events_today":     window_count,
        "eating_events_today":     eating_count,
        "dispatch_events_today":   dispatch_count,
        "cylinder_events_today":   cylinder_count,
        "total_today":             cash_count + stock_count + lift_count + packing_count + idle_count + dress_code_count + oven_count + window_count + eating_count + dispatch_count + cylinder_count,
        "floor_counts":            floor_counts,
    }

