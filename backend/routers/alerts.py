"""
routers/alerts.py — v3.0
=========================
REST API for factory compliance alerts.

Endpoints
---------
  GET    /alerts/                  Main alert feed (default: confirmed only)
  GET    /alerts/stats             Aggregate stats
  GET    /alerts/pending           Pending-review queue (admin review UI)
  PATCH  /alerts/{id}/confirm      Promote pending → confirmed + fire Telegram
  PATCH  /alerts/{id}/dismiss      Mark as dismissed (false positive)
  GET    /alerts/{id}              Single alert detail (includes snapshot_b64)
  DELETE /alerts/{id}              Hard-delete one alert
  DELETE /alerts/                  Hard-delete all alerts for current user
  POST   /alerts/test-push          Send a test push notification (ntfy + Telegram) — admin debug
  POST   /alerts/test-telegram      Alias for test-push (backward compat)
"""

import os
from datetime import datetime, date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db, Alert, Camera, User
from routers.auth import get_current_user
from services.alert_service import confirm_alert, dismiss_alert

router = APIRouter(prefix="/alerts", tags=["alerts"])

SNAPSHOT_BASE = "/uploads/snapshots"


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def alert_to_dict(a: Alert, include_snapshot: bool = True) -> dict:
    snapshot_url = f"/uploads/{a.snapshot_path}" if a.snapshot_path else None
    return {
        "id":              a.id,
        "user_id":         a.user_id,
        "message":         a.message,
        "role":            a.role,
        "severity":        a.severity,
        "detected_issue":  a.detected_issue,
        "confidence":      a.confidence,
        "status":          getattr(a, "status", "confirmed"),   # safe for old rows
        "snapshot_url":    snapshot_url,
        "snapshot_b64":    a.snapshot_b64 if include_snapshot else None,
        "has_snapshot":    snapshot_url is not None or bool(a.snapshot_b64),
        "timestamp":       a.timestamp.strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        "camera_id":       getattr(a, "camera_id", None),
        "floor":           getattr(a, "floor", None),
        "employee_id":     getattr(a, "employee_id", None),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /alerts/
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/")
def get_alerts(
    role:       Optional[str]  = Query(None),
    severity:   Optional[str]  = Query(None),
    status:     Optional[str]  = Query("confirmed"),   # default: main feed = confirmed only
    floor:      Optional[str]  = Query(None),
    camera_id:  Optional[int]  = Query(None),
    date_from:  Optional[date] = Query(None),
    date_to:    Optional[date] = Query(None),
    page:       int            = Query(1, ge=1),
    limit:      int            = Query(20, ge=1, le=100),
    db:         Session        = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    """
    Main alert feed.

    By default returns only "confirmed" alerts (status=confirmed).
    Pass ?status=pending_review for the admin review queue.
    Pass ?status=all to get every alert regardless of status.
    """
    query = db.query(Alert).filter(Alert.user_id == current_user.id)

    # Status filter
    if status and status != "all":
        query = query.filter(Alert.status == status)

    if role:
        query = query.filter(Alert.role == role)
    if severity:
        query = query.filter(Alert.severity == severity)
    if floor:
        query = query.filter(Alert.floor == floor)
    if camera_id:
        query = query.filter(Alert.camera_id == camera_id)
    if date_from:
        query = query.filter(Alert.timestamp >= datetime.combine(date_from, datetime.min.time()))
    if date_to:
        query = query.filter(Alert.timestamp <= datetime.combine(date_to, datetime.max.time()))

    total  = query.count()
    alerts = (
        query
        .order_by(Alert.timestamp.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )
    return {
        "total":  total,
        "page":   page,
        "limit":  limit,
        "status_filter": status,
        "alerts": [alert_to_dict(a, include_snapshot=False) for a in alerts],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /alerts/pending  — review queue shortcut
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/pending")
def get_pending_alerts(
    page:  int     = Query(1, ge=1),
    limit: int     = Query(20, ge=1, le=100),
    db:    Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Admin review queue — all pending_review alerts for this user.
    Rendered separately in the Block 9 review UI so low-confidence alerts
    don't pollute the main confirmed feed.
    """
    query = (
        db.query(Alert)
        .filter(
            Alert.user_id == current_user.id,
            Alert.status  == "pending_review",
        )
        .order_by(Alert.timestamp.desc())
    )
    total  = query.count()
    alerts = query.offset((page - 1) * limit).limit(limit).all()
    return {
        "total":  total,
        "page":   page,
        "limit":  limit,
        "alerts": [alert_to_dict(a, include_snapshot=False) for a in alerts],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /alerts/stats
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    today  = date.today()
    base_q = db.query(Alert).filter(Alert.user_id == current_user.id)

    total          = base_q.count()
    confirmed      = base_q.filter(Alert.status == "confirmed").count()
    pending_review = base_q.filter(Alert.status == "pending_review").count()
    dismissed      = base_q.filter(Alert.status == "dismissed").count()

    today_count = base_q.filter(
        Alert.timestamp >= datetime.combine(today, datetime.min.time()),
        Alert.status == "confirmed",
    ).count()

    critical = base_q.filter(
        Alert.severity == "critical",
        Alert.status   == "confirmed",
    ).count()

    recent = (
        base_q
        .filter(Alert.status == "confirmed")
        .order_by(Alert.timestamp.desc())
        .limit(50)
        .all()
    )

    compliance = max(0, round(100 - (confirmed / max(confirmed + 100, 1)) * 100, 1))

    return {
        "total_alerts":          total,
        "confirmed_alerts":      confirmed,
        "pending_review_alerts": pending_review,
        "dismissed_alerts":      dismissed,
        "today_alerts":          today_count,
        "critical_alerts":       critical,
        "compliance_percentage": compliance,
        "recent_alerts":         [alert_to_dict(a, include_snapshot=False) for a in recent],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PATCH /alerts/{id}/confirm
# ─────────────────────────────────────────────────────────────────────────────

@router.patch("/{alert_id}/confirm")
def confirm_alert_endpoint(
    alert_id:     int,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """
    Promote a pending_review alert to confirmed and send the Telegram notification.

    Only the alert owner (current_user) can confirm their own alerts.
    Returns 409 if the alert is not in pending_review status.
    """
    # Ownership check
    alert = db.query(Alert).filter(
        Alert.id      == alert_id,
        Alert.user_id == current_user.id,
    ).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    try:
        updated = confirm_alert(alert_id, db)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "message":  "Alert confirmed and Telegram notification sent",
        "alert_id": updated.id,
        "status":   updated.status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PATCH /alerts/{id}/dismiss
# ─────────────────────────────────────────────────────────────────────────────

@router.patch("/{alert_id}/dismiss")
def dismiss_alert_endpoint(
    alert_id:     int,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """
    Dismiss a pending_review alert (false positive — not sent to Telegram).
    Returns 409 if the alert is already confirmed.
    """
    alert = db.query(Alert).filter(
        Alert.id      == alert_id,
        Alert.user_id == current_user.id,
    ).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    try:
        updated = dismiss_alert(alert_id, db)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "message":  "Alert dismissed",
        "alert_id": updated.id,
        "status":   updated.status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /alerts/test-telegram  — admin debug / connection test
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/test-telegram")
def test_telegram(
    current_user: User    = Depends(get_current_user),
):
    """
    Send a test message to the configured Telegram chat.
    Useful to verify the token and chat_id are correct before going live.
    Does NOT create a DB alert row.
    """
    from services.notification_service import send_telegram_alert, _is_configured
    if not _is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Telegram is not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file. "
                "See config.py for the setup guide."
            ),
        )

    success = send_telegram_alert(
        message        = "✅ Telegram integration is working! This is a test from the Safety Monitor.",
        severity       = "low",
        detected_issue = "Telegram connection test",
    )

    if success:
        return {"message": "Test Telegram message sent successfully"}
    raise HTTPException(
        status_code=502,
        detail="Telegram API call failed — check token and chat_id, and ensure the bot is in the group"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /alerts/{id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{alert_id}")
def get_alert(
    alert_id:     int,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    alert = db.query(Alert).filter(
        Alert.id      == alert_id,
        Alert.user_id == current_user.id,
    ).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert_to_dict(alert, include_snapshot=True)


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /alerts/{id}
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/{alert_id}")
def delete_alert(
    alert_id:     int,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    alert = db.query(Alert).filter(
        Alert.id      == alert_id,
        Alert.user_id == current_user.id,
    ).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    if alert.snapshot_path:
        try:
            os.remove(os.path.join("uploads", alert.snapshot_path))
        except Exception:
            pass

    db.delete(alert)
    db.commit()
    return {"message": "Alert deleted"}


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /alerts/
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/")
def clear_all_alerts(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    alerts = db.query(Alert).filter(Alert.user_id == current_user.id).all()
    for a in alerts:
        if a.snapshot_path:
            try:
                os.remove(os.path.join("uploads", a.snapshot_path))
            except Exception:
                pass
    db.query(Alert).filter(Alert.user_id == current_user.id).delete()
    db.commit()
    return {"message": "All alerts cleared"}


# ─────────────────────────────────────────────────────────────────────────────
# POST /alerts/test-push  (and legacy alias /alerts/test-telegram)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/test-push")
@router.post("/test-telegram")   # legacy alias
def test_push_notification(current_user: User = Depends(get_current_user)):
    """
    Fire a real test push notification and return a diagnostic report.
    Shows which channels are configured, which succeeded/failed, and why.
    Admin/debug use only.
    """
    from config import settings
    from services.notification_service import (
        send_ntfy_alert, send_telegram_alert,
        _is_ntfy_configured, _is_telegram_configured,
    )

    result = {
        "ntfy": {
            "configured": _is_ntfy_configured(),
            "topic":  settings.NTFY_TOPIC  or "(not set)",
            "server": settings.NTFY_SERVER or "(not set)",
            "sent": False,
        },
        "telegram": {
            "configured": _is_telegram_configured(),
            "token_set":  bool(settings.TELEGRAM_BOT_TOKEN),
            "chat_id":    settings.TELEGRAM_CHAT_ID or "(not set)",
            "sent": False,
        },
    }

    # Try ntfy
    if result["ntfy"]["configured"]:
        result["ntfy"]["sent"] = send_ntfy_alert(
            message        = "Test push from OccuSafe — ntfy is working!",
            severity       = "medium",
            detected_issue = "Test notification",
        )
    else:
        result["ntfy"]["reason"] = "NTFY_TOPIC not set in .env"

    # Try Telegram
    if result["telegram"]["configured"]:
        result["telegram"]["sent"] = send_telegram_alert(
            message        = "Test push from OccuSafe — Telegram is working!",
            severity       = "medium",
            detected_issue = "Test notification",
        )
    else:
        result["telegram"]["reason"] = "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env"

    return result
