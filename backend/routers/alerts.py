"""
Alerts router — v2.0
Returns snapshot_url pointing to the static file endpoint,
plus snapshot_b64 in the single-alert detail view.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, date
from database import get_db, Alert, User
from routers.auth import get_current_user

router = APIRouter(prefix="/alerts", tags=["alerts"])

# Base URL for serving alert snapshots
SNAPSHOT_BASE = "/uploads/snapshots"


def alert_to_dict(a: Alert, include_snapshot: bool = True) -> dict:
    # Build full snapshot URL from path (e.g. "snapshots/20250101_120000_u1.jpg")
    snapshot_url = None
    if a.snapshot_path:
        snapshot_url = f"/uploads/{a.snapshot_path}"

    return {
        "id":              a.id,
        "user_id":         a.user_id,
        "message":         a.message,
        "role":            a.role,
        "severity":        a.severity,
        "detected_issue":  a.detected_issue,
        "confidence":      a.confidence,
        "snapshot_url":    snapshot_url,                                  # persistent file URL
        "snapshot_b64":    a.snapshot_b64 if include_snapshot else None,  # base64 for modal
        "has_snapshot":    snapshot_url is not None or bool(a.snapshot_b64),
        "timestamp":       a.timestamp.isoformat(),
    }


@router.get("/")
def get_alerts(
    role:       Optional[str]  = Query(None),
    severity:   Optional[str]  = Query(None),
    date_from:  Optional[date] = Query(None),
    date_to:    Optional[date] = Query(None),
    page:       int            = Query(1, ge=1),
    limit:      int            = Query(20, ge=1, le=100),
    db:         Session        = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    query = db.query(Alert).filter(Alert.user_id == current_user.id)
    if role:
        query = query.filter(Alert.role == role)
    if severity:
        query = query.filter(Alert.severity == severity)
    if date_from:
        query = query.filter(Alert.timestamp >= datetime.combine(date_from, datetime.min.time()))
    if date_to:
        query = query.filter(Alert.timestamp <= datetime.combine(date_to, datetime.max.time()))
    total  = query.count()
    alerts = query.order_by(Alert.timestamp.desc()).offset((page - 1) * limit).limit(limit).all()
    return {
        "total":  total,
        "page":   page,
        "limit":  limit,
        "alerts": [alert_to_dict(a, include_snapshot=False) for a in alerts],
    }


@router.get("/stats")
def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    total = db.query(Alert).filter(Alert.user_id == current_user.id).count()
    today = date.today()
    today_count = db.query(Alert).filter(
        Alert.user_id == current_user.id,
        Alert.timestamp >= datetime.combine(today, datetime.min.time()),
    ).count()
    critical = db.query(Alert).filter(
        Alert.user_id == current_user.id,
        Alert.severity == "critical",
    ).count()
    recent = (
        db.query(Alert)
        .filter(Alert.user_id == current_user.id)
        .order_by(Alert.timestamp.desc())
        .limit(50)
        .all()
    )
    compliance = max(0, round(100 - (total / max(total + 100, 1)) * 100, 1))
    return {
        "total_alerts":       total,
        "today_alerts":       today_count,
        "critical_alerts":    critical,
        "compliance_percentage": compliance,
        "recent_alerts":      [alert_to_dict(a, include_snapshot=False) for a in recent],
    }


@router.get("/{alert_id}")
def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    alert = db.query(Alert).filter(
        Alert.id == alert_id,
        Alert.user_id == current_user.id,
    ).first()
    if not alert:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert_to_dict(alert, include_snapshot=True)


@router.delete("/{alert_id}")
def delete_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    alert = db.query(Alert).filter(
        Alert.id == alert_id,
        Alert.user_id == current_user.id,
    ).first()
    if not alert:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Alert not found")
    # Optionally delete the snapshot file from disk
    if alert.snapshot_path:
        import os
        try:
            os.remove(os.path.join("uploads", alert.snapshot_path))
        except Exception:
            pass
    db.delete(alert)
    db.commit()
    return {"message": "Alert deleted"}


@router.delete("/")
def clear_all_alerts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    alerts = db.query(Alert).filter(Alert.user_id == current_user.id).all()
    import os
    for a in alerts:
        if a.snapshot_path:
            try:
                os.remove(os.path.join("uploads", a.snapshot_path))
            except Exception:
                pass
    db.query(Alert).filter(Alert.user_id == current_user.id).delete()
    db.commit()
    return {"message": "All alerts cleared"}
