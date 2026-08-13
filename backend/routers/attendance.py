"""
routers/attendance.py — Employee attendance REST API
=====================================================
Endpoints
---------
  GET    /attendance/           List records (date, employee_id filters)
  GET    /attendance/stats      Today's summary (present / absent / late)
  POST   /attendance/clock-in   Manual or face-triggered clock-in
  POST   /attendance/clock-out/{record_id}  Close an open session
  GET    /attendance/export     CSV download for payroll
"""

import csv
import io
import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, AttendanceRecord, Employee
from routers.auth import get_current_user

router = APIRouter(prefix="/attendance", tags=["attendance"])


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class ClockInRequest(BaseModel):
    employee_id: Optional[int] = None
    camera_id:   Optional[int] = None
    method:      str            = "manual"   # manual | face | qr
    notes:       Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/employees")
def list_employees(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Return active employee roster using Employee IDs required by attendance."""
    rows = (
        db.query(Employee)
        .filter(Employee.active.is_(True))
        .order_by(Employee.name, Employee.id)
        .all()
    )
    return [
        {
            "id": employee.id,
            "name": employee.name,
            "role": employee.role,
            "department": employee.department,
        }
        for employee in rows
    ]


@router.get("/stats")
def get_stats(
    db:           Session = Depends(get_db),
    current_user          = Depends(get_current_user),
):
    """Today's attendance summary — present / absent / late counts."""
    from services.attendance_service import get_today_summary
    return get_today_summary(db)


@router.get("/")
def list_records(
    date:        Optional[str] = Query(None, description="YYYY-MM-DD"),
    employee_id: Optional[int] = Query(None),
    limit:       int           = Query(200, le=1000),
    db:          Session       = Depends(get_db),
    current_user               = Depends(get_current_user),
):
    """List attendance records with optional date and employee filters."""
    from services.attendance_service import get_records
    parsed_date = None
    if date:
        try:
            parsed_date = datetime.date.fromisoformat(date)
        except ValueError:
            raise HTTPException(400, "Invalid date format — use YYYY-MM-DD")
    return get_records(db, date=parsed_date, employee_id=employee_id, limit=limit)


@router.post("/clock-in")
def clock_in(
    body:         ClockInRequest,
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """
    Clock a person in.
    - employee_id=None → unrecognized / visitor entry
    - method='manual'  → admin desk entry
    - method='face'    → face-recognition auto-entry
    """
    from services.attendance_service import clock_in as svc_clock_in
    return svc_clock_in(
        db          = db,
        employee_id = body.employee_id,
        camera_id   = body.camera_id,
        method      = body.method,
        notes       = body.notes,
        user_id     = current_user.id,
    )


@router.post("/clock-out/{record_id}")
def clock_out(
    record_id:    int,
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """Close an open attendance session (set clock_out = now)."""
    from services.attendance_service import clock_out as svc_clock_out
    result = svc_clock_out(db, record_id)
    if result is None:
        raise HTTPException(404, f"Attendance record #{record_id} not found")
    return result


@router.get("/export")
def export_csv(
    date:        Optional[str] = Query(None, description="YYYY-MM-DD"),
    employee_id: Optional[int] = Query(None),
    db:          Session       = Depends(get_db),
    current_user               = Depends(get_current_user),
):
    """Download attendance records as CSV for payroll."""
    from services.attendance_service import get_records
    parsed_date = None
    if date:
        try:
            parsed_date = datetime.date.fromisoformat(date)
        except ValueError:
            raise HTTPException(400, "Invalid date format — use YYYY-MM-DD")

    records = get_records(db, date=parsed_date, employee_id=employee_id, limit=5000)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "id", "employee_id", "employee_name", "camera_id", "method",
        "clock_in", "clock_out", "duration_seconds", "notes",
    ])
    writer.writeheader()
    for r in records:
        writer.writerow({
            "id":               r["id"],
            "employee_id":      r["employee_id"] or "",
            "employee_name":    r["employee_name"] or "Unknown",
            "camera_id":        r["camera_id"] or "",
            "method":           r["method"],
            "clock_in":         r["clock_in"] or "",
            "clock_out":        r["clock_out"] or "",
            "duration_seconds": r["duration_seconds"] or "",
            "notes":            r["notes"] or "",
        })

    output.seek(0)
    filename = f"attendance_{date or 'all'}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
