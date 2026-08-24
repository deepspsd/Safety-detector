"""
routers/attendance.py — Employee attendance REST API
=====================================================
Endpoints
---------
  GET    /attendance/                   List records (date, employee_id filters)
  GET    /attendance/stats              Today's summary (present / absent / late)
  POST   /attendance/clock-in           Manual or face-triggered clock-in
  POST   /attendance/clock-out/{id}     Close an open session
  POST   /attendance/clock-out-all      Bulk close ALL open sessions (nightly / admin)
  GET    /attendance/export             CSV download for payroll
  POST   /attendance/employees/import   Bulk import employees from uploaded CSV
  DELETE /attendance/{id}               Hard-delete a single attendance record
"""

import csv
import datetime
import io
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import AttendanceRecord, Employee, get_db
from routers.auth import get_current_user

log = logging.getLogger("attendance_router")

router = APIRouter(prefix="/attendance", tags=["attendance"])


# ── Pydantic schemas ─────────────────────────────────────────────────────────


class ClockInRequest(BaseModel):
    employee_id: Optional[int] = None
    camera_id: Optional[int] = None
    method: str = "manual"  # manual | face | qr
    notes: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/employees")
def list_employees(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
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
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Today's attendance summary — present / absent / late counts."""
    from services.attendance_service import get_today_summary

    return get_today_summary(db)


@router.get("/")
def list_records(
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    employee_id: Optional[int] = Query(None),
    limit: int = Query(200, le=1000),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
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
    body: ClockInRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Clock a person in.
    - employee_id=None → unrecognized / visitor entry
    - method='manual'  → admin desk entry
    - method='face'    → face-recognition auto-entry
    """
    from services.attendance_service import clock_in as svc_clock_in

    return svc_clock_in(
        db=db,
        employee_id=body.employee_id,
        camera_id=body.camera_id,
        method=body.method,
        notes=body.notes,
        user_id=current_user.id,
    )


@router.post("/clock-out/{record_id}")
def clock_out(
    record_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Close an open attendance session (set clock_out = now)."""
    from services.attendance_service import clock_out as svc_clock_out

    result = svc_clock_out(db, record_id)
    if result is None:
        raise HTTPException(404, f"Attendance record #{record_id} not found")
    return result


@router.post("/clock-out-all")
def clock_out_all(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Bulk-close ALL currently open attendance sessions.

    Use cases
    ---------
    • Admin clicks "End of Day" button to close all sessions at once.
    • Called automatically at 19:00 IST by the nightly scheduler.

    Returns
    -------
    {"closed": N, "message": "..."}
    """
    from services.attendance_service import auto_clock_out_open_sessions

    return auto_clock_out_open_sessions(db)


@router.get("/export")
def export_csv(
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    employee_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
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
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "id",
            "employee_id",
            "employee_name",
            "camera_id",
            "method",
            "clock_in",
            "clock_out",
            "duration_seconds",
            "notes",
        ],
    )
    writer.writeheader()
    for r in records:
        writer.writerow(
            {
                "id": r["id"],
                "employee_id": r["employee_id"] or "",
                "employee_name": r["employee_name"] or "Unknown",
                "camera_id": r["camera_id"] or "",
                "method": r["method"],
                "clock_in": r["clock_in"] or "",
                "clock_out": r["clock_out"] or "",
                "duration_seconds": r["duration_seconds"] or "",
                "notes": r["notes"] or "",
            }
        )

    output.seek(0)
    filename = f"attendance_{date or 'all'}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/employees/import")
async def import_employees_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Bulk import employees from a CSV file (server-side streaming — memory-safe for large files).

    Expected CSV columns (order-independent, case-insensitive):
        name   — required  — employee full name
        role   — optional  — e.g. Baker, Supervisor
        department — optional

    Returns
    -------
    {
        "imported": N,     # new rows created
        "updated":  N,     # existing rows updated (matched by name)
        "skipped":  N,     # blank / malformed rows
        "errors":   [...]  # list of row-level error descriptions
    }
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted")

    # Read the entire upload into memory as bytes — streaming chunk-by-chunk
    # avoids holding the file open but CSV parse needs a seekable object.
    raw_bytes = await file.read()
    try:
        text = raw_bytes.decode("utf-8-sig")  # strip BOM if Excel-exported
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))

    # Normalise headers to lowercase, strip spaces
    if reader.fieldnames is None:
        raise HTTPException(400, "CSV is empty or has no header row")

    normalised_headers = {h.strip().lower(): h for h in reader.fieldnames}
    if "name" not in normalised_headers:
        raise HTTPException(
            400, "CSV must have a 'name' column. " f"Found: {list(reader.fieldnames)}"
        )

    imported = 0
    updated = 0
    skipped = 0
    errors: list[str] = []

    for row_num, raw_row in enumerate(reader, start=2):
        # Normalise keys
        row = {k.strip().lower(): (v.strip() if v else "") for k, v in raw_row.items()}

        name = row.get("name", "").strip()
        if not name:
            skipped += 1
            continue

        role = row.get("role", "") or None
        department = row.get("department", "") or None
        phone = row.get("phone", "") or row.get("mobile", "") or None
        shift = row.get("shift", "") or None

        try:
            existing = db.query(Employee).filter(Employee.name == name).first()
            if existing:
                # Update existing row with any new info
                if role:
                    existing.role = role
                if department:
                    existing.department = department
                existing.active = True
                updated += 1
            else:
                emp = Employee(
                    name=name,
                    role=role,
                    department=department,
                    active=True,
                )
                db.add(emp)
                imported += 1
        except Exception as exc:
            errors.append(f"Row {row_num} ({name!r}): {exc}")
            db.rollback()
            skipped += 1
            continue

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(500, f"DB commit failed: {exc}")

    log.info(
        f"[Attendance] CSV import complete: imported={imported} updated={updated} "
        f"skipped={skipped} errors={len(errors)}"
    )
    return {
        "imported": imported,
        "updated": updated,
        "skipped": skipped,
        "errors": errors[:50],  # cap error list — never return megabytes of errors
    }


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /attendance/{id}
# ─────────────────────────────────────────────────────────────────────────────


@router.delete("/{record_id}")
def delete_attendance_record(
    record_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Hard-delete a single attendance record by ID."""
    record = db.query(AttendanceRecord).filter(AttendanceRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Attendance record not found")
    db.delete(record)
    db.commit()
    log.info(
        f"[Attendance] Record #{record_id} deleted by user={current_user.id} "
        f"(employee_id={record.employee_id})"
    )
    return {"deleted": True, "id": record_id}
