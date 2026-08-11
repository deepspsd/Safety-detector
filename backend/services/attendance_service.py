"""
services/attendance_service.py — Employee attendance tracking
=============================================================
Provides clock-in / clock-out logic, today's summary stats,
and hooks for the face-recognition pipeline to auto-fire clock-in.

All DB access uses short-lived per-call sessions (same pattern
as idle_service.py) to avoid session leaks in background threads.
"""

import datetime
import logging
from typing import Optional, List, Dict

log = logging.getLogger("attendance_service")


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def clock_in(
    db,
    employee_id: Optional[int] = None,
    camera_id:   Optional[int] = None,
    method:      str           = "manual",
    notes:       Optional[str] = None,
    user_id:     Optional[int] = None,
) -> dict:
    """
    Create a new open attendance record (clock_out=NULL).

    Parameters
    ----------
    db          : SQLAlchemy Session
    employee_id : DB employees.id — NULL = unrecognised face
    camera_id   : camera that fired the detection (NULL = manual entry)
    method      : 'face' | 'manual' | 'qr'
    notes       : optional admin note
    user_id     : the logged-in user performing a manual entry (NULL = system)

    Returns
    -------
    dict representation of the created AttendanceRecord
    """
    from database import AttendanceRecord

    # Prevent duplicate open sessions for the same employee today
    if employee_id is not None:
        today_start = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        existing = (
            db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.employee_id == employee_id,
                AttendanceRecord.clock_in    >= today_start,
                AttendanceRecord.clock_out   == None,         # noqa: E711
            )
            .first()
        )
        if existing:
            log.info(
                f"[Attendance] Employee {employee_id} already clocked-in "
                f"(record #{existing.id}) — skipping duplicate"
            )
            return _record_to_dict(existing)

    record = AttendanceRecord(
        employee_id = employee_id,
        camera_id   = camera_id,
        method      = method,
        notes       = notes,
        user_id     = user_id,
        clock_in    = datetime.datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    log.info(
        f"[Attendance] Clock-IN recorded — employee={employee_id} "
        f"method={method} record_id={record.id}"
    )
    return _record_to_dict(record)


def clock_out(db, record_id: int) -> Optional[dict]:
    """
    Close an open attendance record by setting clock_out to now.
    Returns None if the record is not found or already closed.
    """
    from database import AttendanceRecord

    record = db.query(AttendanceRecord).filter(AttendanceRecord.id == record_id).first()
    if not record:
        log.warning(f"[Attendance] clock_out: record #{record_id} not found")
        return None
    if record.clock_out is not None:
        log.info(f"[Attendance] clock_out: record #{record_id} already closed")
        return _record_to_dict(record)

    record.clock_out = datetime.datetime.utcnow()
    duration = (record.clock_out - record.clock_in).total_seconds()
    record.duration_seconds = duration
    db.commit()
    db.refresh(record)
    log.info(
        f"[Attendance] Clock-OUT recorded — employee={record.employee_id} "
        f"duration={duration:.0f}s record_id={record.id}"
    )
    return _record_to_dict(record)


def get_today_summary(db) -> dict:
    """
    Return today's attendance stats for the dashboard.

    Returns
    -------
    {
        present_count  : int   # employees who clocked in today
        absent_count   : int   # registered employees with no clock-in today
        late_count     : int   # employees clocked in after their shift start (09:00 default)
        open_sessions  : int   # currently on-site (clock_out=NULL)
        records        : list  # all today's records
    }
    """
    from database import AttendanceRecord, Employee

    today_start = datetime.datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    late_threshold_hour = 9   # 09:00 UTC — adjust per shift

    records = (
        db.query(AttendanceRecord)
        .filter(AttendanceRecord.clock_in >= today_start)
        .order_by(AttendanceRecord.clock_in.desc())
        .all()
    )

    total_employees = db.query(Employee).filter(Employee.active == True).count()  # noqa: E712
    clocked_in_employee_ids = {
        r.employee_id for r in records if r.employee_id is not None
    }
    present_count  = len(clocked_in_employee_ids)
    absent_count   = max(0, total_employees - present_count)
    open_sessions  = sum(1 for r in records if r.clock_out is None)
    late_count     = sum(
        1 for r in records
        if r.clock_in.hour >= late_threshold_hour
        and r.employee_id is not None
    )

    return {
        "present_count":  present_count,
        "absent_count":   absent_count,
        "late_count":     late_count,
        "open_sessions":  open_sessions,
        "total_employees": total_employees,
        "records":        [_record_to_dict(r) for r in records],
    }


def get_records(
    db,
    date:        Optional[datetime.date] = None,
    employee_id: Optional[int]           = None,
    limit:       int                     = 200,
) -> List[dict]:
    """List attendance records with optional filters."""
    from database import AttendanceRecord

    q = db.query(AttendanceRecord)
    if date:
        day_start = datetime.datetime.combine(date, datetime.time.min)
        day_end   = datetime.datetime.combine(date, datetime.time.max)
        q = q.filter(
            AttendanceRecord.clock_in >= day_start,
            AttendanceRecord.clock_in <= day_end,
        )
    if employee_id:
        q = q.filter(AttendanceRecord.employee_id == employee_id)

    records = q.order_by(AttendanceRecord.clock_in.desc()).limit(limit).all()
    return [_record_to_dict(r) for r in records]


# ─────────────────────────────────────────────────────────────────────────────
# Face-recognition hook (called from camera_manager detection loop)
# ─────────────────────────────────────────────────────────────────────────────

def handle_face_match(
    camera_id:   int,
    employee_id: int,
    confidence:  float,
) -> None:
    """
    Auto-clock-in when face-recognition positively identifies an employee.
    Only fires if confidence >= 0.72 to avoid low-quality false matches.
    Uses its own short-lived DB session (background thread safe).
    """
    if confidence < 0.72:
        return

    from database import SessionLocal
    db = None
    try:
        db = SessionLocal()
        clock_in(
            db          = db,
            employee_id = employee_id,
            camera_id   = camera_id,
            method      = "face",
        )
    except Exception as exc:
        log.error(f"[Attendance] handle_face_match failed: {exc}")
        if db:
            try: db.rollback()
            except Exception: pass
    finally:
        if db:
            try: db.close()
            except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _record_to_dict(r) -> dict:
    duration = None
    if r.clock_out and r.clock_in:
        duration = round((r.clock_out - r.clock_in).total_seconds())
    elif hasattr(r, "duration_seconds") and r.duration_seconds is not None:
        duration = round(r.duration_seconds)

    # Fetch employee name if linked
    employee_name = None
    try:
        if r.employee:
            employee_name = r.employee.name
    except Exception:
        pass

    return {
        "id":               r.id,
        "employee_id":      r.employee_id,
        "employee_name":    employee_name,
        "camera_id":        r.camera_id,
        "method":           r.method,
        "notes":            r.notes,
        "clock_in":         r.clock_in.isoformat() if r.clock_in else None,
        "clock_out":        r.clock_out.isoformat() if r.clock_out else None,
        "duration_seconds": duration,
        "is_open":          r.clock_out is None,
    }
