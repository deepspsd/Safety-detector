"""
services/attendance_service.py — Employee attendance tracking
=============================================================
Provides clock-in / clock-out logic, today's summary stats,
and hooks for the face-recognition pipeline to auto-fire clock-in.

All DB access uses short-lived per-call sessions (same pattern
as idle_service.py) to avoid session leaks in background threads.

Timezone policy
---------------
All datetime objects stored in DB are naive UTC (datetime.utcnow()).
All datetimes returned via the API include a trailing 'Z' so
JavaScript correctly parses them as UTC and converts to the user's
local timezone (IST = UTC+05:30) for display.

Late-count comparison is done in IST so 08:30 IST is correctly
recognised as on-time for an 08:00 IST shift, not the wrong UTC hour.
"""

import datetime
import logging
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger("attendance_service")

# IST reference for all shift-time comparisons
_IST = ZoneInfo("Asia/Kolkata")

# Auto clock-out time (IST): configurable via SETTING_DEFAULTS
_AUTO_CLOCKOUT_HOUR_IST = 19  # 7 PM IST — change via rule_engine setting
_AUTO_CLOCKOUT_MIN_IST = 0


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────


def clock_in(
    db,
    employee_id: Optional[int] = None,
    camera_id: Optional[int] = None,
    method: str = "manual",
    notes: Optional[str] = None,
    user_id: Optional[int] = None,
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

    now_utc = datetime.datetime.utcnow()

    # Prevent duplicate open sessions for the same employee today
    if employee_id is not None:
        today_start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        existing = (
            db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.employee_id == employee_id,
                AttendanceRecord.clock_in >= today_start_utc,
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
        employee_id=employee_id,
        camera_id=camera_id,
        method=method,
        notes=notes,
        user_id=user_id,
        clock_in=now_utc,
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

    now_utc = datetime.datetime.utcnow()
    record.clock_out = now_utc
    duration = (now_utc - record.clock_in).total_seconds()
    record.duration_seconds = duration
    db.commit()
    db.refresh(record)
    log.info(
        f"[Attendance] Clock-OUT recorded — employee={record.employee_id} "
        f"duration={duration:.0f}s record_id={record.id}"
    )
    return _record_to_dict(record)


def auto_clock_out_open_sessions(db) -> dict:
    """
    Close ALL open attendance sessions (clock_out=NULL).

    Called automatically by the nightly scheduler at 19:00 IST and
    also available as POST /attendance/clock-out-all for manual bulk close.

    Returns
    -------
    {"closed": N, "message": "..."}
    """
    from database import AttendanceRecord

    now_utc = datetime.datetime.utcnow()
    open_records = (
        db.query(AttendanceRecord).filter(AttendanceRecord.clock_out.is_(None)).all()
    )
    closed = 0
    for r in open_records:
        r.clock_out = now_utc
        try:
            r.duration_seconds = (now_utc - r.clock_in).total_seconds()
        except Exception:
            r.duration_seconds = 0.0
        closed += 1

    if closed:
        db.commit()

    msg = f"Auto clock-out complete: {closed} open session(s) closed."
    log.info(f"[Attendance] {msg}")

    # Push notification to owner via FCM
    if closed:
        try:
            from services.notification_service import send_push_alert

            send_push_alert(
                message=(
                    f"🕖 Nightly auto clock-out ran at "
                    f"{datetime.datetime.now(_IST).strftime('%H:%M IST')}. "
                    f"{closed} open session(s) were closed automatically."
                ),
                severity="low",
                detected_issue="Auto clock-out",
            )
        except Exception as exc:
            log.warning(f"[Attendance] auto clock-out push notify failed: {exc}")

    return {"closed": closed, "message": msg}


def get_today_summary(db) -> dict:
    """
    Return today's attendance stats for the dashboard.

    Returns
    -------
    {
        present_count  : int   # employees who clocked in today
        absent_count   : int   # registered employees with no clock-in today
        late_count     : int   # employees clocked in after their floor shift start (IST)
        open_sessions  : int   # currently on-site (clock_out=NULL)
        records        : list  # all today's records
    }

    Timezone note
    -------------
    "Today" is computed in IST midnight → converted to UTC for the DB query,
    so records at 23:30 UTC (= 05:00 IST next day) are NOT counted as today.
    Late check uses IST hours to match the client's shift times.
    """
    from database import AttendanceRecord, Employee

    # IST midnight → UTC (subtract 5h30m)
    now_ist = datetime.datetime.now(_IST)
    today_midnight_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    # Convert to UTC (naive) for DB query
    today_start_utc = today_midnight_ist.astimezone(datetime.timezone.utc).replace(
        tzinfo=None
    )

    # Default late threshold: 09:00 IST. Pull floor-specific setting if we can.
    # Use the ground-floor setting as the global fallback (most common shift).
    late_threshold_hour_ist = 9
    late_threshold_min_ist = 0
    try:
        from services.rule_engine import get_setting

        val = get_setting("shift_start_ground", db)
        if val and ":" in val:
            h, m = map(int, val.split(":"))
            late_threshold_hour_ist = h
            late_threshold_min_ist = m
    except Exception:
        pass

    records = (
        db.query(AttendanceRecord)
        .filter(AttendanceRecord.clock_in >= today_start_utc)
        .order_by(AttendanceRecord.clock_in.desc())
        .all()
    )

    total_employees = (
        db.query(Employee).filter(Employee.active == True).count()
    )  # noqa: E712
    clocked_in_employee_ids = {
        r.employee_id for r in records if r.employee_id is not None
    }
    present_count = len(clocked_in_employee_ids)
    absent_count = max(0, total_employees - present_count)
    open_sessions = sum(1 for r in records if r.clock_out is None)

    # Late count — compare clock_in converted to IST
    late_count = 0
    for r in records:
        if r.employee_id is None:
            continue
        # r.clock_in is naive UTC → make timezone-aware → convert to IST
        clock_in_utc = r.clock_in.replace(tzinfo=datetime.timezone.utc)
        clock_in_ist = clock_in_utc.astimezone(_IST)
        shift_limit = clock_in_ist.replace(
            hour=late_threshold_hour_ist,
            minute=late_threshold_min_ist,
            second=0,
            microsecond=0,
        )
        if clock_in_ist > shift_limit:
            late_count += 1

    return {
        "present_count": present_count,
        "absent_count": absent_count,
        "late_count": late_count,
        "open_sessions": open_sessions,
        "total_employees": total_employees,
        "records": [_record_to_dict(r) for r in records],
    }


def get_records(
    db,
    date: Optional[datetime.date] = None,
    employee_id: Optional[int] = None,
    limit: int = 200,
) -> List[dict]:
    """List attendance records with optional filters."""
    from database import AttendanceRecord

    q = db.query(AttendanceRecord)
    if date:
        # Interpret the requested date as IST calendar day → UTC range for DB
        day_start_ist = datetime.datetime(
            date.year, date.month, date.day, 0, 0, 0, tzinfo=_IST
        )
        day_end_ist = datetime.datetime(
            date.year, date.month, date.day, 23, 59, 59, tzinfo=_IST
        )
        day_start_utc = day_start_ist.astimezone(datetime.timezone.utc).replace(
            tzinfo=None
        )
        day_end_utc = day_end_ist.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        q = q.filter(
            AttendanceRecord.clock_in >= day_start_utc,
            AttendanceRecord.clock_in <= day_end_utc,
        )
    if employee_id:
        q = q.filter(AttendanceRecord.employee_id == employee_id)

    records = q.order_by(AttendanceRecord.clock_in.desc()).limit(limit).all()
    return [_record_to_dict(r) for r in records]


# ─────────────────────────────────────────────────────────────────────────────
# Face-recognition hook (called from camera_manager detection loop)
# ─────────────────────────────────────────────────────────────────────────────


def handle_face_match(
    camera_id: int,
    employee_id: int,
    confidence: float,
) -> None:
    """
    Auto-clock-in when face-recognition positively identifies an employee.
    Gate: confidence >= 0.40 (i.e. distance <= 0.60), which matches the
    RECOGNITION_TOLERANCE=0.52 setting in config.py.
    handle_face_match deduplicates within the same calendar day via clock_in().
    """
    if confidence < 0.40:
        return
    log.info(
        f"[Attendance] Face match → employee={employee_id} conf={confidence:.2%} cam={camera_id}"
    )

    from database import SessionLocal

    db = None
    try:
        db = SessionLocal()
        clock_in(
            db=db,
            employee_id=employee_id,
            camera_id=camera_id,
            method="face",
        )
    except Exception as exc:
        log.error(f"[Attendance] handle_face_match failed: {exc}")
        if db:
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        if db:
            try:
                db.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helper
# ─────────────────────────────────────────────────────────────────────────────


def _utc_iso(dt: Optional[datetime.datetime]) -> Optional[str]:
    """
    Serialise a naive-UTC datetime to ISO 8601 with trailing 'Z'.
    JavaScript new Date("2024-08-21T03:30:00Z") correctly converts to IST.
    Without 'Z' JS treats the string as local time, causing a 5h30m display error.
    """
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


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
        "id": r.id,
        "employee_id": r.employee_id,
        "employee_name": employee_name,
        "camera_id": r.camera_id,
        "method": r.method,
        "notes": r.notes,
        "clock_in": _utc_iso(r.clock_in),
        "clock_out": _utc_iso(r.clock_out),
        "duration_seconds": duration,
        "is_open": r.clock_out is None,
    }
