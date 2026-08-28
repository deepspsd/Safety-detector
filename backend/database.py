"""
database.py — Safety Monitor / Bakery Factory Monitoring
=========================================================
Schema v2.0

Existing tables (unchanged — admin/supervisor login):
  users, user_configs, face_encodings, alerts (extended)

New tables (multi-camera factory monitoring):
  cameras        — RTSP feeds per floor/zone
  employees      — people being monitored (≠ User login accounts)
  zone_configs   — per-camera polygon zones
  idle_sessions  — ByteTrack idle-timer records
  cylinder_logs  — gas cylinder usage events

Alert table is backward-compatible: three new nullable FK columns
(camera_id, floor, employee_id) added via auto-migration in main.py.
"""

from datetime import datetime

from config import settings
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine, event, inspect, text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

_engine_kwargs = {}
if settings.DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {
        "check_same_thread": False,
        "timeout": 30,
    }

engine = create_engine(settings.DATABASE_URL, **_engine_kwargs)


if engine.dialect.name == "sqlite":

    @event.listens_for(engine, "connect")
    def _configure_sqlite(dbapi_connection, _connection_record):
        """Enable safe SQLite concurrency and referential integrity per connection."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ──────────────────────────────────────────────────────────────────────────────
# EXISTING TABLES  (kept intact — no column removed or renamed)
# ──────────────────────────────────────────────────────────────────────────────


class User(Base):
    """Admin / supervisor login account.  NOT the person being monitored."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    email = Column(String(200), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(50), nullable=True)  # admin, supervisor, …
    created_at = Column(DateTime, default=datetime.utcnow)

    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    config = relationship(
        "UserConfig", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    face_encodings = relationship(
        "FaceEncoding", back_populates="user", cascade="all, delete-orphan"
    )


class Alert(Base):
    """
    Safety / compliance alert.

    v1 columns (all preserved, unchanged):
      id, user_id, message, role, severity, detected_issue,
      confidence, snapshot_b64, snapshot_path, timestamp

    v2 additions (nullable for full backward compat):
      camera_id    — which camera fired the alert
      floor        — ground / first / second / shop (denormalised for fast queries)
      employee_id  — identified violator (NULL = unknown / unidentified)

    v3 additions:
      status       — 'confirmed' | 'pending_review' | 'dismissed'
                     High-confidence detectors write 'confirmed' directly.
                     Low-confidence detectors (Phase 4: chewing, eating,
                     cash-in-pocket, dirty-floor) write 'pending_review'.
                     Admin can promote pending → confirmed via PATCH /alerts/{id}/confirm
                     (which also fires the Telegram notification).
                     'dismissed' = admin decided it was a false positive.
    """

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message = Column(String(500), nullable=False)
    role = Column(String(50), nullable=True)
    severity = Column(String(20), default="medium")  # low / medium / high / critical
    detected_issue = Column(String(200), nullable=True)
    confidence = Column(Float, nullable=True)
    snapshot_b64 = Column(Text, nullable=True)  # base64 JPEG for immediate WS display
    snapshot_path = Column(
        String(500), nullable=True
    )  # path to annotated image on disk
    timestamp = Column(DateTime, default=datetime.utcnow)

    # v2 — nullable so existing rows are untouched
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    floor = Column(
        String(50), nullable=True
    )  # denormalised; values: ground/first/second/shop
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)

    # v3 — confidence tier / review workflow
    # Default = 'confirmed' preserves backward compat for all existing rows.
    status = Column(
        String(20), nullable=False, default="confirmed"
    )  # confirmed | pending_review | dismissed

    # Relationships
    user = relationship("User", back_populates="alerts")
    camera = relationship("Camera", back_populates="alerts", foreign_keys=[camera_id])
    employee = relationship(
        "Employee", back_populates="alerts", foreign_keys=[employee_id]
    )


class UserConfig(Base):
    __tablename__ = "user_configs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    camera_type = Column(String(20), default="webcam")  # webcam / rtsp / upload
    rtsp_url = Column(String(500), nullable=True)
    notify_sound = Column(Boolean, default=True)
    notify_ui = Column(Boolean, default=True)
    detection_sensitivity = Column(Float, default=0.5)
    # JSON-encoded list of violation class names the user wants monitored.
    # e.g. '["NO-Hardhat","NO-Gloves"]'
    custom_ppe_items = Column(Text, nullable=True)
    # When True, ANY detected phone triggers an alert (not just near-ear usage).
    no_phone_zone = Column(Boolean, default=False)

    user = relationship("User", back_populates="config")


class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    label = Column(String(100), nullable=False)  # "owner", custom name
    encoding_data = Column(Text, nullable=False)  # JSON array of floats
    image_path = Column(String(500), nullable=True)
    thumbnail_b64 = Column(Text, nullable=True)  # small base64 face crop for UI preview
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="face_encodings")
    # Back-reference: employees that use this encoding for face-match
    employees = relationship("Employee", back_populates="face_encoding")


# ──────────────────────────────────────────────────────────────────────────────
# NEW TABLES  (multi-camera / multi-floor factory monitoring)
# ──────────────────────────────────────────────────────────────────────────────


class Camera(Base):
    """
    Physical RTSP camera mounted in the factory.

    floor   : ground | first | second | shop
              (stored as String — SQLite has no native ENUM; validated at API layer)
    status  : online | offline | error
    zone_type : free-text description of what this camera covers,
                e.g. "entrance", "dough-mixing", "oven", "packing", "shop-counter"
    """

    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    floor = Column(String(20), nullable=False)  # ground | first | second | shop
    zone_type = Column(String(100), nullable=True)  # entrance, dough-mixing, oven, …
    rtsp_url = Column(String(500), nullable=True)
    status = Column(String(20), default="offline")  # online | offline | error
    last_seen_at = Column(DateTime, nullable=True)  # last heartbeat / frame received
    created_at = Column(DateTime, default=datetime.utcnow)

    # Enterprise camera inventory.  Every field is nullable/defaulted so the
    # original camera CRUD contract remains valid while existing installations
    # are migrated in-place.
    camera_code = Column(String(100), unique=True, nullable=True, index=True)
    department = Column(String(100), nullable=True)
    purpose = Column(String(200), nullable=True)
    camera_type = Column(String(50), nullable=True)
    mount_height_m = Column(Float, nullable=True)
    view_direction = Column(String(100), nullable=True)
    resolution = Column(String(50), nullable=True)
    configured_fps = Column(Float, nullable=True)
    calibration_status = Column(String(40), nullable=False, default="not_calibrated")
    calibration_version = Column(Integer, nullable=False, default=0)
    last_calibrated_at = Column(DateTime, nullable=True)
    rule_profile_id = Column(Integer, nullable=True)
    workflow_profile_id = Column(Integer, nullable=True)
    ai_enabled = Column(Boolean, nullable=False, default=True)
    supports_multi_zone = Column(Boolean, nullable=False, default=True)
    supports_ocr = Column(Boolean, nullable=False, default=False)
    supports_pose = Column(Boolean, nullable=False, default=False)
    supports_tracking = Column(Boolean, nullable=False, default=True)
    supports_recording = Column(Boolean, nullable=False, default=False)
    supports_snapshot = Column(Boolean, nullable=False, default=True)
    heartbeat_at = Column(DateTime, nullable=True)
    health_status = Column(String(40), nullable=False, default="unknown")
    reference_frame_path = Column(String(500), nullable=True)
    drift_score = Column(Float, nullable=True)

    # Vendor-neutral IP-camera inventory.  RTSP credentials are deliberately
    # not stored on this row; CameraCredential holds encrypted values instead.
    manufacturer = Column(String(120), nullable=True)
    model = Column(String(160), nullable=True)
    ip_address = Column(String(64), nullable=True, index=True)
    onvif_endpoint = Column(String(500), nullable=True)
    discovery_id = Column(String(100), nullable=True, unique=True, index=True)
    preferred_stream = Column(String(20), nullable=False, default="sub")
    ai_stream = Column(String(20), nullable=False, default="sub")

    # Children
    zones = relationship(
        "ZoneConfig", back_populates="camera", cascade="all, delete-orphan"
    )
    idle_sessions = relationship("IdleSession", back_populates="camera")
    cylinder_logs = relationship("CylinderLog", back_populates="camera")
    invoice_logs = relationship("InvoiceLog", back_populates="camera")
    order_logs = relationship("OrderFormLog", back_populates="camera")
    alerts = relationship(
        "Alert", back_populates="camera", foreign_keys="Alert.camera_id"
    )
    dirty_baselines = relationship(
        "DirtyFloorBaseline", back_populates="camera", cascade="all, delete-orphan"
    )
    credentials = relationship(
        "CameraCredential",
        back_populates="camera",
        uselist=False,
        cascade="all, delete-orphan",
    )
    streams = relationship(
        "CameraStreamProfile", back_populates="camera", cascade="all, delete-orphan"
    )
    health_records = relationship(
        "CameraHealth", back_populates="camera", cascade="all, delete-orphan"
    )


class CameraCredential(Base):
    """Encrypted camera credentials. Never serialize this model into an API response."""

    __tablename__ = "camera_credentials"

    camera_id = Column(Integer, ForeignKey("cameras.id"), primary_key=True)
    encrypted_username = Column(Text, nullable=False)
    encrypted_password = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    camera = relationship("Camera", back_populates="credentials")


class CameraStreamProfile(Base):
    """One authenticated ONVIF media profile and its encrypted RTSP URI."""

    __tablename__ = "camera_streams"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    profile_token = Column(String(200), nullable=True)
    stream_type = Column(String(20), nullable=False, default="sub")
    codec = Column(String(40), nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    encrypted_rtsp_uri = Column(Text, nullable=False)
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    camera = relationship("Camera", back_populates="streams")


class CameraHealth(Base):
    """Append-only camera health measurements for diagnostics and trends."""

    __tablename__ = "camera_health"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    status = Column(String(40), nullable=False)
    fps = Column(Float, nullable=True)
    bitrate_kbps = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    packet_loss = Column(Float, nullable=True)
    last_frame_at = Column(DateTime, nullable=True)
    reconnect_count = Column(Integer, nullable=False, default=0)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    camera = relationship("Camera", back_populates="health_records")


class Employee(Base):
    """
    Person being monitored on camera.

    This is NOT a login account (has no password).  Admin/supervisor logins
    live in User.  An Employee can optionally be linked to a FaceEncoding so
    that the face-match pipeline can identify them automatically.
    """

    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    role = Column(String(100), nullable=True)  # baker, packer, cashier, …
    department = Column(String(100), nullable=True)  # ground-floor, shop, …
    face_encoding_id = Column(Integer, ForeignKey("face_encodings.id"), nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    face_encoding = relationship(
        "FaceEncoding", back_populates="employees", foreign_keys=[face_encoding_id]
    )
    idle_sessions = relationship("IdleSession", back_populates="employee")
    alerts = relationship(
        "Alert", back_populates="employee", foreign_keys="Alert.employee_id"
    )
    attendance_records = relationship(
        "AttendanceRecord",
        foreign_keys="AttendanceRecord.employee_id",
        back_populates="employee",
        lazy="dynamic",
    )


class ZoneConfig(Base):
    """
    Polygon zone defined for one camera.

    polygon_json : JSON text of [[x,y], [x,y], …] pixel coordinates
                   in the camera's native frame resolution.
    zone_name    : logical name, e.g. "entrance", "cashbox", "window",
                   "dough_table_1", "oven_section".
    """

    __tablename__ = "zone_configs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    zone_name = Column(String(200), nullable=False)
    polygon_json = Column(
        Text, nullable=False
    )  # e.g. "[[120,80],[320,80],[320,300],[120,300]]"
    created_at = Column(DateTime, default=datetime.utcnow)

    # Versioned, rule-ready zone metadata.  zone_name/polygon_json are kept as
    # the legacy public contract; these columns add the requested semantics.
    zone_type = Column(String(100), nullable=True)
    preset_type = Column(String(100), nullable=True)
    display_name = Column(String(200), nullable=True)
    color = Column(String(20), nullable=True)
    priority = Column(Integer, nullable=False, default=0)
    workflow_stage = Column(String(100), nullable=True)
    rule_profile_id = Column(Integer, nullable=True)
    expected_objects_json = Column(Text, nullable=True)
    allowed_objects_json = Column(Text, nullable=True)
    forbidden_objects_json = Column(Text, nullable=True)
    time_constraints_json = Column(Text, nullable=True)
    alert_thresholds_json = Column(Text, nullable=True)
    movement_threshold = Column(Float, nullable=True)
    idle_threshold = Column(Float, nullable=True)
    confidence_threshold = Column(Float, nullable=True)
    visibility_threshold = Column(Float, nullable=True)
    calibration_version = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    camera = relationship("Camera", back_populates="zones")


class IdleSession(Base):
    """
    Records one idle-timer event for a track inside a camera zone.

    track_id     : ephemeral ByteTrack / BoT-SORT integer — resets across
                   restarts, so it is NOT a stable identity.
    employee_id  : filled in only when the face-match pipeline positively
                   identifies who the track belongs to; NULL = unidentified.
    end_time /
    duration_seconds : NULL while the session is still open (person still idle).
                       Closed (filled) when the person moves or exits the zone.
    """

    __tablename__ = "idle_sessions"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    track_id = Column(Integer, nullable=False)  # from ByteTrack
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)  # NULL = still open
    duration_seconds = Column(Float, nullable=True)  # NULL = still open
    zone_name = Column(String(200), nullable=True)  # zone where idle detected

    camera = relationship("Camera", back_populates="idle_sessions")
    employee = relationship("Employee", back_populates="idle_sessions")


class CylinderLog(Base):
    """
    Tracks LPG / gas cylinder usage on any floor.

    event_type     : "detected" — cylinder seen in frame
                     "swapped"  — cylinder replaced (new cylinder detected)
    usage_day_count : running count of days this cylinder has been in use
                      (incremented by the rule engine once per calendar day).
    """

    __tablename__ = "cylinder_logs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    event_type = Column(String(50), nullable=False)  # detected | swapped
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    usage_day_count = Column(Integer, nullable=True)  # cumulative usage days

    camera = relationship("Camera", back_populates="cylinder_logs")


class InvoiceLog(Base):
    """
    Stores a successfully-scanned inward invoice document.

    Populated by ocr_service.scan_document_in_frame() when:
      - direction == "inward"
      - approved == True

    raw_ocr_text is always stored for admin audit, even for approved documents.
    snapshot_b64 holds a JPEG crop of the document area (for visual verification).

    ⚠️  "approved" here means the OCR text matched the invoice heuristic.
        It does NOT trigger any hardware gate — access decisions remain manual.
    """

    __tablename__ = "invoice_logs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(
        Integer, ForeignKey("cameras.id"), nullable=True
    )  # nullable for manual entries
    employee_id = Column(
        Integer, ForeignKey("employees.id"), nullable=True
    )  # filled if face-matched
    direction = Column(String(10), nullable=False, default="inward")  # inward | outward
    raw_ocr_text = Column(
        Text, nullable=True
    )  # full OCR output — always logged for admin review
    approved = Column(Boolean, nullable=False, default=False)
    snapshot_b64 = Column(Text, nullable=True)  # JPEG crop of the document
    ocr_available = Column(
        Boolean, nullable=False, default=True
    )  # False = Tesseract not installed
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    # QR upload tracking
    upload_token = Column(String(500), nullable=True)   # JWT token used for this QR upload session
    submitted_by_phone = Column(Boolean, nullable=False, default=False)  # True = uploaded via QR on phone
    goods_count = Column(Integer, nullable=True)  # extracted goods count from OCR

    camera = relationship("Camera", back_populates="invoice_logs")
    employee = relationship("Employee", foreign_keys=[employee_id])


class OrderFormLog(Base):
    """
    Stores a successfully-scanned outward order-form document.

    Populated by ocr_service.scan_document_in_frame() when:
      - direction == "outward"
      - approved == True

    Same audit/snapshot pattern as InvoiceLog.
    denied (approved=False) outward events generate an Alert instead.

    ⚠️  "approved" here means the OCR text matched the order-form heuristic.
        It does NOT trigger any hardware gate — access decisions remain manual.
    """

    __tablename__ = "order_form_logs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    employee_id = Column(
        Integer, ForeignKey("employees.id"), nullable=True
    )  # person showing the form
    direction = Column(String(10), nullable=False, default="outward")
    raw_ocr_text = Column(Text, nullable=True)
    approved = Column(Boolean, nullable=False, default=False)
    snapshot_b64 = Column(Text, nullable=True)
    ocr_available = Column(Boolean, nullable=False, default=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    # QR upload tracking
    upload_token = Column(String(500), nullable=True)   # JWT token used for this QR upload session
    submitted_by_phone = Column(Boolean, nullable=False, default=False)  # True = uploaded via QR on phone

    camera = relationship("Camera", back_populates="order_logs")
    employee = relationship("Employee", foreign_keys=[employee_id])


# ──────────────────────────────────────────────────────────────────────────────
# ATTENDANCE & WORKFLOW TABLES
# ──────────────────────────────────────────────────────────────────────────────


class AttendanceRecord(Base):
    """
    Employee clock-in / clock-out record.

    method : 'face'   — automatically fired by face-recognition pipeline
             'manual' — desk/admin entry
             'qr'     — QR-code scanner (future)

    employee_id=NULL means an unrecognised visitor or an entry made before
    face-match completes; the admin can link it to an Employee later.
    """

    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    user_id = Column(
        Integer, ForeignKey("users.id"), nullable=True
    )  # who created it (manual)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    clock_in = Column(DateTime, nullable=False, default=datetime.utcnow)
    clock_out = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)  # computed on clock-out
    method = Column(String(20), default="manual")  # face | manual | qr
    notes = Column(String(500), nullable=True)

    employee = relationship(
        "Employee", foreign_keys=[employee_id], back_populates="attendance_records"
    )
    camera = relationship("Camera", foreign_keys=[camera_id])


# Indexes for fast attendance queries (defined after class so SQLAlchemy registers them)
# ix_att_emp_clockin: speeds up today_summary and per-employee history lookups
from sqlalchemy import Index as _Idx

_Idx("ix_att_emp_clockin", AttendanceRecord.employee_id, AttendanceRecord.clock_in)
_Idx("ix_att_clockout_null", AttendanceRecord.clock_out)  # fast open-session scans


class LiftEvent(Base):
    """
    Records one lift zone entry or exit event.

    event_type : 'entry' — person entered the lift zone
                 'exit'  — person exited (duration_sec is filled)
                 'idle'  — person still in lift after idle_limit_sec (alert fired)
    """

    __tablename__ = "lift_events"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    event_type = Column(String(30), nullable=False)  # entry | exit | idle
    floor_from = Column(String(20), nullable=True)  # ground | first | second | shop
    floor_to = Column(String(20), nullable=True)
    duration_sec = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    camera = relationship("Camera", foreign_keys=[camera_id])
    employee = relationship("Employee", foreign_keys=[employee_id])


# ──────────────────────────────────────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────────────────────────────────────


class SystemSettings(Base):
    """
    Admin-editable key/value threshold store.

    Replaces hardcoded Python constants in config.py / yolo_service.py so
    thresholds can be updated via the admin UI without redeploying code.

    All values are stored as strings; the rule_engine casts on read.
    See rule_engine.py for the full list of recognised keys and their defaults.

    Examples
    --------
      key="idle_limit_default"      value="300"    description="Idle alert (seconds)"
      key="shift_start_ground"      value="08:00"  description="Ground floor shift start"
      key="dirty_floor_threshold"   value="0.08"   description="Dirty-floor pixel fraction"
    """

    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(String(500), nullable=False)
    description = Column(String(500), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DirtyFloorBaseline(Base):
    """
    Stores the path of a client-supplied clean-state reference photo for one
    camera / zone combination.

    The rule_engine's DirtyFloorDetector uses these paths to load the
    baseline image against which live frames are compared.

    Workflow:
      1. Client provides a photo of how the zone looks when clean.
      2. Admin uploads via POST /cameras/{id}/baseline (routers/settings.py).
      3. DirtyFloorDetector.reload() is called so the new baseline takes effect
         immediately without a server restart.
    """

    __tablename__ = "dirty_floor_baselines"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    zone_name = Column(String(200), nullable=False)  # e.g. "ground_entrance"
    image_path = Column(
        String(500), nullable=False
    )  # absolute or relative path on disk
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    camera = relationship("Camera", back_populates="dirty_baselines")


# ---------------------------------------------------------------------------
# Enterprise platform tables
# ---------------------------------------------------------------------------
# These are deliberately additive.  Existing User, Alert, Camera and
# ZoneConfig records continue to operate exactly as before while new services
# write durable domain state here.


class CalibrationVersion(Base):
    __tablename__ = "calibration_versions"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    snapshot_path = Column(String(500), nullable=True)
    reference_frame_path = Column(String(500), nullable=True)
    zones_json = Column(Text, nullable=False, default="[]")
    change_note = Column(String(500), nullable=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    restored_from_id = Column(
        Integer, ForeignKey("calibration_versions.id"), nullable=True
    )


class WorkflowProfile(Base):
    __tablename__ = "workflow_profiles"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    floor = Column(String(100), nullable=True)
    definition_json = Column(Text, nullable=False, default="{}")
    enabled = Column(Boolean, nullable=False, default=True)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RuleProfile(Base):
    __tablename__ = "rule_profiles"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    enabled = Column(Boolean, nullable=False, default=True)
    definition_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RuleDefinition(Base):
    __tablename__ = "rule_definitions"

    id = Column(Integer, primary_key=True)
    profile_id = Column(
        Integer, ForeignKey("rule_profiles.id"), nullable=True, index=True
    )
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    priority = Column(String(30), nullable=False, default="warning")
    enabled = Column(Boolean, nullable=False, default=True)
    zone_id = Column(Integer, ForeignKey("zone_configs.id"), nullable=True)
    workflow_stage = Column(String(100), nullable=True)
    required_events_json = Column(Text, nullable=False, default="[]")
    forbidden_events_json = Column(Text, nullable=False, default="[]")
    conditions_json = Column(Text, nullable=False, default="{}")
    time_threshold_sec = Column(Float, nullable=True)
    confidence_threshold = Column(Float, nullable=True)
    cooldown_sec = Column(Float, nullable=False, default=30)
    escalation_json = Column(Text, nullable=False, default="{}")
    notification_targets_json = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SurveillanceEvent(Base):
    __tablename__ = "surveillance_events"

    id = Column(Integer, primary_key=True)
    event_id = Column(String(64), nullable=False, unique=True, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    occurred_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True, index=True)
    zone_id = Column(Integer, ForeignKey("zone_configs.id"), nullable=True, index=True)
    track_id = Column(String(100), nullable=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    workflow_profile_id = Column(
        Integer, ForeignKey("workflow_profiles.id"), nullable=True
    )
    calibration_version = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    correlation_id = Column(String(100), nullable=True, index=True)
    source = Column(String(100), nullable=False, default="platform")
    payload_json = Column(Text, nullable=False, default="{}")


class ContextSnapshot(Base):
    __tablename__ = "context_snapshots"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    track_id = Column(String(100), nullable=False, index=True)
    occurred_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    zone_id = Column(Integer, ForeignKey("zone_configs.id"), nullable=True)
    workflow_stage = Column(String(100), nullable=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    context_json = Column(Text, nullable=False, default="{}")


class TrackSession(Base):
    __tablename__ = "track_sessions"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    track_id = Column(String(100), nullable=False, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    first_position_json = Column(Text, nullable=True)
    last_position_json = Column(Text, nullable=True)
    movement_state = Column(String(50), nullable=True)
    metadata_json = Column(Text, nullable=False, default="{}")


class TrackZoneHistory(Base):
    __tablename__ = "track_zone_history"

    id = Column(Integer, primary_key=True)
    track_session_id = Column(
        Integer, ForeignKey("track_sessions.id"), nullable=True, index=True
    )
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    track_id = Column(String(100), nullable=False, index=True)
    zone_id = Column(Integer, ForeignKey("zone_configs.id"), nullable=True)
    entered_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    exited_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)


class RuleEvaluation(Base):
    __tablename__ = "rule_evaluations"

    id = Column(Integer, primary_key=True)
    rule_id = Column(
        Integer, ForeignKey("rule_definitions.id"), nullable=False, index=True
    )
    event_id = Column(String(64), nullable=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    outcome = Column(String(30), nullable=False)
    reason = Column(Text, nullable=True)
    evaluated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    context_json = Column(Text, nullable=False, default="{}")


class AlertCase(Base):
    __tablename__ = "alert_cases"

    id = Column(Integer, primary_key=True)
    public_id = Column(String(64), nullable=False, unique=True, index=True)
    rule_id = Column(Integer, ForeignKey("rule_definitions.id"), nullable=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True, index=True)
    event_id = Column(String(64), nullable=True, index=True)
    status = Column(String(40), nullable=False, default="open")
    severity = Column(String(30), nullable=False, default="warning")
    title = Column(String(300), nullable=False)
    details_json = Column(Text, nullable=False, default="{}")
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    closed_at = Column(DateTime, nullable=True)


class NotificationDelivery(Base):
    __tablename__ = "notification_deliveries"

    id = Column(Integer, primary_key=True)
    alert_case_id = Column(
        Integer, ForeignKey("alert_cases.id"), nullable=True, index=True
    )
    channel = Column(String(50), nullable=False)
    target = Column(String(300), nullable=True)
    status = Column(String(30), nullable=False, default="queued")
    provider_message_id = Column(String(200), nullable=True)
    attempted_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    payload_json = Column(Text, nullable=False, default="{}")


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True)
    model_key = Column(String(150), nullable=False, unique=True, index=True)
    display_name = Column(String(200), nullable=False)
    model_type = Column(String(50), nullable=False, index=True)
    provider = Column(String(100), nullable=True)
    version = Column(String(100), nullable=False)
    artifact_path = Column(String(500), nullable=True)
    class_map_json = Column(Text, nullable=False, default="{}")
    capabilities_json = Column(Text, nullable=False, default="[]")
    config_json = Column(Text, nullable=False, default="{}")
    enabled = Column(Boolean, nullable=False, default=True)
    health_status = Column(String(40), nullable=False, default="unknown")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class HealthLog(Base):
    __tablename__ = "health_logs"

    id = Column(Integer, primary_key=True)
    component_type = Column(String(80), nullable=False, index=True)
    component_id = Column(String(100), nullable=False, index=True)
    status = Column(String(40), nullable=False)
    measured_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    metrics_json = Column(Text, nullable=False, default="{}")
    message = Column(Text, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    actor_user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    action = Column(String(150), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False)
    entity_id = Column(String(100), nullable=True)
    occurred_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    before_json = Column(Text, nullable=True)
    after_json = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=False, default="{}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables that don't already exist.  Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)


def ensure_enterprise_schema() -> None:
    """Apply additive SQLite migrations for the enterprise camera and zone fields.

    SQLAlchemy's ``create_all`` creates new tables but intentionally never adds
    columns to an existing table.  This small, idempotent migrator keeps the
    original SQLite deployment upgrade-safe without introducing a destructive
    migration dependency.  Production PostgreSQL deployments should run the
    equivalent Alembic revisions before application startup.
    """
    additions = {
        "cameras": {
            "camera_code": "VARCHAR(100)",
            "department": "VARCHAR(100)",
            "purpose": "VARCHAR(200)",
            "camera_type": "VARCHAR(50)",
            "mount_height_m": "FLOAT",
            "view_direction": "VARCHAR(100)",
            "resolution": "VARCHAR(50)",
            "configured_fps": "FLOAT",
            "calibration_status": "VARCHAR(40) NOT NULL DEFAULT 'not_calibrated'",
            "calibration_version": "INTEGER NOT NULL DEFAULT 0",
            "last_calibrated_at": "DATETIME",
            "rule_profile_id": "INTEGER",
            "workflow_profile_id": "INTEGER",
            "ai_enabled": "BOOLEAN NOT NULL DEFAULT 1",
            "supports_multi_zone": "BOOLEAN NOT NULL DEFAULT 1",
            "supports_ocr": "BOOLEAN NOT NULL DEFAULT 0",
            "supports_pose": "BOOLEAN NOT NULL DEFAULT 0",
            "supports_tracking": "BOOLEAN NOT NULL DEFAULT 1",
            "supports_recording": "BOOLEAN NOT NULL DEFAULT 0",
            "supports_snapshot": "BOOLEAN NOT NULL DEFAULT 1",
            "heartbeat_at": "DATETIME",
            "health_status": "VARCHAR(40) NOT NULL DEFAULT 'unknown'",
            "reference_frame_path": "VARCHAR(500)",
            "drift_score": "FLOAT",
        },
        "invoice_logs": {
            "upload_token": "VARCHAR(500)",
            "submitted_by_phone": "BOOLEAN NOT NULL DEFAULT 0",
            "goods_count": "INTEGER",
        },

        "order_form_logs": {
            "upload_token": "VARCHAR(500)",
            "submitted_by_phone": "BOOLEAN NOT NULL DEFAULT 0",
        },
        "zone_configs": {
            "zone_type": "VARCHAR(100)",
            "preset_type": "VARCHAR(100)",
            "display_name": "VARCHAR(200)",
            "color": "VARCHAR(20)",
            "priority": "INTEGER NOT NULL DEFAULT 0",
            "workflow_stage": "VARCHAR(100)",
            "rule_profile_id": "INTEGER",
            "expected_objects_json": "TEXT",
            "allowed_objects_json": "TEXT",
            "forbidden_objects_json": "TEXT",
            "time_constraints_json": "TEXT",
            "alert_thresholds_json": "TEXT",
            "movement_threshold": "FLOAT",
            "idle_threshold": "FLOAT",
            "confidence_threshold": "FLOAT",
            "visibility_threshold": "FLOAT",
            "calibration_version": "INTEGER NOT NULL DEFAULT 0",
            "is_active": "BOOLEAN NOT NULL DEFAULT 1",
            "updated_at": "DATETIME",
        },
    }
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table_name, columns in additions.items():
            if table_name not in inspector.get_table_names():
                continue
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, column_sql in columns.items():
                if column_name not in existing:
                    conn.execute(
                        text(
                            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
                        )
                    )
