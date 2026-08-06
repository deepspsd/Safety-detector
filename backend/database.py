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

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Text, Boolean, ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ──────────────────────────────────────────────────────────────────────────────
# EXISTING TABLES  (kept intact — no column removed or renamed)
# ──────────────────────────────────────────────────────────────────────────────

class User(Base):
    """Admin / supervisor login account.  NOT the person being monitored."""
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String(100), nullable=True)
    email           = Column(String(200), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role            = Column(String(50), nullable=True)   # admin, supervisor, …
    created_at      = Column(DateTime, default=datetime.utcnow)

    alerts        = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    config        = relationship("UserConfig", back_populates="user", uselist=False, cascade="all, delete-orphan")
    face_encodings = relationship("FaceEncoding", back_populates="user", cascade="all, delete-orphan")


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

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    message        = Column(String(500), nullable=False)
    role           = Column(String(50), nullable=True)
    severity       = Column(String(20), default="medium")   # low / medium / high / critical
    detected_issue = Column(String(200), nullable=True)
    confidence     = Column(Float, nullable=True)
    snapshot_b64   = Column(Text, nullable=True)            # base64 JPEG for immediate WS display
    snapshot_path  = Column(String(500), nullable=True)     # path to annotated image on disk
    timestamp      = Column(DateTime, default=datetime.utcnow)

    # v2 — nullable so existing rows are untouched
    camera_id   = Column(Integer, ForeignKey("cameras.id"),   nullable=True)
    floor       = Column(String(50), nullable=True)           # denormalised; values: ground/first/second/shop
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)

    # v3 — confidence tier / review workflow
    # Default = 'confirmed' preserves backward compat for all existing rows.
    status      = Column(String(20), nullable=False, default="confirmed")  # confirmed | pending_review | dismissed

    # Relationships
    user     = relationship("User",     back_populates="alerts")
    camera   = relationship("Camera",   back_populates="alerts",   foreign_keys=[camera_id])
    employee = relationship("Employee", back_populates="alerts",   foreign_keys=[employee_id])


class UserConfig(Base):
    __tablename__ = "user_configs"

    id                   = Column(Integer, primary_key=True, index=True)
    user_id              = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    camera_type          = Column(String(20), default="webcam")   # webcam / rtsp / upload
    rtsp_url             = Column(String(500), nullable=True)
    notify_sound         = Column(Boolean, default=True)
    notify_ui            = Column(Boolean, default=True)
    detection_sensitivity = Column(Float, default=0.5)
    # JSON-encoded list of violation class names the user wants monitored.
    # e.g. '["NO-Hardhat","NO-Gloves"]'
    custom_ppe_items     = Column(Text, nullable=True)
    # When True, ANY detected phone triggers an alert (not just near-ear usage).
    no_phone_zone        = Column(Boolean, default=False)

    user = relationship("User", back_populates="config")


class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    label         = Column(String(100), nullable=False)    # "owner", custom name
    encoding_data = Column(Text, nullable=False)           # JSON array of floats
    image_path    = Column(String(500), nullable=True)
    thumbnail_b64 = Column(Text, nullable=True)            # small base64 face crop for UI preview
    created_at    = Column(DateTime, default=datetime.utcnow)

    user      = relationship("User",     back_populates="face_encodings")
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

    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String(200), nullable=False)
    floor        = Column(String(20),  nullable=False)   # ground | first | second | shop
    zone_type    = Column(String(100), nullable=True)    # entrance, dough-mixing, oven, …
    rtsp_url     = Column(String(500), nullable=True)
    status       = Column(String(20),  default="offline") # online | offline | error
    last_seen_at = Column(DateTime,    nullable=True)    # last heartbeat / frame received
    created_at   = Column(DateTime,    default=datetime.utcnow)

    # Children
    zones          = relationship("ZoneConfig",          back_populates="camera", cascade="all, delete-orphan")
    idle_sessions  = relationship("IdleSession",         back_populates="camera")
    cylinder_logs  = relationship("CylinderLog",         back_populates="camera")
    invoice_logs   = relationship("InvoiceLog",          back_populates="camera")
    order_logs     = relationship("OrderFormLog",        back_populates="camera")
    alerts         = relationship("Alert",               back_populates="camera", foreign_keys="Alert.camera_id")
    dirty_baselines = relationship("DirtyFloorBaseline", back_populates="camera", cascade="all, delete-orphan")


class Employee(Base):
    """
    Person being monitored on camera.

    This is NOT a login account (has no password).  Admin/supervisor logins
    live in User.  An Employee can optionally be linked to a FaceEncoding so
    that the face-match pipeline can identify them automatically.
    """
    __tablename__ = "employees"

    id                = Column(Integer, primary_key=True, index=True)
    name              = Column(String(200), nullable=False)
    role              = Column(String(100), nullable=True)        # baker, packer, cashier, …
    department        = Column(String(100), nullable=True)        # ground-floor, shop, …
    face_encoding_id  = Column(Integer, ForeignKey("face_encodings.id"), nullable=True)
    active            = Column(Boolean,  default=True)
    created_at        = Column(DateTime, default=datetime.utcnow)

    # Relationships
    face_encoding = relationship("FaceEncoding", back_populates="employees", foreign_keys=[face_encoding_id])
    idle_sessions = relationship("IdleSession",  back_populates="employee")
    alerts        = relationship("Alert",        back_populates="employee", foreign_keys="Alert.employee_id")


class ZoneConfig(Base):
    """
    Polygon zone defined for one camera.

    polygon_json : JSON text of [[x,y], [x,y], …] pixel coordinates
                   in the camera's native frame resolution.
    zone_name    : logical name, e.g. "entrance", "cashbox", "window",
                   "dough_table_1", "oven_section".
    """
    __tablename__ = "zone_configs"

    id           = Column(Integer, primary_key=True, index=True)
    camera_id    = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    zone_name    = Column(String(200), nullable=False)
    polygon_json = Column(Text,        nullable=False)   # e.g. "[[120,80],[320,80],[320,300],[120,300]]"
    created_at   = Column(DateTime,    default=datetime.utcnow)

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

    id               = Column(Integer, primary_key=True, index=True)
    camera_id        = Column(Integer, ForeignKey("cameras.id"),   nullable=False)
    track_id         = Column(Integer, nullable=False)              # from ByteTrack
    employee_id      = Column(Integer, ForeignKey("employees.id"), nullable=True)
    start_time       = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time         = Column(DateTime, nullable=True)              # NULL = still open
    duration_seconds = Column(Float,    nullable=True)              # NULL = still open
    zone_name        = Column(String(200), nullable=True)           # zone where idle detected

    camera   = relationship("Camera",   back_populates="idle_sessions")
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

    id              = Column(Integer, primary_key=True, index=True)
    camera_id       = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    event_type      = Column(String(50), nullable=False)   # detected | swapped
    timestamp       = Column(DateTime, nullable=False, default=datetime.utcnow)
    usage_day_count = Column(Integer, nullable=True)        # cumulative usage days

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

    id             = Column(Integer, primary_key=True, index=True)
    camera_id      = Column(Integer, ForeignKey("cameras.id"), nullable=True)   # nullable for manual entries
    employee_id    = Column(Integer, ForeignKey("employees.id"), nullable=True)  # filled if face-matched
    direction      = Column(String(10),  nullable=False, default="inward")       # inward | outward
    raw_ocr_text   = Column(Text,        nullable=True)    # full OCR output — always logged for admin review
    approved       = Column(Boolean,     nullable=False, default=False)
    snapshot_b64   = Column(Text,        nullable=True)    # JPEG crop of the document
    ocr_available  = Column(Boolean,     nullable=False, default=True)   # False = Tesseract not installed
    timestamp      = Column(DateTime,    nullable=False, default=datetime.utcnow)

    camera   = relationship("Camera",   back_populates="invoice_logs")
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

    id             = Column(Integer, primary_key=True, index=True)
    camera_id      = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    employee_id    = Column(Integer, ForeignKey("employees.id"), nullable=True)  # person showing the form
    direction      = Column(String(10),  nullable=False, default="outward")
    raw_ocr_text   = Column(Text,        nullable=True)
    approved       = Column(Boolean,     nullable=False, default=False)
    snapshot_b64   = Column(Text,        nullable=True)
    ocr_available  = Column(Boolean,     nullable=False, default=True)
    timestamp      = Column(DateTime,    nullable=False, default=datetime.utcnow)

    camera   = relationship("Camera",   back_populates="order_logs")
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

    id          = Column(Integer,     primary_key=True, index=True)
    key         = Column(String(100), unique=True, nullable=False, index=True)
    value       = Column(String(500), nullable=False)
    description = Column(String(500), nullable=True)
    updated_at  = Column(DateTime,    default=datetime.utcnow, onupdate=datetime.utcnow)


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

    id          = Column(Integer,     primary_key=True, index=True)
    camera_id   = Column(Integer,     ForeignKey("cameras.id"), nullable=False)
    zone_name   = Column(String(200), nullable=False)          # e.g. "ground_entrance"
    image_path  = Column(String(500), nullable=False)          # absolute or relative path on disk
    uploaded_at = Column(DateTime,    default=datetime.utcnow)

    camera = relationship("Camera", back_populates="dirty_baselines")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables that don't already exist.  Safe to call on every startup."""
    Base.metadata.create_all(bind=engine)
