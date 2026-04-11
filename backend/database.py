from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
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


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    email = Column(String(200), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(50), nullable=True)  # Doctor, Traffic Police, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    config = relationship("UserConfig", back_populates="user", uselist=False, cascade="all, delete-orphan")
    face_encodings = relationship("FaceEncoding", back_populates="user", cascade="all, delete-orphan")


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message = Column(String(500), nullable=False)
    role = Column(String(50), nullable=True)
    severity = Column(String(20), default="medium")  # low, medium, high, critical
    detected_issue = Column(String(200), nullable=True)
    confidence = Column(Float, nullable=True)
    snapshot_b64 = Column(Text, nullable=True)   # base64 JPEG (for immediate display in WS)
    snapshot_path = Column(String(500), nullable=True)  # path to saved ANNOTATED image on disk
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="alerts")


class UserConfig(Base):
    __tablename__ = "user_configs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    camera_type = Column(String(20), default="webcam")  # webcam, rtsp, upload
    rtsp_url = Column(String(500), nullable=True)
    notify_sound = Column(Boolean, default=True)
    notify_ui = Column(Boolean, default=True)
    detection_sensitivity = Column(Float, default=0.5)
    # JSON-encoded list of violation class names the user wants monitored.
    # e.g. '["NO-Hardhat","NO-Gloves"]'
    # Used as active filters when role == "None", or as overrides for other roles.
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
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="face_encodings")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)
