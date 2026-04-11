import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SECRET_KEY: str = "safety-monitor-super-secret-key-2024-change-in-prod"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    DATABASE_URL: str = "sqlite:///./safety.db"
    UPLOAD_DIR: str = "uploads"
    FACE_ENCODINGS_DIR: str = "face_data"

    # ── Detection Model Settings ────────────────────────────────────────────
    # Choices: yolov8n.pt (fastest), yolov8m.pt (balanced), yolov8l.pt (accurate)
    # Set env var YOLO_MODEL=yolov8l.pt to override without changing code.
    YOLO_MODEL: str = "yolov8m.pt"

    # Confidence threshold for YOLO inference (0–1). Higher = fewer false positives.
    DETECTION_CONF: float = 0.50

    # NMS IoU threshold — lower removes more overlapping boxes.
    NMS_IOU: float = 0.45

    # IoU threshold for assigning a PPE bbox to a person bbox.
    # A helmet overlapping ≥15% of person area = "person has helmet".
    IOU_PERSON_PPE: float = 0.15

    # Minimum confidence for a violation to trigger an alert save.
    MIN_VIOLATION_CONF: float = 0.50

    # Seconds between saved alerts per user (prevents DB flooding).
    ALERT_COOLDOWN: int = 5

    # Process every Nth frame in video uploads (~5 fps at 25fps source).
    FRAME_SKIP: int = 3

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.FACE_ENCODINGS_DIR, exist_ok=True)
