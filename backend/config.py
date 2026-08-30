import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str = "safety-monitor-super-secret-key-2024-change-in-prod"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    DATABASE_URL: str = "sqlite:///./safety.db"
    UPLOAD_DIR: str = "uploads"
    FACE_ENCODINGS_DIR: str = "face_data"

    # ── YOLO model chain ────────────────────────────────────────────────────
    # ppe.pt              → original 10-class model (backup — never delete)
    # ppe_factory_v0.pt   → Phase 0: upgraded factory model (ACTIVE)
    # ppe_factory_v1.pt   → Phase 1: 17 classes, public data
    # ppe_factory_v2.pt   → Phase 2: 17 classes, client footage (retrain with real footage)
    YOLO_MODEL: str = "ppe_factory_v0_cash.pt"
    DETECTION_CONF: float = 0.30  # tighten to 0.40 after Phase 2 retraining
    NMS_IOU: float = 0.40
    IOU_PERSON_PPE: float = 0.10
    MIN_VIOLATION_CONF: float = 0.30
    ALERT_COOLDOWN: int = 3
    FRAME_SKIP: int = 3  # process every Nth frame in video uploads

    # ── Idle tracking limits (seconds) ─────────────────────────────────────
    IDLE_LIMITS: dict = {
        "default": 300,  # 5 min — all floors
        "shop": 60,  # 1 min — shop counter
        "camera_standing": 60,  # 1 min — person blocking camera
        "cashbox": 120,  # 2 min — cashbox zone
    }
    IDLE_MOVEMENT_THRESHOLD_PX: int = 8

    # ── Push notifications ──────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = ""  # set in .env — never commit
    TELEGRAM_CHAT_ID: str = ""
    NTFY_TOPIC: str = ""  # primary channel — e.g. "bakery-alerts-xyz"
    NTFY_SERVER: str = "https://ntfy.sh"

    # ── Security ────────────────────────────────────────────────────────────
    # Set APP_ENV=production in .env on the live server.
    # Production startup will refuse the default SECRET_KEY.
    APP_ENV: str = "development"
    # Comma-separated allowed CORS origins (empty = allow all, dev only).
    # Production example: ALLOWED_ORIGINS=http://192.168.1.100,http://192.168.1.100:5173
    ALLOWED_ORIGINS: str = ""

    # ── Local sound alarm ───────────────────────────────────────────────────
    # Plays an audible beep on the SERVER machine for high-severity alerts.
    # Works on Windows (winsound) and Linux (beep command / paplay).
    LOCAL_ALARM_ENABLED: bool = False
    LOCAL_ALARM_FREQ_HZ: int = 1000
    LOCAL_ALARM_DURATION_MS: int = 600

    # ── Inference performance ───────────────────────────────────────────────
    # Reduce YOLO_INFERENCE_WIDTH to 480 for 20+ cameras on 16 GB RAM.
    YOLO_INFERENCE_WIDTH: int = 640
    # Set False to disable MediaPipe (chew/cleanshave) if RAM is constrained.
    MEDIAPIPE_ENABLED: bool = True


settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.FACE_ENCODINGS_DIR, exist_ok=True)
