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
    # ppe_factory_v2.pt   → Phase 2: 17 classes, client footage (retrain)
    YOLO_MODEL: str = "ppe_factory_v0_cash.pt"
    DETECTION_CONF: float = 0.30  # tighten to 0.40 after Phase 2 retraining
    NMS_IOU: float = 0.40
    IOU_PERSON_PPE: float = 0.10
    MIN_VIOLATION_CONF: float = 0.30
    ALERT_COOLDOWN: int = 3
    FRAME_SKIP: int = 3  # process every Nth frame in video uploads

    # ── Idle tracking limits (seconds) ─────────────────────────────────────
    IDLE_LIMITS: dict = {
        "default": 300,      # 5 min — all floors
        "shop": 60,          # 1 min — shop counter
        "camera_standing": 60,  # 1 min — person blocking camera
        "cashbox": 120,      # 2 min — cashbox zone
    }
    IDLE_MOVEMENT_THRESHOLD_PX: int = 8

    # ── Push notifications ──────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = ""  # set in .env — never commit
    TELEGRAM_CHAT_ID: str = ""
    NTFY_TOPIC: str = ""          # primary channel — e.g. "bakery-alerts-xyz"
    NTFY_SERVER: str = "https://ntfy.sh"

    # ── Security ────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    # Comma-separated allowed CORS origins (empty = allow all, dev only).
    # Production: ALLOWED_ORIGINS=http://192.168.1.100,http://192.168.1.100:5173
    ALLOWED_ORIGINS: str = ""

    # ── Local sound alarm ───────────────────────────────────────────────────
    LOCAL_ALARM_ENABLED: bool = False
    LOCAL_ALARM_FREQ_HZ: int = 1000
    LOCAL_ALARM_DURATION_MS: int = 600

    # ── Inference performance ───────────────────────────────────────────────
    YOLO_INFERENCE_WIDTH: int = 640
    MEDIAPIPE_ENABLED: bool = True

    # ── Evidence & Debug Crops ──────────────────────────────────────────────
    EVIDENCE_DIR: str = "uploads/evidence"
    DEBUG_CROPS: bool = False
    DEBUG_CROPS_DIR: str = "debug/crops"

    # ── Image Classifier (Teachable Machine / ONNX) ─────────────────────────
    CLASSIFIER_INTERVAL_MS: int = 1000
    CLASSIFIER_SMOOTHING_WINDOW: int = 5

    # ── Shift schedules (floor → required_start_time HH:MM) ─────────────────
    SHIFT_SCHEDULES: dict = {
        "ground": "08:00",
        "dough": "06:00",
        "second": "05:00",
        "shop": "09:00",
    }

    # ── Shop absence threshold (seconds) ────────────────────────────────────
    SHOP_ABSENCE_THRESHOLD_SEC: int = 60

    # ── Camera standing threshold (seconds) ─────────────────────────────────
    CAMERA_STANDING_THRESHOLD_SEC: int = 60

    # ── Mock / demo mode ────────────────────────────────────────────────────
    # Set MOCK_MODE=true to run the full dashboard without real cameras or ML
    # models.  The mock detector generates fake persons + bounding boxes; the
    # mock camera produces synthetic frames at MOCK_FPS.
    MOCK_MODE: bool = False
    MOCK_CAMERA_IDS: str = ""      # comma-separated camera IDs to inject mocks
    MOCK_PERSON_COUNT: int = 3     # simulated persons per camera per tick
    MOCK_FPS: float = 5.0          # tick rate of the mock frame generator


settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.FACE_ENCODINGS_DIR, exist_ok=True)
os.makedirs(settings.EVIDENCE_DIR, exist_ok=True)
if settings.DEBUG_CROPS:
    os.makedirs(settings.DEBUG_CROPS_DIR, exist_ok=True)
