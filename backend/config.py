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
    # Model chain (do NOT delete older files — keep as rollback):
    #   ppe.pt              → original 10-class model (backup — never delete)
    #   ppe_factory_v1.pt   → Phase 1: 17 classes, public data (deploy after Phase 1)
    #   ppe_factory_v2.pt   → Phase 2: 17 classes, client footage (deploy after Phase 2)
    # Set env var YOLO_MODEL=ppe_factory_v1.pt to override without changing code.
    YOLO_MODEL: str = "ppe_factory_v1.pt"

    # Confidence threshold for YOLO inference (0–1).
    # 0.30 is safe for Phase 1 — new classes trained on public data may score
    # lower confidence; tighten to 0.40 after Phase 2 (client footage).
    DETECTION_CONF: float = 0.30   # Lower = more sensitive (catches distant/partial PPE)
    NMS_IOU: float = 0.40           # NMS threshold — lower removes fewer overlapping boxes
    IOU_PERSON_PPE: float = 0.10    # Lower = PPE assigned to person even at edges

    # Minimum confidence for a violation to trigger an alert save.
    # Must be >= DETECTION_CONF so genuine detections always pass.
    MIN_VIOLATION_CONF: float = 0.30

    # Seconds between saved alerts per user (prevents DB flooding).
    ALERT_COOLDOWN: int = 3

    # Process every Nth frame in video uploads (~5 fps at 25fps source).
    FRAME_SKIP: int = 3

    # ── Idle tracking limits (seconds) ─────────────────────────────────────────
    # zone_name from ZoneConfig maps to a limit here.
    # Falls back to "default" if zone_name not found.
    # Change these without restarting the server by editing .env or config.py.
    IDLE_LIMITS: dict = {
        "default":          300,   # 5 min — all floors / general zones
        "shop":              60,   # 1 min — shop counter / absent from shop
        "camera_standing":   60,   # 1 min — person blocking the camera
        "cashbox":          120,   # 2 min — standing at cashbox
    }

    # Pixels a centroid must move between frames to reset the idle timer.
    # Lower = more sensitive (resets on tiny shifts). 8px is robust to RTSP jitter.
    IDLE_MOVEMENT_THRESHOLD_PX: int = 8

    # Only "confirmed" status alerts are sent; "pending_review" stays in-app only.
    TELEGRAM_BOT_TOKEN: str = ""   # set in .env — never commit this value
    TELEGRAM_CHAT_ID:   str = ""   # group chat id (negative number for group chats)

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.FACE_ENCODINGS_DIR, exist_ok=True)
