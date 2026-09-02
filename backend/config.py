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
    # Primary YOLO detection model
    # yolov8x.pt          → High-accuracy COCO model (person, cell phone, objects)
    # hairnet, fall, anomaly, throwing models loaded via MultiModelRegistry
    YOLO_MODEL: str = "yolov8x.pt"
    DETECTION_CONF: float = 0.60
    NMS_IOU: float = 0.40
    IOU_PERSON_PPE: float = 0.10
    MIN_VIOLATION_CONF: float = 0.60
    ALERT_COOLDOWN: int = 3
    FRAME_SKIP: int = 3  # process every Nth frame in video uploads

    # ── Multi-Model Registry ────────────────────────────────────────────────
    # Controls which AI models are loaded and used for which zones.
    # Set enabled=False to disable a model without removing its config.
    MODEL_REGISTRY: dict = {
        "yolov8x_coco": {
            "path": "yolov8x.pt",
            "type": "yolo",
            "enabled": True,
            "priority": 0,  # PRIMARY - person detection, vehicles, objects
            "conf_threshold": 0.50,
            "target_fps": 10,
            "zones": ["*"],  # All zones
            "description": "YOLOv8x COCO - General object detection (person, vehicle, etc.)",
            "classes": {
                0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
                24: "backpack", 26: "handbag", 28: "suitcase", 
                39: "bottle", 41: "cup", 67: "cell phone"
            }
        },
        "hairnet_glove_detection": {
            "path": "../portable_models_package/hairnet_glove_detection/best.pt",
            "type": "yolo",
            "enabled": True,
            "priority": 1,
            "conf_threshold": 0.60,
            "target_fps": 5,
            "zones": ["dough_mixing", "oven", "packing", "biscuit_cutting", "entrance", "dough_table", "gas_section"],
            "description": "Food safety - hairnet, gloves, hand detection",
            "classes": {
                0: "Back_Palm", 1: "Front_Palm", 2: "Hair", 
                3: "Hair_Cover", 4: "Hand_Gloves"
            }
        },
        "fall_detection": {
            "path": "../portable_models_package/fall_detection/best.pt",
            "type": "yolo",
            "enabled": True,
            "priority": 1,
            "conf_threshold": 0.60,
            "target_fps": 10,
            "zones": ["dough_mixing", "oven", "packing", "biscuit_cutting", "entrance", "lift", "gas_section", "dough_table"],
            "description": "Worker safety - fall detection",
            "classes": {0: "non-fall", 1: "fall"}
        },
        "helmet_detection": {
            "path": "helmet_model.pt",
            "type": "yolo",
            "enabled": False,  # Optional - enable if hardhat required
            "priority": 2,
            "conf_threshold": 0.50,
            "target_fps": 5,
            "zones": ["entrance", "lift"],
            "description": "Helmet/hardhat detection",
        },
        "cash_detection": {
            "path": "../portable_models_package/cash_detection/yolo11m_finetuned.pt",
            "type": "yolo",
            "enabled": True,
            "priority": 2,
            "conf_threshold": 0.60,
            "target_fps": 3,
            "zones": ["shop_counter", "cashbox"],
            "description": "Cash monitoring - banknote detection (EUR/BGN, works generically)",
            "classes": {
                0: "5 BGN", 1: "10 BGN", 2: "20 BGN", 3: "50 BGN", 4: "100 BGN",
                5: "5 EUR", 6: "10 EUR", 7: "20 EUR", 8: "50 EUR", 9: "100 EUR"
            }
        },
        "hand_landmarks": {
            "path": "../portable_models_package/hand_landmarks/hand_landmarker.task",
            "type": "mediapipe",
            "enabled": False,  # Phase 2+, expensive compute
            "priority": 3,
            "target_fps": 2,
            "zones": ["packing", "shop_counter"],
            "description": "Hand tracking - idle detection, cash-in-pocket",
        },
        "person_action_recognition": {
            "path": "../portable_models_package/person_action",
            "type": "openvino",
            "enabled": False,  # Phase 4, R&D
            "priority": 3,
            "target_fps": 1,
            "zones": ["*"],
            "description": "Action recognition - behavioral analysis",
        },
        "machine_visual_anomaly": {
            "path": "../portable_models_package/machine_visual_anomaly/model_metal_nut_64.pt",
            "type": "pytorch",
            "enabled": True,
            "priority": 2,
            "conf_threshold": 0.60,
            "target_fps": 3,
            "zones": ["dough_mixing", "biscuit_cutting", "cutting_machine"],
            "description": "Visual Anomaly Detection for Machinery / Metal Parts (MVTec)",
        },
        "machine_sensor_anomaly": {
            "path": "../portable_models_package/machine_sensor_anomaly/rf_forecast_model.joblib",
            "scaler_path": "../portable_models_package/machine_sensor_anomaly/scaler.joblib",
            "type": "sklearn",
            "enabled": True,
            "priority": 2,
            "conf_threshold": 0.50,
            "target_fps": 2,
            "zones": ["dough_mixing", "biscuit_cutting", "cutting_machine"],
            "description": "IoT Sensor Anomaly Detection - Predictive Maintenance (needs real sensor data)",
        },
        # OLD MODEL - DEPRECATED
        "ppe_factory_v0_cash": {
            "path": "ppe_factory_v0_cash.pt",
            "type": "yolo",
            "enabled": False,  # DISABLED - replaced by yolov8x_coco + specialized models
            "priority": 99,
            "conf_threshold": 0.60,
            "target_fps": 10,
            "zones": ["*"],
            "description": "[DEPRECATED] Old PPE model - disabled",
        },
        # Temporal action recognition - throwing / aggressive motion detection
        "object_throwing": {
            "path": "../portable_models_package/object_throwing/TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar",
            "type": "pytorch_trn",  # custom dispatch in multi_model_detector
            "enabled": True,
            "priority": 2,
            "conf_threshold": 0.60,
            "target_fps": 2,
            "zones": ["entrance", "shop_counter", "cashbox", "lift"],
            "description": "TRN temporal action recognition - throwing / aggressive motion detection",
        },
    }

    # ── Zone-to-Model Mapping (auto-generated helper) ──────────────────────
    # Defines which models should run for each zone type.
    # This is a convenience map; the actual routing uses MODEL_REGISTRY zones field.
    ZONE_MODEL_MAP: dict = {
        "entrance":         ["yolov8x_coco", "hairnet_glove_detection", "fall_detection", "object_throwing"],
        "dough_mixing":     ["yolov8x_coco", "hairnet_glove_detection", "fall_detection", "machine_visual_anomaly", "machine_sensor_anomaly"],
        "dough_table":      ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
        "oven":             ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
        "biscuit_cutting":  ["yolov8x_coco", "hairnet_glove_detection", "fall_detection", "machine_visual_anomaly", "machine_sensor_anomaly"],
        "cutting_machine":  ["yolov8x_coco", "hairnet_glove_detection", "fall_detection", "machine_visual_anomaly", "machine_sensor_anomaly"],
        "packing":          ["yolov8x_coco", "hairnet_glove_detection"],
        "shop_counter":     ["yolov8x_coco", "cash_detection", "object_throwing"],
        "lift":             ["yolov8x_coco", "fall_detection", "object_throwing"],
        "gas_section":      ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
        "cashbox":          ["yolov8x_coco", "cash_detection", "object_throwing"],
        "default":          ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
    }

    # ── Model Performance Tuning ────────────────────────────────────────────
    # Frame rate limits per model (to prevent CPU overload)
    MODEL_FPS_LIMITS: dict = {
        "yolov8x_coco": 10,             # Primary: 10 FPS
        "hairnet_glove_detection": 5,   # Secondary: 5 FPS
        "fall_detection": 10,           # Safety-critical: 10 FPS
        "helmet_detection": 5,          # Optional: 5 FPS
        "cash_detection": 3,            # Shop only: 3 FPS
        "hand_landmarks": 2,            # Expensive: 2 FPS
        "person_action_recognition": 1, # Very expensive: 1 FPS
    }

    # Detection fusion - merge overlapping detections from multiple models
    ENABLE_DETECTION_FUSION: bool = True
    FUSION_IOU_THRESHOLD: float = 0.7  # Merge detections with IoU > 0.7

    # Class mapping for detection fusion (map different class names to canonical violation types)
    VIOLATION_CLASS_MAPPING: dict = {
        # Hair/head violations
        "Hair": "no_hair_cover",
        "NO-Hairnet": "no_hair_cover",
        "NO-Bakery-Head-Cap": "no_hair_cover",
        # Glove violations  
        "Back_Palm": "no_gloves",
        "Front_Palm": "no_gloves",
        "NO-Gloves": "no_gloves",
        # Compliant (keep for reverse mapping)
        "Hair_Cover": "hair_cover_ok",
        "Hand_Gloves": "gloves_ok",
        # Fall detection
        "fall": "worker_fall",
        # Mask violations
        "NO-Mask": "no_mask",
        # Hardhat violations
        "NO-Hardhat": "no_hardhat",
        # Safety vest violations
        "NO-Safety Vest": "no_safety_vest",
    }

    # ── Bakery Head-Cap compliance ───────────────────────────────────────────
    # How many seconds a headcap must be continuously missing before an alert
    # is fired.  Prevents false alerts from single-frame detection gaps.
    HEADCAP_MISSING_SECONDS: float = 3.0
    # After an alert fires for a person, suppress further alerts for this many
    # seconds (unless they become compliant and then violate again).
    HEADCAP_ALERT_COOLDOWN: float = 30.0
    # Minimum YOLO confidence for a Bakery-Head-Cap detection to count as valid.
    HEADCAP_CONF_THRESHOLD: float = 0.25

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
    UNIFORM_MODEL_DIR: str = "../training/models"
    UNIFORM_MISSING_SECONDS: float = 3.0
    UNIFORM_ALERT_COOLDOWN: float = 30.0
    UNIFORM_CONF_THRESHOLD: float = 0.75

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
