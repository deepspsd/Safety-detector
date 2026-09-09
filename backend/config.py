import os

from pydantic_settings import BaseSettings

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)


def _resolve_model_path(env_var: str, default_rel: str) -> str:
    """Resolve model path from environment variable or project root/backend directory."""
    val = os.getenv(env_var)
    if val:
        if os.path.isabs(val) and os.path.exists(val):
            return val
        p_root = os.path.join(_PROJECT_ROOT, val)
        if os.path.exists(p_root):
            return p_root
        p_backend = os.path.join(_BACKEND_DIR, val)
        if os.path.exists(p_backend):
            return p_backend
        return val
    p_def_root = os.path.normpath(os.path.join(_PROJECT_ROOT, default_rel))
    if os.path.exists(p_def_root):
        return p_def_root
    return os.path.normpath(os.path.join(_BACKEND_DIR, default_rel))


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
    DETECTION_CONF: float = 0.25  # Lowered from 0.60 to 0.25 for better person detection
    NMS_IOU: float = 0.40
    IOU_PERSON_PPE: float = 0.10
    MIN_VIOLATION_CONF: float = 0.40  # Lowered from 0.60 to 0.40
    ALERT_COOLDOWN: int = 3
    FRAME_SKIP: int = 3  # process every Nth frame in video uploads
    HAIRNET_CONF_THRESHOLD: float = float(os.getenv("HAIRNET_CONF_THRESHOLD", "0.45"))
    HAIRNET_IMGSZ: int = int(os.getenv("HAIRNET_IMGSZ", "960"))
    DEBUG_HAIRNET_RAW: bool = os.getenv("DEBUG_HAIRNET_RAW", "false").lower() in ("true", "1", "yes")

    # ── Face Recognition Tuning ──────────────────────────────────────────────
    FACE_RECOGNITION_TOLERANCE: float = 0.52  # 0.52 = balanced (0.60 = dlib loose, 0.42 = overly strict)
    FACE_MIN_PIXELS: int = 30  # minimum face bounding box height in pixels

    # ── Multi-Model Registry ────────────────────────────────────────────────
    # Controls which AI models are loaded and used for which zones.
    # Set enabled=False to disable a model without removing its config.
    MODEL_REGISTRY: dict = {
        "yolov8x_coco": {
            "path": _resolve_model_path("YOLO_MODEL_PATH", os.getenv("YOLO_MODEL", "yolov8x.pt")),
            "type": "yolo",
            "enabled": True,
            "priority": 0,  # PRIMARY - person detection, vehicles, objects
            "conf_threshold": 0.25,  # Lowered for sensitive person detection
            "target_fps": 15,
            "zones": ["*"],  # All zones
            "description": "YOLOv8x COCO - General object detection (person, vehicle, phone, etc.)",
            "capabilities": ["person_detection", "phone_detection", "vehicle_detection", "bottle_detection"],
            "classes": {
                0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
                24: "backpack", 26: "handbag", 28: "suitcase", 
                39: "bottle", 41: "cup", 67: "cell phone"
            }
        },
        "hairnet_glove_detection": {
            "path": _resolve_model_path("HAIRNET_MODEL_PATH", "portable_models_package/hairnet_glove_detection/best.pt"),
            "type": "yolo",
            "enabled": True,
            "priority": 1,
            "conf_threshold": float(os.getenv("HAIRNET_CONF_THRESHOLD", "0.40")),
            "imgsz": int(os.getenv("HAIRNET_IMGSZ", "960")),
            "target_fps": 15,
            "zones": ["*", "dough_mixing", "oven", "packing", "biscuit_cutting", "cutting_machine", "entrance", "dough_table", "gas_section", "shop", "shop_counter", "cashbox", "store", "default"],
            "description": "Food safety - hairnet detection (YOLOv8m 2-class head detector)",
            "capabilities": ["head_cover_compliance", "sanitation_check"],
            "classes": {
                0: "hairnet",
                1: "no_hairnet"
            }
        },
        "fall_detection": {
            "path": _resolve_model_path("FALL_MODEL_PATH", "portable_models_package/fall_detection/best.pt"),
            "type": "yolo",
            "enabled": True,
            "priority": 1,
            "conf_threshold": float(os.getenv("FALL_CONF_THRESHOLD", "0.95")),
            "target_fps": 15,
            "zones": ["*"],
            "description": "Worker safety - fall detection",
            "capabilities": ["worker_fall_detection", "person_down_detection"],
            "classes": {1: "fall"}
        },
        "cash_detection": {
            "path": _resolve_model_path("CASH_MODEL_PATH", "portable_models_package/cash_detection/yolo11m_finetuned.pt"),
            "type": "yolo",
            "enabled": True,
            "priority": 2,
            "conf_threshold": 0.50,
            "target_fps": 3,
            "zones": ["shop_counter", "shop", "cashbox", "cash"],
            "description": "Cash monitoring - banknote detection (EUR/BGN/currency)",
            "capabilities": ["cash_monitoring", "banknote_detection"],
            "classes": {
                0: "5 BGN", 1: "10 BGN", 2: "20 BGN", 3: "50 BGN", 4: "100 BGN",
                5: "5 EUR", 6: "10 EUR", 7: "20 EUR", 8: "50 EUR", 9: "100 EUR"
            }
        },
        "helmet_model": {
            "path": _resolve_model_path("HELMET_MODEL_PATH", "helmet_model.pt"),
            "type": "yolo",
            "enabled": True,
            "priority": 2,
            "conf_threshold": 0.40,
            "target_fps": 5,
            "zones": ["loading", "construction", "entrance"],
            "description": "Industrial Safety - Hardhat compliance",
            "capabilities": ["hardhat_compliance", "head_protection"],
            "classes": {0: "Hardhat", 1: "NO-Hardhat"}
        },
        "hand_landmarks": {
            "path": _resolve_model_path("HAND_LANDMARKS_PATH", "portable_models_package/hand_landmarks/hand_landmarker.task"),
            "type": "mediapipe",
            "enabled": False,  # Phase 2+, expensive compute
            "priority": 3,
            "conf_threshold": 0.50,
            "target_fps": 2,
            "zones": ["packing", "shop_counter", "shop"],
            "description": "MediaPipe Hand Tracking - Packing movement & idle hands",
            "capabilities": ["hand_motion_tracking", "packing_motion"]
        },
        "machine_sensor_anomaly": {
            "path": _resolve_model_path("MACHINE_ANOMALY_MODEL_PATH", "portable_models_package/machine_sensor_anomaly/rf_forecast_model.joblib"),
            "scaler_path": _resolve_model_path("MACHINE_ANOMALY_SCALER_PATH", "portable_models_package/machine_sensor_anomaly/scaler.joblib"),
            "type": "sklearn",
            "enabled": True,
            "priority": 2,
            "conf_threshold": 0.50,
            "target_fps": 2,
            "zones": ["dough_mixing", "biscuit_cutting", "cutting_machine"],
            "description": "IoT Sensor Anomaly Detection - Predictive Maintenance",
            "capabilities": ["machine_anomaly_prediction", "predictive_maintenance"]
        },
        "object_throwing": {
            "path": _resolve_model_path("OBJECT_THROWING_PATH", "portable_models_package/object_throwing/TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar"),
            "type": "pytorch_trn",
            "enabled": False,
            "priority": 2,
            "conf_threshold": 0.60,
            "target_fps": 2,
            "zones": ["entrance", "shop_counter", "shop", "cashbox", "lift", "window"],
            "description": "TRN temporal action recognition - throwing / window goods theft",
            "capabilities": ["object_throwing_detection", "theft_trajectory"]
        },
        "person_action_recognition": {
            "path": _resolve_model_path("PERSON_ACTION_MODEL_PATH", "portable_models_package/person_action"),
            "type": "openvino",
            "enabled": False,
            "precision": "FP16",
            "priority": 3,
            "conf_threshold": 0.50,
            "target_fps": 2,
            "zones": ["entrance", "shop_counter", "shop", "packing", "lift"],
            "description": "OpenVINO Person Detection + Action Recognition",
            "capabilities": ["person_action_monitoring"]
        },
    }

    # ── Capability Registry ────────────────────────────────────────────────
    # Maps semantic capabilities to compatible model keys in order of priority.
    CAPABILITY_REGISTRY: dict = {
        "person_detection": ["yolov8x_coco"],
        "phone_detection": ["yolov8x_coco"],
        "vehicle_detection": ["yolov8x_coco"],
        "bottle_detection": ["yolov8x_coco"],
        "head_cover_compliance": ["hairnet_glove_detection"],
        "sanitation_check": ["hairnet_glove_detection"],
        "worker_fall_detection": ["fall_detection"],
        "person_down_detection": ["fall_detection"],
        "cash_monitoring": ["cash_detection"],
        "banknote_detection": ["cash_detection"],
        "hardhat_compliance": ["helmet_model"],
        "head_protection": ["helmet_model", "hairnet_glove_detection"],
        "machine_anomaly_prediction": ["machine_sensor_anomaly"],
        "predictive_maintenance": ["machine_sensor_anomaly"],
        "hand_motion_tracking": ["hand_landmarks"],
        "packing_motion": ["hand_landmarks"],
        "object_throwing_detection": ["object_throwing"],
        "theft_trajectory": ["object_throwing"],
        "person_action_monitoring": ["person_action_recognition"],
    }

    # ── Zone-to-Capability Mapping ─────────────────────────────────────────
    # Default required capabilities for standard zone types.
    ZONE_CAPABILITY_MAP: dict = {
        "entrance":         ["person_detection", "head_cover_compliance", "worker_fall_detection", "vehicle_detection"],
        "dough_mixing":     ["person_detection", "head_cover_compliance", "worker_fall_detection", "machine_anomaly_prediction"],
        "dough_table":      ["person_detection", "head_cover_compliance", "worker_fall_detection"],
        "oven":             ["person_detection", "head_cover_compliance", "worker_fall_detection"],
        "biscuit_cutting":  ["person_detection", "head_cover_compliance", "worker_fall_detection", "machine_anomaly_prediction"],
        "cutting_machine":  ["person_detection", "head_cover_compliance", "worker_fall_detection", "machine_anomaly_prediction"],
        "packing":          ["person_detection", "head_cover_compliance", "worker_fall_detection", "hand_motion_tracking"],
        "shop":             ["person_detection", "cash_monitoring", "head_cover_compliance", "worker_fall_detection"],
        "shop_counter":     ["person_detection", "cash_monitoring", "head_cover_compliance", "worker_fall_detection"],
        "cashbox":          ["person_detection", "cash_monitoring", "head_cover_compliance"],
        "lift":             ["person_detection", "worker_fall_detection"],
        "gas_section":      ["person_detection", "head_cover_compliance", "worker_fall_detection"],
        "passage":          ["person_detection", "worker_fall_detection"],
        "store":            ["person_detection", "head_cover_compliance", "worker_fall_detection"],
        "raw_material":     ["person_detection", "worker_fall_detection"],
        "window":           ["person_detection", "object_throwing_detection"],
        "loading":          ["person_detection", "hardhat_compliance", "vehicle_detection"],
        "default":          ["person_detection", "head_cover_compliance", "worker_fall_detection"],
    }

    # ── Zone-to-Model Mapping (derived helper) ─────────────────────────────
    # Defines fallback models for each zone type.
    ZONE_MODEL_MAP: dict = {
        "entrance":         ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
        "dough_mixing":     ["yolov8x_coco", "hairnet_glove_detection", "fall_detection", "machine_sensor_anomaly"],
        "dough_table":      ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
        "oven":             ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
        "biscuit_cutting":  ["yolov8x_coco", "hairnet_glove_detection", "fall_detection", "machine_sensor_anomaly"],
        "cutting_machine":  ["yolov8x_coco", "hairnet_glove_detection", "fall_detection", "machine_sensor_anomaly"],
        "packing":          ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
        "shop":             ["yolov8x_coco", "hairnet_glove_detection", "fall_detection", "cash_detection"],
        "shop_counter":     ["yolov8x_coco", "hairnet_glove_detection", "fall_detection", "cash_detection"],
        "cashbox":          ["yolov8x_coco", "cash_detection", "hairnet_glove_detection"],
        "lift":             ["yolov8x_coco", "fall_detection"],
        "gas_section":      ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
        "passage":          ["yolov8x_coco", "fall_detection"],
        "store":            ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
        "raw_material":     ["yolov8x_coco", "fall_detection"],
        "window":           ["yolov8x_coco", "object_throwing"],
        "loading":          ["yolov8x_coco", "helmet_model"],
        "default":          ["yolov8x_coco", "hairnet_glove_detection", "fall_detection"],
    }

    # ── Model Performance Tuning ────────────────────────────────────────────
    # Frame rate limits per model (to prevent CPU overload)
    MODEL_FPS_LIMITS: dict = {
        "yolov8x_coco": 10,             # Primary: 10 FPS
        "hairnet_glove_detection": 5,   # Secondary: 5 FPS
        "fall_detection": 10,           # Safety-critical: 10 FPS
        "cash_detection": 3,            # Shop only: 3 FPS
        "machine_sensor_anomaly": 2,    # Machine IoT: 2 FPS
        "hand_landmarks": 2,            # Optional MediaPipe: 2 FPS
        "object_throwing": 2,           # Optional TRN: 2 FPS
        "person_action_recognition": 2, # Optional OpenVINO: 2 FPS
    }

    # Detection fusion - merge overlapping detections from multiple models
    ENABLE_DETECTION_FUSION: bool = True
    FUSION_IOU_THRESHOLD: float = 0.7  # Merge detections with IoU > 0.7

    # Class mapping for detection fusion (map different class names to canonical violation types)
    VIOLATION_CLASS_MAPPING: dict = {
        # Hair / headcap violations & compliant
        "Hair": "no_hair_cover",
        "NO-Hairnet": "no_hair_cover",
        "no_hairnet": "no_hair_cover",
        "NO-Bakery-Head-Cap": "no_hair_cover",
        "hairnet": "hair_cover_ok",
        "Hairnet": "hair_cover_ok",
        "Hair_Cover": "hair_cover_ok",
        "Bakery-Head-Cap": "hair_cover_ok",
        "hair_cover_ok": "hair_cover_ok",

        # Hand / glove violations & compliant
        "Back_Palm": "no_gloves",
        "Front_Palm": "no_gloves",
        "NO-Gloves": "no_gloves",
        "no_gloves": "no_gloves",
        "Hand_Gloves": "gloves_ok",
        "Gloves": "gloves_ok",
        "gloves_ok": "gloves_ok",

        # Bangles (strict food safety violation)
        "Bangles": "bangles",
        "bangles": "bangles",

        # Uniform violations & compliant
        "NO-Uniform": "no_uniform",
        "NON_UNIFORM": "no_uniform",
        "Uniform": "uniform_ok",
        "UNIFORM": "uniform_ok",

        # Mask violations & compliant
        "NO-Mask": "no_mask",
        "Mask": "mask_ok",

        # Worker fall detection
        "fall": "worker_fall",
        "Worker Fall": "worker_fall",
        "worker_fall": "worker_fall",

        # Machinery anomaly detection
        "Machine Anomaly": "machine_anomaly",
        "Machine_Sensor_Anomaly": "machine_anomaly",
        "anomaly": "machine_anomaly",

        # Currency / Cash monitoring
        "Cash": "cash",
        "cash": "cash",
        "5 BGN": "cash", "10 BGN": "cash", "20 BGN": "cash", "50 BGN": "cash", "100 BGN": "cash",
        "5 EUR": "cash", "10 EUR": "cash", "20 EUR": "cash", "50 EUR": "cash", "100 EUR": "cash",

        # Action / Throwing / Motion
        "Object Throwing": "object_throwing",
        "object_throwing": "object_throwing",
        "Object_Throwing": "object_throwing",

        # Hand landmarks
        "Hand-0": "hand",
        "Hand-1": "hand",

        # Construction PPE (legacy compatibility)
        "NO-Hardhat": "no_hardhat",
        "Hardhat": "hardhat_ok",
        "NO-Safety Vest": "no_safety_vest",
        "Safety Vest": "safety_vest_ok",
        "NO-Goggles": "no_goggles",
        "NO-Safety Shoes": "no_safety_shoes",
        "NO-ID Card": "no_id_card",
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

    # ── Push notifications (FCM HTTP v1) ──────────────────────────────────
    # Firebase project ID (e.g. occusafe-4f0c8)
    FCM_PROJECT_ID: str = "occusafe-4f0c8"
    # Path to the downloaded service account JSON key file
    # Firebase Console → Project Settings → Service Accounts → Generate new private key
    FCM_SERVICE_ACCOUNT_PATH: str = "firebase_service_account.json"
    # Seconds between repeated push notifications for the same (camera, issue_type).
    FCM_NOTIFICATION_COOLDOWN_SEC: int = 120

    # ── Event-based notification system ──────────────────────────────────
    # How long an anomaly must be continuously detected before confirmation (seconds)
    EVENT_CONFIRMATION_DURATION: float = 120.0  # 2 minutes
    # Grace period: how long to keep event active after last detection (seconds)
    EVENT_GRACE_PERIOD: float = 5.0  # 5 seconds of missed detections allowed
    # Minimum time between notifications for the same event (seconds)
    EVENT_NOTIFICATION_COOLDOWN: float = 300.0  # 5 minutes between repeated notifications
    # Maximum event duration before auto-resolve (seconds) - prevents zombie events
    EVENT_MAX_DURATION: float = 3600.0  # 1 hour
    # Cleanup: purge resolved events older than this (seconds)
    EVENT_CLEANUP_AGE: float = 86400.0  # 24 hours

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
