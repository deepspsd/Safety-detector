"""
YOLO Violation Detection Service — v3.1  (ppe_factory_v0 edition)
==================================================================
Active model: ppe_factory_v0.pt  (upgraded factory model — 17 classes)
Fallback chain: ppe_factory_v0.pt → ppe_factory_v1.pt → ppe.pt (10-class) → simulation

  Base classes (ppe.pt, 0-9):
    Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
    Person, Safety Cone, Safety Vest, machinery, vehicle

  Factory-extended classes (10-16):
    Bakery-Head-Cap, NO-Bakery-Head-Cap, Bangles,
    Document-in-hand, Cylinder, Exposed-Item, Cashbox

  Violation classes: NO-Hardhat, NO-Mask, NO-Safety Vest,
                     NO-Bakery-Head-Cap, Bangles, Exposed-Item
  Compliant classes: Hardhat, Mask, Safety Vest, Bakery-Head-Cap
  Neutral:           Person, Safety Cone, machinery, vehicle,
                     Document-in-hand, Cylinder, Cashbox

Flow per frame:
  1. Run model inference (conf ≥ DETECTION_CONF).
  2. Separate detections into violations, compliant PPE, persons, neutral.
  3. Associate violations/PPE to nearest Person bbox via IoU + containment.
  4. Draw RED box + label for violators.
  5. Draw GREEN box for persons with ALL required PPE present.
  6. Build role-specific compliance summary.
  7. Return enriched result for the FastAPI router.

Place model in:  backend/ppe_factory_v0.pt
"""

import base64
import datetime
import logging
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from config import settings

log = logging.getLogger("yolo_service")

# ─────────────────────────────────────────────────────────────────
# Class map shared across the model chain (index → name, as trained)
# ppe_factory_v0.pt adds classes 10-16 on top of the base ppe.pt set
# ─────────────────────────────────────────────────────────────────
PPE_CLASS_NAMES = [
    "Hardhat",  # 0  ✅ compliant
    "Mask",  # 1  ✅ compliant
    "NO-Hardhat",  # 2  ❌ violation
    "NO-Mask",  # 3  ❌ violation
    "NO-Safety Vest",  # 4  ❌ violation
    "Person",  # 5  👤 neutral person
    "Safety Cone",  # 6  🟠 neutral
    "Safety Vest",  # 7  ✅ compliant
    "machinery",  # 8  🔵 neutral
    "vehicle",  # 9  🔵 neutral
    # ── Phase 1 new classes (ppe_factory_v1.pt) ────────────────
    "Bakery-Head-Cap",  # 10 ✅ compliant (cloth cap worn correctly)
    "NO-Bakery-Head-Cap",  # 11 ❌ violation  (cap absent / wrong)
    "Bangles",  # 12 ❌ violation  (always flagged in food production)
    "Document-in-hand",  # 13 🔵 neutral  (triggers OCR pipeline at entrance)
    "Cylinder",  # 14 🔵 neutral  (usage counter)
    "Exposed-Item",  # 15 ❌ violation  (stock kept openly)
    "Cashbox",  # 16 🔵 neutral  (zone anchor for cash monitoring)
]

# Sets for fast membership checks
VIOLATION_CLASSES = {
    # Original ppe.pt violations
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    # Phase 1 new violations
    "NO-Bakery-Head-Cap",  # cap absent / incorrectly worn
    "Bangles",  # always a violation in food production
    "Exposed-Item",  # stock kept openly
}
COMPLIANT_CLASSES = {
    "Hardhat",
    "Mask",
    "Safety Vest",
    "Bakery-Head-Cap",  # cloth cap worn correctly
}
PERSON_CLASSES = {"Person"}
NEUTRAL_CLASSES = {
    "Safety Cone",
    "machinery",
    "vehicle",
    "Document-in-hand",
    "Cylinder",
    "Cashbox",  # Phase 1 neutral classes
}

# Human-readable violation → missing item label
# Original ppe.pt violation classes
VIOLATION_LABEL_MAP = {
    "NO-Hardhat": "No Hardhat",
    "NO-Mask": "No Mask",
    "NO-Safety Vest": "No Safety Vest",
    # Phase 1 — bakery-specific violations
    "NO-Bakery-Head-Cap": "No Head Cap",
    "Bangles": "Bangles Detected (Violation)",
    "Exposed-Item": "Stock Kept Openly",
    # Simulated classes (not detected by model natively)
    "NO-Gloves": "No Gloves",
    "NO-Goggles": "No Goggles",
    "NO-Safety Shoes": "No Safety Shoes",
    "NO-ID Card": "No ID Card",
    "NO-Uniform": "No Uniform",
}

# Classes that are purely simulated (not detected by ppe.pt natively)
SIM_ONLY_VIOLATIONS = {
    "NO-Gloves",
    "NO-Goggles",
    "NO-Safety Shoes",
    "NO-ID Card",
    "NO-Uniform",
}

# ── Traffic Police label remap ────────────────────────────────────
# ppe.pt uses "Hardhat" / "NO-Hardhat" — for Traffic Police display
# we remap these human-readable labels to "Helmet" / "No Helmet".
TRAFFIC_POLICE_LABEL_REMAP: Dict[str, str] = {
    "No Hardhat": "No Helmet",
    "Hardhat": "Helmet",
}

# ─────────────────────────────────────────────────────────────────
# Role-based required PPE  (maps to violation class names)
# Fields:
#   required_native    — classes ppe.pt detects natively
#   required_sim       — classes simulated (Gloves, Goggles)
#   severity / alert_prefix — for alert DB records
# ─────────────────────────────────────────────────────────────────
ROLE_RULES: Dict[str, Dict] = {
    "Construction Worker": {
        # ppe.pt natively detects these three violation classes
        "required_violations": ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"],
        "required_compliant": ["Hardhat", "Safety Vest", "Mask"],
        # Simulated PPE (Gloves, Goggles, Safety Shoes)
        "required_sim": ["NO-Gloves", "NO-Goggles", "NO-Safety Shoes"],
        "severity": "critical",
        "alert_prefix": "Construction safety violation",
    },
    "Doctor": {
        "required_violations": ["NO-Mask"],
        "required_compliant": ["Mask"],
        "required_sim": ["NO-Gloves"],
        "severity": "high",
        "alert_prefix": "🩺 Doctor PPE violation",
    },
    "Traffic Police": {
        "required_violations": ["NO-Hardhat"],
        "required_compliant": ["Hardhat"],
        "required_sim": [],
        "severity": "critical",
        "alert_prefix": "🚓 Traffic Police violation",
    },
    "College": {
        "required_violations": [],
        "required_compliant": [],
        "required_sim": ["NO-ID Card", "NO-Uniform"],
        "severity": "medium",
        "alert_prefix": "🎓 College compliance violation",
    },
    "Home": {
        "required_violations": [],
        "required_compliant": [],
        "required_sim": [],
        "severity": "low",
        "alert_prefix": "🏠 Home security",
    },
    # "None" = fully custom: required_violations come from detection_filters at runtime.
    # This entry provides defaults; the pipeline overrides it dynamically.
    "None": {
        "required_violations": [],
        "required_compliant": [],
        "required_sim": [],
        "severity": "high",
        "alert_prefix": "🛠️ Custom safety violation",
    },
    # ── Phase 1: Bakery / Food Factory Worker ─────────────────────────────
    # Violations that ppe_factory_v0.pt detects natively for this role.
    # Bangles is ALWAYS a violation in food production regardless of other PPE.
    "Bakery Worker": {
        "required_violations": [
            "NO-Bakery-Head-Cap",  # cap absent / incorrectly worn
            "Bangles",  # jewellery — always flagged in food production
            "NO-Mask",  # hygiene mask required
        ],
        "required_compliant": [
            "Bakery-Head-Cap",  # cloth cap worn correctly
            "Mask",  # face mask
        ],
        "required_sim": ["NO-Gloves"],
        "severity": "critical",
        "alert_prefix": "🧁 Bakery safety violation",
    },
    # Legacy alias — keep so existing DB rows with 'Factory Worker' still match
    "Factory Worker": {
        "required_violations": [
            "NO-Bakery-Head-Cap",
            "Bangles",
        ],
        "required_compliant": [
            "Bakery-Head-Cap",
        ],
        "required_sim": [],
        "severity": "critical",
        "alert_prefix": "🏭 Factory safety violation",
    },
}


# ─────────────────────────────────────────────────────────────────
# Visual colours  (BGR)
# ─────────────────────────────────────────────────────────────────
COLOR_VIOLATION = (30, 30, 220)  # Red
COLOR_COMPLIANT = (40, 200, 50)  # Green
COLOR_NEUTRAL = (180, 120, 60)  # Blue-grey
COLOR_PERSON = (200, 160, 60)  # Amber

# ─────────────────────────────────────────────────────────────────
# Model state  (loaded once at startup)
# ─────────────────────────────────────────────────────────────────
_model = None  # general PPE model (ppe.pt)
_helmet_model = None  # dedicated helmet model (keremberke / fallback)
_helmet_model_dedicated = False  # True when keremberke model (helmet/head classes)
_phone_model = None  # dedicated phone detection model (COCO class 67)
_use_simulation = False
_model_is_ppe = False  # True when ppe.pt loaded (vs generic COCO)

# ─────────────────────────────────────────────────────────────────
# ByteTrack tracker registry  (one tracker instance per camera_id)
# ─────────────────────────────────────────────────────────────────
# Keyed by camera_id (int).  Use camera_id=-1 as a sentinel for callers
# that don't have a camera_id (e.g. per-WebSocket legacy streams).
# Each tracker is a supervision.ByteTrack instance.
# Registry is protected by _tracker_lock for thread-safety.
import threading as _threading

_trackers: Dict[int, object] = {}  # int → sv.ByteTrack
_tracker_lock = _threading.Lock()


def _get_or_create_tracker(camera_id: int) -> object:
    """
    Return the supervision.ByteTrack instance for camera_id.
    Creates a new one on first call.  Thread-safe.
    """
    with _tracker_lock:
        if camera_id not in _trackers:
            try:
                import supervision as sv

                _trackers[camera_id] = sv.ByteTrack(
                    track_activation_threshold=0.25,
                    lost_track_buffer=30,  # frames to keep lost tracks
                    minimum_matching_threshold=0.8,
                    frame_rate=15,
                    minimum_consecutive_frames=1,
                )
                log.info(f"[Tracker] Created ByteTrack for camera_id={camera_id}")
            except ImportError:
                log.warning(
                    "[Tracker] supervision not installed — "
                    "track_id will be -1 for all persons. "
                    "Run: pip install supervision>=0.21.0"
                )
                _trackers[camera_id] = None  # sentinel: supervision unavailable
        return _trackers.get(camera_id)


def reset_tracker(camera_id: int):
    """
    Discard the ByteTrack state for camera_id (call when camera stops/restarts
    so track IDs don't bleed across sessions).
    """
    with _tracker_lock:
        _trackers.pop(camera_id, None)
    log.info(f"[Tracker] Reset tracker for camera_id={camera_id}")


def _apply_tracking(camera_id: int, persons: List[Dict]) -> List[Dict]:
    """
    Run ByteTrack on the given person detections and attach a stable
    'track_id' (int) to each person dict.

    If supervision is unavailable or camera_id is None, every person gets
    track_id=-1 and the rest of the pipeline is unaffected.

    This function ONLY adds 'track_id' — it does not modify bbox, confidence,
    or any PPE field.  Downstream code (_associate_to_persons, _draw_results)
    is completely unchanged.
    """
    if not persons:
        return persons

    tracker = _get_or_create_tracker(camera_id)
    if tracker is None:
        # supervision unavailable — assign sentinel IDs
        for p in persons:
            p["track_id"] = -1
        return persons

    try:
        import supervision as sv

        bboxes = np.array([p["bbox"] for p in persons], dtype=np.float32)
        confs = np.array([p["confidence"] for p in persons], dtype=np.float32)
        cls_ids = np.zeros(len(persons), dtype=int)  # all "person" class

        sv_dets = sv.Detections(
            xyxy=bboxes,
            confidence=confs,
            class_id=cls_ids,
        )
        tracked = tracker.update_with_detections(sv_dets)

        # tracked.xyxy and tracked.tracker_id are aligned arrays.
        # Match back to original persons by bbox proximity.
        track_ids = tracked.tracker_id  # np.ndarray[int] or None
        if track_ids is None:
            for p in persons:
                p["track_id"] = -1
            return persons

        # Build lookup: rounded bbox tuple → track_id
        bbox_to_tid: Dict[tuple, int] = {}
        for i, bbox in enumerate(tracked.xyxy):
            key = tuple(int(v) for v in bbox)
            bbox_to_tid[key] = int(track_ids[i])

        for p in persons:
            key = tuple(int(v) for v in p["bbox"])
            # Exact match first; fall back to -1 if tracker dropped the detection
            p["track_id"] = bbox_to_tid.get(key, -1)

    except Exception as exc:
        log.error(f"[Tracker] ByteTrack update failed (cam={camera_id}): {exc}")
        for p in persons:
            p["track_id"] = -1

    return persons


# ── Keremberke hard-hat-detection model constants ─────────────────
# https://huggingface.co/keremberke/yolov8m-hard-hat-detection
_HELMET_MODEL_URL = (
    "https://huggingface.co/keremberke/yolov8m-hard-hat-detection/resolve/main/best.pt"
)
_HELMET_MODEL_PATH = "helmet_model.pt"  # local cache path
# Classes in keremberke model — used by _run_traffic_police_helmet_pipeline
_HELMET_COMPLIANT_NAMES = {"helmet", "hard hat", "hardhat", "with helmet"}
_HELMET_VIOLATION_NAMES = {"head", "no helmet", "no_helmet", "without helmet"}


def _get_model_for_role(role: str):
    """
    Return the correct model instance for the given role.
    Always returns ppe.pt — the _helmet_model (yolov8m.pt / COCO) uses different
    class indices and cannot be used with PPE_CLASS_NAMES mapping.
    Traffic Police helmet labels are remapped via TRAFFIC_POLICE_LABEL_REMAP
    after detection in _run_pipeline.
    """
    return _model, _model_is_ppe


def load_model():
    """
    Load models ONCE at startup and cache globally.
    Order of preference for the general model:
      1. ppe_factory_v1.pt  (17-class factory model — Phase 1)
      2. ppe_factory_v2.pt  (17-class factory model — Phase 2, client footage)
      3. ppe.pt             (original 10-class — fallback if v1/v2 not present)
      4. YOLO_MODEL from config (e.g. yolov8m.pt — COCO generic)
      5. Simulation mode

    Additionally loads a specialized helmet detection model for Traffic Police:
      • helmet_model: yolov8m.pt (higher accuracy, helmet-focused via conf boost)
        Falls back to the same ppe model if unavailable.
    """
    global _model, _helmet_model, _phone_model, _use_simulation, _model_is_ppe

    import os

    os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")

    # ── 1. Load general PPE model (ppe.pt) ─────────────────────
    log.info("Loading PPE model…")
    try:
        import torch
        from ultralytics import YOLO

        _orig_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _patched_load

        # Priority: ppe_factory_v0.pt → ppe_factory_v1.pt (17-class) → ppe.pt (10-class) → YOLO_MODEL → simulation
        _factory_candidates = [
            "ppe_factory_v0.pt",
            "ppe_factory_v1.pt",
            "ppe_factory_v2.pt",
            "ppe.pt",
        ]
        _loaded = False
        for _candidate in _factory_candidates:
            try:
                _model = YOLO(_candidate)
                torch.load = _orig_load
                _use_simulation = False
                _model_is_ppe = True
                _nc = len(_model.names)
                log.info(f"✅ PPE model loaded: {_candidate} ({_nc} classes)")
                print(f"✅ PPE model loaded: {_candidate} ({_nc} classes)")
                _loaded = True
                break
            except Exception as e:
                log.warning(f"{_candidate} unavailable: {e}")
        if not _loaded:
            torch.load = _orig_load
    except Exception as e:
        log.warning(f"ppe.pt unavailable: {e}")
        try:
            torch.load = _orig_load
        except Exception:
            pass
        # Fallback: standard YOLO model
        model_name = settings.YOLO_MODEL
        try:
            from ultralytics import YOLO

            _model = YOLO(model_name)
            _use_simulation = False
            _model_is_ppe = False
            log.warning(f"⚠ Falling back to {model_name} (COCO, limited PPE classes)")
            print(f"⚠️  Falling back to {model_name} (COCO)")
        except Exception as e2:
            log.error(f"Model load failed ({e2}) — using simulation mode")
            _use_simulation = True
            print("ℹ️  Running in SIMULATION mode. Place ppe.pt in backend/")
            return

    # ── 2. Load dedicated helmet model for Traffic Police ─────────
    # Runs in a background thread to NEVER block server startup.
    # The file helmet_model.pt must be >10 MB to be considered valid.
    # (A 2-3 MB file = corrupt/partial download → delete and re-download.)
    import os
    import threading

    HELMET_MIN_BYTES = 10 * 1024 * 1024  # 10 MB

    def _load_helmet_bg():
        """Background thread: download + load dedicated helmet model."""
        global _helmet_model, _helmet_model_dedicated
        import urllib.request

        _hm_orig2 = None
        try:
            import torch
            from ultralytics import YOLO as _YOLO_hm

            _hm_orig2 = torch.load

            def _hm_patch2(*a, **kw):
                kw.setdefault("weights_only", False)
                return _hm_orig2(*a, **kw)

            torch.load = _hm_patch2

            # Validate existing file — reject partial downloads
            if os.path.exists(_HELMET_MODEL_PATH):
                sz = os.path.getsize(_HELMET_MODEL_PATH)
                if sz < HELMET_MIN_BYTES:
                    print(
                        f"⚠️  Removing corrupt helmet_model.pt ({sz//1024} KB < 10 MB)"
                    )
                    os.remove(_HELMET_MODEL_PATH)

            if os.path.exists(_HELMET_MODEL_PATH):
                # Valid cached model — just load it
                _helmet_model = _YOLO_hm(_HELMET_MODEL_PATH)
                _helmet_model_dedicated = True
                print(
                    f"✅ Dedicated helmet model loaded from cache: {_HELMET_MODEL_PATH}"
                )
            else:
                # Download from HuggingFace (one-time, ~52 MB)
                print("⏬ [background] Downloading dedicated helmet model…")
                print(f"   URL: {_HELMET_MODEL_URL}")
                urllib.request.urlretrieve(_HELMET_MODEL_URL, _HELMET_MODEL_PATH)  # nosec B310
                if os.path.getsize(_HELMET_MODEL_PATH) < HELMET_MIN_BYTES:
                    raise ValueError(
                        "Downloaded file is too small — likely a network error"
                    )
                _helmet_model = _YOLO_hm(_HELMET_MODEL_PATH)
                _helmet_model_dedicated = True
                print(
                    "✅ Dedicated helmet model ready — keremberke/yolov8m-hard-hat-detection"
                )
        except Exception as _hme:
            try:
                if _hm_orig2:
                    torch.load = _hm_orig2
            except Exception:
                pass
            # Clean up partial download
            try:
                if os.path.exists(_HELMET_MODEL_PATH) and not _helmet_model_dedicated:
                    os.remove(_HELMET_MODEL_PATH)
            except Exception:
                pass
            _helmet_model = None
            _helmet_model_dedicated = False
            print(
                f"⚠️  Helmet model unavailable ({type(_hme).__name__}). Strict ppe.pt logic will be used."
            )
        finally:
            try:
                if _hm_orig2:
                    torch.load = _hm_orig2
            except Exception:
                pass

    # Start the download thread — server startup continues immediately
    _ht = threading.Thread(target=_load_helmet_bg, daemon=True)
    _ht.start()
    print("ℹ️  Helmet model loading in background (does not block startup)…")

    # ── 3. Load dedicated phone detection model ────────────────────
    # Priority cascade: yolov8x.pt (best, ~137 MB, auto-downloads)
    #                 → yolov8l.pt (~87 MB, auto-downloads)
    #                 → yolov8m.pt (52 MB, already on disk — guaranteed fallback)
    # All are COCO models with class 67 = cell phone.
    log.info("Loading dedicated phone detection model…")
    _phone_candidates = ["yolov8x.pt", "yolov8l.pt", "yolov8m.pt"]
    _phone_loaded = False
    for _cand in _phone_candidates:
        try:
            import torch
            from ultralytics import YOLO

            _orig3 = torch.load

            def _p3(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig3(*a, **kw)

            torch.load = _p3
            _phone_model = YOLO(_cand)
            torch.load = _orig3
            log.info(f"✅ Phone model loaded: {_cand} (COCO, class 67 = cell phone)")
            print(f"✅ Phone detection model: {_cand}")
            _phone_loaded = True
            break
        except Exception as e:
            try:
                torch.load = _orig3
            except Exception:
                pass
            log.warning(f"{_cand} unavailable ({type(e).__name__}) — trying next…")
    if not _phone_loaded:
        _phone_model = _helmet_model if not _helmet_model_dedicated else None
        print(
            "⚠️  Phone model unavailable — falling back to simulation for phone detection"
        )


# ─────────────────────────────────────────────────────────────────
# Frame I/O
# ─────────────────────────────────────────────────────────────────


def decode_frame(b64_data: str) -> Optional[np.ndarray]:
    """Decode base64 image → BGR numpy array."""
    try:
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def encode_frame(frame: np.ndarray, quality: int = 78) -> str:
    """Encode BGR numpy array → data-URI base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


# ─────────────────────────────────────────────────────────────────
# IoU + containment helpers
# ─────────────────────────────────────────────────────────────────


def _iou(a: List[int], b: List[int]) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    areaB = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(areaA + areaB - inter)


def _center_in_box(box_small: List[int], box_large: List[int]) -> bool:
    """Return True if center of box_small lies inside box_large."""
    cx = (box_small[0] + box_small[2]) / 2
    cy = (box_small[1] + box_small[3]) / 2
    return box_large[0] <= cx <= box_large[2] and box_large[1] <= cy <= box_large[3]


def _belongs_to_person(person_box: List[int], item_box: List[int]) -> bool:
    """
    True if item_box can be assigned to person_box.
    Criteria (either):
      • IoU ≥ IOU_PERSON_PPE threshold
      • item center falls inside person box
    """
    if _iou(person_box, item_box) >= settings.IOU_PERSON_PPE:
        return True
    return _center_in_box(item_box, person_box)


# ─────────────────────────────────────────────────────────────────
# ppe.pt inference + raw detection parsing
# ─────────────────────────────────────────────────────────────────


def _run_inference_with(frame: np.ndarray, model, is_ppe: bool) -> List[Dict]:
    """
    Run YOLO inference with the specified model.
    Works with both ppe.pt (is_ppe=True) and standard COCO models.

    Returns flat list: {label, confidence, bbox:[x1,y1,x2,y2], det_type}
    """
    results = model(
        frame,
        verbose=False,
        conf=settings.DETECTION_CONF,
        iou=settings.NMS_IOU,
    )
    detections = []
    _raw_box_count = sum(len(r.boxes) for r in results)
    if _raw_box_count > 0:
        _raw_labels = [model.names[int(box.cls[0])] for r in results for box in r.boxes]
        print(f"[INFERENCE] raw_boxes={_raw_box_count} labels={_raw_labels}")
    for r in results:
        for box in r.boxes:
            if is_ppe:
                cls_idx = int(box.cls[0])
                label = (
                    PPE_CLASS_NAMES[cls_idx]
                    if cls_idx < len(PPE_CLASS_NAMES)
                    else "unknown"
                )
            else:
                label = model.names[int(box.cls[0])]

            conf = round(float(box.conf[0]), 3)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            if label in VIOLATION_CLASSES:
                det_type = "violation"
            elif label in COMPLIANT_CLASSES:
                det_type = "compliant"
            elif label in PERSON_CLASSES or label.lower() == "person":
                det_type = "person"
            else:
                det_type = "neutral"

            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "det_type": det_type,
                }
            )
    return detections


# ─────────────────────────────────────────────────────────────────
# Person–PPE association engine
# ─────────────────────────────────────────────────────────────────


def _associate_to_persons(
    persons: List[Dict],
    ppe_detections: List[Dict],
    role: str,
    detection_filters: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Assign violation/compliant PPE detections to their nearest person.
    For role='None': required violations come from detection_filters directly.
    For other roles: req_violations come from ROLE_RULES and the filter acts as a subset mask.
    """
    rules = ROLE_RULES.get(role, ROLE_RULES["Home"])
    req_violations = list(rules.get("required_violations", []))
    req_sim = list(rules.get("required_sim", []))

    # ── "None" role: derive requirements entirely from the user's selection ──
    if role == "None" and detection_filters:
        req_violations = [f for f in detection_filters if f not in SIM_ONLY_VIOLATIONS]
        req_sim = [f for f in detection_filters if f in SIM_ONLY_VIOLATIONS]

    enriched = []
    for person in persons:
        p_box = person["bbox"]

        assigned_violations: List[str] = []
        assigned_compliant: List[str] = []

        for det in ppe_detections:
            if not _belongs_to_person(p_box, det["bbox"]):
                continue
            if det["det_type"] == "violation":
                assigned_violations.append(det["label"])
            elif det["det_type"] == "compliant":
                assigned_compliant.append(det["label"])

        ppe_missing: List[str] = []
        violation_labels: List[str] = []

        # ── Native ppe.pt violations ───────────────────────────────
        # Two detection modes:
        #   1. ACTIVE violation: model detects "NO-Bakery-Head-Cap" → definitely missing
        #   2. ABSENCE violation: model detects a Person but sees NEITHER
        #      the compliant class (Bakery-Head-Cap) NOR the violation class
        #      (NO-Bakery-Head-Cap) → assume missing (model just didn't see it)
        #
        # Mode 2 prevents false negatives on webcam where the PPE model's
        # recall is low — if it can't see the cap at all, it's likely absent.
        ABSENCE_COMPLIANT_MAP = {
            "NO-Bakery-Head-Cap": "Bakery-Head-Cap",
            "NO-Hardhat": "Hardhat",
            "NO-Mask": "Mask",
            "NO-Safety Vest": "Safety Vest",
            "Bangles": None,  # no compliant counterpart — only flagged when detected
        }

        for req_v in req_violations:
            # For non-None roles: skip if this violation class not in active filter
            if (
                role != "None"
                and detection_filters is not None
                and req_v not in detection_filters
            ):
                continue

            # Mode 1: model explicitly detected the violation class
            if req_v in assigned_violations:
                human_label = VIOLATION_LABEL_MAP.get(req_v, req_v)
                ppe_missing.append(req_v)
                violation_labels.append(human_label)
                continue

            # Mode 2: absence detection — if the compliant counterpart is NOT seen
            # and the violation is NOT seen, the item is likely absent.
            # Skip for classes that have no compliant counterpart (e.g. Bangles).
            compliant_class = ABSENCE_COMPLIANT_MAP.get(req_v)
            if (
                compliant_class is not None
                and compliant_class not in assigned_compliant
            ):
                human_label = VIOLATION_LABEL_MAP.get(req_v, req_v)
                ppe_missing.append(req_v)
                violation_labels.append(human_label)

        # ── Simulated / non-native PPE (Gloves, Goggles, ID Card, Safety Shoes) ──
        for sim_v in req_sim:
            if (
                role != "None"
                and detection_filters is not None
                and sim_v not in detection_filters
            ):
                continue
            if _use_simulation:
                # Simulation mode: randomly assign violations
                if sim_v in assigned_violations:
                    human_label = VIOLATION_LABEL_MAP.get(sim_v, sim_v)
                    ppe_missing.append(sim_v)
                    violation_labels.append(human_label)
            # Real model: ppe_extended.pt handles these — if not loaded, skip (do NOT phantom-flag)
            # Extended model results are already merged into assigned_violations before this point

        enriched.append(
            {
                "bbox": p_box,
                "confidence": person["confidence"],
                "track_id": person.get(
                    "track_id", -1
                ),  # passthrough from _apply_tracking
                "ppe_found": list(assigned_compliant),
                "ppe_missing": ppe_missing,
                "assigned_violations": assigned_violations,
                "is_compliant": len(ppe_missing) == 0,
                "violation_labels": violation_labels,
            }
        )

    return enriched


# ─────────────────────────────────────────────────────────────────
# Simulation mode  (when ppe.pt not available)
# ─────────────────────────────────────────────────────────────────


def _simulate(
    frame: np.ndarray,
    role: str,
    detection_filters: Optional[List[str]] = None,
    frame_index: int = -1,
) -> List[Dict]:
    """
    Generate realistic simulated detections for when ppe.pt is unavailable.
    Produces 1–2 persons, each with randomised PPE status.
    Handles both native ppe.pt classes AND simulated classes (Gloves, Goggles).
    Respects detection_filters — only simulates selected PPE classes.
    frame_index >= 0: use deterministic RNG seeded per frame (for video upload)
    frame_index < 0:  use global random (for live feed — varied results)
    """
    # Deterministic RNG for stable per-frame results in video scanning
    _rng = random.Random(frame_index) if frame_index >= 0 else random
    h, w = frame.shape[:2]
    rules = ROLE_RULES.get(role, ROLE_RULES["Home"])
    req_violations = rules.get("required_violations", [])
    req_sim = rules.get("required_sim", [])
    all_req = req_violations + req_sim  # simulate ALL required PPE

    # Apply filter: only simulate selected violation classes
    if detection_filters is not None:
        all_req = [v for v in all_req if v in detection_filters]

    detections: List[Dict] = []
    num_persons = random.randint(1, 2)
    x_offsets = [0.08, 0.52]

    # Compliant class mapping for both native and simulated
    COMPLIANT_MAP = {
        "NO-Hardhat": "Hardhat",
        "NO-Mask": "Mask",
        "NO-Safety Vest": "Safety Vest",
        "NO-Gloves": "Gloves",
        "NO-Goggles": "Safety Goggles",
        "NO-ID Card": "ID Card",
        "NO-Uniform": "Uniform",
    }

    for i in range(num_persons):
        xo = x_offsets[i] if i < len(x_offsets) else 0.2
        px, py = int(w * xo), int(h * 0.04)
        px2, py2 = int(w * (xo + 0.35)), int(h * 0.95)
        px2, py2 = min(px2, w - 1), min(py2, h - 1)

        detections.append(
            {
                "label": "Person",
                "confidence": round(_rng.uniform(0.82, 0.97), 3),
                "bbox": [px, py, px2, py2],
                "det_type": "person",
            }
        )

        for req_v in all_req:
            is_violation = _rng.random() < 0.35  # 35% violation chance per item

            # Vertical zone: helmet/mask/goggles → upper 25%; gloves/vest → mid
            upper = "Hardhat" in req_v or "Mask" in req_v or "Goggles" in req_v
            if upper:
                iy1 = py + int((py2 - py) * 0.00)
                iy2 = py + int((py2 - py) * 0.25)
            elif "Gloves" in req_v:
                iy1 = py + int((py2 - py) * 0.50)
                iy2 = py + int((py2 - py) * 0.70)
            elif "Shoes" in req_v:
                iy1 = py + int((py2 - py) * 0.85)
                iy2 = py + int((py2 - py) * 1.00)
            else:
                # Vest, ID Card, Uniform, etc.
                iy1 = py + int((py2 - py) * 0.20)
                iy2 = py + int((py2 - py) * 0.60)

            ix1 = px + int((px2 - px) * 0.1)
            ix2 = px + int((px2 - px) * 0.9)

            if is_violation:
                label = req_v  # e.g. "NO-Gloves"
                det_type = "violation"
            else:
                label = COMPLIANT_MAP.get(req_v, "Hardhat")
                det_type = "compliant"

            detections.append(
                {
                    "label": label,
                    "confidence": round(_rng.uniform(0.65, 0.95), 3),
                    "bbox": [ix1, iy1, ix2, iy2],
                    "det_type": det_type,
                }
            )

    return detections


# ─────────────────────────────────────────────────────────────────
# Frame annotation
# ─────────────────────────────────────────────────────────────────


def _draw_results(
    frame: np.ndarray,
    enriched_persons: List[Dict],
    raw_detections: List[Dict],
    role: str,
) -> np.ndarray:
    """
    Annotate frame:
    • RED thick box + violation labels on violating persons.
    • GREEN thin box + ✓ on compliant persons.
    • Neutral/PPE item boxes (thin, dim) in background.
    • Top banner showing violation count.
    • Timestamp + role watermark.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    violations = [p for p in enriched_persons if not p["is_compliant"]]
    n_persons = len(enriched_persons)
    n_violations = len(violations)

    # ── Top banner ─────────────────────────────────────────────
    if n_violations > 0:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 46), (0, 0, 170), -1)
        cv2.addWeighted(overlay, 0.58, annotated, 0.42, 0, annotated)
        banner = f"  \u26a0  {n_violations}/{n_persons} PERSON(S) VIOLATING  \u2014  ROLE: {role.upper()}"
        cv2.putText(
            annotated,
            banner,
            (8, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 46), (0, 130, 0), -1)
        cv2.addWeighted(overlay, 0.40, annotated, 0.60, 0, annotated)
        cv2.putText(
            annotated,
            f"  \u2713  ALL {n_persons} PERSON(S) COMPLIANT  \u2014  ROLE: {role.upper()}",
            (8, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ── Draw neutral / PPE item boxes (thin, behind person boxes) ──
    for det in raw_detections:
        if det["det_type"] in ("neutral",):
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_NEUTRAL, 1)
            cv2.putText(
                annotated,
                f"{det['label']} {det['confidence']:.0%}",
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                COLOR_NEUTRAL,
                1,
                cv2.LINE_AA,
            )

    # ── Draw per-person boxes ───────────────────────────────────
    for person in enriched_persons:
        x1, y1, x2, y2 = person["bbox"]
        is_compliant = person["is_compliant"]

        if not is_compliant:
            color = COLOR_VIOLATION
            thick = 3

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

            # Confidence header
            conf_text = f"Person {person['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                annotated,
                conf_text,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Violation label pills (stacked below top edge)
            for idx, vlabel in enumerate(person["violation_labels"]):
                label_y = y1 + 26 + idx * 24
                (lw, lh), _ = cv2.getTextSize(vlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                cv2.rectangle(
                    annotated,
                    (x1 + 4, label_y - lh - 4),
                    (x1 + lw + 14, label_y + 5),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    annotated,
                    vlabel,
                    (x1 + 8, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (90, 90, 255),
                    2,
                    cv2.LINE_AA,
                )

        else:
            color = COLOR_COMPLIANT
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # Role-specific compliant label
            if role == "Traffic Police":
                ok_txt = f"\u2713 Helmet OK {person['confidence']:.0%}"
            elif person.get("ppe_found"):
                ok_txt = (
                    f"\u2713 {person['ppe_found'][0]} OK {person['confidence']:.0%}"
                )
            else:
                ok_txt = f"\u2713 Compliant {person['confidence']:.0%}"
            cv2.putText(
                annotated,
                ok_txt,
                (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                1,
                cv2.LINE_AA,
            )

    # ── Watermarks ─────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        annotated,
        ts,
        (w - 208, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "OccuSafe Monitor v4.0",
        (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (100, 200, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated


# ─────────────────────────────────────────────────────────────────
# Compliance summary
# ─────────────────────────────────────────────────────────────────


def _compliance_summary(
    enriched: List[Dict], role: str
) -> Tuple[bool, List[str], str, str]:
    """Aggregate per-person data → frame-level compliance."""
    rules = ROLE_RULES.get(role, ROLE_RULES["Home"])
    violations = [p for p in enriched if not p["is_compliant"]]

    if not violations:
        return True, [], "", "low"

    all_missing: List[str] = []
    for v in violations:
        for vl in v["violation_labels"]:
            if vl not in all_missing:
                all_missing.append(vl)

    n_v, n_p = len(violations), len(enriched)
    msg = (
        f"{rules['alert_prefix']}: " f"{n_v}/{n_p} person(s) — {', '.join(all_missing)}"
    )
    return False, all_missing, msg, rules["severity"]


# Traffic Police dedicated helmet pipelines
# ─────────────────────────────────────────────────────────────────


def _run_traffic_police_dedicated(frame: np.ndarray) -> Optional[List[Dict]]:
    """
    Run the keremberke/yolov8m-hard-hat-detection model.
    Each detection IS a person — the model outputs:
      'helmet' / 'hard hat' → person wearing helmet  → GREEN (compliant)
      'head'  / 'no helmet' → person without helmet  → RED  (violation)
    Returns None on error (caller will fall back to strict-ppe).
    """
    if _helmet_model is None:
        return None
    try:
        results = _helmet_model(frame, conf=0.28, iou=0.45, verbose=False)
    except Exception as e:
        log.error(f"[TrafficPolice] Helmet model inference error: {e}")
        return None

    enriched: List[Dict] = []
    for r in results:
        for box in r.boxes:
            cls_name = _helmet_model.names.get(int(box.cls[0]), "").lower()
            conf = round(float(box.conf[0]), 3)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            is_helmet = cls_name in _HELMET_COMPLIANT_NAMES
            enriched.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "ppe_found": ["Helmet"] if is_helmet else [],
                    "ppe_missing": [] if is_helmet else ["NO-Hardhat"],
                    "assigned_violations": [] if is_helmet else ["NO-Hardhat"],
                    "is_compliant": is_helmet,
                    "violation_labels": [] if is_helmet else ["No Helmet"],
                }
            )
    return enriched


def _run_traffic_police_strict_ppe(
    frame: np.ndarray,
    frame_index: int = -1,
) -> List[Dict]:
    """
    Fallback when dedicated helmet model is not available.
    Runs ppe.pt (or simulation) with STRICT logic:
      - Person with 'Hardhat' detected → GREEN (compliant)
      - Person with 'NO-Hardhat' detected OR no helmet detected at all → RED (violation)
    This fixes the false-green bug where undetected helmets passed as compliant.
    """
    if _use_simulation or _model is None:
        raw = _simulate(frame, "Traffic Police", frame_index=frame_index)
    else:
        try:
            raw = _run_inference_with(frame, _model, _model_is_ppe)
        except Exception as e:
            log.error(f"[TrafficPolice strict] inference error: {e}")
            raw = _simulate(frame, "Traffic Police", frame_index=frame_index)

    persons = [d for d in raw if d["det_type"] == "person"]
    ppe_dets = [d for d in raw if d["det_type"] in ("violation", "compliant")]

    # Fallback: create one synthetic person spanning all PPE detections if no Person class
    if not persons and ppe_dets:
        xs = [d["bbox"][0] for d in ppe_dets] + [d["bbox"][2] for d in ppe_dets]
        ys = [d["bbox"][1] for d in ppe_dets] + [d["bbox"][3] for d in ppe_dets]
        h, w = frame.shape[:2]
        persons = [
            {
                "label": "Person",
                "confidence": 0.82,
                "bbox": [
                    max(0, min(xs) - 30),
                    max(0, min(ys) - 30),
                    min(w - 1, max(xs) + 30),
                    min(h - 1, max(ys) + 30),
                ],
                "det_type": "person",
            }
        ]

    enriched: List[Dict] = []
    for person in persons:
        p_box = person["bbox"]
        # Find all PPE items belonging to this person
        assigned_c = [
            d["label"]
            for d in ppe_dets
            if d["det_type"] == "compliant" and _belongs_to_person(p_box, d["bbox"])
        ]
        assigned_v = [
            d["label"]
            for d in ppe_dets
            if d["det_type"] == "violation" and _belongs_to_person(p_box, d["bbox"])
        ]

        # STRICT: helmet OK only when ppe.pt EXPLICITLY detects 'Hardhat'
        # If no helmet class detected at all (neither Hardhat nor NO-Hardhat), treat as violation
        hardhat_ok = "Hardhat" in assigned_c
        no_hardhat = "NO-Hardhat" in assigned_v or not hardhat_ok

        is_compliant = hardhat_ok and not no_hardhat
        enriched.append(
            {
                "bbox": p_box,
                "confidence": person["confidence"],
                "ppe_found": ["Helmet"] if hardhat_ok else [],
                "ppe_missing": [] if is_compliant else ["NO-Hardhat"],
                "assigned_violations": assigned_v,
                "is_compliant": is_compliant,
                "violation_labels": [] if is_compliant else ["No Helmet"],
            }
        )
    return enriched


# ─────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────


def _run_pipeline(
    frame: np.ndarray,
    role: str,
    detection_filters: Optional[List[str]] = None,
    no_phone_zone: bool = False,
    frame_index: int = -1,
    # ── OCR gate context (all optional) ──
    ocr_zone_config: Optional[Dict] = None,
    ocr_direction: Optional[str] = None,
    ocr_db_session=None,
    ocr_camera_id: Optional[int] = None,
    ocr_user_id: Optional[int] = None,
    # ── ByteTrack camera context (optional) ───────────────────
    # Pass camera_id to enable per-camera stable track_id.
    # None = skip tracking (track_id=-1 on all persons).
    camera_id: Optional[int] = None,
) -> Dict:
    """
    Shared pipeline used by both process_frame and process_frame_numpy.
    detection_filters: if provided, only these violation classes are evaluated.
    no_phone_zone: if True, any phone detection triggers an alert.

    OCR gate kwargs (ocr_*) are all optional.  When ocr_zone_config is None
    the OCR block is skipped with zero overhead.  Supply them only from callers
    that are processing an entrance-zone camera feed.

    camera_id: when provided, ByteTrack is applied to Person detections using a
    per-camera tracker instance.  This assigns a stable track_id to each person
    dict so downstream services (idle_service) can maintain per-track state.
    """
    # Expose OCR context to the block at the end of this function via local vars.
    _ocr_zone_config = ocr_zone_config
    _ocr_direction = ocr_direction
    _ocr_db_session = ocr_db_session
    _ocr_camera_id = ocr_camera_id
    _ocr_user_id = ocr_user_id

    print(
        f"[PIPELINE] role={role} | sim={_use_simulation} | ded_helmet={_helmet_model_dedicated} | filters={detection_filters}"
    )

    # ── Camera Blockage / Tampering Check ────────────────────────────────
    if frame is not None and frame.size > 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean, stddev = cv2.meanStdDev(gray)
        # stddev < 5 means almost no variation (e.g. covered by hand, totally dark)
        if stddev[0][0] < 5.0:
            msg = "⚠️ Camera Blocked or Covered!"
            if mean[0][0] < 15.0:
                msg = "⚠️ Camera Signal Lost (Completely Black)!"

            ann_frame = frame.copy()
            cv2.rectangle(ann_frame, (0, 0), (ann_frame.shape[1], 80), (0, 0, 200), -1)
            cv2.putText(
                ann_frame,
                msg,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            return {
                "persons": [],
                "violations": [],
                "detections": [],
                "is_compliant": False,
                "missing_items": ["Camera Tampering"],
                "violations_count": 1,
                "persons_count": 0,
                "alert_message": msg,
                "severity": "critical",
                "annotated_frame": encode_frame(ann_frame),
                "snapshot_b64": encode_frame(frame),
                "phone_status": "safe",
                "phone_detected": False,
                "model_mode": "tamper_detection",
                "phone_severity": "low",
            }

    # ── Traffic Police: use dedicated helmet pipeline ─────────────────
    if role == "Traffic Police":
        if _helmet_model_dedicated:
            # Try the dedicated keremberke model first
            enriched = _run_traffic_police_dedicated(frame)
            if enriched is None:
                # Model failed — fall back
                enriched = _run_traffic_police_strict_ppe(
                    frame, frame_index=frame_index
                )
        else:
            # Use ppe.pt with strict logic
            enriched = _run_traffic_police_strict_ppe(frame, frame_index=frame_index)

        is_compliant, missing, alert_msg, severity = _compliance_summary(
            enriched, "Traffic Police"
        )
        annotated = _draw_results(frame, enriched, [], "Traffic Police")
        ann_b64 = encode_frame(annotated)
        snap_b64 = encode_frame(annotated, quality=85) if not is_compliant else None
        violations = [p for p in enriched if not p["is_compliant"]]
        ui_detections = [
            {
                "label": ("Helmet" if p["is_compliant"] else "No Helmet"),
                "confidence": p["confidence"],
                "bbox": p["bbox"],
            }
            for p in enriched
        ]

        # Phone detection still runs for Traffic Police
        from services import phone_service as _phone_svc

        phone_result = _phone_svc.detect_phone_usage(
            frame=annotated,
            no_phone_zone=no_phone_zone,
            coco_model=_phone_model,
            coco_fallback=None,  # helmet model not suitable for phone detection
            use_simulation=_use_simulation,
        )
        annotated_with_phone = phone_result["annotated_frame"]
        ann_b64 = encode_frame(annotated_with_phone)
        snap_b64 = (
            encode_frame(annotated_with_phone, quality=85)
            if (not is_compliant or phone_result["phone_alert"])
            else None
        )
        phone_alert = phone_result.get("phone_alert")
        phone_status = phone_result.get("phone_status", "safe")
        phone_dets = phone_result.get("phone_detections", [])
        if phone_alert:
            is_compliant = False
            alert_msg = f"{alert_msg} | {phone_alert}" if alert_msg else phone_alert
            if not severity or severity == "low":
                severity = phone_result.get("phone_severity", "high")

        return {
            "persons": enriched,
            "violations": violations,
            "detections": ui_detections + phone_dets,
            "is_compliant": is_compliant,
            "missing_items": missing,
            "violations_count": len(violations),
            "persons_count": len(enriched),
            "alert_message": alert_msg if not is_compliant else None,
            "severity": severity if not is_compliant else None,
            "annotated_frame": ann_b64,
            "snapshot_b64": snap_b64,
            "phone_status": phone_status,
            "phone_detected": phone_result.get("phone_detected", False),
            "model_mode": (
                "helmet:keremberke"
                if _helmet_model_dedicated
                else "helmet:ppe.pt(strict)"
            ),
            "phone_severity": phone_result.get("phone_severity", "low"),
        }

    # ── All other roles: general PPE pipeline ───────────────────────
    print(
        f"[PIPELINE] role={role} | sim={_use_simulation} | filters={detection_filters}"
    )
    active_model, active_is_ppe = _get_model_for_role(role)

    # Mapping from violation class → its compliant counterpart (model class names)
    VIOLATION_TO_COMPLIANT = {
        "NO-Hardhat": "Hardhat",
        "NO-Mask": "Mask",
        "NO-Safety Vest": "Safety Vest",
        "NO-Bakery-Head-Cap": "Bakery-Head-Cap",  # bakery cap compliant class
        # Bangles has no "compliant" counterpart — it is always a violation
        # simulated classes have no real compliant class in ppe.pt
    }

    if _use_simulation or active_model is None:
        raw = _simulate(
            frame, role, detection_filters=detection_filters, frame_index=frame_index
        )
    else:
        try:
            raw = _run_inference_with(frame, active_model, active_is_ppe)
        except Exception as e:
            log.error(f"Inference error (role={role}): {e}")
            raw = _simulate(
                frame,
                role,
                detection_filters=detection_filters,
                frame_index=frame_index,
            )

    # Strictly filter raw detections to only keep selected classes
    before_filter = [d["label"] for d in raw]
    if detection_filters is not None and len(detection_filters) > 0:
        # Build the set of compliant labels that correspond to selected violation filters
        allowed_compliant = {
            VIOLATION_TO_COMPLIANT[v]
            for v in detection_filters
            if v in VIOLATION_TO_COMPLIANT
        }
        raw = [
            d
            for d in raw
            if d["det_type"] == "person"  # always keep persons
            or d["det_type"] == "neutral"  # always keep neutral
            or d["label"] in detection_filters  # keep selected violations
            or d["label"] in allowed_compliant  # keep corresponding compliant
        ]
    after_filter = [d["label"] for d in raw]
    print(
        f"[FILTER] before={before_filter} | after={after_filter} | filters={detection_filters}"
    )

    persons = [d for d in raw if d["det_type"] == "person"]
    ppe_dets = [d for d in raw if d["det_type"] in ("violation", "compliant")]

    # ── ByteTrack: assign stable track_id to each person (no-op if camera_id is None) ──
    if camera_id is not None:
        persons = _apply_tracking(camera_id, persons)

    if not persons and ppe_dets:
        xs = [d["bbox"][0] for d in ppe_dets] + [d["bbox"][2] for d in ppe_dets]
        ys = [d["bbox"][1] for d in ppe_dets] + [d["bbox"][3] for d in ppe_dets]
        h, w = frame.shape[:2]
        persons = [
            {
                "label": "Person",
                "confidence": max(d["confidence"] for d in ppe_dets),
                "bbox": [
                    max(0, min(xs) - 30),
                    max(0, min(ys) - 30),
                    min(w - 1, max(xs) + 30),
                    min(h - 1, max(ys) + 30),
                ],
                "det_type": "person",
            }
        ]

    # ── COCO Person fallback ──────────────────────────────────────────────
    # The PPE model (ppe_factory_v0.pt, 14 classes) often fails to detect
    # Person on webcam/laptop cameras.  The phone model (yolov8x.pt, COCO)
    # reliably detects them.  If PPE model found zero persons, run a quick
    # Person-only pass with the COCO model to inject person bounding boxes
    # so violations/compliance can still be evaluated.
    if not persons and _phone_model is not None:
        try:
            _coco_results = _phone_model(
                frame, verbose=False, conf=0.35, iou=0.45, classes=[0]
            )  # class 0 = person in COCO
            for _cr in _coco_results:
                for _cb in _cr.boxes:
                    _cx1, _cy1, _cx2, _cy2 = [int(v) for v in _cb.xyxy[0]]
                    persons.append(
                        {
                            "label": "Person",
                            "confidence": round(float(_cb.conf[0]), 3),
                            "bbox": [_cx1, _cy1, _cx2, _cy2],
                            "det_type": "person",
                        }
                    )
            if persons:
                print(
                    f"[COCO FALLBACK] Injected {len(persons)} person(s) from yolov8x.pt"
                )
        except Exception as _coco_err:
            log.warning(f"[COCO FALLBACK] person detection failed: {_coco_err}")

    enriched = _associate_to_persons(
        persons, ppe_dets, role, detection_filters=detection_filters
    )

    # ── Traffic Police: remap already done inside dedicated/strict pipeline ──
    # (no remap needed here anymore)

    is_compliant, missing, alert_msg, severity = _compliance_summary(enriched, role)

    annotated = _draw_results(frame, enriched, raw, role)
    ann_b64 = encode_frame(annotated)
    snap_b64 = encode_frame(annotated, quality=85) if not is_compliant else None

    violations = [p for p in enriched if not p["is_compliant"]]
    ui_detections = [
        {"label": d["label"], "confidence": d["confidence"], "bbox": d["bbox"]}
        for d in raw
    ]

    # ── Phone usage detection (parallel pipeline) ─────────────
    # Lazy import avoids circular import at module load time
    from services import phone_service as _phone_svc

    phone_result = _phone_svc.detect_phone_usage(
        frame=annotated,
        no_phone_zone=no_phone_zone,
        # Dedicated large COCO model for high-accuracy phone detection
        coco_model=_phone_model,
        # Secondary fallback: yolov8m.pt helmet model (also COCO)
        coco_fallback=(_helmet_model or (_model if not _model_is_ppe else None)),
        use_simulation=_use_simulation,
    )

    # Merge phone annotations onto the annotated frame
    annotated_with_phone = phone_result["annotated_frame"]
    ann_b64 = encode_frame(annotated_with_phone)
    snap_b64 = (
        encode_frame(annotated_with_phone, quality=85)
        if (not is_compliant or phone_result["phone_alert"])
        else None
    )

    # Merge phone alert into overall compliance
    phone_alert = phone_result.get("phone_alert")
    phone_status = phone_result.get("phone_status", "safe")
    phone_dets = phone_result.get("phone_detections", [])

    if phone_alert:
        is_compliant = False
        if alert_msg:
            alert_msg = f"{alert_msg} | {phone_alert}"
        else:
            alert_msg = phone_alert
        if not severity or severity == "low":
            severity = phone_result.get("phone_severity", "high")

    # ── OCR gate: Document-in-hand (class 13) at entrance zone ────────────────
    # Triggered only when the detection pipeline produces at least one
    # "Document-in-hand" bounding box AND the caller has supplied a zone_config
    # dict that includes an "entrance" polygon (via kwarg; see below).
    #
    # ⚠️  HARDWARE DISCLAIMER: calling ocr_service does NOT open/close a physical
    #     door or turnstile.  There is no hardware actuator integrated here.
    #     approved=True  → document saved to DB (InvoiceLog / OrderFormLog).
    #     approved=False → high-severity REVIEW alert saved via alert_service.
    #     A human operator must review alerts and take physical action.
    #     Do not assume or imply hardware integration at any point in this code.
    _doc_dets = [d for d in raw if d["label"] == "Document-in-hand"]
    if _doc_dets and _ocr_zone_config is not None:
        _entrance_poly = _ocr_zone_config.get("entrance")
        for _doc in _doc_dets:
            _cx = (_doc["bbox"][0] + _doc["bbox"][2]) / 2
            _cy = (_doc["bbox"][1] + _doc["bbox"][3]) / 2
            # No polygon = always trigger (entrance not yet calibrated).
            # Uses cv2.pointPolygonTest via zone_service — no shapely dependency.
            from services import zone_service as _zs

            _in_zone = _entrance_poly is None or _zs.point_in_zone(
                _cx, _cy, _entrance_poly
            )
            if not _in_zone:
                continue

            # ── Run OCR ────────────────────────────────────────────────────
            try:
                from services import ocr_service as _ocr

                _direction = _ocr_direction or "inward"
                _ocr_result = _ocr.scan_document_in_frame(
                    frame=frame,
                    bbox=_doc["bbox"],
                    direction=_direction,
                )
            except Exception as _oe:
                log.error(f"[OCR] scan_document_in_frame error: {_oe}")
                continue

            # Always log raw OCR text — even on approval — so admin can audit
            log.info(
                f"[OCR] direction={_ocr_result['direction']} "
                f"approved={_ocr_result['approved']} "
                f"raw_text={_ocr_result['raw_text'][:80]!r}"
            )

            if _ocr_db_session is not None:
                if _ocr_result["approved"]:
                    # ── Save to InvoiceLog / OrderFormLog ──────────────────
                    try:
                        from database import InvoiceLog, OrderFormLog

                        _ts_str = _ocr_result.get("timestamp", "")
                        import datetime as _dt

                        try:
                            _ts = _dt.datetime.fromisoformat(_ts_str.rstrip("Z"))
                        except Exception:
                            _ts = _dt.datetime.utcnow()

                        if _ocr_result["direction"] == "inward":
                            _log_row = InvoiceLog(
                                camera_id=_ocr_camera_id,
                                direction="inward",
                                raw_ocr_text=_ocr_result["raw_text"],
                                approved=True,
                                snapshot_b64=_ocr_result.get("snapshot_b64"),
                                ocr_available=_ocr_result.get("ocr_available", True),
                                timestamp=_ts,
                            )
                        else:
                            _log_row = OrderFormLog(
                                camera_id=_ocr_camera_id,
                                direction="outward",
                                raw_ocr_text=_ocr_result["raw_text"],
                                approved=True,
                                snapshot_b64=_ocr_result.get("snapshot_b64"),
                                ocr_available=_ocr_result.get("ocr_available", True),
                                timestamp=_ts,
                            )
                        _ocr_db_session.add(_log_row)
                        _ocr_db_session.commit()
                        log.info(
                            f"[OCR] {'InvoiceLog' if _ocr_result['direction'] == 'inward' else 'OrderFormLog'} "
                            f"saved (id={_log_row.id})"
                        )
                    except Exception as _dbe:
                        log.error(f"[OCR] DB save error: {_dbe}")

                else:
                    # ── Fire REVIEW alert ───────────────────────────────────
                    # This is a COMPLIANCE LOG alert — not a hardware gate trigger.
                    # A human operator must review this alert and decide on access.
                    try:
                        from services.alert_service import save_alert

                        _ocr_msg = (
                            f"[REVIEW — OCR GATE] "
                            f"{'Inward' if _ocr_result['direction'] == 'inward' else 'Outward'} "
                            f"document at entrance did NOT match expected "
                            f"{'invoice' if _ocr_result['direction'] == 'inward' else 'order-form'} "
                            f"pattern. Raw OCR: {_ocr_result['raw_text'][:120]!r}. "
                            f"ACTION REQUIRED: human admin must verify document manually. "
                            f"⚠️ No hardware gate is connected — this is a logged compliance check only."
                        )
                        save_alert(
                            db=_ocr_db_session,
                            user_id=_ocr_user_id or 0,
                            message=_ocr_msg,
                            role="Factory Worker",
                            severity="high",
                            detected_issue="Invalid/unrecognised document at entrance",
                            confidence=None,
                            snapshot_b64=_ocr_result.get("snapshot_b64"),
                        )
                    except Exception as _ae:
                        log.error(f"[OCR] Alert save error: {_ae}")

    # ── Cashbox zone gate ─────────────────────────────────────────────────────
    # If a "cashbox" zone is configured, mark which enriched persons are inside.
    # Gate is on person centroid (wrist keypoints = Phase 4 pose model).
    # ⚠️  No hardware connected — compliance log + alert only.
    if _ocr_zone_config:
        from services import zone_service as _zs_gate

        _cashbox_poly = _ocr_zone_config.get("cashbox")
        if _cashbox_poly:
            for _ep in enriched:
                _pcx, _pcy = _zs_gate.bbox_center(_ep["bbox"])
                _ep["near_cashbox"] = _zs_gate.point_in_zone(_pcx, _pcy, _cashbox_poly)

    # ── Window zone gate (Phase 4 stub) ───────────────────────────────────────
    # Window-throw trajectory detection inserted here in Phase 4.
    # Zone name: "window". Trigger: object centroid exits through window polygon.
    if _ocr_zone_config and _ocr_zone_config.get("window"):
        log.debug(
            "[zone_gate] 'window' zone configured — trajectory detection is Phase 4."
        )

    return {
        "persons": enriched,
        "violations": violations,
        "detections": ui_detections + phone_dets,
        "is_compliant": is_compliant,
        "missing_items": missing,
        "violations_count": len(violations),
        "persons_count": len(enriched),
        "alert_message": alert_msg if not is_compliant else None,
        "severity": severity if not is_compliant else None,
        "annotated_frame": ann_b64,
        "snapshot_b64": snap_b64,
        "phone_status": phone_status,
        "phone_detected": phone_result.get("phone_detected", False),
        "model_mode": (
            "ppe.pt"
            if _model_is_ppe
            else ("simulation" if _use_simulation else settings.YOLO_MODEL)
        ),
        "phone_severity": phone_result.get("phone_severity", "low"),
    }


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────


def process_frame(
    b64_frame: str,
    role: str,
    detection_filters: Optional[List[str]] = None,
    no_phone_zone: bool = False,
) -> Dict:
    """
    Full violation pipeline from base64 frame.
    Called by the WebSocket detection router.
    """
    frame = decode_frame(b64_frame)
    if frame is None:
        return {"error": "Invalid frame data"}
    return _run_pipeline(
        frame, role, detection_filters=detection_filters, no_phone_zone=no_phone_zone
    )


def process_frame_numpy(
    frame: np.ndarray,
    role: str,
    detection_filters: Optional[List[str]] = None,
    no_phone_zone: bool = False,
    frame_index: int = -1,
) -> Dict:
    """
    Full violation pipeline from a numpy frame directly.
    Called by the video upload router (avoids double encode/decode).
    frame_index: passed through to _simulate for deterministic results in video.
    """
    if frame is None:
        return {"error": "Invalid frame"}
    return _run_pipeline(
        frame,
        role,
        detection_filters=detection_filters,
        no_phone_zone=no_phone_zone,
        frame_index=frame_index,
    )


def process_frame_numpy_tracked(
    frame: np.ndarray,
    role: str,
    camera_id: int,
    detection_filters: Optional[List[str]] = None,
    no_phone_zone: bool = False,
) -> Dict:
    """
    Full violation pipeline with per-camera ByteTrack tracking.

    Called by the camera_manager detection daemon for server-side inference
    that runs independently of browser WebSocket connections.

    Differences from process_frame_numpy:
    • camera_id is required — each camera has its own ByteTrack instance so
      track IDs are stable within a camera but never mixed across cameras.
    • Each person dict in result["persons"] includes "track_id" (int).
      track_id == -1 means supervision is not installed or tracking was skipped.
    • All PPE logic (_associate_to_persons, _draw_results) is unchanged.

    The caller (camera_manager) is responsible for passing result["persons"]
    to idle_service.process_frame() after this returns.
    """
    if frame is None:
        return {"error": "Invalid frame"}
    return _run_pipeline(
        frame,
        role,
        detection_filters=detection_filters,
        no_phone_zone=no_phone_zone,
        camera_id=camera_id,
    )


# =============================================================================
# OCR gate — called directly from camera_manager._detection_loop (REQ-001/004)
# =============================================================================


def run_ocr_gate_for_camera(
    frame,
    raw_dets: list,
    zone_config: dict,
    db_session,
    camera_id: int,
    direction: str = "inward",
) -> None:
    """
    Standalone OCR gate called by camera_manager._detection_loop.

    Checks whether any 'Document-in-hand' (class 13) detection is inside the
    configured 'entrance' zone polygon, then runs OCR and either:
      - Saves an InvoiceLog (inward) or OrderFormLog (outward) on approval.
      - Fires a high-severity REVIEW alert if the document doesn't match.

    No hardware gate is connected — compliance log only.
    """
    doc_dets = [d for d in raw_dets if d.get("label") == "Document-in-hand"]
    if not doc_dets or not zone_config:
        return

    entrance_poly = zone_config.get("entrance")
    from services import zone_service as _zs

    for doc in doc_dets:
        cx = (doc["bbox"][0] + doc["bbox"][2]) / 2
        cy = (doc["bbox"][1] + doc["bbox"][3]) / 2

        in_zone = entrance_poly is None or _zs.point_in_zone(cx, cy, entrance_poly)
        if not in_zone:
            continue

        try:
            from services import ocr_service as _ocr

            ocr_result = _ocr.scan_document_in_frame(
                frame=frame,
                bbox=doc["bbox"],
                direction=direction,
            )
        except Exception as oe:
            log.error(f"[OCR] scan_document_in_frame error: {oe}")
            continue

        log.info(
            f"[OCR] direction={ocr_result['direction']} "
            f"approved={ocr_result['approved']} "
            f"raw_text={ocr_result['raw_text'][:80]!r}"
        )

        if db_session is None:
            continue

        if ocr_result["approved"]:
            try:
                import datetime as _dt

                from database import InvoiceLog, OrderFormLog

                ts_str = ocr_result.get("timestamp", "")
                try:
                    ts = _dt.datetime.fromisoformat(ts_str.rstrip("Z"))
                except Exception:
                    ts = _dt.datetime.utcnow()

                if ocr_result["direction"] == "inward":
                    log_row = InvoiceLog(
                        camera_id=camera_id,
                        direction="inward",
                        raw_ocr_text=ocr_result["raw_text"],
                        approved=True,
                        snapshot_b64=ocr_result.get("snapshot_b64"),
                        ocr_available=ocr_result.get("ocr_available", True),
                        timestamp=ts,
                    )
                else:
                    log_row = OrderFormLog(
                        camera_id=camera_id,
                        direction="outward",
                        raw_ocr_text=ocr_result["raw_text"],
                        approved=True,
                        snapshot_b64=ocr_result.get("snapshot_b64"),
                        ocr_available=ocr_result.get("ocr_available", True),
                        timestamp=ts,
                    )
                db_session.add(log_row)
                db_session.commit()
                log.info(
                    f"[OCR] {'InvoiceLog' if ocr_result['direction'] == 'inward' else 'OrderFormLog'} saved cam={camera_id}"
                )
            except Exception as dbe:
                log.error(f"[OCR] DB save error: {dbe}")
        else:
            try:
                from services.alert_service import save_alert
                from services.rule_engine import _get_rule_engine_user_id

                uid = _get_rule_engine_user_id(db_session)
                ocr_msg = (
                    f"[REVIEW - OCR GATE] "
                    f"{'Inward' if ocr_result['direction'] == 'inward' else 'Outward'} "
                    f"document at entrance did NOT match expected pattern. "
                    f"Raw OCR: {ocr_result['raw_text'][:120]!r}. "
                    f"ACTION REQUIRED: verify document manually. "
                    f"No hardware gate connected - compliance check only."
                )
                save_alert(
                    db=db_session,
                    user_id=uid,
                    message=ocr_msg,
                    role="Factory Worker",
                    severity="high",
                    detected_issue="Invalid/unrecognised document at entrance",
                    confidence=None,
                    snapshot_b64=ocr_result.get("snapshot_b64"),
                    camera_id=camera_id,
                )
            except Exception as ae:
                log.error(f"[OCR] Alert save error: {ae}")
