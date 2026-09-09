"""
YOLO Violation Detection Service — v4.0 (Multi-Model Architecture)
==================================================================
Primary Person & Object Model: yolov8x.pt (COCO standard, 80 classes)
Specialized Models: portable_models_package/ (hairnet, fall, cash, anomaly)
Legacy ppe_factory_v0.pt: DISABLED / REMOVED

Flow per frame:
  1. Detect persons with yolov8x.pt.
  2. Multi-model injection runs specialized detectors (hairnet_glove_detection, fall, etc.).
  3. Associate head coverings / PPE to detected persons via IoU & head region geometry.
  4. Evaluate role-specific compliance rules (Bakery Worker: Bakery-Head-Cap, Bangles, etc.).
  5. Annotate frame and return detection payload.
"""

import base64
import datetime
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from config import settings

log = logging.getLogger("yolo_service")

# ─────────────────────────────────────────────────────────────────
# Class map shared across the model chain (index → name, as trained)
# ppe_factory_v0.pt adds classes 10-16 on top of the base ppe.pt set
# ─────────────────────────────────────────────────────────────────
PPE_CLASS_NAMES = [
    "Hardhat",           # 0  ✅ compliant
    "Mask",              # 1  ✅ compliant
    "NO-Hardhat",        # 2  ❌ violation
    "NO-Mask",           # 3  ❌ violation
    "NO-Safety Vest",    # 4  ❌ violation
    "Person",            # 5  👤 neutral person
    "Safety Cone",       # 6  🟠 neutral
    "Safety Vest",       # 7  ✅ compliant
    "machinery",         # 8  🔵 neutral
    "vehicle",           # 9  🔵 neutral
    # ── Phase 1 new classes (ppe_factory_v0_cash.pt) ───────────────────
    "Bakery-Head-Cap",   # 10 ✅ compliant (cloth cap worn correctly)
    "NO-Bakery-Head-Cap",# 11 ❌ violation  (cap absent / wrong)
    "Bangles",           # 12 ❌ violation  (always flagged in food production)
    "Document-in-hand",  # 13 🔵 neutral  (triggers OCR pipeline at entrance)
    "Cash",              # 14 💵 neutral  (cash monitoring — CashEventTracker)
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
    "Cashbox",
    "Cash",          # Phase 1 cash monitoring — passed through to CashEventTracker
}

# Human-readable violation → missing item label
# Original ppe.pt violation classes
VIOLATION_LABEL_MAP = {
    "NO-Hardhat": "No Hardhat",
    "NO-Mask": "No Mask",
    "NO-Safety Vest": "No Safety Vest",
    # Phase 1 — bakery-specific violations
    "NO-Bakery-Head-Cap": "No Head Cap",
    "NO-Hairnet": "No Hairnet",
    "no_hairnet": "No Hairnet",
    "Bangles": "Hand Item / Bangles Worn (Violation)",
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
            "NO-Bakery-Head-Cap",  # cap absent / loose hair
            "Bangles",             # jewellery — always flagged in food production
        ],
        "required_compliant": [
            "Bakery-Head-Cap",  # hair covered
        ],
        "required_sim": ["NO-Uniform"],
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
        "required_sim": ["NO-Uniform"],
        "severity": "critical",
        "alert_prefix": "🏭 Factory safety violation",
    },
}
ROLE_RULES["Bakery"] = ROLE_RULES["Bakery Worker"]


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

        # Load primary model from settings.YOLO_MODEL (default: yolov8x.pt)
        _candidate = settings.YOLO_MODEL
        _loaded = False
        _backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        _paths_to_try = [
            _candidate,
            os.path.join(_backend_dir, _candidate),
            os.path.join(_backend_dir, "..", _candidate),
            os.path.join(_backend_dir, "..", "training", "models", _candidate),
        ]
        for _p in _paths_to_try:
            if os.path.isfile(_p) or not _p.endswith(".pt"):
                try:
                    # BUG FIX: Explicitly declare global here to ensure assignment works
                    global _model, _use_simulation, _model_is_ppe
                    _model = YOLO(_p)
                    torch.load = _orig_load
                    _use_simulation = False
                    _model_is_ppe = "ppe" in _candidate.lower()
                    _nc = len(_model.names)
                    log.info(f"✅ Primary YOLO model loaded: {_candidate} ({_nc} classes)")
                    print(f"✅ Primary YOLO model loaded: {_candidate} ({_nc} classes)")
                    _loaded = True
                    break
                except Exception as e:
                    log.warning(f"{_p} load failed: {e}")
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

    # ── Warm-up: run one blank inference to trigger JIT compilation ────────
    # Without this, the FIRST live frame pays the PyTorch JIT cost (~2-5s).
    # A tiny black 640×640 frame is enough to trigger compilation with zero
    # visual impact.  Runs synchronously here so it completes before the
    # first WebSocket connection arrives.
    if _model is not None and not _use_simulation:
        try:
            import numpy as _np
            _warmup_frame = _np.zeros((640, 640, 3), dtype=_np.uint8)
            list(_model(_warmup_frame, verbose=False, imgsz=640, stream=True))
            log.info("[YOLO] Warm-up inference complete")
            print("ℹ️  YOLO warm-up complete (first frame JIT cost eliminated)")
        except Exception as _we:
            log.warning(f"[YOLO] Warm-up skipped: {_we}")

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
                _helmet_model = None
                _helmet_model_dedicated = False
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


def _deduplicate_person_boxes(
    persons: List[Dict], iou_thresh: float = 0.35, contain_thresh: float = 0.65
) -> List[Dict]:
    """
    Deduplicate overlapping person bounding boxes.
    Handles duplicate detections from multiple models (e.g. YOLO PPE + YOLO COCO)
    or multi-scale person boxes (torso box vs full body box for the same human).
    """
    if len(persons) <= 1:
        return persons

    # Sort by confidence descending
    sorted_p = sorted(persons, key=lambda p: p.get("confidence", 0.0), reverse=True)
    kept: List[Dict] = []

    for p in sorted_p:
        box1 = p["bbox"]
        area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        is_dup = False

        for k in kept:
            box2 = k["bbox"]
            area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))

            xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
            xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
            inter = max(0, xB - xA) * max(0, yB - yA)

            if inter > 0:
                union = area1 + area2 - inter
                iou = inter / float(union) if union > 0 else 0.0
                containment = inter / float(min(area1, area2))

                if iou > iou_thresh or containment > contain_thresh:
                    is_dup = True
                    break

        if not is_dup:
            kept.append(p)

    return kept


def _deduplicate_headwear_boxes(
    detections: List[Dict], iou_thresh: float = 0.30
) -> List[Dict]:
    """
    Deduplicate overlapping headwear/hairnet detections.
    Ensures only the highest confidence head detection survives for each person head.
    """
    head_dets: List[Dict] = []
    other_dets: List[Dict] = []

    for d in detections:
        lbl_low = str(d.get("label", "")).lower()
        raw_low = str(d.get("raw_label", "")).lower()
        if (
            "hair" in lbl_low
            or "head-cap" in lbl_low
            or "head cap" in lbl_low
            or "hairnet" in raw_low
        ):
            head_dets.append(d)
        else:
            other_dets.append(d)

    if len(head_dets) <= 1:
        return detections

    sorted_h = sorted(head_dets, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept_h: List[Dict] = []

    for d in sorted_h:
        box1 = d["bbox"]
        area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        is_dup = False

        for k in kept_h:
            box2 = k["bbox"]
            area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))

            xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
            xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
            inter = max(0, xB - xA) * max(0, yB - yA)

            if inter > 0:
                union = area1 + area2 - inter
                iou = inter / float(union) if union > 0 else 0.0
                containment = inter / float(min(area1, area2))
                if iou > iou_thresh or containment > 0.50:
                    is_dup = True
                    break

        if not is_dup:
            kept_h.append(d)

    return other_dets + kept_h


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


def _head_belongs_to_person(person_box: List[int], head_box: List[int]) -> bool:
    """
    True if head_box (hairnet / no_hairnet / head-cap) belongs to person_box.
    Evaluates horizontal alignment and upper-body vertical placement.
    """
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = head_box
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)

    hcx = (hx1 + hx2) / 2.0
    hcy = (hy1 + hy2) / 2.0

    # Upper 45% of person height with 15% horizontal margin
    hr_x1 = px1 - 0.15 * pw
    hr_x2 = px2 + 0.15 * pw
    hr_y1 = py1 - 0.10 * ph
    hr_y2 = py1 + 0.45 * ph

    if (hr_x1 <= hcx <= hr_x2) and (hr_y1 <= hcy <= hr_y2):
        return True

    head_rgn = [max(0, int(hr_x1)), max(0, int(hr_y1)), int(hr_x2), int(hr_y2)]
    if _iou(head_rgn, head_box) >= 0.05:
        return True

    return False


# ─────────────────────────────────────────────────────────────────
# ppe.pt inference + raw detection parsing
# ─────────────────────────────────────────────────────────────────


def _run_inference_with(frame: np.ndarray, model, is_ppe: bool) -> List[Dict]:
    """
    Run YOLO inference with the specified model.
    Works with both ppe.pt (is_ppe=True) and standard COCO models.

    Returns flat list: {label, confidence, bbox:[x1,y1,x2,y2], det_type}
    """
    from services.multi_model_detector import normalize_class_label

    # Use FP16 half-precision if GPU available → 2× throughput, half VRAM.
    # On CPU-only (16 GB RAM) gracefully falls back to FP32.
    try:
        import torch as _torch
        _use_half = _torch.cuda.is_available()
    except Exception:
        _use_half = False

    results = list(model(
        frame,
        verbose=False,
        imgsz=480,
        conf=max(0.40, getattr(settings, "DETECTION_CONF", 0.50)),
        iou=settings.NMS_IOU,
        half=_use_half,
        stream=True,
    ))
    detections = []
    _raw_box_count = sum(len(r.boxes) for r in results)
    if _raw_box_count > 0:
        _model_names = getattr(model, "names", {})
        _raw_labels = [
            _model_names.get(int(box.cls[0]), f"class_{int(box.cls[0])}")
            for r in results for box in r.boxes
        ]
        log.debug("[INFERENCE] raw_boxes=%d labels=%s", _raw_box_count, _raw_labels)

    for r in results:
        for box in r.boxes:
            cls_idx = int(box.cls[0])

            # Resolve raw label from model class map
            _model_names = getattr(model, "names", {})
            if isinstance(_model_names, dict) and cls_idx in _model_names:
                raw_label = _model_names[cls_idx]
            elif is_ppe and cls_idx < len(PPE_CLASS_NAMES):
                raw_label = PPE_CLASS_NAMES[cls_idx]
            else:
                raw_label = f"class_{cls_idx}"

            # Normalize label to standard taxonomy
            label, det_type = normalize_class_label(raw_label)
            if not label or det_type == "ignore":
                continue

            # For COCO (non-PPE) models: discard classes not in our detection sets.
            # This prevents car, truck, bicycle, bench, etc. from flooding results.
            if not is_ppe and det_type == "neutral" and label not in NEUTRAL_CLASSES:
                continue

            conf = round(float(box.conf[0]), 3)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

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
    is_shop: bool = False,
) -> List[Dict]:
    """
    Assign violation/compliant PPE detections to their nearest person.
    For role='None': required violations come from detection_filters directly.
    For other roles: req_violations come from ROLE_RULES and the filter acts as a subset mask.
    
    is_shop: if True, bangles and hand/wrist items are permitted (customer/cashier retail zone)
             and excluded from violations. If False, any item worn on hand/wrist is an immediate violation.
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
        head_bbox: Optional[List[int]] = None
        headwear_status: Optional[str] = None
        headwear_label: Optional[str] = None
        headwear_conf: float = 0.0

        for det in ppe_detections:
            lbl = det["label"]
            raw_lbl = det.get("raw_label", lbl)
            is_head = (
                lbl in ("Bakery-Head-Cap", "NO-Bakery-Head-Cap", "Hairnet", "NO-Hairnet")
                or raw_lbl in ("hairnet", "no_hairnet", "Hairnet", "NO-Hairnet")
            )
            belongs = _head_belongs_to_person(p_box, det["bbox"]) if is_head else _belongs_to_person(p_box, det["bbox"])
            if not belongs:
                continue

            if is_head:
                det_conf = det.get("confidence", 0.0)
                if head_bbox is None or det_conf > headwear_conf:
                    head_bbox = det["bbox"]
                    headwear_conf = det_conf
                    headwear_label = raw_lbl
                    headwear_status = "compliant" if det["det_type"] == "compliant" else "violation"

            if det["det_type"] == "violation":
                assigned_violations.append(det["label"])
            elif det["det_type"] == "compliant":
                assigned_compliant.append(det["label"])

        # Spatial check: In shop floor, hand accessories / bangles are permitted
        if is_shop:
            assigned_violations = [v for v in assigned_violations if v not in ("Bangles", "bangles")]

        # Resolve conflicting headwear detections using the highest-confidence prediction
        if headwear_status == "compliant":
            assigned_violations = [v for v in assigned_violations if v not in ("NO-Bakery-Head-Cap", "NO-Hairnet")]
            if "Bakery-Head-Cap" not in assigned_compliant:
                assigned_compliant.append("Bakery-Head-Cap")
        elif headwear_status == "violation":
            assigned_compliant = [c for c in assigned_compliant if c not in ("Bakery-Head-Cap", "Hairnet")]
            if "NO-Bakery-Head-Cap" not in assigned_violations:
                assigned_violations.append("NO-Bakery-Head-Cap")

        ppe_missing: List[str] = []
        violation_labels: List[str] = []

        # Hand/wrist item worn check: on all non-shop floors, any detected hand/wrist accessory is a violation
        if not is_shop and ("Bangles" in assigned_violations or "bangles" in assigned_violations):
            if "Bangles" not in ppe_missing:
                ppe_missing.append("Bangles")
            bangle_lbl = VIOLATION_LABEL_MAP.get("Bangles", "Hand Item / Bangles Worn (Violation)")
            if bangle_lbl not in violation_labels:
                violation_labels.append(bangle_lbl)

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

            # Skip gloves and mask (disabled per client requirements)
            if req_v in ("NO-Mask", "NO-Gloves"):
                continue

            # Mode 1: model explicitly detected the violation class
            if req_v in assigned_violations:
                human_label = VIOLATION_LABEL_MAP.get(req_v, req_v)
                if req_v not in ppe_missing:
                    ppe_missing.append(req_v)
                if human_label not in violation_labels:
                    violation_labels.append(human_label)
                continue

            # Mode 2: absence detection — if the compliant counterpart is NOT seen
            # and the violation is NOT seen, the item is likely absent.
            # Skip for classes that have no compliant counterpart (e.g. Bangles).

            # Head-cap absence is unsafe to infer from a full-frame miss.  Small
            # heads, occlusion, lighting, and crop scale make "not detected" very
            # different from "not worn".  The dedicated HeadCapMonitor performs
            # crop inference + temporal confirmation for this class.
            if req_v == "NO-Bakery-Head-Cap":
                continue

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
                "confidence": person.get("confidence", 0.9),
                "track_id": person.get(
                    "track_id", -1
                ),  # passthrough from _apply_tracking
                "worker_name": person.get("worker_name"),
                "employee_id": person.get("employee_id"),
                "ppe_found": list(assigned_compliant),
                "ppe_missing": ppe_missing,
                "assigned_violations": assigned_violations,
                "is_compliant": len(ppe_missing) == 0,
                "violation_labels": violation_labels,
                "head_bbox": head_bbox,
                "headwear_status": headwear_status,
                "headwear_label": headwear_label,
                "headwear_conf": headwear_conf,
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
        "NO-Bakery-Head-Cap": "Bakery-Head-Cap",
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

            # Vertical zone: helmet/mask/goggles/headcap → upper 25%; gloves/vest → mid
            upper = (
                "Hardhat" in req_v
                or "Mask" in req_v
                or "Goggles" in req_v
                or "Head-Cap" in req_v
                or "Bakery" in req_v
            )
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

    # ── Draw neutral / PPE item boxes (prominent, distinct colors) ──
    _COLOR_CASH = (0, 215, 255)       # Gold/Yellow (BGR)
    _COLOR_CYLINDER = (255, 180, 0)   # Cyan/Teal (BGR)
    _COLOR_DOC = (255, 120, 200)      # Pink/Purple (BGR)

    for det in raw_detections:
        if det["det_type"] in ("neutral",):
            x1, y1, x2, y2 = det["bbox"]
            lbl = det["label"]
            conf = det["confidence"]

            if lbl.lower() == "cash":
                box_color = _COLOR_CASH
                prefix = "💵 "
            elif "cylinder" in lbl.lower():
                box_color = _COLOR_CYLINDER
                prefix = "🛢️ "
            elif "document" in lbl.lower():
                box_color = _COLOR_DOC
                prefix = "📄 "
            else:
                box_color = COLOR_NEUTRAL
                prefix = "📦 "

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Draw text pill background
            text_str = f"{prefix}{lbl} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            pill_y1 = max(0, y1 - th - 8)
            pill_y2 = y1
            cv2.rectangle(annotated, (x1, pill_y1), (x1 + tw + 8, pill_y2), box_color, -1)
            cv2.putText(
                annotated,
                text_str,
                (x1 + 4, pill_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 0) if box_color in (_COLOR_CASH, _COLOR_CYLINDER) else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # ── Draw standalone violation / compliant boxes (multi-model) ──
    # Catches multi-model detections: Worker Fall, Machine Anomaly, Object Throwing,
    # Bangles, standalone Hair/Gloves from hairnet model, etc.
    _COLOR_ANOMALY = (60, 20, 220)     # Deep Orange-Red (BGR)
    _COLOR_THROWING = (180, 40, 200)   # Magenta (BGR)
    _COLOR_FALL = (40, 40, 220)       # Red (BGR)
    _COLOR_BANGLES = (20, 120, 220)   # Amber-Red (BGR)
    _COLOR_HAIRNET_OK = (40, 180, 90) # Teal-Green (BGR)
    _COLOR_GLOVES_OK = (40, 180, 180) # Cyan (BGR)

    for det in raw_detections:
        _dt = det.get("det_type", "")
        if _dt not in ("violation", "compliant"):
            continue
        x1, y1, x2, y2 = det["bbox"]
        lbl = det["label"]
        conf = det["confidence"]
        _lbl_lower = lbl.lower()

        # Skip tiny bboxes that match a person bbox exactly — those are rendered
        # via the per-person enriched section below (red/green person frame).
        _matches_person = False
        for _p in enriched_persons:
            _px1, _py1, _px2, _py2 = _p["bbox"]
            if _px1 == x1 and _py1 == y1 and _px2 == x2 and _py2 == y2:
                _matches_person = True
                break
        if _matches_person:
            continue

        # Pick color + emoji based on label
        if "fall" in _lbl_lower:
            box_color = _COLOR_FALL
            prefix = "🚨 "
        elif "anomaly" in _lbl_lower:
            box_color = _COLOR_ANOMALY
            prefix = "⚠️ "
        elif "throw" in _lbl_lower:
            box_color = _COLOR_THROWING
            prefix = "🏃 "
        elif "bangle" in _lbl_lower:
            box_color = _COLOR_BANGLES
            prefix = "🚫 "
        elif "hair" in _lbl_lower or "head-cap" in _lbl_lower or "head cap" in _lbl_lower:
            raw_n = det.get("raw_label", "")
            lbl = raw_n if raw_n in ("hairnet", "no_hairnet") else lbl
            if _dt == "compliant":
                box_color = _COLOR_HAIRNET_OK
                prefix = "🧢 "
            else:
                box_color = COLOR_VIOLATION
                prefix = "❌ "
        elif "glove" in _lbl_lower:
            if _dt == "compliant":
                box_color = _COLOR_GLOVES_OK
                prefix = "🧤 "
            else:
                box_color = COLOR_VIOLATION
                prefix = "❌ "
        elif _dt == "compliant":
            box_color = COLOR_COMPLIANT
            prefix = "✅ "
        else:
            box_color = COLOR_VIOLATION
            prefix = "❗ "

        thickness = 3 if _dt == "violation" else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        text_str = f"{prefix}{lbl} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        pill_y1 = max(0, y1 - th - 8)
        pill_y2 = y1
        cv2.rectangle(annotated, (x1, pill_y1), (x1 + tw + 8, pill_y2), box_color, -1)
        _text_color = (255, 255, 255)
        cv2.putText(
            annotated,
            text_str,
            (x1 + 4, pill_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            _text_color,
            1,
            cv2.LINE_AA,
        )

    # ── Draw per-person boxes ───────────────────────────────────
    for person in enriched_persons:
        x1, y1, x2, y2 = person["bbox"]
        is_compliant = person["is_compliant"]

        if not is_compliant:
            # If the only violation is headcap (No Head Cap / No Hairnet), omit the giant outer person box
            # so the focused inner headwear box is prominent and uncluttered.
            other_violations = [
                v for v in person.get("violation_labels", [])
                if v not in ("No Head Cap", "No Hairnet")
            ]
            if other_violations:
                color = COLOR_VIOLATION
                thick = 3
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

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

                for idx, vlabel in enumerate(other_violations):
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
    persons = _deduplicate_person_boxes(persons)

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
    zones: Optional[Dict[str, Any]] = None,
    floor: Optional[str] = None,
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

    log.debug(
        "[PIPELINE] role=%s | sim=%s | ded_helmet=%s | filters=%s",
        role, _use_simulation, _helmet_model_dedicated, detection_filters,
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
    log.debug("[PIPELINE] role=%s | sim=%s | filters=%s", role, _use_simulation, detection_filters)
    active_model, active_is_ppe = _get_model_for_role(role)

    # Mapping from violation class → its compliant counterpart (model class names)
    VIOLATION_TO_COMPLIANT = {
        "NO-Hardhat": "Hardhat",
        "NO-Mask": "Mask",
        "NO-Safety Vest": "Safety Vest",
        "NO-Bakery-Head-Cap": "Bakery-Head-Cap",  # bakery cap compliant class
        "NO-Gloves": "Gloves",                   # food safety gloves
        # Bangles has no "compliant" counterpart — it is always a violation
    }

    if _use_simulation:
        # Explicit simulation mode (no real model available at all)
        raw = _simulate(
            frame, role, detection_filters=detection_filters, frame_index=frame_index
        )
    elif active_model is None:
        # No legacy ppe.pt — use multi-model pipeline exclusively.
        # The _mm_detector.detect() block below will inject real person detections
        # (from yolov8x_coco), hairnet, fall, and all other PPE from portable_models.
        # Starting with an empty raw prevents simulated phantom boxes from conflicting.
        raw = []
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

    # ── Zone-Aware Multi-Model Pipeline Injection ────────────────────────
    # Run specialized models from portable_models_package (hairnet, fall, cash, anomaly)
    # based on zone_type (auto-resolved from camera_id or ocr_zone_config)
    try:
        from services.multi_model_detector import get_multi_model_detector
        _mm_detector = get_multi_model_detector()

        # Determine zone_type & floor context:
        # process_frame / process_frame_numpy pack zone_type into ocr_zone_config when called
        _eff_zone = "default"
        if ocr_zone_config and isinstance(ocr_zone_config, dict):
            _eff_zone = ocr_zone_config.get("zone_type") or "default"

        _eff_floor = (floor or "").lower()
        if not _eff_floor and camera_id is not None:
            try:
                from database import Camera as _CamModel, SessionLocal as _SL
                with _SL() as _s:
                    _c = _s.query(_CamModel).filter(_CamModel.id == camera_id).first()
                    if _c and _c.floor:
                        _eff_floor = _c.floor.lower()
                        if _eff_zone == "default" and _c.zone_type:
                            _eff_zone = _c.zone_type.lower()
            except Exception:
                pass

        is_shop_floor = (
            _eff_floor == "shop"
            or _eff_zone in ("shop", "shop_counter", "cashbox", "cash")
            or bool(zones and any(k.lower() in ("shop", "shop_counter", "cashbox", "cash") for k in zones.keys()))
        )

        # If primary model already ran yolov8x, resolve specialized models for this zone
        _specialized_models = None
        if active_model is not None:
            _specialized_models = ["hairnet_glove_detection", "fall_detection"]
            # Auto-include cash_detection for shop/cashbox zones
            _is_cash_target = is_shop_floor or _eff_zone in ("shop", "shop_counter", "cashbox", "cash")
            if _is_cash_target and "cash_detection" not in _specialized_models:
                _specialized_models.append("cash_detection")
            # Auto-include helmet_model for loading zones
            if _eff_zone in ("loading", "construction") and "helmet_model" not in _specialized_models:
                _specialized_models.append("helmet_model")

        _extra_dets = _mm_detector.detect(
            frame, zone_type=_eff_zone, camera_id=camera_id, zones=zones, enabled_models=_specialized_models
        )
        for _ed in _extra_dets:
            _lbl = _ed.label
            _dt = _ed.det_type or "neutral"

            # The multi-model detector already normalized labels via normalize_class_label().
            # Only apply disable-filters for classes the client doesn't want.
            _lbl_lower = _lbl.lower()

            # Gloves disabled per client requirement
            if "glove" in _lbl_lower or "palm" in _lbl_lower:
                continue
            # Mask disabled per client requirement
            if "mask" in _lbl_lower and "no_mask" not in _lbl_lower:
                continue

            raw.append({
                "label": _lbl,
                "confidence": _ed.confidence,
                "bbox": _ed.bbox,
                "det_type": _dt,
                "raw_label": _ed.raw_label or _lbl,
            })
    except Exception as _mm_err:
        log.debug(f"[MultiModel] Error during injection: {_mm_err}")

    # ── Wrist accessory / Hand item / Bangles detection via Pose keypoints ───────────
    # Active across all factory floors (Ground, First, Second) and default zones EXCEPT Shop.
    if not is_shop_floor:
        try:
            from services.pose_layer import pose_adapter
            _wrist_dets = pose_adapter.detect_wrist_accessories(frame, is_shop=False)
            for _wd in _wrist_dets:
                raw.append(_wd)
        except Exception as _wr_err:
            log.debug(f"[Pose] Wrist accessory check error: {_wr_err}")
    else:
        # Shop floor: exclude any bangle / hand item detections
        raw = [
            d for d in raw
            if d.get("label") not in ("Bangles", "bangles")
            and d.get("raw_label") != "hand_wrist_item"
        ]

    # Strictly filter raw detections to only keep selected classes
    # CRITICAL: safety-critical anomalies must NEVER be suppressed by client-side PPE checkboxes!
    CRITICAL_SAFETY_LABELS = {
        "Worker Fall", "fall", "Fall",
        "Machine Anomaly", "anomaly", "Anomaly",
        "Object Throwing", "throw", "Throw",
        "Bangles", "bangles",
        "Cash", "cash",
        "Cylinder", "gas_cylinder",
        "Bakery-Head-Cap", "NO-Bakery-Head-Cap",
        "hairnet", "no_hairnet", "Hairnet", "NO-Hairnet",
    }

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
            or d["label"] in CRITICAL_SAFETY_LABELS  # always keep critical safety detections
        ]
    after_filter = [d["label"] for d in raw]
    # Only log when there's something worth seeing (suppresses empty-frame noise)
    if before_filter or after_filter:
        log.debug(
            "[FILTER] before=%s | after=%s | filters=%s",
            before_filter, after_filter, detection_filters,
        )

    raw = _deduplicate_headwear_boxes(raw)
    persons = [d for d in raw if d["det_type"] == "person"]
    ppe_dets = [d for d in raw if d["det_type"] in ("violation", "compliant")]

    # Deduplicate overlapping person detections (merges multi-model & multi-scale duplicates)
    persons = _deduplicate_person_boxes(persons)

    # ── ByteTrack: assign stable track_id to each person (no-op if camera_id is None) ──
    if camera_id is not None:
        persons = _apply_tracking(camera_id, persons)
        try:
            from services.face_service import get_worker_identity
            for p in persons:
                t_id = p.get("track_id")
                if t_id is not None and t_id != -1:
                    ident = get_worker_identity(camera_id, t_id)
                    if ident:
                        p["worker_name"] = ident.get("worker_name")
                        p["employee_id"] = ident.get("employee_id")
        except Exception as _ident_exc:
            log.debug(f"Worker identity attach error: {_ident_exc}")

    # Only synthesize a person if wearable PPE items (headcap, vest, hardhat, bangles) are detected
    WEARABLE_PPE_CLASSES = {
        "Hardhat", "Safety Vest", "Bakery-Head-Cap",
        "NO-Hardhat", "NO-Safety Vest", "NO-Bakery-Head-Cap",
        "Bangles", "bangles",
    }
    wearable_dets = [d for d in ppe_dets if d["label"] in WEARABLE_PPE_CLASSES]

    if not persons and wearable_dets:
        xs = [d["bbox"][0] for d in wearable_dets] + [d["bbox"][2] for d in wearable_dets]
        ys = [d["bbox"][1] for d in wearable_dets] + [d["bbox"][3] for d in wearable_dets]
        h, w = frame.shape[:2]
        persons = [
            {
                "label": "Person",
                "confidence": max(d["confidence"] for d in wearable_dets),
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
    # Ensure person detection pass runs with primary COCO model (yolov8x.pt)
    # if person list is empty.
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

    # Deduplicate once more in case fallback or synthetic added overlapping boxes
    persons = _deduplicate_person_boxes(persons)

    # ── Specialized model inference (Hairnet / Fall / Cash / Anomaly) ──
    # Note: Full-frame hairnet_glove_detection runs via MultiModelDetector above at imgsz=960.
    enriched = _associate_to_persons(
        persons, ppe_dets, role, detection_filters=detection_filters, is_shop=is_shop_floor
    )

    # ── Uniform classifier check on detected persons ──────────────────────────
    try:
        from services.uniform_monitor import uniform_monitor
        _req_uniform = (
            "NO-Uniform" in (detection_filters or [])
            or (detection_filters is None and "NO-Uniform" in ROLE_RULES.get(role, {}).get("required_sim", []))
            or (detection_filters is None and "NO-Uniform" in ROLE_RULES.get(role, {}).get("required_violations", []))
        )
        for p in enriched:
            res, _ = uniform_monitor.classify_person_box(frame, p["bbox"])
            is_uniform = (res.predicted_class in ("UNIFORM", "Uniform", "Uniform_1", "Uniform_2"))
            p["uniform_prediction"] = res.predicted_class
            p["uniform_confidence"] = res.confidence
            p["has_uniform"] = is_uniform

            if _req_uniform:
                if not is_uniform:
                    if "NO-Uniform" not in p["ppe_missing"]:
                        p["ppe_missing"].append("NO-Uniform")
                    if "No Uniform" not in p["violation_labels"]:
                        p["violation_labels"].append("No Uniform")
                    p["is_compliant"] = False
                else:
                    if "Uniform" not in p["ppe_found"]:
                        p["ppe_found"].append("Uniform")
    except Exception as _ue:
        log.debug(f"[YOLO] uniform check error: {_ue}")

    # ── Head-cap fallback for detected persons (webcam & CCTV) ────────────────
    # If the role requires Bakery-Head-Cap and neither cap nor violation was found,
    # run inference on the upper torso/head region with contextual padding.
    try:
        _req_headcap = (
            "NO-Bakery-Head-Cap" in (detection_filters or [])
            or (detection_filters is None and "NO-Bakery-Head-Cap" in ROLE_RULES.get(role, {}).get("required_violations", []))
        )
        if _req_headcap and enriched:
            from services.model_manager import ModelManager
            _mm = ModelManager()
            for p in enriched:
                # If already assigned from full frame, keep it
                has_cap = "Bakery-Head-Cap" in p.get("ppe_found", [])
                has_no_cap = "NO-Bakery-Head-Cap" in p.get("ppe_missing", [])
                if has_cap or has_no_cap:
                    continue

                # Contextual upper-body/head crop (top 50% + 20% horizontal margin)
                px1, py1, px2, py2 = p["bbox"]
                pw = max(1, px2 - px1)
                ph = max(1, py2 - py1)
                fh, fw = frame.shape[:2]
                cx1 = max(0, int(px1 - 0.20 * pw))
                cy1 = max(0, int(py1 - 0.05 * ph))
                cx2 = min(fw, int(px2 + 0.20 * pw))
                cy2 = min(fh, int(py1 + 0.55 * ph))

                if cx2 > cx1 and cy2 > cy1:
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        crop_dets = _mm.infer("hairnet_glove_detection", crop, conf=0.18)
                        best_cap = None
                        best_no_cap = None
                        for cd in crop_dets:
                            c_lbl = getattr(cd, "class_name", "") or ""
                            c_conf = float(getattr(cd, "confidence", 0.0))
                            if c_lbl in ("hairnet", "Bakery-Head-Cap", "Hair_Cover"):
                                if best_cap is None or c_conf > best_cap["confidence"]:
                                    best_cap = {"confidence": c_conf, "bbox": cd.bbox}
                            elif c_lbl in ("no_hairnet", "NO-Bakery-Head-Cap", "Hair"):
                                if best_no_cap is None or c_conf > best_no_cap["confidence"]:
                                    best_no_cap = {"confidence": c_conf, "bbox": cd.bbox}

                        if best_cap and (not best_no_cap or best_cap["confidence"] >= best_no_cap["confidence"]):
                            # Cap confirmed on crop
                            if "Bakery-Head-Cap" not in p["ppe_found"]:
                                p["ppe_found"].append("Bakery-Head-Cap")
                            bx1, by1, bx2, by2 = best_cap["bbox"]
                            mapped_box = [cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2]
                            raw.append({
                                "label": "Bakery-Head-Cap",
                                "confidence": round(best_cap["confidence"], 3),
                                "bbox": mapped_box,
                                "det_type": "compliant",
                                "raw_label": "hairnet",
                            })
                        elif best_no_cap:
                            # Explicit bare hair / no hairnet on crop
                            if "NO-Bakery-Head-Cap" not in p["ppe_missing"]:
                                p["ppe_missing"].append("NO-Bakery-Head-Cap")
                            if "No Head Cap" not in p["violation_labels"]:
                                p["violation_labels"].append("No Head Cap")
                            p["is_compliant"] = False
                            bx1, by1, bx2, by2 = best_no_cap["bbox"]
                            mapped_box = [cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2]
                            raw.append({
                                "label": "NO-Bakery-Head-Cap",
                                "confidence": round(best_no_cap["confidence"], 3),
                                "bbox": mapped_box,
                                "det_type": "violation",
                                "raw_label": "no_hairnet",
                            })
                        else:
                            # Absence fallback: person present, role requires cap, neither seen
                            if "NO-Bakery-Head-Cap" not in p["ppe_missing"]:
                                p["ppe_missing"].append("NO-Bakery-Head-Cap")
                            if "No Head Cap" not in p["violation_labels"]:
                                p["violation_labels"].append("No Head Cap")
                            p["is_compliant"] = False
                            # Synthesize inner headwear box for clear visual feedback
                            head_synth_box = [
                                max(0, int(px1 + 0.15 * pw)),
                                max(0, int(py1)),
                                min(fw, int(px2 - 0.15 * pw)),
                                min(fh, int(py1 + 0.35 * ph)),
                            ]
                            raw.append({
                                "label": "NO-Bakery-Head-Cap",
                                "confidence": round(p.get("confidence", 0.5), 2),
                                "bbox": head_synth_box,
                                "det_type": "violation",
                                "raw_label": "no_hairnet",
                            })
    except Exception as _he:
        log.debug(f"[YOLO] headcap check error: {_he}")

    # ── Traffic Police: remap already done inside dedicated/strict pipeline ──
    # (no remap needed here anymore)

    is_compliant, missing, alert_msg, severity = _compliance_summary(enriched, role)

    annotated = _draw_results(frame, enriched, raw, role)
    ann_b64 = encode_frame(annotated)
    snap_b64 = encode_frame(annotated, quality=85) if not is_compliant else None

    violations = [p for p in enriched if not p["is_compliant"]]
    ui_detections = [
        {
            "label": d["label"],
            "confidence": d["confidence"],
            "bbox": d["bbox"],
            "raw_label": d.get("raw_label", d["label"]),
            "det_type": d.get("det_type", "neutral"),
        }
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

    # ── Cash detection summary ────────────────────────────────────────────────
    # Extract Cash objects (class 14) from raw detections so callers can
    # display a yellow badge and/or feed the CashEventTracker.
    _cash_dets = [
        d for d in raw
        if str(d.get("label", "")).lower() == "cash" or d.get("class_id") == 14
    ]

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
        # Cash monitoring fields
        "cash_detected":   bool(_cash_dets),
        "cash_detections": _cash_dets,
    }


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────


def process_frame(
    b64_frame: str,
    role: str,
    detection_filters: Optional[List[str]] = None,
    no_phone_zone: bool = False,
    zone_type: Optional[str] = None,
) -> Dict:
    """
    Full violation pipeline from base64 frame.
    Called by the WebSocket detection router.
    """
    frame = decode_frame(b64_frame)
    if frame is None:
        return {"error": "Invalid frame data"}
    _zone_cfg = {"zone_type": zone_type} if zone_type else None
    return _run_pipeline(
        frame,
        role,
        detection_filters=detection_filters,
        no_phone_zone=no_phone_zone,
        ocr_zone_config=_zone_cfg,
    )


def process_frame_numpy(
    frame: np.ndarray,
    role: str,
    detection_filters: Optional[List[str]] = None,
    no_phone_zone: bool = False,
    frame_index: int = -1,
    zone_type: Optional[str] = None,
    camera_id: Optional[int] = None,
    zones: Optional[Dict[str, Any]] = None,
    floor: Optional[str] = None,
) -> Dict:
    """
    Full violation pipeline from a numpy frame directly.
    Called by the video upload router (avoids double encode/decode).
    frame_index: passed through to _simulate for deterministic results in video.
    """
    if frame is None:
        return {"error": "Invalid frame"}
    _zone_cfg = {"zone_type": zone_type} if zone_type else None
    return _run_pipeline(
        frame,
        role,
        detection_filters=detection_filters,
        no_phone_zone=no_phone_zone,
        frame_index=frame_index,
        ocr_zone_config=_zone_cfg,
        camera_id=camera_id,
        zones=zones,
        floor=floor,
    )


def process_frame_numpy_tracked(
    frame: np.ndarray,
    role: str,
    camera_id: int,
    detection_filters: Optional[List[str]] = None,
    no_phone_zone: bool = False,
    zone_type: Optional[str] = None,
    zones: Optional[Dict[str, Any]] = None,
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
    _zone_cfg = {"zone_type": zone_type} if zone_type else None
    return _run_pipeline(
        frame,
        role,
        detection_filters=detection_filters,
        no_phone_zone=no_phone_zone,
        camera_id=camera_id,
        ocr_zone_config=_zone_cfg,
        zones=zones,
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
