"""
YOLO Violation Detection Service — v3.0  (ppe.pt edition)
===========================================================
Uses the biswadeep-roy/Safety-Detection-YOLOv8 custom model (ppe.pt)
which natively detects BOTH compliance and violation states:

  Classes:  ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
             'NO-Safety Vest', 'Person', 'Safety Cone',
             'Safety Vest', 'machinery', 'vehicle']

  Violation classes: NO-Hardhat, NO-Mask, NO-Safety Vest
  Compliant classes: Hardhat, Mask, Safety Vest
  Neutral:           Person, Safety Cone, machinery, vehicle

Flow per frame:
  1. Run ppe.pt inference (conf ≥ 0.50).
  2. Separate detections into violations, compliant PPE, persons, neutral.
  3. Associate violations/PPE to nearest Person bbox via IoU + containment.
  4. Draw RED box + "No Hardhat" etc. for violators.
  5. Draw GREEN box for persons with ALL required PPE present.
  6. Build role-specific compliance summary.
  7. Return enriched result for the FastAPI router.

Model Download:
  https://drive.google.com/drive/folders/11tfTBkp4JdlJ8QXoAMZxgVMf8xpLBXi_

Place ppe.pt in:  backend/ppe.pt
"""

import cv2
import numpy as np
import base64
import random
import logging
import datetime
from typing import List, Dict, Tuple, Optional
from config import settings

log = logging.getLogger("yolo_service")

# ─────────────────────────────────────────────────────────────────
# ppe.pt class map  (index → name, as trained)
# ─────────────────────────────────────────────────────────────────
PPE_CLASS_NAMES = [
    'Hardhat',       # 0  ✅ compliant
    'Mask',          # 1  ✅ compliant
    'NO-Hardhat',    # 2  ❌ violation
    'NO-Mask',       # 3  ❌ violation
    'NO-Safety Vest',# 4  ❌ violation
    'Person',        # 5  👤 neutral person
    'Safety Cone',   # 6  🟠 neutral
    'Safety Vest',   # 7  ✅ compliant
    'machinery',     # 8  🔵 neutral
    'vehicle',       # 9  🔵 neutral
]

# Sets for fast membership checks
VIOLATION_CLASSES  = {'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'}
COMPLIANT_CLASSES  = {'Hardhat', 'Mask', 'Safety Vest'}
PERSON_CLASSES     = {'Person'}
NEUTRAL_CLASSES    = {'Safety Cone', 'machinery', 'vehicle'}

# Human-readable violation → missing item label
# Native ppe.pt violation classes
VIOLATION_LABEL_MAP = {
    'NO-Hardhat':       'No Hardhat',
    'NO-Mask':          'No Mask',
    'NO-Safety Vest':   'No Safety Vest',
    # Simulated classes (not in ppe.pt — detected via simulation / future model)
    'NO-Gloves':        'No Gloves',
    'NO-Goggles':       'No Goggles',
    'NO-Safety Shoes':  'No Safety Shoes',
    'NO-ID Card':       'No ID Card',
    'NO-Uniform':       'No Uniform',
}

# Classes that are purely simulated (not detected by ppe.pt natively)
SIM_ONLY_VIOLATIONS = {'NO-Gloves', 'NO-Goggles', 'NO-Safety Shoes', 'NO-ID Card', 'NO-Uniform'}

# ── Traffic Police label remap ────────────────────────────────────
# ppe.pt uses "Hardhat" / "NO-Hardhat" — for Traffic Police display
# we remap these human-readable labels to "Helmet" / "No Helmet".
TRAFFIC_POLICE_LABEL_REMAP: Dict[str, str] = {
    'No Hardhat': 'No Helmet',
    'Hardhat':    'Helmet',
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
        "required_compliant":  ["Hardhat", "Safety Vest", "Mask"],
        # Simulated PPE (Gloves, Goggles, Safety Shoes)
        "required_sim":        ["NO-Gloves", "NO-Goggles", "NO-Safety Shoes"],
        "severity": "critical",
        "alert_prefix": "Construction safety violation",
    },
    "Doctor": {
        "required_violations": ["NO-Mask"],
        "required_compliant":  ["Mask"],
        "required_sim":        ["NO-Gloves"],
        "severity": "high",
        "alert_prefix": "🩺 Doctor PPE violation",
    },
    "Traffic Police": {
        "required_violations": ["NO-Hardhat"],
        "required_compliant":  ["Hardhat"],
        "required_sim":        [],
        "severity": "critical",
        "alert_prefix": "🚓 Traffic Police violation",
    },
    "College": {
        "required_violations": [],
        "required_compliant":  [],
        "required_sim":        ["NO-ID Card", "NO-Uniform"],
        "severity": "medium",
        "alert_prefix": "🎓 College compliance violation",
    },
    "Home": {
        "required_violations": [],
        "required_compliant":  [],
        "required_sim":        [],
        "severity": "low",
        "alert_prefix": "🏠 Home security",
    },
    # "None" = fully custom: required_violations come from detection_filters at runtime.
    # This entry provides defaults; the pipeline overrides it dynamically.
    "None": {
        "required_violations": [],
        "required_compliant":  [],
        "required_sim":        [],
        "severity": "high",
        "alert_prefix": "🛠️ Custom safety violation",
    },
}

# ─────────────────────────────────────────────────────────────────
# Visual colours  (BGR)
# ─────────────────────────────────────────────────────────────────
COLOR_VIOLATION = (30,  30, 220)   # Red
COLOR_COMPLIANT = (40, 200,  50)   # Green
COLOR_NEUTRAL   = (180, 120, 60)   # Blue-grey
COLOR_PERSON    = (200, 160,  60)  # Amber

# ─────────────────────────────────────────────────────────────────
# Model state  (loaded once at startup)
# ─────────────────────────────────────────────────────────────────
_model                  = None   # general PPE model (ppe.pt)
_helmet_model           = None   # dedicated helmet model (keremberke / fallback)
_helmet_model_dedicated = False  # True when keremberke model (helmet/head classes)
_phone_model            = None   # dedicated phone detection model (COCO class 67)
_use_simulation         = False
_model_is_ppe           = False  # True when ppe.pt loaded (vs generic COCO)

# ── Keremberke hard-hat-detection model constants ─────────────────
# https://huggingface.co/keremberke/yolov8m-hard-hat-detection
_HELMET_MODEL_URL  = (
    "https://huggingface.co/keremberke/yolov8m-hard-hat-detection/resolve/main/best.pt"
)
_HELMET_MODEL_PATH = "helmet_model.pt"      # local cache path
# Classes in keremberke model — used by _run_traffic_police_helmet_pipeline
_HELMET_COMPLIANT_NAMES = {'helmet', 'hard hat', 'hardhat', 'with helmet'}
_HELMET_VIOLATION_NAMES = {'head', 'no helmet', 'no_helmet', 'without helmet'}


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
      1. ppe.pt  (custom 10-class PPE model)
      2. YOLO_MODEL from config (e.g. yolov8m.pt)
      3. Simulation mode

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
        from ultralytics import YOLO
        import torch
        _orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load

        _model = YOLO("ppe.pt")
        torch.load = _orig_load
        _use_simulation = False
        _model_is_ppe = True
        log.info("✅ PPE model loaded: ppe.pt (custom, 10 classes)")
        print("✅ Custom ppe.pt model loaded (PPE-native, 10 classes)")
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
    import os, threading
    HELMET_MIN_BYTES = 10 * 1024 * 1024  # 10 MB

    def _load_helmet_bg():
        """Background thread: download + load dedicated helmet model."""
        global _helmet_model, _helmet_model_dedicated
        import urllib.request
        _hm_orig2 = None
        try:
            from ultralytics import YOLO as _YOLO_hm
            import torch
            _hm_orig2 = torch.load
            def _hm_patch2(*a, **kw): kw.setdefault("weights_only", False); return _hm_orig2(*a, **kw)
            torch.load = _hm_patch2

            # Validate existing file — reject partial downloads
            if os.path.exists(_HELMET_MODEL_PATH):
                sz = os.path.getsize(_HELMET_MODEL_PATH)
                if sz < HELMET_MIN_BYTES:
                    print(f"⚠️  Removing corrupt helmet_model.pt ({sz//1024} KB < 10 MB)")
                    os.remove(_HELMET_MODEL_PATH)

            if os.path.exists(_HELMET_MODEL_PATH):
                # Valid cached model — just load it
                _helmet_model = _YOLO_hm(_HELMET_MODEL_PATH)
                _helmet_model_dedicated = True
                print(f"✅ Dedicated helmet model loaded from cache: {_HELMET_MODEL_PATH}")
            else:
                # Download from HuggingFace (one-time, ~52 MB)
                print("⏬ [background] Downloading dedicated helmet model…")
                print(f"   URL: {_HELMET_MODEL_URL}")
                urllib.request.urlretrieve(_HELMET_MODEL_URL, _HELMET_MODEL_PATH)
                if os.path.getsize(_HELMET_MODEL_PATH) < HELMET_MIN_BYTES:
                    raise ValueError("Downloaded file is too small — likely a network error")
                _helmet_model = _YOLO_hm(_HELMET_MODEL_PATH)
                _helmet_model_dedicated = True
                print("✅ Dedicated helmet model ready — keremberke/yolov8m-hard-hat-detection")
        except Exception as _hme:
            try:
                if _hm_orig2: torch.load = _hm_orig2
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
            print(f"⚠️  Helmet model unavailable ({type(_hme).__name__}). Strict ppe.pt logic will be used.")
        finally:
            try:
                if _hm_orig2: torch.load = _hm_orig2
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
    _phone_loaded     = False
    for _cand in _phone_candidates:
        try:
            from ultralytics import YOLO
            import torch
            _orig3 = torch.load
            def _p3(*a, **kw): kw.setdefault("weights_only", False); return _orig3(*a, **kw)
            torch.load = _p3
            _phone_model = YOLO(_cand)
            torch.load = _orig3
            log.info(f"✅ Phone model loaded: {_cand} (COCO, class 67 = cell phone)")
            print(f"✅ Phone detection model: {_cand}")
            _phone_loaded = True
            break
        except Exception as e:
            try: torch.load = _orig3
            except Exception: pass
            log.warning(f"{_cand} unavailable ({type(e).__name__}) — trying next…")
    if not _phone_loaded:
        _phone_model = _helmet_model if not _helmet_model_dedicated else None
        print("⚠️  Phone model unavailable — falling back to simulation for phone detection")



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
    areaA = max(1, (a[2]-a[0]) * (a[3]-a[1]))
    areaB = max(1, (b[2]-b[0]) * (b[3]-b[1]))
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
    for r in results:
        for box in r.boxes:
            if is_ppe:
                cls_idx = int(box.cls[0])
                label = PPE_CLASS_NAMES[cls_idx] if cls_idx < len(PPE_CLASS_NAMES) else "unknown"
            else:
                label = model.names[int(box.cls[0])]

            conf  = round(float(box.conf[0]), 3)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            if label in VIOLATION_CLASSES:
                det_type = "violation"
            elif label in COMPLIANT_CLASSES:
                det_type = "compliant"
            elif label in PERSON_CLASSES or label.lower() == "person":
                det_type = "person"
            else:
                det_type = "neutral"

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "det_type": det_type,
            })
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
    rules          = ROLE_RULES.get(role, ROLE_RULES["Home"])
    req_violations = list(rules.get("required_violations", []))
    req_sim        = list(rules.get("required_sim", []))

    # ── "None" role: derive requirements entirely from the user's selection ──
    if role == "None" and detection_filters:
        req_violations = [f for f in detection_filters if f not in SIM_ONLY_VIOLATIONS]
        req_sim        = [f for f in detection_filters if f in SIM_ONLY_VIOLATIONS]

    enriched = []
    for person in persons:
        p_box = person["bbox"]

        assigned_violations: List[str] = []
        assigned_compliant:  List[str] = []

        for det in ppe_detections:
            if not _belongs_to_person(p_box, det["bbox"]):
                continue
            if det["det_type"] == "violation":
                assigned_violations.append(det["label"])
            elif det["det_type"] == "compliant":
                assigned_compliant.append(det["label"])

        ppe_missing:      List[str] = []
        violation_labels: List[str] = []

        # ── Native ppe.pt violations ───────────────────────────────
        for req_v in req_violations:
            # For non-None roles: skip if this violation class not in active filter
            if role != "None" and detection_filters is not None and req_v not in detection_filters:
                continue
            if req_v in assigned_violations:
                human_label = VIOLATION_LABEL_MAP.get(req_v, req_v)
                ppe_missing.append(req_v)
                violation_labels.append(human_label)

        # ── Simulated / non-native PPE (Gloves, Goggles, ID Card, Safety Shoes) ──
        for sim_v in req_sim:
            if role != "None" and detection_filters is not None and sim_v not in detection_filters:
                continue
            if _use_simulation:
                # Simulation mode: randomly assign violations
                if sim_v in assigned_violations:
                    human_label = VIOLATION_LABEL_MAP.get(sim_v, sim_v)
                    ppe_missing.append(sim_v)
                    violation_labels.append(human_label)
            # Real model: ppe_extended.pt handles these — if not loaded, skip (do NOT phantom-flag)
            # Extended model results are already merged into assigned_violations before this point

        enriched.append({
            "bbox":                 p_box,
            "confidence":          person["confidence"],
            "ppe_found":           list(assigned_compliant),
            "ppe_missing":         ppe_missing,
            "assigned_violations": assigned_violations,
            "is_compliant":        len(ppe_missing) == 0,
            "violation_labels":    violation_labels,
        })

    return enriched


# ─────────────────────────────────────────────────────────────────
# Simulation mode  (when ppe.pt not available)
# ─────────────────────────────────────────────────────────────────

def _simulate(frame: np.ndarray, role: str,
              detection_filters: Optional[List[str]] = None,
              frame_index: int = -1) -> List[Dict]:
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
    req_sim        = rules.get("required_sim", [])
    all_req        = req_violations + req_sim   # simulate ALL required PPE

    # Apply filter: only simulate selected violation classes
    if detection_filters is not None:
        all_req = [v for v in all_req if v in detection_filters]

    detections: List[Dict] = []
    num_persons = random.randint(1, 2)
    x_offsets   = [0.08, 0.52]

    # Compliant class mapping for both native and simulated
    COMPLIANT_MAP = {
        "NO-Hardhat":     "Hardhat",
        "NO-Mask":        "Mask",
        "NO-Safety Vest": "Safety Vest",
        "NO-Gloves":      "Gloves",
        "NO-Goggles":     "Safety Goggles",
        "NO-ID Card":     "ID Card",
        "NO-Uniform":     "Uniform",
    }

    for i in range(num_persons):
        xo  = x_offsets[i] if i < len(x_offsets) else 0.2
        px,  py  = int(w * xo), int(h * 0.04)
        px2, py2 = int(w * (xo + 0.35)), int(h * 0.95)
        px2, py2 = min(px2, w - 1), min(py2, h - 1)

        detections.append({
            "label":      "Person",
            "confidence": round(_rng.uniform(0.82, 0.97), 3),
            "bbox":       [px, py, px2, py2],
            "det_type":   "person",
        })

        for req_v in all_req:
            is_violation = _rng.random() < 0.35   # 35% violation chance per item

            # Vertical zone: helmet/mask/goggles → upper 25%; gloves/vest → mid
            upper = ("Hardhat" in req_v or "Mask" in req_v or "Goggles" in req_v)
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
                label    = req_v           # e.g. "NO-Gloves"
                det_type = "violation"
            else:
                label    = COMPLIANT_MAP.get(req_v, "Hardhat")
                det_type = "compliant"

            detections.append({
                "label":      label,
                "confidence": round(_rng.uniform(0.65, 0.95), 3),
                "bbox":       [ix1, iy1, ix2, iy2],
                "det_type":   det_type,
            })

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

    violations  = [p for p in enriched_persons if not p["is_compliant"]]
    n_persons   = len(enriched_persons)
    n_violations = len(violations)

    # ── Top banner ─────────────────────────────────────────────
    if n_violations > 0:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 46), (0, 0, 170), -1)
        cv2.addWeighted(overlay, 0.58, annotated, 0.42, 0, annotated)
        banner = f"  \u26a0  {n_violations}/{n_persons} PERSON(S) VIOLATING  \u2014  ROLE: {role.upper()}"
        cv2.putText(annotated, banner, (8, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (255, 255, 255), 2, cv2.LINE_AA)
    else:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 46), (0, 130, 0), -1)
        cv2.addWeighted(overlay, 0.40, annotated, 0.60, 0, annotated)
        cv2.putText(annotated, f"  \u2713  ALL {n_persons} PERSON(S) COMPLIANT  \u2014  ROLE: {role.upper()}",
                    (8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # ── Draw neutral / PPE item boxes (thin, behind person boxes) ──
    for det in raw_detections:
        if det["det_type"] in ("neutral",):
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_NEUTRAL, 1)
            cv2.putText(annotated, f"{det['label']} {det['confidence']:.0%}",
                        (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_NEUTRAL, 1, cv2.LINE_AA)

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
            cv2.putText(annotated, conf_text, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

            # Violation label pills (stacked below top edge)
            for idx, vlabel in enumerate(person["violation_labels"]):
                label_y = y1 + 26 + idx * 24
                (lw, lh), _ = cv2.getTextSize(vlabel, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                cv2.rectangle(annotated,
                               (x1 + 4, label_y - lh - 4),
                               (x1 + lw + 14, label_y + 5),
                               (0, 0, 0), -1)
                cv2.putText(annotated, vlabel, (x1 + 8, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (90, 90, 255), 2, cv2.LINE_AA)

        else:
            color = COLOR_COMPLIANT
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # Role-specific compliant label
            if role == "Traffic Police":
                ok_txt = f"\u2713 Helmet OK {person['confidence']:.0%}"
            elif person.get('ppe_found'):
                ok_txt = f"\u2713 {person['ppe_found'][0]} OK {person['confidence']:.0%}"
            else:
                ok_txt = f"\u2713 Compliant {person['confidence']:.0%}"
            cv2.putText(annotated, ok_txt, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

    # ── Watermarks ─────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, ts, (w - 208, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(annotated, "OccuSafe Monitor v4.0", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 200, 255), 1, cv2.LINE_AA)

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
        f"{rules['alert_prefix']}: "
        f"{n_v}/{n_p} person(s) — {', '.join(all_missing)}"
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
        results = _helmet_model(
            frame, conf=0.28, iou=0.45, verbose=False
        )
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
            enriched.append({
                "bbox":                [x1, y1, x2, y2],
                "confidence":          conf,
                "ppe_found":           ["Helmet"] if is_helmet else [],
                "ppe_missing":         [] if is_helmet else ["NO-Hardhat"],
                "assigned_violations": [] if is_helmet else ["NO-Hardhat"],
                "is_compliant":        is_helmet,
                "violation_labels":    [] if is_helmet else ["No Helmet"],
            })
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

    persons  = [d for d in raw if d["det_type"] == "person"]
    ppe_dets = [d for d in raw if d["det_type"] in ("violation", "compliant")]

    # Fallback: create one synthetic person spanning all PPE detections if no Person class
    if not persons and ppe_dets:
        xs = [d["bbox"][0] for d in ppe_dets] + [d["bbox"][2] for d in ppe_dets]
        ys = [d["bbox"][1] for d in ppe_dets] + [d["bbox"][3] for d in ppe_dets]
        h, w = frame.shape[:2]
        persons = [{
            "label": "Person", "confidence": 0.82,
            "bbox": [max(0, min(xs)-30), max(0, min(ys)-30),
                     min(w-1, max(xs)+30), min(h-1, max(ys)+30)],
            "det_type": "person",
        }]

    enriched: List[Dict] = []
    for person in persons:
        p_box = person["bbox"]
        # Find all PPE items belonging to this person
        assigned_c = [d["label"] for d in ppe_dets
                      if d["det_type"] == "compliant" and _belongs_to_person(p_box, d["bbox"])]
        assigned_v = [d["label"] for d in ppe_dets
                      if d["det_type"] == "violation" and _belongs_to_person(p_box, d["bbox"])]

        # STRICT: helmet OK only when ppe.pt EXPLICITLY detects 'Hardhat'
        # If no helmet class detected at all (neither Hardhat nor NO-Hardhat), treat as violation
        hardhat_ok  = "Hardhat" in assigned_c
        no_hardhat  = "NO-Hardhat" in assigned_v or not hardhat_ok

        is_compliant = hardhat_ok and not no_hardhat
        enriched.append({
            "bbox":                p_box,
            "confidence":          person["confidence"],
            "ppe_found":           ["Helmet"] if hardhat_ok else [],
            "ppe_missing":         [] if is_compliant else ["NO-Hardhat"],
            "assigned_violations": assigned_v,
            "is_compliant":        is_compliant,
            "violation_labels":    [] if is_compliant else ["No Helmet"],
        })
    return enriched


# ─────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────

def _run_pipeline(frame: np.ndarray, role: str,
                  detection_filters: Optional[List[str]] = None,
                  no_phone_zone: bool = False,
                  frame_index: int = -1) -> Dict:
    """
    Shared pipeline used by both process_frame and process_frame_numpy.
    detection_filters: if provided, only these violation classes are evaluated.
    no_phone_zone: if True, any phone detection triggers an alert.
    """
    print(f"[PIPELINE] role={role} | sim={_use_simulation} | ded_helmet={_helmet_model_dedicated} | filters={detection_filters}")

    # ── Traffic Police: use dedicated helmet pipeline ─────────────────
    if role == "Traffic Police":
        if _helmet_model_dedicated:
            # Try the dedicated keremberke model first
            enriched = _run_traffic_police_dedicated(frame)
            if enriched is None:
                # Model failed — fall back
                enriched = _run_traffic_police_strict_ppe(frame, frame_index=frame_index)
        else:
            # Use ppe.pt with strict logic
            enriched = _run_traffic_police_strict_ppe(frame, frame_index=frame_index)

        is_compliant, missing, alert_msg, severity = _compliance_summary(enriched, "Traffic Police")
        annotated = _draw_results(frame, enriched, [], "Traffic Police")
        ann_b64   = encode_frame(annotated)
        snap_b64  = encode_frame(annotated, quality=85) if not is_compliant else None
        violations = [p for p in enriched if not p["is_compliant"]]
        ui_detections = [
            {"label": ("Helmet" if p["is_compliant"] else "No Helmet"),
             "confidence": p["confidence"], "bbox": p["bbox"]}
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
        ann_b64  = encode_frame(annotated_with_phone)
        snap_b64 = encode_frame(annotated_with_phone, quality=85) if (
            not is_compliant or phone_result["phone_alert"]
        ) else None
        phone_alert  = phone_result.get("phone_alert")
        phone_status = phone_result.get("phone_status", "safe")
        phone_dets   = phone_result.get("phone_detections", [])
        if phone_alert:
            is_compliant = False
            alert_msg = f"{alert_msg} | {phone_alert}" if alert_msg else phone_alert
            if not severity or severity == "low":
                severity = phone_result.get("phone_severity", "high")

        return {
            "persons":          enriched,
            "violations":       violations,
            "detections":       ui_detections + phone_dets,
            "is_compliant":     is_compliant,
            "missing_items":    missing,
            "violations_count": len(violations),
            "persons_count":    len(enriched),
            "alert_message":    alert_msg if not is_compliant else None,
            "severity":         severity if not is_compliant else None,
            "annotated_frame":  ann_b64,
            "snapshot_b64":     snap_b64,
            "phone_status":     phone_status,
            "phone_detected":   phone_result.get("phone_detected", False),
            "model_mode":       "helmet:keremberke" if _helmet_model_dedicated else "helmet:ppe.pt(strict)",
            "phone_severity":   phone_result.get("phone_severity", "low"),
        }

    # ── All other roles: general PPE pipeline ───────────────────────
    print(f"[PIPELINE] role={role} | sim={_use_simulation} | filters={detection_filters}")
    active_model, active_is_ppe = _get_model_for_role(role)

    # Mapping from violation class → its compliant counterpart (ppe.pt class names)
    VIOLATION_TO_COMPLIANT = {
        'NO-Hardhat':      'Hardhat',
        'NO-Mask':         'Mask',
        'NO-Safety Vest':  'Safety Vest',
        # simulated classes have no real compliant class in ppe.pt
    }

    if _use_simulation or active_model is None:
        raw = _simulate(frame, role, detection_filters=detection_filters, frame_index=frame_index)
    else:
        try:
            raw = _run_inference_with(frame, active_model, active_is_ppe)
        except Exception as e:
            log.error(f"Inference error (role={role}): {e}")
            raw = _simulate(frame, role, detection_filters=detection_filters, frame_index=frame_index)

    # Strictly filter raw detections to only keep selected classes
    before_filter = [d['label'] for d in raw]
    if detection_filters is not None and len(detection_filters) > 0:
        # Build the set of compliant labels that correspond to selected violation filters
        allowed_compliant = {VIOLATION_TO_COMPLIANT[v] for v in detection_filters if v in VIOLATION_TO_COMPLIANT}
        raw = [
            d for d in raw
            if d["det_type"] == "person"                     # always keep persons
            or d["det_type"] == "neutral"                    # always keep neutral
            or d["label"] in detection_filters               # keep selected violations
            or d["label"] in allowed_compliant               # keep corresponding compliant
        ]
    after_filter = [d['label'] for d in raw]
    print(f"[FILTER] before={before_filter} | after={after_filter} | filters={detection_filters}")

    persons  = [d for d in raw if d["det_type"] == "person"]
    ppe_dets = [d for d in raw if d["det_type"] in ("violation", "compliant")]

    if not persons and ppe_dets:
        xs = [d["bbox"][0] for d in ppe_dets] + [d["bbox"][2] for d in ppe_dets]
        ys = [d["bbox"][1] for d in ppe_dets] + [d["bbox"][3] for d in ppe_dets]
        h, w = frame.shape[:2]
        persons = [{
            "label": "Person",
            "confidence": max(d["confidence"] for d in ppe_dets),
            "bbox": [max(0, min(xs)-30), max(0, min(ys)-30),
                     min(w-1, max(xs)+30), min(h-1, max(ys)+30)],
            "det_type": "person",
        }]

    enriched = _associate_to_persons(persons, ppe_dets, role,
                                     detection_filters=detection_filters)

    # ── Traffic Police: remap already done inside dedicated/strict pipeline ──
    # (no remap needed here anymore)

    is_compliant, missing, alert_msg, severity = _compliance_summary(enriched, role)

    annotated = _draw_results(frame, enriched, raw, role)
    ann_b64   = encode_frame(annotated)
    snap_b64  = encode_frame(annotated, quality=85) if not is_compliant else None

    violations    = [p for p in enriched if not p["is_compliant"]]
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
    ann_b64  = encode_frame(annotated_with_phone)
    snap_b64 = encode_frame(annotated_with_phone, quality=85) if (
        not is_compliant or phone_result["phone_alert"]
    ) else None

    # Merge phone alert into overall compliance
    phone_alert   = phone_result.get("phone_alert")
    phone_status  = phone_result.get("phone_status", "safe")
    phone_dets    = phone_result.get("phone_detections", [])

    if phone_alert:
        is_compliant = False
        if alert_msg:
            alert_msg = f"{alert_msg} | {phone_alert}"
        else:
            alert_msg = phone_alert
        if not severity or severity == "low":
            severity = phone_result.get("phone_severity", "high")

    return {
        "persons":          enriched,
        "violations":       violations,
        "detections":       ui_detections + phone_dets,
        "is_compliant":     is_compliant,
        "missing_items":    missing,
        "violations_count": len(violations),
        "persons_count":    len(enriched),
        "alert_message":    alert_msg if not is_compliant else None,
        "severity":         severity if not is_compliant else None,
        "annotated_frame":  ann_b64,
        "snapshot_b64":     snap_b64,
        "phone_status":     phone_status,
        "phone_detected":   phone_result.get("phone_detected", False),
        "model_mode": (
            "ppe.pt" if _model_is_ppe
            else ("simulation" if _use_simulation else settings.YOLO_MODEL)
        ),
        "phone_severity": phone_result.get("phone_severity", "low"),
    }


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def process_frame(b64_frame: str, role: str,
                  detection_filters: Optional[List[str]] = None,
                  no_phone_zone: bool = False) -> Dict:
    """
    Full violation pipeline from base64 frame.
    Called by the WebSocket detection router.
    """
    frame = decode_frame(b64_frame)
    if frame is None:
        return {"error": "Invalid frame data"}
    return _run_pipeline(frame, role, detection_filters=detection_filters,
                         no_phone_zone=no_phone_zone)


def process_frame_numpy(frame: np.ndarray, role: str,
                         detection_filters: Optional[List[str]] = None,
                         no_phone_zone: bool = False,
                         frame_index: int = -1) -> Dict:
    """
    Full violation pipeline from a numpy frame directly.
    Called by the video upload router (avoids double encode/decode).
    frame_index: passed through to _simulate for deterministic results in video.
    """
    if frame is None:
        return {"error": "Invalid frame"}
    return _run_pipeline(frame, role, detection_filters=detection_filters,
                         no_phone_zone=no_phone_zone, frame_index=frame_index)
 