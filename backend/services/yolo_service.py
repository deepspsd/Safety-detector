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
_model         = None   # general PPE model (ppe.pt)
_helmet_model  = None   # specialized helmet model for Traffic Police role
_use_simulation = False
_model_is_ppe   = False  # True when ppe.pt loaded (vs generic COCO)


def _get_model_for_role(role: str):
    """Return the correct model instance for the given role."""
    if role == "Traffic Police" and _helmet_model is not None:
        return _helmet_model, True   # (model, is_ppe)
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
    global _model, _helmet_model, _use_simulation, _model_is_ppe

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

    # ── 2. Load specialized helmet model for Traffic Police ─────
    log.info("Loading Traffic Police helmet model…")
    try:
        from ultralytics import YOLO
        import torch
        _orig2 = torch.load
        def _p2(*a, **kw): kw.setdefault("weights_only", False); return _orig2(*a, **kw)
        torch.load = _p2
        # yolov8m.pt: higher accuracy model for helmet (motorcycle/construction) detection.
        # Swap this path for a dedicated helmet .pt file if available.
        _helmet_model = YOLO("yolov8m.pt")
        torch.load = _orig2
        log.info("✅ Helmet model loaded: yolov8m.pt (Traffic Police role)")
        print("✅ Helmet model loaded for Traffic Police role (yolov8m.pt)")
    except Exception as e:
        try: torch.load = _orig2
        except Exception: pass
        log.warning(f"Helmet model unavailable: {e} — Traffic will use ppe.pt")
        _helmet_model = None
        print("⚠️  Helmet model unavailable — Traffic Police will use ppe.pt")


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
                if sim_v in assigned_violations:
                    human_label = VIOLATION_LABEL_MAP.get(sim_v, sim_v)
                    ppe_missing.append(sim_v)
                    violation_labels.append(human_label)
            else:
                # Real inference: ~25% estimated miss rate for non-native classes
                if random.random() < 0.25:
                    human_label = VIOLATION_LABEL_MAP.get(sim_v, sim_v)
                    ppe_missing.append(sim_v)
                    violation_labels.append(human_label)

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
              detection_filters: Optional[List[str]] = None) -> List[Dict]:
    """
    Generate realistic simulated detections for when ppe.pt is unavailable.
    Produces 1–2 persons, each with randomised PPE status.
    Handles both native ppe.pt classes AND simulated classes (Gloves, Goggles).
    Respects detection_filters — only simulates selected PPE classes.
    """
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
            "confidence": round(random.uniform(0.82, 0.97), 3),
            "bbox":       [px, py, px2, py2],
            "det_type":   "person",
        })

        for req_v in all_req:
            is_violation = random.random() < 0.35   # 35% violation chance per item

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
                "confidence": round(random.uniform(0.65, 0.95), 3),
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


# ─────────────────────────────────────────────────────────────────
# Core pipeline helpers
# ─────────────────────────────────────────────────────────────────

def _run_pipeline(frame: np.ndarray, role: str,
                  detection_filters: Optional[List[str]] = None,
                  no_phone_zone: bool = False) -> Dict:
    """
    Shared pipeline used by both process_frame and process_frame_numpy.
    detection_filters: if provided, only these violation classes are evaluated.
    no_phone_zone: if True, any phone detection triggers an alert.
    """
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
        raw = _simulate(frame, role, detection_filters=detection_filters)
    else:
        try:
            raw = _run_inference_with(frame, active_model, active_is_ppe)
        except Exception as e:
            log.error(f"Inference error (role={role}): {e}")
            raw = _simulate(frame, role, detection_filters=detection_filters)

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
    # (yolo_service and phone_service are both in the same package).
    from services import phone_service as _phone_svc
    phone_result = _phone_svc.detect_phone_usage(
        frame=annotated,
        no_phone_zone=no_phone_zone,
        # Primary: yolov8m.pt loaded for Traffic Police (COCO, has class 67)
        coco_model=_helmet_model,
        # Fallback: use _model only if it's a COCO model (not the PPE-specific ppe.pt)
        coco_fallback=(_model if not _model_is_ppe else None),
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
            "Traffic:yolov8m" if (role == "Traffic Police" and _helmet_model is not None)
            else ("ppe.pt" if _model_is_ppe else ("simulation" if _use_simulation else settings.YOLO_MODEL))
        ),
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
                         no_phone_zone: bool = False) -> Dict:
    """
    Full violation pipeline from a numpy frame directly.
    Called by the video upload router (avoids double encode/decode).
    """
    if frame is None:
        return {"error": "Invalid frame"}
    return _run_pipeline(frame, role, detection_filters=detection_filters,
                         no_phone_zone=no_phone_zone)
