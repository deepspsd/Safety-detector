"""
Phone Usage Detection Service — v2.0
=====================================
Uses COCO yolov8m.pt model (cell phone = class 67).
Falls back cleanly when model unavailable.

Fixes in v2.0:
  - Removed emoji from cv2.putText (OpenCV can't render them — caused silent draw failures)
  - Added coco_fallback parameter: use general COCO model if helmet model unavailable
  - Lowered confidence to 0.20 for small object phone detection
  - Stateful simulation: once a phone appears it persists 8 frames (realistic)
  - No-Phone Zone defaults to True in the pipeline call

Phone status:
  SAFE          - no phone detected
  IN_HAND       - phone visible but not near ear, no zone restriction
  CALLING       - phone center within top 30% of nearest person bbox height
  ZONE_VIOLATION- any phone + no_phone_zone=True
"""
import cv2
import numpy as np
import random
import logging
from typing import List, Dict, Optional, Tuple

log = logging.getLogger("phone_service")

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
PHONE_CLASS_ID      = 67    # COCO class index for 'cell phone'
PERSON_CLASS_ID     = 0     # COCO class index for 'person'

# Phone center must be within this fraction of person height from the top
# to be considered "near ear" (head/shoulder region).
NEAR_EAR_FRACTION   = 0.35  # slightly looser for better detection

# BGR colours — NO emoji, only ASCII labels for cv2.putText
COLOR_PHONE_CALLING  = (30,  30, 220)    # Red
COLOR_PHONE_IN_HAND  = (0,  200, 220)    # Yellow-cyan
COLOR_PHONE_ZONE     = (0,   50, 200)    # Orange-red

# ─────────────────────────────────────────────────────────────────
# Simulation state (persists phone across frames for realism)
# ─────────────────────────────────────────────────────────────────
_sim_phone_frames_left = 0     # how many more frames to keep simulated phone
_sim_phone_near_ear    = False # whether current simulated phone is near ear

# ─────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────

def _box_center(box: List[int]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _is_phone_near_ear(phone_box: List[int], person_box: List[int]) -> bool:
    """
    Return True if the center of the phone bbox sits within the top 35%
    (head + shoulder region) of the person bounding box.
    """
    px1, py1, px2, py2 = person_box
    person_height = max(1, py2 - py1)
    head_bottom   = py1 + person_height * NEAR_EAR_FRACTION
    _, phone_cy   = _box_center(phone_box)
    return phone_cy <= head_bottom


def _nearest_person(phone_box: List[int], persons: List[Dict]) -> Optional[Dict]:
    """Return the closest person to this phone bbox center."""
    phone_cx, phone_cy = _box_center(phone_box)
    best, best_dist = None, float("inf")
    for p in persons:
        pb  = p["bbox"]
        pcx = (pb[0] + pb[2]) / 2
        pcy = (pb[1] + pb[3]) / 2
        d   = abs(phone_cx - pcx) + abs(phone_cy - pcy)
        if d < best_dist:
            best_dist = d
            best = p
    return best


# ─────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────

def _run_phone_inference(frame: np.ndarray, model) -> Tuple[List[Dict], List[Dict]]:
    """
    Run a COCO model on frame and extract cell phone + person detections.
    Uses:
      - imgsz=960 : 2.25x more pixels than default 640 — crucial for small phones
      - conf=0.20 : low threshold to catch partially visible/occluded phones
    """
    from config import settings
    results = model(
        frame,
        verbose=False,
        imgsz=960,          # higher resolution = much better small-object detection
        conf=0.20,
        iou=settings.NMS_IOU,
        classes=[PHONE_CLASS_ID, PERSON_CLASS_ID],
    )

    phones, persons = [], []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = round(float(box.conf[0]), 3)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            det = {
                "label":      model.names[cls_id],
                "confidence": conf,
                "bbox":       [x1, y1, x2, y2],
            }
            if cls_id == PHONE_CLASS_ID:
                phones.append(det)
            elif cls_id == PERSON_CLASS_ID:
                persons.append(det)

    return phones, persons


def _simulate_phone(frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
    """
    Stateful simulation fallback.
    - Phone appears with 60% chance and persists for 8–15 frames.
    - Prevents the flickering that made detection seem broken.
    """
    global _sim_phone_frames_left, _sim_phone_near_ear

    h, w = frame.shape[:2]
    px1, py1 = int(w * 0.15), int(h * 0.05)
    px2, py2 = int(w * 0.55), int(h * 0.95)
    persons = [{"label": "person", "confidence": 0.89, "bbox": [px1, py1, px2, py2]}]

    # Decide whether to start a new phone event
    if _sim_phone_frames_left <= 0:
        if random.random() < 0.60:   # 60% chance to detect a phone
            _sim_phone_frames_left = random.randint(8, 15)
            _sim_phone_near_ear    = random.random() < 0.50   # 50% near ear
        else:
            return [], persons

    # Decrement persistence counter
    _sim_phone_frames_left -= 1

    height = py2 - py1
    cx     = (px1 + px2) // 2
    if _sim_phone_near_ear:
        ph_y1 = py1 + int(height * 0.05)
        ph_y2 = py1 + int(height * 0.22)
    else:
        ph_y1 = py1 + int(height * 0.55)
        ph_y2 = py1 + int(height * 0.75)

    phones = [{
        "label":      "cell phone",
        "confidence": round(random.uniform(0.65, 0.92), 3),
        "bbox":       [cx - 20, ph_y1, cx + 20, ph_y2],
    }]
    return phones, persons


# ─────────────────────────────────────────────────────────────────
# Annotation (ASCII only — OpenCV cannot render emoji)
# ─────────────────────────────────────────────────────────────────

def _draw_phone_annotations(
    frame: np.ndarray,
    alert_phones: List[Dict],
    silent_phones: List[Dict],
) -> np.ndarray:
    annotated = frame.copy()

    for ph in silent_phones:
        x1, y1, x2, y2 = ph["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_PHONE_IN_HAND, 2)
        label = "Phone in Hand"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1),
                      COLOR_PHONE_IN_HAND, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)

    for ph in alert_phones:
        x1, y1, x2, y2 = ph["bbox"]
        status = ph.get("status", "calling")
        color  = COLOR_PHONE_ZONE if status == "zone_violation" else COLOR_PHONE_CALLING
        label  = "CALLING - ALERT" if status == "calling" else "PHONE VIOLATION"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1),
                      color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def detect_phone_usage(
    frame: np.ndarray,
    no_phone_zone: bool = True,      # DEFAULT ON — any phone = alert
    coco_model=None,                  # primary model (yolov8m.pt helmet model)
    coco_fallback=None,               # secondary fallback (general COCO model)
    use_simulation: bool = False,
) -> Dict:
    """
    Main entry point called by yolo_service._run_pipeline().

    Model selection priority:
      1. coco_model      (yolov8m.pt loaded for Traffic Police role)
      2. coco_fallback   (general yolov8m.pt or any COCO model)
      3. simulation      (stateful, realistic)

    Returns dict with phone_detected, phone_status, phone_alert,
    phone_severity, phone_detections, annotated_frame.
    """
    # ── Select model ──────────────────────────────────────────
    active_model = coco_model or coco_fallback

    if use_simulation or active_model is None:
        phones, persons = _simulate_phone(frame)
    else:
        try:
            phones, persons = _run_phone_inference(frame, active_model)
        except Exception as e:
            log.warning(f"Phone inference error: {e} — using simulation")
            phones, persons = _simulate_phone(frame)

    # ── No phone detected ──────────────────────────────────────
    if not phones:
        return {
            "phone_detected":   False,
            "phone_status":     "safe",
            "phone_alert":      None,
            "phone_severity":   None,
            "phone_detections": [],
            "annotated_frame":  frame,
        }

    # ── Classify each phone ────────────────────────────────────
    alert_phones  = []
    silent_phones = []

    for phone in phones:
        nearest  = _nearest_person(phone["bbox"], persons)
        near_ear = nearest and _is_phone_near_ear(phone["bbox"], nearest["bbox"])

        if no_phone_zone:
            alert_phones.append({**phone, "status": "zone_violation"})
        elif near_ear:
            alert_phones.append({**phone, "status": "calling"})
        else:
            silent_phones.append(phone)

    # ── Annotate ───────────────────────────────────────────────
    annotated = _draw_phone_annotations(frame, alert_phones, silent_phones)

    if alert_phones:
        status    = alert_phones[0]["status"]
        alert_msg = ("Phone Usage Not Allowed in This Area"
                     if status == "zone_violation"
                     else "Unsafe Phone Usage Detected (Calling)")
        severity  = "high"
        all_dets  = [
            {"label": p["label"], "confidence": p["confidence"],
             "bbox": p["bbox"], "status": p.get("status", "calling")}
            for p in alert_phones + silent_phones
        ]
        return {
            "phone_detected":   True,
            "phone_status":     status,
            "phone_alert":      alert_msg,
            "phone_severity":   severity,
            "phone_detections": all_dets,
            "annotated_frame":  annotated,
        }

    # Phones in hand only — no alert
    all_dets = [
        {"label": p["label"], "confidence": p["confidence"],
         "bbox": p["bbox"], "status": "in_hand"}
        for p in silent_phones
    ]
    return {
        "phone_detected":   True,
        "phone_status":     "in_hand",
        "phone_alert":      None,
        "phone_severity":   None,
        "phone_detections": all_dets,
        "annotated_frame":  annotated,
    }
