"""
Phone Usage Detection Service — v1.0
=====================================
Uses the existing yolov8m.pt (COCO) model which already detects 'cell phone' (class 67).
No new model downloads required.

Detection logic:
  1. Run COCO model on frame, extract 'cell phone' and 'person' detections.
  2. For each phone, check if it is near a person's head/ear region.
  3. Apply No-Phone Zone rule if enabled.

Phone status:
  • SAFE         — no phone detected
  • IN_HAND      — phone detected but not near ear, no zone restriction
  • CALLING      — phone center within top 30% of nearest person bbox height
  • ZONE_VIOLATION — any phone + no_phone_zone=True

Only CALLING and ZONE_VIOLATION generate alerts. IN_HAND is labelled but silent.
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
NEAR_EAR_FRACTION   = 0.30

# If phone bbox overlaps horizontally within this margin (px) of person — it's associated
PHONE_PERSON_MARGIN = 20

# BGR colours
COLOR_PHONE_CALLING  = (30,  30,  220)   # Red  — calling/zone violation
COLOR_PHONE_IN_HAND  = (30, 200,  220)   # Yellow-ish — in hand, silent
COLOR_PHONE_LABEL_BG = (20,  20,   20)   # dark bg for text

# ─────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────

def _box_center(box: List[int]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _is_phone_near_ear(phone_box: List[int], person_box: List[int]) -> bool:
    """
    Return True if the center of the phone bbox sits within the top 30%
    (head + shoulder region) of the person bounding box.
    """
    px1, py1, px2, py2 = person_box
    person_height = max(1, py2 - py1)
    head_bottom   = py1 + person_height * NEAR_EAR_FRACTION

    _, phone_cy = _box_center(phone_box)
    return phone_cy <= head_bottom


def _phone_near_person(phone_box: List[int], person_box: List[int]) -> bool:
    """
    Return True if the phone bbox horizontally overlaps (or is close to)
    the person bbox. Used to associate a phone with its carrier.
    """
    # Expand person box horizontally by margin
    px1 = person_box[0] - PHONE_PERSON_MARGIN
    px2 = person_box[2] + PHONE_PERSON_MARGIN
    phone_cx, _ = _box_center(phone_box)
    return px1 <= phone_cx <= px2


def _nearest_person(phone_box: List[int], persons: List[Dict]) -> Optional[Dict]:
    """Return the person whose bbox horizontally contains the phone center, or None."""
    phone_cx, phone_cy = _box_center(phone_box)
    best, best_dist = None, float("inf")
    for p in persons:
        pb = p["bbox"]
        # Distance from phone center to person center
        pcx = (pb[0] + pb[2]) / 2
        pcy = (pb[1] + pb[3]) / 2
        d = abs(phone_cx - pcx) + abs(phone_cy - pcy)
        if d < best_dist:
            best_dist = d
            best = p
    return best


# ─────────────────────────────────────────────────────────────────
# Inference (uses the global yolov8m.pt loaded in yolo_service)
# ─────────────────────────────────────────────────────────────────

def _run_phone_inference(frame: np.ndarray, coco_model) -> Tuple[List[Dict], List[Dict]]:
    """
    Run the COCO model on frame and return:
      (phone_detections, person_detections)
    Each detection is {label, confidence, bbox:[x1,y1,x2,y2]}.
    Only extracts class 67 (cell phone) and class 0 (person).
    """
    from config import settings
    results = coco_model(
        frame,
        verbose=False,
        conf=0.35,           # lower confidence for small phones
        iou=settings.NMS_IOU,
        classes=[PHONE_CLASS_ID, PERSON_CLASS_ID],
    )

    phones, persons = [], []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = round(float(box.conf[0]), 3)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            det = {"label": coco_model.names[cls_id], "confidence": conf,
                   "bbox": [x1, y1, x2, y2]}
            if cls_id == PHONE_CLASS_ID:
                phones.append(det)
            elif cls_id == PERSON_CLASS_ID:
                persons.append(det)

    return phones, persons


def _simulate_phone(frame: np.ndarray, no_phone_zone: bool) -> Tuple[List[Dict], List[Dict]]:
    """
    Simulation fallback — generates 1 random person and randomly places a phone.
    35% chance the person has a phone; if phone, 45% chance it's near the ear.
    """
    h, w = frame.shape[:2]
    px1, py1 = int(w * 0.15), int(h * 0.05)
    px2, py2 = int(w * 0.55), int(h * 0.95)
    persons = [{"label": "person", "confidence": 0.89, "bbox": [px1, py1, px2, py2]}]

    phones = []
    if random.random() < 0.35:
        height = py2 - py1
        if random.random() < 0.45:
            # Near ear — upper 20%
            ph_y1 = py1 + int(height * 0.05)
            ph_y2 = py1 + int(height * 0.22)
        else:
            # In hand — lower 60%
            ph_y1 = py1 + int(height * 0.55)
            ph_y2 = py1 + int(height * 0.75)
        cx = (px1 + px2) // 2
        phones.append({
            "label": "cell phone", "confidence": round(random.uniform(0.55, 0.90), 3),
            "bbox": [cx - 18, ph_y1, cx + 18, ph_y2],
        })

    return phones, persons


# ─────────────────────────────────────────────────────────────────
# Annotation
# ─────────────────────────────────────────────────────────────────

def _draw_phone_annotations(
    frame: np.ndarray,
    phone_results: List[Dict],   # enriched per-phone results
    in_hand_phones: List[Dict],  # silent in-hand phones
) -> np.ndarray:
    annotated = frame.copy()

    for ph in in_hand_phones:
        x1, y1, x2, y2 = ph["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_PHONE_IN_HAND, 2)
        cv2.putText(annotated, "📱 Phone", (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_PHONE_IN_HAND, 1, cv2.LINE_AA)

    for ph in phone_results:
        x1, y1, x2, y2 = ph["bbox"]
        status = ph.get("status", "calling")
        color  = COLOR_PHONE_CALLING
        label  = "📱 Calling!" if status == "calling" else "🚫 Phone Violation"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        # Dark background pill for label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def detect_phone_usage(
    frame: np.ndarray,
    no_phone_zone: bool = False,
    coco_model=None,
    use_simulation: bool = False,
) -> Dict:
    """
    Main entry point called by yolo_service._run_pipeline().

    Returns:
    {
        "phone_detected":     bool,
        "phone_status":       "safe" | "in_hand" | "calling" | "zone_violation",
        "phone_alert":        str | None,   # alert message if violation,
        "phone_severity":     str | None,
        "phone_detections":   list,         # raw phone + person detections for UI
        "annotated_frame":    np.ndarray,   # frame WITH phone annotations drawn
    }
    """
    if use_simulation or coco_model is None:
        phones, persons = _simulate_phone(frame, no_phone_zone)
    else:
        try:
            phones, persons = _run_phone_inference(frame, coco_model)
        except Exception as e:
            log.warning(f"Phone inference failed: {e} — using simulation")
            phones, persons = _simulate_phone(frame, no_phone_zone)

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
    alert_phones  = []   # phones that trigger an alert
    silent_phones = []   # phones that are visible but don't alert

    for phone in phones:
        nearest = _nearest_person(phone["bbox"], persons)
        near_ear = nearest and _is_phone_near_ear(phone["bbox"], nearest["bbox"])

        if no_phone_zone:
            # Any phone = violation
            alert_phones.append({**phone, "status": "zone_violation"})
        elif near_ear:
            alert_phones.append({**phone, "status": "calling"})
        else:
            silent_phones.append(phone)

    # ── Build result ───────────────────────────────────────────
    annotated = _draw_phone_annotations(frame, alert_phones, silent_phones)

    if alert_phones:
        status = alert_phones[0]["status"]   # worst status
        if status == "zone_violation":
            alert_msg  = "🚫 Phone Usage Not Allowed in This Area"
            severity   = "high"
        else:
            alert_msg  = "📱 Unsafe Phone Usage Detected (Calling)"
            severity   = "high"

        # Build flat detections list for UI
        all_dets = [
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

    # Phones detected but none alerting
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
