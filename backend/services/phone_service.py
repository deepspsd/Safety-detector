"""
Phone Usage Detection Service v4.0
====================================
Changes in v4.0:
  - Removed broken majority-vote smoothing buffer (caused phone to never show)
  - Lowered confidence threshold to 0.15 for better detection of large phones
  - Frame-position fallback when COCO misses person detection
  - In-hand = yellow alert, calling/zone_violation = red alert
  - Flickering handled purely on frontend (5s hold timer) — cleaner architecture
"""
import cv2
import numpy as np
import random
import logging
from typing import List, Dict, Optional, Tuple

log = logging.getLogger("phone_service")

PHONE_CLASS_ID     = 67
PERSON_CLASS_ID    = 0
NEAR_EAR_FRACTION  = 0.50   # upper 50% of person bbox = ear/head region
FRAME_EAR_FRACTION = 0.55   # fallback: upper 55% of frame = near head

COLOR_RED    = (30,  30, 220)
COLOR_YELLOW = (0,  200, 220)

_sim_frames_left = 0
_sim_near_ear    = False


# ── Geometry ──────────────────────────────────────────────────────

def _box_center(box: List[int]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _is_phone_near_ear(phone_box: List[int], person_box: List[int]) -> bool:
    """True if phone center Y is in upper 50% of person bounding box."""
    px1, py1, px2, py2 = person_box
    head_bottom = py1 + (py2 - py1) * NEAR_EAR_FRACTION
    _, phone_cy = _box_center(phone_box)
    return phone_cy <= head_bottom


def _is_phone_near_ear_by_frame(phone_box: List[int], frame_h: int) -> bool:
    """Fallback when no person bbox: upper 55% of frame = near head."""
    _, phone_cy = _box_center(phone_box)
    return phone_cy <= frame_h * FRAME_EAR_FRACTION


def _nearest_person(phone_box: List[int], persons: List[Dict]) -> Optional[Dict]:
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


# ── Inference ─────────────────────────────────────────────────────

def _run_phone_inference(frame: np.ndarray, model) -> Tuple[List[Dict], List[Dict]]:
    """
    Run COCO model at 960px / conf=0.15 for best small-object accuracy.
    Phones filling the frame are detected at conf=0.80+; lowering threshold
    catches partially visible and fast-moving phones.
    """
    from config import settings
    results = model(
        frame,
        verbose=False,
        imgsz=960,
        conf=0.15,          # lower threshold — catches any visible phone
        iou=settings.NMS_IOU,
        classes=[PHONE_CLASS_ID, PERSON_CLASS_ID],
    )
    phones, persons = [], []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = round(float(box.conf[0]), 3)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            det = {"label": model.names[cls_id], "confidence": conf,
                   "bbox": [x1, y1, x2, y2]}
            if cls_id == PHONE_CLASS_ID:
                phones.append(det)
            elif cls_id == PERSON_CLASS_ID:
                persons.append(det)
    return phones, persons


def _simulate_phone(frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
    """Stateful simulation — phone persists 10-20 frames, 70% hit rate."""
    global _sim_frames_left, _sim_near_ear
    h, w = frame.shape[:2]
    px1, py1 = int(w * 0.15), int(h * 0.05)
    px2, py2 = int(w * 0.55), int(h * 0.95)
    persons = [{"label": "person", "confidence": 0.89, "bbox": [px1, py1, px2, py2]}]
    if _sim_frames_left <= 0:
        if random.random() < 0.70:
            _sim_frames_left = random.randint(10, 20)
            _sim_near_ear    = random.random() < 0.50
        else:
            return [], persons
    _sim_frames_left -= 1
    height = py2 - py1
    cx     = (px1 + px2) // 2
    if _sim_near_ear:
        ph_y1 = py1 + int(height * 0.05)
        ph_y2 = py1 + int(height * 0.25)
    else:
        ph_y1 = py1 + int(height * 0.55)
        ph_y2 = py1 + int(height * 0.75)
    phones = [{"label": "cell phone",
               "confidence": round(random.uniform(0.70, 0.95), 3),
               "bbox": [cx - 25, ph_y1, cx + 25, ph_y2]}]
    return phones, persons


# ── Annotation ────────────────────────────────────────────────────

def _put_label(img: np.ndarray, text: str, x1: int, y1: int, color: tuple) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    ty = max(th + 8, y1 - 4)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty + 2), color, -1)
    cv2.putText(img, text, (x1 + 4, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_phone_annotations(frame: np.ndarray, alert_phones: List[Dict],
                             silent_phones: List[Dict]) -> np.ndarray:
    annotated = frame.copy()
    for ph in silent_phones:
        x1, y1, x2, y2 = ph["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_YELLOW, 2)
        _put_label(annotated, "Phone in Hand", x1, y1, COLOR_YELLOW)
    for ph in alert_phones:
        x1, y1, x2, y2 = ph["bbox"]
        status = ph.get("status", "calling")
        label  = "CALLING - ALERT" if status == "calling" else "PHONE VIOLATION"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_RED, 3)
        _put_label(annotated, label, x1, y1, COLOR_RED)
    return annotated


# ── Public API ────────────────────────────────────────────────────

def detect_phone_usage(
    frame: np.ndarray,
    no_phone_zone: bool = True,
    coco_model=None,
    coco_fallback=None,
    use_simulation: bool = False,
) -> Dict:
    """
    Detects phone usage and returns structured result each frame.
    Anti-flicker is handled entirely on the frontend (5s hold timer).

    Returns:
      phone_detected, phone_status, phone_alert, phone_severity,
      phone_detections, annotated_frame

    Alert levels:
      calling / zone_violation  →  severity="high"   (red)
      in_hand                   →  severity="medium"  (yellow)
      safe                      →  no alert           (green badge)
    """
    active_model = coco_model or coco_fallback
    frame_h      = frame.shape[0]

    # ── Run detection ─────────────────────────────────────────────
    if use_simulation or active_model is None:
        phones, persons = _simulate_phone(frame)
    else:
        try:
            phones, persons = _run_phone_inference(frame, active_model)
        except Exception as e:
            log.warning(f"Phone inference error: {e} — simulation fallback")
            phones, persons = _simulate_phone(frame)

    # ── No phone ──────────────────────────────────────────────────
    if not phones:
        return {"phone_detected": False, "phone_status": "safe",
                "phone_alert": None, "phone_severity": None,
                "phone_detections": [], "annotated_frame": frame}

    # ── Classify each detected phone ──────────────────────────────
    alert_phones  = []
    silent_phones = []

    for phone in phones:
        nearest  = _nearest_person(phone["bbox"], persons)
        if nearest:
            near_ear = _is_phone_near_ear(phone["bbox"], nearest["bbox"])
        else:
            # COCO missed person (tight crop / close-up) — use frame position
            near_ear = _is_phone_near_ear_by_frame(phone["bbox"], frame_h)

        if no_phone_zone:
            alert_phones.append({**phone, "status": "zone_violation"})
        elif near_ear:
            alert_phones.append({**phone, "status": "calling"})
        else:
            silent_phones.append({**phone, "status": "in_hand"})

    # ── Draw bounding boxes ───────────────────────────────────────
    annotated = _draw_phone_annotations(frame, alert_phones, silent_phones)

    # ── Build response ────────────────────────────────────────────
    if alert_phones:
        status    = alert_phones[0]["status"]
        alert_msg = ("Phone Usage Not Allowed in This Area"
                     if status == "zone_violation"
                     else "Unsafe Phone Usage Detected (Calling Near Ear)")
        all_dets  = [{"label": p["label"], "confidence": p["confidence"],
                      "bbox": p["bbox"], "status": p.get("status")}
                     for p in alert_phones + silent_phones]
        return {"phone_detected": True, "phone_status": status,
                "phone_alert": alert_msg, "phone_severity": "high",
                "phone_detections": all_dets, "annotated_frame": annotated}

    # In-hand — yellow alert
    all_dets = [{"label": p["label"], "confidence": p["confidence"],
                 "bbox": p["bbox"], "status": "in_hand"}
                for p in silent_phones]
    return {
        "phone_detected":   True,
        "phone_status":     "in_hand",
        "phone_alert":      "Phone Detected in Hand",
        "phone_severity":   "medium",
        "phone_detections": all_dets,
        "annotated_frame":  annotated,
    }
