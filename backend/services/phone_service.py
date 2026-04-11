"""
Phone Usage Detection Service v3.0
====================================
Fixes in v3.0:
  - Frame-position fallback when COCO misses person detection:
    upper 55% of frame = near head/ear
  - NEAR_EAR_FRACTION raised to 0.50 (upper half of person = ear region)
  - In-hand phone now generates YELLOW alert (was completely silent before)
  - Severity: calling/zone_violation=high (red), in_hand=medium (yellow)
"""
import cv2
import numpy as np
import random
import logging
from typing import List, Dict, Optional, Tuple

log = logging.getLogger("phone_service")

PHONE_CLASS_ID    = 67
PERSON_CLASS_ID   = 0
NEAR_EAR_FRACTION  = 0.50   # upper 50% of person bbox = ear/head region
FRAME_EAR_FRACTION = 0.55   # fallback: upper 55% of frame = near head

COLOR_RED    = (30,  30, 220)
COLOR_YELLOW = (0,  200, 220)

_sim_frames_left = 0
_sim_near_ear    = False

# Rolling window buffer — holds last 6 statuses for smoothing
# Only report a new status if it appears in majority of recent frames.
_BUFFER_SIZE    = 6
_MAJORITY_VOTES = 4   # need 4/6 frames to confirm a status change
_status_buffer: list = []   # list of recent 'phone_status' strings
_last_result:   dict = {}   # last emitted result (reused during hold)


# ── Geometry ──────────────────────────────────────────────────────

def _box_center(box: List[int]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _is_phone_near_ear(phone_box: List[int], person_box: List[int]) -> bool:
    px1, py1, px2, py2 = person_box
    head_bottom = py1 + (py2 - py1) * NEAR_EAR_FRACTION
    _, phone_cy = _box_center(phone_box)
    return phone_cy <= head_bottom


def _is_phone_near_ear_by_frame(phone_box: List[int], frame_h: int) -> bool:
    """Fallback when no person bbox available — uses frame height."""
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
    from config import settings
    results = model(
        frame,
        verbose=False,
        imgsz=960,
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
            det = {"label": model.names[cls_id], "confidence": conf,
                   "bbox": [x1, y1, x2, y2]}
            if cls_id == PHONE_CLASS_ID:
                phones.append(det)
            elif cls_id == PERSON_CLASS_ID:
                persons.append(det)
    return phones, persons


def _simulate_phone(frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
    global _sim_frames_left, _sim_near_ear
    h, w = frame.shape[:2]
    px1, py1 = int(w * 0.15), int(h * 0.05)
    px2, py2 = int(w * 0.55), int(h * 0.95)
    persons = [{"label": "person", "confidence": 0.89, "bbox": [px1, py1, px2, py2]}]
    if _sim_frames_left <= 0:
        if random.random() < 0.60:
            _sim_frames_left = random.randint(8, 15)
            _sim_near_ear    = random.random() < 0.50
        else:
            return [], persons
    _sim_frames_left -= 1
    height = py2 - py1
    cx     = (px1 + px2) // 2
    if _sim_near_ear:
        ph_y1 = py1 + int(height * 0.05)
        ph_y2 = py1 + int(height * 0.22)
    else:
        ph_y1 = py1 + int(height * 0.55)
        ph_y2 = py1 + int(height * 0.75)
    phones = [{"label": "cell phone",
               "confidence": round(random.uniform(0.65, 0.92), 3),
               "bbox": [cx - 20, ph_y1, cx + 20, ph_y2]}]
    return phones, persons


# ── Annotation ────────────────────────────────────────────────────

def _put_label(img: np.ndarray, text: str, x1: int, y1: int, color: tuple) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    ty = max(th + 6, y1 - 4)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
    cv2.putText(img, text, (x1 + 3, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)


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
    Returns:
      phone_detected, phone_status, phone_alert, phone_severity,
      phone_detections, annotated_frame

    Alert levels:
      calling / zone_violation → severity="high"   (🔴 red)
      in_hand                  → severity="medium"  (🟡 yellow)
      safe                     → no alert            (🟢 green badge)
    """
    active_model = coco_model or coco_fallback
    frame_h      = frame.shape[0]

    if use_simulation or active_model is None:
        phones, persons = _simulate_phone(frame)
    else:
        try:
            phones, persons = _run_phone_inference(frame, active_model)
        except Exception as e:
            log.warning(f"Phone inference error: {e} — simulation fallback")
            phones, persons = _simulate_phone(frame)

    # ── Rolling-window smoothing ──────────────────────────────
    # Compute raw status this frame, then vote across recent frames.
    # This prevents single missed/extra detections from causing flicker.
    global _status_buffer, _last_result

    if not phones:
        raw_status = "safe"
    else:
        # Quick classify to get raw status
        frame_h   = frame.shape[0]
        has_alert = False
        has_hand  = False
        for phone in phones:
            nearest  = _nearest_person(phone["bbox"], persons)
            near_ear = (_is_phone_near_ear(phone["bbox"], nearest["bbox"])
                        if nearest
                        else _is_phone_near_ear_by_frame(phone["bbox"], frame_h))
            if no_phone_zone or near_ear:
                has_alert = True
            else:
                has_hand = True
        if has_alert:
            raw_status = "zone_violation" if no_phone_zone else "calling"
        else:
            raw_status = "in_hand"

    _status_buffer.append(raw_status)
    if len(_status_buffer) > _BUFFER_SIZE:
        _status_buffer.pop(0)

    # Vote: find which status appears most frequently
    from collections import Counter
    vote_counts  = Counter(_status_buffer)
    top_status, top_count = vote_counts.most_common(1)[0]

    # Only switch if top status has enough votes to be reliable
    STATUS_PRIORITY = {"zone_violation": 3, "calling": 3, "in_hand": 2, "safe": 1}
    if top_count >= _MAJORITY_VOTES or top_status == "safe":
        smooth_status = top_status
    else:
        # Not enough votes — keep last known non-safe status if available
        smooth_status = (_last_result.get("phone_status") or "safe")

    # If smooth result is safe, return safe immediately
    if smooth_status == "safe":
        _last_result = {"phone_detected": False, "phone_status": "safe",
                        "phone_alert": None, "phone_severity": None,
                        "phone_detections": [], "annotated_frame": frame}
        return _last_result

    # ── Build full result for non-safe status ─────────────────

    alert_phones  = []
    silent_phones = []

    for phone in phones:
        nearest  = _nearest_person(phone["bbox"], persons)
        if nearest:
            near_ear = _is_phone_near_ear(phone["bbox"], nearest["bbox"])
        else:
            # COCO missed person detection (tight crop / close-up)
            # Use frame-position heuristic instead
            near_ear = _is_phone_near_ear_by_frame(phone["bbox"], frame_h)

        if no_phone_zone:
            alert_phones.append({**phone, "status": "zone_violation"})
        elif near_ear:
            alert_phones.append({**phone, "status": "calling"})
        else:
            silent_phones.append({**phone, "status": "in_hand"})

    annotated = _draw_phone_annotations(frame, alert_phones, silent_phones)

    if alert_phones:
        status    = alert_phones[0]["status"]
        alert_msg = ("Phone Usage Not Allowed in This Area"
                     if status == "zone_violation"
                     else "Unsafe Phone Usage Detected (Calling Near Ear)")
        all_dets  = [{"label": p["label"], "confidence": p["confidence"],
                      "bbox": p["bbox"], "status": p.get("status")}
                     for p in alert_phones + silent_phones]
        _last_result = {"phone_detected": True, "phone_status": status,
                        "phone_alert": alert_msg, "phone_severity": "high",
                        "phone_detections": all_dets, "annotated_frame": annotated}
        return _last_result

    # In-hand only — yellow alert
    all_dets = [{"label": p["label"], "confidence": p["confidence"],
                 "bbox": p["bbox"], "status": "in_hand"}
                for p in silent_phones]
    _last_result = {
        "phone_detected":   True,
        "phone_status":     "in_hand",
        "phone_alert":      "Phone Detected in Hand",
        "phone_severity":   "medium",
        "phone_detections": all_dets,
        "annotated_frame":  annotated,
    }
    return _last_result
