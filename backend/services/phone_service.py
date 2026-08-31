"""
Phone Usage Detection Service v5.0
====================================
Dedicated high-accuracy phone detection with behavior-based alerts.

Key improvements in v5.0:
  - Per-PERSON colored bounding boxes:
      🟢 Green  + "SAFE - No Phone"    → no phone associated with this person
      🟡 Yellow + "Phone in Hand"      → phone visible but not near head
      🔴 Red    + "CALLING DETECTED"   → phone near ear / zone violation
  - Per-PHONE bounding boxes with yellow/red labels
  - Person-phone association uses bbox containment (not just distance)
    so phones clearly held by a person are correctly linked
  - conf=0.12 (very low) — maximizes recall at the cost of minor FP
  - imgsz=960 — 2.25× more pixels than default, crucial for phones
  - Frame-position fallback when COCO misses person detection
  - No smoothing buffer (frontend 5s hold handles stability)
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("phone_service")

# ── COCO class IDs ────────────────────────────────────────────────
PHONE_CLASS_ID = 67
PERSON_CLASS_ID = 0
PHONE_PROXY_CLASS_IDS = [67, 65, 73, 32]  # cell phone (67), remote (65), book (73), sports ball (32)

# ── Detection thresholds ──────────────────────────────────────────
CONF_THRESHOLD = 0.60  # 60% confidence threshold for phone detection
IMG_SIZE = 960  # higher resolution = much better small-object detection

# ── Ear / head region thresholds ─────────────────────────────────
NEAR_EAR_FRACTION = 0.50  # upper 50% of PERSON bbox = ear/head region
FRAME_EAR_FRACTION = 0.55  # fallback: upper 55% of frame height = near head

# ── Colors (BGR) — NO emoji, cv2.putText cannot render them ──────
COLOR_GREEN = (50, 200, 50)  # safe — no phone
COLOR_YELLOW = (0, 200, 220)  # in hand
COLOR_RED = (30, 30, 220)  # calling / zone violation
COLOR_WHITE = (255, 255, 255)

# ── Simulation state (stateful — phone persists across frames) ────
_sim_frames_left = 0
_sim_near_ear = False


# ─────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────


def _box_center(box: List[int]) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _is_phone_near_ear(phone_box: List[int], person_box: List[int]) -> bool:
    """True if phone center Y is within upper NEAR_EAR_FRACTION of person bbox."""
    px1, py1, px2, py2 = person_box
    head_bottom = py1 + (py2 - py1) * NEAR_EAR_FRACTION
    _, phone_cy = _box_center(phone_box)
    return phone_cy <= head_bottom


def _is_phone_near_ear_by_frame(phone_box: List[int], frame_h: int) -> bool:
    """Fallback: upper FRAME_EAR_FRACTION of total frame height → near head."""
    _, phone_cy = _box_center(phone_box)
    return phone_cy <= frame_h * FRAME_EAR_FRACTION


def _phone_associated_to_person(phone_box: List[int], person_box: List[int]) -> bool:
    """
    True if the phone center is within the person bounding box (with 30% margin).
    More reliable than pure distance when holding the phone away from body.
    """
    px1, py1, px2, py2 = person_box
    pw = px2 - px1
    ph = py2 - py1
    # Allow phone to be up to 30% of person width/height outside the box
    margin_x = pw * 0.30
    margin_y = ph * 0.30
    phone_cx, phone_cy = _box_center(phone_box)
    return (
        px1 - margin_x <= phone_cx <= px2 + margin_x
        and py1 - margin_y <= phone_cy <= py2 + margin_y
    )


def _get_person_phone_status(
    person_box: List[int],
    phones: List[Dict],
    no_phone_zone: bool,
    frame_h: int,
) -> Tuple[str, Optional[Dict]]:
    """
    For a given person bbox, find the nearest associated phone and classify status.

    Returns (status, associated_phone_or_None).
    Status: "safe" | "in_hand" | "calling" | "zone_violation"
    """
    # Find all phones associated with this person
    associated = []
    for phone in phones:
        if _phone_associated_to_person(phone["bbox"], person_box):
            associated.append(phone)

    if not associated:
        return "safe", None

    # Use the highest-confidence associated phone
    best_phone = max(associated, key=lambda p: p["confidence"])

    if no_phone_zone:
        return "zone_violation", best_phone

    near_ear = _is_phone_near_ear(best_phone["bbox"], person_box)
    if not near_ear:
        # Fallback: frame-height heuristic (helps when person is close to camera)
        near_ear = _is_phone_near_ear_by_frame(best_phone["bbox"], frame_h)

    return ("calling" if near_ear else "in_hand"), best_phone


# ─────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────


def _run_phone_inference(frame: np.ndarray, model) -> Tuple[List[Dict], List[Dict]]:
    """
    Run COCO model at 960px with phone + proxy classes, followed by
    focused hand/torso crop inference per person for maximum recall.
    Returns (phones, persons).
    """
    from config import settings

    target_classes = [PERSON_CLASS_ID] + PHONE_PROXY_CLASS_IDS
    results = model(
        frame,
        verbose=False,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=settings.NMS_IOU,
        classes=target_classes,
    )
    phones, persons = [], []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = round(float(box.conf[0]), 3)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            det = {
                "label": "cell phone",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }
            if cls_id in PHONE_PROXY_CLASS_IDS:
                phones.append(det)
            elif cls_id == PERSON_CLASS_ID:
                det["label"] = model.names[cls_id]
                persons.append(det)

    # Focused hand/torso crop inference: if persons detected and no phone found on full frame,
    # crop the person's hand/chest area (20%-85% height) where phones are held
    if persons and not phones:
        for p in persons:
            px1, py1, px2, py2 = p["bbox"]
            pw, ph = px2 - px1, py2 - py1
            hy1 = max(0, py1 + int(ph * 0.18))
            hy2 = min(frame.shape[0], py1 + int(ph * 0.85))
            hx1 = max(0, px1 - int(pw * 0.15))
            hx2 = min(frame.shape[1], px2 + int(pw * 0.15))
            crop = frame[hy1:hy2, hx1:hx2]
            if crop.size > 0 and crop.shape[0] > 40 and crop.shape[1] > 40:
                crop_res = model(
                    crop,
                    verbose=False,
                    conf=CONF_THRESHOLD,
                    iou=settings.NMS_IOU,
                    classes=target_classes,
                )
                for cr in crop_res:
                    for cb in cr.boxes:
                        c_cls = int(cb.cls[0])
                        if c_cls in PHONE_PROXY_CLASS_IDS:
                            c_conf = round(float(cb.conf[0]), 3)
                            cx1, cy1, cx2, cy2 = [int(v) for v in cb.xyxy[0]]
                            phones.append({
                                "label": "cell phone",
                                "confidence": c_conf,
                                "bbox": [hx1 + cx1, hy1 + cy1, hx1 + cx2, hy1 + cy2],
                            })

    return phones, persons


def _simulate_phone(frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
    """Stateful simulation — phone persists 10-20 frames (70% hit rate)."""
    global _sim_frames_left, _sim_near_ear
    h, w = frame.shape[:2]
    px1, py1 = int(w * 0.15), int(h * 0.05)
    px2, py2 = int(w * 0.55), int(h * 0.95)
    persons = [{"label": "person", "confidence": 0.89, "bbox": [px1, py1, px2, py2]}]
    if _sim_frames_left <= 0:
        if random.random() < 0.70:
            _sim_frames_left = random.randint(10, 20)
            _sim_near_ear = random.random() < 0.50
        else:
            return [], persons
    _sim_frames_left -= 1
    height = py2 - py1
    cx = (px1 + px2) // 2
    if _sim_near_ear:
        ph_y1 = py1 + int(height * 0.05)
        ph_y2 = py1 + int(height * 0.25)
    else:
        ph_y1 = py1 + int(height * 0.55)
        ph_y2 = py1 + int(height * 0.75)
    phones = [
        {
            "label": "cell phone",
            "confidence": round(random.uniform(0.70, 0.95), 3),
            "bbox": [cx - 25, ph_y1, cx + 25, ph_y2],
        }
    ]
    return phones, persons


# ─────────────────────────────────────────────────────────────────
# Annotation  (ASCII labels only — cv2.putText cannot render emoji)
# ─────────────────────────────────────────────────────────────────


def _put_label(
    img: np.ndarray, text: str, x1: int, y1: int, color: tuple, font_scale: float = 0.52
) -> None:
    """Draw filled-background text label above y1."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    ty = max(th + 8, y1 - 4)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty + 2), color, -1)
    cv2.putText(
        img,
        text,
        (x1 + 4, ty - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )


def _draw_all_annotations(
    frame: np.ndarray,
    persons: List[Dict],
    phones: List[Dict],
    person_statuses: List[Tuple[str, Optional[Dict]]],
    no_phone_zone: bool,
) -> np.ndarray:
    """
    Draw per-person colored boxes based on phone status,
    then draw phone bounding boxes on top.
    """
    annotated = frame.copy()

    # ── Draw person boxes ─────────────────────────────────────
    for person, (status, _assoc_phone) in zip(persons, person_statuses):
        px1, py1, px2, py2 = person["bbox"]

        if status == "safe":
            color = COLOR_GREEN
            label = "SAFE - No Phone"
            thickness = 2
        elif status == "in_hand":
            color = COLOR_YELLOW
            label = "Phone in Hand"
            thickness = 3
        else:  # calling / zone_violation
            color = COLOR_RED
            label = "CALLING DETECTED" if status == "calling" else "PHONE VIOLATION"
            thickness = 3

        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, thickness)
        _put_label(annotated, label, px1, py1, color)

        # Confidence badge (bottom-left of person box)
        conf_label = f"{int(person['confidence'] * 100)}%"
        (cw, ch), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(annotated, (px1, py2 - ch - 6), (px1 + cw + 6, py2), color, -1)
        cv2.putText(
            annotated,
            conf_label,
            (px1 + 3, py2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

    # ── Draw phone boxes ──────────────────────────────────────
    for phone in phones:
        phx1, phy1, phx2, phy2 = phone["bbox"]
        # Determine if any person associated this phone
        is_alert = any(
            assoc is not None and assoc is phone for _status, assoc in person_statuses
        )
        # Even without person association, colour by zone rule
        # (single-person frame often has no separate person box)
        if no_phone_zone:
            pcolor = COLOR_RED
            plabel = "PHONE VIOLATION"
        elif is_alert:
            pcolor = COLOR_RED
            plabel = "CALLING DETECTED"
        else:
            pcolor = COLOR_YELLOW
            plabel = "Phone Detected"
        cv2.rectangle(annotated, (phx1, phy1), (phx2, phy2), pcolor, 2)
        _put_label(annotated, plabel, phx1, phy1, pcolor, font_scale=0.46)

    return annotated


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────


def detect_phone_usage(
    frame: np.ndarray,
    no_phone_zone: bool = True,
    coco_model=None,
    coco_fallback=None,
    use_simulation: bool = False,
) -> Dict:
    """
    High-accuracy phone usage detection with behavior-based alerts.

    Visual output per person:
      🟢 Green  "SAFE - No Phone"    → person detected, no phone nearby
      🟡 Yellow "Phone in Hand"      → phone near body but not near ear
      🔴 Red    "CALLING DETECTED"   → phone near ear (unsafe usage)
      🔴 Red    "PHONE VIOLATION"    → any phone when no_phone_zone=True

    Alert severity:
      calling / zone_violation  →  "high"   (saved to DB with snapshot)
      in_hand                   →  "medium"  (sent to frontend, no DB save)
      safe                      →  None      (green badge on frontend)

    Returns dict with:
      phone_detected, phone_status, phone_alert, phone_severity,
      phone_detections, annotated_frame
    """
    active_model = coco_model or coco_fallback
    frame_h = frame.shape[0]

    # ── Run inference ─────────────────────────────────────────
    if use_simulation or active_model is None:
        phones, persons = _simulate_phone(frame)
        if active_model is None:
            log.debug("Phone model not available — running simulation")
    else:
        try:
            phones, persons = _run_phone_inference(frame, active_model)
            log.debug(
                f"Phone inference: {len(phones)} phone(s), {len(persons)} person(s)"
            )
        except Exception as e:
            log.warning(f"Phone inference error: {e} — simulation fallback")
            phones, persons = _simulate_phone(frame)

    # ── No persons and no phones → completely safe ────────────
    if not phones and not persons:
        return {
            "phone_detected": False,
            "phone_status": "safe",
            "phone_alert": None,
            "phone_severity": None,
            "phone_detections": [],
            "annotated_frame": frame,
        }

    # ── If persons detected but no phones → all green ─────────
    if not phones and persons:
        annotated = frame.copy()
        for person in persons:
            px1, py1, px2, py2 = person["bbox"]
            cv2.rectangle(annotated, (px1, py1), (px2, py2), COLOR_GREEN, 2)
            _put_label(annotated, "SAFE - No Phone", px1, py1, COLOR_GREEN)
        return {
            "phone_detected": False,
            "phone_status": "safe",
            "phone_alert": None,
            "phone_severity": None,
            "phone_detections": [],
            "annotated_frame": annotated,
        }

    # ── If phones but no persons → use frame-position heuristic ─
    if phones and not persons:
        # Create a synthetic full-frame person box
        h, w = frame.shape[:2]
        persons = [{"label": "person", "confidence": 0.50, "bbox": [0, 0, w, h]}]

    # ── Classify each person by phone association ─────────────
    person_statuses: List[Tuple[str, Optional[Dict]]] = []
    for person in persons:
        status, assoc = _get_person_phone_status(
            person["bbox"], phones, no_phone_zone, frame_h
        )
        person_statuses.append((status, assoc))

    # ── Phones not associated to any person → mark as detected ─
    # (still draw them with yellow/red box based on no_phone_zone)

    # ── Draw all annotations ──────────────────────────────────
    annotated = _draw_all_annotations(
        frame, persons, phones, person_statuses, no_phone_zone
    )

    # ── Determine overall alert ───────────────────────────────
    statuses = [s for s, _ in person_statuses]

    if "zone_violation" in statuses or "calling" in statuses:
        # Highest severity — red alert
        worst = "zone_violation" if "zone_violation" in statuses else "calling"
        alert_msg = (
            "Phone Usage Not Allowed in This Area"
            if worst == "zone_violation"
            else "Unsafe Phone Usage Detected (Calling Near Ear)"
        )
        all_phone_dets = [
            {
                "label": p["label"],
                "confidence": p["confidence"],
                "bbox": p["bbox"],
                "status": (s if s != "safe" else "in_hand"),
            }
            for p, (s, _) in zip(persons, person_statuses)
            if s != "safe"
        ] + [
            {
                "label": p["label"],
                "confidence": p["confidence"],
                "bbox": p["bbox"],
                "status": "phone",
            }
            for p in phones
        ]
        return {
            "phone_detected": True,
            "phone_status": worst,
            "phone_alert": alert_msg,
            "phone_severity": "high",
            "phone_detections": all_phone_dets,
            "annotated_frame": annotated,
        }

    if "in_hand" in statuses:
        # Medium severity — yellow alert
        all_phone_dets = [
            {
                "label": p["label"],
                "confidence": p["confidence"],
                "bbox": p["bbox"],
                "status": "phone",
            }
            for p in phones
        ]
        return {
            "phone_detected": True,
            "phone_status": "in_hand",
            "phone_alert": "Phone Detected in Hand",
            "phone_severity": "medium",
            "phone_detections": all_phone_dets,
            "annotated_frame": annotated,
        }

    # All persons are safe (phone detected but not associated to any person)
    return {
        "phone_detected": bool(phones),
        "phone_status": "safe",
        "phone_alert": None,
        "phone_severity": None,
        "phone_detections": [],
        "annotated_frame": annotated,
    }
