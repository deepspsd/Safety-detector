"""Person Cropping Engine
========================
Crops a region from a camera frame based on a person bounding box, preparing
it for downstream image classifiers (uniform, head cap, bangle, etc.).

Usage
─────
    from services.person_cropper import crop_person, CropMode

    crop = crop_person(
        frame=frame,
        bbox=[x1, y1, x2, y2],
        crop_mode=CropMode.UPPER_BODY,
        padding=0.10,
        upper_body_ratio=0.65,
        target_size=(224, 224),
    )
    if crop is not None:
        result = classifier.predict(crop)

Crop modes
──────────
FULL_PERSON   — the complete bounding box with padding
UPPER_BODY    — top `upper_body_ratio` fraction of the box (for uniform/head cap)
HEAD          — top fraction of the box used for head-cap checks
CUSTOM        — caller supplies explicit ratios via `y_start_ratio` / `y_end_ratio`

All modes:
  1. Start with the raw bbox.
  2. Apply padding (fractional, relative to box size) on all sides.
  3. Clamp to image boundaries.
  4. Validate minimum size.
  5. Resize to target_size.

Debug saving
────────────
Set DEBUG_CROPS=true in .env to save every crop to:
  debug/crops/<camera_id>/<YYYY-MM-DD>/<track_id>/<classifier_name>/

Filename format:
  <timestamp>_<predicted_class>_<confidence>.jpg
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import settings

log = logging.getLogger("person_cropper")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_CROP_PX = 16   # crops smaller than this in either dimension are rejected
DEFAULT_PADDING = 0.05
DEFAULT_UPPER_BODY_RATIO = 0.65
DEFAULT_HEAD_RATIO = 0.30
DEFAULT_TARGET_SIZE = (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# Crop mode enum
# ─────────────────────────────────────────────────────────────────────────────


class CropMode(str, Enum):
    FULL_PERSON = "full_person"
    UPPER_BODY = "upper_body"
    HEAD = "head"
    CUSTOM = "custom"


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────


def crop_person(
    frame: np.ndarray,
    bbox: List[int],
    crop_mode: str = CropMode.FULL_PERSON,
    padding: float = DEFAULT_PADDING,
    upper_body_ratio: float = DEFAULT_UPPER_BODY_RATIO,
    head_ratio: float = DEFAULT_HEAD_RATIO,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    y_start_ratio: float = 0.0,  # CUSTOM mode only
    y_end_ratio: float = 1.0,    # CUSTOM mode only
) -> Optional[np.ndarray]:
    """Crop a person region from *frame*.

    Args:
        frame:            BGR numpy array from OpenCV.
        bbox:             [x1, y1, x2, y2] — person detection box.
        crop_mode:        One of CropMode values.
        padding:          Fractional padding applied to each side of the box.
        upper_body_ratio: Fraction of box height kept for UPPER_BODY mode.
        head_ratio:       Fraction of box height kept for HEAD mode.
        target_size:      (width, height) to resize the crop to.
        y_start_ratio:    CUSTOM mode — vertical start as fraction of box height.
        y_end_ratio:      CUSTOM mode — vertical end as fraction of box height.

    Returns:
        Resized BGR crop, or None if the crop is invalid/too small.
    """
    if frame is None or frame.size == 0:
        return None

    img_h, img_w = frame.shape[:2]
    x1_raw, y1_raw, x2_raw, y2_raw = bbox

    # Validate raw bbox
    if x2_raw <= x1_raw or y2_raw <= y1_raw:
        log.debug("crop_person: invalid bbox %s", bbox)
        return None

    box_w = x2_raw - x1_raw
    box_h = y2_raw - y1_raw

    # ── Apply vertical crop mode ──────────────────────────────────────────────
    mode = CropMode(crop_mode) if isinstance(crop_mode, str) else crop_mode

    if mode == CropMode.UPPER_BODY:
        y2_raw = y1_raw + int(box_h * upper_body_ratio)
    elif mode == CropMode.HEAD:
        y2_raw = y1_raw + int(box_h * head_ratio)
    elif mode == CropMode.CUSTOM:
        y1_raw = y1_raw + int(box_h * y_start_ratio)
        y2_raw = y1_raw + int(box_h * (y_end_ratio - y_start_ratio))
    # FULL_PERSON — no change

    # ── Apply padding ─────────────────────────────────────────────────────────
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)
    x1 = x1_raw - pad_x
    y1 = y1_raw - pad_y
    x2 = x2_raw + pad_x
    y2 = y2_raw + pad_y

    # ── Clamp to image boundaries ─────────────────────────────────────────────
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # ── Validate minimum size ─────────────────────────────────────────────────
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < MIN_CROP_PX or crop_h < MIN_CROP_PX:
        log.debug("crop_person: crop too small (%dx%d)", crop_w, crop_h)
        return None

    # ── Slice and resize ──────────────────────────────────────────────────────
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    try:
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    except cv2.error as exc:
        log.warning("crop_person: resize failed: %s", exc)
        return None

    return resized


# ─────────────────────────────────────────────────────────────────────────────
# Debug crop saving
# ─────────────────────────────────────────────────────────────────────────────


def save_debug_crop(
    crop: np.ndarray,
    camera_id: int,
    track_id: str,
    classifier_name: str,
    predicted_class: str,
    confidence: float,
) -> Optional[str]:
    """Save *crop* to the debug directory if DEBUG_CROPS is enabled.

    Returns:
        Path of saved file, or None if debug saving is disabled or fails.
    """
    if not settings.DEBUG_CROPS:
        return None

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    ts = datetime.utcnow().strftime("%H%M%S_%f")
    safe_cls = predicted_class.replace("/", "-").replace(" ", "_")
    filename = f"{ts}_{safe_cls}_{confidence:.2f}.jpg"

    dir_path = os.path.join(
        settings.DEBUG_CROPS_DIR,
        str(camera_id),
        date_str,
        str(track_id),
        classifier_name,
    )
    try:
        os.makedirs(dir_path, exist_ok=True)
        full_path = os.path.join(dir_path, filename)
        cv2.imwrite(full_path, crop)
        return full_path
    except Exception as exc:
        log.warning("save_debug_crop: failed: %s", exc)
        return None
