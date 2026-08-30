"""
Debug overlay renderer.

Draws a rich development overlay on a frame showing:
- Bounding box per track
- Track ID
- Current zone
- Classifier predictions (Uniform / Head Cap / etc.)
- Idle timer
- Movement state

Only active when DEBUG_OVERLAY=true in config (independent of DEBUG_CROPS).

Usage in camera loop:
    from services.debug_overlay import draw_debug_overlay
    annotated = draw_debug_overlay(frame, camera_id, persons, zones)
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import settings

# ── Colours (BGR) ────────────────────────────────────────────────────────────
_COLOUR_BOX = (0, 200, 0)        # green person box
_COLOUR_IDLE = (0, 80, 220)      # orange-red for idle state
_COLOUR_TEXT_BG = (20, 20, 20)   # near-black label background
_COLOUR_ZONE = (255, 200, 0)     # cyan zone label
_COLOUR_CLF_YES = (0, 220, 0)    # green for positive classifier
_COLOUR_CLF_NO = (0, 60, 220)    # red-ish for violation

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.42
_FONT_THICK = 1
_LINE_H = 16   # pixels per text line


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _label_colour(clf_name: str, class_name: str) -> tuple:
    """Green for safe / positive, red-ish for violation."""
    violations = {"NO_UNIFORM", "NO_HEAD_CAP", "BANGLE", "IDLE", "NO_MASK", "NO-Hardhat", "NO-Mask"}
    return _COLOUR_CLF_NO if class_name.upper() in violations else _COLOUR_CLF_YES


def _draw_label_block(
    frame: np.ndarray,
    x1: int, y1: int,
    lines: List[str],
    colours: Optional[List[tuple]] = None,
) -> None:
    """Draw a semi-transparent label block above the bounding box."""
    if not lines:
        return
    block_w = max(cv2.getTextSize(l, _FONT, _FONT_SCALE, _FONT_THICK)[0][0] for l in lines) + 8
    block_h = len(lines) * _LINE_H + 4
    bx1 = max(0, x1)
    by1 = max(0, y1 - block_h)
    bx2 = bx1 + block_w
    by2 = bx1 + block_h  # note: relative

    # Overlay rect
    roi = frame[by1:by1 + block_h, bx1:bx2]
    if roi.size:
        overlay = roi.copy()
        overlay[:] = _COLOUR_TEXT_BG
        cv2.addWeighted(overlay, 0.65, roi, 0.35, 0, roi)

    for i, line in enumerate(lines):
        col = colours[i] if (colours and i < len(colours)) else (220, 220, 220)
        cv2.putText(
            frame, line,
            (bx1 + 3, by1 + (i + 1) * _LINE_H - 3),
            _FONT, _FONT_SCALE, col, _FONT_THICK, cv2.LINE_AA,
        )


def draw_debug_overlay(
    frame: np.ndarray,
    camera_id: int,
    persons: List[Dict[str, Any]],
    zones: Optional[Dict] = None,
) -> np.ndarray:
    """
    Annotate *frame* in-place (or on a copy) with tracking debug info.
    Returns the annotated frame.
    """
    if frame is None:
        return frame

    out = frame.copy()

    from services.tracking_layer import tracker as _tracker

    now = time.time()

    for person in persons:
        tid = str(person.get("track_id", ""))
        bbox = person.get("bbox", [])
        if not tid or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), _COLOUR_BOX, 2)

        obs = _tracker.get_track(camera_id, tid)

        # ── Build label lines ────────────────────────────────────────────────
        lines: List[str] = []
        colours: List[tuple] = []

        lines.append(f"#{tid}")
        colours.append((255, 255, 255))

        if obs:
            # Zone
            zone_name = obs.current_zone or "—"
            lines.append(f"Zone: {zone_name}")
            colours.append(_COLOUR_ZONE)

            # Idle timer
            if obs.last_moved_at:
                idle_s = now - obs.last_moved_at
                lines.append(f"Idle: {_fmt_duration(idle_s)}")
                colours.append(_COLOUR_IDLE if idle_s > 60 else (160, 160, 160))

            # Classifier predictions
            preds = obs.last_attribute_predictions or {}
            for clf_name, result in preds.items():
                if isinstance(result, dict):
                    cls = result.get("predicted_class", "?")
                    conf = result.get("confidence", 0.0)
                    label = f"{clf_name.replace('_', ' ').title()}: {cls} {conf:.2f}"
                    lines.append(label)
                    colours.append(_label_colour(clf_name, cls))

        _draw_label_block(out, x1, y1, lines, colours)

    # ── Zone polygons (if provided) ──────────────────────────────────────────
    if zones:
        for zone_name, polygon in zones.items():
            if not polygon or len(polygon) < 3:
                continue
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts], True, _COLOUR_ZONE, 1)
            cx = int(np.mean([p[0] for p in polygon]))
            cy = int(np.mean([p[1] for p in polygon]))
            cv2.putText(
                out, zone_name.upper(),
                (cx, cy), _FONT, 0.38, _COLOUR_ZONE, 1, cv2.LINE_AA,
            )

    # ── Debug watermark ──────────────────────────────────────────────────────
    cv2.putText(
        out, "DEBUG MODE",
        (6, out.shape[0] - 6), _FONT, 0.35, (0, 0, 255), 1, cv2.LINE_AA,
    )

    return out
