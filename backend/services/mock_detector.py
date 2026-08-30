"""
Mock detector — generates plausible fake detections for MOCK_MODE.

In mock mode the real YOLO model is never loaded. This lets the entire
dashboard, rule engine, and alert pipeline be exercised without any
GPU, camera, or ML model files.

Usage (set in .env or environment):
    MOCK_MODE=true
"""
from __future__ import annotations

import math
import random
import time
from typing import Dict, List

from config import settings

# Stable random state per camera so persons appear to move smoothly
_camera_states: Dict[int, dict] = {}


def _get_state(camera_id: int) -> dict:
    if camera_id not in _camera_states:
        rng = random.Random(camera_id)
        persons = []
        for i in range(settings.MOCK_PERSON_COUNT):
            persons.append(
                {
                    "track_id": i + 1,
                    "x": rng.uniform(0.1, 0.85),
                    "y": rng.uniform(0.1, 0.75),
                    "vx": rng.uniform(-0.003, 0.003),
                    "vy": rng.uniform(-0.002, 0.002),
                    "idle_ticks": 0,
                }
            )
        _camera_states[camera_id] = {"persons": persons, "tick": 0}
    return _camera_states[camera_id]


# ── Class catalogue ──────────────────────────────────────────────────────────
_PPE_CLASSES = [
    "Hardhat", "NO-Hardhat", "Mask", "NO-Mask",
    "Safety Vest", "NO-Safety Vest",
]
_EXTRA_CLASSES = ["Bakery-Head-Cap", "NO-Bakery-Head-Cap"]


def generate_mock_detections(camera_id: int, frame_shape: tuple) -> dict:
    """
    Advance the mock state machine by one tick and return a detection dict
    in the same format as the real inference pool / enterprise_runtime.

    Returns
    -------
    dict with keys:
        tracks      list[dict]   — same structure as ByteTrack output
        detections  list[dict]   — same structure as YOLO raw detections
    """
    state = _get_state(camera_id)
    state["tick"] += 1
    tick = state["tick"]

    h, w = (frame_shape[0], frame_shape[1]) if frame_shape else (480, 640)

    tracks = []
    detections = []

    for p in state["persons"]:
        # Smooth sinusoidal walk so the person appears to move naturally
        t = tick * 0.05 + p["track_id"]
        p["x"] = max(0.05, min(0.9, p["x"] + p["vx"] + 0.002 * math.sin(t)))
        p["y"] = max(0.05, min(0.85, p["y"] + p["vy"] + 0.001 * math.cos(t * 1.3)))

        # Occasionally stop (simulate idling)
        if random.random() < 0.02:
            p["vx"] = random.uniform(-0.003, 0.003)
            p["vy"] = random.uniform(-0.002, 0.002)

        bw, bh = int(w * 0.08), int(h * 0.25)
        x1 = int(p["x"] * w)
        y1 = int(p["y"] * h)
        x2 = min(w - 1, x1 + bw)
        y2 = min(h - 1, y1 + bh)

        track = {
            "track_id": p["track_id"],
            "bbox": [x1, y1, x2, y2],
            "confidence": round(random.uniform(0.75, 0.97), 2),
            "class_name": "person",
            "class_id": 0,
        }
        tracks.append(track)

        # Person detection
        detections.append(
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": track["confidence"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "center_x": (x1 + x2) // 2,
                "center_y": (y1 + y2) // 2,
            }
        )

        # Occasional PPE violation overlay
        if random.random() < 0.25:
            cls = random.choice(_PPE_CLASSES + _EXTRA_CLASSES)
            detections.append(
                {
                    "class_id": len(_PPE_CLASSES),
                    "class_name": cls,
                    "confidence": round(random.uniform(0.60, 0.95), 2),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2 - bh // 2,
                    "center_x": (x1 + x2) // 2,
                    "center_y": y1 + bh // 4,
                }
            )

    return {"tracks": tracks, "detections": detections}


def is_mock_camera(camera_id: int) -> bool:
    """Return True when this camera should use mock instead of real RTSP."""
    if not settings.MOCK_MODE:
        return False
    ids_str = settings.MOCK_CAMERA_IDS.strip()
    if not ids_str:
        return True  # all cameras mocked when no list is given
    try:
        ids = {int(i.strip()) for i in ids_str.split(",") if i.strip()}
        return camera_id in ids
    except ValueError:
        return True
