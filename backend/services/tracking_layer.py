"""Tracking layer: identity, trajectory and zone-transition-ready state only."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from math import hypot
from typing import Dict, List, Optional, Tuple

from services.detection_layer import Detection


@dataclass
class TrackObservation:
    track_id: str
    bbox: List[int]
    confidence: float
    velocity: Tuple[float, float]
    direction: str
    movement_state: str
    first_seen_at: datetime
    last_seen_at: datetime

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["first_seen_at"] = self.first_seen_at.isoformat()
        data["last_seen_at"] = self.last_seen_at.isoformat()
        return data


class ByteTrackAdapter:
    """Uses the existing per-camera ByteTrack registry, with no policy logic."""

    def __init__(self) -> None:
        self._state: Dict[Tuple[int, str], Dict] = {}
        self._lock = threading.Lock()

    def update(
        self, camera_id: int, detections: List[Detection]
    ) -> List[TrackObservation]:
        from services import yolo_service

        persons = [
            {"bbox": item.bbox, "confidence": item.confidence}
            for item in detections
            if item.label.lower() == "person"
        ]
        tracked = yolo_service._apply_tracking(camera_id, persons) if persons else []
        now = datetime.utcnow()
        observations: List[TrackObservation] = []
        active_keys = set()
        with self._lock:
            for person in tracked:
                raw_id = int(person.get("track_id", -1))
                if raw_id < 0:
                    continue
                track_id = str(raw_id)
                key = (camera_id, track_id)
                active_keys.add(key)
                center = self._center(person["bbox"])
                prior = self._state.get(key)
                velocity = (0.0, 0.0)
                direction = "stationary"
                movement_state = "stationary"
                first_seen = now
                if prior:
                    old_center = prior["center"]
                    elapsed = max((now - prior["seen"]).total_seconds(), 0.001)
                    velocity = (
                        (center[0] - old_center[0]) / elapsed,
                        (center[1] - old_center[1]) / elapsed,
                    )
                    speed = hypot(*velocity)
                    movement_state = "moving" if speed >= 8 else "stationary"
                    direction = (
                        self._direction(velocity)
                        if movement_state == "moving"
                        else "stationary"
                    )
                    first_seen = prior["first_seen"]
                self._state[key] = {
                    "center": center,
                    "seen": now,
                    "first_seen": first_seen,
                }
                observations.append(
                    TrackObservation(
                        track_id,
                        person["bbox"],
                        person["confidence"],
                        velocity,
                        direction,
                        movement_state,
                        first_seen,
                        now,
                    )
                )
            # Track lifetime is bounded by the detector buffer.  Consumers see
            # a lost event from the context engine on the next absent frame.
            for key in [
                k
                for k, state in self._state.items()
                if k[0] == camera_id
                and k not in active_keys
                and (now - state["seen"]).total_seconds() > 3
            ]:
                self._state.pop(key, None)
        return observations

    def reset_camera(self, camera_id: int) -> None:
        """Forget transient tracking state for one camera only."""
        with self._lock:
            for key in [key for key in self._state if key[0] == camera_id]:
                self._state.pop(key, None)

    @staticmethod
    def _center(bbox: List[int]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    @staticmethod
    def _direction(velocity: Tuple[float, float]) -> str:
        x, y = velocity
        if abs(x) >= abs(y):
            return "right" if x > 0 else "left"
        return "down" if y > 0 else "up"


tracker = ByteTrackAdapter()
