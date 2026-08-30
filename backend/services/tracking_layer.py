"""Tracking layer: identity, trajectory and zone-transition-ready state only.

No alert, workflow or business-policy decision belongs here.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import hypot
from typing import Any, Dict, List, Optional, Tuple

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
    # Human-readable object class (e.g. "person")
    object_class: str = "person"
    # Epoch timestamp of the last significant movement (used for idle timing)
    last_moved_at: Optional[float] = None
    # Zone context — populated by the zone engine after detection
    current_zone: Optional[str] = None
    previous_zone: Optional[str] = None
    # Rolling position history: list of {bbox, ts} dicts (last N samples)
    history: List[Dict[str, Any]] = field(default_factory=list)
    # Latest classifier predictions keyed by classifier name
    last_attribute_predictions: Dict[str, Any] = field(default_factory=dict)

    _HISTORY_MAX = 60  # keep last 60 position samples

    # Convenience aliases for the /tracks API
    @property
    def first_seen(self) -> str:
        return self.first_seen_at.isoformat()

    @property
    def last_seen(self) -> str:
        return self.last_seen_at.isoformat()

    def update_history(self, cx: float, cy: float) -> None:
        import time as _time
        entry = {"bbox": list(self.bbox), "ts": _time.time(), "cx": cx, "cy": cy}
        self.history.append(entry)
        if len(self.history) > self._HISTORY_MAX:
            self.history = self.history[-self._HISTORY_MAX:]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["first_seen_at"] = self.first_seen_at.isoformat()
        data["last_seen_at"] = self.last_seen_at.isoformat()
        return data


class ByteTrackAdapter:
    """Uses the existing per-camera ByteTrack registry, with no policy logic."""

    def __init__(self) -> None:
        self._state: Dict[Tuple[int, str], Dict] = {}
        self._observations: Dict[Tuple[int, str], TrackObservation] = {}
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
                # Retrieve or create persistent observation for zone/attr state
                obs = self._observations.get(key)
                if obs is None:
                    obs = TrackObservation(
                        track_id=track_id,
                        bbox=person["bbox"],
                        confidence=person["confidence"],
                        velocity=velocity,
                        direction=direction,
                        movement_state=movement_state,
                        first_seen_at=first_seen,
                        last_seen_at=now,
                    )
                    self._observations[key] = obs
                else:
                    obs.bbox = person["bbox"]
                    obs.confidence = person["confidence"]
                    obs.velocity = velocity
                    obs.direction = direction
                    obs.movement_state = movement_state
                    obs.last_seen_at = now
                # Update last_moved_at when person is actually moving
                if movement_state == "moving":
                    import time as _time
                    obs.last_moved_at = _time.time()
                elif obs.last_moved_at is None:
                    import time as _time
                    obs.last_moved_at = _time.time()
                obs.update_history(center[0], center[1])
                observations.append(obs)

            # Expire lost tracks after 3 s
            for key in [
                k
                for k, state in self._state.items()
                if k[0] == camera_id
                and k not in active_keys
                and (now - state["seen"]).total_seconds() > 3
            ]:
                self._state.pop(key, None)
                self._observations.pop(key, None)
        return observations

    def get_track(self, camera_id: int, track_id: str) -> Optional[TrackObservation]:
        """Return the current observation for a specific track, or None."""
        with self._lock:
            return self._observations.get((camera_id, track_id))

    def all_tracks(self, camera_id: int) -> List[TrackObservation]:
        """Return all current observations for a camera."""
        with self._lock:
            return [
                obs
                for (cam_id, _), obs in self._observations.items()
                if cam_id == camera_id
            ]

    def all_track_keys(self) -> List[Tuple[int, str]]:
        """Return (camera_id, track_id) tuples for every active observation."""
        with self._lock:
            return list(self._observations.keys())

    def reset_camera(self, camera_id: int) -> None:
        """Forget transient tracking state for one camera only."""
        with self._lock:
            for key in [key for key in self._state if key[0] == camera_id]:
                self._state.pop(key, None)
                self._observations.pop(key, None)

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
