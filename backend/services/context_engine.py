"""Context engine: turns independent vision outputs into semantic track context."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from services import zone_service
from services.detection_layer import Detection
from services.platform_events import emit
from services.tracking_layer import TrackObservation


@dataclass
class TrackContext:
    camera_id: int
    track_id: str
    occurred_at: datetime
    zone_id: Optional[int]
    zone_name: Optional[str]
    workflow_stage: Optional[str]
    calibration_version: int
    calibration_status: str
    position: List[int]
    velocity: Tuple[float, float]
    direction: str
    movement_state: str
    objects: List[Dict]
    ppe: Dict[str, List[str]]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["occurred_at"] = self.occurred_at.isoformat()
        return data


class ContextEngine:
    def __init__(self) -> None:
        self._previous_zones: Dict[Tuple[int, str], Optional[int]] = {}
        self._lock = threading.RLock()

    def build(
        self,
        camera_id: int,
        tracks: List[TrackObservation],
        detections: List[Detection],
        db,
    ) -> List[TrackContext]:
        from database import Camera, ZoneConfig

        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            return []
        # A moved camera invalidates the relationship between pixels and zones.
        # Context remains available, but no zone semantics are exposed until a
        # calibration version has been restored or recaptured.
        zones = (
            []
            if camera.calibration_status == "required"
            else (
                db.query(ZoneConfig)
                .filter(
                    ZoneConfig.camera_id == camera_id, ZoneConfig.is_active.is_(True)
                )
                .order_by(ZoneConfig.priority.desc())
                .all()
            )
        )
        contexts: List[TrackContext] = []
        for track in tracks:
            zone = self._resolve_zone(track.bbox, zones)
            assigned = [
                item for item in detections if self._belongs(track.bbox, item.bbox)
            ]
            ppe = {
                "present": [
                    x.label
                    for x in assigned
                    if not x.label.startswith("NO-")
                    and x.label in {"Bakery-Head-Cap", "Hardhat", "Mask", "Safety Vest"}
                ],
                "missing": [x.label for x in assigned if x.label.startswith("NO-")],
                "violations": [
                    x.label for x in assigned if x.label in {"Bangles", "Exposed-Item"}
                ],
            }
            context = TrackContext(
                camera_id=camera_id,
                track_id=track.track_id,
                occurred_at=track.last_seen_at,
                zone_id=zone.id if zone else None,
                zone_name=zone.zone_name if zone else None,
                workflow_stage=zone.workflow_stage if zone else None,
                calibration_version=camera.calibration_version or 0,
                calibration_status=camera.calibration_status or "unknown",
                position=track.bbox,
                velocity=track.velocity,
                direction=track.direction,
                movement_state=track.movement_state,
                objects=[x.to_dict() for x in assigned],
                ppe=ppe,
            )
            self._emit_context_events(context)
            self._persist_context(context, db)
            contexts.append(context)
        return contexts

    @staticmethod
    def _resolve_zone(bbox: List[int], zones) -> Optional[object]:
        for zone in zones:
            try:
                if zone_service.bbox_in_zone(bbox, json.loads(zone.polygon_json)):
                    return zone
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        return None

    @staticmethod
    def _belongs(person: List[int], item: List[int]) -> bool:
        cx, cy = (item[0] + item[2]) / 2, (item[1] + item[3]) / 2
        return person[0] <= cx <= person[2] and person[1] <= cy <= person[3]

    def _emit_context_events(self, context: TrackContext) -> None:
        key = (context.camera_id, context.track_id)
        with self._lock:
            old_zone = self._previous_zones.get(key)
            self._previous_zones[key] = context.zone_id
        payload = context.to_dict()
        base = dict(
            camera_id=context.camera_id,
            track_id=context.track_id,
            zone_id=context.zone_id,
            calibration_version=context.calibration_version,
            payload=payload,
            source="context-engine",
        )
        emit("PERSON_DETECTED", **base)
        if old_zone != context.zone_id:
            if old_zone is not None:
                emit(
                    "ZONE_EXITED",
                    camera_id=context.camera_id,
                    track_id=context.track_id,
                    zone_id=old_zone,
                    calibration_version=context.calibration_version,
                    payload=context.to_dict(),
                    source="context-engine",
                )
            if context.zone_id is not None:
                emit("ZONE_ENTERED", **base)
        signal_base = {key: value for key, value in base.items() if key != "payload"}
        for label in context.ppe["missing"]:
            emit(
                "PPE_MISSING",
                confidence=None,
                **signal_base,
                payload={**context.to_dict(), "label": label},
            )
        for label in context.ppe["violations"]:
            emit(
                "OBJECT_POLICY_SIGNAL",
                confidence=None,
                **signal_base,
                payload={**context.to_dict(), "label": label},
            )

    @staticmethod
    def _persist_context(context: TrackContext, db) -> None:
        try:
            from database import ContextSnapshot

            db.add(
                ContextSnapshot(
                    camera_id=context.camera_id,
                    track_id=context.track_id,
                    occurred_at=context.occurred_at,
                    zone_id=context.zone_id,
                    workflow_stage=context.workflow_stage,
                    context_json=json.dumps(context.to_dict()),
                )
            )
            db.commit()
        except Exception:
            db.rollback()


context_engine = ContextEngine()
