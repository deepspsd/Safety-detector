"""Configurable workflow projection; it consumes context and emits events only."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from services.platform_events import emit


class WorkflowEngine:
    def __init__(self) -> None:
        self._stage_by_track: Dict[Tuple[int, str], Optional[str]] = {}

    def apply(self, context, workflow_profile_id: Optional[int] = None) -> None:
        # The stage is supplied by the calibrated zone/profile.  Valid paths,
        # roles and transitions are data in WorkflowProfile.definition_json;
        # this component deliberately does not encode bakery process names.
        key = (context.camera_id, context.track_id)
        prior = self._stage_by_track.get(key)
        current = context.workflow_stage
        if current and current != prior:
            self._stage_by_track[key] = current
            emit(
                "WORKFLOW_STAGE_ENTERED",
                camera_id=context.camera_id,
                zone_id=context.zone_id,
                track_id=context.track_id,
                calibration_version=context.calibration_version,
                payload={
                    "workflow_profile_id": workflow_profile_id,
                    "stage": current,
                    "previous_stage": prior,
                    "context": context.to_dict(),
                },
                source="workflow-engine",
            )


workflow_engine = WorkflowEngine()
