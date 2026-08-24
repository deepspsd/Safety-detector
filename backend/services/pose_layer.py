"""Pose extension point. It deliberately emits observations, never policy decisions."""

from __future__ import annotations

from typing import Dict, List


class PoseAdapter:
    def analyse(self, frame, person_boxes: List[List[int]]) -> List[Dict]:
        # A registered pose model can replace this adapter. Keeping this stable
        # allows hand/head/body signals to enter Context without rule rewrites.
        return [
            {"bbox": bbox, "keypoints": [], "available": False} for bbox in person_boxes
        ]


pose_adapter = PoseAdapter()
