"""
test_headcap.py — Unit and state-machine tests for crop-based HeadCapMonitor
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.headcap_monitor import (
    HeadCapMonitor,
    PRED_HEAD_CAP,
    PRED_NO_HEAD_CAP,
    PRED_UNCERTAIN,
    STATE_COMPLIANT,
    STATE_PENDING,
    STATE_ALERT,
    _derive_head_crop_box,
    _map_confidence_to_prediction,
)


class TestHeadCapMonitor(unittest.TestCase):
    def test_derive_head_crop_box(self):
        person_bbox = [100, 100, 200, 400]  # w=100, h=300
        frame_shape = (720, 1280, 3)
        crop = _derive_head_crop_box(person_bbox, frame_shape)

        # Top 30% of height = 90px -> y1=100, y2=190
        # 10% x padding = 10px -> x1=90, x2=210
        self.assertEqual(crop[0], 90)
        self.assertEqual(crop[1], 100)
        self.assertEqual(crop[2], 210)
        self.assertEqual(crop[3], 190)

    def test_confidence_mapping(self):
        self.assertEqual(_map_confidence_to_prediction(0.85), PRED_HEAD_CAP)
        self.assertEqual(_map_confidence_to_prediction(0.25), PRED_HEAD_CAP)
        self.assertEqual(_map_confidence_to_prediction(0.22), PRED_UNCERTAIN)
        self.assertEqual(_map_confidence_to_prediction(0.20), PRED_NO_HEAD_CAP)
        self.assertEqual(_map_confidence_to_prediction(0.05), PRED_NO_HEAD_CAP)

    def test_headcap_monitor_lifecycle(self):
        monitor = HeadCapMonitor()
        cam_id = 99
        track_id = "test_track_1"

        persons = [{"track_id": track_id, "bbox": [50, 50, 150, 350]}]
        raw_dets = [{"label": "Bakery-Head-Cap", "confidence": 0.85, "bbox": [60, 50, 140, 100]}]

        # 1. First tick with cap present
        statuses = monitor.update_camera(
            camera_id=cam_id,
            persons=persons,
            raw_dets=raw_dets,
            db=None,
            frame=None,
        )
        self.assertEqual(len(statuses), 1)
        st = statuses[0]
        self.assertEqual(st.track_id, track_id)

        # 2. Check diagnostic extraction
        diags = monitor.get_diagnostics(cam_id)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].track_id, track_id)

        # 3. Clean reset
        monitor.reset_camera(cam_id)
        self.assertEqual(len(monitor.get_diagnostics(cam_id)), 0)


if __name__ == "__main__":
    unittest.main()
