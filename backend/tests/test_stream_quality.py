"""
test_stream_quality.py — Tests for CameraReader frame validation and stream health
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routers.cctv import (
    CameraReader,
    STREAM_ONLINE,
    STREAM_DEGRADED,
    STREAM_OFFLINE,
    STREAM_RECOVERING,
)


class TestStreamQuality(unittest.TestCase):
    def setUp(self):
        self.reader = CameraReader("0")

    def test_green_frame_detection(self):
        # 1. Normal colored frame
        normal_frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
        self.assertFalse(self.reader._is_green_frame(normal_frame))

        # 2. Pure synthetic green frame (H.264 decoder desync corruption)
        green_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        green_frame[:, :, 1] = 240  # High green channel
        green_frame[:, :, 0] = 5    # Low blue
        green_frame[:, :, 2] = 5    # Low red
        self.assertTrue(self.reader._is_green_frame(green_frame))

    def test_frozen_frame_detection(self):
        frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)

        # Single frame is not frozen initially
        self.assertFalse(self.reader._is_frozen_frame(frame))

        # Repeated identical frame increments frozen counter
        for _ in range(35):
            is_frozen = self.reader._is_frozen_frame(frame)

        self.assertTrue(is_frozen)

    def test_corrupted_frame_detection(self):
        # 1. Black/zero frame
        black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertTrue(self.reader._is_corrupted_frame(black_frame))

        # 2. Normal varied frame
        normal_frame = np.random.randint(20, 240, (100, 100, 3), dtype=np.uint8)
        self.assertFalse(self.reader._is_corrupted_frame(normal_frame))

    def test_camera_reader_status_metrics(self):
        self.assertEqual(self.reader.stream_status, STREAM_OFFLINE)

        # Metric reporting check
        metrics = self.reader.metrics()
        self.assertIn("stream_status", metrics)
        self.assertIn("bad_frame_count", metrics)
        self.assertIn("decoder_error_count", metrics)
        self.assertEqual(metrics["stream_status"], STREAM_OFFLINE)


if __name__ == "__main__":
    unittest.main()
