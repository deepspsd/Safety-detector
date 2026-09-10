"""
Unit tests for surveillance upgrades:
1. Fall vs. Sitting classification (FallKinematicsAnalyzer)
2. Fight / Physical Altercation detection (FightAggressionDetector)
3. Indian Cash Handling Integrity & zone spatial restriction (CashEventTracker)
"""

import os
import sys
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.pose_layer import fall_analyzer, FallKinematicsAnalyzer
from services.action_recognition import aggression_detector, FightAggressionDetector
from services.cash_monitor import check_cash_zone, _is_cash_label, CashEventTracker, CashState


class TestFallVsSittingKinematics(unittest.TestCase):
    def setUp(self):
        self.analyzer = FallKinematicsAnalyzer(history_len=15)
        self.cam_id = 1
        self.track_id = 42

    def test_sitting_posture_suppresses_fall(self):
        """A worker sitting down with an upright torso should NOT trigger fall alert."""
        t0 = time.time()
        # Simulate 5 frames of sitting: height=180, width=150, AR=1.2, torso upright=75 deg
        for i in range(5):
            t = t0 + i * 0.1
            bbox = [100, 200, 250, 380]  # w=150, h=180, AR=1.2
            keypoints = {
                "left_shoulder": (150, 230, 0.9),
                "right_shoulder": (200, 230, 0.9),
                "left_hip": (150, 320, 0.9),
                "right_hip": (200, 320, 0.9),
            }
            res = self.analyzer.update_track(self.cam_id, self.track_id, bbox, keypoints, now=t)

        self.assertTrue(res["is_sitting"])
        self.assertFalse(res["is_fall"])
        self.assertEqual(res["state"], "sitting")

        # Test filter_fall_false_positives
        raw_falls = [{"label": "Worker Fall", "confidence": 0.91, "bbox": [100, 200, 250, 380]}]
        persons = [{"track_id": self.track_id, "bbox": [100, 200, 250, 380]}]
        verified, suppressed = self.analyzer.filter_fall_false_positives(
            self.cam_id, raw_falls, persons, (720, 1280)
        )
        self.assertEqual(suppressed, 1)
        self.assertEqual(len(verified), 0, "Sitting person must not trigger fall detection")

    def test_genuine_fall_collapse_triggers_fall(self):
        """A worker collapsing to horizontal floor pose (AR < 0.85, angle < 35 deg, motionless) triggers fall."""
        t0 = time.time()
        # Initial standing pose
        self.analyzer.update_track(
            self.cam_id, 101, [100, 100, 180, 320],  # h=220, w=80, AR=2.75
            {"left_shoulder": (140, 140, 0.9), "right_shoulder": (160, 140, 0.9),
             "left_hip": (140, 220, 0.9), "right_hip": (160, 220, 0.9)},
            now=t0
        )
        # Fast collapse to horizontal floor pose
        res = None
        for i in range(1, 8):
            t = t0 + i * 0.25
            # Fallen person on floor: h=70, w=240, AR=0.29, torso angle < 20 deg
            bbox = [100, 400, 340, 470]
            keypoints = {
                "left_shoulder": (130, 430, 0.9),
                "right_shoulder": (130, 440, 0.9),
                "left_hip": (280, 435, 0.9),
                "right_hip": (280, 445, 0.9),
            }
            res = self.analyzer.update_track(self.cam_id, 101, bbox, keypoints, now=t)

        self.assertIsNotNone(res)
        self.assertTrue(res["is_fall"])
        self.assertEqual(res["state"], "fallen")


class TestFightAggressionDetection(unittest.TestCase):
    def setUp(self):
        self.detector = FightAggressionDetector(history_len=15)
        self.cam_id = 2

    def test_peaceful_distance_no_fight(self):
        """Two individuals standing far apart do not trigger fight detection."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        persons = [
            {"track_id": 1, "bbox": [50, 100, 130, 350]},
            {"track_id": 2, "bbox": [400, 100, 480, 350]},
        ]
        dets = self.detector.detect_aggression(dummy_frame, persons, camera_id=self.cam_id, now=100.0)
        self.assertEqual(len(dets), 0)

    def test_peaceful_close_standing_no_fight(self):
        """Two individuals close together without rapid velocity/acceleration do not trigger fight."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Feed 4 consecutive frames with stationary close positions
        for i in range(4):
            t = 100.0 + i * 0.1
            persons = [
                {"track_id": 1, "bbox": [200, 100, 270, 350]},
                {"track_id": 2, "bbox": [260, 100, 330, 350]},
            ]
            dets = self.detector.detect_aggression(dummy_frame, persons, camera_id=self.cam_id, now=t)
            self.assertEqual(len(dets), 0)

    def test_violent_physical_altercation_detected(self):
        """Two individuals with rapid erratic movement and high relative acceleration trigger physical altercation."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = 200.0

        # Frame 1: close contact
        self.detector.detect_aggression(
            dummy_frame,
            [{"track_id": 10, "bbox": [200, 100, 280, 350]}, {"track_id": 20, "bbox": [270, 100, 350, 350]}],
            camera_id=self.cam_id, now=t0
        )
        # Frame 2: rapid high-speed strike & lunge
        self.detector.detect_aggression(
            dummy_frame,
            [{"track_id": 10, "bbox": [240, 100, 320, 350]}, {"track_id": 20, "bbox": [230, 110, 310, 360]}],
            camera_id=self.cam_id, now=t0 + 0.08
        )
        # Frame 3: violent recoil and struggle
        dets = self.detector.detect_aggression(
            dummy_frame,
            [{"track_id": 10, "bbox": [190, 95, 270, 345]}, {"track_id": 20, "bbox": [280, 120, 360, 370]}],
            camera_id=self.cam_id, now=t0 + 0.16
        )

        self.assertTrue(len(dets) >= 1)
        self.assertEqual(dets[0]["label"], "Physical Altercation")
        self.assertEqual(dets[0]["det_type"], "violation")

    def test_peaceful_persons_standing_together_not_altercation(self):
        """Two individuals standing side-by-side or overlapping in camera perspective must NOT trigger altercation."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = 300.0

        # Simulate 10 consecutive frames of two coworkers standing close together (overlapping bboxes)
        for f in range(10):
            dets = self.detector.detect_aggression(
                dummy_frame,
                [
                    {"track_id": 101, "bbox": [200, 100, 300, 380]},
                    {"track_id": 102, "bbox": [240, 110, 340, 390]}  # heavy 2D overlap (IoU > 0.35)
                ],
                camera_id=self.cam_id,
                now=t0 + (f * 0.1)
            )
            self.assertEqual(len(dets), 0, f"False positive altercation detected on frame {f} while standing calmly!")


class TestIndianCashIntegrity(unittest.TestCase):
    def test_indian_rupee_labels(self):
        """All Indian Rupee denomination aliases must be recognized as Cash."""
        inr_cases = [
            "10 inr", "20 inr", "50 inr", "100 inr", "200 inr", "500 inr", "2000 inr",
            "10 INR", "500 INR", "2000 INR",
            "₹10", "₹50", "₹100", "₹500", "₹2000",
            "Rs 500", "rs.100", "rupee 200", "Cash", "banknote"
        ]
        for c in inr_cases:
            self.assertTrue(_is_cash_label(c), f"Expected '{c}' to be recognized as cash label")

        # Non-cash labels should be rejected
        for non_c in ["person", "hairnet", "Hardhat", "Safety Vest", "bottle"]:
            self.assertFalse(_is_cash_label(non_c), f"Expected '{non_c}' to NOT be recognized as cash")

    def test_cash_monitoring_bypassed_outside_cash_zone(self):
        """Cash monitor must strictly exit without alerts when on non-cash floor and no cashbox polygon."""
        res = check_cash_zone(
            db=None,
            camera_id=99,
            detections=[{"label": "500 INR", "confidence": 0.95, "bbox": [100, 100, 150, 130]}],
            persons=[],
            cashbox_polygon=None,
            floor="dough_mixing",  # production floor, NOT shop
        )
        self.assertFalse(res["cash_detected"])
        self.assertIsNone(res["theft_alert"])

    def test_cash_deposited_in_cashbox(self):
        """Cash entering cashbox polygon reaches DEPOSITED state."""
        tracker = CashEventTracker(camera_id=5)
        cb_poly = [[100, 100], [200, 100], [200, 200], [100, 200]]

        # Frame 1: cash in cashbox
        tracker.update(
            db=None,
            cash_detections=[{"label": "500 INR", "confidence": 0.92, "bbox": [130, 130, 170, 160], "track_id": "c1"}],
            person_detections=[],
            cashbox_polygon=cb_poly,
            floor="shop",
        )
        track = tracker._tracks.get("c1")
        self.assertIsNotNone(track)
        # Advance time by 1.2 seconds to confirm deposit
        track.in_cashbox_since = time.monotonic() - 1.5
        tracker.update(
            db=None,
            cash_detections=[{"label": "500 INR", "confidence": 0.92, "bbox": [130, 130, 170, 160], "track_id": "c1"}],
            person_detections=[],
            cashbox_polygon=cb_poly,
            floor="shop",
        )
        self.assertEqual(track.state, CashState.DEPOSITED)


if __name__ == "__main__":
    unittest.main()
