"""
Unit tests for the Person Cropper, Tracker, Zone Engine, Classifier Adapter, and Rule State Machines.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from services.person_cropper import CropMode, crop_person
from services.classifier_adapter import (
    ClassificationResult,
    ClassifierCache,
    MockClassifier,
    TemporalSmoother,
)
from services.tracking_layer import ByteTrackAdapter, TrackObservation
from services.zone_service import compute_zone_for_track, check_line_crossing
from services.rule_engine_v2 import (
    IdleRuleStateMachine,
    AbsenceRuleStateMachine,
    CameraStandingChecker,
)
from services.mock_detector import generate_mock_detections


class TestPersonCropper:
    def test_crop_modes_and_clamping(self):
        # Create a dummy image 480x640
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = [100, 100, 300, 400]

        # Full person
        crop_full = crop_person(frame, bbox, crop_mode=CropMode.FULL_PERSON)
        assert crop_full is not None
        assert crop_full.shape == (224, 224, 3)

        # Upper body
        crop_upper = crop_person(frame, bbox, crop_mode=CropMode.UPPER_BODY, upper_body_ratio=0.65)
        assert crop_upper is not None
        assert crop_upper.shape == (224, 224, 3)

        # Head
        crop_head = crop_person(frame, bbox, crop_mode=CropMode.HEAD)
        assert crop_head is not None
        assert crop_head.shape == (224, 224, 3)

    def test_invalid_bbox_returns_none(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert crop_person(frame, [100, 100, 100, 100]) is None  # zero width/height
        assert crop_person(frame, [300, 300, 100, 100]) is None  # inverted


class TestClassifierAdapterAndSmoothing:
    def test_mock_classifier_predictions(self):
        clf = MockClassifier("test_clf", ["A", "B"])
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        res = clf.predict(img)
        assert isinstance(res, ClassificationResult)
        assert res.predicted_class in ["A", "B"]
        assert res.confidence >= 0.0

    def test_classifier_cache_rate_limiting(self):
        cache = ClassifierCache(interval_ms=500)
        assert cache.should_run("track_1", "uniform") is True
        cache.record("track_1", "uniform")
        assert cache.should_run("track_1", "uniform") is False

    def test_temporal_smoother_majority_voting(self):
        smoother = TemporalSmoother(window=5)
        # Push 3 UNIFORM and 1 NO_UNIFORM
        for _ in range(3):
            smoother.push("track_1", "uniform", ClassificationResult("UNIFORM", 0.9, {}))
        smoother.push("track_1", "uniform", ClassificationResult("NO_UNIFORM", 0.6, {}))

        smoothed = smoother.vote("track_1", "uniform")
        assert smoothed is not None
        assert smoothed.predicted_class == "UNIFORM"


class TestTrackingLayer:
    def test_track_observation_history(self):
        tracker = ByteTrackAdapter()
        obs = TrackObservation(
            track_id="17",
            bbox=[50, 50, 150, 200],
            confidence=0.95,
            velocity=(0.0, 0.0),
            direction="stationary",
            movement_state="stationary",
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        obs.update_history(100.0, 125.0)
        assert len(obs.history) == 1
        assert obs.history[0]["cx"] == 100.0


class TestZoneEngine:
    def test_point_in_polygon_zone(self):
        zones = {
            "packing": [[0, 0], [200, 0], [200, 200], [0, 200]],
            "shop": [[300, 300], [500, 300], [500, 500], [300, 500]],
        }
        assert compute_zone_for_track(1, "1", 100, 100, zones) == "packing"
        assert compute_zone_for_track(1, "2", 400, 400, zones) == "shop"
        assert compute_zone_for_track(1, "3", 250, 250, zones) is None

    def test_line_crossing(self):
        # Line at y=200 from x=100 to x=300
        # side = (300-100)*(py-200) = 200*(py-200)
        # py=250 -> positive side; py=150 -> negative side
        # Moving positive -> negative = INWARD
        lines = {"entry": [[100, 200], [300, 200]]}
        res_in = check_line_crossing(None, "1", (150, 250), (150, 150), lines)
        assert res_in is not None
        assert res_in["direction"] == "INWARD"

        res_out = check_line_crossing(None, "1", (150, 150), (150, 250), lines)
        assert res_out is not None
        assert res_out["direction"] == "OUTWARD"


class TestRuleStateMachines:
    def test_idle_rule_state_machine(self):
        sm = IdleRuleStateMachine(idle_threshold_sec=2, cooldown_sec=2)
        sm.update(1, "1", "moving")
        assert sm.get_state(1, "1") == "ACTIVE"
        sm.update(1, "1", "stationary")
        assert sm.get_state(1, "1") == "IDLE_TIMER_RUNNING"

    def test_absence_rule_state_machine(self):
        sm = AbsenceRuleStateMachine(absence_threshold_sec=2, cooldown_sec=2)
        sm.update(1, persons_in_shop=1)
        assert sm.get_state(1) == "PERSON_PRESENT"
        sm.update(1, persons_in_shop=0)
        assert sm.get_state(1) == "ABSENT_TIMER_RUNNING"

    def test_mock_detector_generator(self):
        dets = generate_mock_detections(1, (480, 640, 3))
        assert "tracks" in dets
        assert "detections" in dets
        assert len(dets["tracks"]) > 0
