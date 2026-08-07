"""Versioned polygon calibration and ORB-based camera drift assessment."""
from __future__ import annotations

from datetime import datetime
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


CALIBRATION_DIR = os.path.join("uploads", "calibrations")
os.makedirs(CALIBRATION_DIR, exist_ok=True)


def validate_polygon(polygon: List[List[int]], width: int | None = None, height: int | None = None) -> None:
    if len(polygon) < 3:
        raise ValueError("Polygon requires at least three vertices")
    points = np.array(polygon, dtype=np.float32)
    if len(np.unique(points, axis=0)) < 3:
        raise ValueError("Polygon requires three distinct vertices")
    if abs(cv2.contourArea(points.astype(np.int32))) < 4:
        raise ValueError("Polygon area is too small")
    if width is not None and height is not None and any(x < 0 or y < 0 or x > width or y > height for x, y in polygon):
        raise ValueError("Polygon vertex lies outside the camera frame")


def save_version(db, camera_id: int, user_id: int | None, frame: np.ndarray | None, note: str = "") -> int:
    from database import Camera, CalibrationVersion, ZoneConfig
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise ValueError("Camera not found")
    zones = db.query(ZoneConfig).filter(ZoneConfig.camera_id == camera_id).all()
    version = int(camera.calibration_version or 0) + 1
    snapshot_path = None
    if frame is not None:
        snapshot_path = os.path.join(CALIBRATION_DIR, f"camera_{camera_id}_v{version}.jpg")
        cv2.imwrite(snapshot_path, frame)
    zones_json = json.dumps([{"zone_name": z.zone_name, "polygon_json": z.polygon_json, "zone_type": z.zone_type,
                              "display_name": z.display_name, "color": z.color, "priority": z.priority,
                              "workflow_stage": z.workflow_stage} for z in zones])
    db.add(CalibrationVersion(camera_id=camera_id, version=version, snapshot_path=snapshot_path,
                              reference_frame_path=snapshot_path or camera.reference_frame_path, zones_json=zones_json,
                              change_note=note, created_by_user_id=user_id))
    camera.calibration_version = version
    camera.calibration_status = "calibrated"
    camera.last_calibrated_at = datetime.utcnow()
    if snapshot_path:
        camera.reference_frame_path = snapshot_path
    for zone in zones:
        zone.calibration_version = version
    db.commit()
    return version


def assess_drift(camera, frame: np.ndarray, threshold: float = 18.0) -> Tuple[bool, float, str]:
    """Return (drifted, score, method). Scores are median feature displacement."""
    path = camera.reference_frame_path
    if not path or not os.path.exists(path):
        return False, 0.0, "reference_missing"
    reference = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if reference is None:
        return False, 0.0, "reference_unreadable"
    current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current = cv2.resize(current, (reference.shape[1], reference.shape[0]))
    extractor = cv2.ORB_create(nfeatures=800)
    k1, d1 = extractor.detectAndCompute(reference, None)
    k2, d2 = extractor.detectAndCompute(current, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return False, 0.0, "insufficient_features"
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    matches = sorted(matches, key=lambda item: item.distance)[:80]
    if len(matches) < 8:
        return False, 0.0, "insufficient_matches"
    displacement = [float(np.linalg.norm(np.array(k1[m.queryIdx].pt) - np.array(k2[m.trainIdx].pt))) for m in matches]
    score = float(np.median(displacement))
    return score > threshold, round(score, 2), "orb_median_displacement"
