"""
Face Recognition Service
Handles face encoding storage, known/unknown face identification.
Uses the face_recognition library (dlib-based).
Falls back to simulation if dlib unavailable.
"""
import json
import os
import numpy as np
import base64
import cv2
import random
from typing import List, Dict, Optional, Tuple
from config import settings

_use_simulation = False

try:
    import face_recognition
    _use_simulation = False
    print("[OK] face_recognition library loaded")
except ImportError:
    print("[WARN] face_recognition not available, using simulation mode")
    _use_simulation = True


def decode_image(b64_data: str) -> Optional[np.ndarray]:
    """Decode base64 to RGB numpy array for face_recognition."""
    try:
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def encode_face_from_image(b64_image: str) -> Optional[List[float]]:
    """Extract face encoding from a reference image."""
    if _use_simulation:
        return [random.uniform(-1, 1) for _ in range(128)]
    rgb_image = decode_image(b64_image)
    if rgb_image is None:
        return None
    encodings = face_recognition.face_encodings(rgb_image)
    if not encodings:
        return None
    return encodings[0].tolist()


def load_known_encodings(user_id: int, db) -> Tuple[List[np.ndarray], List[str]]:
    """Load all face encodings for a user from DB."""
    from database import FaceEncoding
    records = db.query(FaceEncoding).filter(FaceEncoding.user_id == user_id).all()
    encodings = []
    labels = []
    for rec in records:
        enc = json.loads(rec.encoding_data)
        encodings.append(np.array(enc))
        labels.append(rec.label)
    return encodings, labels


def process_face_frame(b64_frame: str, user_id: int, db) -> Dict:
    """Detect faces in frame, compare with known encodings, return results."""
    if _use_simulation:
        return _simulate_face_detection(b64_frame, user_id, db)

    rgb_frame = decode_image(b64_frame)
    if rgb_frame is None:
        return {"error": "Invalid frame"}

    known_encodings, known_labels = load_known_encodings(user_id, db)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    faces = []
    unknown_detected = False

    for enc, loc in zip(face_encodings, face_locations):
        label = "Unknown"
        confidence = 0.0
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, enc)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            confidence = round(max(0, 1 - best_dist), 2)
            if best_dist < 0.5:
                label = known_labels[best_idx]
            else:
                unknown_detected = True
        else:
            unknown_detected = True

        top, right, bottom, left = loc
        faces.append({
            "label": label,
            "confidence": confidence,
            "bbox": [left, top, right, bottom],
            "is_unknown": label == "Unknown"
        })

    # Draw on frame
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    annotated = _draw_faces(bgr_frame, faces, unknown_detected)
    annotated_b64 = _encode_frame(annotated)
    snapshot_b64 = _encode_frame(bgr_frame) if unknown_detected else None

    return {
        "faces": faces,
        "unknown_detected": unknown_detected,
        "is_compliant": not unknown_detected,
        "alert_message": "⚠️ Unknown person detected!" if unknown_detected else None,
        "severity": "critical" if unknown_detected else None,
        "annotated_frame": annotated_b64,
        "snapshot_b64": snapshot_b64
    }


def _simulate_face_detection(b64_frame: str, user_id: int, db) -> Dict:
    """Simulate face detection for demo."""
    from database import FaceEncoding
    known_count = db.query(FaceEncoding).filter(FaceEncoding.user_id == user_id).count()
    unknown_detected = (known_count == 0) or (random.random() < 0.2)
    label = "Unknown" if unknown_detected else "Owner"

    # Decode and annotate frame
    try:
        raw = b64_frame.split(",")[1] if "," in b64_frame else b64_frame
        nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return {"error": "Invalid frame"}

    h, w = frame.shape[:2]
    fx, fy = int(w * 0.35), int(h * 0.15)
    fw, fh = int(w * 0.3), int(h * 0.45)
    conf = round(random.uniform(0.78, 0.96), 2)
    faces = [{"label": label, "confidence": conf, "bbox": [fx, fy, fx + fw, fy + fh], "is_unknown": unknown_detected}]

    annotated = _draw_faces(frame, faces, unknown_detected)
    annotated_b64 = _encode_frame(annotated)
    snapshot_b64 = _encode_frame(frame) if unknown_detected else None

    return {
        "faces": faces,
        "unknown_detected": unknown_detected,
        "is_compliant": not unknown_detected,
        "alert_message": "⚠️ Unknown person detected!" if unknown_detected else None,
        "severity": "critical" if unknown_detected else None,
        "annotated_frame": annotated_b64,
        "snapshot_b64": snapshot_b64
    }


def _draw_faces(frame: np.ndarray, faces: List[Dict], unknown_detected: bool) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    if unknown_detected:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (180, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
        cv2.putText(annotated, "⚠ ALERT: UNKNOWN PERSON DETECTED",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        color = (50, 50, 255) if face["is_unknown"] else (50, 220, 50)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{face['label']} {face['confidence']:.0%}"
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    ts = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated, ts, (w - 200, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return annotated


def _encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
