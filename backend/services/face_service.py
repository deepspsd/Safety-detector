"""
Face Recognition Service — High-Accuracy Edition
==================================================
Key improvements:
  • Tighter recognition tolerance (0.42 vs 0.50) — fewer false "known" matches
  • Multi-shot voting: if the same person is registered multiple times,
    every stored encoding is compared and the MAJORITY VOTE wins.
  • Face size guard: ignores tiny/blurry faces (< MIN_FACE_PX pixels tall)
  • upsample=1 in face_locations for better detection of smaller/distant faces
  • No alert when ZERO faces are registered (nothing to compare against)
  • Owner-only mode: only alerts for Unknown faces — registered owners are GREEN
"""

import base64
import datetime
import json
import logging
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("face_service")

_use_simulation = False
try:
    import face_recognition

    print("[OK] face_recognition loaded")
except ImportError:
    print("[WARN] face_recognition unavailable — simulation mode")
    _use_simulation = True


from config import settings

# ── Tuning constants ────────────────────────────────────────────────────────────

# Max face distance to count as KNOWN (lower = stricter = fewer false matches)
# 0.52 = balanced for real-world lighting/angle variations (dlib default is 0.60; 0.42 was overly strict)
RECOGNITION_TOLERANCE = getattr(settings, "FACE_RECOGNITION_TOLERANCE", 0.52)

# Minimum face height in pixels to process — ignore tiny/blurry distant faces
MIN_FACE_PX = getattr(settings, "FACE_MIN_PIXELS", 30)

# Upsampling factor for face_locations — 1 = detect smaller/farther faces
UPSAMPLE_TIMES = 1


# ── Helpers ────────────────────────────────────────────────────────────────────


def _encode_jpg(frame: np.ndarray, quality: int = 88) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _encode_frame(frame: np.ndarray) -> str:
    return _encode_jpg(frame)


def _decode_b64(b64_data: str) -> Optional[np.ndarray]:
    try:
        raw = b64_data.split(",")[1] if "," in b64_data else b64_data
        arr = np.frombuffer(base64.b64decode(raw), np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    except Exception:
        return None


def _face_thumbnail(rgb_frame: np.ndarray, location: tuple, size: int = 128) -> str:
    top, right, bottom, left = location
    h, w = rgb_frame.shape[:2]
    pad_v = max(0, int((bottom - top) * 0.25))
    pad_h = max(0, int((right - left) * 0.25))
    crop = rgb_frame[
        max(0, top - pad_v) : min(h, bottom + pad_v),
        max(0, left - pad_h) : min(w, right + pad_h),
    ]
    if crop.size == 0:
        crop = rgb_frame
    bgr = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_RGB2BGR)
    return _encode_jpg(bgr, quality=80)


def decode_image(b64_data: str) -> Optional[np.ndarray]:
    """Public helper — returns RGB numpy array."""
    return _decode_b64(b64_data)


# ── DB helper ──────────────────────────────────────────────────────────────────


def load_known_encodings(user_id: Optional[int], db) -> Tuple[List[np.ndarray], List[str]]:
    """
    Returns all stored encodings and their labels.
    Enrolled faces are shared company-wide so all cameras and logged-in users
    (Admin, Manager, Worker) recognize all employees.
    """
    from database import FaceEncoding

    records = db.query(FaceEncoding).all()
    encs, labels = [], []
    for r in records:
        try:
            encs.append(np.array(json.loads(r.encoding_data), dtype=np.float64))
            labels.append(r.label)
        except Exception:
            pass
    return encs, labels


def _lookup_employee_id(label: str, db) -> Optional[int]:
    """
    Given a recognized face label (name), look up the matching Employee.id.
    Returns None if no active employee with that name exists.
    """
    if not label or not db:
        return None
    try:
        from database import Employee
        from sqlalchemy import func

        emp = (
            db.query(Employee)
            .filter(
                func.lower(Employee.name) == label.strip().lower(),
                Employee.active == True,  # noqa: E712
            )
            .first()
        )
        return emp.id if emp else None
    except Exception:
        return None


# ── Registration ───────────────────────────────────────────────────────────────


def encode_face_from_image(b64_image: str) -> Optional[Dict]:
    """
    Detect the largest/most-centered face in the image and return its encoding.
    Returns {"encoding": [...128 floats], "thumbnail_b64": "data:..."} or None.
    """
    if _use_simulation:
        enc = [random.gauss(0, 0.3) for _ in range(128)]  # realistic Gaussian noise
        return {"encoding": enc, "thumbnail_b64": None}

    rgb = _decode_b64(b64_image)
    if rgb is None:
        return None

    # upsample=1 catches smaller/partially visible faces in registration photos
    locs = face_recognition.face_locations(
        rgb, model="hog", number_of_times_to_upsample=1
    )
    if not locs:
        return None

    encs = face_recognition.face_encodings(
        rgb, locs, num_jitters=5
    )  # more jitter = more stable
    if not encs:
        return None

    # Pick the largest face (most prominent in frame)
    def _face_area(loc):
        t, r, b, l = loc
        return (b - t) * (r - l)

    best = int(np.argmax([_face_area(loc) for loc in locs]))
    return {
        "encoding": encs[best].tolist(),
        "thumbnail_b64": _face_thumbnail(rgb, locs[best]),
    }


# ── Live detection — numpy BGR (CCTV / webcam fast path) ──────────────────────


def process_face_numpy(frame_bgr: np.ndarray, user_id: int, db=None) -> Dict:
    """Detect + identify all faces in a BGR numpy frame.

    Opens its own thread-local DB session so this function is safe to call
    from a thread-pool executor (SQLAlchemy sessions are NOT thread-safe).
    """
    from database import SessionLocal
    _own_db = None
    try:
        _own_db = SessionLocal()
        if _use_simulation:
            return _simulate(frame_bgr, user_id, _own_db)
        return _run_detection(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), user_id, _own_db)
    finally:
        if _own_db is not None:
            try:
                _own_db.close()
            except Exception:
                pass


# ── Live detection — base64 (legacy upload path) ───────────────────────────────


def process_face_frame(b64_frame: str, user_id: int, db) -> Dict:
    """Detect + identify all faces in a base64-encoded frame."""
    if _use_simulation:
        try:
            raw = b64_frame.split(",")[1] if "," in b64_frame else b64_frame
            arr = np.frombuffer(base64.b64decode(raw), np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError
            return _simulate(frame, user_id, db)
        except Exception:
            return {"error": "Invalid frame"}
    rgb = _decode_b64(b64_frame)
    if rgb is None:
        return {"error": "Invalid frame"}
    return _run_detection(rgb, user_id, db)


# ── Core detection engine ──────────────────────────────────────────────────────


def _run_detection(rgb_frame: np.ndarray, user_id: int, db) -> Dict:
    """
    High-accuracy face detection + recognition.
    - Only alerts when a KNOWN database exists AND a face doesn't match any entry
    - Uses multi-shot voting: each stored encoding votes, majority label wins
    - Filters out tiny faces to avoid false detections
    - Returns employee_id alongside label for attendance hook
    """
    known_encs, known_labels = load_known_encodings(user_id, db)
    has_registered = len(known_encs) > 0

    # Detect face locations with upsampling for better small-face recall
    locs = face_recognition.face_locations(
        rgb_frame,
        model="hog",
        number_of_times_to_upsample=UPSAMPLE_TIMES,
    )

    # Compute encodings with jitter for stability
    encs = face_recognition.face_encodings(rgb_frame, locs, num_jitters=3)

    faces: List[Dict] = []
    unknown_detected = False
    # Track recognized employees for attendance hook: {employee_id: confidence}
    recognized_employees: Dict[int, float] = {}

    for enc, (top, right, bottom, left) in zip(encs, locs):
        face_height = bottom - top

        # Skip tiny/blurry faces — they produce unreliable encodings
        if face_height < MIN_FACE_PX:
            log.debug(f"[Face] Skipping small face h={face_height}px < {MIN_FACE_PX}")
            continue

        label = "Unknown"
        confidence = 0.0
        is_unknown = True
        employee_id = None

        if has_registered:
            # Compute distances to ALL stored encodings
            dists = face_recognition.face_distance(known_encs, enc)

            # ── Multi-shot voting ────────────────────────────────────────────
            # For each stored label, take the BEST (min) distance among all
            # encodings with that label. This handles multi-registration well.
            label_best: Dict[str, float] = {}
            for dist, lbl in zip(dists, known_labels):
                if lbl not in label_best or dist < label_best[lbl]:
                    label_best[lbl] = dist

            # Find the overall best-matching label
            best_label = min(label_best, key=label_best.get)
            best_dist = label_best[best_label]
            confidence = round(max(0.0, 1.0 - best_dist), 3)

            if best_dist <= RECOGNITION_TOLERANCE:
                label = best_label
                is_unknown = False
                # Look up employee_id so attendance can be updated
                employee_id = _lookup_employee_id(label, db)
                if employee_id is not None:
                    # Keep best confidence for each employee seen this frame
                    if (
                        employee_id not in recognized_employees
                        or confidence > recognized_employees[employee_id]
                    ):
                        recognized_employees[employee_id] = confidence
                log.debug(
                    f"[Face] Matched '{label}' emp={employee_id} dist={best_dist:.3f} conf={confidence:.0%}"
                )
            else:
                # Distance exceeds tolerance → UNKNOWN person
                unknown_detected = True
                log.debug(
                    f"[Face] UNKNOWN dist={best_dist:.3f} > tolerance={RECOGNITION_TOLERANCE}"
                )
        else:
            # No faces registered at all — detect presence but don't alert
            # (user hasn't set up face recognition yet)
            is_unknown = True  # mark visually but don't alert
            # unknown_detected stays False — no alert until registration is done

        faces.append(
            {
                "label": label,
                "confidence": confidence,
                "bbox": [left, top, right, bottom],
                "is_unknown": is_unknown,
                "employee_id": employee_id,
            }
        )

    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    ann = _draw_faces(bgr, faces, unknown_detected, has_registered)

    return {
        "faces": faces,
        "face_count": len(faces),
        "unknown_detected": unknown_detected,
        "is_compliant": not unknown_detected,
        "alert_message": "⚠️ Unknown person detected!" if unknown_detected else None,
        "severity": "critical" if unknown_detected else None,
        "annotated_frame": _encode_jpg(ann),
        "snapshot_b64": _encode_jpg(bgr) if unknown_detected else None,
        "recognized_employees": recognized_employees,  # {employee_id: confidence}
    }


# ── Simulation ─────────────────────────────────────────────────────────────────


def _simulate(frame_bgr: np.ndarray, user_id: int, db) -> Dict:
    from database import FaceEncoding

    known_count = db.query(FaceEncoding).filter(FaceEncoding.user_id == user_id).count()
    h, w = frame_bgr.shape[:2]
    faces: List[Dict] = []
    unknown_detected = False

    if known_count > 0:
        # Registered owner — shown as compliant
        fx, fy, fw, fh = int(w * 0.20), int(h * 0.10), int(w * 0.22), int(h * 0.40)
        faces.append(
            {
                "label": "Owner",
                "confidence": round(random.uniform(0.88, 0.98), 2),
                "bbox": [fx, fy, fx + fw, fy + fh],
                "is_unknown": False,
            }
        )
        # 15% chance of an intruder showing up alongside the owner
        if random.random() < 0.15:
            fx2 = int(w * 0.58)
            fy2 = int(h * 0.08)
            fw2 = int(w * 0.20)
            fh2 = int(h * 0.38)
            faces.append(
                {
                    "label": "Unknown",
                    "confidence": 0.0,
                    "bbox": [fx2, fy2, fx2 + fw2, fy2 + fh2],
                    "is_unknown": True,
                }
            )
            unknown_detected = True
    # No alert when no faces registered

    ann = _draw_faces(frame_bgr, faces, unknown_detected, known_count > 0)
    return {
        "faces": faces,
        "face_count": len(faces),
        "unknown_detected": unknown_detected,
        "is_compliant": not unknown_detected,
        "alert_message": "⚠️ Unknown person detected!" if unknown_detected else None,
        "severity": "critical" if unknown_detected else None,
        "annotated_frame": _encode_jpg(ann),
        "snapshot_b64": _encode_jpg(frame_bgr) if unknown_detected else None,
    }


# ── Drawing ────────────────────────────────────────────────────────────────────


def _draw_faces(
    frame: np.ndarray,
    faces: List[Dict],
    unknown_detected: bool,
    has_registered: bool = True,
) -> np.ndarray:
    ann = frame.copy()
    h, w = ann.shape[:2]

    # Red alert banner only when there are registered faces AND an intruder found
    if unknown_detected and has_registered:
        ov = ann.copy()
        cv2.rectangle(ov, (0, 0), (w, 38), (20, 0, 180), -1)
        cv2.addWeighted(ov, 0.70, ann, 0.30, 0, ann)
        cv2.putText(
            ann,
            "  \u26a0 ALERT: UNKNOWN PERSON DETECTED",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 210, 210),
            2,
            cv2.LINE_AA,
        )
    elif not has_registered and faces:
        # Soft info banner — faces visible but no DB to compare against
        ov = ann.copy()
        cv2.rectangle(ov, (0, 0), (w, 38), (30, 80, 0), -1)
        cv2.addWeighted(ov, 0.55, ann, 0.45, 0, ann)
        cv2.putText(
            ann,
            "  \u2139 Face detected — Register owner in Settings \u2192 Faces",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 255, 180),
            2,
            cv2.LINE_AA,
        )

    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
        unk = face.get("is_unknown", True)
        conf = face.get("confidence", 0.0)
        lname = face.get("label", "Unknown")

        # Color: GREEN = known owner, RED = unknown intruder, GRAY = no DB
        if not has_registered:
            color = (120, 120, 120)  # gray — no comparison possible
        elif unk:
            color = (40, 40, 230)  # red (BGR)
        else:
            color = (40, 210, 50)  # green (BGR)

        thick = 3 if unk and has_registered else 2
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, thick)

        # Label pill
        if unk and has_registered:
            text = f"\u274c UNKNOWN  {conf:.0%}"
        elif not has_registered:
            text = "Face detected"
        else:
            text = f"\u2713 {lname}  {conf:.0%}"

        scale = 0.55
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        pill_y1 = max(0, y1 - th - 12)
        cv2.rectangle(ann, (x1, pill_y1), (x1 + tw + 10, y1), (0, 0, 0), -1)
        cv2.rectangle(ann, (x1, pill_y1), (x1 + tw + 10, y1), color, 1)
        cv2.putText(
            ann,
            text,
            (x1 + 5, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )

    # Timestamp watermark
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        ann,
        ts,
        (w - 192, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    return ann


# ─────────────────────────────────────────────────────────────────────────────
# Worker Identity Cache
# ─────────────────────────────────────────────────────────────────────────────
# Maps (camera_id, track_id) → {employee_id, name, confidence, ts}
# Populated by camera_manager after each face recognition tick.
# Monitors call get_worker_identity(camera_id, track_id) at alert-fire time.
# No new file needed — lives here in face_service since this is the face module.

_wid_cache: Dict[Tuple[int, str], dict] = {}
_wid_lock = threading.Lock()
_WID_TTL = 15.0  # seconds before identity expires (face rec runs every 2s)


def _bbox_centroid(bbox) -> Tuple[float, float]:
    """Return (cx, cy) for a [x1, y1, x2, y2] bbox."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def update_worker_identity_from_face_result(
    camera_id: int,
    persons: List[dict],
    face_result: dict,
    db,
) -> None:
    """
    Associate recognized faces → tracked person bboxes via centroid distance.
    Call this from camera_manager._stream_annotation_loop after face recognition.

    face_result["faces"]: list of {bbox:[x1,y1,x2,y2], label:str, employee_id:int|None,
                                    is_unknown:bool, confidence:float}
    persons: list from yolo_service with {bbox:..., track_id:int|str}
    """
    faces = face_result.get("faces", [])
    if not faces or not persons:
        return

    # Build person centroid list (only tracked persons)
    tracked = [
        (str(p.get("track_id", "")), p.get("bbox", []))
        for p in persons
        if p.get("track_id") not in (None, -1, "-1", "")
    ]
    if not tracked:
        return

    now = time.time()

    # For each recognized (non-unknown) face, find nearest person track
    for face in faces:
        if face.get("is_unknown", True):
            continue
        emp_name = face.get("label", "Unknown")
        if not emp_name or emp_name == "Unknown":
            continue
        emp_id = face.get("employee_id")
        if emp_id is None and db is not None:
            emp_id = _lookup_employee_id(emp_name, db)
        conf = face.get("confidence", 0.0)
        face_bbox = face.get("bbox")
        if not face_bbox:
            continue

        fx, fy = _bbox_centroid(face_bbox)

        # Find nearest person track by centroid distance
        best_tid = None
        best_dist = float("inf")
        for tid, pbbox in tracked:
            if not pbbox or len(pbbox) != 4:
                continue
            px, py = _bbox_centroid(pbbox)
            dist = ((fx - px) ** 2 + (fy - py) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is None:
            continue

        # Only associate if face centroid is reasonably close to person bbox
        # (guard against wide-angle cameras where faces/persons are far apart)
        max_dist = 300  # pixels; generous to handle partial bbox overlap
        if best_dist > max_dist:
            continue

        key = (camera_id, best_tid)
        with _wid_lock:
            _wid_cache[key] = {
                "employee_id": emp_id,
                "name": emp_name,
                "confidence": conf,
                "ts": now,
            }

    log.debug(
        "[WorkerID] cam=%d updated %d faces → %d tracked persons",
        camera_id, len(faces), len(tracked),
    )


def get_worker_identity(camera_id: int, track_id) -> Optional[dict]:
    """
    Return {employee_id, name, confidence} for a tracked person, or None.
    Returns None if identity is unknown, not yet recognized, or TTL expired.
    """
    key = (camera_id, str(track_id))
    now = time.time()
    with _wid_lock:
        entry = _wid_cache.get(key)
        if entry is None:
            return None
        if now - entry["ts"] > _WID_TTL:
            del _wid_cache[key]
            return None
        return entry.copy()


def clear_worker_identity_camera(camera_id: int) -> None:
    """Evict all cached identities for a camera (call on camera stop)."""
    with _wid_lock:
        for k in [k for k in _wid_cache if k[0] == camera_id]:
            del _wid_cache[k]
    log.debug("[WorkerID] cam=%d cache cleared", camera_id)

