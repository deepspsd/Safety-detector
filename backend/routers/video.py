"""
Video upload & processing router — v3.0  (Smart Timeline Edition)

Changes vs v2.0:
  • Generates an annotated output video (MP4) with bounding boxes drawn
  • Stores violation_timestamps list: [{ts, items, severity, persons, thumbnail_b64}]
  • Exposes GET /video/result/{job_id} to stream the annotated video
  • Keeps all existing alert saving & ppe_summary logic
"""

import os
import uuid
from typing import Optional, Tuple

import cv2
from config import settings
from database import User, get_db
from fastapi import (APIRouter, BackgroundTasks, Depends, File, Form, HTTPException,
                     UploadFile)
from fastapi.responses import FileResponse
from routers.auth import get_current_user
from services import face_service, yolo_service
from services.alert_service import save_alert
from sqlalchemy.orm import Session

router = APIRouter(prefix="/video", tags=["video"])

# In-memory job store (use Redis in production)
_job_status: dict = {}


def _get_annotated_path(job_id: str) -> str:
    return os.path.join(settings.UPLOAD_DIR, f"annotated_{job_id}.mp4")


def _create_video_writer(annotated_path: str, fps: float, width: int, height: int) -> Tuple[Optional[cv2.VideoWriter], bool]:
    """Return the best available working VideoWriter."""
    for cc in ("mp4v", "avc1", "XVID", "MJPG"):
        try:
            fcc = cv2.VideoWriter_fourcc(*cc)
            writer = cv2.VideoWriter(annotated_path, fcc, fps, (width, height))
            if writer is not None and writer.isOpened():
                return writer, True
        except Exception:
            continue
    return None, False


def process_video_job(job_id: str, video_path: str, role: str, user_id: int, zone_type: str = "default"):
    """Background task: process video frame-by-frame, write annotated video."""
    _job_status[job_id] = {
        "status": "processing",
        "progress": 0,
        "alerts": [],
        "frames_processed": 0,
        "total_violations": 0,
        "annotated_video_url": None,
        "violation_timestamps": [],
    }

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    frame_skip = max(1, int(fps // 5))  # analyse ~5 frames/sec

    annotated_path = _get_annotated_path(job_id)
    out_writer, writer_ok = _create_video_writer(annotated_path, fps, width, height)

    alerts_found: list = []
    violation_timestamps: list = []  # [{ts, items, severity, persons, thumbnail_b64}]
    frame_num = 0
    processed = 0
    total_violations = 0
    ppe_summary: dict = {}

    from database import SessionLocal

    db = SessionLocal()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            progress = int((frame_num / max(total_frames, 1)) * 100)
            _job_status[job_id]["progress"] = progress

            # ── Determine if this frame will be analysed ──────────────
            analyse_this = frame_num % frame_skip == 0

            if analyse_this:
                processed += 1
                _job_status[job_id]["frames_processed"] = processed

                # ── Run detection pipeline ────────────────────────────
                if role == "Home":
                    b64 = yolo_service.encode_frame(frame)
                    result = face_service.process_face_frame(b64, user_id, db)
                else:
                    result = yolo_service.process_frame_numpy(
                        frame, role, frame_index=frame_num, zone_type=zone_type
                    )

                ts_sec = round(frame_num / fps, 1)

                # ── Extract the annotated frame from the result ───────
                # process_frame_numpy → returns annotated_frame as base64
                # We decode it back to numpy for the video writer.
                annotated_frame = frame  # default: original frame
                ann_b64 = result.get("annotated_frame")
                if ann_b64:
                    try:
                        import base64

                        import numpy as np

                        _, b64data = (
                            ann_b64.split(",", 1) if "," in ann_b64 else ("", ann_b64)
                        )
                        img_bytes = base64.b64decode(b64data)
                        arr = np.frombuffer(img_bytes, np.uint8)
                        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if decoded is not None and decoded.shape[:2] == (height, width):
                            annotated_frame = decoded
                    except Exception:
                        pass

                # ── Write frame to output video ───────────────────────
                if writer_ok:
                    out_writer.write(annotated_frame)

                # ── Record violations ─────────────────────────────────
                if not result.get("is_compliant") and result.get("alert_message"):
                    n_violations = result.get("violations_count", 0)
                    total_violations += n_violations

                    for item in result.get("missing_items", []):
                        ppe_summary[item] = ppe_summary.get(item, 0) + 1

                    # Violation timestamp entry (used by frontend timeline)
                    vt_entry = {
                        "ts": ts_sec,
                        "items": result.get("missing_items", []),
                        "severity": result.get("severity", "medium"),
                        "persons": result.get("persons_count", 0),
                        "violations": n_violations,
                    }
                    # Attach thumbnail for first 20 timestamps
                    if len(violation_timestamps) < 20 and result.get("snapshot_b64"):
                        vt_entry["thumbnail_b64"] = result["snapshot_b64"]
                    violation_timestamps.append(vt_entry)

                    alert_info = {
                        "timestamp_sec": ts_sec,
                        "message": result["alert_message"],
                        "severity": result["severity"],
                        "missing_items": result.get("missing_items", []),
                        "violations_count": n_violations,
                        "persons_count": result.get("persons_count", 0),
                    }
                    # Full annotated frame thumbnail for first 10 alerts
                    if len(alerts_found) < 10 and result.get("annotated_frame"):
                        alert_info["thumbnail_b64"] = result["annotated_frame"]

                    alerts_found.append(alert_info)

                    if len(alerts_found) <= 20:
                        save_alert(
                            db=db,
                            user_id=user_id,
                            message=result["alert_message"],
                            role=role,
                            severity=result["severity"],
                            detected_issue=", ".join(result.get("missing_items", [])),
                            confidence=None,
                            snapshot_b64=result.get("snapshot_b64"),
                        )
            else:
                # Non-analysed frame: write the original frame to keep timing
                if writer_ok:
                    out_writer.write(frame)

        # ── Finalise ──────────────────────────────────────────────────
        out_writer.release()

        # Build video URL (served at /uploads/annotated_{job_id}.mp4)
        video_url = f"/uploads/annotated_{job_id}.mp4" if writer_ok else None

        _job_status[job_id] = {
            "status": "complete",
            "progress": 100,
            "frames_processed": processed,
            "total_alerts": len(alerts_found),
            "total_violations": total_violations,
            "alerts": alerts_found[:50],
            "violation_timestamps": violation_timestamps,
            "ppe_summary": dict(
                sorted(ppe_summary.items(), key=lambda x: x[1], reverse=True)
            ),
            "annotated_video_url": video_url,
            "video_duration_sec": round(total_frames / fps, 1),
        }

    except Exception as e:
        try:
            out_writer.release()
        except Exception:
            pass
        _job_status[job_id] = {"status": "error", "error": str(e)}
    finally:
        cap.release()
        db.close()
        try:
            os.remove(video_path)
        except Exception:
            pass


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    zone_type: str = Form("default"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {allowed}",
        )

    job_id = str(uuid.uuid4())
    filename = f"{job_id}{ext}"
    video_path = os.path.join(settings.UPLOAD_DIR, filename)

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    role = current_user.role or "Home"
    background_tasks.add_task(
        process_video_job, job_id, video_path, role, current_user.id, zone_type
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "zone_type": zone_type,
        "message": "Video upload accepted — violation scanning started",
    }


@router.get("/status/{job_id}")
def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    status = _job_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/result/{job_id}")
def download_annotated_video(
    job_id: str, current_user: User = Depends(get_current_user)
):
    """Stream the annotated output video to the browser."""
    path = _get_annotated_path(job_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Annotated video not ready yet")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"occusafe_analysis_{job_id[:8]}.mp4",
    )
