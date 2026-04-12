"""
Video upload & processing router — v2.0
Accepts an uploaded video file, processes frames using
the violation detection pipeline (process_frame_numpy),
and returns a summary of violations found.
"""
import os
import cv2
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from database import get_db, User
from routers.auth import get_current_user
from services import yolo_service, face_service
from services.alert_service import save_alert
from config import settings

router = APIRouter(prefix="/video", tags=["video"])

# In-memory job store (use Redis in production)
_job_status: dict = {}


def process_video_job(job_id: str, video_path: str, role: str, user_id: int):
    """Background task: process video file frame-by-frame."""
    _job_status[job_id] = {
        "status": "processing",
        "progress": 0,
        "alerts": [],
        "frames_processed": 0,
        "total_violations": 0,
    }

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_skip = max(1, int(fps // 5))   # ~5 frames/sec

    alerts_found = []
    frame_num = 0
    processed = 0
    total_violations = 0
    ppe_summary: dict = {}   # {ppe_item_label: count_of_frames_violated}

    from database import SessionLocal
    db = SessionLocal()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num % frame_skip != 0:
                continue

            processed += 1
            progress = int((frame_num / max(total_frames, 1)) * 100)
            _job_status[job_id]["progress"] = progress
            _job_status[job_id]["frames_processed"] = processed

            # ── Run detection pipeline (numpy → no encode/decode) ──
            if role == "Home":
                # Face service needs base64 — encode once
                b64 = yolo_service.encode_frame(frame)
                result = face_service.process_face_frame(b64, user_id, db)
            else:
                result = yolo_service.process_frame_numpy(
                    frame, role, frame_index=frame_num
                )

            if not result.get("is_compliant") and result.get("alert_message"):
                ts_sec = round(frame_num / fps, 1)
                n_violations = result.get("violations_count", 0)
                total_violations += n_violations

                # Track per-PPE-item violation counts for the summary section
                for item in result.get("missing_items", []):
                    ppe_summary[item] = ppe_summary.get(item, 0) + 1

                alert_info = {
                    "timestamp_sec":  ts_sec,
                    "message":        result["alert_message"],
                    "severity":       result["severity"],
                    "missing_items":  result.get("missing_items", []),
                    "violations_count": n_violations,
                    "persons_count":  result.get("persons_count", 0),
                }
                # Attach thumbnail for first 10 alerts only (keeps memory usage low)
                if len(alerts_found) < 10 and result.get("annotated_frame"):
                    alert_info["thumbnail_b64"] = result["annotated_frame"]

                alerts_found.append(alert_info)

                # Save to DB (cap at 20 distinct alerts)
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

        _job_status[job_id] = {
            "status":          "complete",
            "progress":        100,
            "frames_processed": processed,
            "total_alerts":    len(alerts_found),
            "total_violations": total_violations,
            "alerts":          alerts_found[:50],
            # Per-PPE-item breakdown (sorted by most-violated first)
            "ppe_summary":     dict(
                sorted(ppe_summary.items(), key=lambda x: x[1], reverse=True)
            ),
        }

    except Exception as e:
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

    job_id    = str(uuid.uuid4())
    filename  = f"{job_id}{ext}"
    video_path = os.path.join(settings.UPLOAD_DIR, filename)

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    role = current_user.role or "Home"
    background_tasks.add_task(
        process_video_job, job_id, video_path, role, current_user.id
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video upload accepted — violation scanning started",
    }


@router.get("/status/{job_id}")
def get_job_status(
    job_id: str, current_user: User = Depends(get_current_user)
):
    status = _job_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status
