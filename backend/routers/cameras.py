"""
routers/cameras.py — Admin CRUD endpoints for Camera management
===============================================================
These endpoints manage Camera rows in the DB AND hot-wire the
camera_manager to start/stop/restart reader threads without
requiring a server restart.

Auth: all endpoints require a valid JWT token.  Role enforcement
is lenient for now (any authenticated user can manage cameras);
tighten with a supervisor/admin role check once roles are finalised.

Endpoints
──────────
  GET    /cameras              → list all cameras + live status overlay
  POST   /cameras              → create camera in DB + start reader
  GET    /cameras/{id}         → single camera detail + live status
  PUT    /cameras/{id}         → update config (name/url/floor/zone_type)
  DELETE /cameras/{id}         → stop reader + mark status=offline
  POST   /cameras/{id}/restart → stop + restart reader thread
"""

from __future__ import annotations

import datetime
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db, Camera as CameraModel
from routers.auth import get_current_user
from database import User
from services import camera_manager

log = logging.getLogger("cameras_router")
router = APIRouter(prefix="/cameras", tags=["cameras"])


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class CameraCreate(BaseModel):
    name:      str             = Field(..., min_length=1, max_length=200)
    floor:     str             = Field(..., description="ground | first | second | shop")
    zone_type: Optional[str]   = Field(None, max_length=100)
    rtsp_url:  Optional[str]   = Field(None, max_length=500)
    # If status is omitted, defaults to "offline" (reader starts only when rtsp_url present)
    status:    str             = Field("offline", description="online | offline | error")


class CameraUpdate(BaseModel):
    name:      Optional[str]  = Field(None, min_length=1, max_length=200)
    floor:     Optional[str]  = None
    zone_type: Optional[str]  = None
    rtsp_url:  Optional[str]  = None
    status:    Optional[str]  = None


def _camera_to_dict(cam: CameraModel, live: Optional[dict] = None) -> dict:
    """
    Serialize a Camera ORM row to a dict.
    Optionally merges in live reader info (fps, has_frame, error) from camera_manager.
    """
    d = {
        "id":           cam.id,
        "name":         cam.name,
        "floor":        cam.floor,
        "zone_type":    cam.zone_type,
        "rtsp_url":     cam.rtsp_url,
        "status":       cam.status,
        "last_seen_at": cam.last_seen_at.isoformat() if cam.last_seen_at else None,
        "created_at":   cam.created_at.isoformat()   if cam.created_at   else None,
        # Live state overlay (present only when reader is running)
        "live": None,
    }
    if live:
        d["live"] = live
    elif camera_manager.is_running(cam.id):
        d["live"] = {
            "fps":       camera_manager.get_reader_fps(cam.id),
            "error":     camera_manager.get_reader_error(cam.id),
            "has_frame": camera_manager.get_latest_frame(cam.id) is not None,
        }
    return d


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/")
def list_cameras(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all Camera rows, with live reader state overlaid where running."""
    cameras = db.query(CameraModel).order_by(CameraModel.id).all()
    return [_camera_to_dict(c) for c in cameras]


@router.post("/", status_code=201)
def create_camera(
    payload: CameraCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a Camera row in the DB.
    If rtsp_url is provided and status is not 'offline', the reader thread
    is started immediately without restarting the server.
    """
    cam = CameraModel(
        name       = payload.name,
        floor      = payload.floor,
        zone_type  = payload.zone_type,
        rtsp_url   = payload.rtsp_url,
        status     = payload.status,
        created_at = datetime.datetime.utcnow(),
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)

    # Hot-start reader if URL present and not explicitly offline
    if cam.rtsp_url and cam.rtsp_url.strip() and cam.status != "offline":
        try:
            camera_manager.start_camera(cam.id, cam.name, cam.rtsp_url)
            log.info(f"[cameras] Reader started live for cam {cam.id}")
        except Exception as exc:
            log.error(f"[cameras] Failed to start reader for cam {cam.id}: {exc}")

    return _camera_to_dict(cam)


@router.get("/status")
def live_status(current_user: User = Depends(get_current_user)):
    """
    Return real-time reader state for all cameras currently in the registry.
    Useful for a dashboard status widget.
    """
    return camera_manager.list_status()


@router.get("/{camera_id}")
def get_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return _camera_to_dict(cam)


@router.put("/{camera_id}")
def update_camera(
    camera_id: int,
    payload: CameraUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update camera configuration.
    If rtsp_url changes and the camera is currently running, restarts the reader.
    """
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    url_changed = payload.rtsp_url is not None and payload.rtsp_url != cam.rtsp_url

    if payload.name      is not None: cam.name      = payload.name
    if payload.floor     is not None: cam.floor     = payload.floor
    if payload.zone_type is not None: cam.zone_type = payload.zone_type
    if payload.rtsp_url  is not None: cam.rtsp_url  = payload.rtsp_url
    if payload.status    is not None: cam.status    = payload.status

    db.commit()
    db.refresh(cam)

    # Restart reader if URL changed and camera should be running
    if url_changed and cam.rtsp_url and cam.status != "offline":
        try:
            camera_manager.restart_camera(cam.id, cam.name, cam.rtsp_url)
            log.info(f"[cameras] Reader restarted for cam {cam.id} (URL changed)")
        except Exception as exc:
            log.error(f"[cameras] Restart failed for cam {cam.id}: {exc}")

    return _camera_to_dict(cam)


@router.delete("/{camera_id}")
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Stop the camera reader thread and mark the camera as offline in the DB.
    The row is NOT deleted — use the DB directly for hard deletes if needed.
    This is a soft-stop to preserve historical alert/log data.
    """
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Stop the live reader (no-op if not running)
    camera_manager.stop_camera(camera_id)

    # Mark offline in DB
    cam.status = "offline"
    db.commit()
    db.refresh(cam)

    log.info(f"[cameras] Camera {camera_id} stopped and marked offline")
    return _camera_to_dict(cam)


@router.post("/{camera_id}/restart")
def restart_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Stop then restart the reader thread for a camera (e.g. to recover a stuck stream).
    The camera must have a valid rtsp_url in the DB.
    """
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not cam.rtsp_url or not cam.rtsp_url.strip():
        raise HTTPException(status_code=400, detail="Camera has no rtsp_url — set one first")

    try:
        camera_manager.restart_camera(cam.id, cam.name, cam.rtsp_url)
        cam.status = "online"
        db.commit()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Restart failed: {exc}")

    log.info(f"[cameras] Camera {camera_id} restarted via API")
    return {"message": f"Camera {camera_id} restarted", **_camera_to_dict(cam)}
