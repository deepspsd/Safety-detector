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

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import (get_db, Camera as CameraModel, CameraCredential,
                      CameraHealth, CameraStreamProfile)
from routers.auth import get_current_user
from database import User
from services import camera_manager
from services.camera_credentials import (CredentialConfigurationError, decrypt,
                                         encrypt, encrypt_credentials)
from services.onvif_client import OnvifConnectionError, OnvifUnavailable, inspect_camera
from services.camera_discovery import discover_onvif, probe_onvif_endpoints, hikvision_quick_add

log = logging.getLogger("cameras_router")
router = APIRouter(prefix="/cameras", tags=["cameras"])


# ── WebRTC Endpoint ────────────────────────────────────────────────────────
class WebRTCOffer(BaseModel):
    sdp: str
    type: str

@router.post("/{camera_id}/webrtc/offer")
async def webrtc_offer(
    camera_id: int,
    offer: WebRTCOffer,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from services.webrtc_streamer import webrtc_manager
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # We don't strictly require the reader to be online, 
    # but if it's offline webrtc_streamer will just send black frames.
    
    answer = await webrtc_manager.handle_offer(camera_id, offer.sdp, offer.type)
    return answer


# ── HikVision Quick-Add ────────────────────────────────────────────────────

class HikVisionQuickAdd(BaseModel):
    name:           str
    ip:             str
    username:       str
    password:       str
    floor:          str            = "ground"
    channel:        int            = 1
    ai_stream:      str            = "sub"    # "main" | "sub"
    display_stream: str            = "main"

@router.post("/hikvision-quick-add", summary="Add a HikVision camera by IP + credentials")
def hikvision_add(
    payload:      HikVisionQuickAdd,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    """
    One-step HikVision camera add.
    Builds the RTSP URL from the supplied IP, username, password and channel,
    then creates the camera row and starts the reader — no manual URL needed.
    """
    cam_dict = hikvision_quick_add(
        name           = payload.name,
        ip             = payload.ip,
        username       = payload.username,
        password       = payload.password,
        floor          = payload.floor,
        channel        = payload.channel,
        ai_stream      = payload.ai_stream,
        display_stream = payload.display_stream,
    )
    cam = CameraModel(**{
        k: v for k, v in cam_dict.items()
        if hasattr(CameraModel, k)
    })
    cam.created_at = datetime.datetime.utcnow()
    db.add(cam)
    db.commit()
    db.refresh(cam)

    # Start reader thread immediately
    try:
        camera_manager.start_camera(cam.id, cam.name, cam_dict["rtsp_url"], floor=cam.floor)
    except Exception as exc:
        log.warning("[cameras] hikvision-quick-add reader start warning: %s", exc)

    return {"id": cam.id, "name": cam.name, "rtsp_url": cam_dict["rtsp_url"], "status": "started"}


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class CameraCreate(BaseModel):
    name:      str             = Field(..., min_length=1, max_length=200)
    floor:     str             = Field(..., description="ground | first | second | shop")
    zone_type: Optional[str]   = Field(None, max_length=100)
    rtsp_url:  Optional[str]   = Field(None, max_length=500)
    # If status is omitted, defaults to "offline" (reader starts only when rtsp_url present)
    status:    str             = Field("offline", description="online | offline | error")
    camera_code: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    purpose: Optional[str] = Field(None, max_length=200)
    camera_type: Optional[str] = Field(None, max_length=50)
    mount_height_m: Optional[float] = None
    view_direction: Optional[str] = Field(None, max_length=100)
    resolution: Optional[str] = Field(None, max_length=50)
    configured_fps: Optional[float] = Field(None, ge=0)
    rule_profile_id: Optional[int] = None
    workflow_profile_id: Optional[int] = None
    ai_enabled: bool = True
    supports_multi_zone: bool = True
    supports_ocr: bool = False
    supports_pose: bool = False
    supports_tracking: bool = True
    supports_recording: bool = False
    supports_snapshot: bool = True


class CameraUpdate(BaseModel):
    name:      Optional[str]  = Field(None, min_length=1, max_length=200)
    floor:     Optional[str]  = None
    zone_type: Optional[str]  = None
    rtsp_url:  Optional[str]  = None
    status:    Optional[str]  = None
    camera_code: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    purpose: Optional[str] = Field(None, max_length=200)
    camera_type: Optional[str] = Field(None, max_length=50)
    mount_height_m: Optional[float] = None
    view_direction: Optional[str] = Field(None, max_length=100)
    resolution: Optional[str] = Field(None, max_length=50)
    configured_fps: Optional[float] = Field(None, ge=0)
    rule_profile_id: Optional[int] = None
    workflow_profile_id: Optional[int] = None
    ai_enabled: Optional[bool] = None
    supports_multi_zone: Optional[bool] = None
    supports_ocr: Optional[bool] = None
    supports_pose: Optional[bool] = None
    supports_tracking: Optional[bool] = None
    supports_recording: Optional[bool] = None
    supports_snapshot: Optional[bool] = None


class CameraRegistration(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    floor: str = Field(..., pattern="^(ground|first|second|shop)$")
    zone_type: Optional[str] = Field(None, max_length=100)
    onvif_endpoint: str = Field(..., max_length=500)
    username: str = Field(..., min_length=1, max_length=120)
    password: str = Field(..., min_length=1, max_length=500)
    preferred_stream: str = Field("sub", pattern="^(main|sub)$")
    ai_stream: str = Field("sub", pattern="^(main|sub)$")


class DiscoveryRequest(BaseModel):
    timeout_seconds: float = Field(5.0, ge=1.0, le=20.0)
    interface: Optional[str] = None
    retries: int = Field(1, ge=1, le=3)
    fallback_subnet: Optional[str] = None


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
        "rtsp_configured": bool(cam.rtsp_url or cam.streams),
        "status":       cam.status,
        "last_seen_at": cam.last_seen_at.isoformat() if cam.last_seen_at else None,
        "created_at":   cam.created_at.isoformat()   if cam.created_at   else None,
        "camera_code": cam.camera_code,
        "department": cam.department,
        "purpose": cam.purpose,
        "camera_type": cam.camera_type,
        "mount_height_m": cam.mount_height_m,
        "view_direction": cam.view_direction,
        "resolution": cam.resolution,
        "configured_fps": cam.configured_fps,
        "calibration_status": cam.calibration_status,
        "calibration_version": cam.calibration_version,
        "last_calibrated_at": cam.last_calibrated_at.isoformat() if cam.last_calibrated_at else None,
        "rule_profile_id": cam.rule_profile_id,
        "workflow_profile_id": cam.workflow_profile_id,
        "ai_enabled": cam.ai_enabled,
        "capabilities": {"multi_zone": cam.supports_multi_zone, "ocr": cam.supports_ocr,
                         "pose": cam.supports_pose, "tracking": cam.supports_tracking,
                         "recording": cam.supports_recording, "snapshot": cam.supports_snapshot},
        "heartbeat": cam.heartbeat_at.isoformat() if cam.heartbeat_at else None,
        "health_status": cam.health_status,
        "drift_score": cam.drift_score,
        "manufacturer": cam.manufacturer,
        "model": cam.model,
        "ip_address": cam.ip_address,
        "onvif_endpoint": cam.onvif_endpoint,
        "preferred_stream": cam.preferred_stream,
        "ai_stream": cam.ai_stream,
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
    cameras = db.query(CameraModel).filter(CameraModel.status != "deleted").order_by(CameraModel.id).all()
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
        camera_code = payload.camera_code,
        department = payload.department,
        purpose = payload.purpose,
        camera_type = payload.camera_type,
        mount_height_m = payload.mount_height_m,
        view_direction = payload.view_direction,
        resolution = payload.resolution,
        configured_fps = payload.configured_fps,
        rule_profile_id = payload.rule_profile_id,
        workflow_profile_id = payload.workflow_profile_id,
        ai_enabled = payload.ai_enabled,
        supports_multi_zone = payload.supports_multi_zone,
        supports_ocr = payload.supports_ocr,
        supports_pose = payload.supports_pose,
        supports_tracking = payload.supports_tracking,
        supports_recording = payload.supports_recording,
        supports_snapshot = payload.supports_snapshot,
        created_at = datetime.datetime.utcnow(),
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)

    # Hot-start reader if URL present and not explicitly offline
    if cam.rtsp_url and cam.rtsp_url.strip() and cam.status != "offline":
        try:
            camera_manager.start_camera(cam.id, cam.name, cam.rtsp_url, floor=cam.floor)
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


@router.post("/register", status_code=201)
def register_onvif_camera(
    payload: CameraRegistration,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Register a real ONVIF camera without placing credentials in the camera row.

    The endpoint authenticates against the device, resolves its own media
    profiles/RTSP URIs, encrypts credentials and stream URIs, then starts the
    selected stream. Passwords are never serialized or logged.
    """
    try:
        inspected = inspect_camera(payload.onvif_endpoint, payload.username, payload.password)
        encrypted_username, encrypted_password = encrypt_credentials(payload.username, payload.password)
    except CredentialConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except OnvifUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except OnvifConnectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    cam = CameraModel(
        name=payload.name, floor=payload.floor, zone_type=payload.zone_type,
        status="offline", manufacturer=inspected.get("manufacturer"),
        model=inspected.get("model"), ip_address=inspected["ip_address"],
        onvif_endpoint=payload.onvif_endpoint, preferred_stream=payload.preferred_stream,
        ai_stream=payload.ai_stream, created_at=datetime.datetime.utcnow(),
    )
    db.add(cam)
    db.flush()
    db.add(CameraCredential(camera_id=cam.id, encrypted_username=encrypted_username, encrypted_password=encrypted_password))
    for stream in inspected["streams"]:
        db.add(CameraStreamProfile(
            camera_id=cam.id, profile_token=stream.get("profile_token"), stream_type=stream["stream_type"],
            codec=stream.get("codec"), width=stream.get("width"), height=stream.get("height"), fps=stream.get("fps"),
            encrypted_rtsp_uri=encrypt(stream["rtsp_uri"]),
        ))
    db.commit()
    db.refresh(cam)
    _start_selected_stream(cam, db)
    return _camera_to_dict(cam)


def _selected_stream(cam: CameraModel, db: Session, stream_type: str | None = None) -> CameraStreamProfile | None:
    desired = stream_type or cam.preferred_stream or "sub"
    return (db.query(CameraStreamProfile)
            .filter(CameraStreamProfile.camera_id == cam.id, CameraStreamProfile.active.is_(True), CameraStreamProfile.stream_type == desired)
            .order_by(CameraStreamProfile.id)
            .first()) or (db.query(CameraStreamProfile)
                         .filter(CameraStreamProfile.camera_id == cam.id, CameraStreamProfile.active.is_(True))
                         .order_by(CameraStreamProfile.id).first())


def _stream_url(cam: CameraModel, db: Session, stream_type: str | None = None) -> str:
    profile = _selected_stream(cam, db, stream_type)
    if profile:
        return decrypt(profile.encrypted_rtsp_uri)
    if cam.rtsp_url:
        # Legacy cameras created before encrypted registration continue to work.
        return cam.rtsp_url
    raise HTTPException(status_code=409, detail="Camera has no configured stream")


def _start_selected_stream(cam: CameraModel, db: Session) -> None:
    try:
        camera_manager.start_camera(cam.id, cam.name, _stream_url(cam, db), floor=cam.floor)
        cam.status = "online"
        db.commit()
    except CredentialConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        cam.status = "error"
        db.commit()
        log.warning("Camera %s could not start: %s", cam.id, type(exc).__name__)


@router.get("/{camera_id}/health")
def camera_health(camera_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    metrics = camera_manager.get_metrics(camera_id)
    return {"camera_id": camera_id, "status": cam.health_status or cam.status, **metrics}


@router.get("/{camera_id}/streams")
def camera_streams(camera_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    profiles = db.query(CameraStreamProfile).filter(CameraStreamProfile.camera_id == camera_id, CameraStreamProfile.active.is_(True)).all()
    return [{"id": p.id, "profile_token": p.profile_token, "stream_type": p.stream_type, "codec": p.codec,
             "width": p.width, "height": p.height, "fps": p.fps} for p in profiles]


@router.post("/{camera_id}/streams/select")
def select_camera_stream(camera_id: int, stream_type: str = Query(..., pattern="^(main|sub)$"), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    profile = _selected_stream(cam, db, stream_type)
    if not profile:
        raise HTTPException(status_code=409, detail="Requested stream is unavailable")
    cam.preferred_stream = profile.stream_type
    camera_manager.restart_camera(cam.id, cam.name, _stream_url(cam, db, profile.stream_type), floor=cam.floor)
    cam.status = "online"
    db.commit()
    return {"camera_id": cam.id, "stream_type": profile.stream_type, "status": "streaming"}


@router.post("/{camera_id}/connect")
def connect_camera(camera_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    _start_selected_stream(cam, db)
    return {"camera_id": cam.id, "status": cam.status}


@router.post("/{camera_id}/disconnect")
def disconnect_camera(camera_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    camera_manager.stop_camera(camera_id)
    cam.status = "offline"
    db.commit()
    return {"camera_id": camera_id, "status": "offline"}


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
    for field in ("camera_code", "department", "purpose", "camera_type", "mount_height_m", "view_direction",
                  "resolution", "configured_fps", "rule_profile_id", "workflow_profile_id", "ai_enabled",
                  "supports_multi_zone", "supports_ocr", "supports_pose", "supports_tracking", "supports_recording",
                  "supports_snapshot"):
        value = getattr(payload, field)
        if value is not None:
            setattr(cam, field, value)

    db.commit()
    db.refresh(cam)

    # Restart reader if URL changed and camera should be running
    if url_changed and cam.rtsp_url and cam.status != "offline":
        try:
            camera_manager.restart_camera(cam.id, cam.name, cam.rtsp_url, floor=cam.floor)
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
    Stop the camera reader thread and delete the camera from the DB.
    If there are foreign key constraints (e.g. existing alerts), it falls back
    to soft-deleting by setting status="deleted".
    """
    from sqlalchemy.exc import IntegrityError
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Stop the live reader (no-op if not running)
    camera_manager.stop_camera(camera_id)

    # Try hard delete
    try:
        db.delete(cam)
        db.commit()
        log.info(f"[cameras] Camera {camera_id} hard-deleted")
        return {"id": camera_id, "status": "deleted"}
    except IntegrityError:
        # Fallback to soft delete
        db.rollback()
        cam.status = "deleted"
        db.commit()
        db.refresh(cam)
        log.info(f"[cameras] Camera {camera_id} soft-deleted (status=deleted)")
        return _camera_to_dict(cam)


@router.post("/{camera_id}/restart")
def restart_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Stop then restart the reader thread for a camera (e.g. to recover a stuck stream).
    Supports both legacy plaintext URLs and encrypted ONVIF stream profiles.
    """
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    try:
        stream_url = _stream_url(cam, db)
        camera_manager.restart_camera(cam.id, cam.name, stream_url, floor=cam.floor)
        cam.status = "online"
        db.commit()
    except CredentialConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Restart failed: {exc}")

    log.info(f"[cameras] Camera {camera_id} restarted via API")
    return {"message": f"Camera {camera_id} restarted", **_camera_to_dict(cam)}


# ── Zone endpoints ─────────────────────────────────────────────────────────────
# Zones are polygon regions drawn in the camera frame's pixel space.
# zone_name examples: "entrance", "cashbox", "window", "dough_table_1"
# polygon_json: JSON string of [[x,y], [x,y], ...] (≥3 points)

from database import ZoneConfig
from services import zone_service as _zone_svc


def _json_load(value: Optional[str], default):
    import json
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


class ZoneCreate(BaseModel):
    zone_name:    str = Field(..., min_length=1, max_length=200)
    polygon_json: str = Field(
        ...,
        description=(
            "JSON array of [x,y] pixel points, e.g. [[10,10],[200,10],[200,200],[10,200]]"
        ),
    )
    zone_type: Optional[str] = Field(None, max_length=100)
    preset_type: Optional[str] = Field(None, max_length=100)
    display_name: Optional[str] = Field(None, max_length=200)
    color: Optional[str] = Field(None, max_length=20)
    priority: int = 0
    workflow_stage: Optional[str] = Field(None, max_length=100)
    rule_profile_id: Optional[int] = None
    expected_objects: Optional[list[str]] = None
    allowed_objects: Optional[list[str]] = None
    forbidden_objects: Optional[list[str]] = None
    time_constraints: Optional[dict] = None
    alert_thresholds: Optional[dict] = None
    movement_threshold: Optional[float] = None
    idle_threshold: Optional[float] = None
    confidence_threshold: Optional[float] = None
    visibility_threshold: Optional[float] = None


@router.get("/{camera_id}/zones")
def get_zones(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return all configured zone polygons for this camera.
    Response: list of { id, zone_name, polygon_json, created_at }
    """
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    zones = db.query(ZoneConfig).filter(ZoneConfig.camera_id == camera_id).all()
    return [
        {
            "id":           z.id,
            "zone_name":    z.zone_name,
            "polygon_json": z.polygon_json,
            "created_at":   z.created_at.isoformat() if z.created_at else None,
            "zone_type": z.zone_type,
            "preset_type": z.preset_type,
            "display_name": z.display_name,
            "color": z.color,
            "priority": z.priority,
            "workflow_stage": z.workflow_stage,
            "rule_profile_id": z.rule_profile_id,
            "expected_objects": _json_load(z.expected_objects_json, []),
            "allowed_objects": _json_load(z.allowed_objects_json, []),
            "forbidden_objects": _json_load(z.forbidden_objects_json, []),
            "time_constraints": _json_load(z.time_constraints_json, {}),
            "alert_thresholds": _json_load(z.alert_thresholds_json, {}),
            "movement_threshold": z.movement_threshold,
            "idle_threshold": z.idle_threshold,
            "confidence_threshold": z.confidence_threshold,
            "visibility_threshold": z.visibility_threshold,
            "is_active": z.is_active,
            "calibration_version": z.calibration_version,
        }
        for z in zones
    ]


@router.post("/{camera_id}/zones", status_code=201)
def create_or_replace_zone(
    camera_id: int,
    payload: ZoneCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create or replace a named zone polygon for this camera.
    If a zone with the same zone_name already exists, its polygon is replaced
    (upsert by zone_name — keeps calibration idempotent).

    Polygon coordinates are raw pixel values in the camera's frame resolution.
    Re-calibrate if the camera resolution changes.

    After saving, the in-process zone cache is invalidated so the detection
    daemon picks up the new polygon immediately.
    """
    cam = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Validate polygon JSON
    import json as _json
    from services.calibration_service import validate_polygon
    try:
        poly = _json.loads(payload.polygon_json)
        if not isinstance(poly, list) or len(poly) < 3:
            raise ValueError("Polygon must have at least 3 points")
        for pt in poly:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                raise ValueError(f"Each point must be [x, y], got: {pt}")
        validate_polygon(poly)
    except (ValueError, _json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid polygon_json: {exc}")

    # Upsert
    existing = (
        db.query(ZoneConfig)
        .filter(ZoneConfig.camera_id == camera_id, ZoneConfig.zone_name == payload.zone_name)
        .first()
    )
    if existing:
        existing.polygon_json = payload.polygon_json
        for field, value in {
            "zone_type": payload.zone_type, "preset_type": payload.preset_type,
            "display_name": payload.display_name, "color": payload.color,
            "priority": payload.priority, "workflow_stage": payload.workflow_stage,
            "rule_profile_id": payload.rule_profile_id,
            "expected_objects_json": _json.dumps(payload.expected_objects) if payload.expected_objects is not None else None,
            "allowed_objects_json": _json.dumps(payload.allowed_objects) if payload.allowed_objects is not None else None,
            "forbidden_objects_json": _json.dumps(payload.forbidden_objects) if payload.forbidden_objects is not None else None,
            "time_constraints_json": _json.dumps(payload.time_constraints) if payload.time_constraints is not None else None,
            "alert_thresholds_json": _json.dumps(payload.alert_thresholds) if payload.alert_thresholds is not None else None,
            "movement_threshold": payload.movement_threshold, "idle_threshold": payload.idle_threshold,
            "confidence_threshold": payload.confidence_threshold, "visibility_threshold": payload.visibility_threshold,
        }.items():
            if value is not None:
                setattr(existing, field, value)
        db.commit()
        db.refresh(existing)
        zone = existing
        log.info(f"[cameras] Zone '{payload.zone_name}' updated for cam {camera_id}")
    else:
        zone = ZoneConfig(
            camera_id    = camera_id,
            zone_name    = payload.zone_name,
            polygon_json = payload.polygon_json,
            zone_type = payload.zone_type,
            preset_type = payload.preset_type,
            display_name = payload.display_name,
            color = payload.color,
            priority = payload.priority,
            workflow_stage = payload.workflow_stage,
            rule_profile_id = payload.rule_profile_id,
            expected_objects_json = _json.dumps(payload.expected_objects) if payload.expected_objects is not None else None,
            allowed_objects_json = _json.dumps(payload.allowed_objects) if payload.allowed_objects is not None else None,
            forbidden_objects_json = _json.dumps(payload.forbidden_objects) if payload.forbidden_objects is not None else None,
            time_constraints_json = _json.dumps(payload.time_constraints) if payload.time_constraints is not None else None,
            alert_thresholds_json = _json.dumps(payload.alert_thresholds) if payload.alert_thresholds is not None else None,
            movement_threshold = payload.movement_threshold,
            idle_threshold = payload.idle_threshold,
            confidence_threshold = payload.confidence_threshold,
            visibility_threshold = payload.visibility_threshold,
        )
        db.add(zone)
        db.commit()
        db.refresh(zone)
        log.info(f"[cameras] Zone '{payload.zone_name}' created for cam {camera_id}")

    # Bust zone cache so daemon picks it up immediately
    _zone_svc.invalidate_zone_cache(camera_id)

    return {
        "id":           zone.id,
        "camera_id":    camera_id,
        "zone_name":    zone.zone_name,
        "polygon_json": zone.polygon_json,
        "created_at":   zone.created_at.isoformat() if zone.created_at else None,
        "zone_type": zone.zone_type,
        "preset_type": zone.preset_type,
        "display_name": zone.display_name,
        "color": zone.color,
        "priority": zone.priority,
        "workflow_stage": zone.workflow_stage,
        "rule_profile_id": zone.rule_profile_id,
        "expected_objects": _json_load(zone.expected_objects_json, []),
        "allowed_objects": _json_load(zone.allowed_objects_json, []),
        "forbidden_objects": _json_load(zone.forbidden_objects_json, []),
        "time_constraints": _json_load(zone.time_constraints_json, {}),
        "alert_thresholds": _json_load(zone.alert_thresholds_json, {}),
        "movement_threshold": zone.movement_threshold,
        "idle_threshold": zone.idle_threshold,
        "confidence_threshold": zone.confidence_threshold,
        "visibility_threshold": zone.visibility_threshold,
        "calibration_version": zone.calibration_version,
    }


@router.delete("/{camera_id}/zones/{zone_name}", status_code=200)
def delete_zone(
    camera_id: int,
    zone_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a named zone polygon for this camera.
    Busts the in-process zone cache immediately.
    """
    zone = (
        db.query(ZoneConfig)
        .filter(ZoneConfig.camera_id == camera_id, ZoneConfig.zone_name == zone_name)
        .first()
    )
    if not zone:
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{zone_name}' not found for camera {camera_id}"
        )
    db.delete(zone)
    db.commit()
    _zone_svc.invalidate_zone_cache(camera_id)
    log.info(f"[cameras] Zone '{zone_name}' deleted for cam {camera_id}")
    return {"message": f"Zone '{zone_name}' deleted"}


# ── Snapshot endpoint ──────────────────────────────────────────────────────────

import base64 as _b64
import cv2 as _cv2


@router.get("/{camera_id}/snapshot")
def get_snapshot(
    camera_id: int,
    current_user: User = Depends(get_current_user),
):
    """
    Return the latest frame from the camera daemon as a JPEG base64 data URI.
    Used by the zone calibration UI to display the live frame as a canvas background.
    Returns { frame_b64: "data:image/jpeg;base64,..." } or { frame_b64: null }
    if the camera is offline or has no frame yet.
    """
    frame = camera_manager.get_latest_frame(camera_id)
    if frame is None:
        return {"frame_b64": None}
    try:
        ok, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return {"frame_b64": None}
        b64 = _b64.b64encode(buf.tobytes()).decode("ascii")
        return {"frame_b64": f"data:image/jpeg;base64,{b64}"}
    except Exception as exc:
        log.error(f"[cameras] Snapshot encode failed for cam {camera_id}: {exc}")
        return {"frame_b64": None}


# ── LAN camera discovery ───────────────────────────────────────────────────────
# Probes the local subnet for IP cameras by scanning common RTSP (554) and
# HTTP (80) ports.  This is a connectivity ping — no authentication is
# attempted.  The admin confirms and manually enters credentials before adding.
#
# Approach: pure Python socket.connect_ex scan (no zeep/WSDL/ONVIF library
# needed for discovery), similar to what Hikvision's iVMS does internally.
# Defaults to the server's own /24 subnet.

import ipaddress as _ipaddress
import socket as _socket
import concurrent.futures as _futures


def _probe_host(ip: str, ports: list[int], timeout: float = 0.4) -> Optional[dict]:
    """Try connecting to each port. Return host info if any port responds."""
    for port in ports:
        try:
            with _socket.create_connection((ip, port), timeout=timeout):
                # Try to get hostname
                try:
                    hostname = _socket.gethostbyaddr(ip)[0]
                except Exception:
                    hostname = ""

                # Guess RTSP URL patterns (Hikvision / Dahua / generic)
                rtsp_guesses = []
                if port == 554:
                    rtsp_guesses = [
                        f"rtsp://<user>:<pass>@{ip}:554/Streaming/Channels/101",   # Hikvision
                        f"rtsp://<user>:<pass>@{ip}:554/cam/realmonitor?channel=1&subtype=0",  # Dahua
                        f"rtsp://{ip}:554/stream1",                                 # Generic
                    ]
                elif port == 80:
                    rtsp_guesses = [f"http://{ip}/video"]

                return {
                    "ip":           ip,
                    "open_port":    port,
                    "hostname":     hostname,
                    "rtsp_guesses": rtsp_guesses,
                }
        except (ConnectionRefusedError, TimeoutError, OSError):
            continue
    return None


@router.post("/discover")
def discover_onvif_cameras(
    payload: DiscoveryRequest,
    current_user: User = Depends(get_current_user),
):
    """Discover ONVIF devices first; optionally run a constrained private-LAN fallback.

    Fallback candidates are deliberately *not* registered or labelled as
    cameras until the operator supplies credentials and ONVIF registration
    validates them.
    """
    devices = discover_onvif(payload.timeout_seconds, payload.interface, payload.retries)
    fallback: list[dict] = []
    if payload.fallback_subnet:
        try:
            fallback = probe_onvif_endpoints(payload.fallback_subnet)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    deduped = {device["onvif_endpoint"]: device for device in devices}
    for device in fallback:
        deduped.setdefault(device["onvif_endpoint"], device)
    values = list(deduped.values())
    log.info("ONVIF discovery returned %s validated device(s), %s candidate(s)", len(devices), len(fallback))
    return {"devices": values, "onvif_devices": len(devices), "fallback_candidates": len(fallback)}


@router.post("/discover/legacy-probe")
def discover_cameras(
    current_user: User = Depends(get_current_user),
):
    """
    Scan the local LAN subnet for IP cameras by probing ports 554 (RTSP)
    and 80 (HTTP).  Returns a list of hosts that responded.

    ⚠️  This is a best-effort ping scan — it does NOT guarantee the host is
    a camera, and it does NOT attempt authentication.  The admin should
    review results, select real cameras, enter credentials, and use
    POST /cameras to add them officially.

    Scans the /24 subnet of the server's primary outbound interface.
    Max 254 hosts × 2 ports = 508 probes with 0.4 s timeout, parallelised
    in a thread pool (takes ~3-8 seconds on a typical LAN).
    """
    # Determine local outbound IP
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "192.168.1.1"

    # Build /24 host list (skip .0 and .255)
    try:
        network = _ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
        hosts   = [str(h) for h in network.hosts()]
    except Exception:
        hosts = [f"192.168.1.{i}" for i in range(1, 255)]

    ports_to_try = [554, 80]

    found = []
    with _futures.ThreadPoolExecutor(max_workers=64) as pool:
        futs = {pool.submit(_probe_host, ip, ports_to_try): ip for ip in hosts}
        for fut in _futures.as_completed(futs):
            result = fut.result()
            if result:
                found.append(result)

    found.sort(key=lambda x: _ipaddress.IPv4Address(x["ip"]))
    log.info(f"[cameras] Discovery scan found {len(found)} hosts on {local_ip}/24")
    return {
        "subnet":       f"{local_ip}/24",
        "hosts_scanned": len(hosts),
        "cameras_found": found,
    }
