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


# ── Zone endpoints ─────────────────────────────────────────────────────────────
# Zones are polygon regions drawn in the camera frame's pixel space.
# zone_name examples: "entrance", "cashbox", "window", "dough_table_1"
# polygon_json: JSON string of [[x,y], [x,y], ...] (≥3 points)

from database import ZoneConfig
from services import zone_service as _zone_svc


class ZoneCreate(BaseModel):
    zone_name:    str = Field(..., min_length=1, max_length=200)
    polygon_json: str = Field(
        ...,
        description=(
            "JSON array of [x,y] pixel points, e.g. [[10,10],[200,10],[200,200],[10,200]]"
        ),
    )


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
    try:
        poly = _json.loads(payload.polygon_json)
        if not isinstance(poly, list) or len(poly) < 3:
            raise ValueError("Polygon must have at least 3 points")
        for pt in poly:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                raise ValueError(f"Each point must be [x, y], got: {pt}")
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
        db.commit()
        db.refresh(existing)
        zone = existing
        log.info(f"[cameras] Zone '{payload.zone_name}' updated for cam {camera_id}")
    else:
        zone = ZoneConfig(
            camera_id    = camera_id,
            zone_name    = payload.zone_name,
            polygon_json = payload.polygon_json,
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
