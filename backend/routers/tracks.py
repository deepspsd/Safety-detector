"""
Tracks router — exposes live tracking state to the dashboard.

Endpoints
---------
GET /tracks/{camera_id}           – all currently-tracked persons for a camera
GET /tracks/{camera_id}/{track_id} – single track with full history
GET /tracks                        – latest snapshot across ALL cameras
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from auth_utils import get_current_user
from database import SessionLocal, User
from services.tracking_layer import tracker as _tracker

router = APIRouter(prefix="/tracks", tags=["tracks"])


def _db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _serialise_track(camera_id: int, track_id: str) -> Optional[Dict[str, Any]]:
    """Convert a TrackObservation into a JSON-safe dict."""
    obs = _tracker.get_track(camera_id, track_id)
    if obs is None:
        return None

    now = time.time()
    idle_seconds = round(now - obs.last_moved_at, 1) if obs.last_moved_at else None

    return {
        "camera_id": camera_id,
        "track_id": track_id,
        "object_class": obs.object_class,
        "bbox": obs.bbox,
        "confidence": obs.confidence,
        "first_seen": obs.first_seen,
        "last_seen": obs.last_seen,
        "current_zone": obs.current_zone,
        "previous_zone": obs.previous_zone,
        "movement_state": obs.movement_state,
        "idle_seconds": idle_seconds,
        "velocity": obs.velocity,
        "attribute_predictions": obs.last_attribute_predictions or {},
        "history_length": len(obs.history),
    }


@router.get("")
def get_all_tracks(
    current_user: User = Depends(get_current_user),
):
    """Return the latest tracking snapshot for every camera."""
    result = []
    for (cam_id, tid) in _tracker.all_track_keys():
        s = _serialise_track(cam_id, tid)
        if s:
            result.append(s)
    return {"total": len(result), "tracks": result}


@router.get("/{camera_id}")
def get_camera_tracks(
    camera_id: int,
    current_user: User = Depends(get_current_user),
):
    """Return all currently-tracked persons for a specific camera."""
    keys = [(c, t) for (c, t) in _tracker.all_track_keys() if c == camera_id]
    tracks = []
    for (cam_id, tid) in keys:
        s = _serialise_track(cam_id, tid)
        if s:
            tracks.append(s)
    return {"camera_id": camera_id, "total": len(tracks), "tracks": tracks}


@router.get("/{camera_id}/{track_id}")
def get_single_track(
    camera_id: int,
    track_id: str,
    current_user: User = Depends(get_current_user),
):
    """Return a single track with full attribute history."""
    obs = _tracker.get_track(camera_id, track_id)
    if obs is None:
        raise HTTPException(status_code=404, detail="Track not found or expired")

    now = time.time()
    return {
        "camera_id": camera_id,
        "track_id": track_id,
        "object_class": obs.object_class,
        "bbox": obs.bbox,
        "confidence": obs.confidence,
        "first_seen": obs.first_seen,
        "last_seen": obs.last_seen,
        "current_zone": obs.current_zone,
        "previous_zone": obs.previous_zone,
        "movement_state": obs.movement_state,
        "idle_seconds": round(now - obs.last_moved_at, 1) if obs.last_moved_at else None,
        "velocity": obs.velocity,
        "attribute_predictions": obs.last_attribute_predictions or {},
        "history": [
            {"bbox": h["bbox"], "ts": h["ts"]}
            for h in (obs.history[-50:] if obs.history else [])
        ],
    }
