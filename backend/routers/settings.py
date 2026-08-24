"""
routers/settings.py — Admin settings & cylinder-log API
=========================================================
Endpoints
---------
  GET    /settings/              List all SystemSettings rows
  GET    /settings/{key}         Get one setting by key
  PUT    /settings/{key}         Update a threshold value (admin only)
  GET    /settings/cylinder-logs Paginated CylinderLog history
  POST   /cameras/{id}/baseline  Upload a dirty-floor baseline image
  DELETE /cameras/{id}/baseline  Remove the baseline for a camera
"""

import logging
import os
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Camera, CylinderLog, DirtyFloorBaseline, SystemSettings, get_db
from routers.auth import get_current_user
from services import rule_engine

log = logging.getLogger("routers.settings")

router = APIRouter(prefix="/settings", tags=["Admin Settings"])

# Directory where baseline images are stored
BASELINE_DIR = os.path.join("uploads", "baselines")
os.makedirs(BASELINE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────


class SettingOut(BaseModel):
    key: str
    value: str
    description: str | None = None

    class Config:
        from_attributes = True


class SettingUpdate(BaseModel):
    value: str


class CylinderLogOut(BaseModel):
    id: int
    camera_id: int
    event_type: str
    usage_day_count: int | None
    timestamp: str

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────────────────
# GET /settings/
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/", response_model=list[SettingOut])
def list_settings(
    db: Session = Depends(get_db),
    _current_user=Depends(get_current_user),
):
    """Return all system threshold settings (admin only)."""
    return db.query(SystemSettings).order_by(SystemSettings.key).all()


# ─────────────────────────────────────────────────────────────────────────────
# GET /settings/cylinder-logs
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/cylinder-logs", response_model=list[CylinderLogOut])
def list_cylinder_logs(
    camera_id: int | None = None,
    event_type: str | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    _current_user=Depends(get_current_user),
):
    """Paginated cylinder usage history with optional filters."""
    q = db.query(CylinderLog)
    if camera_id:
        q = q.filter(CylinderLog.camera_id == camera_id)
    if event_type:
        q = q.filter(CylinderLog.event_type == event_type)
    rows = q.order_by(CylinderLog.timestamp.desc()).limit(limit).all()
    return [
        CylinderLogOut(
            id=r.id,
            camera_id=r.camera_id,
            event_type=r.event_type,
            usage_day_count=r.usage_day_count,
            timestamp=r.timestamp.isoformat(),
        )
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# GET /settings/{key}
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/{key}", response_model=SettingOut)
def get_setting(
    key: str,
    db: Session = Depends(get_db),
    _current_user=Depends(get_current_user),
):
    row = db.query(SystemSettings).filter(SystemSettings.key == key).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Setting {key!r} not found")
    return row


# ─────────────────────────────────────────────────────────────────────────────
# PUT /settings/{key}
# ─────────────────────────────────────────────────────────────────────────────


@router.put("/{key}", response_model=SettingOut)
def update_setting(
    key: str,
    body: SettingUpdate,
    db: Session = Depends(get_db),
    _current_user=Depends(get_current_user),
):
    """
    Update a threshold setting.  Unknown keys are rejected — only keys that
    have default values in rule_engine.SETTING_DEFAULTS can be set.
    This prevents accidental typos from silently creating orphan rows.
    """
    if key not in rule_engine.SETTING_DEFAULTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown setting key {key!r}. "
                f"Valid keys: {sorted(rule_engine.SETTING_DEFAULTS)}"
            ),
        )

    row = db.query(SystemSettings).filter(SystemSettings.key == key).first()
    if row:
        row.value = body.value
    else:
        desc = rule_engine.SETTING_DEFAULTS[key][1]
        row = SystemSettings(key=key, value=body.value, description=desc)
        db.add(row)

    db.commit()
    db.refresh(row)

    # Bust the in-process cache so the new value takes effect immediately
    rule_engine.invalidate_settings_cache(key)
    log.info(f"[settings] Updated {key!r} = {body.value!r}")
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Baseline image management  (/cameras/{id}/baseline)
# ─────────────────────────────────────────────────────────────────────────────

baseline_router = APIRouter(tags=["Camera Baselines"])


@baseline_router.post("/cameras/{camera_id}/baseline", status_code=201)
async def upload_baseline(
    camera_id: int,
    zone_name: str = "default",
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _current_user=Depends(get_current_user),
):
    """
    Upload a clean-state reference photo for the dirty-floor detector.

    The admin should take this photo from the exact camera angle after morning
    cleaning, before the shift starts.  One photo per camera / zone is stored.

    The file is saved to uploads/baselines/cam_{camera_id}_{zone_name}.jpg
    and the DirtyFloorDetector for this camera is hot-reloaded immediately.
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG / PNG / WebP baseline images are accepted",
        )

    # Save to disk
    ext = os.path.splitext(file.filename or "baseline.jpg")[1] or ".jpg"
    filename = f"cam_{camera_id}_{zone_name}{ext}"
    abs_path = os.path.join(BASELINE_DIR, filename)
    with open(abs_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Upsert DB row
    existing = (
        db.query(DirtyFloorBaseline)
        .filter(
            DirtyFloorBaseline.camera_id == camera_id,
            DirtyFloorBaseline.zone_name == zone_name,
        )
        .first()
    )
    if existing:
        existing.image_path = abs_path
        existing.uploaded_at = __import__("datetime").datetime.utcnow()
    else:
        db.add(
            DirtyFloorBaseline(
                camera_id=camera_id,
                zone_name=zone_name,
                image_path=abs_path,
            )
        )
    db.commit()

    # Hot-reload the detector (no server restart needed)
    rule_engine.reload_detector(camera_id)

    log.info(
        f"[settings] Baseline uploaded cam={camera_id} zone={zone_name!r} path={abs_path!r}"
    )
    return {
        "camera_id": camera_id,
        "zone_name": zone_name,
        "image_path": abs_path,
        "message": "Baseline saved and detector reloaded",
    }


@baseline_router.delete("/cameras/{camera_id}/baseline", status_code=200)
def delete_baseline(
    camera_id: int,
    zone_name: str = "default",
    db: Session = Depends(get_db),
    _current_user=Depends(get_current_user),
):
    """Remove the dirty-floor baseline for a camera / zone."""
    row = (
        db.query(DirtyFloorBaseline)
        .filter(
            DirtyFloorBaseline.camera_id == camera_id,
            DirtyFloorBaseline.zone_name == zone_name,
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Baseline not found")

    # Delete file from disk if it exists
    try:
        if os.path.exists(row.image_path):
            os.remove(row.image_path)
    except Exception as exc:
        log.warning(f"[settings] Could not delete baseline file: {exc}")

    db.delete(row)
    db.commit()

    # Evict cached detector so it's rebuilt without a baseline next tick
    rule_engine.reload_detector(camera_id)

    return {"message": f"Baseline for camera {camera_id} / zone {zone_name!r} deleted"}
