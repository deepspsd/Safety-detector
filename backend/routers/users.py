from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from database import get_db, User, UserConfig
from routers.auth import get_current_user

router = APIRouter(prefix="/users", tags=["users"])


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None


class ConfigUpdate(BaseModel):
    camera_type: Optional[str] = None
    rtsp_url: Optional[str] = None
    notify_sound: Optional[bool] = None
    notify_ui: Optional[bool] = None
    detection_sensitivity: Optional[float] = None


@router.put("/me")
def update_profile(data: ProfileUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if data.name is not None:
        current_user.name = data.name
    if data.role is not None:
        current_user.role = data.role
    db.commit()
    db.refresh(current_user)
    return {"id": current_user.id, "email": current_user.email, "name": current_user.name, "role": current_user.role}


@router.get("/me/config")
def get_config(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    config = db.query(UserConfig).filter(UserConfig.user_id == current_user.id).first()
    if not config:
        config = UserConfig(user_id=current_user.id)
        db.add(config)
        db.commit()
        db.refresh(config)
    return {
        "camera_type": config.camera_type,
        "rtsp_url": config.rtsp_url,
        "notify_sound": config.notify_sound,
        "notify_ui": config.notify_ui,
        "detection_sensitivity": config.detection_sensitivity
    }


@router.put("/me/config")
def update_config(data: ConfigUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    config = db.query(UserConfig).filter(UserConfig.user_id == current_user.id).first()
    if not config:
        config = UserConfig(user_id=current_user.id)
        db.add(config)
    if data.camera_type is not None:
        config.camera_type = data.camera_type
    if data.rtsp_url is not None:
        config.rtsp_url = data.rtsp_url
    if data.notify_sound is not None:
        config.notify_sound = data.notify_sound
    if data.notify_ui is not None:
        config.notify_ui = data.notify_ui
    if data.detection_sensitivity is not None:
        config.detection_sensitivity = data.detection_sensitivity
    db.commit()
    return {"message": "Config updated"}
