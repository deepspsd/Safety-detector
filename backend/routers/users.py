import json
from typing import List, Optional

from database import User, UserConfig, get_db
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from routers.auth import get_current_user
from sqlalchemy.orm import Session

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
    # Custom PPE list — list of violation class names e.g. ["NO-Hardhat","NO-Gloves"]
    custom_ppe_items: Optional[List[str]] = None
    # Phone zone setting — True = any phone detected triggers an alert
    no_phone_zone: Optional[bool] = None


def _get_or_create_config(db: Session, user_id: int) -> UserConfig:
    config = db.query(UserConfig).filter(UserConfig.user_id == user_id).first()
    if not config:
        config = UserConfig(user_id=user_id)
        db.add(config)
        db.commit()
        db.refresh(config)
    return config


def _parse_custom_ppe(config: UserConfig) -> List[str]:
    """Safely parse the JSON-encoded custom_ppe_items column."""
    if not config.custom_ppe_items:
        return []
    try:
        val = json.loads(config.custom_ppe_items)
        return val if isinstance(val, list) else []
    except Exception:
        return []


@router.put("/me")
def update_profile(
    data: ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    valid_roles = {
        "Construction Worker",
        "Doctor",
        "Traffic Police",
        "College",
        "Home",
        "None",
        "Bakery Worker",
    }
    if data.name is not None:
        current_user.name = data.name
    if data.role is not None:
        current_user.role = data.role if data.role in valid_roles else current_user.role
    db.commit()
    db.refresh(current_user)
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "role": current_user.role,
    }


@router.get("/me/config")
def get_config(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    config = _get_or_create_config(db, current_user.id)
    return {
        "camera_type": config.camera_type,
        "rtsp_url": config.rtsp_url,
        "notify_sound": config.notify_sound,
        "notify_ui": config.notify_ui,
        "detection_sensitivity": config.detection_sensitivity,
        "custom_ppe_items": _parse_custom_ppe(config),
        "no_phone_zone": (
            bool(config.no_phone_zone) if config.no_phone_zone is not None else False
        ),
    }


@router.put("/me/config")
def update_config(
    data: ConfigUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    config = _get_or_create_config(db, current_user.id)
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
    if data.custom_ppe_items is not None:
        config.custom_ppe_items = json.dumps(data.custom_ppe_items)
    if data.no_phone_zone is not None:
        config.no_phone_zone = data.no_phone_zone
    db.commit()
    return {
        "message": "Config updated",
        "custom_ppe_items": _parse_custom_ppe(config),
        "no_phone_zone": bool(config.no_phone_zone),
    }


# ── FCM push token endpoints ──────────────────────────────────────────────────


class FcmTokenRequest(BaseModel):
    token: str
    device_name: Optional[str] = None


@router.post("/me/fcm-token")
def register_fcm_token(
    data: FcmTokenRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Register (or refresh) an FCM device token for the current user."""
    from services.fcm_service import register_token as _reg

    result = _reg(
        user_id=current_user.id,
        token=data.token,
        device_name=data.device_name,
        db=db,
    )
    return result


@router.delete("/me/fcm-token")
def unregister_fcm_token(
    data: FcmTokenRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Remove an FCM device token (logout / device swap)."""
    from services.fcm_service import delete_token as _del

    removed = _del(token=data.token, user_id=current_user.id, db=db)
    return {"removed": removed}


@router.get("/me/fcm-tokens")
def list_fcm_tokens(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all registered FCM tokens for the current user."""
    from database import FcmToken

    tokens = db.query(FcmToken).filter(FcmToken.user_id == current_user.id).all()
    return [
        {
            "id": t.id,
            "device_name": t.device_name,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in tokens
    ]


@router.post("/me/fcm-test")
def test_fcm_notification(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Send an immediate test push notification to all registered devices."""
    from services.fcm_service import send_fcm_alert
    success = send_fcm_alert(
        message="🔔 Test Alert from OccuSafe! Push notifications are working on this device.",
        severity="high",
        detected_issue="Test Notification",
        camera_name="Test Chamber",
        floor="Ground",
        db=db,
    )
    return {"success": success, "user_id": current_user.id}

