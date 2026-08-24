"""
Faces router — Multi-Person Face Registration & Management
• Any authenticated user can register faces (not just Home role)
• Saves thumbnail_b64 for UI preview
• Supports rename endpoint
"""

import json

from database import Employee, FaceEncoding, User, get_db
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from routers.auth import get_current_user
from services.face_service import encode_face_from_image
from sqlalchemy.orm import Session

router = APIRouter(prefix="/faces", tags=["faces"])


class FaceRegisterRequest(BaseModel):
    label: str
    image_b64: str  # base64 data-URL


class FaceRenameRequest(BaseModel):
    label: str


@router.post("/register")
def register_face(
    data: FaceRegisterRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not data.label.strip():
        raise HTTPException(status_code=400, detail="Label cannot be empty")

    result = encode_face_from_image(data.image_b64)
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please use a clear, front-facing photo with good lighting.",
        )

    # Try saving with thumbnail first; fall back without it if column doesn't exist yet
    try:
        face_record = FaceEncoding(
            user_id=current_user.id,
            label=data.label.strip(),
            encoding_data=json.dumps(result["encoding"]),
            thumbnail_b64=result.get("thumbnail_b64"),
        )
        db.add(face_record)
        db.commit()
        db.refresh(face_record)
    except Exception:
        db.rollback()
        # Fallback: save without thumbnail (column might not be migrated yet)
        face_record = FaceEncoding(
            user_id=current_user.id,
            label=data.label.strip(),
            encoding_data=json.dumps(result["encoding"]),
        )
        db.add(face_record)
        db.commit()
        db.refresh(face_record)

    # Upsert employee: reuse existing record if one with this label exists,
    # otherwise create fresh. Prevents duplicate inactive + active employee rows.
    employee = db.query(Employee).filter(Employee.name == face_record.label).first()
    if employee:
        employee.face_encoding_id = face_record.id
        employee.active = True
    else:
        employee = Employee(
            name=face_record.label,
            face_encoding_id=face_record.id,
            active=True,
        )
        db.add(employee)
    db.commit()
    db.refresh(employee)

    return {
        "message": f"Face '{face_record.label}' registered successfully",
        "id": face_record.id,
        "employee_id": employee.id,
        "label": face_record.label,
        "thumbnail_b64": getattr(face_record, "thumbnail_b64", None),
        "created_at": face_record.created_at.isoformat(),
    }


@router.get("/")
def list_faces(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    faces = (
        db.query(FaceEncoding)
        .filter(FaceEncoding.user_id == current_user.id)
        .order_by(FaceEncoding.created_at.desc())
        .all()
    )

    return [
        {
            "id": f.id,
            "label": f.label,
            "thumbnail_b64": f.thumbnail_b64,
            "created_at": f.created_at.isoformat(),
        }
        for f in faces
    ]


@router.put("/{face_id}/label")
def rename_face(
    face_id: int,
    data: FaceRenameRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    face = (
        db.query(FaceEncoding)
        .filter(
            FaceEncoding.id == face_id,
            FaceEncoding.user_id == current_user.id,
        )
        .first()
    )
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")
    if not data.label.strip():
        raise HTTPException(status_code=400, detail="Label cannot be empty")
    face.label = data.label.strip()
    for employee in face.employees:
        employee.name = face.label
    db.commit()
    return {"message": "Label updated", "id": face.id, "label": face.label}


@router.delete("/{face_id}")
def delete_face(
    face_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    face = (
        db.query(FaceEncoding)
        .filter(
            FaceEncoding.id == face_id,
            FaceEncoding.user_id == current_user.id,
        )
        .first()
    )
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")
    for employee in face.employees:
        db.delete(employee)
    db.delete(face)
    db.commit()
    return {"message": "Face deleted"}
