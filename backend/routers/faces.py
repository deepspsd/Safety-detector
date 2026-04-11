"""
Face registration router — allows Home users to register known faces.
"""
import json
import os
import base64
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import get_db, User, FaceEncoding
from routers.auth import get_current_user
from services.face_service import encode_face_from_image
from config import settings

router = APIRouter(prefix="/faces", tags=["faces"])


class FaceRegisterRequest(BaseModel):
    label: str
    image_b64: str  # base64 image


@router.post("/register")
def register_face(
    data: FaceRegisterRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "Home":
        raise HTTPException(status_code=403, detail="Face registration is only for Home role")

    encoding = encode_face_from_image(data.image_b64)
    if encoding is None:
        raise HTTPException(status_code=400, detail="No face detected in the provided image")

    # Remove existing encoding with same label
    db.query(FaceEncoding).filter(
        FaceEncoding.user_id == current_user.id,
        FaceEncoding.label == data.label
    ).delete()

    face_record = FaceEncoding(
        user_id=current_user.id,
        label=data.label,
        encoding_data=json.dumps(encoding)
    )
    db.add(face_record)
    db.commit()
    return {"message": f"Face '{data.label}' registered successfully"}


@router.get("/")
def list_faces(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    faces = db.query(FaceEncoding).filter(FaceEncoding.user_id == current_user.id).all()
    return [{"id": f.id, "label": f.label, "created_at": f.created_at.isoformat()} for f in faces]


@router.delete("/{face_id}")
def delete_face(face_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    face = db.query(FaceEncoding).filter(
        FaceEncoding.id == face_id, FaceEncoding.user_id == current_user.id
    ).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")
    db.delete(face)
    db.commit()
    return {"message": "Face encoding deleted"}
