from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from auth_utils import (
    create_access_token,
    decode_token,
    get_password_hash,
    verify_password,
)
from database import User, UserConfig, get_db

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    role: str = "Construction Worker"  # Default: Manufacturing / Construction


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
    user = db.query(User).filter(User.email == payload.get("sub")).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )
    return user


@router.post("/signup", response_model=TokenResponse)
def signup(data: SignupRequest, db: Session = Depends(get_db)):
    try:
        existing = db.query(User).filter(User.email == data.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        if not data.email or "@" not in data.email:
            raise HTTPException(status_code=422, detail="Invalid email address")
        if len(data.password) < 6:
            raise HTTPException(
                status_code=422, detail="Password must be at least 6 characters"
            )

        valid_roles = {
            "Construction Worker",
            "Doctor",
            "Traffic Police",
            "College",
            "Home",
            "None",
            "Bakery Worker",
        }
        role = data.role if data.role in valid_roles else "Construction Worker"

        user = User(
            email=data.email,
            name=data.name or data.email.split("@")[0],
            hashed_password=get_password_hash(data.password),
            role=role,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        config = UserConfig(user_id=user.id)
        db.add(config)
        db.commit()
        token = create_access_token({"sub": user.email})
        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


@router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter(User.email == form_data.username).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Invalid credentials")
        token = create_access_token({"sub": user.email})
        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "role": current_user.role,
        "created_at": current_user.created_at.isoformat(),
    }
