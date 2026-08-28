"""
routers/documents.py — Invoice / Order-Form document scan API
=============================================================
Endpoints
---------
  GET    /documents/           Unified log (invoice + order forms, with direction filter)
  GET    /documents/stats      Approved/rejected counts by day
  POST   /documents/scan       Manual document image upload → OCR → save log
  GET    /documents/qr-token   Generate a signed QR session token (admin only)
  POST   /documents/mobile-upload  Mobile phone upload — public, token-validated
  GET    /documents/pending-approval  Records awaiting admin approval from phone uploads
  PATCH  /documents/{id}/approve   Admin override: approve a rejected scan
  PATCH  /documents/{id}/reject    Admin override: reject an approved scan
"""

import base64
import datetime
import logging
import secrets
from typing import Optional

import numpy as np
from database import InvoiceLog, OrderFormLog, get_db
from fastapi import (APIRouter, Depends, File, Form, HTTPException, Query,
                     UploadFile)
from pydantic import BaseModel
from routers.auth import get_current_user
from sqlalchemy.orm import Session

log = logging.getLogger("documents_router")
router = APIRouter(prefix="/documents", tags=["documents"])

# ── QR Token helpers ──────────────────────────────────────────────────────────
import hashlib
import hmac
import json
import time
from config import settings

_QR_TOKEN_TTL = 15 * 60   # 15 minutes in seconds


def _make_qr_token(direction: str) -> str:
    """Create a signed token: base64(payload_json).hmac_hex"""
    payload = json.dumps({"dir": direction, "exp": int(time.time()) + _QR_TOKEN_TTL, "nonce": secrets.token_hex(8)})
    payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode()
    sig = hmac.new(settings.SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def _validate_qr_token(token: str) -> dict:
    """Validate token, return payload dict or raise HTTPException."""
    try:
        payload_b64, sig = token.rsplit(".", 1)
        expected = hmac.new(settings.SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(401, "Invalid QR token — tampering detected")
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode())
        if time.time() > payload["exp"]:
            raise HTTPException(410, "QR token expired — ask admin to generate a new QR")
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Malformed QR token")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _doc_to_dict(row, table: str) -> dict:
    return {
        "id": row.id,
        "table": table,
        "camera_id": row.camera_id,
        "employee_id": row.employee_id,
        "direction": row.direction,
        "raw_ocr_text": row.raw_ocr_text,
        "approved": row.approved,
        "ocr_available": row.ocr_available,
        "timestamp": row.timestamp.isoformat(),
        "has_snapshot": bool(row.snapshot_b64),
        "snapshot_b64": row.snapshot_b64,
        "submitted_by_phone": getattr(row, "submitted_by_phone", False),
        "goods_count": getattr(row, "goods_count", None),
        "person_snapshot_b64": getattr(row, "person_snapshot_b64", None),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/stats")
def get_stats(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Today's document scan counts."""
    today = datetime.datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    inv_q = db.query(InvoiceLog).filter(InvoiceLog.timestamp >= today)
    ord_q = db.query(OrderFormLog).filter(OrderFormLog.timestamp >= today)

    inv_rows = inv_q.all()
    ord_rows = ord_q.all()
    all_rows = list(inv_rows) + list(ord_rows)

    return {
        "today_total": len(all_rows),
        "today_inward": sum(1 for r in all_rows if r.direction == "inward"),
        "today_outward": sum(1 for r in all_rows if r.direction == "outward"),
        "today_approved": sum(1 for r in all_rows if r.approved),
        "today_rejected": sum(1 for r in all_rows if not r.approved),
    }


@router.get("/")
def list_documents(
    direction: Optional[str] = Query(None, description="inward | outward"),
    approved: Optional[bool] = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Unified document log (invoices + order forms), newest first."""
    # Query both tables and merge
    inv_q = db.query(InvoiceLog)
    ord_q = db.query(OrderFormLog)

    if direction:
        inv_q = inv_q.filter(InvoiceLog.direction == direction)
        ord_q = ord_q.filter(OrderFormLog.direction == direction)
    if approved is not None:
        inv_q = inv_q.filter(InvoiceLog.approved == approved)
        ord_q = ord_q.filter(OrderFormLog.approved == approved)

    inv_rows = inv_q.order_by(InvoiceLog.timestamp.desc()).limit(limit).all()
    ord_rows = ord_q.order_by(OrderFormLog.timestamp.desc()).limit(limit).all()

    combined = [_doc_to_dict(r, "invoice") for r in inv_rows] + [
        _doc_to_dict(r, "order_form") for r in ord_rows
    ]
    combined.sort(key=lambda x: x["timestamp"], reverse=True)
    return combined[:limit]


@router.post("/scan")
async def manual_scan(
    direction: str = Form("inward"),
    camera_id: Optional[int] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Upload a document image → run OCR → save to invoice_logs or order_form_logs.
    Returns the OCR result immediately.
    """
    import cv2

    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image — upload JPEG or PNG")

    # Use the full image as the bbox
    h, w = frame.shape[:2]
    bbox = [0, 0, w, h]

    from services.ocr_service import scan_document_in_frame

    result = scan_document_in_frame(frame, bbox, direction)

    # Save to DB
    try:
        if direction == "inward":
            row = InvoiceLog(
                camera_id=camera_id,
                direction="inward",
                raw_ocr_text=result["raw_text"],
                goods_count=result.get("goods_count"),
                approved=result["approved"],
                snapshot_b64=result["snapshot_b64"],
                ocr_available=result["ocr_available"],
                timestamp=datetime.datetime.utcnow(),
            )
        else:
            # Outward: also capture a face/person snapshot from the camera frame
            # if a live camera_id was supplied (dual-snapshot requirement REQ-039)
            person_snap = None
            if camera_id:
                try:
                    from services.camera_manager import get_latest_frame

                    live = get_latest_frame(camera_id)
                    if live is not None:
                        import base64

                        import cv2

                        _, buf = cv2.imencode(
                            ".jpg", live, [cv2.IMWRITE_JPEG_QUALITY, 75]
                        )
                        person_snap = (
                            "data:image/jpeg;base64," + base64.b64encode(buf).decode()
                        )
                except Exception as snap_exc:
                    log.debug("[Documents] person snapshot failed: %s", snap_exc)

            row = OrderFormLog(
                camera_id=camera_id,
                direction="outward",
                raw_ocr_text=result["raw_text"],
                approved=result["approved"],
                snapshot_b64=result["snapshot_b64"],
                person_snapshot_b64=person_snap,
                ocr_available=result["ocr_available"],
                timestamp=datetime.datetime.utcnow(),
            )
        db.add(row)
        db.commit()
        db.refresh(row)
        table = "invoice" if direction == "inward" else "order_form"
        return {**result, "saved": True, "record": _doc_to_dict(row, table)}
    except Exception as exc:
        log.error(f"[Documents] DB save failed: {exc}")
        return {**result, "saved": False, "error": str(exc)}


@router.patch("/{doc_id}/approve")
def approve_document(
    doc_id: int,
    table: str = Query("invoice", description="invoice | order_form"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Admin override: mark a document scan as approved."""
    Model = InvoiceLog if table == "invoice" else OrderFormLog
    row = db.query(Model).filter(Model.id == doc_id).first()
    if not row:
        raise HTTPException(404, f"Document #{doc_id} not found in {table}")
    row.approved = True
    db.commit()
    return {"id": doc_id, "approved": True}


@router.patch("/{doc_id}/reject")
def reject_document(
    doc_id: int,
    table: str = Query("invoice", description="invoice | order_form"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Admin override: mark a document scan as rejected."""
    Model = InvoiceLog if table == "invoice" else OrderFormLog
    row = db.query(Model).filter(Model.id == doc_id).first()
    if not row:
        raise HTTPException(404, f"Document #{doc_id} not found in {table}")
    row.approved = False
    db.commit()
    return {"id": doc_id, "approved": False}


@router.delete("/{doc_id}")
def delete_document(
    doc_id: int,
    table: str = Query("invoice", description="invoice | order_form"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Permanently delete a document scan record."""
    Model = InvoiceLog if table == "invoice" else OrderFormLog
    row = db.query(Model).filter(Model.id == doc_id).first()
    if not row:
        raise HTTPException(404, f"Document #{doc_id} not found in {table}")
    db.delete(row)
    db.commit()
    return {"id": doc_id, "deleted": True}


# ── QR Token generation ───────────────────────────────────────────────────────


@router.get("/qr-token")
def generate_qr_token(
    direction: str = Query("inward", description="inward | outward"),
    current_user=Depends(get_current_user),
):
    """
    Generate a signed QR session token for mobile phone invoice upload.
    Token is valid for 15 minutes. Embed in a QR code URL and display to worker.
    """
    token = _make_qr_token(direction)
    return {
        "token": token,
        "direction": direction,
        "expires_in_seconds": _QR_TOKEN_TTL,
        "expires_at": (datetime.datetime.utcnow() + datetime.timedelta(seconds=_QR_TOKEN_TTL)).isoformat(),
    }


# ── Mobile phone upload (PUBLIC — no auth, token-validated) ───────────────────


@router.post("/mobile-upload")
async def mobile_upload(
    token: str = Form(...),
    direction: str = Form("inward"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Public endpoint — called from worker's phone after scanning the QR code.
    Validates the signed token, runs OCR, saves to DB with submitted_by_phone=True.
    No JWT auth — access is controlled by the short-lived signed token.
    """
    import cv2

    # Validate QR token
    payload = _validate_qr_token(token)
    direction = payload.get("dir", direction)   # token direction overrides form

    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image — upload JPEG or PNG")

    h, w = frame.shape[:2]
    bbox = [0, 0, w, h]

    from services.ocr_service import scan_document_in_frame
    result = scan_document_in_frame(frame, bbox, direction)

    try:
        if direction == "inward":
            row = InvoiceLog(
                direction="inward",
                raw_ocr_text=result["raw_text"],
                goods_count=result.get("goods_count"),
                approved=result["approved"],
                snapshot_b64=result["snapshot_b64"],
                ocr_available=result["ocr_available"],
                upload_token=token,
                submitted_by_phone=True,
                timestamp=datetime.datetime.utcnow(),
            )
        else:
            row = OrderFormLog(
                direction="outward",
                raw_ocr_text=result["raw_text"],
                approved=result["approved"],
                snapshot_b64=result["snapshot_b64"],
                ocr_available=result["ocr_available"],
                upload_token=token,
                submitted_by_phone=True,
                timestamp=datetime.datetime.utcnow(),
            )
        db.add(row)
        db.commit()
        db.refresh(row)
        table = "invoice" if direction == "inward" else "order_form"
        log.info("[QR Mobile Upload] %s saved id=%s phone=True", table, row.id)
        return {**result, "saved": True, "submitted_by_phone": True}
    except Exception as exc:
        log.error("[QR Mobile Upload] DB save failed: %s", exc)
        return {**result, "saved": False, "error": str(exc)}


# ── Pending approval queue (phone uploads awaiting admin review) ───────────────


@router.get("/pending-approval")
def pending_approval(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Returns all records submitted via QR phone upload that have not been
    manually approved or rejected by an admin yet (approved=False, submitted_by_phone=True).
    Used to display the pending badge count and approval queue in the Documents page.
    """
    try:
        inv_rows = (
            db.query(InvoiceLog)
            .filter(InvoiceLog.submitted_by_phone == True, InvoiceLog.approved == False)
            .order_by(InvoiceLog.timestamp.desc())
            .all()
        )
        ord_rows = (
            db.query(OrderFormLog)
            .filter(OrderFormLog.submitted_by_phone == True, OrderFormLog.approved == False)
            .order_by(OrderFormLog.timestamp.desc())
            .all()
        )
        combined = [_doc_to_dict(r, "invoice") for r in inv_rows] + [
            _doc_to_dict(r, "order_form") for r in ord_rows
        ]
        combined.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"count": len(combined), "records": combined}
    except Exception as exc:
        log.warning("[Documents] pending-approval query failed (columns may not exist yet): %s", exc)
        return {"count": 0, "records": []}

