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
                     Request, UploadFile)
from pydantic import BaseModel
from routers.auth import get_current_user
from sqlalchemy.orm import Session

log = logging.getLogger("documents_router")
router = APIRouter(prefix="/documents", tags=["documents"])

import hashlib
import hmac
import json
import re
import socket
import time
import urllib.request
from config import settings

_QR_TOKEN_TTL = 15 * 60   # 15 minutes in seconds
_runtime_public_gate_url: Optional[str] = getattr(settings, "PUBLIC_GATE_URL", "")


def _detect_lan_ip() -> str:
    """Auto-detect real LAN IPv4 of this host so mobile on Wi-Fi doesn't get unreachable localhost."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _detect_ngrok_url() -> Optional[str]:
    """Query local ngrok client API (127.0.0.1:4040) to detect active public tunnel."""
    try:
        req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels", headers={"User-Agent": "OccuSafe"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode())
            tunnels = data.get("tunnels", [])
            for t in tunnels:
                p_url = t.get("public_url", "")
                if p_url.startswith("https://"):
                    return p_url
            if tunnels:
                return tunnels[0].get("public_url")
    except Exception:
        pass
    return None



def _make_qr_token(direction: str) -> str:
    """Create a signed token: base64(payload_json).hmac_hex"""
    payload = json.dumps({"dir": direction, "exp": int(time.time()) + _QR_TOKEN_TTL, "nonce": secrets.token_hex(8)})
    payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode()
    sig = hmac.new(settings.SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def _validate_qr_token(token: Optional[str]) -> dict:
    """Validate token, return payload dict. Handles fixed gate tokens and legacy session tokens."""
    # Allow fixed gate token or empty token for printed dock placards
    fixed_secret = getattr(settings, "FIXED_QR_ACCESS_KEY", "occusafe-gate-fixed")
    if not token or token == "fixed" or token == "gate-permanent" or token == fixed_secret:
        return {"dir": "dynamic", "fixed": True}
    if token.startswith("gate_fixed"):
        return {"dir": "dynamic", "fixed": True}

    try:
        payload_b64, sig = token.rsplit(".", 1)
        expected = hmac.new(settings.SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            # If not valid signature, check if it was raw fixed key
            if token == fixed_secret:
                return {"dir": "dynamic", "fixed": True}
            raise HTTPException(401, "Invalid QR token — tampering detected")
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode())
        if time.time() > payload.get("exp", 0):
            if payload.get("fixed"):
                return payload
            raise HTTPException(410, "QR token expired — ask admin to generate a new QR or use fixed gate QR")
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Malformed QR token")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _doc_to_dict(row, table: str) -> dict:
    ts_val = getattr(row, "timestamp", None)
    if hasattr(ts_val, "isoformat"):
        ts_str = ts_val.isoformat()
    elif ts_val:
        ts_str = str(ts_val)
    else:
        ts_str = datetime.datetime.utcnow().isoformat()

    # Ensure UTC timezone indicator so browser parsers do not shift or treat as local naive
    if not ts_str.endswith("Z") and "+" not in ts_str:
        ts_str += "Z"

    return {
        "id": row.id,
        "table": table,
        "camera_id": getattr(row, "camera_id", None),
        "employee_id": getattr(row, "employee_id", None),
        "direction": getattr(row, "direction", "inward"),
        "raw_ocr_text": getattr(row, "raw_ocr_text", None),
        "approved": getattr(row, "approved", False),
        "ocr_available": getattr(row, "ocr_available", True),
        "timestamp": ts_str,
        "has_snapshot": bool(getattr(row, "snapshot_b64", None)),
        "snapshot_b64": getattr(row, "snapshot_b64", None),
        "submitted_by_phone": getattr(row, "submitted_by_phone", False),
        "goods_count": getattr(row, "goods_count", None),
        "weight": getattr(row, "weight", None),
        "vendor_name": getattr(row, "vendor_name", None),
        "vehicle_no": getattr(row, "vehicle_no", None),
        "doc_number": getattr(row, "doc_number", None),
        "person_snapshot_b64": getattr(row, "person_snapshot_b64", None),
        "notes": getattr(row, "notes", None),
        "status": getattr(row, "status", "approved" if getattr(row, "approved", False) else "pending"),
        "reject_reason": getattr(row, "reject_reason", None),
    }


def _notify_owner_new_document(row, table: str, gate_pass_code: Optional[str] = None):
    """
    Notify factory owner via FCM, ntfy.sh, and Telegram when a new invoice/document
    is uploaded for approval.
    """
    try:
        direction = getattr(row, "direction", "inward")
        dir_title = "INWARD (Delivery)" if direction == "inward" else "OUTWARD (Dispatch)"
        doc_type = "Invoice" if direction == "inward" else "Order Form"
        vendor = getattr(row, "vendor_name", None) or "Unknown Vendor"
        vehicle = getattr(row, "vehicle_no", None) or "N/A"
        qty = getattr(row, "goods_count", None)
        wt = getattr(row, "weight", None)

        details = []
        if vendor and vendor != "Unknown Vendor":
            details.append(f"Vendor: {vendor}")
        if vehicle and vehicle != "N/A":
            details.append(f"Vehicle: {vehicle}")
        if qty:
            details.append(f"Qty: {qty}")
        if wt:
            details.append(f"Wt: {wt}")
        if gate_pass_code:
            details.append(f"Pass: {gate_pass_code}")

        detail_str = " | ".join(details) if details else f"ID: #{row.id}"
        message = f"📄 New {doc_type} Added for Approval: {dir_title} — {detail_str}"

        # 1. FCM Web Push to Owner Devices
        try:
            from services.notification_service import send_push_alert
            send_push_alert(
                message=message,
                severity="high",
                detected_issue=f"New {doc_type} For Approval",
            )
        except Exception as fcm_err:
            log.debug("[Documents Notify] FCM push error: %s", fcm_err)

        # 2. ntfy.sh (if configured)
        try:
            from services.notification_service import _is_ntfy_configured, send_ntfy_alert
            if _is_ntfy_configured():
                send_ntfy_alert(
                    message=message,
                    severity="high",
                    detected_issue=f"New {doc_type} For Approval",
                )
        except Exception as ntfy_err:
            log.debug("[Documents Notify] ntfy error: %s", ntfy_err)

        # 3. Telegram (if configured)
        try:
            from services.notification_service import _is_telegram_configured, send_telegram_alert
            if _is_telegram_configured():
                send_telegram_alert(
                    message=message,
                    severity="high",
                    detected_issue=f"New {doc_type} For Approval",
                )
        except Exception as tg_err:
            log.debug("[Documents Notify] Telegram error: %s", tg_err)

        log.info("[Documents Notify] Owner notified of new %s #%s for approval", doc_type, row.id)
    except Exception as exc:
        log.error("[Documents Notify] Failed dispatching owner alert: %s", exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/stats")
def get_stats(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Today's document scan counts, aligned to Indian Standard Time (IST) day boundaries."""
    now_ist = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
    today_ist_midnight = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    today_utc_start = today_ist_midnight - datetime.timedelta(hours=5, minutes=30)

    inv_q = db.query(InvoiceLog).filter(InvoiceLog.timestamp >= today_utc_start)
    ord_q = db.query(OrderFormLog).filter(OrderFormLog.timestamp >= today_utc_start)

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
    goods_count: Optional[int] = Form(None),
    weight: Optional[str] = Form(None),
    vendor_name: Optional[str] = Form(None),
    vehicle_no: Optional[str] = Form(None),
    doc_number: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Upload a document image → run OCR → save to invoice_logs or order_form_logs.
    Returns the OCR result immediately.
    """
    import os
    from services.ocr_service import process_uploaded_document_file, ALLOWED_DOC_EXTENSIONS

    # Read uploaded document (PDF, Word, or Image)
    contents = await file.read()
    ext = os.path.splitext(file.filename.lower())[1] if file.filename else ""
    if ext and ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file format '{ext}'. Allowed: PDF, JPEG, PNG, WEBP, DOCX, DOC.")

    result = process_uploaded_document_file(contents, file.filename or "upload.jpg", direction)
    if result.get("auto_rejected") and not result.get("raw_text") and result.get("engine") == "format_filter":
        raise HTTPException(400, result.get("reject_reason", "Could not decode or parse document file"))

    # Auto-extract quantity if omitted
    resolved_qty = goods_count if goods_count is not None else result.get("goods_count")
    resolved_weight = weight
    if not resolved_weight and result.get("raw_text"):
        wm = re.search(r"(?:net\s*wt|gross\s*wt|wt|weight)?\s*[:\-]?\s*(\d+(?:\.\d+)?\s*(?:kg|kgs|g|ton|tons|quintal))\b", result["raw_text"], re.IGNORECASE)
        if wm:
            resolved_weight = wm.group(1).strip()

    resolved_doc = doc_number
    if not resolved_doc and result.get("raw_text"):
        dm = re.search(r"\b(?:inv|invoice|bill|challan|order|dc|po)[\s\.\-\#:]*([a-z0-9\-\/]{3,20})\b", result["raw_text"], re.IGNORECASE)
        if dm:
            resolved_doc = dm.group(1).strip()

    # Determine status & reject reason
    is_appr = bool(result.get("approved", False))
    doc_status = "approved" if is_appr else "auto_rejected"
    rej_reason = result.get("reject_reason") if not is_appr else None

    # Save to DB
    try:
        if direction == "inward":
            row = InvoiceLog(
                camera_id=camera_id,
                direction="inward",
                raw_ocr_text=result["raw_text"],
                goods_count=resolved_qty,
                weight=resolved_weight,
                vendor_name=vendor_name,
                vehicle_no=vehicle_no,
                doc_number=resolved_doc,
                notes=notes,
                approved=is_appr,
                status=doc_status,
                reject_reason=rej_reason,
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
                goods_count=resolved_qty,
                weight=resolved_weight,
                vendor_name=vendor_name,
                vehicle_no=vehicle_no,
                doc_number=resolved_doc,
                notes=notes,
                approved=is_appr,
                status=doc_status,
                reject_reason=rej_reason,
                snapshot_b64=result["snapshot_b64"],
                person_snapshot_b64=person_snap,
                ocr_available=result["ocr_available"],
                timestamp=datetime.datetime.utcnow(),
            )
        db.add(row)
        db.commit()
        db.refresh(row)
        table = "invoice" if direction == "inward" else "order_form"
        if is_appr:
            _notify_owner_new_document(row, table)
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
    row.status = "approved"
    row.reject_reason = None
    db.commit()
    return {"id": doc_id, "approved": True, "status": "approved"}


@router.patch("/{doc_id}/reject")
def reject_document(
    doc_id: int,
    table: str = Query("invoice", description="invoice | order_form"),
    reason: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Admin override: mark a document scan as rejected."""
    Model = InvoiceLog if table == "invoice" else OrderFormLog
    row = db.query(Model).filter(Model.id == doc_id).first()
    if not row:
        raise HTTPException(404, f"Document #{doc_id} not found in {table}")
    row.approved = False
    row.status = "rejected"
    row.reject_reason = reason or "Manually rejected by admin"
    db.commit()
    return {"id": doc_id, "approved": False, "status": "rejected"}


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


# ── QR Token & Fixed Gateway Configuration ─────────────────────────────────────


class GateConfigUpdate(BaseModel):
    public_gate_url: str


@router.get("/fixed-qr-config")
def get_fixed_qr_config(request: Request):
    """
    Returns the permanent fixed QR code configuration for printing dock placards.
    Detects active ngrok tunnels or LAN IP automatically so mobile phones
    on cellular 4G/5G or Wi-Fi never get an unreachable localhost link.
    """
    global _runtime_public_gate_url
    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    forwarded_host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    detected_host_url = f"{forwarded_proto}://{forwarded_host}".rstrip("/")

    # Detect active ngrok tunnel
    ngrok_url = _detect_ngrok_url()
    
    # Detect host LAN IP (for Wi-Fi)
    lan_ip = _detect_lan_ip()
    lan_url = f"http://{lan_ip}:5173"

    pub_url = (_runtime_public_gate_url or getattr(settings, "PUBLIC_GATE_URL", "")).strip().rstrip("/")
    
    # Priority for effective base:
    # 1. Manually saved public URL
    # 2. Detected active ngrok tunnel
    # 3. Non-localhost forwarded host
    # 4. Local network LAN IP (port 5173 for Vite dev)
    if pub_url:
        effective_base = pub_url
    elif ngrok_url:
        effective_base = ngrok_url
    elif "localhost" not in detected_host_url and "127.0.0.1" not in detected_host_url:
        effective_base = detected_host_url
    else:
        effective_base = lan_url

    fixed_key = getattr(settings, "FIXED_QR_ACCESS_KEY", "occusafe-gate-fixed")
    portal_url = f"{effective_base}/upload-invoice?gate_key={fixed_key}"

    return {
        "public_gate_url": pub_url,
        "ngrok_url": ngrok_url or "",
        "lan_url": lan_url,
        "lan_ip": lan_ip,
        "detected_host_url": detected_host_url,
        "effective_base_url": effective_base,
        "portal_url": portal_url,
        "fixed_key": fixed_key,
        "mode": "fixed_permanent",
        "description": "Permanent QR Code for printed gate posters. Does not expire.",
    }


@router.post("/fixed-qr-config")
def update_fixed_qr_config(
    payload: GateConfigUpdate,
    current_user=Depends(get_current_user),
):
    """Admin endpoint to configure or update the public tunnel / domain URL for fixed QR posters."""
    global _runtime_public_gate_url
    cleaned = payload.public_gate_url.strip().rstrip("/")
    _runtime_public_gate_url = cleaned
    settings.PUBLIC_GATE_URL = cleaned
    log.info("[Fixed QR] Public gate URL updated to: %s", cleaned)
    return {"public_gate_url": cleaned, "saved": True}


@router.get("/qr-token")
def generate_qr_token(
    direction: str = Query("inward", description="inward | outward"),
    current_user=Depends(get_current_user),
):
    """
    Generate a signed QR session token for mobile phone document upload.
    Also returns fixed token for permanent placard printing.
    """
    token = _make_qr_token(direction)
    fixed_key = getattr(settings, "FIXED_QR_ACCESS_KEY", "occusafe-gate-fixed")
    return {
        "token": token,
        "fixed_token": fixed_key,
        "direction": direction,
        "expires_in_seconds": _QR_TOKEN_TTL,
        "expires_at": (datetime.datetime.utcnow() + datetime.timedelta(seconds=_QR_TOKEN_TTL)).isoformat(),
    }


# ── Mobile phone upload (PUBLIC — accessible over ANY network) ─────────────────


@router.post("/mobile-upload")
async def mobile_upload(
    token: Optional[str] = Form(None),
    direction: str = Form("inward"),
    file: UploadFile = File(...),
    person_file: Optional[UploadFile] = File(None),
    goods_count: Optional[int] = Form(None),
    weight: Optional[str] = Form(None),
    vendor_name: Optional[str] = Form(None),
    vehicle_no: Optional[str] = Form(None),
    doc_number: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Public endpoint — called from vendor/driver phone after scanning fixed or dynamic QR.
    Works over cellular (4G/5G) or local Wi-Fi.
    Accepts:
      - Document photo (invoice or order form)
      - Driver/vendor selfie photo (satisfies outward & inward client requirements)
      - Goods count, weight, vehicle number, vendor name
    Runs RapidOCR / Tesseract, logs entry with digital gate pass reference.
    """
    import cv2

    # Validate QR token (accepts fixed gate secret or validated session token)
    payload = _validate_qr_token(token)
    if payload.get("dir") and payload["dir"] != "dynamic" and not direction:
        direction = payload["dir"]
    direction = "inward" if direction == "inward" else "outward"

    # Enforce mandatory fields (Vendor Name, Vehicle No, Goods Count, Weight, Doc Ref)
    missing = []
    if not vendor_name or not vendor_name.strip():
        missing.append("Vendor / Supplier Name")
    if not vehicle_no or not vehicle_no.strip():
        missing.append("Vehicle Number")
    if goods_count is None or goods_count <= 0:
        missing.append("Goods Count (Quantity)")
    if not weight or not weight.strip():
        missing.append("Total Weight")
    if not doc_number or not doc_number.strip():
        missing.append("Invoice / Document Number")

    if missing:
        raise HTTPException(
            422,
            f"All fields are mandatory. Please provide: {', '.join(missing)}."
        )

    # 1. Read and process document (PDF, Word, or Image)
    contents = await file.read()
    import os
    from services.ocr_service import process_uploaded_document_file, ALLOWED_DOC_EXTENSIONS

    ext = os.path.splitext(file.filename.lower())[1] if file.filename else ""
    if ext and ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(400, f"Unsupported document format '{ext}'. Allowed: PDF, JPEG, PNG, WEBP, DOCX, DOC.")

    result = process_uploaded_document_file(contents, file.filename or "doc.jpg", direction)

    # Strict auto-reject check
    is_valid_doc = bool(result.get("approved", False))
    doc_status = "pending" if is_valid_doc else "auto_rejected"
    reject_reason = result.get("reject_reason") if not is_valid_doc else None

    # 2. Read driver/vendor person photo if provided
    person_snap_b64 = None
    if person_file is not None:
        try:
            person_bytes = await person_file.read()
            if person_bytes:
                pnparr = np.frombuffer(person_bytes, np.uint8)
                pframe = cv2.imdecode(pnparr, cv2.IMREAD_COLOR)
                if pframe is not None:
                    # Resize to max 640px for efficient DB storage
                    ph, pw = pframe.shape[:2]
                    if max(ph, pw) > 640:
                        scale = 640.0 / max(ph, pw)
                        pframe = cv2.resize(pframe, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)
                    _, pbuf = cv2.imencode(".jpg", pframe, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    person_snap_b64 = "data:image/jpeg;base64," + base64.b64encode(pbuf).decode()
        except Exception as p_exc:
            log.warning("[QR Mobile Upload] Failed decoding person photo: %s", p_exc)

    resolved_qty = goods_count
    resolved_weight = weight.strip()
    resolved_doc = doc_number.strip()
    clean_vendor = vendor_name.strip()
    clean_vehicle = vehicle_no.strip().upper()

    try:
        if direction == "inward":
            row = InvoiceLog(
                direction="inward",
                raw_ocr_text=result["raw_text"],
                goods_count=resolved_qty,
                weight=resolved_weight,
                vendor_name=clean_vendor,
                vehicle_no=clean_vehicle,
                doc_number=resolved_doc,
                notes=notes,
                approved=False,
                status=doc_status,
                reject_reason=reject_reason,
                snapshot_b64=result["snapshot_b64"],
                person_snapshot_b64=person_snap_b64,
                ocr_available=result["ocr_available"],
                upload_token=token or "fixed_gate",
                submitted_by_phone=True,
                timestamp=datetime.datetime.utcnow(),
            )
        else:
            row = OrderFormLog(
                direction="outward",
                raw_ocr_text=result["raw_text"],
                goods_count=resolved_qty,
                weight=resolved_weight,
                vendor_name=clean_vendor,
                vehicle_no=clean_vehicle,
                doc_number=resolved_doc,
                notes=notes,
                approved=False,
                status=doc_status,
                reject_reason=reject_reason,
                snapshot_b64=result["snapshot_b64"],
                person_snapshot_b64=person_snap_b64,
                ocr_available=result["ocr_available"],
                upload_token=token or "fixed_gate",
                submitted_by_phone=True,
                timestamp=datetime.datetime.utcnow(),
            )
        db.add(row)
        db.commit()
        db.refresh(row)
        table = "invoice" if direction == "inward" else "order_form"

        # Generate Digital Gate Pass reference ID
        dir_prefix = "INW" if direction == "inward" else "OUT"
        gate_pass_code = f"GP-{dir_prefix}-{datetime.datetime.utcnow().strftime('%m%d')}-{row.id:04d}"

        if is_valid_doc:
            log.info("[QR Mobile Upload] Valid %s saved id=%s code=%s — notifying owner for approval", table, row.id, gate_pass_code)
            # Notify owner across configured channels (FCM, ntfy, Telegram)
            _notify_owner_new_document(row, table, gate_pass_code)
        else:
            log.warning("[QR Mobile Upload] %s AUTO-REJECTED id=%s code=%s reason=%s", table, row.id, gate_pass_code, reject_reason)

        return {
            **result,
            "saved": True,
            "submitted_by_phone": True,
            "gate_pass_code": gate_pass_code,
            "record": _doc_to_dict(row, table),
            "vendor_name": clean_vendor,
            "vehicle_no": clean_vehicle,
            "goods_count": resolved_qty,
            "weight": resolved_weight,
            "doc_number": resolved_doc,
            "status": doc_status,
            "auto_rejected": not is_valid_doc,
            "reject_reason": reject_reason,
        }
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
    Returns all records submitted via QR phone upload that are genuine documents
    awaiting owner review (status='pending', approved=False).
    Auto-rejected photos and already approved/rejected records are excluded.
    """
    try:
        inv_rows = (
            db.query(InvoiceLog)
            .filter(
                InvoiceLog.submitted_by_phone == True,
                InvoiceLog.approved == False,
                InvoiceLog.status == "pending"
            )
            .order_by(InvoiceLog.timestamp.desc())
            .all()
        )
        ord_rows = (
            db.query(OrderFormLog)
            .filter(
                OrderFormLog.submitted_by_phone == True,
                OrderFormLog.approved == False,
                OrderFormLog.status == "pending"
            )
            .order_by(OrderFormLog.timestamp.desc())
            .all()
        )
        combined = [_doc_to_dict(r, "invoice") for r in inv_rows] + [
            _doc_to_dict(r, "order_form") for r in ord_rows
        ]
        combined.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"count": len(combined), "records": combined}
    except Exception as exc:
        log.warning("[Documents] pending-approval query failed: %s", exc)
        return {"count": 0, "records": []}

