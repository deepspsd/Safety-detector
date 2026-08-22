"""
routers/documents.py — Invoice / Order-Form document scan API
=============================================================
Endpoints
---------
  GET    /documents/           Unified log (invoice + order forms, with direction filter)
  GET    /documents/stats      Approved/rejected counts by day
  POST   /documents/scan       Manual document image upload → OCR → save log
  PATCH  /documents/{id}/approve   Admin override: approve a rejected scan
  PATCH  /documents/{id}/reject    Admin override: reject an approved scan
"""

import base64
import datetime
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db, InvoiceLog, OrderFormLog
from routers.auth import get_current_user

log = logging.getLogger("documents_router")
router = APIRouter(prefix="/documents", tags=["documents"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_to_dict(row, table: str) -> dict:
    return {
        "id":            row.id,
        "table":         table,
        "camera_id":     row.camera_id,
        "employee_id":   row.employee_id,
        "direction":     row.direction,
        "raw_ocr_text":  row.raw_ocr_text,
        "approved":      row.approved,
        "ocr_available": row.ocr_available,
        "timestamp":     row.timestamp.isoformat(),
        "has_snapshot":  bool(row.snapshot_b64),
        "snapshot_b64":  row.snapshot_b64,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats(
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """Today's document scan counts."""
    today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    inv_q   = db.query(InvoiceLog).filter(InvoiceLog.timestamp >= today)
    ord_q   = db.query(OrderFormLog).filter(OrderFormLog.timestamp >= today)

    inv_rows = inv_q.all()
    ord_rows = ord_q.all()
    all_rows = list(inv_rows) + list(ord_rows)

    return {
        "today_total":    len(all_rows),
        "today_inward":   sum(1 for r in all_rows if r.direction == "inward"),
        "today_outward":  sum(1 for r in all_rows if r.direction == "outward"),
        "today_approved": sum(1 for r in all_rows if r.approved),
        "today_rejected": sum(1 for r in all_rows if not r.approved),
    }


@router.get("/")
def list_documents(
    direction:    Optional[str] = Query(None, description="inward | outward"),
    approved:     Optional[bool] = Query(None),
    limit:        int            = Query(100, le=500),
    db:           Session        = Depends(get_db),
    current_user                  = Depends(get_current_user),
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

    combined = (
        [_doc_to_dict(r, "invoice") for r in inv_rows] +
        [_doc_to_dict(r, "order_form") for r in ord_rows]
    )
    combined.sort(key=lambda x: x["timestamp"], reverse=True)
    return combined[:limit]


@router.post("/scan")
async def manual_scan(
    direction:    str          = Form("inward"),
    camera_id:    Optional[int] = Form(None),
    file:         UploadFile    = File(...),
    db:           Session       = Depends(get_db),
    current_user                = Depends(get_current_user),
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
                camera_id     = camera_id,
                direction     = "inward",
                raw_ocr_text  = result["raw_text"],
                goods_count   = result.get("goods_count"),
                approved      = result["approved"],
                snapshot_b64  = result["snapshot_b64"],
                ocr_available = result["ocr_available"],
                timestamp     = datetime.datetime.utcnow(),
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
                        import cv2, base64
                        _, buf = cv2.imencode(".jpg", live, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        person_snap = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
                except Exception as snap_exc:
                    log.debug("[Documents] person snapshot failed: %s", snap_exc)

            row = OrderFormLog(
                camera_id          = camera_id,
                direction          = "outward",
                raw_ocr_text       = result["raw_text"],
                approved           = result["approved"],
                snapshot_b64       = result["snapshot_b64"],
                person_snapshot_b64= person_snap,
                ocr_available      = result["ocr_available"],
                timestamp          = datetime.datetime.utcnow(),
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
    doc_id:       int,
    table:        str     = Query("invoice", description="invoice | order_form"),
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
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
    doc_id:       int,
    table:        str     = Query("invoice", description="invoice | order_form"),
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
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
    doc_id:       int,
    table:        str     = Query("invoice", description="invoice | order_form"),
    db:           Session = Depends(get_db),
    current_user           = Depends(get_current_user),
):
    """Permanently delete a document scan record."""
    Model = InvoiceLog if table == "invoice" else OrderFormLog
    row = db.query(Model).filter(Model.id == doc_id).first()
    if not row:
        raise HTTPException(404, f"Document #{doc_id} not found in {table}")
    db.delete(row)
    db.commit()
    return {"id": doc_id, "deleted": True}
