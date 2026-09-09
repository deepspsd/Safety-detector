"""
ocr_service.py — Document scan at factory entrance gate
=========================================================
Triggered by yolo_service when class 13 "Document-in-hand" is detected
inside the "entrance" zone polygon (see bakery_cv_plan.md §9.5).

Engine choice: pytesseract (Tesseract OCR)
  • Preferred over EasyOCR on CPU-only servers: lower RAM, no neural-network
    load time, runs synchronously without GPU.
  • Requires system installation: https://github.com/UB-Mannheim/tesseract/wiki
    (Windows: winget install UB-Mannheim.TesseractOCR)
    (Linux:   sudo apt install tesseract-ocr)
  • Falls back gracefully if Tesseract is not installed: approved=False,
    raw_text="[OCR_UNAVAILABLE]", so the pipeline logs a REVIEW alert rather
    than crashing.

⚠️  IMPORTANT — hardware limitation disclaimer:
    "approved=False" does NOT physically block a door or turnstile.
    There is no hardware actuator integrated here.  The result of this
    function is a LOGGED COMPLIANCE CHECK + optional ALERT sent to an admin.
    A human must review the alert and take any physical action.
    Do not assume or imply hardware integration.

Approval heuristics (deliberately lenient v1 — see inline notes):
  • "inward"  → invoice-like: text must contain ≥1 digit string matching
                common invoice patterns (numbers, dates, INR amounts).
  • "outward" → order-form-like: same heuristic but description in alert
                differs.
  Both heuristics log raw_text unconditionally so admins can review
  edge cases where the OCR misfires.  Do not tighten without more data.
"""

import base64
import datetime
import io
import logging
import os
import re
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

log = logging.getLogger("ocr_service")

# ──────────────────────────────────────────────────────────────────────────────
# Multi-Tier OCR Engines: RapidOCR (Primary) → PaddleOCR (Fallback 1) → Tesseract (Fallback 2)
# ──────────────────────────────────────────────────────────────────────────────
_RAPID_OCR_INSTANCE = None
_PADDLE_OCR_INSTANCE = None

try:
    from rapidocr_onnxruntime import RapidOCR
    _RAPID_OCR_OK = True
    log.info("✅ RapidOCR (PaddleOCR ONNX engine) available — PRIMARY")
except ImportError:
    _RAPID_OCR_OK = False
    log.info("ℹ️  rapidocr_onnxruntime not installed.")

try:
    from paddleocr import PaddleOCR
    _PADDLE_OCR_OK = True
    log.info("✅ PaddleOCR engine available — FALLBACK")
except ImportError:
    _PADDLE_OCR_OK = False

# Tesseract availability guard (Legacy fallback)
try:
    import os
    import pytesseract

    _WIN_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    _WIN_DEFAULT_X86 = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    if os.name == "nt":
        if os.path.isfile(_WIN_DEFAULT):
            pytesseract.pytesseract.tesseract_cmd = _WIN_DEFAULT
            log.info(f"✅ pytesseract: using Tesseract at {_WIN_DEFAULT}")
        elif os.path.isfile(_WIN_DEFAULT_X86):
            pytesseract.pytesseract.tesseract_cmd = _WIN_DEFAULT_X86
            log.info(f"✅ pytesseract: using Tesseract at {_WIN_DEFAULT_X86}")
        else:
            import shutil

            tess = shutil.which("tesseract")
            if tess:
                pytesseract.pytesseract.tesseract_cmd = tess
                log.info(f"✅ pytesseract: found Tesseract in PATH at {tess}")

    _TESSERACT_OK = True
    log.info("✅ pytesseract available — Tesseract OCR available as fallback")
except (ImportError, Exception):
    _TESSERACT_OK = False
    log.info("ℹ️  pytesseract not available.")

# ──────────────────────────────────────────────────────────────────────────────
# Approval patterns
# ──────────────────────────────────────────────────────────────────────────────

# Inward (invoice): at least one sequence of ≥3 digits OR a date-like string
# e.g. "INV-20240801", "123456", "01/08/2024", "2024-08-01", "₹1,200"
_INVOICE_PATTERNS = [
    re.compile(r"\b\d{3,}\b"),  # ≥3 consecutive digits
    re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"),  # date: 01/08/2024 or 1-8-24
    re.compile(r"(?:INV|GST|PO|ORD|REF)[\/\-#]?\w+", re.IGNORECASE),  # common prefixes
    re.compile(r"[₹\$]\s*[\d,]+"),  # currency amount
]

# Outward (order form): same pragmatic approach — order refs look like invoices
# A stricter check (e.g. "ORDER" keyword) risks false-rejects on handwritten forms.
_ORDER_PATTERNS = (
    _INVOICE_PATTERNS  # identical v1; differentiated in v2 with real samples
)


# Commercial & invoice keywords (lower-case tokens)
COMMERCIAL_DOCUMENT_KEYWORDS = {
    # Document types & titles
    "invoice", "tax invoice", "retail invoice", "bill", "bills", "challan",
    "delivery", "dispatch", "gate pass", "gatepass", "purchase order", "po",
    "order", "order form", "work order", "consignment", "bilty", "lr no", "receipt",
    "memo", "slip", "proforma", "quotation", "statement", "vessel", "unpaid", "paid",
    # Parties & transport
    "vendor", "supplier", "consignor", "consignee", "buyer", "customer",
    "billed to", "shipped to", "ship to", "bill to", "sold to", "client",
    "transporter", "transport", "vehicle", "truck", "driver", "carrier",
    # Line items, units & quantities
    "qty", "quantity", "pieces", "pcs", "nos", "units", "box", "boxes",
    "bags", "packs", "cartons", "ctn", "kg", "kgs", "weight", "gross", "net wt", "tare",
    "rate", "price", "amount", "total", "subtotal", "sub total", "grand total",
    "taxable", "mrp", "discount", "disc", "cgst", "sgst", "igst", "gst",
    "hsn", "sac", "item", "description", "particulars",
    # Banking & authorization
    "signature", "sign", "authorised", "authorized", "bank", "account",
    "a/c", "ifsc", "rupees", "balance", "payment", "date"
}

# Appliance / remote control keywords that identify a non-document object
APPLIANCE_OBJECT_KEYWORDS = {
    "temp", "fan speed", "swing", "turbo", "sleep mode", "eco mode",
    "cool mode", "heat mode", "dry mode", "on/off", "air conditioner",
    "voltas", "daikin", "remote controller", "set temp", "room temp",
    "power", "celsius"
}

_GSTIN_REGEX = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b", re.IGNORECASE)
_DOC_NUM_REGEX = re.compile(r"\b(?:inv|invoice|bill|challan|ch|po|ord|ref|memo|tax|dc|lr)[\s\.\-\#:\/]*[a-z0-9\-\/]{3,20}\b", re.IGNORECASE)
_CURRENCY_AMOUNT_REGEX = re.compile(r"(?:total|amount|amt|inr|rs\.?|₹)[\s\.\:\-]*[\d,]+(?:\.\d{2})?", re.IGNORECASE)


def is_valid_invoice_document(text: str, direction: str = "inward") -> Tuple[bool, str]:
    """
    Validate whether OCR text represents a genuine commercial invoice or delivery document.
    Auto-rejects non-document objects (such as AC remotes, appliances, blank photos).
    """
    if not text:
        return False, "No text detected in image (blank, blurred, or non-document photo)"

    clean = text.strip()
    clean_lower = clean.lower()

    # 1. Immediate rejection on appliance/remote keywords
    appliance_matches = [k for k in APPLIANCE_OBJECT_KEYWORDS if k in clean_lower]
    if appliance_matches and not _GSTIN_REGEX.search(clean):
        return False, f"Non-document photo detected (appliance/remote controls found: {', '.join(appliance_matches)})"

    # 2. Minimum length check: A genuine commercial document has substantial text
    if len(clean) < 25:
        return False, f"Insufficient text for a commercial document ({len(clean)} characters; minimum 25 required)"

    # 3. Minimum word count
    words = re.findall(r"\b[A-Za-z]{2,}\b", clean)
    if len(words) < 3:
        return False, f"Too few recognizable words ({len(words)} words) to be an invoice or delivery slip"

    # 4. Search for high-confidence document indicators
    has_gstin = bool(_GSTIN_REGEX.search(clean))
    has_doc_num = bool(_DOC_NUM_REGEX.search(clean))
    has_currency_amount = bool(_CURRENCY_AMOUNT_REGEX.search(clean))

    matched_kws = [kw for kw in COMMERCIAL_DOCUMENT_KEYWORDS if kw in clean_lower]

    if has_gstin:
        return True, f"Verified invoice (GSTIN verified, {len(matched_kws)} keywords matched)"

    if has_doc_num and len(matched_kws) >= 1:
        return True, f"Verified document (Doc ref + keywords: {', '.join(matched_kws[:3])})"

    if has_currency_amount and len(matched_kws) >= 1:
        return True, f"Verified invoice (Total amount + keywords: {', '.join(matched_kws[:3])})"

    if len(matched_kws) >= 2:
        return True, f"Verified document ({len(matched_kws)} commercial keywords matched: {', '.join(matched_kws[:4])})"

    return False, "Not recognized as an invoice or order document. Missing invoice keywords, document numbers, or amounts."


def _text_looks_like_document(text: str, direction: str) -> bool:
    """Backward compatibility wrapper."""
    valid, _ = is_valid_invoice_document(text, direction)
    return valid


# Regex patterns to extract goods quantity from OCR text.
# Matches common invoice formats:
#   "Qty: 48", "Quantity 24", "48 bags", "48 packs", "48 nos", "48 units", "48 pcs"
_QTY_PATTERNS = [
    re.compile(
        r"(?:qty|quantity|nos|pcs|packs|bags|units)\s*[:\-]?\s*(\d+)", re.IGNORECASE
    ),
    re.compile(r"(\d+)\s*(?:qty|nos|pcs|packs|bags|units|pieces)", re.IGNORECASE),
    re.compile(r"\bqty\s+(\d+)\b", re.IGNORECASE),
]


def _extract_goods_count(text: str) -> Optional[int]:
    """
    Parse a numeric goods count from invoice OCR text.
    Returns None if no plausible quantity is found.
    Tries patterns in priority order; returns the first match.
    """
    for pattern in _QTY_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                val = int(m.group(1))
                if 1 <= val <= 100_000:  # sanity range — avoid OCR noise
                    return val
            except (IndexError, ValueError):
                continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Image pre-processing
# ──────────────────────────────────────────────────────────────────────────────


def _looks_like_camera_photo(img: np.ndarray) -> bool:
    """
    Heuristic: camera photos have non-uniform variance across regions
    (shadows, background, glare). Compute local variance in 4 quadrants;
    if the ratio of max/min variance is large, it's a camera photo.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    quads = [
        gray[:mid_h, :mid_w],
        gray[:mid_h, mid_w:],
        gray[mid_h:, :mid_w],
        gray[mid_h:, mid_w:],
    ]
    variances = [float(np.var(q)) for q in quads if q.size > 0]
    if not variances or min(variances) < 1:
        return True  # near-zero variance in a quadrant = probably uneven photo
    return (max(variances) / min(variances)) > 4.0


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp-mask sharpening — makes thin strokes crisper for Tesseract."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def _remove_noise_dots(binary: np.ndarray) -> np.ndarray:
    """
    Morphological open (erode→dilate) removes isolated 1–2px speckles that
    Tesseract misreads as random letters ('l', 'i', '.', etc.).
    Uses a small 2×2 kernel so actual text strokes are preserved.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _preprocess_for_scan(gray: np.ndarray) -> np.ndarray:
    """For clean uploaded scans/PDFs: sharpen → Otsu global thresholding."""
    sharpened = _sharpen(gray)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _remove_noise_dots(binary)


def _preprocess_for_camera(gray: np.ndarray) -> np.ndarray:
    """
    For camera photos: denoise → sharpen → CLAHE → adaptive threshold → denoise dots.
    Handles shadows, glare, and uneven lighting from handheld documents.
    """
    denoised = cv2.fastNlMeansDenoising(gray, h=12)
    sharpened = _sharpen(denoised)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=12,
    )
    return _remove_noise_dots(binary)


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Route to the correct preprocessor based on image characteristics.
    Upscales to ~300 DPI equivalent for Tesseract — below 150px wide is unusable.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape[:2]

    # Target ~300 DPI minimum — Tesseract accuracy degrades sharply below this
    target_width = 1200
    if w < target_width:
        scale = target_width / max(w, 1)
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if _looks_like_camera_photo(gray):
        return _preprocess_for_camera(gray)
    return _preprocess_for_scan(gray)


def _clean_ocr_text(text: str) -> str:
    """
    Post-process raw Tesseract output to remove common noise artifacts:
    - Isolated single characters on their own line (common Tesseract artefact)
    - Lines that are pure punctuation / whitespace
    - Duplicate whitespace
    This preserves all real invoice content (numbers, words, dates).
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Drop lines that are only 1 char OR pure punctuation/symbols
        if len(stripped) <= 1:
            continue
        if re.match(r"^[^a-zA-Z0-9₹\$\.,%/\-]+$", stripped):
            continue
        cleaned.append(stripped)
    return " | ".join(cleaned) if cleaned else text.strip()


def _crop_with_padding(
    frame: np.ndarray, bbox: list, pad_frac: float = 0.10
) -> np.ndarray:
    """
    Crop bbox from frame with 10% padding on each side.
    Matches the approach in bakery_cv_plan.md §9.5.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    pad_x = int((x2 - x1) * pad_frac)
    pad_y = int((y2 - y1) * pad_frac)
    x1c = max(0, x1 - pad_x)
    y1c = max(0, y1 - pad_y)
    x2c = min(w, x2 + pad_x)
    y2c = min(h, y2 + pad_y)
    return frame[y1c:y2c, x1c:x2c]


def _encode_crop(crop: np.ndarray) -> Optional[str]:
    """Encode a BGR numpy crop as data-URI base64 JPEG (for DB + WebSocket)."""
    try:
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Tier Engine Runners
# ──────────────────────────────────────────────────────────────────────────────


def _run_rapid_ocr(crop: np.ndarray) -> Optional[Tuple[str, float]]:
    """
    Run RapidOCR (PP-OCRv4 ONNX) — Primary Engine.
    Ultra-lightweight (~150MB RAM), fast C++ ONNX inference (~80ms), handles
    skewed, angled, and distorted camera invoice captures without random characters.
    """
    global _RAPID_OCR_INSTANCE
    if not _RAPID_OCR_OK:
        return None
    try:
        if _RAPID_OCR_INSTANCE is None:
            from rapidocr_onnxruntime import RapidOCR

            _RAPID_OCR_INSTANCE = RapidOCR()

        results, _ = _RAPID_OCR_INSTANCE(crop)
        if not results:
            return ("", 0.0)

        lines = []
        confs = []
        for item in results:
            # item format: [box_coordinates, text, confidence]
            if len(item) >= 3:
                txt = str(item[1]).strip()
                if txt:
                    lines.append(txt)
                    try:
                        confs.append(float(item[2]))
                    except (ValueError, TypeError):
                        pass

        full_text = " | ".join(lines) if lines else ""
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return (full_text, round(avg_conf, 3))
    except Exception as e:
        log.error(f"[OCR] RapidOCR error: {e}")
        return None


def _run_paddle_ocr(crop: np.ndarray) -> Optional[Tuple[str, float]]:
    """
    Run native PaddleOCR — Secondary Fallback Engine.
    Runs if rapidocr fails or is unavailable.
    """
    global _PADDLE_OCR_INSTANCE
    if not _PADDLE_OCR_OK:
        return None
    try:
        if _PADDLE_OCR_INSTANCE is None:
            from paddleocr import PaddleOCR

            _PADDLE_OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        results = _PADDLE_OCR_INSTANCE.ocr(crop, cls=True)
        if not results or not results[0]:
            return ("", 0.0)

        lines = []
        confs = []
        for line in results[0]:
            if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                txt = str(line[1][0]).strip()
                if txt:
                    lines.append(txt)
                    try:
                        confs.append(float(line[1][1]))
                    except (ValueError, TypeError):
                        pass

        full_text = " | ".join(lines) if lines else ""
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return (full_text, round(avg_conf, 3))
    except Exception as e:
        log.error(f"[OCR] PaddleOCR error: {e}")
        return None


def _run_tesseract(crop: np.ndarray) -> Optional[Tuple[str, float]]:
    """
    Run Tesseract OCR — Tertiary Fallback Engine (Legacy).
    Applies image preprocessing and attempts multiple PSM modes.
    """
    if not _TESSERACT_OK:
        return None

    _PSM_MODES = [
        "--psm 6 --oem 3",
        "--psm 11 --oem 3",
        "--psm 3 --oem 3",
    ]
    try:
        processed = _preprocess_for_ocr(crop)
        best_text = ""
        for psm_config in _PSM_MODES:
            try:
                candidate = pytesseract.image_to_string(
                    processed, config=psm_config, lang="eng"
                ).strip()
                candidate = _clean_ocr_text(candidate)
                if len(candidate) > len(best_text):
                    best_text = candidate
                    if len(best_text) > 60:
                        break
            except Exception:
                continue
        return (best_text, 0.70)
    except Exception as exc:
        log.error(f"[OCR] Tesseract error: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def scan_document_in_frame(
    frame: np.ndarray,
    bbox: list,
    direction: str,  # "inward" | "outward"
) -> dict:
    """
    Main entry point — called by yolo_service when 'Document-in-hand' (class 13)
    is detected inside the 'entrance' zone polygon.

    Uses a tiered architecture:
      1. RapidOCR (PP-OCRv4 ONNX) — Primary (low memory, high speed, high accuracy)
      2. PaddleOCR               — Fallback 1
      3. Tesseract               — Fallback 2 (Legacy)
      4. Graceful Degrade        — If no OCR engine available

    Parameters
    ----------
    frame     : BGR numpy array (the full camera frame)
    bbox      : [x1, y1, x2, y2] bounding box of the detected document
    direction : "inward" (goods coming in) | "outward" (goods going out)

    Returns
    -------
    dict with keys:
        approved       : bool — True if document text passes heuristic
        raw_text       : str  — always populated (for admin manual review)
        goods_count    : int | None — parsed items quantity from invoice
        confidence     : float — OCR recognition confidence
        engine         : str — "rapidocr" | "paddleocr" | "tesseract" | "none"
        timestamp      : str  — ISO-8601 UTC
        direction      : str  — echoed back
        snapshot_b64   : str | None — base64 JPEG of the cropped document area
        ocr_available  : bool — False if no OCR engine installed
    """
    ts = datetime.datetime.utcnow().isoformat() + "Z"

    # Step 1 — crop with boundary padding
    crop = _crop_with_padding(frame, bbox)
    snapshot_b64 = _encode_crop(crop)

    active_engine = None
    raw_text = ""
    conf = 0.0

    # Tier 1 — RapidOCR (PaddleOCR ONNX, Primary)
    if _RAPID_OCR_OK:
        rapid_res = _run_rapid_ocr(crop)
        if rapid_res and rapid_res[0]:
            raw_text, conf = rapid_res
            active_engine = "rapidocr"
            log.info(
                f"[OCR:RapidOCR] direction={direction} | conf={conf:.2f} | "
                f"text_len={len(raw_text)} | preview={raw_text[:80]!r}"
            )

    # Tier 2 — PaddleOCR (Fallback 1)
    if not raw_text and _PADDLE_OCR_OK:
        paddle_res = _run_paddle_ocr(crop)
        if paddle_res and paddle_res[0]:
            raw_text, conf = paddle_res
            active_engine = "paddleocr"
            log.info(
                f"[OCR:PaddleOCR] direction={direction} | conf={conf:.2f} | "
                f"text_len={len(raw_text)} | preview={raw_text[:80]!r}"
            )

    # Tier 3 — Tesseract (Fallback 2, Legacy)
    if not raw_text and _TESSERACT_OK:
        tess_res = _run_tesseract(crop)
        if tess_res and tess_res[0]:
            raw_text, conf = tess_res
            active_engine = "tesseract"
            log.info(
                f"[OCR:Tesseract] direction={direction} | text_len={len(raw_text)} | "
                f"preview={raw_text[:80]!r}"
            )

    # If no OCR engine produced text or available
    if not active_engine:
        if not _RAPID_OCR_OK and not _PADDLE_OCR_OK and not _TESSERACT_OK:
            log.warning("OCR called but no OCR engine installed. Returning REVIEW alert.")
            return {
                "approved": False,
                "raw_text": "[OCR_UNAVAILABLE — install rapidocr_onnxruntime]",
                "goods_count": None,
                "confidence": 0.0,
                "engine": None,
                "timestamp": ts,
                "direction": direction,
                "snapshot_b64": snapshot_b64,
                "ocr_available": False,
            }
        active_engine = "none"

    # Step 4 — strict commercial document verification (rejects AC remotes, non-documents, random objects)
    approved, validation_reason = is_valid_invoice_document(raw_text, direction)

    if not approved:
        log.warning(
            f"[OCR:Auto-Reject] Document rejected | reason={validation_reason} | engine={active_engine} | direction={direction} | "
            f"raw_text={raw_text[:120]!r}"
        )
    else:
        log.info(f"[OCR:Approved] Valid document detected | reason={validation_reason} | engine={active_engine} | direction={direction}")

    return {
        "approved": approved,
        "auto_rejected": not approved,
        "status": "approved" if approved else "auto_rejected",
        "reject_reason": None if approved else validation_reason,
        "validation_reason": validation_reason,
        "raw_text": raw_text,
        "goods_count": _extract_goods_count(raw_text),
        "confidence": conf,
        "engine": active_engine,
        "timestamp": ts,
        "direction": direction,
        "snapshot_b64": snapshot_b64,
        "ocr_available": True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Universal Document Processor (Images, PDFs, Word Docs)
# ──────────────────────────────────────────────────────────────────────────────

ALLOWED_DOC_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".doc", ".docx"}


def _create_text_document_preview(title: str, text_lines: list, filename: str) -> str:
    """Generates an 800x1000 rendered preview image for text/word documents."""
    preview_img = np.full((1000, 800, 3), 250, dtype=np.uint8)
    cv2.rectangle(preview_img, (0, 0), (800, 75), (30, 41, 59), -1)
    cv2.putText(preview_img, f"DOC: {filename[:38]}", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    y = 125
    for line in text_lines[:25]:
        clean = line.strip().replace('\t', ' ')
        if clean:
            cv2.putText(preview_img, clean[:65], (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 41, 59), 1)
            y += 32
            if y > 950:
                break
    _, buf = cv2.imencode(".jpg", preview_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def process_uploaded_document_file(
    file_bytes: bytes,
    filename: str,
    direction: str = "inward",
) -> dict:
    """
    Processes an uploaded document file of type PDF, Word (.docx/.doc), or Image (.jpg/.png/.webp).
    Executes the exact same OCR flow, strict invoice heuristic validation, snapshot creation,
    and auto-rejection rules.
    """
    ext = os.path.splitext(filename.lower())[1]
    ts = datetime.datetime.utcnow().isoformat() + "Z"

    if ext not in ALLOWED_DOC_EXTENSIONS:
        return {
            "approved": False,
            "auto_rejected": True,
            "status": "auto_rejected",
            "reject_reason": f"Unsupported format '{ext}'. Allowed: PDF, JPG, PNG, WEBP, DOCX, DOC.",
            "validation_reason": f"Unsupported format '{ext}'",
            "raw_text": "",
            "goods_count": None,
            "confidence": 0.0,
            "engine": "format_filter",
            "timestamp": ts,
            "direction": direction,
            "snapshot_b64": None,
            "ocr_available": True,
        }

    # ── CASE 1: PDF Document ──────────────────────────────────────────────────
    if ext == ".pdf":
        digital_text = ""
        # 1. Extract digital text with pypdf if available
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages[:3]:
                txt = page.extract_text() or ""
                if txt.strip():
                    digital_text += "\n" + txt.strip()
        except Exception as pdf_read_err:
            log.debug("[PDF Text Extract] Notice: %s", pdf_read_err)

        # 2. Render Page 1 to BGR image with pypdfium2 for optical OCR
        page_bgr = None
        try:
            import pypdfium2 as pdfium
            pdf_doc = pdfium.PdfDocument(io.BytesIO(file_bytes))
            if len(pdf_doc) > 0:
                page0 = pdf_doc[0]
                bitmap = page0.render(scale=2.0)  # 2x scale for sharp text recognition
                pil_img = bitmap.to_pil()
                page_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as pdf_render_err:
            log.warning("[PDF Render] pypdfium2 failed: %s", pdf_render_err)

        if page_bgr is not None:
            # Run optical OCR through standard pipeline
            h, w = page_bgr.shape[:2]
            res = scan_document_in_frame(page_bgr, [0, 0, w, h], direction)
            # Merge digital text with OCR text for maximum accuracy
            combined_text = (res["raw_text"] + "\n" + digital_text).strip()
            # Validate combined text with commercial invoice rules
            appr, reason = is_valid_invoice_document(combined_text, direction)
            res["raw_text"] = combined_text
            res["goods_count"] = _extract_goods_count(combined_text) or res["goods_count"]
            res["approved"] = appr
            res["auto_rejected"] = not appr
            res["status"] = "approved" if appr else "auto_rejected"
            res["reject_reason"] = None if appr else reason
            res["validation_reason"] = reason
            return res

        # Fallback if rendering failed but digital text exists
        if digital_text.strip():
            appr, reason = is_valid_invoice_document(digital_text, direction)
            snap_b64 = _create_text_document_preview("PDF Document", digital_text.splitlines(), filename)
            return {
                "approved": appr,
                "auto_rejected": not appr,
                "status": "approved" if appr else "auto_rejected",
                "reject_reason": None if appr else reason,
                "validation_reason": reason,
                "raw_text": digital_text.strip(),
                "goods_count": _extract_goods_count(digital_text),
                "confidence": 0.95,
                "engine": "pypdf_digital",
                "timestamp": ts,
                "direction": direction,
                "snapshot_b64": snap_b64,
                "ocr_available": True,
            }

        return {
            "approved": False,
            "auto_rejected": True,
            "status": "auto_rejected",
            "reject_reason": "Could not read text or render pages from uploaded PDF.",
            "validation_reason": "Unreadable PDF document",
            "raw_text": "",
            "goods_count": None,
            "confidence": 0.0,
            "engine": "pdf_reader",
            "timestamp": ts,
            "direction": direction,
            "snapshot_b64": None,
            "ocr_available": True,
        }

    # ── CASE 2: Word Document (.docx / .doc) ──────────────────────────────────
    if ext in {".docx", ".doc"}:
        extracted_lines = []
        full_text = ""
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            for p in doc.paragraphs:
                p_txt = p.text.strip()
                if p_txt:
                    extracted_lines.append(p_txt)
            for table in doc.tables:
                for row in table.rows:
                    cells_txt = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if cells_txt:
                        extracted_lines.append(cells_txt)
            full_text = "\n".join(extracted_lines).strip()
        except Exception as docx_err:
            log.warning("[DOCX Parse] Standard docx parse failed: %s", docx_err)
            # Binary string extraction fallback for legacy .doc
            try:
                raw_matches = re.findall(r"[\x20-\x7E]{4,}", file_bytes.decode("latin-1", errors="ignore"))
                full_text = "\n".join(m.strip() for m in raw_matches if len(m.strip()) > 3)
                extracted_lines = full_text.splitlines()[:50]
            except Exception:
                pass

        if full_text:
            appr, reason = is_valid_invoice_document(full_text, direction)
            snap_b64 = _create_text_document_preview("Word Document", extracted_lines, filename)
            return {
                "approved": appr,
                "auto_rejected": not appr,
                "status": "approved" if appr else "auto_rejected",
                "reject_reason": None if appr else reason,
                "validation_reason": reason,
                "raw_text": full_text,
                "goods_count": _extract_goods_count(full_text),
                "confidence": 0.95,
                "engine": "docx_parser",
                "timestamp": ts,
                "direction": direction,
                "snapshot_b64": snap_b64,
                "ocr_available": True,
            }

        return {
            "approved": False,
            "auto_rejected": True,
            "status": "auto_rejected",
            "reject_reason": "Could not extract readable text from uploaded Word document.",
            "validation_reason": "Unreadable Word document",
            "raw_text": "",
            "goods_count": None,
            "confidence": 0.0,
            "engine": "docx_parser",
            "timestamp": ts,
            "direction": direction,
            "snapshot_b64": None,
            "ocr_available": True,
        }

    # ── CASE 3: Standard Image (.jpg, .jpeg, .png, .webp) ─────────────────────
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {
            "approved": False,
            "auto_rejected": True,
            "status": "auto_rejected",
            "reject_reason": "Could not decode document image. Please upload a clear photo or document.",
            "validation_reason": "Invalid or corrupted image format",
            "raw_text": "",
            "goods_count": None,
            "confidence": 0.0,
            "engine": "none",
            "timestamp": ts,
            "direction": direction,
            "snapshot_b64": None,
            "ocr_available": True,
        }

    h, w = frame.shape[:2]
    return scan_document_in_frame(frame, [0, 0, w, h], direction)

