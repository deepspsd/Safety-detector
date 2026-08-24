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
import logging
import re
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("ocr_service")

# ──────────────────────────────────────────────────────────────────────────────
# Tesseract availability guard
# ──────────────────────────────────────────────────────────────────────────────
try:
    import os

    import pytesseract

    # ── Windows: set tesseract_cmd to the default UB-Mannheim install path ──────
    # winget / installer puts the binary here; PATH may not be refreshed yet
    # in the current process without a restart.
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
            # Try to find via PATH anyway
            import shutil

            tess = shutil.which("tesseract")
            if tess:
                pytesseract.pytesseract.tesseract_cmd = tess
                log.info(f"✅ pytesseract: found Tesseract in PATH at {tess}")
            else:
                log.warning(
                    "⚠️  Tesseract binary not found at default Windows paths. "
                    "Install via: winget install UB-Mannheim.TesseractOCR\n"
                    "  Expected: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
                )

    _TESSERACT_OK = True
    log.info("✅ pytesseract available — Tesseract OCR will be used")
except ImportError:
    _TESSERACT_OK = False
    log.warning(
        "⚠️  pytesseract not installed. OCR will be unavailable. "
        "Run: pip install pytesseract && install Tesseract system binary."
    )

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


def _text_looks_like_document(text: str, direction: str) -> bool:
    """
    Heuristic approval: True if OCR text contains at least one pattern
    matching the expected document type.

    v1 intentionally lenient — a single number/date match passes.
    Tighten after collecting real document samples from the client.
    """
    patterns = _INVOICE_PATTERNS if direction == "inward" else _ORDER_PATTERNS
    stripped = text.strip()
    if len(stripped) < 4:  # almost certainly a bad crop / OCR failure
        return False
    return any(p.search(stripped) for p in patterns)


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
        timestamp      : str  — ISO-8601 UTC
        direction      : str  — echoed back
        snapshot_b64   : str | None — base64 JPEG of the cropped document area
        ocr_available  : bool — False if Tesseract not installed (degrades gracefully)

    ⚠️  This function returns a compliance-check result only.
        It does NOT interact with any physical door/gate hardware.
        Any access-control decision must be made by a human operator
        reviewing the generated alert in the dashboard.
    """
    ts = datetime.datetime.utcnow().isoformat() + "Z"

    # Step 1 — crop
    crop = _crop_with_padding(frame, bbox)
    snapshot_b64 = _encode_crop(crop)

    # Step 2 — Tesseract unavailable → degrade gracefully
    if not _TESSERACT_OK:
        log.warning("OCR called but pytesseract not installed. Returning REVIEW alert.")
        return {
            "approved": False,
            "raw_text": "[OCR_UNAVAILABLE — install pytesseract + Tesseract binary]",
            "timestamp": ts,
            "direction": direction,
            "snapshot_b64": snapshot_b64,
            "ocr_available": False,
        }

    # Step 3 — pre-process + run Tesseract
    # Try multiple PSM modes; pick the one that extracts the most text.
    # PSM 6  = single uniform block — best for printed invoices / receipts
    # PSM 11 = sparse text — best for camera shots where doc is in a scene
    # PSM 3  = full auto layout — fallback for complex layouts
    # OEM 3  = LSTM (most accurate Tesseract engine)
    _PSM_MODES = [
        "--psm 6 --oem 3",
        "--psm 11 --oem 3",
        "--psm 3 --oem 3",
    ]
    raw_text = ""
    try:
        processed = _preprocess_for_ocr(crop)
        best_text = ""
        for psm_config in _PSM_MODES:
            try:
                candidate = pytesseract.image_to_string(
                    processed, config=psm_config, lang="eng"
                ).strip()
                # Clean noise immediately after each attempt
                candidate = _clean_ocr_text(candidate)
                if len(candidate) > len(best_text):
                    best_text = candidate
                    if len(best_text) > 60:
                        break  # good enough — stop trying other modes
            except Exception:
                continue
        raw_text = best_text
        log.info(
            f"[OCR] direction={direction} | text_len={len(raw_text)} | "
            f"preview={raw_text[:80]!r}"
        )
    except Exception as exc:
        log.error(f"[OCR] Tesseract error: {exc}")
        raw_text = f"[OCR_ERROR: {type(exc).__name__}]"

    # Step 4 — heuristic approval (always log raw_text regardless of outcome)
    approved = _text_looks_like_document(raw_text, direction)

    if not approved:
        log.warning(
            f"[OCR] Document NOT approved | direction={direction} | "
            f"raw_text={raw_text[:120]!r}"
        )
    else:
        log.info(f"[OCR] Document approved | direction={direction}")

    return {
        "approved": approved,
        "raw_text": raw_text,
        "goods_count": _extract_goods_count(raw_text),
        "timestamp": ts,
        "direction": direction,
        "snapshot_b64": snapshot_b64,
        "ocr_available": True,
    }
