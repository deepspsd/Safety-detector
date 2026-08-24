"""OCR integration boundary: document signals become neutral events only."""

from __future__ import annotations

from services.platform_events import emit


class OcrAdapter:
    def scan(
        self,
        camera_id: int,
        frame,
        bbox,
        *,
        direction: str = "inward",
        track_id: str | None = None,
        zone_id: int | None = None,
    ):
        from services.ocr_service import scan_document_in_frame

        result = scan_document_in_frame(frame, bbox, direction)
        passed = bool(result and result.get("approved"))
        emit(
            "OCR_SUCCESS" if passed else "OCR_FAILED",
            camera_id=camera_id,
            zone_id=zone_id,
            track_id=track_id,
            source="ocr-layer",
            payload={"result": result or {}},
        )
        return result


ocr_adapter = OcrAdapter()
