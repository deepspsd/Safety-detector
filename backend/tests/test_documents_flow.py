import unittest
import json
import io
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from main import app
from database import get_db, Base, engine, InvoiceLog, OrderFormLog

client = TestClient(app)

class TestDocumentsWorkflow(unittest.TestCase):
    def test_fixed_qr_config_endpoint(self):
        """Verify GET /documents/fixed-qr-config returns portal URL and fixed key."""
        res = client.get("/documents/fixed-qr-config")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("portal_url", data)
        self.assertIn("fixed_key", data)
        self.assertEqual(data["fixed_key"], "occusafe-gate-fixed")
        self.assertIn("upload-invoice", data["portal_url"])

    def test_mobile_upload_with_fixed_token(self):
        """Verify mobile phone upload over any network works with fixed gate key and metadata."""
        # Create a 50x50 dummy white JPEG
        import cv2
        import numpy as np
        dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.putText(dummy_img, "INV-9999", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        _, img_bytes = cv2.imencode(".jpg", dummy_img)

        # Upload with inward direction and full vendor metadata
        files = {
            "file": ("invoice.jpg", img_bytes.tobytes(), "image/jpeg")
        }
        data = {
            "token": "occusafe-gate-fixed",
            "direction": "inward",
            "vendor_name": "Sunrise Flour Mills",
            "vehicle_no": "MH-12-DE-5555",
            "goods_count": 45,
            "weight": "2250 kg",
            "doc_number": "INV-9999",
            "notes": "Raw flour sacks delivered to Ground Floor dock",
        }

        res = client.post("/documents/mobile-upload", data=data, files=files)
        self.assertEqual(res.status_code, 200)
        resp_json = res.json()
        self.assertTrue(resp_json.get("saved"))
        self.assertTrue(resp_json.get("submitted_by_phone"))
        self.assertIn("gate_pass_code", resp_json)
        self.assertTrue(resp_json["gate_pass_code"].startswith("GP-INW-"))
        self.assertEqual(resp_json.get("vendor_name"), "Sunrise Flour Mills")
        self.assertEqual(resp_json.get("vehicle_no"), "MH-12-DE-5555")

    def test_mobile_upload_missing_mandatory_fields(self):
        """Verify mobile upload rejects submission with 422 if mandatory fields are missing."""
        import cv2
        import numpy as np
        dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, img_bytes = cv2.imencode(".jpg", dummy_img)
        files = {"file": ("invoice.jpg", img_bytes.tobytes(), "image/jpeg")}

        # Omit vendor_name and vehicle_no
        data = {
            "token": "occusafe-gate-fixed",
            "direction": "inward",
            "goods_count": 10,
            "weight": "100 kg",
            "doc_number": "INV-1234",
        }
        res = client.post("/documents/mobile-upload", data=data, files=files)
        self.assertEqual(res.status_code, 422)
        self.assertIn("mandatory", res.json()["detail"].lower())

    def test_auto_reject_non_invoice_image(self):
        """Verify non-invoice photo (e.g. blank/ac remote) is auto-rejected and excluded from pending queue."""
        import cv2
        import numpy as np
        # Create image simulating remote control buttons / non-invoice
        dummy_img = np.ones((150, 150, 3), dtype=np.uint8) * 200
        cv2.putText(dummy_img, "TEMP 24 ON/OFF", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        _, img_bytes = cv2.imencode(".jpg", dummy_img)

        files = {"file": ("remote.jpg", img_bytes.tobytes(), "image/jpeg")}
        data = {
            "token": "occusafe-gate-fixed",
            "direction": "inward",
            "vendor_name": "Test Vendor",
            "vehicle_no": "MH-01-AA-1111",
            "goods_count": 5,
            "weight": "50 kg",
            "doc_number": "DOC-9999",
        }
        res = client.post("/documents/mobile-upload", data=data, files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("auto_rejected"))
        self.assertEqual(data.get("status"), "auto_rejected")
        self.assertFalse(data.get("approved"))
        self.assertIsNotNone(data.get("reject_reason"))

    def test_pdf_upload_and_ocr_flow(self):
        """Verify uploading a PDF document triggers the same OCR flow and validation."""
        from PIL import Image, ImageDraw
        # Create an invoice page rendered into a PDF in memory
        img = Image.new('RGB', (800, 1000), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((50, 50), "TAX INVOICE", fill=(0, 0, 0))
        d.text((50, 80), "Supplier: Bharat Flour Mills Ltd | GSTIN: 27AAAAA0000A1Z5", fill=(0, 0, 0))
        d.text((50, 110), "Invoice No: INV-2026-7777 | Date: 09/09/2026", fill=(0, 0, 0))
        d.text((50, 140), "Item: Wheat Flour Bags | Quantity: 100 | Total Amount: Rs 150000", fill=(0, 0, 0))
        pdf_buf = io.BytesIO()
        img.save(pdf_buf, format='PDF')

        files = {"file": ("bill.pdf", pdf_buf.getvalue(), "application/pdf")}
        data = {
            "token": "occusafe-gate-fixed",
            "direction": "inward",
            "vendor_name": "Bharat Flour Mills Ltd",
            "vehicle_no": "MH-14-CC-9999",
            "goods_count": 100,
            "weight": "5000 kg",
            "doc_number": "INV-2026-7777",
        }
        res = client.post("/documents/mobile-upload", data=data, files=files)
        self.assertEqual(res.status_code, 200)
        resp_json = res.json()
        self.assertTrue(resp_json.get("saved"))
        self.assertEqual(resp_json.get("status"), "pending")
        self.assertFalse(resp_json.get("auto_rejected"))
        self.assertIsNotNone(resp_json.get("snapshot_b64"))
        self.assertTrue(resp_json["gate_pass_code"].startswith("GP-INW-"))

    def test_docx_upload_flow(self):
        """Verify uploading a Word document (.docx) extracts text and runs validation."""
        import docx
        doc = docx.Document()
        doc.add_heading("COMMERCIAL TAX INVOICE", 0)
        doc.add_paragraph("Vendor: Premier Packaging Supplies | GSTIN: 27BBBBB1111B1Z2")
        doc.add_paragraph("Invoice No: INV-DOCX-101 | Total Amount: Rs. 45000.00")
        table = doc.add_table(rows=1, cols=2)
        row = table.rows[0].cells
        row[0].text = "Corrugated Bakery Boxes"
        row[1].text = "5000 Units"
        docx_buf = io.BytesIO()
        doc.save(docx_buf)

        files = {"file": ("invoice.docx", docx_buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        data = {
            "token": "occusafe-gate-fixed",
            "direction": "inward",
            "vendor_name": "Premier Packaging Supplies",
            "vehicle_no": "MH-04-XX-4321",
            "goods_count": 5000,
            "weight": "800 kg",
            "doc_number": "INV-DOCX-101",
        }
        res = client.post("/documents/mobile-upload", data=data, files=files)
        self.assertEqual(res.status_code, 200)
        resp_json = res.json()
        self.assertTrue(resp_json.get("saved"))
        self.assertEqual(resp_json.get("status"), "pending")
        self.assertFalse(resp_json.get("auto_rejected"))

    def test_unsupported_format_rejected(self):
        """Verify invalid document format (e.g. zip/exe) is rejected with 400."""
        files = {"file": ("archive.zip", b"PK\x03\x04fakezipcontent", "application/zip")}
        data = {
            "token": "occusafe-gate-fixed",
            "direction": "inward",
            "vendor_name": "Test Vendor",
            "vehicle_no": "MH-01-AA-1111",
            "goods_count": 5,
            "weight": "50 kg",
            "doc_number": "DOC-9999",
        }
        res = client.post("/documents/mobile-upload", data=data, files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("unsupported", res.json()["detail"].lower())

if __name__ == "__main__":
    unittest.main()
