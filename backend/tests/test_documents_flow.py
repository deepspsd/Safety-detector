import unittest
import json
import io
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

if __name__ == "__main__":
    unittest.main()
