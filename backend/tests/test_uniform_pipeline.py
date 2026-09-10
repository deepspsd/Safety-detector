import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from config import settings
from services.uniform_monitor import uniform_monitor
from services.yolo_service import process_frame_numpy

print("Uniform model path in config:", settings.MODEL_REGISTRY["uniform_detector"]["path"])
print("Uniform conf threshold in config:", settings.MODEL_REGISTRY["uniform_detector"]["conf_threshold"])

img_path = os.path.abspath("backend/tests/test_images/cctv_synthetic/cctv_factory_worker_floor.jpg")
img = cv2.imread(img_path)
print("Loaded test image:", img.shape)

res = process_frame_numpy(
    frame=img,
    role="Bakery Worker",
    detection_filters=["NO-Bakery-Head-Cap", "NO-Uniform", "Bangles"],
)

print("\n--- RESULTS ---")
print("is_compliant:", res["is_compliant"])
print("alert_message:", res["alert_message"])
print("missing_items:", res["missing_items"])
print("persons detected:", len(res["persons"]))
for i, p in enumerate(res["persons"]):
    print(f"Person {i}: bbox={p['bbox']}, compliant={p['is_compliant']}, found={p.get('ppe_found')}, missing={p.get('ppe_missing')}, u_pred={p.get('uniform_prediction')}, u_conf={p.get('uniform_confidence')}")

print("\n--- DETECTIONS ---")
for d in res.get("detections", []):
    lbl = d.get("label", "")
    if any(k in lbl.lower() for k in ("uniform", "cap", "hairnet", "bangle")):
        print(f"  {lbl}: conf={d.get('confidence')}, det_type={d.get('det_type')}, bbox={d.get('bbox')}")

from services.yolo_service import decode_frame

out_path = os.path.abspath("backend/tests/test_images/cctv_synthetic/output_uniform_test.jpg")
ann_img = decode_frame(res["annotated_frame"])
cv2.imwrite(out_path, ann_img)
print("\nSuccessfully saved annotated test output to:", out_path)
