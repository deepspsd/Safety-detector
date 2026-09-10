"""Test uniform detection on user uploaded photo."""
import sys
import os
import cv2
import shutil

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.uniform_monitor import uniform_monitor
from services.yolo_service import process_frame_numpy, load_model
from ultralytics import YOLO

load_model()

img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_images", "cctv_synthetic", "cctv_workers_eating_uniform_test.jpg"))
if not os.path.exists(img_path):
    print(f"Error: image not found at {img_path}")
    sys.exit(1)

frame = cv2.imread(img_path)
print(f"Loaded image: {frame.shape[1]}x{frame.shape[0]}")

# 1. Direct uniform_monitor test on persons detected
person_model = YOLO("yolov8x.pt")
results = person_model(frame, verbose=False)[0]
person_boxes = []
for box in results.boxes:
    cls_id = int(box.cls[0].item())
    conf = float(box.conf[0].item())
    if cls_id == 0 and conf >= 0.35:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        person_boxes.append(xyxy)

print(f"Found {len(person_boxes)} person(s)")

for idx, p_box in enumerate(person_boxes):
    px1, py1, px2, py2 = p_box
    res = uniform_monitor.detect_person_uniform(frame, p_box)
    print(f"Person {idx+1} ({px1},{py1},{px2},{py2}):")
    print(f"  Uniform status : {res['prediction']} (has_uniform={res['has_uniform']})")
    print(f"  Confidence     : {res['confidence']:.3f}")
    print(f"  Raw class      : {res['raw_label']}")
    print(f"  Mapped box     : {res['mapped_bbox']}")

# 2. Full pipeline run (Bakery Worker role)
processed = process_frame_numpy(
    frame=frame,
    role="Bakery Worker",
    detection_filters=["NO-Bakery-Head-Cap", "NO-Uniform", "Bangles"]
)

from services.yolo_service import decode_frame

print("\nFull Pipeline Summary:")
print(f"Is compliant        : {processed.get('is_compliant')}")
print(f"Alert message       : {processed.get('alert_message')}")
print(f"Missing items       : {processed.get('missing_items')}")
print(f"Persons detected    : {len(processed.get('persons', []))}")
for i, p in enumerate(processed.get("persons", [])):
    print(f"  Person {i+1}: compliant={p.get('is_compliant')}, missing={p.get('ppe_missing')}, uniform_pred={p.get('uniform_prediction')}, uniform_conf={p.get('uniform_confidence')}")

print("\nDetections:")
for d in processed.get("detections", []):
    print(f"  - {d.get('label')} (conf={d.get('confidence'):.2f}, bbox={d.get('bbox')})")

# Save annotated image
out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_images", "cctv_synthetic", "cctv_user_photo_annotated.jpg"))
ann_img = decode_frame(processed["annotated_frame"])
cv2.imwrite(out_path, ann_img)
print(f"\nSaved annotated image: {out_path}")

# Copy to brain artifact directory for display
artifact_dir = r"C:\Users\svdy1\.gemini\antigravity-ide\brain\530c6115-354e-4ea3-afc1-eaa093aa1bc0"
artifact_out = os.path.join(artifact_dir, "cctv_user_photo_annotated.jpg")
shutil.copyfile(out_path, artifact_out)
print(f"Copied to artifact dir: {artifact_out}")
