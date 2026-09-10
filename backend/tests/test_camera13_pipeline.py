"""Test bangle, uniform, and hairnet models on Camera 13 CCTV image."""
import os
import sys
import shutil
import cv2

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.yolo_service import process_frame_numpy, load_model, decode_frame
from services.uniform_monitor import uniform_monitor
from services.pose_layer import pose_adapter
from ultralytics import YOLO

# 1. Image path
src_img_path = r"C:\Users\svdy1\.gemini\antigravity-ide\brain\530c6115-354e-4ea3-afc1-eaa093aa1bc0\.user_uploaded\media_1789035226337.png"
dst_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_images", "cctv_synthetic", "cctv_camera13_cutting_test.jpg"))

os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
shutil.copyfile(src_img_path, dst_img_path)
print(f"Copied image to: {dst_img_path}")

frame = cv2.imread(dst_img_path)
print(f"Loaded image shape: {frame.shape} (H={frame.shape[0]}, W={frame.shape[1]})")

# Preload models
load_model()

# 2. Test Person Detection
person_model = YOLO("yolov8x.pt")
p_res = person_model(frame, verbose=False)[0]
person_boxes = []
for b in p_res.boxes:
    cls_id = int(b.cls[0].item())
    conf = float(b.conf[0].item())
    if cls_id == 0 and conf >= 0.25:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        person_boxes.append((conf, xyxy))

print(f"\n--- 1. PERSON DETECTION ---")
print(f"Found {len(person_boxes)} person(s):")
for i, (conf, box) in enumerate(person_boxes):
    print(f"  Person {i+1}: conf={conf:.3f}, bbox={list(box)}")

# 3. Test Uniform Model directly on each detected person
print(f"\n--- 2. UNIFORM MODEL TEST (best_uniform_detector.pt, conf=0.45) ---")
for i, (conf, box) in enumerate(person_boxes):
    ures = uniform_monitor.detect_person_uniform(frame, box, conf_threshold=0.45)
    print(f"  Person {i+1} bbox={list(box)}:")
    print(f"    has_uniform : {ures['has_uniform']}")
    print(f"    prediction  : {ures['prediction']}")
    print(f"    confidence  : {ures['confidence']:.3f}")
    print(f"    raw_label   : {ures['raw_label']}")
    print(f"    mapped_bbox : {ures['mapped_bbox']}")

# 4. Test Hairnet / Headcap Model directly
print(f"\n--- 3. HAIRNET / HEADCAP MODEL TEST ---")
from services.multi_model_detector import get_multi_model_detector
mm_detector = get_multi_model_detector()
h_dets = mm_detector.detect(frame, zone_type="default")
print(f"MultiModelDetector detections count: {len(h_dets)}")
for hd in h_dets:
    print(f"  Detection: label={hd.label}, raw_label={hd.raw_label}, conf={hd.confidence:.3f}, bbox={hd.bbox}")

# Also test raw hairnet model directly
hairnet_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "portable_models_package", "hairnet_glove_detection", "best.pt"))
if os.path.exists(hairnet_model_path):
    hn_model = YOLO(hairnet_model_path)
    hn_res = hn_model(frame, conf=0.15, imgsz=960, verbose=False)[0]
    print(f"\nRaw hairnet model detections count: {len(hn_res.boxes)}")
    for b in hn_res.boxes:
        cls_id = int(b.cls[0].item())
        c = float(b.conf[0].item())
        box = b.xyxy[0].cpu().numpy().astype(int)
        cname = hn_model.names.get(cls_id, str(cls_id))
        print(f"  Raw hairnet: {cname} @ {c:.3f}, bbox={list(box)}")


# 5. Test Wrist Accessories / Bangles directly
print(f"\n--- 4. WRIST ACCESSORY / BANGLE DETECTION ---")
wrist_dets = pose_adapter.detect_wrist_accessories(frame, is_shop=False)
print(f"Wrist accessory detections count: {len(wrist_dets)}")
for wd in wrist_dets:
    print(f"  Wrist detection: label={wd.get('label')}, raw={wd.get('raw_label')}, conf={wd.get('confidence')}, bbox={wd.get('bbox')}")

# 6. Full Integrated Pipeline Run
print(f"\n--- 5. FULL PIPELINE RUN (Bakery Worker role) ---")
pipeline_out = process_frame_numpy(
    frame=frame,
    role="Bakery Worker",
    zone_type="default",
    detection_filters=["NO-Bakery-Head-Cap", "NO-Uniform", "Bangles"]
)

print(f"Is compliant     : {pipeline_out.get('is_compliant')}")
print(f"Alert message    : {pipeline_out.get('alert_message')}")
print(f"Missing items    : {pipeline_out.get('missing_items')}")
print(f"Persons count    : {len(pipeline_out.get('persons', []))}")
for i, p in enumerate(pipeline_out.get("persons", [])):
    print(f"  Worker {i+1}:")
    print(f"    bbox             : {p.get('bbox')}")
    print(f"    compliant        : {p.get('is_compliant')}")
    print(f"    ppe_missing      : {p.get('ppe_missing')}")
    print(f"    ppe_found        : {p.get('ppe_found')}")
    print(f"    uniform_pred     : {p.get('uniform_prediction')} (conf={p.get('uniform_confidence')})")
    print(f"    head_cap         : {p.get('head_cap')}")

print("\nAll Rendered Detections on Frame:")
for d in pipeline_out.get("detections", []):
    print(f"  [{d.get('det_type', '').upper()}] {d.get('label')} @ {d.get('confidence'):.2f} - bbox={d.get('bbox')}")

# 7. Save annotated frame
out_annotated = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_images", "cctv_synthetic", "cctv_camera13_annotated.jpg"))
ann_img = decode_frame(pipeline_out["annotated_frame"])
cv2.imwrite(out_annotated, ann_img)
print(f"\nSaved annotated image to: {out_annotated}")

# Copy to brain artifact directory
artifact_dir = r"C:\Users\svdy1\.gemini\antigravity-ide\brain\530c6115-354e-4ea3-afc1-eaa093aa1bc0"
artifact_out = os.path.join(artifact_dir, "cctv_camera13_annotated.jpg")
shutil.copyfile(out_annotated, artifact_out)
print(f"Copied to artifact dir: {artifact_out}")
