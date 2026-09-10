import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.yolo_service import process_frame_numpy, decode_frame

img_path = os.path.join(os.path.dirname(__file__), "test_images", "cctv_synthetic", "cctv_factory_altercation.jpg")
img = cv2.imread(img_path)
assert img is not None, f"Failed to load image from {img_path}"

# Process frame with yolo_service pipeline
res = process_frame_numpy(img, role="Factory Worker")

# Check detections
fight_dets = [d for d in res.get("detections", []) if "altercation" in d.get("label", "").lower() or "fight" in d.get("label", "").lower()]
print(f"[*] Fight Detections Found: {len(fight_dets)}")
for fd in fight_dets:
    print(f"    -> Label: {fd['label']} | Conf: {fd['confidence']} | Box: {fd['bbox']}")

# Decode annotated frame and save to disk
ann_b64 = res.get("annotated_frame")
assert ann_b64, "No annotated frame returned!"
ann_img = decode_frame(ann_b64)

out_path1 = os.path.join(os.path.dirname(__file__), "test_images", "cctv_synthetic", "cctv_factory_altercation_marked.jpg")
cv2.imwrite(out_path1, ann_img)
print(f"[+] Saved marked image to: {out_path1}")

# Also save to artifacts folder if accessible
artifact_dir = r"C:\Users\svdy1\.gemini\antigravity-ide\brain\244cfe7f-b767-4aec-8e6f-08d4393c9163"
if os.path.exists(artifact_dir):
    out_path2 = os.path.join(artifact_dir, "cctv_factory_altercation_marked.jpg")
    cv2.imwrite(out_path2, ann_img)
    print(f"[+] Saved artifact copy to: {out_path2}")
