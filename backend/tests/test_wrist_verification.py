import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from services.pose_layer import pose_adapter

tests = [
    ("cctv_bangles_positive.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bangles_positive.jpg", 1),
    ("cctv_bangles_wrist.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bangles_wrist.jpg", 1),
    ("cctv_bare_wrists.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bare_wrists.jpg", 0),
    ("Camera 13 (Coconut Cutters)", "backend/tests/test_images/cctv_synthetic/cctv_camera13_cutting_test.jpg", 1),
    ("User Full Photo", r"C:\Users\svdy1\.gemini\antigravity-ide\brain\530c6115-354e-4ea3-afc1-eaa093aa1bc0\.user_uploaded\media_1789033791901.jpg", 1),
]

all_ok = True
print("=== VERIFYING WRIST ACCURACY & CROPPING ===")
for name, path, expected in tests:
    img = cv2.imread(path)
    if img is None:
        print(f"SKIP {name}: file not found")
        continue
    dets = pose_adapter.detect_wrist_accessories(img, is_shop=False)
    n = len(dets)
    ok = (n > 0) == (expected > 0)
    if not ok:
        all_ok = False
    status = "PASS" if ok else "FAIL"
    exp_str = ">0 (Positive)" if expected else "0 (Negative)"
    print(f"[{status}] {name}: detected {n} (expected {exp_str})")
print("\n=== VERIFYING FULL LIVE MONITOR YOLO PIPELINE ===")
from services.yolo_service import process_frame_numpy

pipeline_tests = [
    ("cctv_bangles_positive.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bangles_positive.jpg", True),
    ("cctv_bangles_wrist.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bangles_wrist.jpg", True),
    ("cctv_bare_wrists.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bare_wrists.jpg", False),
    ("Camera 13 (Coconut Cutters)", "backend/tests/test_images/cctv_synthetic/cctv_camera13_cutting_test.jpg", True),
]

for name, path, expected in pipeline_tests:
    img = cv2.imread(path)
    res = process_frame_numpy(img, role="Factory Worker", floor="ground")
    dets = res.get("detections", [])
    bangles = [d for d in dets if "bangle" in str(d.get("label", "")).lower()]
    has_b = len(bangles) > 0
    ok = (has_b == expected)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Full Live Pipeline {name}: {len(bangles)} bangle(s) in detections")
    if res.get("annotated_frame"):
        print(f"       annotated_frame base64 generated: {len(res['annotated_frame'])} chars")

print("\nOVERALL RESULT:", "ALL PASSED" if all_ok else "SOME FAILED")

