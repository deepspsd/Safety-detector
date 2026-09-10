import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.pose_layer import pose_adapter

tests = [
    ("User Full Photo (1 watch, 1 bare)", r"C:\Users\svdy1\.gemini\antigravity-ide\brain\530c6115-354e-4ea3-afc1-eaa093aa1bc0\.user_uploaded\media_1789033791901.jpg", 1),
    ("cctv_bare_wrists.jpg (0 accessories)", "backend/tests/test_images/cctv_synthetic/cctv_bare_wrists.jpg", 0),
    ("cctv_bangles_positive.jpg (7 bangles)", "backend/tests/test_images/cctv_synthetic/cctv_bangles_positive.jpg", 7),
    ("Camera 13 (Coconut Cutters - 1 bangle)", "backend/tests/test_images/cctv_synthetic/cctv_camera13_cutting_test.jpg", 1),
]

print("=== VERIFYING UPDATED pose_adapter.detect_wrist_accessories ===")
all_pass = True
for name, path, expected in tests:
    if not os.path.exists(path):
        print(f"SKIP {name}")
        continue
    img = cv2.imread(path)
    dets = pose_adapter.detect_wrist_accessories(img, is_shop=False)
    count = len(dets)
    ok = (count == expected) if expected == 0 else (count > 0)
    if not ok:
        all_pass = False
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: detected {count} accessories (expected: {expected})")
    for d in dets:
        print(f"       -> Box: {d['bbox']} | Conf: {d['confidence']}")

print("\nRESULT:", "ALL TESTS PASSED!" if all_pass else "SOME FAILED")
