import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.pose_layer import pose_adapter

tests = [
    ("cctv_bare_wrists.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bare_wrists.jpg"),
    ("cctv_bangles_positive.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bangles_positive.jpg"),
    ("cctv_bangles_wrist.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bangles_wrist.jpg"),
]

for name, path in tests:
    img = cv2.imread(path)
    fh, fw = img.shape[:2]
    poses = pose_adapter.analyse(img)
    print(f"\n================ {name} ================")
    for p_idx, p in enumerate(poses):
        kps = p.get("keypoints", {})
        p_box = p.get("bbox", [0, 0, fw, fh])
        p_h = max(1, p_box[3] - p_box[1])
        rw = max(18, min(36, int(p_h * 0.06)))
        for side in ["left", "right"]:
            w_kp = kps.get(f"{side}_wrist")
            if not w_kp or w_kp[2] < 0.20:
                continue
            wx, wy = int(w_kp[0]), int(w_kp[1])
            e_kp = kps.get(f"{side}_elbow")
            s_kp = kps.get(f"{side}_shoulder")
            if e_kp and e_kp[2] >= 0.20:
                adx, ady = wx - int(e_kp[0]), wy - int(e_kp[1])
            elif s_kp and s_kp[2] >= 0.20:
                adx, ady = wx - int(s_kp[0]), wy - int(s_kp[1])
            else:
                body_cx = (p_box[0] + p_box[2]) / 2.0
                body_cy = (p_box[1] + p_box[3]) / 2.0
                adx, ady = wx - body_cx, wy - body_cy

            norm = np.hypot(adx, ady)
            ux, uy = (adx / norm, ady / norm) if norm >= 1e-3 else (0.0, 1.0)
            shift = min(14, max(4, int(rw * 0.25)))
            crop_cx = int(wx - ux * shift)
            crop_cy = int(wy - uy * shift)

            x1 = max(0, crop_cx - rw)
            y1 = max(0, crop_cy - rw)
            x2 = min(fw, crop_cx + rw)
            y2 = min(fh, crop_cy + rw)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ch, cw = crop.shape[:2]
            center_crop = crop[int(ch*0.25):int(ch*0.75), int(cw*0.25):int(cw*0.75)]
            c_hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            c_m1 = cv2.inRange(c_hsv, np.array([0, 20, 45]), np.array([25, 210, 255]))
            c_m2 = cv2.inRange(c_hsv, np.array([165, 20, 45]), np.array([180, 210, 255]))
            c_skin = cv2.bitwise_or(c_m1, c_m2)
            c_skin_ratio = np.count_nonzero(c_skin) / float(center_crop.shape[0] * center_crop.shape[1])
            c_non_skin = 1.0 - c_skin_ratio

            print(f"P{p_idx+1} {side:5s} | center_skin={c_skin_ratio:.2f} | center_non_skin={c_non_skin:.2f}")
