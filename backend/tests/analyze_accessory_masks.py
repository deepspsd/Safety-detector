import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

def analyze_crop(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Detect skin in HSV
    # Hue: 0-25 or 165-180
    # Sat: 25-180
    # Val: 45-255
    m1 = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([25, 200, 255]))
    m2 = cv2.inRange(hsv, np.array([165, 20, 40]), np.array([180, 200, 255]))
    skin_mask = cv2.bitwise_or(m1, m2)
    skin_cnt = np.count_nonzero(skin_mask)
    total_cnt = crop.shape[0] * crop.shape[1]
    skin_ratio = skin_cnt / float(total_cnt) if total_cnt > 0 else 0.0

    # Non-skin ratio
    non_skin_ratio = 1.0 - skin_ratio
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    dark_watch_mask = (v < 50) | ((s < 30) & (v < 85)) # black/dark watch/strap
    metallic_silver_mask = (s < 25) & (v > 160)        # silver / steel watch/kada
    vivid_bangle_mask = (s > 100) & (v > 60)          # red/gold/colorful bangles
    accessory_mask = dark_watch_mask | metallic_silver_mask | vivid_bangle_mask
    accessory_ratio = np.count_nonzero(accessory_mask) / float(total_cnt)

    return {
        "skin_ratio": skin_ratio,
        "non_skin_ratio": non_skin_ratio,
        "accessory_ratio": accessory_ratio,
    }

from services.pose_layer import pose_adapter

tests = [
    ("User Full Photo", r"C:\Users\svdy1\.gemini\antigravity-ide\brain\530c6115-354e-4ea3-afc1-eaa093aa1bc0\.user_uploaded\media_1789033791901.jpg"),
    ("cctv_bare_wrists.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bare_wrists.jpg"),
    ("cctv_bangles_positive.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bangles_positive.jpg"),
    ("cctv_bangles_wrist.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bangles_wrist.jpg"),
]

for name, path in tests:
    if not os.path.exists(path):
        continue
    img = cv2.imread(path)
    fh, fw = img.shape[:2]
    poses = pose_adapter.analyse(img)
    print(f"\n--- {name} ---")
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
            res = analyze_crop(crop)
            print(f"P{p_idx+1} {side:5s} | skin={res['skin_ratio']:.2f} | non_skin={res['non_skin_ratio']:.2f} | acc_mask={res['accessory_ratio']:.2f}")
