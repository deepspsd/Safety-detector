import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from services.pose_layer import pose_adapter

img_paths = [
    r"C:\Users\svdy1\.gemini\antigravity-ide\brain\530c6115-354e-4ea3-afc1-eaa093aa1bc0\.user_uploaded\media_1789033791901.jpg",
    "backend/tests/test_images/cctv_synthetic/cctv_bare_wrists.jpg",
    "backend/tests/test_images/cctv_synthetic/cctv_bangles_positive.jpg",
]

for p in img_paths:
    if not os.path.exists(p):
        continue
    img = cv2.imread(p)
    fh, fw = img.shape[:2]
    poses = pose_adapter.analyse(img)
    print(f"\n================ IMAGE: {os.path.basename(p)} ({fw}x{fh}) ================")
    for p_idx, pose in enumerate(poses):
        kp_dict = pose.get("keypoints", {})
        p_box = pose.get("bbox", [0, 0, fw, fh])
        p_h = max(1, p_box[3] - p_box[1])
        rw = max(18, min(36, int(p_h * 0.06)))
        for side in ["left", "right"]:
            w_kp = kp_dict.get(f"{side}_wrist")
            if not w_kp or w_kp[2] < 0.20:
                continue
            wx, wy = int(w_kp[0]), int(w_kp[1])
            e_kp = kp_dict.get(f"{side}_elbow")
            s_kp = kp_dict.get(f"{side}_shoulder")
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
            px, py = -uy, ux

            shift = min(14, max(4, int(rw * 0.25)))
            crop_cx = int(wx - ux * shift)
            crop_cy = int(wy - uy * shift)

            x1 = max(0, crop_cx - rw)
            y1 = max(0, crop_cy - rw)
            x2 = min(fw, crop_cx + rw)
            y2 = min(fh, crop_cy + rw)

            long_samples = []
            for d in range(-10, 11, 2):
                pts = []
                for w_off in range(-6, 7, 2):
                    sx = int(crop_cx + d * ux + w_off * px)
                    sy = int(crop_cy + d * uy + w_off * py)
                    if 0 <= sx < fw and 0 <= sy < fh:
                        pts.append(img[sy, sx])
                if pts:
                    long_samples.append(np.mean(pts, axis=0))

            if len(long_samples) < 5:
                continue

            long_samples_arr = np.array(long_samples)
            intensity = np.mean(long_samples_arr, axis=1)
            lap = np.abs(intensity[1:-1] - (intensity[:-2] + intensity[2:]) / 2.0)
            max_contrast = float(np.max(lap)) if len(lap) > 0 else 0.0
            chroma_var = float(np.sum(np.std(long_samples_arr, axis=0)))

            # Crop skin ratio test
            crop = img[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # Standard skin masks
            mask1 = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([25, 200, 255]))
            mask2 = cv2.inRange(hsv, np.array([165, 25, 45]), np.array([180, 200, 255]))
            skin_mask = cv2.bitwise_or(mask1, mask2)
            skin_ratio = float(np.count_nonzero(skin_mask)) / float(crop.shape[0] * crop.shape[1])
            non_skin_ratio = 1.0 - skin_ratio

            # Edge density
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)

            # Transverse band edge check (edges aligned with transverse vector px, py)
            print(f"  Person {p_idx+1} {side.upper()} wrist: max_contrast={max_contrast:.2f}, chroma_var={chroma_var:.2f}, skin_ratio={skin_ratio:.2f}, non_skin_ratio={non_skin_ratio:.2f}, edge_ratio={edge_ratio:.3f}")
