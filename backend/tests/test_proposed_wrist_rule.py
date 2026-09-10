import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from services.pose_layer import pose_adapter

tests = [
    ("User Full Photo", r"C:\Users\svdy1\.gemini\antigravity-ide\brain\530c6115-354e-4ea3-afc1-eaa093aa1bc0\.user_uploaded\media_1789033791901.jpg", 1), # Only 1 wrist has watch! Left is bare!
    ("cctv_bare_wrists.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bare_wrists.jpg", 0), # 0 accessories
    ("cctv_bangles_positive.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bangles_positive.jpg", 1), # bangles
    ("cctv_bangles_wrist.jpg", "backend/tests/test_images/cctv_synthetic/cctv_bangles_wrist.jpg", 1), # bangles
    ("Camera 13 (Coconut Cutters)", "backend/tests/test_images/cctv_synthetic/cctv_camera13_cutting_test.jpg", 1), # 1 bangle on Worker 2
]

def evaluate_wrist(img, p_box, kps, side, fw, fh):
    w_kp = kps.get(f"{side}_wrist")
    if not w_kp or w_kp[2] < 0.20:
        return False, "no_kp"
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
    px, py = -uy, ux

    p_h = max(1, p_box[3] - p_box[1])
    rw = max(18, min(36, int(p_h * 0.06)))
    shift = min(14, max(4, int(rw * 0.25)))
    crop_cx = int(wx - ux * shift)
    crop_cy = int(wy - uy * shift)

    x1 = max(0, crop_cx - rw)
    y1 = max(0, crop_cy - rw)
    x2 = min(fw, crop_cx + rw)
    y2 = min(fh, crop_cy + rw)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or (x2 - x1) < 14 or (y2 - y1) < 14:
        return False, "small_crop"

    # 1. Central wrist band analysis (where accessory sits)
    ch, cw = crop.shape[:2]
    center_crop = crop[int(ch*0.20):int(ch*0.80), int(cw*0.20):int(cw*0.80)]
    c_hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    c_m1 = cv2.inRange(c_hsv, np.array([0, 20, 45]), np.array([25, 210, 255]))
    c_m2 = cv2.inRange(c_hsv, np.array([165, 20, 45]), np.array([180, 210, 255]))
    c_skin = cv2.bitwise_or(c_m1, c_m2)
    c_skin_ratio = np.count_nonzero(c_skin) / float(center_crop.shape[0] * center_crop.shape[1])
    c_non_skin = 1.0 - c_skin_ratio

    # 2. Transverse & longitudinal gradient along arm
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
        return False, "too_few_samples"

    long_samples_arr = np.array(long_samples)
    intensity = np.mean(long_samples_arr, axis=1)
    lap = np.abs(intensity[1:-1] - (intensity[:-2] + intensity[2:]) / 2.0)
    max_contrast = float(np.max(lap)) if len(lap) > 0 else 0.0
    chroma_var = float(np.sum(np.std(long_samples_arr, axis=0)))

    # Edge density on center crop
    gray_c = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
    edges_c = cv2.Canny(gray_c, 40, 120)
    edge_ratio_c = float(np.count_nonzero(edges_c)) / float(edges_c.size)

    # 3. Vivid bangle color check (saturated non-skin colors like red/gold/green on center crop)
    c_s = c_hsv[:, :, 1]
    c_v = c_hsv[:, :, 2]
    # Saturated pixel that is NOT standard human skin tone
    vivid_foreign_mask = (c_s > 90) & (c_v > 60) & (~c_skin)
    vivid_ratio = float(np.count_nonzero(vivid_foreign_mask)) / float(center_crop.shape[0] * center_crop.shape[1])

    # 4. DECISION RULE:
    # A wrist accessory (watch, bangle, kada, bracelet, thread) requires:
    # (a) NOT clean bare skin (if center_skin > 0.75 and c_non_skin < 0.25 and vivid_ratio < 0.08, it's bare skin!)
    is_pure_skin = (c_skin_ratio >= 0.70 and c_non_skin < 0.30 and vivid_ratio < 0.08 and chroma_var < 28.0)
    if is_pure_skin:
        return False, f"bare_skin (skin={c_skin_ratio:.2f}, non_skin={c_non_skin:.2f})"

    # (b) Evidence of worn object:
    # 1. Watch / dark / metallic band: substantial non-skin in center (>= 0.35) with contrast >= 5.0
    has_watch_or_strap = (c_non_skin >= 0.35 and max_contrast >= 5.0)
    # 2. Colorful bangles: vivid ratio >= 0.10 or high chroma variance >= 30.0 with contrast >= 5.0
    has_vivid_bangle = ((vivid_ratio >= 0.10 or chroma_var >= 30.0) and max_contrast >= 4.5)
    # 3. Thin kada / jewelry / thread with edges: center_non_skin >= 0.25 and edge_ratio >= 0.15 and contrast >= 6.0
    has_jewelry_edges = (c_non_skin >= 0.25 and edge_ratio_c >= 0.15 and max_contrast >= 6.0)
    # 4. Very high contrast interruption across wrist (> 12.0) with non-skin >= 0.25
    has_sharp_interruption = (max_contrast >= 11.0 and c_non_skin >= 0.25)

    is_accessory = (has_watch_or_strap or has_vivid_bangle or has_jewelry_edges or has_sharp_interruption)
    reason = f"mc={max_contrast:.1f}, cv={chroma_var:.1f}, c_non_skin={c_non_skin:.2f}, vivid={vivid_ratio:.2f}"
    return is_accessory, reason

for name, path, exp in tests:
    img = cv2.imread(path)
    fh, fw = img.shape[:2]
    poses = pose_adapter.analyse(img)
    print(f"\n================ {name} ================")
    detected_count = 0
    for p_idx, p in enumerate(poses):
        kp = p.get("keypoints", {})
        has_shoulders = any(
            k in kp and kp[k][2] > 0.35
            for k in ("left_shoulder", "right_shoulder")
        )
        if not has_shoulders:
            continue
        p_box = p.get("bbox", [0, 0, fw, fh])
        for side in ["left", "right"]:
            is_acc, r = evaluate_wrist(img, p_box, kp, side, fw, fh)
            if is_acc:
                detected_count += 1
                print(f"  [+] P{p_idx+1} {side.upper()} wrist: ACCESSORY DETECTED ({r})")
            else:
                print(f"  [-] P{p_idx+1} {side.upper()} wrist: BARE / NO ACCESSORY ({r})")
    print(f"  --> Total Accessories Detected: {detected_count}")
