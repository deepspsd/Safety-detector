import os
import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger("pose_layer")


class PoseAdapter:
    """
    Adapter for YOLOv8-Pose model.
    Extracts keypoints (wrists, hips, pocket regions) to track person gestures
    and cash trajectory heading toward cashbox vs pocket.
    """

    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(self):
        self._model = None
        self._initialized = False

    def _load_model(self):
        if not self._initialized:
            self._initialized = True
            model_path = os.getenv("POSE_MODEL_PATH", "yolov8n-pose.pt")
            if not os.path.isabs(model_path):
                candidates = [
                    os.path.abspath(model_path),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", model_path)),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", model_path)),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        model_path = c
                        break

            if os.path.exists(model_path):
                try:
                    from ultralytics import YOLO
                    self._model = YOLO(model_path)
                    log.info(f"Loaded YOLO-Pose model from {model_path}")
                except Exception as exc:
                    log.warning(f"Failed to initialize YOLO-Pose model: {exc}")
            else:
                log.debug(f"YOLO-Pose model not found at {model_path}")
        return self._model

    def _load_hand_landmarker(self):
        """
        Lazily initialize Google MediaPipe HandLandmarker from portable_models_package.
        Provides robust detection of close-up hands and isolated wrists (e.g. webcam).
        """
        if not hasattr(self, "_hand_detector_loaded"):
            self._hand_detector_loaded = True
            self._hand_detector = None
            try:
                task_path = os.getenv(
                    "HAND_LANDMARKS_PATH",
                    "portable_models_package/hand_landmarks/hand_landmarker.task",
                )
                candidates = [
                    os.path.abspath(task_path),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", task_path)),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", task_path)),
                ]
                resolved = None
                for c in candidates:
                    if os.path.exists(c):
                        resolved = c
                        break
                if resolved:
                    import mediapipe as mp
                    from mediapipe.tasks import python
                    from mediapipe.tasks.python import vision

                    base_opts = python.BaseOptions(model_asset_path=resolved)
                    opts = vision.HandLandmarkerOptions(
                        base_options=base_opts,
                        num_hands=4,
                        min_hand_detection_confidence=0.30,
                        min_hand_presence_confidence=0.30,
                    )
                    self._hand_detector = vision.HandLandmarker.create_from_options(opts)
                    log.info(f"Loaded MediaPipe HandLandmarker from {resolved}")
            except Exception as e:
                log.warning(f"Failed to initialize MediaPipe HandLandmarker: {e}")
        return getattr(self, "_hand_detector", None)

    def analyse(self, frame: np.ndarray, person_boxes: Optional[List[List[int]]] = None) -> List[Dict]:
        """
        Run pose estimation on frame. Returns list of pose observations per person:
        - bbox: [x1, y1, x2, y2]
        - keypoints: dict of {name: (x, y, conf)}
        - wrists: [(x, y), ...]
        - hips: [(x, y), ...]
        - pocket_zone: [[x, y], ...] 4-point polygon around hips/upper thigh
        - available: bool
        """
        if frame is None or frame.size == 0:
            return [{"bbox": bbox, "keypoints": {}, "available": False} for bbox in (person_boxes or [])]

        model = self._load_model()
        if model is None:
            return [{"bbox": bbox, "keypoints": {}, "available": False} for bbox in (person_boxes or [])]

        try:
            results = model(frame, verbose=False, conf=0.18, imgsz=960)
            if not results or len(results) == 0:
                return [{"bbox": bbox, "keypoints": {}, "available": False} for bbox in (person_boxes or [])]

            res = results[0]
            if res.keypoints is None or len(res.keypoints) == 0:
                return [{"bbox": bbox, "keypoints": {}, "available": False} for bbox in (person_boxes or [])]

            kpts_data = res.keypoints.data.cpu().numpy()  # (N, 17, 3) or (N, 17, 2)
            boxes_data = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

            poses: List[Dict] = []
            for i in range(len(kpts_data)):
                p_box = [int(v) for v in boxes_data[i]] if i < len(boxes_data) else [0, 0, 0, 0]
                kpts = kpts_data[i]
                kp_dict = {}
                for idx, name in enumerate(self.KEYPOINT_NAMES):
                    if idx < len(kpts):
                        x, y = float(kpts[idx][0]), float(kpts[idx][1])
                        c = float(kpts[idx][2]) if len(kpts[idx]) > 2 else 1.0
                        kp_dict[name] = (x, y, c)

                # Extract wrists & hips
                wrists = [kp_dict[k][:2] for k in ("left_wrist", "right_wrist") if k in kp_dict and kp_dict[k][2] > 0.30]
                hips = [kp_dict[k][:2] for k in ("left_hip", "right_hip") if k in kp_dict and kp_dict[k][2] > 0.30]

                # Compute pocket zone from hips
                pocket_poly = []
                if len(hips) >= 2:
                    hx1 = min(hips[0][0], hips[1][0]) - 30
                    hx2 = max(hips[0][0], hips[1][0]) + 30
                    hy1 = min(hips[0][1], hips[1][1]) - 15
                    hy2 = max(hips[0][1], hips[1][1]) + 80  # down to upper thigh / pocket
                    pocket_poly = [
                        [int(hx1), int(hy1)],
                        [int(hx2), int(hy1)],
                        [int(hx2), int(hy2)],
                        [int(hx1), int(hy2)],
                    ]
                elif p_box != [0, 0, 0, 0]:
                    # Fallback to lower body fraction of bbox
                    x1, y1, x2, y2 = p_box
                    h = y2 - y1
                    pocket_poly = [
                        [x1, int(y1 + h * 0.45)],
                        [x2, int(y1 + h * 0.45)],
                        [x2, int(y1 + h * 0.80)],
                        [x1, int(y1 + h * 0.80)],
                    ]

                poses.append({
                    "bbox": p_box,
                    "keypoints": kp_dict,
                    "wrists": wrists,
                    "hips": hips,
                    "pocket_zone": pocket_poly,
                    "available": True,
                })

            return poses
        except Exception as exc:
            log.warning(f"Pose estimation error: {exc}")
            return [{"bbox": bbox, "keypoints": {}, "available": False} for bbox in (person_boxes or [])]

    def detect_wrist_accessories(
        self,
        frame: np.ndarray,
        poses: Optional[List[Dict]] = None,
        is_shop: bool = False,
    ) -> List[Dict]:
        """
        Detect any item worn on the hand or wrist: bangles, kadas, bracelets,
        watches, rings, jewelry, ribbons, threads, or bands.
        
        Spatial policy:
          Disabled in shop floor (customers and cashiers naturally wear items).
          Active across all factory floors (Ground, First, Second) and default zones.
        
        Returns list of detections:
          { 'label': 'Bangles', 'confidence': float, 'bbox': [x1, y1, x2, y2], 'det_type': 'violation', 'raw_label': 'hand_wrist_item' }
        """
        if is_shop or frame is None or frame.size == 0:
            return []

        import cv2

        if poses is None:
            poses = self.analyse(frame)

        accessory_dets = []
        fh, fw = frame.shape[:2]

        for p in (poses or []):
            if not p.get("available"):
                continue

            p_box = p.get("bbox", [0, 0, 0, 0])
            p_w = max(10, p_box[2] - p_box[0]) if len(p_box) == 4 else 100
            p_h = max(10, p_box[3] - p_box[1]) if len(p_box) == 4 else 200

            # Reject tiny/distant people (h<150px) — keypoints too unreliable at that scale.
            # Also reject heads near y=0 (ceiling hallucinations from overhead CCTV).
            if p_h < 150 or p_box[1] < 15:
                continue

            kp_dict = p.get("keypoints", {})

            # Real humans with wrists need detectable upper body (shoulders OR elbows).
            has_upper_body = any(
                kp_dict.get(k, (0, 0, 0))[2] > 0.35
                for k in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]
            )
            if not has_upper_body:
                continue

            # Scale-adaptive wrist region radius (responsive to camera distance / worker scale)
            # A human wrist is ~5-6% of total body height
            rw = max(18, min(36, int(p_h * 0.06)))

            wrist_kps = [
                ("left", kp_dict.get("left_wrist")),
                ("right", kp_dict.get("right_wrist")),
            ]

            for side, w_kp in wrist_kps:
                # Raise wrist confidence threshold — low-confidence wrist keypoints hit background.
                if not w_kp or w_kp[2] < 0.25:
                    continue

                wx, wy = int(w_kp[0]), int(w_kp[1])

                # 1. Obtain forearm vector pointing from proximal (elbow/shoulder/body) toward wrist
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
                if norm < 1e-3:
                    ux, uy = 0.0, 1.0
                else:
                    ux, uy = adx / norm, ady / norm
                px, py = -uy, ux

                # 2. Anatomical crop centering:
                # Bangle / kada / watch / bracelet sits proximal along forearm axis (toward elbow) from carpal joint.
                shift = min(14, max(4, int(rw * 0.25)))
                crop_cx = int(wx - ux * shift)
                crop_cy = int(wy - uy * shift)

                x1 = max(0, crop_cx - rw)
                y1 = max(0, crop_cy - rw)
                x2 = min(fw, crop_cx + rw)
                y2 = min(fh, crop_cy + rw)

                if (x2 - x1) < 14 or (y2 - y1) < 14:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # 3. Robust dual-space skin detection (HSV + YCrCb combined)
                ch, cw = crop.shape[:2]
                center_crop = crop[int(ch * 0.20):int(ch * 0.80), int(cw * 0.20):int(cw * 0.80)]
                if center_crop.size == 0:
                    continue
                tot_pix = float(center_crop.shape[0] * center_crop.shape[1])

                c_hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
                c_ycrcb = cv2.cvtColor(center_crop, cv2.COLOR_BGR2YCrCb)

                # Vivid foreign jewelry / sacred thread check:
                # Saturated jewelry hues (red thread / kalava, gold, lacquer, green glass bangles)
                # Evaluated FIRST so saturated red thread isn't mistaken for human skin.
                c_h = c_hsv[:, :, 0]
                c_s = c_hsv[:, :, 1]
                c_v = c_hsv[:, :, 2]
                is_jewelry_hue = ((c_h <= 35) | (c_h >= 160) | ((c_h >= 40) & (c_h <= 85)))
                vivid_foreign_mask = (c_s > 80) & (c_v > 50) & is_jewelry_hue
                vivid_ratio = float(np.count_nonzero(vivid_foreign_mask)) / tot_pix

                # Robust dual-space skin detection (HSV + YCrCb combined)
                c_m1 = cv2.inRange(c_hsv, np.array([0, 18, 38]), np.array([30, 230, 255]))
                c_m2 = cv2.inRange(c_hsv, np.array([165, 18, 38]), np.array([180, 230, 255]))
                c_skin_hsv = cv2.bitwise_or(c_m1, c_m2)
                c_skin_ycrcb = cv2.inRange(c_ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
                c_skin = cv2.bitwise_or(c_skin_hsv, c_skin_ycrcb)
                # Artificial vivid jewelry/thread pixels are strictly non-skin
                c_skin[vivid_foreign_mask] = 0
                c_skin_ratio = float(np.count_nonzero(c_skin)) / tot_pix
                c_non_skin = 1.0 - c_skin_ratio

                # 4. Longitudinal & transverse sampling along forearm axis
                long_samples = []
                for d in range(-10, 11, 2):
                    pts = []
                    for w_off in range(-6, 7, 2):
                        sx = int(crop_cx + d * ux + w_off * px)
                        sy = int(crop_cy + d * uy + w_off * py)
                        if 0 <= sx < fw and 0 <= sy < fh:
                            pts.append(frame[sy, sx])
                    if pts:
                        long_samples.append(np.mean(pts, axis=0))

                if len(long_samples) < 5:
                    continue

                long_samples_arr = np.array(long_samples)
                intensity = np.mean(long_samples_arr, axis=1)
                lap = np.abs(intensity[1:-1] - (intensity[:-2] + intensity[2:]) / 2.0)
                max_contrast = float(np.max(lap)) if len(lap) > 0 else 0.0

                # 5. BARE SKIN REJECTION GATES:
                # Gate A — High skin ratio with no vivid jewelry = guaranteed bare arm.
                if c_skin_ratio >= 0.60 and vivid_ratio < 0.06 and max_contrast < 30.0:
                    continue

                # Gate B — Check forearm continuity: sample arm-side and hand-side bands.
                # If both proximal (arm) and distal (hand) are also skin, the wrist is bare.
                def _band_skin(offset):
                    pts = []
                    for w_off in range(-6, 7, 2):
                        sx = int(crop_cx + offset * ux + w_off * px)
                        sy = int(crop_cy + offset * uy + w_off * py)
                        if 0 <= sx < fw and 0 <= sy < fh:
                            pts.append(frame[sy, sx])
                    if not pts:
                        return 0.0
                    patch = np.array([pts], dtype=np.uint8)
                    p_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    p_ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
                    pm1 = cv2.inRange(p_hsv, np.array([0, 18, 38]), np.array([30, 230, 255]))
                    pm2 = cv2.inRange(p_hsv, np.array([165, 18, 38]), np.array([180, 230, 255]))
                    pm_ycrcb = cv2.inRange(p_ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
                    pm_comb = cv2.bitwise_or(cv2.bitwise_or(pm1, pm2), pm_ycrcb)
                    return float(np.count_nonzero(pm_comb)) / float(len(pts))

                prox_skin = _band_skin(-8)   # forearm side
                dist_skin = _band_skin(8)    # hand side
                if prox_skin >= 0.45 and c_skin_ratio >= 0.45 and dist_skin >= 0.45 and vivid_ratio < 0.06 and max_contrast < 30.0:
                    continue

                # Gate C — Low-skin crop: verify the wrist keypoint is inside the person's bounding box.
                # YOLO-pose already detected this person, so a wrist KP inside their bbox IS a real human wrist
                # (works for any skin tone, including very dark skin that defeats color-space skin masks).
                # KPs that land on background / another person's clothing fall OUTSIDE the pose bbox.
                if c_skin_ratio < 0.30:
                    kp_in_person_box = (
                        len(p_box) == 4
                        and p_box[0] <= wx <= p_box[2]
                        and p_box[1] <= wy <= p_box[3]
                    )
                    if not kp_in_person_box:
                        continue

                # 6. WORN ACCESSORY TESTS:
                # (a) Colorful bangles (glass, lac, gold, red thread): vivid jewelry hues detected
                #     NOTE: chroma_var deliberately removed — background texture triggers too many FPs.
                has_vivid_bangle = (vivid_ratio >= 0.08 and max_contrast >= 5.0)

                # (b) Watch / dark strap / fitness tracker:
                #     Replaces skin at wrist (high non-skin, >= 0.65) with sharp contrast band (>= 7.0)
                has_watch_or_strap = (c_non_skin >= 0.65 and max_contrast >= 7.0)

                # (c) High-contrast metallic kada, glass/gold bangles, or thick jewelry (steel, gold, silver):
                #     Extreme contrast interruption across wrist (>= 35.0) indicates a foreign band/bangle
                #     Or sharp contrast (>= 20.0) with non-skin presence (>= 0.35) or vivid hue (>= 0.04)
                has_metallic_kada = (max_contrast >= 35.0) or (max_contrast >= 20.0 and (vivid_ratio >= 0.04 or c_non_skin >= 0.35))

                is_worn_accessory = (has_vivid_bangle or has_watch_or_strap or has_metallic_kada)

                if is_worn_accessory:
                    conf = min(0.98, max(0.60, round(0.50 + max_contrast * 0.012 + vivid_ratio * 1.5, 2)))
                    accessory_dets.append({
                        "label": "Bangles",
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "det_type": "violation",
                        "raw_label": "hand_wrist_item",
                        "description": "Item/accessory worn on hand or wrist (Bangle/Watch/Jewelry)",
                    })

        # Close-up hand inspection fallback (e.g. webcam or close-up check at security entry)
        if not accessory_dets and not any(
            p.get("available") and any(
                p.get("keypoints", {}).get(f"{s}_wrist", (0, 0, 0))[2] >= 0.20
                for s in ["left", "right"]
            )
            for p in (poses or [])
        ):
            hl = self._load_hand_landmarker()
            if hl:
                try:
                    import mediapipe as mp
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    h_res = hl.detect(mp_img)
                    for hand in (h_res.hand_landmarks or []):
                        w_lm = hand[0]
                        m_lm = hand[9]
                        wx = int(w_lm.x * fw)
                        wy = int(w_lm.y * fh)
                        mx = int(m_lm.x * fw)
                        my = int(m_lm.y * fh)
                        scale = np.hypot(mx - wx, my - wy)
                        if scale < 10:
                            continue
                        rw = max(18, min(70, int(scale * 0.70)))
                        ux = (wx - mx) / scale
                        uy = (wy - my) / scale
                        px, py = -uy, ux
                        shift = min(25, max(6, int(rw * 0.35)))
                        crop_cx = int(wx + ux * shift)
                        crop_cy = int(wy + uy * shift)
                        x1 = max(0, crop_cx - rw)
                        y1 = max(0, crop_cy - rw)
                        x2 = min(fw, crop_cx + rw)
                        y2 = min(fh, crop_cy + rw)
                        if (x2 - x1) < 14 or (y2 - y1) < 14:
                            continue

                        ch, cw = (y2 - y1), (x2 - x1)
                        center_crop = frame[y1 + int(ch * 0.20):y1 + int(ch * 0.80), x1 + int(cw * 0.20):x1 + int(cw * 0.80)]
                        if center_crop.size == 0:
                            continue
                        c_hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
                        c_m1 = cv2.inRange(c_hsv, np.array([0, 20, 45]), np.array([25, 210, 255]))
                        c_m2 = cv2.inRange(c_hsv, np.array([165, 20, 45]), np.array([180, 210, 255]))
                        c_skin = cv2.bitwise_or(c_m1, c_m2)
                        c_skin_ratio = float(np.count_nonzero(c_skin)) / float(center_crop.shape[0] * center_crop.shape[1])
                        c_non_skin = 1.0 - c_skin_ratio

                        c_s = c_hsv[:, :, 1]
                        c_v = c_hsv[:, :, 2]
                        vivid_foreign_mask = (c_s > 90) & (c_v > 60) & (~c_skin)
                        vivid_ratio = float(np.count_nonzero(vivid_foreign_mask)) / float(center_crop.shape[0] * center_crop.shape[1])

                        long_samples = []
                        d_span = max(8, min(20, int(rw * 0.50)))
                        w_span = max(5, min(14, int(rw * 0.35)))
                        step = max(1, int(rw / 14))
                        for d in range(-d_span, d_span + 1, step):
                            pts = []
                            for w_off in range(-w_span, w_span + 1, step):
                                sx = int(crop_cx + d * ux + w_off * px)
                                sy = int(crop_cy + d * uy + w_off * py)
                                if 0 <= sx < fw and 0 <= sy < fh:
                                    pts.append(frame[sy, sx])
                            if pts:
                                long_samples.append(np.mean(pts, axis=0))
                        if len(long_samples) >= 5:
                            arr = np.array(long_samples)
                            intensity = np.mean(arr, axis=1)
                            lap = np.abs(intensity[1:-1] - (intensity[:-2] + intensity[2:]) / 2.0)
                            max_c = float(np.max(lap)) if len(lap) > 0 else 0.0
                            chroma = float(np.sum(np.std(arr, axis=0)))

                            # Bare skin filter
                            if c_skin_ratio >= 0.70 and c_non_skin < 0.30 and vivid_ratio < 0.08 and chroma < 28.0:
                                continue

                            has_watch_or_strap = (c_non_skin >= 0.35 and max_c >= 5.0)
                            has_vivid_bangle = ((vivid_ratio >= 0.08 or chroma >= 32.0) and max_c >= 4.5)
                            has_sharp_interruption = (max_c >= 11.0 and c_non_skin >= 0.25)

                            if has_watch_or_strap or has_vivid_bangle or has_sharp_interruption:
                                conf = min(0.98, max(0.60, round(0.50 + max_c * 0.015 + chroma * 0.003, 2)))
                                accessory_dets.append({
                                    "label": "Bangles",
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "det_type": "violation",
                                    "raw_label": "hand_wrist_item",
                                    "description": "Item/accessory worn on hand or wrist (Bangle/Watch/Jewelry)",
                                })
                except Exception as _hle:
                    log.debug(f"[HandLandmarker] Close-up wrist check error: {_hle}")

        return accessory_dets



class FallKinematicsAnalyzer:
    """
    Temporal kinematics engine to accurately classify Fall vs Sitting.
    Eliminates false positives on sitting, bending, and kneeling.

    Operates across ALL zones (default, production, entrances, shop).

    Kinematic Indicators:
    1. Vertical velocity of descent (vy): rapid collapse vs controlled lowering.
    2. Aspect ratio dynamics (AR = height / width): sudden flip from >1.4 to <0.85.
    3. Torso orientation angle (theta): upright (>55 deg) for sitting/standing vs flat (<35 deg) on ground.
    4. Post-fall immobility duration: motionless for >= 2.0s after collapse.
    5. Controlled lowering into chair vs uncontrolled impact.
    """

    def __init__(self, history_len: int = 30):
        import collections
        import threading
        self.history_len = history_len
        self._history: Dict[Tuple[int, int], collections.deque] = {}
        self._fall_track_state: Dict[Tuple[int, int], Dict] = {}
        self._states: Dict[Tuple[int, int], Dict] = {}
        self._lock = threading.Lock()

    def update_track(
        self,
        camera_id: int,
        track_id: int,
        bbox: List[int],
        keypoints: Optional[Dict] = None,
        now: Optional[float] = None,
    ) -> Dict:
        """
        Record person observation and evaluate state machine:
        Returns: {
            "state": "standing" | "sitting" | "falling" | "fallen",
            "is_sitting": bool,
            "is_fall": bool,
            "confidence": float,
            "reason": str,
            "aspect_ratio": float,
            "torso_angle": float,
        }
        """
        import collections
        import math
        now = now or time.time()
        key = (camera_id, track_id)

        x1, y1, x2, y2 = bbox[:4]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect_ratio = float(h) / float(w)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Calculate torso angle from keypoints if available
        torso_angle = 80.0  # default upright
        hip_y = float(y1 + h * 0.6)
        shoulder_y = float(y1 + h * 0.2)

        if keypoints:
            l_sh = keypoints.get("left_shoulder")
            r_sh = keypoints.get("right_shoulder")
            l_hp = keypoints.get("left_hip")
            r_hp = keypoints.get("right_hip")

            sh_pts = [p for p in (l_sh, r_sh) if p and p[2] > 0.25]
            hp_pts = [p for p in (l_hp, r_hp) if p and p[2] > 0.25]

            if sh_pts and hp_pts:
                sx = sum(p[0] for p in sh_pts) / len(sh_pts)
                sy = sum(p[1] for p in sh_pts) / len(sh_pts)
                hx = sum(p[0] for p in hp_pts) / len(hp_pts)
                hy = sum(p[1] for p in hp_pts) / len(hp_pts)

                shoulder_y = sy
                hip_y = hy

                # Torso angle relative to vertical axis (0 deg = horizontal lying down, 90 deg = vertical upright)
                dx = abs(sx - hx)
                dy = abs(sy - hy)
                torso_angle = math.atan2(dy, dx) * (180.0 / math.pi)

        snap = {
            "time": now,
            "bbox": bbox,
            "cx": cx,
            "cy": cy,
            "hip_y": hip_y,
            "shoulder_y": shoulder_y,
            "ar": aspect_ratio,
            "torso_angle": torso_angle,
        }

        with self._lock:
            if key not in self._history:
                self._history[key] = collections.deque(maxlen=self.history_len)
            hist = self._history[key]
            hist.append(snap)

            # Analyze kinematic trajectory across history
            is_sitting = False
            is_fall = False
            state = "standing"
            reason = "normal standing"
            conf = 0.50

            # Single-frame fallback: extreme horizontal posture detected immediately
            # (works for static-image validation without multi-frame history)
            if aspect_ratio < 0.72 and torso_angle < 35.0:
                state = "fallen"
                is_fall = True
                conf = min(0.95, max(0.80, 0.80 + (0.72 - aspect_ratio) * 0.20))
                reason = f"extreme horizontal posture (AR={aspect_ratio:.2f}, angle={torso_angle:.0f}deg) — single-frame fallback"
                self._fall_track_state.setdefault(key, {"fall_ts": now, "confirmed": True})

            # Need at least 3 frames for full velocity analysis
            elif len(hist) >= 3:
                # Compare against oldest frame within 0.4s - 1.2s window
                prev = hist[0]
                dt = max(0.05, now - prev["time"])
                d_hip_y = (hip_y - prev["hip_y"]) / dt   # positive = moving down
                ar_prev = prev["ar"]
                ar_diff = aspect_ratio - ar_prev        # negative = collapsed

                # Check if person is sitting upright
                # Sitting criteria:
                # - Torso is upright (angle > 55 deg)
                # - Aspect ratio moderate (0.85 <= AR <= 1.4)
                # - Descent is controlled (d_hip_y < 250 px/s or stabilized)
                if torso_angle >= 55.0 and 0.85 <= aspect_ratio <= 1.45:
                    is_sitting = True
                    state = "sitting"
                    reason = f"controlled upright posture (angle={torso_angle:.0f}deg, AR={aspect_ratio:.2f})"
                    conf = 0.92

                # Fall criteria (temporal, with history):
                # - Aspect ratio collapsed to horizontal (AR < 0.85)
                # - Torso tilted toward ground (torso_angle < 45 deg)
                elif aspect_ratio < 0.85 and torso_angle < 45.0:
                    state = "fallen"
                    # Check immobility in fallen pose
                    recent_cxs = [s["cx"] for s in list(hist)[-5:]]
                    recent_cys = [s["cy"] for s in list(hist)[-5:]]
                    displacement = max(recent_cxs) - min(recent_cxs) + max(recent_cys) - min(recent_cys)

                    st = self._fall_track_state.setdefault(key, {"fall_ts": now, "confirmed": False})
                    time_in_fall = now - st.get("fall_ts", now)

                    if time_in_fall >= 0.5 or displacement < 25.0:
                        is_fall = True
                        conf = min(0.98, max(0.85, 0.85 + (1.0 - aspect_ratio) * 0.15))
                        reason = f"horizontal collapse confirmed (AR={aspect_ratio:.2f}, angle={torso_angle:.0f}deg, immobile={time_in_fall:.1f}s)"

                else:
                    self._fall_track_state.pop(key, None)

            # Update cached state
            res = {
                "state": state,
                "is_sitting": is_sitting,
                "is_fall": is_fall,
                "confidence": conf,
                "reason": reason,
                "aspect_ratio": aspect_ratio,
                "torso_angle": torso_angle,
            }
            self._states[key] = res
            return res

    def filter_fall_false_positives(
        self,
        camera_id: int,
        raw_fall_detections: List[Dict],
        tracked_persons: List[Dict],
        frame_shape: Tuple[int, int],
    ) -> Tuple[List[Dict], int]:
        """
        Cross-validates raw fall bounding boxes against pose kinematics.
        If the person is sitting or upright, suppresses the fall alert.
        Returns: (verified_fall_detections, suppressed_count)
        """
        if not raw_fall_detections:
            return [], 0

        verified = []
        suppressed = 0

        for fdet in raw_fall_detections:
            fb = fdet.get("bbox", [])
            if len(fb) < 4:
                continue

            fx1, fy1, fx2, fy2 = fb
            fw = max(1, fx2 - fx1)
            fh = max(1, fy2 - fy1)
            f_ar = float(fh) / float(fw)

            # Match with closest tracked person
            matched_person = None
            best_iou = 0.15
            for p in tracked_persons:
                pb = p.get("bbox", [])
                if len(pb) < 4:
                    continue
                # Intersection
                ix1 = max(fx1, pb[0])
                iy1 = max(fy1, pb[1])
                ix2 = min(fx2, pb[2])
                iy2 = min(fy2, pb[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = fw * fh + (pb[2] - pb[0]) * (pb[3] - pb[1]) - inter
                    iou = inter / max(1.0, union)
                    if iou > best_iou:
                        best_iou = iou
                        matched_person = p

            if matched_person:
                tid = int(matched_person.get("track_id", -1))
                kstate = self._states.get((camera_id, tid))
                if kstate and kstate.get("is_sitting"):
                    # Suppress sitting false positive
                    suppressed += 1
                    log.info(
                        f"[FallKinematics] Suppressed sitting false positive for cam={camera_id} track={tid} ({kstate.get('reason')})"
                    )
                    continue

                if kstate and kstate.get("torso_angle", 90.0) > 60.0 and f_ar >= 0.90:
                    # Upright posture — definitely not a horizontal fall
                    suppressed += 1
                    log.info(
                        f"[FallKinematics] Suppressed upright posture false positive for cam={camera_id} (AR={f_ar:.2f})"
                    )
                    continue

            # If aspect ratio is genuinely horizontal (< 0.85) or kinematics confirmed:
            verified.append(fdet)

        return verified, suppressed


pose_adapter = PoseAdapter()
fall_analyzer = FallKinematicsAnalyzer()
