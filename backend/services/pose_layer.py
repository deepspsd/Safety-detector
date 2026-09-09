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
            results = model(frame, verbose=False, conf=0.35, imgsz=640)
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

        for p in poses:
            if not p.get("available"):
                continue

            p_box = p.get("bbox", [0, 0, 0, 0])
            p_h = max(40, p_box[3] - p_box[1]) if len(p_box) == 4 else 200
            # Adapt crop radius dynamically with person bbox size (works from 480p to 1440p)
            rw = max(18, min(80, int(p_h * 0.075)))

            kp_dict = p.get("keypoints", {})
            wrist_kps = [
                ("left", kp_dict.get("left_wrist")),
                ("right", kp_dict.get("right_wrist")),
            ]

            for side, w_kp in wrist_kps:
                if not w_kp or w_kp[2] < 0.25:
                    continue

                wx, wy = int(w_kp[0]), int(w_kp[1])
                e_kp = kp_dict.get(f"{side}_elbow")

                # Sample both wrist center and extended hand/palm center
                sample_centers = [(wx, wy)]
                if e_kp and e_kp[2] > 0.25:
                    ex, ey = int(e_kp[0]), int(e_kp[1])
                    dx, dy = wx - ex, wy - ey
                    dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
                    # Hand center extends ~30% beyond wrist along forearm vector
                    hx = int(wx + (dx / dist) * (rw * 0.6))
                    hy = int(wy + (dy / dist) * (rw * 0.6))
                    sample_centers.append((hx, hy))

                for cx, cy in sample_centers:
                    x1 = max(0, cx - rw)
                    y1 = max(0, cy - rw)
                    x2 = min(fw, cx + rw)
                    y2 = min(fh, cy + rw)

                    if (x2 - x1) < 14 or (y2 - y1) < 14:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # 1. Skin tone mask in HSV (including red wrap-around)
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    mask1 = cv2.inRange(
                        hsv,
                        np.array([0, 20, 35], dtype=np.uint8),
                        np.array([25, 255, 255], dtype=np.uint8),
                    )
                    mask2 = cv2.inRange(
                        hsv,
                        np.array([165, 20, 35], dtype=np.uint8),
                        np.array([180, 255, 255], dtype=np.uint8),
                    )
                    skin_mask = cv2.bitwise_or(mask1, mask2)

                    # Non-skin pixels in the hand/wrist region
                    non_skin = cv2.bitwise_not(skin_mask)
                    total_px = float(crop.shape[0] * crop.shape[1])
                    non_skin_ratio = float(cv2.countNonZero(non_skin)) / total_px

                    # 2. Edge / boundary analysis (dials, circular bangles, straps, rings)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 40, 110)
                    edge_ratio = float(cv2.countNonZero(edges)) / total_px

                    # Foreign item criteria:
                    # - High edge density combined with non-skin presence
                    # - Or predominantly non-skin material (dark watch, bright colored bangle/band)
                    # - Or prominent structural edges (metallic jewelry, watch bezel)
                    is_item_detected = (
                        (edge_ratio > 0.06 and non_skin_ratio > 0.22)
                        or (non_skin_ratio > 0.40)
                        or (edge_ratio > 0.12 and non_skin_ratio > 0.15)
                    )

                    if is_item_detected:
                        conf = min(0.98, max(0.60, round(0.50 + edge_ratio * 1.5 + non_skin_ratio * 0.4, 2)))
                        accessory_dets.append({
                            "label": "Bangles",
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "det_type": "violation",
                            "raw_label": "hand_wrist_item",
                            "description": "Item/accessory worn on hand or wrist (Bangle/Watch/Jewelry)",
                        })
                        # Stop sampling other centers for this hand to avoid duplicate overlapping boxes
                        break

        return accessory_dets


pose_adapter = PoseAdapter()

