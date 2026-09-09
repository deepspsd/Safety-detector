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


pose_adapter = PoseAdapter()

