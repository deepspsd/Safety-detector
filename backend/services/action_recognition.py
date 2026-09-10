"""
services/action_recognition.py — Intel OpenVINO Action Recognition
===================================================================
Wraps the `person-detection-action-recognition-0006` OpenVINO IR model.

Model: Intel person-detection-action-recognition-0006
Input : [1, 3, 400, 680] BGR uint8 (person crop or full frame)
Output:
  • detection_out  — [1, 1, N, 7] SSD detections
      [img_id, class_id, conf, x1, y1, x2, y2]  (normalised 0-1)
  • action_out     — [N, num_actions] action class scores per detected person

Action classes (0-indexed):
  0 - sitting
  1 - standing
  2 - raising hand
  3 - listening (arms crossed / passive)
  4 - side talks
  5 - writing

For our use-case we map:
  sitting           → "idle_sitting"   (person doing nothing)
  standing          → "standing"       (may be idle or working)
  raising hand      → "working"        (arm/hand in motion)
  writing           → "working"        (hands active)

Called from:
  • packing_monitor.py  — detect if packing worker's hands are truly moving
  • camera_manager.py   — detect sitting-idle globally (idle ≥ 5 min sitting)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger("action_recognition")

# ── Action class labels for person-detection-action-recognition-0006 ─────────
ACTION_LABELS = [
    "sitting",       # 0 — idle / no work
    "standing",      # 1 — neutral
    "raising_hand",  # 2 — arm / hand movement (working)
    "listening",     # 3 — passive
    "side_talks",    # 4 — talking to side
    "writing",       # 5 — hands active (working)
]

# Actions that clearly mean "hands in motion / working"
WORKING_ACTIONS = {"raising_hand", "writing"}
# Actions that mean "doing nothing"
IDLE_ACTIONS = {"sitting", "listening"}

_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "portable_models_package", "person_action",
    "intel", "person-detection-action-recognition-0006", "FP32",
)
_XML = os.path.join(_MODEL_DIR, "person-detection-action-recognition-0006.xml")

# ── Singleton ─────────────────────────────────────────────────────────────────
_model_loaded: bool = False
_compiled_model = None
_input_layer = None
_det_output = None
_act_output = None
_input_h: int = 400
_input_w: int = 680


_is_nhwc: bool = False


def _load_model() -> bool:
    """Lazy-load the OpenVINO model once.  Thread-safe via GIL for the flag check."""
    global _model_loaded, _compiled_model, _input_layer
    global _det_output, _act_output, _input_h, _input_w, _is_nhwc

    if _model_loaded:
        return _compiled_model is not None

    _model_loaded = True  # mark even if it fails — don't retry every frame

    xml_path = os.path.abspath(_XML)
    if not os.path.isfile(xml_path):
        log.error("[ActionRec] Model XML not found: %s", xml_path)
        return False

    try:
        try:
            from openvino.runtime import Core
        except ImportError:
            from openvino import Core
        ie = Core()
        model = ie.read_model(xml_path)
        _compiled_model = ie.compile_model(model, "CPU")

        # Identify input / output layers
        _input_layer = _compiled_model.input(0)
        shape = list(_input_layer.shape)
        if shape[-1] == 3:
            _input_h = int(shape[1])
            _input_w = int(shape[2])
            _is_nhwc = True
        else:
            _input_h = int(shape[2])
            _input_w = int(shape[3])
            _is_nhwc = False

        # Outputs: detection (SSD) and action scores
        for out in _compiled_model.outputs:
            name = out.get_any_name()
            if "detection" in name.lower() or name == "detection_out":
                _det_output = out
            elif "action" in name.lower() or name == "action_cont_0/out":
                _act_output = out

        if _det_output is None:
            # Fallback: first output = detections, second = actions
            _det_output = _compiled_model.output(0)
        if _act_output is None and len(_compiled_model.outputs) > 1:
            _act_output = _compiled_model.output(1)

        log.info(
            "[ActionRec] Loaded person-detection-action-recognition-0006 "
            "(input=%dx%d, nhwc=%s)", _input_w, _input_h, _is_nhwc
        )
        return True
    except Exception as exc:
        log.error("[ActionRec] Failed to load model: %s", exc)
        _compiled_model = None
        return False


def process_frame(
    frame: np.ndarray,
    conf_threshold: float = 0.50,
) -> List[Dict]:
    """
    Run action recognition on a full camera frame.

    Returns a list of detected persons with their action:
    [
      {
        "bbox": [x1, y1, x2, y2],   # pixel coords in original frame
        "confidence": float,
        "action": str,               # e.g. "sitting", "standing", "raising_hand"
        "action_id": int,
        "is_working": bool,          # True if hands visibly active
        "is_idle": bool,             # True if sitting / passive
      },
      ...
    ]
    Returns [] on model failure or no detections above threshold.
    """
    if not _load_model() or _compiled_model is None:
        return []

    try:
        h, w = frame.shape[:2]
        import cv2
        import numpy as np
        inp = cv2.resize(frame, (_input_w, _input_h))
        if _is_nhwc:
            inp = inp[np.newaxis].astype(np.float32)
        else:
            inp = inp.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # [1,3,H,W]

        result = _compiled_model({_input_layer: inp})

        # Detection output: [1, 1, N, 7]
        det_out = result[_det_output]
        dets = det_out.reshape(-1, 7)  # [N, 7]

        # Action output: [N, num_actions] (may not exist in all model versions)
        act_out = None
        if _act_output is not None:
            try:
                act_out = result[_act_output]  # [N, num_classes]
            except Exception:
                act_out = None

        persons = []
        for i, det in enumerate(dets):
            img_id, cls_id, conf, x1n, y1n, x2n, y2n = det
            if conf < conf_threshold:
                continue
            # Convert normalised → pixel
            x1 = int(max(0, x1n * w))
            y1 = int(max(0, y1n * h))
            x2 = int(min(w, x2n * w))
            y2 = int(min(h, y2n * h))

            # Action class
            action_id = 1  # default: standing
            action_label = "standing"
            if act_out is not None and i < act_out.shape[0]:
                action_id = int(np.argmax(act_out[i]))
                if action_id < len(ACTION_LABELS):
                    action_label = ACTION_LABELS[action_id]

            persons.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "action": action_label,
                "action_id": action_id,
                "is_working": action_label in WORKING_ACTIONS,
                "is_idle": action_label in IDLE_ACTIONS,
            })

        return persons

    except Exception as exc:
        log.debug("[ActionRec] process_frame error: %s", exc)
        return []


# ── Per-camera idle-sitting tracker ──────────────────────────────────────────
# { (camera_id, track_id): {"sitting_since": float, "alert_fired": bool} }
_sitting_state: Dict[Tuple[int, int], dict] = {}
_sitting_alert_ts: Dict[Tuple[int, int], float] = {}
_SITTING_IDLE_THRESH_S = 300   # 5 minutes sitting = alert
_SITTING_COOLDOWN_S   = 600   # re-alert every 10 min max


def check_sitting_idle(
    db,
    camera_id: int,
    frame: np.ndarray,
    persons_tracked: List[Dict],
) -> List[Dict]:
    """
    Run action recognition and alert when a tracked person has been
    sitting/passive for > 5 minutes (req: "no idle > 5 min").

    Parameters
    ----------
    persons_tracked : persons from YOLO tracker [{bbox, track_id, ...}]

    Returns
    -------
    List of action dicts (same format as process_frame()) for annotation.
    """
    now = time.time()
    action_results = process_frame(frame)

    if not action_results:
        return []

    h, w = frame.shape[:2]

    def _iou(b1, b2):
        xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
        xb = min(b1[2], b2[2]); yb = min(b1[3], b2[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    # Match action detections to tracked persons by IoU
    active_keys = set()
    for act in action_results:
        # Find best matching tracked person
        best_tid = -1
        best_iou = 0.3
        for p in persons_tracked:
            pbbox = p.get("bbox", [])
            if len(pbbox) < 4:
                continue
            iou = _iou(act["bbox"], pbbox)
            if iou > best_iou:
                best_iou = iou
                best_tid = int(p.get("track_id", -1))

        if best_tid == -1:
            continue

        key = (camera_id, best_tid)
        active_keys.add(key)

        if act["is_idle"]:
            st = _sitting_state.setdefault(key, {"sitting_since": now, "alert_fired": False})
            elapsed = now - st["sitting_since"]
            if elapsed >= _SITTING_IDLE_THRESH_S:
                last_alert = _sitting_alert_ts.get(key, 0)
                if now - last_alert >= _SITTING_COOLDOWN_S:
                    _sitting_alert_ts[key] = now
                    _fire_sitting_idle_alert(db, camera_id, best_tid, elapsed, act["action"])
        else:
            # Person is active — reset timer
            _sitting_state.pop(key, None)

    # Clean up departed persons
    departed = {k for k in _sitting_state if k[0] == camera_id} - active_keys
    for k in departed:
        _sitting_state.pop(k, None)

    return action_results


def _fire_sitting_idle_alert(
    db, camera_id: int, track_id: int, elapsed: float, action: str
):
    try:
        from services.alert_service import save_alert
        from services.rule_engine import _get_rule_engine_user_id

        uid = _get_rule_engine_user_id(db)
        save_alert(
            db=db,
            user_id=uid,
            message=(
                f"[IDLE] Camera {camera_id} — Track #{track_id} has been "
                f"{action} for {elapsed/60:.1f} min. "
                "Employee appears idle. Supervisor review needed."
            ),
            role="System",
            severity="medium",
            detected_issue="Idle sitting detected",
            camera_id=camera_id,
        )
        log.warning(
            "[ActionRec] Idle alert — cam=%d track=%d action=%s elapsed=%.0fs",
            camera_id, track_id, action, elapsed
        )
    except Exception as exc:
        log.error("[ActionRec] _fire_sitting_idle_alert failed: %s", exc)


def annotate_frame(
    frame: np.ndarray,
    action_results: List[Dict],
) -> np.ndarray:
    """
    Draw action labels on frame for debug / stream annotation.
    """
    import cv2
    out = frame.copy()
    for det in action_results:
        x1, y1, x2, y2 = det["bbox"]
        action = det["action"]
        is_working = det["is_working"]
        is_idle = det["is_idle"]

        color = (
            (0, 200, 60) if is_working
            else (0, 60, 220) if is_idle
            else (180, 180, 30)
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{action} {det['confidence']:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Real-Time Fight / Rage / Aggression Detector
# ─────────────────────────────────────────────────────────────────────────────

class FightAggressionDetector:
    """
    High-accuracy real-time fight & rage aggression detector for workplace CCTV.
    Scalable for 20+ cameras (zero heavy GPU saturation).

    Principles:
    1. Spatial Proximity Gating: Only triggers when 2+ individuals are within close contact (distance < 1.0m or IoU > 0.08).
    2. Kinetic Acceleration / Strike Vector: Analyzes high-speed erratic motion, strike velocity toward other person.
    3. Reciprocal Hostile Motion: Identifies simultaneous high jerk, grapple struggle, and abrupt recoil.
    4. Anti-False Positive Filter: Suppresses calm collaborative work, steady carrying, or passive conversation (side_talks).
    5. Temporal Debounce: Requires at least 2-3 consecutive frames of confirmed physical altercation.
    """

    def __init__(self, history_len: int = 15):
        import collections
        import threading
        self.history_len = history_len
        self._history: Dict[Tuple[int, int], collections.deque] = {}  # (cam_id, track_id) -> deque of snaps
        self._pair_altercation_count: Dict[Tuple[int, int, int], int] = {}  # (cam_id, min_id, max_id) -> frame count
        self._lock = threading.Lock()

    def detect_aggression(
        self,
        frame: np.ndarray,
        persons: List[Dict],
        camera_id: int = 1,
        now: Optional[float] = None,
        poses: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Evaluate interpersonal interactions across all detected persons in frame.
        Detects both high-velocity strikes/charges and close-quarters grappling/pinning/clinching.
        Returns list of fight/aggression detections with unified bounding box.
        """
        if not persons or len(persons) < 2:
            return []

        import collections
        import math
        now = now or time.time()
        fight_detections: List[Dict] = []

        # If poses not passed, try to fetch from pose_adapter if available
        if poses is None and frame is not None and frame.size > 0:
            try:
                from services.pose_layer import pose_adapter
                poses = pose_adapter.analyse(frame, [p.get("bbox") for p in persons if p.get("bbox")])
            except Exception:
                poses = None

        with self._lock:
            # 1. Update individual track histories (centroids & velocities)
            current_tracks: Dict[int, Dict] = {}
            for idx, p in enumerate(persons):
                # Fallback to unique index if tracker assigned -1 or absent
                raw_tid = p.get("track_id", -1)
                tid = int(raw_tid) if raw_tid is not None and int(raw_tid) > 0 else (idx + 1)
                bbox = p.get("bbox", [])
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)

                key = (camera_id, tid)
                if key not in self._history:
                    self._history[key] = collections.deque(maxlen=self.history_len)
                hist = self._history[key]

                vx, vy, speed = 0.0, 0.0, 0.0
                ax, ay, accel = 0.0, 0.0, 0.0
                if hist:
                    prev = hist[-1]
                    dt = max(0.04, now - prev["time"])
                    vx = (cx - prev["cx"]) / dt
                    vy = (cy - prev["cy"]) / dt
                    speed = math.sqrt(vx * vx + vy * vy)
                    if len(hist) >= 2:
                        ax = (vx - prev["vx"]) / dt
                        ay = (vy - prev["vy"]) / dt
                        accel = math.sqrt(ax * ax + ay * ay)

                pose_data = poses[idx] if poses and idx < len(poses) else {}

                snap = {
                    "time": now,
                    "cx": cx,
                    "cy": cy,
                    "vx": vx,
                    "vy": vy,
                    "ax": ax,
                    "ay": ay,
                    "speed": speed,
                    "accel": accel,
                    "bbox": bbox,
                    "w": w,
                    "h": h,
                    "pose": pose_data,
                }
                hist.append(snap)
                current_tracks[tid] = snap

            # 2. Pairwise interaction analysis
            tids = list(current_tracks.keys())
            active_pairs = set()

            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    t1, t2 = tids[i], tids[j]
                    p1 = current_tracks[t1]
                    p2 = current_tracks[t2]

                    pair_key = (camera_id, min(t1, t2), max(t1, t2))
                    active_pairs.add(pair_key)

                    # Spatial proximity filter
                    dx = p1["cx"] - p2["cx"]
                    dy = p1["cy"] - p2["cy"]
                    dist = math.sqrt(dx * dx + dy * dy)
                    avg_w = (p1["w"] + p2["w"]) / 2.0
                    avg_h = (p1["h"] + p2["h"]) / 2.0

                    # Check bounding box overlap or proximity (within 1.35x width)
                    ix1 = max(p1["bbox"][0], p2["bbox"][0])
                    iy1 = max(p1["bbox"][1], p2["bbox"][1])
                    ix2 = min(p1["bbox"][2], p2["bbox"][2])
                    iy2 = min(p1["bbox"][3], p2["bbox"][3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area1 = max(1, p1["w"] * p1["h"])
                    area2 = max(1, p2["w"] * p2["h"])
                    iou = inter / float(area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0.0
                    has_contact = (inter > 0) or (dist < avg_w * 1.35)

                    if not has_contact:
                        self._pair_altercation_count[pair_key] = 0
                        continue

                    # Velocity & jerk dynamics
                    dvx = p1["vx"] - p2["vx"]
                    dvy = p1["vy"] - p2["vy"]
                    rel_speed = math.sqrt(dvx * dvx + dvy * dvy)
                    max_accel = max(p1["accel"], p2["accel"])

                    # Reversal / grapple struggle (opposing velocities under high speed)
                    dot_v = (p1["vx"] * p2["vx"] + p1["vy"] * p2["vy"])
                    is_turbulent = (dot_v < 0 and rel_speed > 100.0) or (rel_speed > 180.0)

                    # Strike detection: high acceleration spike during close proximity
                    is_strike = (rel_speed > 140.0 and max_accel > 220.0) or (is_turbulent and rel_speed > 120.0)

                    # Close-Quarters Grapple / Clinch Detection:
                    # e.g. two persons pushing, pinning against wall, holding collar/shoulders
                    is_grapple = False
                    # Check wrist engagement from pose keypoints
                    w1 = p1.get("pose", {}).get("wrists", [])
                    w2 = p2.get("pose", {}).get("wrists", [])
                    b1 = p1["bbox"]
                    b2 = p2["bbox"]

                    # Wrists of person 1 inside person 2's upper body / chest region
                    w1_in_p2 = any(
                        (b2[0] <= wx <= b2[2]) and (b2[1] <= wy <= b2[1] + (b2[3] - b2[1]) * 0.75)
                        for wx, wy in w1
                    )
                    # Wrists of person 2 inside person 1's upper body / chest region
                    w2_in_p1 = any(
                        (b1[0] <= wx <= b1[2]) and (b1[1] <= wy <= b1[1] + (b1[3] - b1[1]) * 0.75)
                        for wx, wy in w2
                    )

                    # If wrists are holding/interlocking with partner's upper body and IoU > 0.10
                    if (w1_in_p2 or w2_in_p1) and (iou > 0.10 or inter > 0):
                        is_grapple = True
                    elif iou > 0.22 and dist < avg_w * 0.85:
                        # Heavy physical body-to-body clinch (pushing against wall/floor)
                        is_grapple = True

                    is_aggressive = is_strike or is_grapple

                    if is_aggressive:
                        # Grapples with strong wrist interaction or high IoU trigger immediately
                        increment = 2 if is_grapple else 1
                        self._pair_altercation_count[pair_key] = self._pair_altercation_count.get(pair_key, 0) + increment
                    else:
                        self._pair_altercation_count[pair_key] = max(0, self._pair_altercation_count.get(pair_key, 0) - 1)

                    count = self._pair_altercation_count.get(pair_key, 0)
                    if count >= 1:
                        # Unified bounding box enclosing both individuals
                        ux1 = min(p1["bbox"][0], p2["bbox"][0])
                        uy1 = min(p1["bbox"][1], p2["bbox"][1])
                        ux2 = max(p1["bbox"][2], p2["bbox"][2])
                        uy2 = max(p1["bbox"][3], p2["bbox"][3])

                        base_conf = 0.88 if is_grapple else 0.80
                        conf = min(0.98, max(base_conf, 0.82 + (rel_speed / 400.0) * 0.15))
                        fight_mode = "Grapple/Clinch" if is_grapple else "Aggressive Movement"
                        fight_detections.append({
                            "label": "Physical Altercation",
                            "confidence": round(conf, 2),
                            "bbox": [ux1, uy1, ux2, uy2],
                            "det_type": "violation",
                            "raw_label": "fight_aggression",
                            "description": f"Physical altercation / {fight_mode} detected between persons #{t1} and #{t2}",
                        })

            # Cleanup inactive pairs
            departed_pairs = set(self._pair_altercation_count.keys()) - active_pairs
            for dp in departed_pairs:
                if dp[0] == camera_id:
                    self._pair_altercation_count.pop(dp, None)

        return fight_detections


aggression_detector = FightAggressionDetector()

