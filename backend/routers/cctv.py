"""
CCTV / IP Camera detection router  —  High-Speed Edition
=========================================================

Key improvements over v1:
  • Dual-pipeline inference: PPE/YOLO + Face recognition run simultaneously
    on every frame via concurrent thread-pool executors.
  • Camera reader aggressively flushes the capture buffer so you always get
    the NEWEST frame (no lag buildup).
  • Target 15 fps detection throughput (100 ms inter-frame).
  • Supports: MJPEG HTTP, RTSP, or still-JPEG polling.
  • Face recognition always active alongside PPE — detects known/unknown
    persons and overlays identity labels on the annotated frame.
  • Frame pre-scaling: resize to 640×480 before inference for max YOLO speed.
  • Adaptive backpressure: if inference is slower than capture, the pipeline
    drops intermediate frames rather than queuing them.
"""

import asyncio
import base64
import datetime
import json
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import cv2
import numpy as np
from auth_utils import decode_token
from config import settings
from database import User, UserConfig, get_db
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from routers.users import _parse_custom_ppe
from services import camera_manager  # server-managed stream registry
from services import face_service, yolo_service
from services.alert_service import save_alert
from sqlalchemy.orm import Session

router = APIRouter(tags=["cctv"])

# Three executors: more parallelism = lower latency under load
_yolo_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="cctv-yolo")
_face_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cctv-face")

_last_alert_time: dict = {}
_last_phone_alert_time: dict = {}

# ── Inference resolution: accuracy vs speed  ──────────────────────────────────
# 960×720 = high accuracy (recommended) | 640×480 = balanced | 480×360 = fastest
INFER_WIDTH = 960
INFER_HEIGHT = 720


# ─────────────────────────────────────────────────────────────────────────────
# Camera Reader — high-speed, always-fresh frame
# ─────────────────────────────────────────────────────────────────────────────


class CameraReader:
    """
    Background thread that continuously reads and discards all but the LATEST
    frame from the camera stream.  Only the freshest frame is kept in memory
    so the inference pipeline never processes stale data.

    Supports:
      • MJPEG/HTTP  (IP Webcam Android: http://ip:8080/video)
      • RTSP        (rtsp://user:pass@ip/stream)
      • JPEG poll   (IP Webcam: http://ip:8080/shot.jpg)
    """

    # Minimum time between consecutive cap.read() calls when the camera itself
    # is the bottleneck (avoids a tight spin that wastes CPU).
    _READ_INTERVAL = 0.03  # 30 ms → headroom for 30 fps cameras

    def __init__(self, url: str):
        self.url = url
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._error: Optional[str] = None
        self._fps = 0.0
        self._frame_count = 0
        self._last_frame_ts = 0.0
        self._reconnect_count = 0
        u = url.lower()
        self._is_shot = any(
            k in u for k in ("shot.jpg", "photo.jpg", "snap", "capture")
        )
        self._is_http = u.startswith("http://") or u.startswith("https://")
        self._is_rtsp = u.startswith("rtsp://") or u.startswith("rtsps://")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_with_reconnect, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def last_error(self) -> Optional[str]:
        return self._error

    def fps(self) -> float:
        return round(self._fps, 1)

    def metrics(self) -> dict:
        return {
            "fps": self.fps(),
            "last_frame_at": (
                datetime.datetime.utcfromtimestamp(self._last_frame_ts).isoformat()
                + "Z"
                if self._last_frame_ts
                else None
            ),
            "reconnect_count": self._reconnect_count,
            "last_error": self._error,
        }

    # ── Internal read loop ────────────────────────────────────────────────────

    def _read_loop(self):
        if self._is_shot:
            self._poll_jpeg_loop()
        elif self._is_rtsp:
            self._rtsp_loop()  # OpenCV/FFmpeg handles RTSP well
        else:
            self._mjpeg_http_loop()  # urllib handles HTTP MJPEG reliably

    def _run_with_reconnect(self):
        """One reader lifecycle with no concurrent reconnect workers."""
        delay = 1.0
        while self._running:
            self._read_loop()
            if not self._running:
                break
            self._reconnect_count += 1
            time.sleep(delay)
            delay = min(30.0, delay * 2)

    # ── HTTP MJPEG reader (urllib — no FFmpeg) ────────────────────────────────

    def _mjpeg_http_loop(self):
        """
        Pure-Python MJPEG-over-HTTP reader.
        Fixes:
          - Buffer trim no longer discards in-progress JPEG data
          - Large chunk reads (65 KB) so high-res frames arrive fast
          - Content-Length aware: reads exact frame bytes when header present
          - Non-multipart fallback polls /shot.jpg equivalent
        """
        import urllib.error
        import urllib.request

        print(f"[CCTV] Connecting to HTTP stream: {self.url}")
        failures = 0
        MAX_FAIL = 8
        CHUNK = 65536  # 64 KB per read — covers most MJPEG frames in 1–3 reads
        t_fps = time.time()
        fps_frames = 0

        while self._running:
            try:
                req = urllib.request.Request(
                    self.url, headers={"User-Agent": "Mozilla/5.0"}
                )

                # Bypass SSL validation for local cameras with self-signed certs
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

                with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:  # nosec B310
                    ct = resp.headers.get("Content-Type", "")
                    print(f"[CCTV] Connected — Content-Type: {ct}")

                    if "multipart" in ct:
                        buf = b""
                        while self._running:
                            chunk = resp.read(CHUNK)
                            if not chunk:
                                print("[CCTV] Stream ended by server — reconnecting")
                                break
                            buf += chunk

                            # Extract ALL complete JPEGs from the current buffer
                            while self._running:
                                soi = buf.find(b"\xff\xd8")  # JPEG Start-Of-Image
                                if soi == -1:
                                    # No JPEG started yet — discard everything before
                                    # but keep last 3 bytes in case SOI is split
                                    if len(buf) > 3:
                                        buf = buf[-3:]
                                    break

                                eoi = buf.find(
                                    b"\xff\xd9", soi + 2
                                )  # JPEG End-Of-Image
                                if eoi == -1:
                                    # JPEG started but not finished yet —
                                    # Keep everything from SOI onward, read more data
                                    if soi > 0:
                                        buf = buf[soi:]  # discard pre-SOI garbage ONLY
                                    break

                                # We have a complete JPEG: [soi .. eoi+2)
                                jpg_data = buf[soi : eoi + 2]
                                buf = buf[eoi + 2 :]  # continue AFTER this JPEG

                                arr = np.frombuffer(jpg_data, np.uint8)
                                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    frame = cv2.resize(
                                        frame, (INFER_WIDTH, INFER_HEIGHT)
                                    )
                                    with self._lock:
                                        self._frame = frame
                                        self._frame_count += 1
                                    fps_frames += 1
                                    elapsed = time.time() - t_fps
                                    if elapsed >= 2.0:
                                        self._fps = fps_frames / elapsed
                                        fps_frames = 0
                                        t_fps = time.time()
                                    failures = 0  # reset on any good frame
                        # end of stream from server — loop outer while to reconnect
                        failures += 1
                        if failures >= MAX_FAIL:
                            self._error = (
                                f"IP Webcam stream at {self.url} kept disconnecting.\n"
                                f"Ensure the phone screen is ON and IP Webcam is streaming."
                            )
                            return
                        time.sleep(0.5)

                    elif "image/jpeg" in ct or "image/jpg" in ct:
                        # Single-shot JPEG endpoint — read one image and return
                        data = resp.read()
                        arr = np.frombuffer(data, np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
                            with self._lock:
                                self._frame = frame
                                self._frame_count += 1
                        # Poll: loop outer while to reconnect and fetch next frame
                        time.sleep(self._READ_INTERVAL)

                    else:
                        # Unknown content type — log and try JPEG polling fallback
                        print(f"[CCTV] Unknown Content-Type '{ct}' — polling as JPEG")
                        # Try reading raw bytes and decoding as JPEG
                        data = resp.read(1 * 1024 * 1024)  # max 1 MB
                        if data:
                            soi = data.find(b"\xff\xd8")
                            eoi = data.rfind(b"\xff\xd9")
                            if soi != -1 and eoi > soi:
                                arr = np.frombuffer(data[soi : eoi + 2], np.uint8)
                                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    frame = cv2.resize(
                                        frame, (INFER_WIDTH, INFER_HEIGHT)
                                    )
                                    with self._lock:
                                        self._frame = frame
                                        self._frame_count += 1
                                    failures = 0
                        time.sleep(self._READ_INTERVAL)

            except urllib.error.URLError as e:
                failures += 1
                reason = str(e.reason) if hasattr(e, "reason") else str(e)
                print(f"[CCTV] HTTP error ({failures}/{MAX_FAIL}): {reason}")
                if failures >= MAX_FAIL:
                    self._error = (
                        f"Cannot reach camera at {self.url}\n"
                        f"Error: {reason}\n"
                        f"• Open {self.url} in your PC browser — if it doesn't load, it's a network issue\n"
                        f"• Make sure phone and PC are on the SAME WiFi\n"
                        f"• IP Webcam app must be running and showing the server URL"
                    )
                    return
                time.sleep(1.5)
            except Exception as e:
                failures += 1
                print(
                    f"[CCTV] Unexpected error ({failures}/{MAX_FAIL}): {type(e).__name__}: {e}"
                )
                if failures >= MAX_FAIL:
                    self._error = f"Stream error after {MAX_FAIL} retries: {e}"
                    return
                time.sleep(1.0)

    # ── RTSP loop (OpenCV/FFmpeg) ─────────────────────────────────────────────

    def _rtsp_loop(self):
        """OpenCV loop for RTSP streams — FFmpeg handles RTSP well."""
        self._opencv_loop()

    # ── OpenCV generic loop ───────────────────────────────────────────────────

    def _opencv_loop(self):
        """Generic OpenCV VideoCapture loop (RTSP / fallback)."""
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        if not cap.isOpened():
            self._error = (
                f"OpenCV could not open stream: {self.url}\n"
                f"For RTSP: check credentials and port.\n"
                f"For HTTP: ensure the IP Webcam app is running."
            )
            return

        failures = 0
        MAX_FAIL = 60
        t_fps = time.time()
        fps_frames = 0

        while self._running:
            t0 = time.time()
            ret, frame = cap.read()

            if ret and frame is not None:
                failures = 0
                for _ in range(2):
                    ok, f = cap.read()
                    if ok and f is not None:
                        frame = f
                frame_resized = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
                with self._lock:
                    self._frame = frame_resized
                    self._frame_count += 1
                fps_frames += 1
                elapsed = time.time() - t_fps
                if elapsed >= 2.0:
                    self._fps = fps_frames / elapsed
                    fps_frames = 0
                    t_fps = time.time()
            else:
                failures += 1
                if failures >= MAX_FAIL:
                    self._error = (
                        f"Camera stream lost after {MAX_FAIL} failures — "
                        f"check URL: {self.url}"
                    )
                    break
                time.sleep(0.05)

            spent = time.time() - t0
            if spent < self._READ_INTERVAL:
                time.sleep(self._READ_INTERVAL - spent)

        cap.release()

    def _poll_jpeg_loop(self):
        """Polling loop for single-frame endpoints like /shot.jpg."""
        import ssl
        import urllib.request

        failures = 0
        MAX_FAIL = 20

        while self._running:
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

                with urllib.request.urlopen(self.url, timeout=3, context=ctx) as resp:
                    data = resp.read()
                arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
                    with self._lock:
                        self._frame = frame
                        self._frame_count += 1
                    failures = 0
            except Exception as e:
                failures += 1
                if failures >= MAX_FAIL:
                    self._error = f"JPEG poll failed {MAX_FAIL} times: {e}"
                    break
            time.sleep(self._READ_INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
# Dual inference: PPE + Face combined annotation
# ─────────────────────────────────────────────────────────────────────────────


def _run_combined_inference(
    frame: np.ndarray,
    role: str,
    user_id: int,
    db: Session,
    det_filters: Optional[List[str]],
    no_phone_zone: bool,
    frame_idx: int,
    enable_face: bool,
) -> Dict:
    """
    Run PPE detection and optional face recognition simultaneously.
    Returns a merged result dict with a single annotated_frame.
    """
    ppe_result = {}
    face_result = {}

    # --- PPE / YOLO detection ------------------------------------------------
    if role != "Home":
        ppe_result = yolo_service.process_frame_numpy(
            frame,
            role,
            det_filters,
            no_phone_zone,
            frame_idx,
        )
    else:
        # Home role → PPE pipeline still runs (for phone detection)
        ppe_result = yolo_service.process_frame_numpy(
            frame,
            "Home",
            [],  # no PPE requirements
            no_phone_zone,
            frame_idx,
        )

    # --- Face recognition (numpy path - no double encode) --------------------
    if enable_face:
        face_result = face_service.process_face_numpy(frame, user_id, db)

    # --- Merge annotations ---------------------------------------------------
    # Start from PPE-annotated frame (or raw if PPE did nothing)
    ann_b64 = ppe_result.get("annotated_frame")
    if ann_b64 and face_result and face_result.get("faces"):
        # Decode PPE-annotated frame and overlay face boxes on top
        try:
            raw = ann_b64.split(",")[1] if "," in ann_b64 else ann_b64
            arr = np.frombuffer(base64.b64decode(raw), np.uint8)
            ann_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if ann_frame is not None:
                ann_frame = _overlay_faces(ann_frame, face_result)
                _, buf = cv2.imencode(".jpg", ann_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                ann_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        except Exception:
            pass  # keep PPE-only annotation on any error

    elif face_result and face_result.get("annotated_frame") and not ann_b64:
        ann_b64 = face_result.get("annotated_frame")

    # --- Merge compliance / alerts -------------------------------------------
    is_compliant = ppe_result.get("is_compliant", True)
    alert_message = ppe_result.get("alert_message")
    severity = ppe_result.get("severity")

    face_unknown = face_result.get("unknown_detected", False)
    if enable_face and face_unknown:
        is_compliant = False
        face_alert = face_result.get("alert_message", "⚠️ Unknown person detected!")
        alert_message = (
            f"{alert_message} | {face_alert}" if alert_message else face_alert
        )
        severity = "critical"

    merged = {**ppe_result}
    merged["annotated_frame"] = ann_b64
    merged["is_compliant"] = is_compliant
    merged["alert_message"] = alert_message if not is_compliant else None
    merged["severity"] = severity if not is_compliant else None
    merged["face_result"] = (
        {
            "faces": face_result.get("faces", []),
            "unknown_detected": face_unknown,
            "face_count": len(face_result.get("faces", [])),
            "recognized_employees": face_result.get("recognized_employees", {}),
        }
        if enable_face
        else None
    )
    merged["source"] = "cctv"
    return merged


def _overlay_faces(frame: np.ndarray, face_result: Dict) -> np.ndarray:
    """Draw face recognition boxes on top of an already-annotated PPE frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for face in face_result.get("faces", []):
        x1, y1, x2, y2 = face["bbox"]
        is_unknown = face.get("is_unknown", True)
        color = (50, 50, 255) if is_unknown else (50, 220, 50)
        thick = 2

        # Draw face bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

        # Identity label pill
        name = face["label"]
        conf_pct = f"{face['confidence']:.0%}"
        label = f"{'❌ UNKNOWN' if is_unknown else '✓ ' + name}  {conf_pct}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        (lw, lh), _ = cv2.getTextSize(label, font, scale, 2)
        # Background pill above face box
        cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 8, y1), (0, 0, 0), -1)
        cv2.putText(
            annotated, label, (x1 + 4, y1 - 4), font, scale, color, 2, cv2.LINE_AA
        )

    if face_result.get("unknown_detected"):
        # Red banner for unknown person (below the existing PPE banner)
        cv2.rectangle(annotated, (0, 46), (w, 84), (0, 0, 180), -1)
        cv2.addWeighted(
            annotated[46:84].copy(), 0.0, annotated[46:84], 1.0, 0, annotated[46:84]
        )
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 46), (w, 84), (30, 0, 160), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        cv2.putText(
            annotated,
            "  🔴 FACE ALERT: UNKNOWN PERSON DETECTED",
            (8, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 220, 220),
            2,
            cv2.LINE_AA,
        )

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Auth helper
# ─────────────────────────────────────────────────────────────────────────────


async def _get_user(token: str, db: Session) -> Optional[User]:
    payload = decode_token(token)
    if not payload:
        return None
    return db.query(User).filter(User.email == payload.get("sub")).first()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────


@router.websocket("/ws/detect-cctv")
async def cctv_detection_websocket(websocket: WebSocket):
    """
    High-speed CCTV detection WebSocket.

    Handshake JSON (client → server):
    {
      "token":        "<jwt_token>",
      "camera_url":   "http://10.62.212.243:8080/video",
      "filters":      ["NO-Hardhat", ...],   // optional
      "no_phone_zone": true,                 // optional
      "enable_face":  true                   // optional — enables face recognition
    }

    After handshake, client may send:
    {
      "filters":       [...],
      "no_phone_zone": bool,
      "enable_face":   bool
    }
    """
    await websocket.accept()
    db: Session = next(get_db())
    # `camera` is only set in legacy (camera_url) mode.
    # In managed (camera_id) mode all frame reads go through camera_manager.
    camera: Optional[CameraReader] = None
    managed_camera_id: Optional[int] = None  # set when using camera_manager

    try:
        # ── Handshake ────────────────────────────────────────────────────────
        raw = await websocket.receive_text()
        auth_data = json.loads(raw)
        user = await _get_user(auth_data.get("token", ""), db)

        if not user:
            await websocket.send_json({"error": "Unauthorized"})
            await websocket.close()
            return

        role = user.role or "Home"

        # ── Resolve camera source ─────────────────────────────────────────────
        # Prefer camera_id (managed mode) over camera_url (legacy mode)
        _cam_id_raw = auth_data.get("camera_id")
        camera_url = auth_data.get("camera_url", "").strip()

        if _cam_id_raw is not None:
            # ── MANAGED MODE: subscriber reads from camera_manager ───────────
            managed_camera_id = int(_cam_id_raw)
            if not camera_manager.is_running(managed_camera_id):
                await websocket.send_json(
                    {
                        "error": (
                            f"Camera {managed_camera_id} is not currently streaming. "
                            f"Start it via POST /cameras/{managed_camera_id}/restart or "
                            f"check that the camera is registered and online."
                        )
                    }
                )
                await websocket.close()
                return
            display_url = f"managed:{managed_camera_id}"
            print(f"\n[CCTV-v3] ===== SUBSCRIBER SESSION =====")
            print(
                f"[CCTV-v3] User: {user.name} | Role: {role} | CameraID: {managed_camera_id}"
            )

        elif camera_url:
            # ── LEGACY MODE: open a dedicated CameraReader for this session ──
            display_url = camera_url
            print(f"\n[CCTV-v2] ===== LEGACY SESSION =====")
            print(f"[CCTV-v2] User: {user.name} | Role: {role} | URL: {camera_url}")

        else:
            await websocket.send_json({"error": "camera_id or camera_url is required"})
            await websocket.close()
            return

        # PPE filters
        handshake_filters = list(auth_data.get("filters", []))
        if role == "None" and not handshake_filters:
            db_config = (
                db.query(UserConfig).filter(UserConfig.user_id == user.id).first()
            )
            if db_config:
                handshake_filters = _parse_custom_ppe(db_config)

        state = {
            "filters": handshake_filters,
            "no_phone_zone": bool(auth_data.get("no_phone_zone", True)),
            "enable_face": bool(auth_data.get("enable_face", True)),
            "frame_count": 0,
            "alive": True,
            "cam_fps": 0.0,
        }

        await websocket.send_json(
            {
                "status": "connected",
                "role": role,
                "user": user.name,
                "camera_source": display_url,
                "face_enabled": state["enable_face"],
                "active_filters": state["filters"],
                "mode": "managed" if managed_camera_id else "legacy",
            }
        )

        # ── Start camera reader (legacy mode only) ───────────────────────────
        if managed_camera_id is None:
            # Legacy: open a private CameraReader for this WebSocket session
            camera = CameraReader(camera_url)
            camera.start()

            # Ramp-up: wait up to 8 s for first frame
            for _ in range(80):
                if not state["alive"]:
                    return
                if camera.latest_frame() is not None:
                    break
                if camera.last_error():
                    break
                await asyncio.sleep(0.1)

            if camera.last_error():
                await websocket.send_json({"error": camera.last_error()})
                await websocket.close()
                return

            if camera.latest_frame() is None:
                await websocket.send_json(
                    {
                        "error": (
                            f"No frames received from {camera_url}\n\n"
                            f"Quick fix: open  {camera_url}  in your PC browser.\n"
                            f"• If the video loads → reconnect in OccuSafe\n"
                            f"• If it doesn't load → phone and PC are on different networks\n"
                            f"• Make sure IP Webcam app shows 'Server started'"
                        )
                    }
                )
                await websocket.close()
                return

        # In managed mode, the frame is already flowing — no ramp-up needed.

        loop = asyncio.get_event_loop()

        # ── Coroutine A: receive config updates from client ──────────────────
        async def recv_messages():
            while state["alive"]:
                try:
                    raw_msg = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.5
                    )
                    msg = json.loads(raw_msg)
                    if "filters" in msg:
                        state["filters"] = msg["filters"]
                    if "no_phone_zone" in msg:
                        state["no_phone_zone"] = bool(msg["no_phone_zone"])
                    if "enable_face" in msg:
                        state["enable_face"] = bool(msg["enable_face"])
                    if msg.get("stop"):
                        state["alive"] = False
                except asyncio.TimeoutError:
                    pass  # normal — keep looping
                except WebSocketDisconnect:
                    state["alive"] = False
                    return

        # ── Coroutine B: grab → infer → send at max speed ───────────────────
        async def process_frames():
            MIN_INTERVAL_MS = 50  # hard floor: don't send faster than 20 fps
            last_sent = time.time()

            while state["alive"]:
                now = time.time()

                # ── Get latest frame — managed or legacy ─────────────────────
                if managed_camera_id is not None:
                    # Subscriber mode: read from server-managed registry
                    cam_error = camera_manager.get_reader_error(managed_camera_id)
                    if cam_error:
                        await websocket.send_json({"error": cam_error})
                        state["alive"] = False
                        return
                    frame = camera_manager.get_latest_frame(managed_camera_id)
                    cam_fps = camera_manager.get_reader_fps(managed_camera_id)
                else:
                    # Legacy mode: read from private CameraReader
                    if camera.last_error():
                        await websocket.send_json({"error": camera.last_error()})
                        state["alive"] = False
                        return
                    frame = camera.latest_frame()
                    cam_fps = camera.fps()

                if frame is None:
                    await asyncio.sleep(0.015)  # wait for first frame
                    continue

                # Enforce minimum send interval so WebSocket isn't flooded
                elapsed = now - last_sent
                if elapsed < MIN_INTERVAL_MS / 1000:
                    await asyncio.sleep((MIN_INTERVAL_MS / 1000) - elapsed)
                    continue

                state["frame_count"] += 1
                fn = state["frame_count"]
                det_filters = list(state["filters"]) if state["filters"] else None
                nph = state["no_phone_zone"]
                ef = state["enable_face"]

                try:
                    # Run combined PPE + Face inference in thread pool
                    result = await loop.run_in_executor(
                        _yolo_executor,
                        lambda fr=frame, fi=fn, df=det_filters, nz=nph, efa=ef: (
                            _run_combined_inference(
                                fr, role, user.id, db, df, nz, fi, efa
                            )
                        ),
                    )

                    response = {
                        "annotated_frame": result.get("annotated_frame"),
                        "detections": result.get("detections", []),
                        "is_compliant": result.get("is_compliant", True),
                        "missing_items": result.get("missing_items", []),
                        "violations_count": result.get("violations_count", 0),
                        "persons_count": result.get("persons_count", 0),
                        "alert_message": result.get("alert_message"),
                        "severity": result.get("severity"),
                        "frame_count": fn,
                        "persons": result.get("persons", []),
                        "model_mode": result.get("model_mode", "cctv"),
                        "active_filters": state["filters"],
                        "phone_status": result.get("phone_status", "safe"),
                        "phone_detected": result.get("phone_detected", False),
                        "face_result": result.get("face_result"),
                        "cam_fps": cam_fps,
                        "source": "cctv",
                    }

                    # ── PPE alert save ─────────────────────────────────────
                    if not result.get("is_compliant") and result.get("alert_message"):
                        uid = user.id
                        now_t = time.time()
                        dets = result.get("detections", [])
                        # Use person confidence as fallback so violations without
                        # a PPE bbox (College, Gloves, Goggles) still trigger alerts.
                        persons_c = [
                            p.get("confidence", 0) for p in result.get("persons", [])
                        ]
                        dets_c = [d.get("confidence", 0) for d in dets]
                        top_conf = max(persons_c + dets_c, default=0.5)
                        cooldown = settings.ALERT_COOLDOWN
                        if (
                            top_conf >= settings.MIN_VIOLATION_CONF
                            and (now_t - _last_alert_time.get(uid, 0)) > cooldown
                        ):
                            _last_alert_time[uid] = now_t
                            missing = result.get("missing_items", [])
                            save_alert(
                                db=db,
                                user_id=uid,
                                message=result["alert_message"],
                                role=role,
                                severity=result["severity"],
                                detected_issue=(
                                    ", ".join(missing)
                                    if missing
                                    else result["alert_message"]
                                ),
                                confidence=round(top_conf, 3),
                                snapshot_b64=result.get("snapshot_b64"),
                            )
                            response["alert_saved"] = True

                    # ── Phone alert save ───────────────────────────────────
                    if result.get("phone_severity") == "high" and result.get(
                        "phone_alert"
                    ):
                        uid = user.id
                        now_t = time.time()
                        if now_t - _last_phone_alert_time.get(uid, 0) > 15:
                            _last_phone_alert_time[uid] = now_t
                            save_alert(
                                db=db,
                                user_id=uid,
                                message=result["phone_alert"],
                                role=role,
                                severity="high",
                                detected_issue=result["phone_alert"],
                                confidence=0.85,
                                snapshot_b64=result.get("snapshot_b64"),
                            )
                            response["phone_alert_saved"] = True

                    # ── Face unknown alert save ────────────────────────────
                    if result.get("face_result", {}) and result["face_result"].get(
                        "unknown_detected"
                    ):
                        uid = user.id
                        now_t = time.time()
                        if now_t - _last_alert_time.get(f"face_{uid}", 0) > 20:
                            _last_alert_time[f"face_{uid}"] = now_t
                            save_alert(
                                db=db,
                                user_id=uid,
                                message="Unknown person detected via CCTV",
                                role=role,
                                severity="critical",
                                detected_issue="Unknown face",
                                confidence=0.90,
                                snapshot_b64=result.get("snapshot_b64"),
                            )
                            response["face_alert_saved"] = True

                    # ── Attendance auto clock-in on face match ─────────────
                    # Fires for ANY camera that recognizes a registered employee.
                    # handle_face_match already deduplicates within the same day.
                    face_res = result.get("face_result") or {}
                    recognized = face_res.get("recognized_employees", {})
                    if recognized and ef:
                        from services.attendance_service import \
                            handle_face_match as _attn_hook

                        _cam_id_for_attn = managed_camera_id  # None in legacy mode
                        for _emp_id, _conf in recognized.items():
                            _attn_hook(
                                camera_id=_cam_id_for_attn,
                                employee_id=_emp_id,
                                confidence=_conf,
                            )

                    await websocket.send_json(response)
                    last_sent = time.time()

                except WebSocketDisconnect:
                    state["alive"] = False
                    return
                except Exception as e:
                    if state["alive"]:
                        print(f"[CCTV-v2] Frame#{fn} error: {type(e).__name__}: {e}")
                    state["alive"] = False
                    return

        # ── Run both coroutines concurrently ─────────────────────────────────
        try:
            await asyncio.gather(recv_messages(), process_frames())
        except WebSocketDisconnect:
            print("[CCTV-v2] Client disconnected cleanly")
        except Exception as err:
            print(f"[CCTV-v2] Session error: {err}")
        finally:
            state["alive"] = False

    except WebSocketDisconnect:
        print("[CCTV] Disconnected during handshake")
    except Exception as e:
        print(f"[CCTV] Fatal error: {type(e).__name__}: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        # Only stop the reader in legacy mode — managed readers stay alive.
        if camera is not None:
            camera.stop()
        db.close()
        mode = "managed" if managed_camera_id else "legacy"
        print(f"[CCTV] Session closed ({mode}) — resources freed")
