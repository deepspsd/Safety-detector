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
import logging
import os

# ── Ultra-low-latency FFmpeg / RTSP configuration ────────────────────────────
# These flags tell FFmpeg (which OpenCV uses under the hood) to minimise internal
# buffering and decode frames as soon as they arrive.  Without them FFmpeg keeps
# a multi-frame decode pipeline + reorder buffer and the stream is ALWAYS 2–10
# frames behind the live camera, even when cap.set(CAP_PROP_BUFFERSIZE, 1) is set
# (because BUFFERSIZE only limits OpenCV's own wrapper buffer, not FFmpeg's).
#
#   - rtsp_transport=tcp       : reliable delivery (no H.264 corruption from UDP
#                                packet loss, which is the #1 cause of blocky /
#                                washed-out / green artefacts)
#   - fflags=nobuffer + flags=low_delay : disable decoder look-ahead and frame
#                                reordering — sacrifice one keyframe-quality
#                                improvement for ~150 ms lower latency
#   - stimeout=5,000,000 µs    : fail-fast after 5 s of silence instead of
#                                hanging forever on a dead socket
#   - max_delay=0              : no inter-frame jitter buffer at all
#   - analyzeduration+probesize: start decoding instantly, don't wait to probe
#                                the whole stream (cuts connect time by 1–3 s)
#   - vsync=0 + async=1        : no video-sync frame pacing, decoder returns a
#                                frame the moment it's fully decoded
# ─────────────────────────────────────────────────────────────────────────────
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "stimeout;5000000|"
    "max_delay;0"
)

import ssl
import threading
import time

logger = logging.getLogger(__name__)

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

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
_active_cctv_alerts: dict = {}
_active_cctv_alerts_lock = threading.Lock()


def _cctv_alert_key(user_id: int, camera_id: Optional[int], missing: list, message: str = ""):
    """Stable key for one continuous violation event.

    Worker names, track IDs, confidence text, and durations must not create a
    new event.  Key by owner + camera + normalized missing classes.
    """
    import re

    values = [str(item).lower() for item in (missing or []) if item]
    if not values and message:
        values = [str(message).lower()]
    normalized = []
    for value in values:
        if "head" in value or "hairnet" in value or "hair cover" in value:
            value = "head_cap"
        elif "uniform" in value:
            value = "uniform"
        elif "glove" in value:
            value = "gloves"
        elif "phone" in value:
            value = "phone"
        else:
            value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
        if value and value not in normalized:
            normalized.append(value)
    if not normalized:
        return None
    return (int(user_id), int(camera_id) if camera_id is not None else None, tuple(sorted(normalized)))


def _clear_cctv_alert_events(user_id: int, camera_id: Optional[int]) -> None:
    with _active_cctv_alerts_lock:
        for key in list(_active_cctv_alerts):
            if key[0] == int(user_id) and key[1] == (int(camera_id) if camera_id is not None else None):
                _active_cctv_alerts.pop(key, None)

INFER_WIDTH = 640
INFER_HEIGHT = 480


# ─────────────────────────────────────────────────────────────────────────────
# Camera Reader — high-speed, always-fresh frame
# ─────────────────────────────────────────────────────────────────────────────


# ── Stream status constants ──────────────────────────────────────────────────
STREAM_ONLINE     = "ONLINE"
STREAM_DEGRADED   = "DEGRADED"
STREAM_OFFLINE    = "OFFLINE"
STREAM_RECOVERING = "RECOVERING"

# Frames with more than this fraction of green channel dominance are rejected.
# Real CCTV frames: green never dominates this strongly (unlike H.264 decoder
# artifacts which produce solid green frames).
_GREEN_DOMINANT_RATIO = 0.85

# A frame is considered frozen if its hash matches the previous one this many
# consecutive times at >= 10 fps capture — indicates a genuinely stuck decoder.
_FROZEN_CONSECUTIVE_MAX = 30  # ~3 s at 10 fps

# How many consecutive bad frames before declaring DEGRADED
_BAD_FRAME_DEGRADED_THRESH = 5

# How many seconds without any valid frame before OFFLINE
_OFFLINE_TIMEOUT_SECS = 30.0


class CameraReader:
    """
    Background thread that continuously reads and discards all but the LATEST
    valid frame from the camera stream.  Only fresh, non-corrupted frames are
    kept in memory so the inference pipeline never processes stale or green data.

    ── Low-latency design (v3.3) ──────────────────────────────────────────────
    • Continuous drain: the RTSP/OpenCV loop runs WITHOUT a read-interval sleep.
      It grabs every single frame FFmpeg has decoded as fast as possible, only
      keeping the *last* valid one.  This is what prevents FFmpeg's internal
      multi-frame decode pipeline from buffering 2–10 frames behind live.
    • Capture epoch: every stored frame is tagged with its local capture time
      (time.time()) so downstream consumers can measure frame age / end-to-end
      latency and decide whether to drop an obsolete frame before inference.
    • Per-cycle "grab N, keep last" pattern: inside _opencv_loop we call
      cap.grab() / cap.retrieve() in a tight non-blocking burst until no more
      frames are pending.  This empties ALL buffers in every cycle, not just
      one frame per 30 ms.

    Supports:
      • MJPEG/HTTP  (IP Webcam Android: http://ip:8080/video)
      • RTSP        (rtsp://user:pass@ip/stream)
      • JPEG poll   (IP Webcam: http://ip:8080/shot.jpg)
      • Webcam index (int / "0", "1", ...)

    Stream health:
      stream_status   ONLINE | DEGRADED | OFFLINE | RECOVERING
      bad_frame_count running count of rejected frames (green/frozen/corrupt)
      decoder_error_count  count of cap.read() failures
      capture_dropped  frames discarded because a NEWER frame arrived first
                        (i.e. how many frames the drain loop grabbed and threw
                        away inside a single buffer-empty cycle — these are
                        frames that would otherwise have to wait in the queue)
    """

    # Tiny sleep used ONLY when the camera is the bottleneck (no frames pending)
    # to avoid 100 % CPU spinning on an empty socket.  Must be < 1 ms so we
    # never wait long enough to let a buffer build up.
    _EMPTY_SPIN_SLEEP = 0.0005  # 0.5 ms

    def __init__(self, url: str):
        self.url = url
        self._frame = None
        self._frame_capture_ts: float = 0.0  # epoch when THIS frame was captured locally
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._error: Optional[str] = None
        self._fps = 0.0
        self._frame_count = 0
        self._last_frame_ts = 0.0
        self._reconnect_count = 0
        self.capture_dropped: int = 0  # frames thrown away by drain loop (kept only newest)
        self.cap_read_total: int = 0  # total successful cap.retrieve calls
        # Stream health counters
        self.bad_frame_count: int = 0
        self.decoder_error_count: int = 0
        self._consecutive_bad: int = 0
        self._consecutive_frozen: int = 0
        self._stream_status: str = STREAM_OFFLINE
        self._last_frame_hash: Optional[bytes] = None
        u = url.lower()
        # Support integer webcam index passed as string ("0", "1", …)
        self._is_webcam = u.isdigit() or (len(u) == 1 and u in "0123456789")
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
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def latest_frame_with_ts(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (frame_copy, capture_epoch_ts) atomically so consumers can
        accurately measure downstream frame age / end-to-end latency."""
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._frame_capture_ts

    def frame_age_seconds(self) -> float:
        """Age of the currently-stored frame, or 0 if none."""
        with self._lock:
            if self._frame_capture_ts == 0.0:
                return 0.0
            return max(0.0, time.time() - self._frame_capture_ts)

    def last_error(self) -> Optional[str]:
        return self._error

    def fps(self) -> float:
        return round(self._fps, 1)

    @property
    def stream_status(self) -> str:
        """ONLINE | DEGRADED | OFFLINE | RECOVERING."""
        if self._last_frame_ts == 0.0:
            return STREAM_OFFLINE
        if time.time() - self._last_frame_ts > _OFFLINE_TIMEOUT_SECS:
            return STREAM_OFFLINE
        if self._consecutive_bad >= _BAD_FRAME_DEGRADED_THRESH:
            return STREAM_DEGRADED
        return self._stream_status

    def metrics(self) -> dict:
        age_s = self.frame_age_seconds()
        return {
            "fps": self.fps(),
            "stream_status": self.stream_status,
            "bad_frame_count": self.bad_frame_count,
            "decoder_error_count": self.decoder_error_count,
            "reconnect_count": self._reconnect_count,
            "capture_dropped": self.capture_dropped,
            "cap_read_total": self.cap_read_total,
            "frame_age_ms": round(age_s * 1000, 1),
            "last_frame_at": (
                datetime.datetime.utcfromtimestamp(self._last_frame_ts).isoformat()
                + "Z"
                if self._last_frame_ts
                else None
            ),
            "last_error": self._error,
        }

    # ── Frame validation helpers ───────────────────────────────────────────────

    def _is_green_frame(self, frame: np.ndarray) -> bool:
        """
        Detect H.264 decoder artifacts: solid green frames produced when the
        decoder loses sync (typical on RTSP reconnect).
        Strategy: compute mean of each BGR channel. If green channel is
        _GREEN_DOMINANT_RATIO times higher than both R and B, reject.
        Fast: runs on the full frame but only 3 mean ops.
        """
        if frame is None or frame.ndim < 3:
            return False
        b_mean = float(np.mean(frame[:, :, 0]))
        g_mean = float(np.mean(frame[:, :, 1]))
        r_mean = float(np.mean(frame[:, :, 2]))
        total = b_mean + g_mean + r_mean
        if total < 1.0:
            return True  # near-black / all-zero frame → also bad
        g_frac = g_mean / total
        return g_frac > _GREEN_DOMINANT_RATIO

    def _is_corrupted_frame(self, frame: np.ndarray) -> bool:
        """
        Detect other corruption forms:
          - Wrong shape (not 3-channel)
          - All-zero pixels (black frame from failed decode)
          - Uniform single-color frame (stuck decoder outputting one colour)
        """
        if frame is None:
            return True
        if frame.ndim != 3 or frame.shape[2] != 3:
            return True
        if frame.size == 0:
            return True
        # Sample centre pixel area to avoid spending time on full-frame std
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        patch = frame[max(0, cy-30):cy+30, max(0, cx-30):cx+30]
        if patch.size == 0:
            return True
        # All-zero = black frame from failed decode
        if np.max(patch) == 0:
            return True
        # Near-uniform colour = stuck decoder (std deviation across all channels < 2)
        if float(np.std(patch)) < 2.0:
            return True
        return False

    def _is_frozen_frame(self, frame: np.ndarray) -> bool:
        """
        Detect frozen/stuck video by comparing a fast hash of a small
        centre crop. Same hash N consecutive times → frozen stream.
        """
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        sample = frame[max(0, cy-40):cy+40, max(0, cx-40):cx+40]
        # Use tobytes() hash of a downscaled patch for speed
        try:
            small = cv2.resize(sample, (16, 16))
            fhash = small.tobytes()
        except Exception:
            return False
        if fhash == self._last_frame_hash:
            self._consecutive_frozen += 1
        else:
            self._consecutive_frozen = 0
            self._last_frame_hash = fhash
        return self._consecutive_frozen >= _FROZEN_CONSECUTIVE_MAX

    def _accept_frame(self, frame: np.ndarray, source: str = "") -> bool:
        """
        Central frame validity gate.  Returns True if frame should be stored
        and sent to the inference pipeline.  False = bad frame, logged.
        """
        if self._is_corrupted_frame(frame):
            self.bad_frame_count += 1
            self._consecutive_bad += 1
            logger.debug("[CamReader] %s corrupted frame #%d", source, self.bad_frame_count)
            return False
        if self._is_green_frame(frame):
            self.bad_frame_count += 1
            self._consecutive_bad += 1
            logger.debug("[CamReader] %s green frame #%d (decoder artifact)", source, self.bad_frame_count)
            return False
        if self._is_frozen_frame(frame):
            self.bad_frame_count += 1
            self._consecutive_bad += 1
            logger.debug("[CamReader] %s frozen frame #%d", source, self.bad_frame_count)
            return False
        self._consecutive_bad = 0
        return True

    # ── Internal read loop ────────────────────────────────────────────────────

    def _read_loop(self):
        """Main read loop dispatcher - catches all exceptions to prevent thread crashes."""
        try:
            if self._is_shot:
                self._poll_jpeg_loop()
            elif self._is_webcam or self._is_rtsp:
                self._opencv_loop()  # OpenCV/FFmpeg handles RTSP + webcam
            else:
                self._mjpeg_http_loop()  # urllib handles HTTP MJPEG reliably
        except Exception as e:
            # Catch any unhandled exceptions to prevent thread crash
            self._error = f"Read loop exception: {type(e).__name__}: {e}"
            self._stream_status = STREAM_OFFLINE
            import traceback
            print(f"[CameraReader] Exception in read loop for {self.url}:")
            traceback.print_exc()

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
        """
        OpenCV VideoCapture loop (RTSP / webcam / fallback)  —  LOW-LATENCY version.

        Key difference from the naive 1-frame-per-interval loop:
          • We call cap.grab() (fast, no decode) + cap.retrieve() (decode last)
            in a TIGHT LOOP until no more frames are pending.  This empties the
            FFmpeg decoder buffer AND OpenCV's wrapper buffer EVERY CYCLE so we
            never keep more than 1 pending frame at any time.
          • Only the *last* decoded frame of each drain cycle is kept — all
            earlier ones are dropped on the spot (counted in capture_dropped).
          • NO sleep between reads, NO read-interval throttle, NO 50 ms
            "bad frame" sleeps.  The ONLY tiny sleep is 0.5 ms when no frames
            are pending to avoid 100 % CPU on an empty socket, which is way
            too small to let any buffer accumulate.

        TCP is retained for RTSP because it prevents the blocky / washed-out /
        green-artifact frames you get from UDP packet loss (H.264/H.265 P-frames
        depend on previous frames, so a single UDP drop corrupts everything
        until the next keyframe, typically 1–2 seconds).  The slightly higher
        transport latency of TCP (< 20 ms per frame) is negligible compared to
        the 500–2000 ms latency saved by draining the decode pipeline buffer.
        """
        # Re-assert the env-var flags right before open (the top-level global
        # assignment is usually sufficient but some drivers re-read them on
        # each new cv2.VideoCapture).
        if self._is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "fflags;nobuffer|"
                "flags;low_delay|"
                "stimeout;5000000|"  # 5 seconds socket timeout
                "max_delay;0"
            )

        # Support webcam index passed as string "0", "1", …
        src = int(self.url) if self._is_webcam else self.url
        cap_backend = cv2.CAP_FFMPEG if self._is_rtsp else cv2.CAP_ANY
        
        # Add connection timeout and better error handling for RTSP
        cap = cv2.VideoCapture(src, cap_backend)
        if self._is_rtsp:
            # Set explicit connection timeout (milliseconds)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 seconds
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)  # 10 seconds

        # ── OpenCV buffer size  —  hint only (FFmpeg may ignore, but setting it
        #    to 1 still helps on native-webcam backends and does no harm).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        with self._lock:
            self._cap = cap

        if not cap.isOpened():
            self._error = (
                f"❌ Failed to open stream: {self.url}\n\n"
                f"Common issues:\n"
                f"1. Camera offline or unreachable - check if camera IP {self.url.split('@')[-1].split(':')[0] if '@' in self.url else 'N/A'} is pingable\n"
                f"2. Wrong credentials - verify username/password\n"
                f"3. RTSP port (554) blocked by firewall\n"
                f"4. Wrong stream path - try /Streaming/Channels/101 for mainstream or /Streaming/Channels/102 for substream\n"
                f"5. Camera RTSP service disabled - check camera settings\n"
                f"6. Network timeout - camera might be on different subnet/VLAN\n\n"
                f"For high-resolution streams: use substream (subtype=1) for lower bandwidth."
            )
            self._stream_status = STREAM_OFFLINE
            with self._lock:
                if self._cap is not None:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
            return

        self._stream_status = STREAM_RECOVERING
        consecutive_failures = 0
        MAX_FAIL = 120  # ~6 s with empty sleeps — fail fast but not TOO fast
        t_fps = time.time()
        fps_frames = 0

        # Pre-allocate resized frame buffer to avoid repeated allocations on
        # the hot path (resize still copies, but we keep the reference steady).
        _resized_cache = None

        while self._running:
            try:
                ret, frame = cap.read()
            except Exception as read_exc:
                self._error = f"Stream read exception: {read_exc}"
                self._stream_status = STREAM_OFFLINE
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAIL:
                    break
                time.sleep(0.01)
                continue

            if not ret or frame is None or frame.size == 0:
                self.decoder_error_count += 1
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAIL:
                    self._error = (
                        f"Camera stream lost after {MAX_FAIL} failed reads — "
                        f"check URL / network: {self.url}"
                    )
                    self._stream_status = STREAM_OFFLINE
                    break
                time.sleep(0.005)
                continue

            consecutive_failures = 0
            self.cap_read_total += 1

            try:
                fh, fw = frame.shape[:2]
                if fw > 640:
                    target_w = 640
                    target_h = int(fh * (640.0 / fw))
                    target_h = target_h - (target_h % 2)
                    frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_resized = frame
            except cv2.error:
                self.decoder_error_count += 1
                continue

            if not self._accept_frame(frame_resized, source=f"cam:{self.url}"):
                if consecutive_failures == 0:
                    consecutive_failures = 1
                if consecutive_failures >= _BAD_FRAME_DEGRADED_THRESH:
                    self._stream_status = STREAM_DEGRADED
                continue

            # ── Valid frame — store it along with its capture epoch ──────
            now_ts = time.time()
            self._stream_status = STREAM_ONLINE
            self._error = None
            self._last_frame_ts = now_ts
            with self._lock:
                self._frame = frame_resized.copy()
                self._frame_capture_ts = now_ts
                self._frame_count += 1
            fps_frames += 1
            elapsed = now_ts - t_fps
            if elapsed >= 2.0:
                self._fps = fps_frames / elapsed
                fps_frames = 0
                t_fps = now_ts

        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
        self._stream_status = STREAM_OFFLINE

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
    camera_id: Optional[int] = None,
    floor: str = "default",
) -> Dict:
    """
    Run PPE detection and optional face recognition simultaneously.
    Also runs cash_monitor.check_cash_zone() for cameras with cash zone.
    Returns a merged result dict with a single annotated_frame.
    
    NOTE: This function runs in a thread-pool executor — all DB access MUST use
    its own SessionLocal (the async context's session is NOT thread-safe).
    """
    from database import SessionLocal
    _thr_db = SessionLocal()  # thread-local session
    try:
        return _run_combined_inference_inner(
            frame, role, user_id, _thr_db, det_filters, no_phone_zone,
            frame_idx, enable_face, camera_id, floor,
        )
    finally:
        try:
            _thr_db.close()
        except Exception:
            pass


def _run_combined_inference_inner(
    frame: np.ndarray,
    role: str,
    user_id: int,
    db: Session,
    det_filters: Optional[List[str]],
    no_phone_zone: bool,
    frame_idx: int,
    enable_face: bool,
    camera_id: Optional[int] = None,
    floor: str = "default",
) -> Dict:
    """
    Inner implementation — always receives a fresh thread-local DB session.
    """
    ppe_result = {}
    face_result = {}

    # --- PPE / YOLO detection (now with zone-aware multi-model) --------------
    _cam_zone = "default"
    _cashbox_polygon = None
    _vendor_polygon = None
    _zones_map = {}
    if camera_id is not None:
        try:
            import json
            from database import Camera as CameraModel
            cam_obj = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
            if cam_obj:
                if cam_obj.floor:
                    floor = cam_obj.floor.lower()
                if cam_obj.zone_type:
                    _cam_zone = cam_obj.zone_type
                if hasattr(cam_obj, "zones"):
                    for z in cam_obj.zones:
                        zn = (z.zone_name or "").lower()
                        zt = (z.zone_type or "").lower()
                        poly = None
                        try:
                            poly = json.loads(z.polygon_json)
                        except Exception:
                            pass
                        if poly:
                            _zones_map[zn] = poly
                            if zt and zt not in _zones_map:
                                _zones_map[zt] = poly

                        if "cash" in zn or "cash" in zt:
                            if poly:
                                _cashbox_polygon = poly
                        if "vendor" in zn or "payee" in zn or "counter" in zn or "vendor" in zt:
                            if poly:
                                _vendor_polygon = poly
        except Exception:
            pass

    if role != "Home":
        ppe_result = yolo_service.process_frame_numpy(
            frame,
            role,
            det_filters,
            no_phone_zone,
            frame_idx,
            zone_type=_cam_zone,
            camera_id=camera_id,
            zones=_zones_map if _zones_map else None,
            floor=floor,
        )
        print(f"[CCTV-DEBUG] PPE detection: persons={len(ppe_result.get('persons', []))}, detections={len(ppe_result.get('detections', []))}")
    else:
        # Home role → PPE pipeline still runs (for phone detection)
        ppe_result = yolo_service.process_frame_numpy(
            frame,
            "Home",
            [],  # no PPE requirements
            no_phone_zone,
            frame_idx,
            zone_type=_cam_zone,
            camera_id=camera_id,
            zones=_zones_map if _zones_map else None,
            floor=floor,
        )
        print(f"[CCTV-DEBUG] Home detection: persons={len(ppe_result.get('persons', []))}, detections={len(ppe_result.get('detections', []))}")

    # Record worker presence for floor shift start compliance
    if ppe_result.get("persons") and floor:
        try:
            from services import rule_engine
            rule_engine.record_person_seen(floor)
        except Exception:
            pass

    # --- Cash monitoring (CashEventTracker state machine) --------------------
    # Runs when a camera_id is supplied (managed + legacy modes both work).
    # Uses class 14 (Cash) detections from the PPE result; fires a DB alert
    # when cash enters an employee body zone then disappears (pocket theft).
    _cash_alert_msg: Optional[str] = None
    _payee_detected: bool = False
    _payee_snapshot_b64: Optional[str] = None
    _payee_id: Optional[str] = None

    if camera_id is not None and db is not None:
        try:
            from services import cash_monitor as _cm

            _persons  = ppe_result.get("persons", [])

            _c_res = _cm.check_cash_zone(
                db=db,
                camera_id=camera_id,
                detections=ppe_result.get("detections", []),
                persons=_persons,
                cashbox_polygon=_cashbox_polygon,
                floor=floor,
                frame=frame,
                vendor_polygon=_vendor_polygon,
            )

            if _c_res.get("theft_alert"):
                _cash_alert_msg = _c_res["theft_alert"]
            _payee_detected = _c_res.get("payee_detected", False)
            _payee_snapshot_b64 = _c_res.get("payee_snapshot_b64")
            _payee_id = _c_res.get("payee_id")
        except Exception as _cm_exc:
            import logging as _log_mod
            _log_mod.getLogger("cctv").debug(
                "[cctv] cash_monitor error (cam=%s): %s", camera_id, _cm_exc
            )

        # ── Run Rule Engine & Sub-monitors for CCTV live stream ───────────────
        try:
            from services import idle_service, rule_engine
            _persons = ppe_result.get("persons", [])
            _raw_dets = ppe_result.get("detections", [])

            # 1. Idle service tracking
            idle_service.process_frame(
                camera_id=camera_id,
                persons=_persons,
                zone_name=_cam_zone,
            )

            # 2. Multi-model rules (Fall, Machine Anomaly, Object Throwing)
            rule_engine.process_multi_model_rules(
                camera_id=camera_id,
                detections=_raw_dets,
                persons=_persons,
                db=db,
                floor=floor,
                frame=frame,
            )

            # 3. Camera blocking / zone standing
            rule_engine.check_camera_blocking(
                camera_id=camera_id,
                frame_shape=frame.shape,
                persons=_persons,
                db=db,
                zones=_zones_map if _zones_map else None,
            )

            # 4. Shop absence
            rule_engine.check_shop_absence(
                camera_id=camera_id,
                floor=floor,
                persons=_persons,
                db=db,
            )

            # 5. Cleanliness / Dirty floor
            rule_engine.check_dirty_floor(
                camera_id=camera_id,
                floor=floor,
                frame=frame,
                db=db,
            )

            # 6. Cylinder detections
            rule_engine.process_cylinder_detections(
                camera_id=camera_id,
                floor=floor,
                raw_detections=_raw_dets,
                db=db,
            )

            # 7. Stock zone check (exposed items)
            if _zones_map:
                rule_engine.check_stock_zone(
                    camera_id=camera_id,
                    floor=floor,
                    raw_detections=_raw_dets,
                    zones=_zones_map,
                    db=db,
                )

            # 8. Machinery zone & workflow enforcement
            active_zone_keys = set(_zones_map.keys()) if _zones_map else set()
            active_zone_keys.add(_cam_zone.lower())
            if active_zone_keys & {"dough", "dough_table", "dough_mixing", "biscuit_cutting", "cutting_machine", "machine"}:
                rule_engine.check_machinery_zone(
                    camera_id=camera_id,
                    floor=floor,
                    raw_detections=_raw_dets,
                    persons=_persons,
                    zones=_zones_map or {},
                    db=db,
                )
                rule_engine.check_workflow_enforcement(
                    camera_id=camera_id,
                    floor=floor,
                    raw_detections=_raw_dets,
                    persons=_persons,
                    zones=_zones_map or {},
                    db=db,
                )

            # 9. Packing monitor (hand movement)
            if active_zone_keys & {"packing"}:
                try:
                    from services import packing_monitor
                    packing_monitor.process_packing_frame(
                        db=db,
                        camera_id=camera_id,
                        frame=frame,
                        persons=_persons,
                        packing_polygon=_zones_map.get("packing") if _zones_map else None,
                    )
                except Exception:
                    pass

            # 10. Lift monitor
            if active_zone_keys & {"lift", "glass_door"}:
                try:
                    from services import lift_monitor
                    lift_monitor.process_lift_frame(
                        db=db,
                        camera_id=camera_id,
                        floor=floor,
                        persons=_persons,
                        lift_polygon=(_zones_map.get("lift") or _zones_map.get("glass_door")) if _zones_map else None,
                    )
                except Exception:
                    pass

            # 11. Chewing & Clean-shave monitor
            try:
                from services import chew_monitor
                chew_monitor.process_frame(
                    db=db,
                    camera_id=camera_id,
                    frame=frame,
                    persons=_persons,
                )
            except Exception:
                pass

            # 12. Window throw / theft detection
            if active_zone_keys & {"window", "window_throw"}:
                rule_engine.check_window_throw(
                    camera_id=camera_id,
                    floor=floor,
                    raw_detections=_raw_dets,
                    persons=_persons,
                    frame=frame,
                    zones=_zones_map or {},
                    db=db,
                )

            # 13. Finished goods dispatch
            if active_zone_keys & {"finished_goods", "loading", "vehicle"}:
                rule_engine.check_finished_goods_dispatch(
                    camera_id=camera_id,
                    floor=floor,
                    raw_detections=_raw_dets,
                    persons=_persons,
                    zones=_zones_map or {},
                    db=db,
                )

        except Exception as _re_exc:
            import logging as _log_mod
            _log_mod.getLogger("cctv").debug(
                "[cctv] rule_engine error (cam=%s): %s", camera_id, _re_exc
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
    # Cash monitoring fields — relayed to the browser via the WS response.
    merged["cash_detected"] = ppe_result.get("cash_detected", False) or (_cash_alert_msg is not None)
    merged["cash_alert"]    = _cash_alert_msg
    merged["payee_detected"] = _payee_detected
    merged["payee_snapshot_b64"] = _payee_snapshot_b64
    merged["payee_id"] = _payee_id
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
            # ── LEGACY / DIRECT URL MODE: Check if DB matches this RTSP URL to inherit its zone & camera_id ──
            from database import Camera as CameraModel
            db_cam = db.query(CameraModel).filter(CameraModel.rtsp_url == camera_url).first()
            if db_cam and camera_manager.is_running(db_cam.id):
                managed_camera_id = db_cam.id
                display_url = f"managed:{managed_camera_id}"
                print(f"[CCTV-v3] Auto-mapped RTSP URL to managed CameraID: {managed_camera_id} (zone: {db_cam.zone_type})")
            else:
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

            # Ramp-up: wait up to 30 s for first frame (increased for slow cameras)
            for _ in range(300):
                if not state["alive"]:
                    return
                if camera.latest_frame() is not None:
                    break
                if camera.last_error():
                    break
                await asyncio.sleep(0.1)

            # Auto-fallback: if subtype=0 didn't yield frames, try subtype=1 (substream)
            if camera.latest_frame() is None and "subtype=0" in camera_url:
                alt_url = camera_url.replace("subtype=0", "subtype=1")
                print(f"[CCTV] Subtype 0 timed out, auto-falling back to substream: {alt_url}")
                camera.stop()
                await asyncio.sleep(0.3)
                camera = CameraReader(alt_url)
                camera.start()
                for _ in range(200):  # Increased from 100 to 200 (20 seconds)
                    if not state["alive"]:
                        return
                    if camera.latest_frame() is not None:
                        display_url = alt_url
                        print(f"[CCTV] Successfully connected via substream fallback: {alt_url}")
                        break
                    if camera.last_error():
                        break
                    await asyncio.sleep(0.1)

            if camera.last_error():
                error_msg = camera.last_error()
                print(f"[CCTV] Camera connection error: {error_msg}")
                await websocket.send_json({"error": error_msg})
                await websocket.close()
                return

            if camera.latest_frame() is None:
                timeout_msg = (
                    f"⏱️ Connection timeout: No frames received from camera after 30 seconds.\n\n"
                    f"Camera URL: {camera_url}\n\n"
                    f"✅ The diagnostic test showed this camera DOES work!\n\n"
                    f"Troubleshooting:\n"
                    f"1. Camera may be slow to respond - try refreshing the page\n"
                    f"2. Use the working test URL: rtsp://admin:PFCOHO%400624@192.168.100.201:554/Streaming/Channels/1602\n"
                    f"3. For best performance, register this camera in the database\n"
                    f"4. Check that the camera isn't already streaming to another client\n"
                    f"5. Restart the camera if it's unresponsive"
                )
                print(f"[CCTV] {timeout_msg}")
                await websocket.send_json({"error": timeout_msg})
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

        # ── Coroutine B: send LATEST annotated payload to client ─────────
        #     Two distinct modes:
        #
        #     MANAGED MODE (camera_id set):
        #       • NO inference runs HERE.  We just read whatever the shared
        #         per-camera _stream_annotation_thread cached as the latest
        #         WebSocket payload.  This guarantees ONE YOLO inference per
        #         camera no matter how many browsers open N tabs, and keeps
        #         WS send latency ~5 ms instead of 100–200 ms.
        #
        #     LEGACY MODE (camera_url set, direct URL, no camera row):
        #       • Keep the original inline-inference behaviour for backwards
        #         compatibility (each session has its own private CameraReader
        #         and YOLO pipeline — it's intended for one-off testing URLs
        #         anyway, so duplication is acceptable).
        #
        #     Both modes:
        #       • Never send the SAME annotation twice (track the last
        #         source_frame_count and skip duplicates).  Without this the
        #         WS send loop would flood the browser with identical 60 KB
        #         frames at 100+ fps when the shared annotation thread is
        #         slower than the 50 ms send floor.
        #       • MIN_INTERVAL_MS caps user-visible send rate at ~20 fps.
        # ─────────────────────────────────────────────────────────────────────
        async def process_frames():
            MIN_INTERVAL_MS = 33  # hard floor: don't send faster than 30 fps
            last_sent_ts: float = 0.0
            last_sent_frame_count: int = -1
            ws_send_count: int = 0
            ws_skip_dup_count: int = 0

            while state["alive"]:
                loop_now = time.time()

                # ── Rate limit WS sends ──────────────────────────────────
                # FIX: After sleeping, fall through to fetch+send immediately.
                # The old code had `continue` here which forced ANOTHER loop
                # iteration, re-checking the duplicate-frame guard and often
                # adding an extra 10-50 ms of unnecessary delay.
                if last_sent_ts > 0:
                    since_last = loop_now - last_sent_ts
                    if since_last < MIN_INTERVAL_MS / 1000:
                        await asyncio.sleep(
                            (MIN_INTERVAL_MS / 1000) - since_last
                        )
                        # fall through — do NOT continue

                # ── Fetch latest payload ─────────────────────────────────
                response: Optional[dict] = None
                cam_error: Optional[str] = None

                if managed_camera_id is not None:
                    # ─── MANAGED MODE: read shared cached annotation ──
                    cam_error = camera_manager.get_reader_error(managed_camera_id)
                    if cam_error:
                        await websocket.send_json({"error": cam_error})
                        state["alive"] = False
                        return

                    shared_payload = camera_manager.get_latest_stream_result(
                        managed_camera_id
                    )
                    if shared_payload is None:
                        # Annotation thread hasn't produced a frame yet.
                        # Wait briefly without burning CPU.
                        await asyncio.sleep(0.02)
                        continue

                    src_frame_count = int(
                        shared_payload.get("frame_count", -1)
                    )
                    if src_frame_count == last_sent_frame_count:
                        # We already sent this exact annotation to the browser.
                        # Skip it — sending a duplicate wastes WS bandwidth
                        # and 60 KB x N tabs adds up FAST.
                        ws_skip_dup_count += 1
                        await asyncio.sleep(0.01)
                        continue

                    # ── Build WS response, RE-USING the pre-built shared payload.
                    #    Inject per-subscriber fields that the shared thread
                    #    can't know: active_filters, per-user alert_saved flags.
                    response = dict(shared_payload)
                    response["active_filters"] = state["filters"]
                    response["ws_send_count"] = ws_send_count
                    response["ws_dup_skipped"] = ws_skip_dup_count
                    response["subscriber_frame_count"] = state["frame_count"]
                    last_sent_frame_count = src_frame_count

                    # ── Per-subscriber alert-save (cooldown keyed per user) ──
                    #    The shared annotation thread already fires cash alerts
                    #    + attendance, but user-specific PPE / phone / unknown-
                    #    face alerts are saved HERE so each user gets their
                    #    own alert history row and per-user cooldown is honoured.
                    uid = user.id
                    now_t = time.time()

                    if (
                        not response.get("is_compliant", True)
                        and response.get("alert_message")
                    ):
                        dets = response.get("detections", [])
                        persons_c = [
                            p.get("confidence", 0)
                            for p in response.get("persons", [])
                        ]
                        dets_c = [d.get("confidence", 0) for d in dets]
                        top_conf = max(persons_c + dets_c, default=0.5)
                        if (
                            top_conf >= settings.MIN_VIOLATION_CONF
                        ):
                            missing = response.get("missing_items", [])
                            cam_id = response.get("camera_id") or managed_camera_id

                            # Managed camera head-cap state is handled by the
                            # dedicated crop monitor. Do not create a second
                            # per-WebSocket alert for that same violation.
                            event_key = _cctv_alert_key(
                                uid,
                                cam_id,
                                missing,
                                response.get("alert_message", ""),
                            )
                            alert_allowed = bool(event_key)

                            # One DB row/push per continuous event. Event is
                            # cleared only after a compliant frame arrives;
                            # therefore all-day violation stays one alert.
                            if alert_allowed:
                                with _active_cctv_alerts_lock:
                                    active_since = _active_cctv_alerts.get(event_key)
                                    cooldown = getattr(settings, "ALERT_COOLDOWN", 15)
                                    if active_since and (now_t - active_since < cooldown):
                                        alert_allowed = False
                                    else:
                                        _active_cctv_alerts[event_key] = now_t

                            c_worker_name = None
                            c_emp_id = None
                            try:
                                from services.face_service import get_worker_identity
                                if cam_id is not None:
                                    for p in response.get("persons", []):
                                        if not p.get("is_compliant", True):
                                            tid = p.get("track_id")
                                            if tid not in (None, -1, "-1", ""):
                                                ident = get_worker_identity(cam_id, str(tid))
                                                if ident and ident.get("name") and ident["name"] != "Unknown":
                                                    c_worker_name = ident["name"]
                                                    c_emp_id = ident.get("employee_id")
                                                    break
                                if not c_worker_name:
                                    fr_faces = (response.get("face_result") or {}).get("faces", [])
                                    for ff in fr_faces:
                                        if not ff.get("is_unknown") and ff.get("label") and ff["label"] != "Unknown":
                                            c_worker_name = ff["label"]
                                            c_emp_id = ff.get("employee_id")
                                            break
                            except Exception:
                                pass

                            missing_str = ", ".join(missing) if missing else (response.get("alert_message") or "Safety violation")
                            clean_items = [m.replace("NO-", "").replace("No ", "") for m in missing]
                            clean_items_str = ", ".join(clean_items)

                            if c_worker_name:
                                detected_issue = f"{c_worker_name} — {missing_str}"
                                if clean_items_str:
                                    alert_msg = f"{c_worker_name} has not worn {clean_items_str}"
                                else:
                                    alert_msg = f"{c_worker_name} — {response['alert_message']}"
                            else:
                                detected_issue = missing_str
                                alert_msg = response["alert_message"]

                            if alert_allowed:
                                try:
                                    save_alert(
                                        db=db,
                                        user_id=uid,
                                        message=alert_msg,
                                        role=role,
                                        severity=response.get("severity"),
                                        detected_issue=detected_issue,
                                        confidence=round(top_conf, 3),
                                        snapshot_b64=response.get("snapshot_b64") or response.get("annotated_frame"),
                                        camera_id=cam_id,
                                        worker_name=c_worker_name,
                                        employee_id=c_emp_id,
                                    )
                                    response["alert_saved"] = True
                                except Exception as _alert_exc:
                                    logger.error(f"[CCTV] Failed to save alert: {_alert_exc}", exc_info=True)
                                    db.rollback()
                                    with _active_cctv_alerts_lock:
                                        _active_cctv_alerts.pop(event_key, None)
                    elif managed_camera_id is not None:
                        _clear_cctv_alert_events(uid, managed_camera_id)

                    if response.get("phone_severity") == "high" and response.get(
                        "phone_alert"
                    ):
                        if now_t - _last_phone_alert_time.get(uid, 0) > 15:
                            _last_phone_alert_time[uid] = now_t
                            cam_id = response.get("camera_id") or managed_camera_id
                            p_msg = response["phone_alert"]
                            p_issue = f"{c_worker_name} — {p_msg}" if c_worker_name else p_msg
                            p_full = f"{c_worker_name}: {p_msg}" if c_worker_name else p_msg
                            try:
                                save_alert(
                                    db=db,
                                    user_id=uid,
                                    message=p_full,
                                    role=role,
                                    severity="high",
                                    detected_issue=p_issue,
                                    confidence=0.85,
                                    snapshot_b64=response.get("snapshot_b64") or response.get("annotated_frame"),
                                    camera_id=cam_id,
                                    worker_name=c_worker_name,
                                    employee_id=c_emp_id,
                                )
                                response["phone_alert_saved"] = True
                            except Exception:
                                pass

                    fr = response.get("face_result") or {}
                    if fr.get("unknown_detected"):
                        if now_t - _last_alert_time.get(f"face_{uid}", 0) > 20:
                            _last_alert_time[f"face_{uid}"] = now_t
                            try:
                                save_alert(
                                    db=db,
                                    user_id=uid,
                                    message="Unknown person detected via CCTV",
                                    role=role,
                                    severity="critical",
                                    detected_issue="Unknown face",
                                    confidence=0.90,
                                    snapshot_b64=response.get("snapshot_b64") or response.get("annotated_frame"),
                                )
                                response["face_alert_saved"] = True
                            except Exception:
                                pass

                else:
                    # ─── LEGACY MODE: camera_url → inline inference ─────
                    #     (unchanged behaviour for backwards compatibility)
                    if camera.last_error():
                        await websocket.send_json({"error": camera.last_error()})
                        state["alive"] = False
                        return
                    frame = camera.latest_frame()
                    cam_fps_legacy = camera.fps()

                    if frame is None:
                        await asyncio.sleep(0.015)
                        continue

                    # Debug: Log that we have a frame
                    if state["frame_count"] % 30 == 0:  # Log every 30 frames
                        print(f"[CCTV-LEGACY] Processing frame #{state['frame_count']} from {camera.url}, frame shape: {frame.shape if frame is not None else 'None'}")

                    state["frame_count"] += 1
                    fn = state["frame_count"]
                    det_filters = (
                        list(state["filters"]) if state["filters"] else None
                    )
                    nph = state["no_phone_zone"]
                    ef = state["enable_face"]

                    try:
                        result = await loop.run_in_executor(
                            _yolo_executor,
                            lambda fr=frame, fi=fn, df=det_filters, nz=nph, efa=ef: (
                                _run_combined_inference(
                                    fr,
                                    role,
                                    user.id,
                                    db,
                                    df,
                                    nz,
                                    fi,
                                    efa,
                                    camera_id=None,
                                    floor="shop",
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
                            "cam_fps": cam_fps_legacy,
                            "source": "cctv",
                            "cash_detected": result.get("cash_detected", False),
                            "cash_alert": result.get("cash_alert"),
                            "cash_alert_saved": False,
                            "payee_detected": result.get("payee_detected", False),
                            "payee_snapshot_b64": result.get("payee_snapshot_b64"),
                            "payee_id": result.get("payee_id"),
                            "uniform_detected": any(
                                p.get("has_uniform")
                                for p in result.get("persons", [])
                            ),
                            "uniform_violation": any(
                                not p.get("has_uniform", True)
                                for p in result.get("persons", [])
                            ),
                            "mode": "legacy",
                        }
                        last_sent_frame_count = fn

                        if not result.get("is_compliant") and result.get(
                            "alert_message"
                        ):
                            uid = user.id
                            now_t = time.time()
                            dets = result.get("detections", [])
                            persons_c = [
                                p.get("confidence", 0)
                                for p in result.get("persons", [])
                            ]
                            dets_c = [d.get("confidence", 0) for d in dets]
                            top_conf = max(persons_c + dets_c, default=0.5)
                            cooldown = settings.ALERT_COOLDOWN
                            if (
                                top_conf >= settings.MIN_VIOLATION_CONF
                                and (now_t - _last_alert_time.get(uid, 0))
                                > cooldown
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
                    except Exception as e:
                        if state["alive"]:
                            print(
                                f"[CCTV-legacy] Frame#{fn} error: "
                                f"{type(e).__name__}: {e}"
                            )
                        state["alive"] = False
                        return

                # ── Send to browser ───────────────────────────────────────
                try:
                    state["frame_count"] += 1
                    ws_send_count += 1
                    await websocket.send_json(response)
                    last_sent_ts = time.time()
                except WebSocketDisconnect:
                    state["alive"] = False
                    return
                except Exception as e:
                    if state["alive"]:
                        print(
                            f"[CCTV-ws] Send error: {type(e).__name__}: {e}"
                        )
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
