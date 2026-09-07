"""
services/inference_pool.py — Shared YOLO inference pool for multi-camera setups
================================================================================
Problem solved
--------------
Running one YOLO inference thread per camera causes:
  • N × model RAM usage (each thread loads its own model copy)
  • N × CPU cores saturated simultaneously
  • Server crash on 16 GB RAM with >3 cameras

Solution
--------
One background inference worker thread processes frames from ALL cameras in
sequence.  Each camera submits its latest frame to a shared queue and reads
back the most recent result — no model duplication, bounded CPU load.

Watchdog (added v3.1)
---------------------
A second daemon thread checks the worker every 30 s.  If it is not alive
(unhandled exception / OOM crash), the watchdog:
  1. Restarts the worker thread.
  2. Sends a Telegram alert so the admin knows a restart happened.
  3. Increments _restart_count (visible on GET /health).

Performance on 16 GB RAM server (single camera = baseline)
-----------------------------------------------------------
  Cameras   YOLO fps/cam   Rule-engine fps/cam   RAM saved
  1         ~5 fps         5 fps                 baseline
  2         ~2.5 fps       5 fps                 1 model copy
  4         ~1.25 fps      5 fps                 3 model copies
  8         ~0.6 fps       5 fps                 7 model copies

Note: rule-engine checks (idle timers, shift-start, dirty-floor, face-rec)
still run at full camera_manager tick rate (~5 fps) because they do NOT need
a new YOLO inference result every frame — they consume the *latest available*
result.  Only YOLO throughput scales down with more cameras; compliance logic
remains responsive.

Public API
----------
  put_frame(camera_id, frame)  → submit a frame for inference (non-blocking)
  get_result(camera_id)        → latest inference result dict | None
  is_healthy()                 → True if worker thread is alive
  restart_count                → int — number of automatic restarts since startup
  start()                      → called once at FastAPI startup
  stop()                       → called once at FastAPI shutdown
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

log = logging.getLogger("inference_pool")

# ── Tuning ───────────────────────────────────────────────────────────────────
# How many frames may accumulate per camera before the oldest is dropped.
# LATEST-FRAME-ONLY discipline: maxsize=1 → every NEW frame overwrites the
# single pending slot.  Ensures inference never processes stale frames.
_MAX_QUEUE_DEPTH = 1

# How long (seconds) the worker sleeps when the queue is empty.
_IDLE_SLEEP = 0.033

# Rolling FPS window for per-camera inference FPS estimate.
_FPS_WINDOW_SECONDS = 5.0


def _resize_for_inference(frame: np.ndarray, target_width: int) -> np.ndarray:
    """
    Downscale frame to target_width while preserving aspect ratio.
    Skipped if frame is already at or below target_width.
    Using INTER_AREA for downscaling (best quality, avoids moire).
    """
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    new_w = target_width
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class _InferencePool:
    """
    Singleton shared YOLO inference worker.

    Lifecycle
    ---------
    1. start() — loads the YOLO model once, spawns worker + watchdog threads.
    2. put_frame(camera_id, frame) — camera detection loops call this.
    3. get_result(camera_id) — camera detection loops read the latest result.
    4. stop() — signals both threads to exit cleanly.
    """

    # How often the watchdog checks worker liveness (seconds).
    _WATCHDOG_INTERVAL = 30

    def __init__(self) -> None:
        # { camera_id_or_crop_key: queue.Queue(maxsize=_MAX_QUEUE_DEPTH) }
        # Accepts both int (camera IDs) and str (temporary crop keys)
        self._queues: Dict[Any, "queue.Queue[np.ndarray]"] = {}
        # { camera_id_or_crop_key: inference result dict }
        self._results: Dict[Any, dict] = {}
        # { camera_id: zone_type str | None } for multi-model routing
        self._camera_zones: Dict[Any, Optional[str]] = {}
        # Per-camera metrics — all counter updates happen under self._lock.
        self._metrics: Dict[Any, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._watchdog: Optional[threading.Thread] = None
        self.restart_count: int = 0  # public — exposed on /health

    # ── Metrics helpers ───────────────────────────────────────────────────────

    def _ensure_metrics(self, camera_id: Any) -> Dict[str, Any]:
        """Create a fresh per-camera metrics dict if missing.  Caller holds lock."""
        m = self._metrics.get(camera_id)
        if m is None:
            m = {
                "frames_submitted": 0,
                "frames_dropped_at_put": 0,
                "frames_processed": 0,
                "last_inference_latency_ms": 0.0,
                "inference_fps": 0.0,
                "_process_times": [],  # rolling end-times for fps calc
            }
            self._metrics[camera_id] = m
        return m

    @staticmethod
    def _rolling_fps(process_times: list, now: float) -> float:
        """Drop stale entries and return rolling FPS over window."""
        cutoff = now - _FPS_WINDOW_SECONDS
        i = 0
        for i, t in enumerate(process_times):
            if t > cutoff:
                break
        del process_times[:i]
        if len(process_times) < 2:
            return 0.0
        span = process_times[-1] - process_times[0]
        if span <= 0:
            return 0.0
        return (len(process_times) - 1) / span

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the inference worker and watchdog threads.  Safe to call multiple times."""
        if self._running:
            return
        self._running = True
        self._spawn_worker()
        self._spawn_watchdog()
        log.info("[InferencePool] Worker and watchdog started")

    def stop(self) -> None:
        """Signal both threads to exit.  Called from FastAPI shutdown."""
        self._running = False
        log.info("[InferencePool] Shutting down")

    def is_healthy(self) -> bool:
        """Return True if the inference worker thread is alive."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def put_frame(self, camera_id: int, frame: np.ndarray, zone_type: Optional[str] = None) -> None:
        """
        Submit a frame for inference.  Non-blocking — latest-frame-only.

        The per-camera queue has maxsize=1.  If a pending frame already
        occupies the slot it is **discarded and counted as a put-drop**
        before the new frame is enqueued.  This guarantees the worker
        never pulls a frame older than one submission cycle.

        Args:
            camera_id: Camera ID
            frame: Frame to process
            zone_type: Zone type for multi-model routing (e.g., "dough_mixing")
        """
        with self._lock:
            if camera_id not in self._queues:
                self._queues[camera_id] = queue.Queue(maxsize=_MAX_QUEUE_DEPTH)
            self._camera_zones[camera_id] = zone_type
            q = self._queues[camera_id]
            metrics = self._ensure_metrics(camera_id)
            metrics["frames_submitted"] += 1
            # LATEST-FRAME-ONLY purge-if-full inside lock so drop counting
            # cannot race with put_nowait below.
            if q.full():
                try:
                    q.get_nowait()
                    metrics["frames_dropped_at_put"] += 1
                except queue.Empty:
                    pass
        try:
            q.put_nowait(frame)
        except queue.Full:
            with self._lock:
                m2 = self._ensure_metrics(camera_id)
                m2["frames_dropped_at_put"] += 1

    def get_result(self, camera_id: Any) -> Optional[dict]:
        """
        Return the most recent inference result for this camera (or crop key).
        Returns None if no result is available yet.
        """
        with self._lock:
            return self._results.get(camera_id)

    def get_metrics(self, camera_id: Any = None) -> Dict[str, Any]:
        """
        Return per-camera metrics dict, or aggregate across all cameras
        when camera_id is None.

        Fields: frames_submitted, frames_dropped_at_put, frames_processed,
                last_inference_latency_ms, inference_fps.
        """
        with self._lock:
            if camera_id is not None:
                raw = self._metrics.get(camera_id)
                if raw is None:
                    return {
                        "frames_submitted": 0,
                        "frames_dropped_at_put": 0,
                        "frames_processed": 0,
                        "last_inference_latency_ms": 0.0,
                        "inference_fps": 0.0,
                    }
                return {k: raw[k] for k in
                        ("frames_submitted", "frames_dropped_at_put",
                         "frames_processed", "last_inference_latency_ms",
                         "inference_fps")}
            agg: Dict[str, Any] = {
                "frames_submitted": 0,
                "frames_dropped_at_put": 0,
                "frames_processed": 0,
                "cameras": {},
            }
            min_lat: Optional[float] = None
            max_lat: Optional[float] = None
            sum_lat = 0.0
            sum_lat_n = 0
            fps_sum = 0.0
            fps_n = 0
            for cid, raw in self._metrics.items():
                agg["frames_submitted"] += raw["frames_submitted"]
                agg["frames_dropped_at_put"] += raw["frames_dropped_at_put"]
                agg["frames_processed"] += raw["frames_processed"]
                agg["cameras"][str(cid)] = {
                    k: raw[k] for k in
                    ("frames_submitted", "frames_dropped_at_put",
                     "frames_processed", "last_inference_latency_ms",
                     "inference_fps")
                }
                lat = raw["last_inference_latency_ms"]
                if lat > 0:
                    if min_lat is None or lat < min_lat:
                        min_lat = lat
                    if max_lat is None or lat > max_lat:
                        max_lat = lat
                    sum_lat += lat
                    sum_lat_n += 1
                if raw["inference_fps"] > 0:
                    fps_sum += raw["inference_fps"]
                    fps_n += 1
            agg["avg_inference_latency_ms"] = (
                sum_lat / sum_lat_n if sum_lat_n else 0.0
            )
            agg["min_inference_latency_ms"] = min_lat if min_lat is not None else 0.0
            agg["max_inference_latency_ms"] = max_lat if max_lat is not None else 0.0
            agg["avg_inference_fps"] = fps_sum / fps_n if fps_n else 0.0
            return agg

    def clear_result(self, key: Any) -> None:
        """
        Remove a result and queue from the pool.
        Used by crop-based head-cap inference to clean up temp keys after reading.
        """
        with self._lock:
            self._queues.pop(key, None)
            self._results.pop(key, None)
            self._camera_zones.pop(key, None)
            self._metrics.pop(key, None)

    def remove_camera(self, camera_id: Any) -> None:
        """Clean up state when a camera is stopped."""
        with self._lock:
            self._queues.pop(camera_id, None)
            self._results.pop(camera_id, None)
            self._camera_zones.pop(camera_id, None)
            self._metrics.pop(camera_id, None)

    # ── Internal: thread management ───────────────────────────────────────────

    def _spawn_worker(self) -> None:
        """Create and start the inference worker daemon thread."""
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="inference-pool-worker",
            daemon=True,
        )
        self._thread.start()
        log.info("[InferencePool] Worker thread started — shared YOLO model active")

    def _spawn_watchdog(self) -> None:
        """Create and start the watchdog daemon thread."""
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="inference-pool-watchdog",
            daemon=True,
        )
        self._watchdog.start()
        log.info(
            "[InferencePool] Watchdog thread started (interval=%ds)",
            self._WATCHDOG_INTERVAL,
        )

    def _watchdog_loop(self) -> None:
        """
        Periodically checks whether the inference worker is still alive.
        If not, restarts it and fires a Telegram alert so the admin is notified.
        """
        while self._running:
            time.sleep(self._WATCHDOG_INTERVAL)
            if not self._running:
                break
            if self._thread is not None and not self._thread.is_alive():
                self.restart_count += 1
                log.error(
                    "[InferencePool] Worker thread died (restart #%d) — restarting",
                    self.restart_count,
                )
                self._spawn_worker()
                self._notify_restart(self.restart_count)

    @staticmethod
    def _notify_restart(restart_count: int) -> None:
        """
        Send a Telegram/ntfy alert informing the admin that the inference
        pool crashed and was automatically restarted.
        Failures are silently swallowed — never block the watchdog loop.
        """
        try:
            from services.notification_service import send_push_alert

            send_push_alert(
                message=(
                    f"⚠️ OccuSafe: YOLO inference pool crashed and was auto-restarted "
                    f"(restart #{restart_count}). Detection was paused briefly. "
                    "Check server logs for the root cause."
                ),
                severity="high",
                detected_issue="Inference pool restart",
            )
        except Exception as exc:
            log.debug("[InferencePool] Watchdog notify failed: %s", exc)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Main inference loop — runs in a single daemon thread.

        Iterates over all registered camera queues in round-robin order,
        pops one frame per camera per cycle, and runs multi-model inference
        based on each camera's zone_type.
        Sleeps briefly when all queues are empty.
        """
        # Lazy-import detector so the pool can be imported before models load
        try:
            from services.multi_model_detector import get_multi_model_detector
            detector = get_multi_model_detector()
            log.info("[InferencePool] Multi-model detector loaded successfully")
        except Exception as exc:
            log.error(f"[InferencePool] Could not import multi-model detector: {exc}")
            # Fallback to single-model detector
            try:
                from services.detection_layer import detector
                log.warning("[InferencePool] Falling back to single-model detector")
            except Exception as exc2:
                log.error(f"[InferencePool] Could not import any detector: {exc2}")
                return

        log.info("[InferencePool] Inference loop running")

        while self._running:
            processed_any = False

            with self._lock:
                camera_ids = list(self._queues.keys())

            for camera_id in camera_ids:
                if not self._running:
                    break

                with self._lock:
                    q = self._queues.get(camera_id)
                if q is None:
                    continue

                try:
                    frame = q.get_nowait()
                except queue.Empty:
                    continue

                processed_any = True
                infer_start = time.perf_counter()
                try:
                    from config import settings

                    infer_frame = _resize_for_inference(
                        frame, settings.YOLO_INFERENCE_WIDTH
                    )

                    zone_type: Optional[str]
                    with self._lock:
                        zone_type = self._camera_zones.get(camera_id)

                    if hasattr(detector, 'detect'):
                        if zone_type:
                            detections = detector.detect(
                                infer_frame, zone_type=zone_type, camera_id=camera_id
                            )
                        else:
                            detections = detector.detect(infer_frame)
                    else:
                        detections = []

                    det_dicts = [d.to_dict() for d in detections]
                except Exception as exc:
                    log.debug(
                        f"[InferencePool] Inference error cam={camera_id}: {exc}",
                        exc_info=True,
                    )
                    det_dicts = []
                latency_ms = (time.perf_counter() - infer_start) * 1000.0
                now = time.time()

                with self._lock:
                    self._results[camera_id] = {"detections": det_dicts}
                    metrics = self._ensure_metrics(camera_id)
                    metrics["frames_processed"] += 1
                    metrics["last_inference_latency_ms"] = latency_ms
                    metrics["_process_times"].append(now)
                    metrics["inference_fps"] = self._rolling_fps(
                        metrics["_process_times"], now
                    )

            if not processed_any:
                time.sleep(_IDLE_SLEEP)


# ── Module-level singleton ────────────────────────────────────────────────────
inference_pool = _InferencePool()
