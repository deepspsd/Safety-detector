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
  start()                      → called once at FastAPI startup
  stop()                       → called once at FastAPI shutdown
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Dict, Optional

import numpy as np

log = logging.getLogger("inference_pool")

# ── Tuning ───────────────────────────────────────────────────────────────────
# How many frames may accumulate per camera before the oldest is dropped.
# Keeps latency low when inference is slow — we always want the freshest frame.
_MAX_QUEUE_DEPTH = 2

# How long (seconds) the worker sleeps when the queue is empty.
_IDLE_SLEEP = 0.05


class _InferencePool:
    """
    Singleton shared YOLO inference worker.

    Lifecycle
    ---------
    1. start() — loads the YOLO model once, spawns one daemon thread.
    2. put_frame(camera_id, frame) — camera detection loops call this.
    3. get_result(camera_id) — camera detection loops read the latest result.
    4. stop() — signals the worker thread to exit cleanly.
    """

    def __init__(self) -> None:
        # { camera_id: queue.Queue(maxsize=_MAX_QUEUE_DEPTH) }
        self._queues:  Dict[int, "queue.Queue[np.ndarray]"] = {}
        # { camera_id: inference result dict }
        self._results: Dict[int, dict] = {}
        self._lock     = threading.Lock()
        self._running  = False
        self._thread:  Optional[threading.Thread] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the inference worker thread.  Safe to call multiple times."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="inference-pool-worker",
            daemon=True,
        )
        self._thread.start()
        log.info("[InferencePool] Worker thread started — shared YOLO model active")

    def stop(self) -> None:
        """Signal the worker to exit.  Called from FastAPI shutdown."""
        self._running = False
        log.info("[InferencePool] Shutting down")

    def put_frame(self, camera_id: int, frame: np.ndarray) -> None:
        """
        Submit a frame for inference.  Non-blocking — drops oldest frame if
        the per-camera queue is full so we always process fresh frames.
        """
        with self._lock:
            if camera_id not in self._queues:
                self._queues[camera_id] = queue.Queue(maxsize=_MAX_QUEUE_DEPTH)
            q = self._queues[camera_id]

        # Drop oldest frame if queue is full (keep latency low)
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        try:
            q.put_nowait(frame)
        except queue.Full:
            pass  # race condition — ignore

    def get_result(self, camera_id: int) -> Optional[dict]:
        """
        Return the most recent inference result for this camera.
        Returns None if no result is available yet (first startup frames).
        """
        with self._lock:
            return self._results.get(camera_id)

    def remove_camera(self, camera_id: int) -> None:
        """Clean up state when a camera is stopped."""
        with self._lock:
            self._queues.pop(camera_id, None)
            self._results.pop(camera_id, None)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Main inference loop — runs in a single daemon thread.

        Iterates over all registered camera queues in round-robin order,
        pops one frame per camera per cycle, and runs YOLO inference.
        Sleeps briefly when all queues are empty.
        """
        # Lazy-import YOLO so the pool can be imported before the model loads.
        try:
            from services.detection_layer import detector
        except Exception as exc:
            log.error(f"[InferencePool] Could not import detector: {exc}")
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
                try:
                    detections = detector.detect(frame)
                    det_dicts  = [d.to_dict() for d in detections]
                except Exception as exc:
                    log.debug(f"[InferencePool] Inference error cam={camera_id}: {exc}")
                    det_dicts = []

                with self._lock:
                    self._results[camera_id] = {"detections": det_dicts}

            if not processed_any:
                time.sleep(_IDLE_SLEEP)


# ── Module-level singleton ────────────────────────────────────────────────────
inference_pool = _InferencePool()
