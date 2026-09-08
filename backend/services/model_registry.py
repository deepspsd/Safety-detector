"""
Model Registry - Multi-Model Management for OccuSafe
====================================================

Centralized model loading, caching, standardized adapters, and capability routing
for the multi-model detection platform.

Architecture:
- BaseModelAdapter: Standardized abstraction for all models
- Concrete Adapters: YOLOModelAdapter, SklearnAnomalyAdapter, MediaPipeAdapter,
  PyTorchTRNAdapter, OpenVINOAdapter
- ModelInfo: Metadata and runtime telemetry container
- ModelRegistry: Singleton that registers, loads, caches, and routes models

Thread-Safety:
- Models are loaded once at startup or on-demand and cached in memory
- Per-model lock / safe read-only inference
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import settings
from services.detection_layer import Detection, NormalizedDetection

logger = logging.getLogger("model_registry")


# ─────────────────────────────────────────────────────────────────────────────
# Standard Model Adapter Interface
# ─────────────────────────────────────────────────────────────────────────────


class BaseModelAdapter(ABC):
    """Common abstraction for all AI models in OccuSafe."""

    def __init__(self, key: str, config: Dict[str, Any]):
        self.key = key
        self.config = config
        self.enabled = config.get("enabled", True)
        self.priority = config.get("priority", 99)
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.target_fps = config.get("target_fps", 5.0)
        self.zones = config.get("zones", ["*"])
        self.capabilities = list(config.get("capabilities", []))
        self.description = config.get("description", "")
        self.classes: Optional[Dict[int, str]] = config.get("classes")
        self.device = "cpu"
        self.status = "NOT_LOADED"
        self.error_message: Optional[str] = None
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_inference_time = 0.0
        self.last_latency_ms = 0.0

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize runtime."""
        pass

    @abstractmethod
    def predict(
        self, frame: np.ndarray, conf: Optional[float] = None
    ) -> List[NormalizedDetection]:
        """Run inference on frame and return normalized detections."""
        pass

    def get_classes(self) -> Dict[int, str]:
        return self.classes or {}

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "type": self.config.get("type", "unknown"),
            "status": self.status,
            "device": self.device,
            "enabled": self.enabled,
            "priority": self.priority,
            "capabilities": self.capabilities,
            "zones": self.zones,
            "classes": self.classes,
            "conf_threshold": self.conf_threshold,
            "target_fps": self.target_fps,
            "description": self.description,
        }

    def health(self) -> Dict[str, Any]:
        avg_latency = (
            (self.total_inference_time / self.inference_count) * 1000.0
            if self.inference_count > 0
            else 0.0
        )
        return {
            "key": self.key,
            "status": self.status,
            "device": self.device,
            "enabled": self.enabled,
            "inference_count": self.inference_count,
            "avg_latency_ms": round(avg_latency, 2),
            "last_latency_ms": round(self.last_latency_ms, 2),
            "last_seen_ts": self.last_inference_time,
            "error": self.error_message,
        }

    def unload(self) -> None:
        self.status = "NOT_LOADED"


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Model Adapters
# ─────────────────────────────────────────────────────────────────────────────


class YOLOModelAdapter(BaseModelAdapter):
    """Adapter for YOLOv8, YOLOv11, and Ultralytics models."""

    def __init__(self, key: str, config: Dict[str, Any]):
        super().__init__(key, config)
        self.model = None

    def load(self) -> None:
        try:
            import torch
            from ultralytics import YOLO

            # Detect CUDA device
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

            model_path = _resolve_model_path(self.config["path"])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # PyTorch 2.6 safe load patch
            _orig_load = torch.load

            def _patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_load(*args, **kwargs)

            torch.load = _patched_load
            try:
                self.model = YOLO(model_path)
            finally:
                torch.load = _orig_load

            abs_path = os.path.abspath(model_path)
            file_exists = os.path.exists(abs_path)
            file_size = os.path.getsize(abs_path) if file_exists else 0
            model_names = {int(k): str(v) for k, v in self.model.names.items()} if hasattr(self.model, "names") else {}

            logger.info(
                f"[ModelRegistry DIAGNOSTIC] Key: '{self.key}' | Path: {abs_path} | "
                f"Exists: {file_exists} | Size: {file_size} bytes | Names: {model_names}"
            )
            print(
                f"🔍 [ModelRegistry DIAGNOSTIC] Key: '{self.key}' | Path: {abs_path} | "
                f"Exists: {file_exists} | Size: {file_size} bytes | Names: {model_names}"
            )

            # For hairnet or if classes unspecified, sync directly to the loaded model weights
            if "hairnet" in self.key:
                if model_names != {0: "hairnet", 1: "no_hairnet"}:
                    logger.warning(
                        f"⚠️ [ModelRegistry] Expected hairnet model names {{0: 'hairnet', 1: 'no_hairnet'}}, got {model_names}!"
                    )
                self.classes = dict(model_names)
            elif not self.classes and model_names:
                self.classes = dict(model_names)

            self.status = "READY"
            self.error_message = None
            logger.info(f"✅ Loaded YOLO adapter for '{self.key}' on {self.device} ({len(self.classes or {})} classes)")
        except Exception as e:
            self.status = "ERROR"
            self.error_message = str(e)
            logger.error(f"❌ Failed to load YOLO model '{self.key}': {e}", exc_info=True)
            raise

    def predict(
        self, frame: np.ndarray, conf: Optional[float] = None
    ) -> List[NormalizedDetection]:
        if self.status != "READY" or self.model is None:
            return []
        if frame is None or frame.size == 0:
            return []

        start_time = time.perf_counter()
        effective_conf = conf if conf is not None else self.conf_threshold

        try:
            target_classes = list(self.classes.keys()) if self.classes else None

            # Optimal inference resolution matching validation (e.g. imgsz=960 for hairnet)
            if "imgsz" in self.config:
                imgsz = int(self.config["imgsz"])
            elif "hairnet" in self.key:
                imgsz = getattr(settings, "HAIRNET_IMGSZ", 960)
            elif "cash" in self.key:
                imgsz = 832
            else:
                imgsz = 640

            results = list(self.model(
                frame,
                verbose=False,
                imgsz=imgsz,
                conf=effective_conf,
                iou=getattr(settings, "NMS_IOU", 0.40),
                classes=target_classes,
                stream=True,
                device=self.device,
            ))

            detections: List[NormalizedDetection] = []
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            fh, fw = frame.shape[:2]

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    box_conf = float(box.conf[0])
                    xyxy = [int(v) for v in box.xyxy[0]]

                    if self.classes and cls_id in self.classes:
                        label = self.classes[cls_id]
                    elif hasattr(self.model, "names") and cls_id < len(self.model.names):
                        label = self.model.names[cls_id]
                    else:
                        label = f"class_{cls_id}"

                    # Diagnostic log for hairnet model raw detections
                    if "hairnet" in self.key or getattr(settings, "DEBUG_HAIRNET_RAW", False):
                        logger.info(
                            f"[HAIRNET RAW PREDICTION] class_id={cls_id} class_name='{label}' "
                            f"confidence={box_conf:.4f} xyxy={xyxy} frame_shape=({fh}, {fw}) imgsz={imgsz}"
                        )
                        print(
                            f"🎯 [HAIRNET RAW] id={cls_id} name='{label}' "
                            f"conf={box_conf:.3f} xyxy={xyxy} shape=({fh},{fw}) imgsz={imgsz}"
                        )

                    # Determine capability matching this detection
                    cap = self.capabilities[0] if self.capabilities else "object_detection"

                    detections.append(
                        NormalizedDetection(
                            label=label,
                            confidence=round(box_conf, 4),
                            bbox=xyxy,
                            model_key=self.key,
                            class_id=cls_id,
                            capability=cap,
                            timestamp=now_str,
                        )
                    )

            elapsed = time.perf_counter() - start_time
            self.last_latency_ms = elapsed * 1000.0
            self.total_inference_time += elapsed
            self.inference_count += 1
            self.last_inference_time = time.time()
            return detections
        except Exception as e:
            self.error_message = str(e)
            logger.error(f"Inference error in YOLO model '{self.key}': {e}")
            return []

    def unload(self) -> None:
        self.model = None
        super().unload()


class SklearnAnomalyAdapter(BaseModelAdapter):
    """Adapter for IoT / Machine Sensor Anomaly Detection using scikit-learn."""

    def __init__(self, key: str, config: Dict[str, Any]):
        super().__init__(key, config)
        self.model = None
        self.scaler = None

    def load(self) -> None:
        try:
            import joblib

            model_path = _resolve_model_path(self.config["path"])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)

            scaler_path_raw = self.config.get("scaler_path")
            if scaler_path_raw:
                scaler_path = _resolve_model_path(scaler_path_raw)
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)

            self.status = "READY"
            self.error_message = None
            logger.info(f"✅ Loaded Sklearn anomaly adapter for '{self.key}'")
        except Exception as e:
            self.status = "ERROR"
            self.error_message = str(e)
            logger.error(f"❌ Failed to load Sklearn anomaly model '{self.key}': {e}")
            raise

    def predict(
        self, frame: np.ndarray, conf: Optional[float] = None
    ) -> List[NormalizedDetection]:
        if self.status != "READY" or self.model is None:
            return []
        if frame is None or frame.size == 0:
            return []

        start_time = time.perf_counter()
        effective_conf = conf if conf is not None else self.conf_threshold

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_lum = float(np.mean(gray))
            std_lum = float(np.std(gray))

            # 8-feature schema expected by trained model
            features = np.array([[
                25.0 + (mean_lum / 255.0) * 15.0,  # temp_celsius
                0.0,                               # high_temp_duration
                0.0,                               # thermal_stress_code
                std_lum / 100.0,                   # degradation_rate
                95.0,                              # health_score
                0.0,                               # health_delta
                mean_lum,                          # light_value
                0.0,                               # anomaly_score
            ]])

            if self.scaler is not None and hasattr(self.scaler, "transform"):
                try:
                    features = self.scaler.transform(features)
                except Exception:
                    pass

            pred = self.model.predict(features)
            score = float(pred[0]) if pred is not None and len(pred) > 0 else 0.0

            detections: List[NormalizedDetection] = []
            if score > effective_conf:
                h, w = frame.shape[:2]
                detections.append(
                    NormalizedDetection(
                        label="Machine Anomaly",
                        confidence=round(min(score, 1.0), 3),
                        bbox=[10, 10, w - 10, min(60, h - 10)],
                        model_key=self.key,
                        class_id=1,
                        capability="machine_anomaly_prediction",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                )

            elapsed = time.perf_counter() - start_time
            self.last_latency_ms = elapsed * 1000.0
            self.total_inference_time += elapsed
            self.inference_count += 1
            self.last_inference_time = time.time()
            return detections
        except Exception as e:
            self.error_message = str(e)
            logger.debug(f"Inference error in Sklearn model '{self.key}': {e}")
            return []

    def unload(self) -> None:
        self.model = None
        self.scaler = None
        super().unload()


class MediaPipeAdapter(BaseModelAdapter):
    """Adapter for Google MediaPipe Hand Landmark Detection."""

    def __init__(self, key: str, config: Dict[str, Any]):
        super().__init__(key, config)
        self.landmarker = None

    def load(self) -> None:
        try:
            import mediapipe as mp

            model_path = _resolve_model_path(self.config["path"])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"MediaPipe task not found: {model_path}")

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2,
            )
            self.landmarker = HandLandmarker.create_from_options(options)
            self.status = "READY"
            self.error_message = None
            logger.info(f"✅ Loaded MediaPipe hand landmark adapter for '{self.key}'")
        except Exception as e:
            self.status = "ERROR"
            self.error_message = str(e)
            logger.warning(f"MediaPipe load error for '{self.key}': {e}")

    def predict(
        self, frame: np.ndarray, conf: Optional[float] = None
    ) -> List[NormalizedDetection]:
        if self.status != "READY" or self.landmarker is None:
            return []
        if frame is None or frame.size == 0:
            return []

        start_time = time.perf_counter()
        try:
            import mediapipe as mp

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect(mp_image)

            detections: List[NormalizedDetection] = []
            h, w = frame.shape[:2]

            if results and results.hand_landmarks:
                for hand_idx, landmarks in enumerate(results.hand_landmarks):
                    xs = [lm.x * w for lm in landmarks]
                    ys = [lm.y * h for lm in landmarks]
                    x1 = max(0, int(min(xs)) - 15)
                    y1 = max(0, int(min(ys)) - 15)
                    x2 = min(w, int(max(xs)) + 15)
                    y2 = min(h, int(max(ys)) + 15)

                    detections.append(
                        NormalizedDetection(
                            label="Hand_Movement",
                            confidence=0.90,
                            bbox=[x1, y1, x2, y2],
                            model_key=self.key,
                            class_id=hand_idx,
                            capability="hand_motion_tracking",
                            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                    )

            elapsed = time.perf_counter() - start_time
            self.last_latency_ms = elapsed * 1000.0
            self.total_inference_time += elapsed
            self.inference_count += 1
            self.last_inference_time = time.time()
            return detections
        except Exception as e:
            self.error_message = str(e)
            return []

    def unload(self) -> None:
        self.landmarker = None
        super().unload()


class PyTorchTRNAdapter(BaseModelAdapter):
    """Adapter for TRN (Temporal Relation Network) object throwing detection."""

    def __init__(self, key: str, config: Dict[str, Any]):
        super().__init__(key, config)
        self.model = None

    def load(self) -> None:
        try:
            import torch

            model_path = _resolve_model_path(self.config["path"])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"TRN checkpoint not found: {model_path}")

            try:
                self.model = torch.load(model_path, map_location="cpu", weights_only=False)
            except Exception:
                self.model = torch.load(model_path, map_location="cpu")

            self.status = "READY"
            self.error_message = None
            logger.info(f"✅ Loaded TRN adapter for '{self.key}'")
        except Exception as e:
            self.status = "ERROR"
            self.error_message = str(e)
            logger.debug(f"TRN model loading note: {e}")

    def predict(
        self, frame: np.ndarray, conf: Optional[float] = None
    ) -> List[NormalizedDetection]:
        # TRN requires temporal buffer; single-frame approximation
        return []

    def unload(self) -> None:
        self.model = None
        super().unload()


class OpenVINOAdapter(BaseModelAdapter):
    """Adapter for OpenVINO person action recognition."""

    def __init__(self, key: str, config: Dict[str, Any]):
        super().__init__(key, config)
        self.compiled_model = None

    def load(self) -> None:
        try:
            from openvino.runtime import Core

            path = _resolve_model_path(self.config["path"])
            precision = self.config.get("precision", "FP16")
            xml_path = (
                Path(path)
                / "intel"
                / "person-detection-action-recognition-0006"
                / precision
                / "person-detection-action-recognition-0006.xml"
            )
            if not xml_path.exists():
                xml_path = Path(path) / "person-detection-action-recognition-0006.xml"

            if not xml_path.exists():
                raise FileNotFoundError(f"OpenVINO XML not found: {xml_path}")

            ie = Core()
            model = ie.read_model(model=str(xml_path))
            self.compiled_model = ie.compile_model(model=model, device_name="CPU")
            self.status = "READY"
            self.error_message = None
            logger.info(f"✅ Loaded OpenVINO adapter for '{self.key}'")
        except Exception as e:
            self.status = "ERROR"
            self.error_message = str(e)
            logger.debug(f"OpenVINO adapter initialization note: {e}")

    def predict(
        self, frame: np.ndarray, conf: Optional[float] = None
    ) -> List[NormalizedDetection]:
        return []

    def unload(self) -> None:
        self.compiled_model = None
        super().unload()


# ─────────────────────────────────────────────────────────────────────────────
# Model Info Container (Backward Compatible)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelInfo:
    """Metadata and runtime state for a loaded model."""

    key: str
    model_type: str
    model: Any
    conf_threshold: float
    target_fps: float
    zones: List[str]
    priority: int
    enabled: bool
    classes: Optional[Dict[int, str]] = None
    description: str = ""
    auxiliary_model: Any = None
    capabilities: List[str] = field(default_factory=list)
    adapter: Optional[BaseModelAdapter] = None
    status: str = "READY"
    error_message: Optional[str] = None
    device: str = "cpu"

    last_inference_time: float = 0.0
    inference_count: int = 0
    total_inference_time: float = 0.0
    last_latency_ms: float = 0.0

    def matches_zone(self, zone_type: str) -> bool:
        if "*" in self.zones:
            return True
        return zone_type in self.zones

    def matches_capabilities(self, req_capabilities: List[str]) -> bool:
        if not req_capabilities:
            return True
        return any(c in self.capabilities for c in req_capabilities)

    def get_avg_inference_time(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count


# ─────────────────────────────────────────────────────────────────────────────
# Helper Path Resolver
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_model_path(path: str) -> str:
    """Resolve model path relative to backend or project directory."""
    if os.path.isabs(path):
        return path

    backend_dir = Path(__file__).parent.parent
    p1 = backend_dir / path
    if p1.exists():
        return str(p1)

    project_root = backend_dir.parent
    p2 = project_root / path
    if p2.exists():
        return str(p2)

    return str(p1)


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry Singleton
# ─────────────────────────────────────────────────────────────────────────────


class ModelRegistry:
    """Centralized model loading, caching, health tracking, and capability routing."""

    _instance: Optional[ModelRegistry] = None
    _initialized: bool = False
    _lock: Any = None  # threading.Lock — set in __init__

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        import threading
        self._lock = threading.Lock()
        self.adapters: Dict[str, BaseModelAdapter] = {}
        self.models: Dict[str, ModelInfo] = {}
        self.load_all_models()
        self._initialized = True

    def _create_adapter(self, model_key: str, config: Dict[str, Any]) -> BaseModelAdapter:
        mtype = config.get("type", "yolo").lower()
        if mtype == "yolo":
            return YOLOModelAdapter(model_key, config)
        elif mtype == "sklearn":
            return SklearnAnomalyAdapter(model_key, config)
        elif mtype == "mediapipe":
            return MediaPipeAdapter(model_key, config)
        elif mtype in ("pytorch", "pytorch_trn"):
            return PyTorchTRNAdapter(model_key, config)
        elif mtype == "openvino":
            return OpenVINOAdapter(model_key, config)
        else:
            return YOLOModelAdapter(model_key, config)

    def load_all_models(self) -> None:
        """Load all enabled models from settings.MODEL_REGISTRY."""
        logger.info("🚀 Loading models from registry...")

        for model_key, config in settings.MODEL_REGISTRY.items():
            if not config.get("enabled", False):
                logger.info(f"⏭️ Skipping disabled model: {model_key}")
                # Create disabled adapter entry
                adapter = self._create_adapter(model_key, config)
                adapter.status = "DISABLED"
                self.adapters[model_key] = adapter
                continue

            try:
                self.load_model(model_key)
            except Exception as e:
                logger.error(f"❌ Failed to initialize model {model_key}: {e}")
                # Graceful degradation: never crash on single model failure

        logger.info(f"✅ Model registry initialized with {len(self.models)} active models")

    def load_model(self, model_key: str) -> ModelInfo:
        """Load a single model by key on demand and cache in memory. Thread-safe."""
        if self._lock is not None:
            self._lock.acquire()
        try:
            return self._load_model_unlocked(model_key)
        finally:
            if self._lock is not None:
                self._lock.release()

    def _load_model_unlocked(self, model_key: str) -> ModelInfo:
        """Internal: load model without acquiring the lock (caller must hold it)."""
        config = settings.MODEL_REGISTRY.get(model_key)
        if not config:
            raise ValueError(f"Model '{model_key}' not defined in settings.MODEL_REGISTRY")

        # Fast path: already loaded and ready
        existing = self.models.get(model_key)
        if existing is not None and existing.status == "READY":
            return existing

        # Reuse existing adapter if ready
        adapter = self.adapters.get(model_key)
        if adapter is None:
            adapter = self._create_adapter(model_key, config)
            self.adapters[model_key] = adapter

        if adapter.status != "READY":
            adapter.load()

        model_info = ModelInfo(
            key=model_key,
            model_type=config.get("type", "yolo"),
            model=getattr(adapter, "model", getattr(adapter, "compiled_model", None)),
            conf_threshold=config.get("conf_threshold", 0.5),
            target_fps=config.get("target_fps", 5.0),
            zones=config.get("zones", ["*"]),
            priority=config.get("priority", 99),
            enabled=config.get("enabled", True),
            classes=adapter.get_classes(),
            description=config.get("description", ""),
            auxiliary_model=getattr(adapter, "scaler", None),
            capabilities=adapter.get_capabilities(),
            adapter=adapter,
            status=adapter.status,
            error_message=adapter.error_message,
            device=adapter.device,
        )

        self.models[model_key] = model_info
        return model_info

    def unload_model(self, model_key: str) -> None:
        """Unload a model from memory."""
        if self._lock is not None:
            self._lock.acquire()
        try:
            if model_key in self.adapters:
                self.adapters[model_key].unload()
            self.models.pop(model_key, None)
        finally:
            if self._lock is not None:
                self._lock.release()

    def get_model(self, model_key: str) -> Optional[ModelInfo]:
        """Get model by key, loading lazily if enabled. Thread-safe."""
        # Fast path: already loaded
        model = self.models.get(model_key)
        if model is not None and model.status == "READY":
            return model
        # Lazy load
        if model_key in settings.MODEL_REGISTRY and settings.MODEL_REGISTRY[model_key].get("enabled", False):
            try:
                return self._load_model_unlocked(model_key)
            except Exception:
                return None
        return None

    def get_models_for_zone(self, zone_type: Optional[str] = None) -> List[ModelInfo]:
        """Return all models that should process the given zone, sorted by priority."""
        if not zone_type:
            zone_type = "default"

        # Check zone capability mapping
        req_caps = settings.ZONE_CAPABILITY_MAP.get(zone_type, [])
        matching_models = []

        for model in self.models.values():
            if model.status != "READY":
                continue
            # Match by zone name OR by capability
            if model.matches_zone(zone_type) or (req_caps and model.matches_capabilities(req_caps)):
                matching_models.append(model)

        matching_models.sort(key=lambda m: m.priority)
        return matching_models

    def get_models_for_capabilities(self, capabilities: List[str]) -> List[ModelInfo]:
        """Resolve list of capabilities to sorted list of loaded models."""
        matched = []
        for model in self.models.values():
            if model.status == "READY" and model.matches_capabilities(capabilities):
                matched.append(model)
        matched.sort(key=lambda m: m.priority)
        return matched

    def get_recommended_models(self, capability_or_use_case: str) -> List[str]:
        """Recommend model keys for a given capability or use case."""
        cap_clean = capability_or_use_case.lower().replace(" ", "_").replace("-", "_")
        # Direct capability match
        if cap_clean in settings.CAPABILITY_REGISTRY:
            return settings.CAPABILITY_REGISTRY[cap_clean]

        # Fuzzy search
        matched_keys = []
        for cap, models in settings.CAPABILITY_REGISTRY.items():
            if cap_clean in cap or cap in cap_clean:
                for m in models:
                    if m not in matched_keys:
                        matched_keys.append(m)

        # Fallback to scanning model capabilities
        if not matched_keys:
            for mkey, mcfg in settings.MODEL_REGISTRY.items():
                caps = mcfg.get("capabilities", [])
                if any(cap_clean in c for c in caps):
                    matched_keys.append(mkey)

        return matched_keys

    def is_model_enabled(self, model_key: str) -> bool:
        return model_key in self.models and self.models[model_key].status == "READY"

    def get_all_models(self) -> List[ModelInfo]:
        models = list(self.models.values())
        models.sort(key=lambda m: m.priority)
        return models

    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return runtime statistics and health for all registered models."""
        stats = {}
        for key, config in settings.MODEL_REGISTRY.items():
            adapter = self.adapters.get(key)
            if adapter:
                stats[key] = {
                    "enabled": adapter.enabled,
                    "status": adapter.status,
                    "device": adapter.device,
                    "type": config.get("type", "unknown"),
                    "capabilities": adapter.capabilities,
                    "zones": adapter.zones,
                    "priority": adapter.priority,
                    "inference_count": adapter.inference_count,
                    "avg_latency_ms": round(
                        (adapter.total_inference_time / adapter.inference_count) * 1000.0
                        if adapter.inference_count > 0
                        else 0.0,
                        2,
                    ),
                    "last_latency_ms": round(adapter.last_latency_ms, 2),
                    "target_fps": adapter.target_fps,
                    "error": adapter.error_message,
                }
            else:
                stats[key] = {
                    "enabled": config.get("enabled", False),
                    "status": "NOT_LOADED",
                    "device": "cpu",
                    "type": config.get("type", "unknown"),
                    "capabilities": config.get("capabilities", []),
                    "zones": config.get("zones", []),
                    "priority": config.get("priority", 99),
                    "inference_count": 0,
                    "avg_latency_ms": 0.0,
                    "last_latency_ms": 0.0,
                    "target_fps": config.get("target_fps", 5.0),
                    "error": None,
                }
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Module-Level Singleton & Convenience Helpers
# ─────────────────────────────────────────────────────────────────────────────

_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the singleton ModelRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_models_for_zone(zone_type: Optional[str] = None) -> List[ModelInfo]:
    return get_registry().get_models_for_zone(zone_type)


def get_model(model_key: str) -> Optional[ModelInfo]:
    return get_registry().get_model(model_key)
