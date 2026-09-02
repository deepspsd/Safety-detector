"""
Model Registry - Multi-Model Management for OccuSafe
====================================================

Centralized model loading, caching, and routing for the multi-model detection pipeline.

Architecture:
- ModelRegistry: Singleton that loads and caches all enabled models
- get_models_for_zone(): Returns list of models that should process a given zone
- ModelInfo: Metadata container for each loaded model

Thread-Safety:
- All models are loaded once at startup
- Read-only access after initialization (thread-safe)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model Info Container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelInfo:
    """Metadata and runtime state for a loaded model."""

    key: str  # Model key from MODEL_REGISTRY
    model_type: str  # yolo, mediapipe, openvino, pytorch, pytorch_trn, sklearn
    model: Any  # Loaded model instance
    conf_threshold: float
    target_fps: float
    zones: List[str]
    priority: int
    enabled: bool
    classes: Optional[Dict[int, str]] = None
    description: str = ""
    auxiliary_model: Any = None  # Companion object e.g. sklearn scaler

    # Runtime state (frame rate limiting)
    last_inference_time: float = 0.0
    inference_count: int = 0
    total_inference_time: float = 0.0

    def matches_zone(self, zone_type: str) -> bool:
        """Check if this model should process the given zone type."""
        if "*" in self.zones:
            return True
        return zone_type in self.zones

    def get_avg_inference_time(self) -> float:
        """Get average inference time in seconds."""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry Singleton
# ─────────────────────────────────────────────────────────────────────────────


class ModelRegistry:
    """Centralized model loading and management."""

    _instance: Optional[ModelRegistry] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.models: Dict[str, ModelInfo] = {}
        self.load_all_models()
        self._initialized = True

    def load_all_models(self):
        """Load all enabled models from MODEL_REGISTRY."""
        logger.info("🚀 Loading models from registry...")

        for model_key, config in settings.MODEL_REGISTRY.items():
            if not config.get("enabled", False):
                logger.info(f"⏭️  Skipping disabled model: {model_key}")
                continue

            try:
                model_info = self._load_model(model_key, config)
                self.models[model_key] = model_info
                logger.info(
                    f"✅ Loaded {model_key} ({config['type']}) - "
                    f"zones: {config.get('zones', ['*'])}, priority: {config.get('priority', 99)}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to load model {model_key}: {e}", exc_info=True)
                # Continue loading other models

        logger.info(f"✅ Model registry initialized with {len(self.models)} models")

    def _load_model(self, model_key: str, config: Dict) -> ModelInfo:
        """Load a single model based on its type."""
        model_type = config["type"]
        model_path = self._resolve_model_path(config["path"])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model based on type
        if model_type == "yolo":
            model = self._load_yolo_model(model_path)
        elif model_type == "mediapipe":
            model = self._load_mediapipe_model(model_path)
        elif model_type == "openvino":
            model = self._load_openvino_model(model_path, config)
        elif model_type in ("pytorch", "pytorch_trn"):  
            model = self._load_pytorch_model(model_path)
        elif model_type == "sklearn":
            model = self._load_sklearn_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create ModelInfo
        # For sklearn models load companion scaler if path defined in config
        auxiliary = None
        if model_type == "sklearn":
            scaler_path_raw = config.get("scaler_path")
            if scaler_path_raw:
                scaler_path = self._resolve_model_path(scaler_path_raw)
                if os.path.exists(scaler_path):
                    import joblib
                    auxiliary = joblib.load(scaler_path)
                    logger.info(f"✅ Loaded scaler for {model_key} from {scaler_path}")
                else:
                    logger.warning(f"Scaler not found for {model_key}: {scaler_path}")

        return ModelInfo(
            key=model_key,
            model_type=model_type,
            model=model,
            conf_threshold=config.get("conf_threshold", 0.5),
            target_fps=config.get("target_fps", 5.0),
            zones=config.get("zones", ["*"]),
            priority=config.get("priority", 99),
            enabled=config.get("enabled", True),
            classes=config.get("classes"),
            description=config.get("description", ""),
            auxiliary_model=auxiliary,
        )

    def _resolve_model_path(self, path: str) -> str:
        """Resolve model path relative to backend directory."""
        if os.path.isabs(path):
            return path

        # Try relative to backend directory
        backend_dir = Path(__file__).parent.parent
        model_path = backend_dir / path

        if model_path.exists():
            return str(model_path)

        # Try relative to project root
        project_root = backend_dir.parent
        model_path = project_root / path

        if model_path.exists():
            return str(model_path)

        # Return as-is and let the error handler deal with it
        return path

    def _load_yolo_model(self, path: str) -> Any:
        """Load YOLO model using Ultralytics."""
        from ultralytics import YOLO

        model = YOLO(path)
        logger.debug(f"Loaded YOLO model from {path}")
        return model

    def _load_mediapipe_model(self, path: str) -> Any:
        """Load MediaPipe model."""
        try:
            import mediapipe as mp

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=path),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2,
            )

            landmarker = HandLandmarker.create_from_options(options)
            logger.debug(f"Loaded MediaPipe model from {path}")
            return landmarker
        except ImportError:
            logger.error("MediaPipe not installed. Run: pip install mediapipe")
            raise

    def _load_openvino_model(self, path: str, config: Dict) -> Any:
        """Load OpenVINO model."""
        try:
            from openvino.runtime import Core

            # OpenVINO models are directories with .xml/.bin files
            precision = config.get("precision", "FP32")
            precision_map = {
                "FP32": ("fp32_xml", "fp32_bin"),
                "FP16": ("fp16_xml", "fp16_bin"),
                "FP16-INT8": ("fp16_int8_xml", "fp16_int8_bin"),
            }

            xml_key, _ = precision_map[precision]

            # For person_action_recognition, construct path to XML
            if "person_action" in path:
                xml_path = Path(path) / "intel" / "person-detection-action-recognition-0006" / precision / "person-detection-action-recognition-0006.xml"
            else:
                xml_path = Path(path)

            if not xml_path.exists():
                raise FileNotFoundError(f"OpenVINO model XML not found: {xml_path}")

            ie = Core()
            model = ie.read_model(model=str(xml_path))
            compiled_model = ie.compile_model(model=model, device_name="CPU")

            logger.debug(f"Loaded OpenVINO model from {xml_path}")
            return compiled_model
        except ImportError:
            logger.error("OpenVINO not installed. Run: pip install openvino")
            raise

    def _load_pytorch_model(self, path: str) -> Any:
        """Load PyTorch model (handles PyTorch 2.6+ safe-globals requirement).

        MVTec anomaly model embeds pandas.DataFrame in its pickle.
        We try three strategies in order:
          1. weights_only=False (most permissive, always works for trusted files)
          2. add_safe_globals for pandas + weights_only=True (cleaner in 2.6+)
          3. bare torch.load (legacy fallback)
        """
        import torch

        # Strategy 1: permissive load (trusted local file)
        try:
            model = torch.load(path, map_location="cpu", weights_only=False)
            logger.debug(f"Loaded PyTorch model (weights_only=False) from {path}")
            return model
        except Exception as e1:
            logger.debug(f"weights_only=False failed ({e1}), trying safe_globals...")

        # Strategy 2: allowlist pandas globals for weights_only=True
        try:
            import pandas as pd
            with torch.serialization.safe_globals([
                pd.core.frame.DataFrame,
                pd.core.series.Series,
            ]):
                model = torch.load(path, map_location="cpu", weights_only=True)
            logger.debug(f"Loaded PyTorch model (safe_globals) from {path}")
            return model
        except Exception as e2:
            logger.debug(f"safe_globals load failed ({e2}), bare load fallback...")

        # Strategy 3: legacy bare load
        model = torch.load(path, map_location="cpu")
        logger.debug(f"Loaded PyTorch model (bare) from {path}")
        return model

    def _load_sklearn_model(self, path: str) -> Any:
        """Load scikit-learn / joblib model."""
        import joblib

        model = joblib.load(path)
        logger.debug(f"Loaded sklearn model from {path}")
        return model

    def get_model(self, model_key: str) -> Optional[ModelInfo]:
        """Get a specific model by key."""
        return self.models.get(model_key)

    def get_models_for_zone(
        self, zone_type: Optional[str] = None
    ) -> List[ModelInfo]:
        """
        Get all models that should process the given zone, sorted by priority.

        Args:
            zone_type: Zone type (e.g., "dough_mixing", "shop_counter")
                      If None or "default", returns only primary models.

        Returns:
            List of ModelInfo sorted by priority (0 = highest priority)
        """
        if zone_type is None:
            zone_type = "default"

        matching_models = [
            model for model in self.models.values() if model.matches_zone(zone_type)
        ]

        # Sort by priority (lower number = higher priority)
        matching_models.sort(key=lambda m: m.priority)

        return matching_models

    def get_all_models(self) -> List[ModelInfo]:
        """Get all loaded models, sorted by priority."""
        models = list(self.models.values())
        models.sort(key=lambda m: m.priority)
        return models

    def is_model_enabled(self, model_key: str) -> bool:
        """Check if a model is loaded and enabled."""
        return model_key in self.models

    def get_model_stats(self) -> Dict[str, Dict]:
        """Get runtime statistics for all models."""
        stats = {}
        for key, model in self.models.items():
            stats[key] = {
                "enabled": model.enabled,
                "type": model.model_type,
                "zones": model.zones,
                "priority": model.priority,
                "inference_count": model.inference_count,
                "avg_inference_time_ms": model.get_avg_inference_time() * 1000,
                "target_fps": model.target_fps,
            }
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton instance
# ─────────────────────────────────────────────────────────────────────────────

_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the singleton ModelRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_models_for_zone(zone_type: Optional[str] = None) -> List[ModelInfo]:
    """Convenience function to get models for a zone."""
    return get_registry().get_models_for_zone(zone_type)


def get_model(model_key: str) -> Optional[ModelInfo]:
    """Convenience function to get a specific model."""
    return get_registry().get_model(model_key)


# ─────────────────────────────────────────────────────────────────────────────
# Initialization check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test model loading
    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("MODEL REGISTRY TEST")
    print("="*70)

    registry = get_registry()

    print(f"\n✅ Loaded {len(registry.models)} models\n")

    for model_key, model_info in registry.models.items():
        print(f"📦 {model_key}")
        print(f"   Type: {model_info.model_type}")
        print(f"   Zones: {model_info.zones}")
        print(f"   Priority: {model_info.priority}")
        print(f"   Confidence: {model_info.conf_threshold}")
        print(f"   Target FPS: {model_info.target_fps}")
        print()

    # Test zone routing
    print("="*70)
    print("ZONE ROUTING TEST")
    print("="*70)

    test_zones = ["entrance", "dough_mixing", "shop_counter", "unknown"]

    for zone in test_zones:
        models = get_models_for_zone(zone)
        print(f"\n🏢 Zone: {zone}")
        print(f"   Models ({len(models)}): {[m.key for m in models]}")
