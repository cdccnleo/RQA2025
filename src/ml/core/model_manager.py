from dataclasses import dataclass
from typing import Any, Optional

try:
    from src.ml.model_manager import ModelManager, ModelType  # re-export
except ImportError:
    from ml.model_manager import ModelManager, ModelType  # re-export


@dataclass
class ModelPrediction:
    request_id: str
    model_type: str
    prediction: Optional[Any]
    confidence: float
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None


__all__ = ["ModelManager", "ModelType", "ModelPrediction"]

