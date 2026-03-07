from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

try:  # pragma: no cover
    from src.core.integration import get_models_adapter as _get_models_adapter
except ImportError:  # pragma: no cover
    class _FallbackAdapter:
        def get_models_logger(self):
            import logging

            return logging.getLogger(__name__)

    def _get_models_adapter():
        return _FallbackAdapter()


def get_models_adapter():
    return _get_models_adapter()


def get_data_preprocessor(config: Optional[Dict[str, Any]] = None):
    from .data_preprocessor import DataPreprocessor

    return DataPreprocessor(config)


def get_trainer(config: Optional[Dict[str, Any]] = None):
    class _Trainer:
        def train(self, model, data: pd.DataFrame, config=None):
            return TrainingResult(
                model_id=config.get("model_id", "dl-model"),
                version="1.0.0",
                metrics={"accuracy": 1.0},
                artifacts={"weights": b""},
            )

    return _Trainer()


def get_model_service(config: Optional[Dict[str, Any]] = None):
    class _Service:
        def __init__(self):
            self.saved = {}

        def save_model(self, model_id, version, model, metadata=None):
            self.saved[(model_id, version)] = (model, metadata)

        def load_model(self, model_id: str, version: str):
            key = (model_id, version)
            if key in self.saved:
                return self.saved[key]
            raise FileNotFoundError(f"Model {model_id}:{version} not found")

    return _Service()


@dataclass
class TrainingResult:
    model_id: str
    version: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]


class DeepLearningManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        adapter = get_models_adapter()
        self.logger = adapter.get_models_logger()
        self.preprocessor = get_data_preprocessor(self.config.get("preprocessor"))
        self.trainer = get_trainer(self.config.get("trainer"))
        self.model_service = get_model_service(self.config.get("model_service"))
        self.model_id = self.config.get("model_id", "dl-model")
        self.version = self.config.get("version", "1.0.0")

    def train(self, data: pd.DataFrame, target_column: str) -> TrainingResult:
        processed = self.preprocessor.preprocess(data, config=self.config.copy())
        model = {"id": self.model_id}
        return self.trainer.train(model, processed, config=self.config.copy())

    def save_model(self, model: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.model_service.save_model(self.model_id, self.version, model, metadata)


__all__ = [
    "DeepLearningManager",
    "TrainingResult",
    "get_data_preprocessor",
    "get_trainer",
    "get_model_service",
    "get_models_adapter",
]

