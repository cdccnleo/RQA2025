import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field

import pandas as pd

def _load_feature_engineer():
    from importlib import import_module

    module = import_module("src.ml.engine.feature_engineering")
    return getattr(module, "FeatureEngineer")

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    from src.infrastructure.integration import get_models_adapter as _get_models_adapter
except ImportError:  # pragma: no cover
    class _FallbackAdapter:
        def get_models_logger(self):
            return logging.getLogger(__name__)

    def _get_models_adapter():
        return _FallbackAdapter()


def get_models_adapter():
    return _get_models_adapter()


def get_feature_engineer(config: Optional[Dict[str, Any]] = None):
    factory = globals().get("FeatureEngineer")
    return factory(config)


def get_model_trainer(config: Optional[Dict[str, Any]] = None):
    class _Trainer:
        def train(self, data: pd.DataFrame, labels: pd.Series, cfg: Dict[str, Any]):
            return {"model": {"weights": len(data)}, "config": cfg}

    return _Trainer()


def get_model_predictor(config: Optional[Dict[str, Any]] = None):
    class _Predictor:
        def predict(self, model: Dict[str, Any], data: pd.DataFrame):
            return [0.0] * len(data)

    return _Predictor()


@dataclass
class FeatureEngineeringComponent:
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        adapter = get_models_adapter()
        self.logger = adapter.get_models_logger()
        self.pipeline_name = self.config.get("pipeline_name")
        self.engineer = get_feature_engineer(self.config.get("feature_engineer"))

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.pipeline_name:
            raise ValueError("pipeline_name 未配置")
        return self.engineer.process_data(data, self.pipeline_name)


@dataclass
class ModelTrainingComponent:
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        adapter = get_models_adapter()
        self.logger = adapter.get_models_logger()
        self.trainer = get_model_trainer(self.config.get("trainer"))

    def train(self, data: pd.DataFrame, label_column: str) -> Dict[str, Any]:
        if label_column not in data:
            raise ValueError("label_column 不存在")
        features = data.drop(columns=[label_column])
        labels = data[label_column]
        result = self.trainer.train(features, labels, self.config.copy())
        return result


@dataclass
class PredictionPipeline:
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        adapter = get_models_adapter()
        self.logger = adapter.get_models_logger()
        self.trainer = get_model_trainer(self.config.get("trainer"))
        self.predictor = get_model_predictor(self.config.get("predictor"))
        self.model: Optional[Any] = None

    def train(self, data: pd.DataFrame, label_column: str) -> None:
        if label_column not in data:
            raise ValueError("label_column 不存在")
        features = data.drop(columns=[label_column])
        labels = data[label_column]
        result = self.trainer.train(features, labels, self.config.copy())
        self.model = result.get("model")

    def predict(self, data: pd.DataFrame):
        if self.model is None:
            raise ValueError("模型尚未训练")
        return self.predictor.predict(self.model, data)


__all__ = [
    "FeatureEngineeringComponent",
    "ModelTrainingComponent",
    "PredictionPipeline",
    "FeatureEngineer",
    "get_feature_engineer",
    "get_model_trainer",
    "get_model_predictor",
    "get_models_adapter",
]


FeatureEngineer = _load_feature_engineer()
