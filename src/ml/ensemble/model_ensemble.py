#!/usr/bin/env python3
"""轻量级模型集成实现（满足单测需求）。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


class EnsembleMethod(Enum):
    AVERAGE = "average"
    WEIGHTED = "weighted"


class WeightUpdateRule(Enum):
    EQUAL = "equal"
    PERFORMANCE = "performance"


@dataclass
class EnsembleResult:
    prediction: np.ndarray
    model_weights: Dict[str, float]
    performance_metrics: Dict[str, float]
    uncertainty: Optional[np.ndarray] = None


class ModelEnsemble:
    def __init__(self, models: Dict[str, object]):
        if not models:
            raise ValueError("模型字典不能为空")
        self.models = models

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Optional[EnsembleResult]:
        warnings.warn("ModelEnsemble 基类未实现具体预测逻辑。")
        return None


class AverageEnsemble(ModelEnsemble):
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> EnsembleResult:
        stacked = np.stack(
            [model.predict(X) for model in self.models.values()], axis=0  # type: ignore
        ).astype(np.float64)
        avg_pred = stacked.mean(axis=0)
        weights = {name: 1.0 / len(self.models) for name in self.models}
        uncertainty = stacked.var(axis=0)
        return EnsembleResult(avg_pred, weights, {}, uncertainty)


class WeightedEnsemble(ModelEnsemble):
    def __init__(
        self,
        models: Dict[str, object],
        update_rule: WeightUpdateRule = WeightUpdateRule.EQUAL,
    ):
        super().__init__(models)
        self.update_rule = update_rule
        self.performance_history = {name: [] for name in models}

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> EnsembleResult:
        predictions = {
            name: model.predict(X).astype(np.float64)  # type: ignore
            for name, model in self.models.items()
        }
        stacked = np.stack(list(predictions.values()), axis=0)

        weights = self._compute_weights(predictions, y)

        weighted_pred = np.zeros_like(stacked[0])
        for (name, pred) in predictions.items():
            weighted_pred += weights[name] * pred

        metrics: Dict[str, float] = {}
        if y is not None:
            metrics["accuracy"] = self._accuracy(weighted_pred, y)

        uncertainty = stacked.var(axis=0)
        return EnsembleResult(weighted_pred, weights, metrics, uncertainty)

    def _compute_weights(
        self,
        predictions: Dict[str, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> Dict[str, float]:
        if self.update_rule == WeightUpdateRule.PERFORMANCE and y is not None:
            scores = {}
            for name, pred in predictions.items():
                mse = float(np.mean((pred - y) ** 2) + 1e-8)
                score = 1.0 / mse
                self.performance_history[name].append(score)
                scores[name] = score
            total = sum(scores.values())
            return {name: score / total for name, score in scores.items()}

        return {name: 1.0 / len(predictions) for name in predictions}

    def _accuracy(self, prediction: np.ndarray, y: Union[pd.Series, np.ndarray]) -> float:
        if prediction.ndim == 1:
            return float(np.mean((prediction > 0.5).astype(int) == y))
        labels = np.argmax(prediction, axis=1)
        return float(np.mean(labels == y))


class EnsembleMonitor:
    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        self.model_performance = {name: [] for name in model_names}
        self.ensemble_performance: List[float] = []
        self.correlation_matrix = pd.DataFrame()

    def update(
        self,
        predictions: Dict[str, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        ensemble_pred: np.ndarray,
    ) -> None:
        for name, pred in predictions.items():
            mse = float(np.mean((pred - y) ** 2))
            self.model_performance[name].append(mse)

        mse_ensemble = float(np.mean((ensemble_pred - y) ** 2))
        self.ensemble_performance.append(mse_ensemble)

        df = pd.DataFrame(predictions)
        if not df.empty:
            self.correlation_matrix = df.corr()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_performance": {
                name: np.mean(scores) if scores else None
                for name, scores in self.model_performance.items()
            },
            "ensemble_performance": {
                "mean": np.mean(self.ensemble_performance) if self.ensemble_performance else None,
                "history": self.ensemble_performance,
            },
            "correlation_matrix": self.correlation_matrix,
        }


__all__ = [
    "AverageEnsemble",
    "WeightedEnsemble",
    "ModelEnsemble",
    "EnsembleResult",
    "WeightUpdateRule",
    "EnsembleMethod",
    "EnsembleMonitor",
]

