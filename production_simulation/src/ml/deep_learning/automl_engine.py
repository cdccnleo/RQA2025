from __future__ import annotations

from typing import Any, Dict, Optional

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


def get_automl_pipeline(config: Optional[Dict[str, Any]] = None):
    class _Pipeline:
        def fit(self, data, target_column, config=None):
            return {"status": "trained"}

        def predict(self, data):
            return [0] * len(data)

    return _Pipeline()


def get_evaluator(config: Optional[Dict[str, Any]] = None):
    class _Evaluator:
        def evaluate(self, predictions, actual, metrics=None):
            return {"accuracy": 0.0}

    return _Evaluator()


class AutoMLEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        adapter = get_models_adapter()
        self.logger = adapter.get_models_logger()
        self.pipeline = get_automl_pipeline(self.config.get("pipeline"))
        self.evaluator = get_evaluator(self.config.get("evaluator"))
        self.enable_optuna = self.config.get("enable_optuna", True)
        self.health = {"runs": 0, "failures": 0}

    def train(self, data, target_column: str) -> Dict[str, Any]:
        try:
            result = self.pipeline.fit(data, target_column, config=self.config.copy())
            if self.enable_optuna:
                self.logger.info("AutoML training succeeded with optuna enabled")
            else:
                self.logger.warning("Optuna disabled, using basic pipeline")
            self.health["runs"] += 1
            return result
        except Exception as exc:  # pragma: no cover - safety net
            self.logger.error("AutoML training failed: %s", exc)
            self.health["failures"] += 1
            raise

    def evaluate(
        self,
        predictions,
        actual,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scores = self.evaluator.evaluate(predictions, actual, metrics)
        scores["runs"] = self.health["runs"]
        scores["failures"] = self.health["failures"]
        return scores


__all__ = ["AutoMLEngine", "get_automl_pipeline", "get_evaluator", "get_models_adapter"]

