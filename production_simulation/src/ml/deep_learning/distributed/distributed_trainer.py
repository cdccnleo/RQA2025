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


def get_distributed_backend(config: Optional[Dict[str, Any]] = None):
    class _Backend:
        def __init__(self):
            self.world_size = 1

        def barrier(self):
            return None

    return _Backend()


def get_trainer(config: Optional[Dict[str, Any]] = None):
    class _Trainer:
        def train(self, data, epochs, learning_rate):
            return {"epochs": epochs, "lr": learning_rate, "metrics": {"loss": 0.0}}

    return _Trainer()


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 0.001


class DistributedTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        adapter = get_models_adapter()
        self.logger = adapter.get_models_logger()
        self.backend = get_distributed_backend()
        self.trainer = get_trainer()

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        self.backend.barrier()
        return self.trainer.train(
            data, epochs=self.config.epochs, learning_rate=self.config.learning_rate
        )


__all__ = [
    "DistributedTrainer",
    "TrainingConfig",
    "get_distributed_backend",
    "get_trainer",
    "get_models_adapter",
]

