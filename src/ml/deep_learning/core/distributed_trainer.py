from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class DistributedConfig:
    epochs: int = 1
    learning_rate: float = 0.001


class DistributedTrainer:
    """占位式分布式训练器，仅返回输入配置。"""

    def __init__(self, config: DistributedConfig):
        self.config = config

    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        if data.empty:
            raise ValueError("训练数据不能为空")
        return {
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "metrics": {"loss": 0.0},
        }


__all__ = ["DistributedTrainer", "DistributedConfig"]

