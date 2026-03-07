from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DeepLearningModel:
    name: str
    parameters: Dict[str, Any]

    def predict(self, inputs):
        return inputs


__all__ = ["DeepLearningModel"]

