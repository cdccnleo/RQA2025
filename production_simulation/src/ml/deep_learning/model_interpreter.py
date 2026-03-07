from __future__ import annotations

from typing import Any, Dict


class ModelInterpreter:
    def explain(self, model: Any, data: Any) -> Dict[str, float]:
        return {"importance": 1.0}


__all__ = ["ModelInterpreter"]

