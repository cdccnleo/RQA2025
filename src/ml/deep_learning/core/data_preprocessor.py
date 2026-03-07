from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


class DataPreprocessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def preprocess(self, data: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if data.empty:
            raise ValueError("输入数据为空")
        cfg = config or self.config
        dropna = cfg.get("dropna", False)
        normalized = data.copy()
        if dropna:
            normalized = normalized.dropna()
        return normalized


__all__ = ["DataPreprocessor"]

