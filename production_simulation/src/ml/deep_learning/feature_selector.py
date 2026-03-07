from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class FeatureSelector:
    top_k: int = 5

    def select(self, data: pd.DataFrame) -> List[str]:
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        return numeric_cols[: self.top_k]


__all__ = ["FeatureSelector"]

