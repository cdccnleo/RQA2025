from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass
class PipelineSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class DataPipeline:
    """轻量级数据流水线，仅包含单测所需的拆分逻辑。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.test_size = self.config.get("test_size", 0.2)
        self.val_size = self.config.get("val_size", 0.1)

    def split(self, data: pd.DataFrame) -> PipelineSplit:
        if data.empty:
            raise ValueError("输入数据为空")

        total = len(data)
        test_count = max(1, int(total * self.test_size))
        val_count = max(1, int(total * self.val_size))

        test = data.iloc[-test_count:]
        remaining = data.iloc[:-test_count]
        validation = remaining.iloc[-val_count:]
        train = remaining.iloc[:-val_count] if val_count < len(remaining) else remaining.iloc[:0]

        return PipelineSplit(train=train.reset_index(drop=True),
                             validation=validation.reset_index(drop=True),
                             test=test.reset_index(drop=True))


__all__ = ["DataPipeline", "PipelineSplit"]

