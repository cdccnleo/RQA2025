from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Iterator
import os

import pandas as pd


@dataclass
class DataBatch:
    """数据批次"""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    batch_id: int = 0


@dataclass
class PipelineSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class DataSource:
    """数据源基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False

    def connect(self) -> bool:
        """连接数据源"""
        self.connected = True
        return True

    def disconnect(self) -> None:
        """断开连接"""
        self.connected = False

    def read_data(self, batch_size: int = 100) -> Iterator[DataBatch]:
        """读取数据"""
        raise NotImplementedError


class CSVDataSource(DataSource):
    """CSV数据源"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.file_path = config.get('file_path', '')
        self.chunk_size = config.get('chunk_size', 1000)
        self.current_chunk = 0

    def read_data(self, batch_size: int = 100) -> Iterator[DataBatch]:
        """读取CSV数据"""
        if not self.connected:
            raise RuntimeError("数据源未连接")

        try:
            # 读取CSV文件
            df = pd.read_csv(self.file_path)

            # 分批返回数据
            for i in range(0, len(df), batch_size):
                batch_data = df.iloc[i:i+batch_size].copy()
                batch = DataBatch(
                    data=batch_data,
                    metadata={
                        'source': 'csv',
                        'file_path': self.file_path,
                        'batch_size': len(batch_data),
                        'total_rows': len(df)
                    },
                    batch_id=self.current_chunk
                )
                self.current_chunk += 1
                yield batch

        except Exception as e:
            raise RuntimeError(f"读取CSV文件失败: {e}")


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

    def create_data_source(self, config: Dict[str, Any]) -> DataSource:
        """创建数据源"""
        source_type = config.get('type', 'csv')
        if source_type == 'csv':
            return CSVDataSource(config)
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")


__all__ = ["DataPipeline", "PipelineSplit", "DataBatch", "DataSource", "CSVDataSource"]

