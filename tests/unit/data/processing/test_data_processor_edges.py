import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pandas as pd
import numpy as np
from src.data.processing.data_processor import DataProcessor


class _FakeModel:
    def __init__(self, data: pd.DataFrame, frequency: str = "1d", metadata: dict = None):
        self.data = data
        self._frequency = frequency
        self._metadata = metadata or {}

    def get_frequency(self):
        return self._frequency

    def get_metadata(self):
        return self._metadata


def test_process_expected_dtypes_value_ranges_duplicates_zscore():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 3],
            "value": [10.0, 10.0, 1000.0, np.nan],  # 含重复与异常/缺失
            "category": [1.1, 1.1, 2.9, 3.0],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    model = _FakeModel(df, frequency="1d", metadata={"source": "unit"})
    p = DataProcessor()
    out = p.process(
        model,
        fill_method="mean",
        remove_duplicates=True,
        outlier_method="zscore",
        expected_dtypes={"category": "int64"},
        value_ranges={"value": (0, 100)},
    )
    assert isinstance(out, _FakeModel)
    # 断言类型转换与范围裁剪生效
    assert out.data["category"].dtype == "int64"
    assert out.data["value"].max() <= 100
    # 断言元数据包含处理信息
    assert "processed_at" in out._metadata and "processor" in out._metadata


def test_align_index_time_and_required_columns():
    df = pd.DataFrame(
        {
            "id": [10, 11],
            "date": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "v": [1.0, 2.0],
        }
    )
    model = _FakeModel(df, frequency="1d", metadata={"source": "unit"})
    p = DataProcessor()
    out = p.process(
        model,
        index_col="id",
        time_col="date",
        required_columns=["missing_col"],
    )
    assert "missing_col" in out.data.columns
    # 已按时间排序并设置为索引
    assert out.data.index.name in {"date", "id"}


def test_empty_dataframe_returns_original():
    df = pd.DataFrame(columns=["a", "b"])
    model = _FakeModel(df, frequency="1d", metadata={"source": "unit"})
    p = DataProcessor()
    out = p.process(model)
    # 空数据应直接返回原始对象
    assert out is model


