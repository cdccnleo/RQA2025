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


import os
import json
import pandas as pd
import pytest
from pathlib import Path

from src.data.export.data_exporter import DataExporter


class _Model:
    def __init__(self, df: pd.DataFrame, metadata: dict):
        self.data = df
        self._metadata = metadata

    def get_data(self):
        return self.data

    def get_metadata(self):
        return self._metadata


def test_export_csv_and_history(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    m = _Model(df, {"source": "unit", "symbol": "TEST"})
    exp = DataExporter(str(tmp_path))
    out = exp.export(m, "csv", filename="unit_export.csv", include_metadata=True)
    assert Path(out).exists()
    meta_file = Path(out).with_suffix(".metadata.json")
    assert meta_file.exists()
    hist = exp.get_export_history()
    assert len(hist) >= 1 and hist[-1]["format"] == "csv"


def test_export_to_buffer_json_and_pickle():
    df = pd.DataFrame({"a": [1]})
    m = _Model(df, {"source": "unit", "symbol": "X"})
    exp = DataExporter("reports")  # 使用现有可写目录
    buf_json = exp.export_to_buffer(m, "json", include_metadata=True)
    j = json.loads(buf_json.decode("utf-8"))
    assert "data" in j and "metadata" in j
    buf_pickle = exp.export_to_buffer(m, "pickle", include_metadata=False)
    assert isinstance(buf_pickle, (bytes, bytearray))


def test_export_multiple_zip(tmp_path: Path):
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})
    m1 = _Model(df1, {"source": "unit", "symbol": "S1"})
    m2 = _Model(df2, {"source": "unit", "symbol": "S2"})
    exp = DataExporter(str(tmp_path))
    zip_path = exp.export_multiple([m1, m2], "csv", zip_filename="batch.zip", include_metadata=False)
    assert Path(zip_path).exists()


def test_export_unsupported_format_raises(tmp_path: Path):
    df = pd.DataFrame({"a": [1]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    with pytest.raises(Exception):
        exp.export(m, "unknownfmt")


