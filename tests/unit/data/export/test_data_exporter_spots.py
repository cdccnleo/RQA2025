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


import json
from pathlib import Path

import pandas as pd
import pytest

import sys
import types

# 兼容导入：确保 src.interfaces 提供 IDataModel 以避免并行环境下的解析歧义
if "src.interfaces" not in sys.modules:
    _mod = types.ModuleType("src.interfaces")
    class IDataModel:  # type: ignore
        pass
    _mod.IDataModel = IDataModel  # type: ignore
    sys.modules["src.interfaces"] = _mod

from src.data.export.data_exporter import DataExporter


class _StubModel:
    def __init__(self):
        self._df = pd.DataFrame(
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        )
        self._meta = {"source": "stub", "symbol": "TEST"}

    def get_data(self):
        return self._df

    def get_metadata(self):
        return self._meta


def test_load_history_handles_corrupted_json(tmp_path: Path):
    # 准备损坏的历史文件
    exp = DataExporter(str(tmp_path))
    history_file = tmp_path / "export_history.json"
    history_file.write_text("{invalid json")
    # 重新初始化以触发 _load_history 异常分支
    exp2 = DataExporter(str(tmp_path))
    assert isinstance(exp2.history, list)
    assert exp2.history == []


def test_export_excel_writes_metadata_sheet(tmp_path: Path):
    model = _StubModel()
    exporter = DataExporter(str(tmp_path))
    # 导出 Excel，期待写入 Metadata sheet
    out = exporter.export(model, "excel", filename="data.xlsx", include_metadata=True)
    out_path = Path(out)
    assert out_path.exists()
    # 读取并校验两个sheet是否存在
    xl = pd.ExcelFile(out_path)
    assert set(xl.sheet_names) >= {"Data", "Metadata"}


def test_export_pickle_with_and_without_metadata(tmp_path: Path):
    model = _StubModel()
    exporter = DataExporter(str(tmp_path))
    # with metadata
    p1 = exporter.export(model, "pickle", filename="with_meta.pkl", include_metadata=True)
    obj1 = pd.read_pickle(p1)
    assert "data" in obj1 and "metadata" in obj1
    # without metadata
    p2 = exporter.export(model, "pickle", filename="no_meta.pkl", include_metadata=False)
    obj2 = pd.read_pickle(p2)
    assert "data" in obj2 and obj2.get("metadata") is None


def test_export_to_buffer_json_includes_metadata(tmp_path: Path):
    model = _StubModel()
    exporter = DataExporter(str(tmp_path))
    buf = exporter.export_to_buffer(model, "json", include_metadata=True)
    payload = json.loads(buf.decode("utf-8"))
    assert "data" in payload and "metadata" in payload
    assert payload["metadata"]["source"] == "stub"

def test_export_parquet_writes_sidecar_metadata_file(tmp_path: Path, monkeypatch):
    model = _StubModel()
    exporter = DataExporter(str(tmp_path))

    # 伪造 DataFrame.to_parquet，避免依赖 pyarrow/fastparquet
    calls = {"to_parquet": 0}
    def _fake_to_parquet(self, path, **kwargs):
        calls["to_parquet"] += 1
        Path(path).write_bytes(b"PARQUET")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet, raising=True)

    out = exporter.export(model, "parquet", filename="data.parquet", include_metadata=True)
    out_path = Path(out)
    assert out_path.exists()
    assert calls["to_parquet"] == 1
    # 校验旁路元数据文件
    meta_path = out_path.with_suffix(".metadata.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta.get("source") == "stub"

def test_export_hdf_puts_metadata_series(tmp_path: Path, monkeypatch):
    model = _StubModel()
    exporter = DataExporter(str(tmp_path))

    class _FakeHDF:
        def __init__(self, path):
            self.path = path
            self.put_calls = []
            Path(path).write_bytes(b"HDF")
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def put(self, key, obj, **kwargs):
            # 记录 put 调用（data 与 metadata）
            self.put_calls.append(key)

    fake_store_calls = {"instances": []}
    def _fake_HDFStore(path):
        obj = _FakeHDF(path)
        fake_store_calls["instances"].append(obj)
        return obj

    monkeypatch.setattr(pd, "HDFStore", _fake_HDFStore, raising=True)

    out = exporter.export(model, "hdf", filename="data.h5", include_metadata=True)
    out_path = Path(out)
    assert out_path.exists()
    # 校验 put 调用包含 data 与 metadata
    assert fake_store_calls["instances"], "HDFStore should be instantiated"
    puts = fake_store_calls["instances"][0].put_calls
    assert "data" in puts and "metadata" in puts


