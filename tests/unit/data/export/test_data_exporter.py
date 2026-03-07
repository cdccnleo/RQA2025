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


import importlib
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

try:
    interfaces_module = importlib.import_module("src.interfaces")
except ModuleNotFoundError:
    interfaces_module = ModuleType("src.interfaces")
    sys.modules["src.interfaces"] = interfaces_module

if not hasattr(interfaces_module, "IDataModel"):
    class _IDataModel:
        ...

    interfaces_module.IDataModel = _IDataModel

from src.data.export.data_exporter import DataExporter
from src.infrastructure.utils.exceptions import DataLoaderError


class DummyDataModel:
    def __init__(self, symbol="AAA", source="test_source"):
        self.data = pd.DataFrame(
            {
                "value": [1, 2, 3],
                "symbol": [symbol] * 3,
            }
        )
        self._metadata = {
            "symbol": symbol,
            "source": source,
            "created_at": "2025-01-01",
        }

    def get_metadata(self):
        return dict(self._metadata)

    def get_data(self):
        return self.data


@pytest.fixture
def exporter(tmp_path):
    return DataExporter(export_dir=str(tmp_path / "exports"))


def test_export_csv_creates_file_and_history(exporter, tmp_path):
    model = DummyDataModel(symbol="TEST")
    output = exporter.export(model, "csv", filename="custom.csv")

    csv_path = Path(output)
    assert csv_path.exists()
    metadata_path = csv_path.with_suffix(".metadata.json")
    assert metadata_path.exists()

    with metadata_path.open() as f:
        meta = json.load(f)
    assert meta["symbol"] == "TEST"

    history_file = exporter.history_file
    with history_file.open() as f:
        history = json.load(f)
    assert history[-1]["format"] == "csv"
    assert Path(history[-1]["filepath"]).name == "custom.csv"


def test_export_unsupported_format_raises(exporter):
    model = DummyDataModel()
    with pytest.raises(DataLoaderError):
        exporter.export(model, "xml")


def test_export_multiple_creates_zip_with_files(exporter, tmp_path):
    models = [DummyDataModel(symbol="AAA"), DummyDataModel(symbol="BBB")]
    zip_path = exporter.export_multiple(models, format="csv", zip_filename="batch.zip")

    assert Path(zip_path).exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    assert any("AAA" in name for name in names)
    assert any("BBB" in name for name in names)


def test_export_auto_filename_includes_metadata(exporter, monkeypatch):
    model = DummyDataModel(symbol="XYZ", source="alpha")

    class FakeDT:
        @staticmethod
        def now():
            return pd.Timestamp("2025-02-01T10:20:30")

    monkeypatch.setattr("src.data.export.data_exporter.datetime", FakeDT)
    output = exporter.export(model, "json", filename=None, include_metadata=False)
    path = Path(output)
    assert path.exists()
    assert path.suffix == ".json"
    assert "alpha" in path.name
    # metadata disabled => no metadata side file
    assert not path.with_suffix(".metadata.json").exists()


def test_export_handles_export_function_failure(exporter, monkeypatch):
    model = DummyDataModel()
    history_before = len(exporter.history)

    def failing_export(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setitem(exporter.supported_formats, "csv", failing_export)
    with pytest.raises(DataLoaderError):
        exporter.export(model, "csv", filename="fail.csv")
    # 历史记录不应增加
    assert len(exporter.history) == history_before


def test_export_to_buffer_csv_success(exporter):
    model = DummyDataModel()
    data = exporter.export_to_buffer(model, "csv")
    assert isinstance(data, bytes)
    assert b"value" in data


def test_export_to_buffer_unsupported_format(exporter):
    model = DummyDataModel()
    with pytest.raises(DataLoaderError):
        exporter.export_to_buffer(model, "xml")


def test_export_to_buffer_data_none(exporter):
    class EmptyModel(DummyDataModel):
        def __init__(self):
            self.data = None
            self._metadata = {}

    with pytest.raises(DataLoaderError):
        exporter.export_to_buffer(EmptyModel(), "csv")


def test_export_multiple_includes_metadata_files(exporter):
    models = [DummyDataModel(symbol="AAA")]
    zip_path = exporter.export_multiple(models, format="csv", include_metadata=True, zip_filename="meta.zip")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    assert any(name.endswith(".metadata.json") for name in names)


def test_export_multiple_cleanup_on_failure(exporter, monkeypatch, tmp_path):
    models = [DummyDataModel(symbol="AAA")]
    temp_dir = Path(exporter.export_dir) / "temp_export"

    original_write = zipfile.ZipFile.write

    def failing_write(self, filename, arcname=None, compress_type=None):
        raise RuntimeError("zip failure")

    monkeypatch.setattr(zipfile.ZipFile, "write", failing_write)
    with pytest.raises(DataLoaderError):
        exporter.export_multiple(models, format="csv", zip_filename="fail.zip")

    assert not temp_dir.exists()
    monkeypatch.setattr(zipfile.ZipFile, "write", original_write)


def test_history_helpers(exporter):
    model = DummyDataModel(symbol="AAA")
    exporter.export(model, "csv", filename="hist.csv")
    assert exporter.get_export_history(limit=1)

    exporter.clear_history()
    assert exporter.get_export_history() == []


def test_get_supported_formats(exporter):
    formats = exporter.get_supported_formats()
    assert "csv" in formats
    assert "json" in formats


