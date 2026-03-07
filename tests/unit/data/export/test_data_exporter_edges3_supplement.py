"""
data_exporter.py 边界测试补充 - 第3批
覆盖未覆盖的导出格式和异常处理路径
"""
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


import pytest
import pandas as pd
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import shutil

from src.data.export.data_exporter import DataExporter
from src.infrastructure.utils.exceptions import DataLoaderError


class _Model:
    """测试用的数据模型"""
    def __init__(self, df: pd.DataFrame, metadata: dict):
        self.data = df
        self._metadata = metadata

    def get_data(self):
        return self.data

    def get_metadata(self):
        return self._metadata


class _ModelWithoutData:
    """没有 data 属性的模型"""
    def __init__(self, metadata: dict):
        self._metadata = metadata

    def get_metadata(self):
        return self._metadata


@pytest.fixture
def tmp_export_dir(tmp_path):
    """创建临时导出目录"""
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    return str(export_dir)


def test_export_csv_with_metadata(tmp_export_dir):
    """测试导出 CSV（包含元数据，覆盖 335-353 行）"""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    m = _Model(df, {"source": "test", "symbol": "TEST"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "csv", filename="test.csv", include_metadata=True)
    
    assert Path(result).exists()
    # 检查元数据文件是否存在
    metadata_file = Path(result).with_suffix('.metadata.json')
    assert metadata_file.exists()
    
    # 验证元数据内容
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    assert metadata["source"] == "test"
    assert metadata["symbol"] == "TEST"


def test_export_csv_without_metadata(tmp_export_dir):
    """测试导出 CSV（不包含元数据，覆盖 335-348 行）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "test"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "csv", filename="test.csv", include_metadata=False)
    
    assert Path(result).exists()
    # 元数据文件不应该存在
    metadata_file = Path(result).with_suffix('.metadata.json')
    assert not metadata_file.exists()


def test_export_excel_with_metadata(tmp_export_dir):
    """测试导出 Excel（包含元数据，覆盖 354-374 行）"""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    m = _Model(df, {"source": "test", "symbol": "TEST"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "excel", filename="test.xlsx", include_metadata=True)
    
    assert Path(result).exists()
    # 验证 Excel 文件包含两个工作表
    with pd.ExcelWriter(result, mode='r') as writer:
        assert 'Data' in writer.sheets
        assert 'Metadata' in writer.sheets


def test_export_excel_without_metadata(tmp_export_dir):
    """测试导出 Excel（不包含元数据，覆盖 354-368 行）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "test"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "excel", filename="test.xlsx", include_metadata=False)
    
    assert Path(result).exists()
    # 验证 Excel 文件只包含数据工作表
    with pd.ExcelWriter(result, mode='r') as writer:
        assert 'Data' in writer.sheets
        assert 'Metadata' not in writer.sheets


def test_export_json_with_metadata(tmp_export_dir):
    """测试导出 JSON（包含元数据，覆盖 376-394 行）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "test", "symbol": "TEST"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "json", filename="test.json", include_metadata=True)
    
    assert Path(result).exists()
    # 验证 JSON 内容
    with open(result, 'r') as f:
        data = json.load(f)
    assert "data" in data
    assert "metadata" in data
    assert data["metadata"]["source"] == "test"


def test_export_json_without_metadata(tmp_export_dir):
    """测试导出 JSON（不包含元数据，覆盖 376-394 行）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "test"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "json", filename="test.json", include_metadata=False)
    
    assert Path(result).exists()
    # 验证 JSON 内容
    with open(result, 'r') as f:
        data = json.load(f)
    assert "data" in data
    assert data["metadata"] is None


def test_export_parquet_with_metadata(tmp_export_dir):
    """测试导出 Parquet（包含元数据，覆盖 396-413 行）"""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    m = _Model(df, {"source": "test", "symbol": "TEST"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "parquet", filename="test.parquet", include_metadata=True)
    
    assert Path(result).exists()
    # 检查元数据文件是否存在
    metadata_file = Path(result).with_suffix('.metadata.json')
    assert metadata_file.exists()


def test_export_parquet_without_metadata(tmp_export_dir):
    """测试导出 Parquet（不包含元数据，覆盖 396-409 行）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "test"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "parquet", filename="test.parquet", include_metadata=False)
    
    assert Path(result).exists()
    # 元数据文件不应该存在
    metadata_file = Path(result).with_suffix('.metadata.json')
    assert not metadata_file.exists()


def test_export_pickle_with_metadata(tmp_export_dir):
    """测试导出 Pickle（包含元数据，覆盖 415-432 行）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "test", "symbol": "TEST"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "pickle", filename="test.pkl", include_metadata=True)
    
    assert Path(result).exists()
    # 验证 Pickle 文件内容
    import pickle
    with open(result, 'rb') as f:
        obj = pickle.load(f)
    assert "data" in obj
    assert "metadata" in obj
    assert obj["metadata"]["source"] == "test"


def test_export_pickle_without_metadata(tmp_export_dir):
    """测试导出 Pickle（不包含元数据，覆盖 415-432 行）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "test"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "pickle", filename="test.pkl", include_metadata=False)
    
    assert Path(result).exists()
    # 验证 Pickle 文件内容
    import pickle
    with open(result, 'rb') as f:
        obj = pickle.load(f)
    assert "data" in obj
    assert obj["metadata"] is None


def test_export_hdf_with_metadata(tmp_export_dir):
    """测试导出 HDF5（包含元数据，覆盖 434-450 行）"""
    try:
        import tables
    except ImportError:
        pytest.skip("pytables not installed")
    
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    m = _Model(df, {"source": "test", "symbol": "TEST"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "hdf", filename="test.h5", include_metadata=True)
    
    assert Path(result).exists()
    # 验证 HDF5 文件内容
    with pd.HDFStore(result, mode='r') as store:
        assert 'data' in store
        assert 'metadata' in store


def test_export_hdf_without_metadata(tmp_export_dir):
    """测试导出 HDF5（不包含元数据，覆盖 434-447 行）"""
    try:
        import tables
    except ImportError:
        pytest.skip("pytables not installed")
    
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "test"})
    exp = DataExporter(tmp_export_dir)
    
    result = exp.export(m, "hdf", filename="test.h5", include_metadata=False)
    
    assert Path(result).exists()
    # 验证 HDF5 文件内容
    with pd.HDFStore(result, mode='r') as store:
        assert 'data' in store
        assert 'metadata' not in store


def test_export_exception_handling(tmp_export_dir, monkeypatch):
    """测试导出异常处理（覆盖 155-157 行）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "test"})
    exp = DataExporter(tmp_export_dir)
    
    # Mock 导出函数抛出异常
    def failing_export(*args, **kwargs):
        raise RuntimeError("Export failed")
    
    exp.supported_formats["csv"] = failing_export
    
    with pytest.raises(DataLoaderError, match="Failed to export data"):
        exp.export(m, "csv", filename="test.csv")


def test_load_history_exception_handling(tmp_export_dir, monkeypatch):
    """测试加载历史记录异常处理（覆盖 72-77 行）"""
    exp = DataExporter(tmp_export_dir)
    
    # 创建历史文件但内容无效
    history_file = Path(tmp_export_dir) / 'export_history.json'
    history_file.write_text("invalid json")
    
    # 重新加载应该不抛出异常，返回空列表
    new_exp = DataExporter(tmp_export_dir)
    assert new_exp.history == []


def test_save_history_exception_handling(tmp_export_dir, monkeypatch):
    """测试保存历史记录异常处理（覆盖 79-85 行）"""
    exp = DataExporter(tmp_export_dir)
    exp.history = [{"test": "data"}]
    
    # Mock open 抛出异常
    with patch("builtins.open", side_effect=IOError("Permission denied")):
        # _save_history 应该不抛出异常，只记录错误
        exp._save_history()
    
    # 历史记录应该仍然存在（虽然保存失败）
    assert len(exp.history) == 1


def test_export_multiple_with_metadata_files(tmp_export_dir):
    """测试批量导出（包含元数据文件，覆盖 308-312 行）"""
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})
    m1 = _Model(df1, {"source": "unit", "symbol": "S1"})
    m2 = _Model(df2, {"source": "unit", "symbol": "S2"})
    exp = DataExporter(tmp_export_dir)
    
    zip_path = exp.export_multiple([m1, m2], "csv", zip_filename="test.zip", include_metadata=True)
    
    assert Path(zip_path).exists()
    # 验证 ZIP 文件包含元数据文件
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        files = zipf.namelist()
        # 应该包含 CSV 文件和元数据文件
        assert any(f.endswith('.csv') for f in files)
        assert any(f.endswith('.metadata.json') for f in files)


def test_export_multiple_temp_dir_cleanup_failure(tmp_export_dir, monkeypatch):
    """测试批量导出（临时目录清理失败，覆盖 328-333 行）"""
    df1 = pd.DataFrame({"a": [1]})
    m1 = _Model(df1, {"source": "unit", "symbol": "S1"})
    exp = DataExporter(tmp_export_dir)
    
    # Mock rmdir 抛出异常
    original_rmdir = Path.rmdir
    def failing_rmdir(self):
        raise OSError("Directory not empty")
    
    monkeypatch.setattr(Path, "rmdir", failing_rmdir)
    
    # 导出应该成功，即使临时目录清理失败
    zip_path = exp.export_multiple([m1], "csv", zip_filename="test.zip")
    
    assert Path(zip_path).exists()
    
    # 恢复原始方法
    monkeypatch.setattr(Path, "rmdir", original_rmdir)




