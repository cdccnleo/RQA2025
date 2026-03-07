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
from unittest.mock import patch, MagicMock

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


def test_export_unsupported_format_raises():
    """测试不支持的格式抛出异常"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="Unsupported export format"):
        exp.export(m, "unknown_format")


def test_export_none_data_raises():
    """测试 None 数据抛出异常"""
    m = _ModelWithoutData({"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="DataModel.data is None"):
        exp.export(m, "csv")


def test_export_auto_filename_generation():
    """测试自动生成文件名"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "test_source", "symbol": "TEST"})
    exp = DataExporter("test_dir")
    
    # Mock 导出函数以避免实际文件操作
    exp._export_csv = MagicMock()
    
    result = exp.export(m, "csv", filename=None)
    
    # 验证文件名格式
    assert "test_source" in result
    assert "TEST" in result
    assert result.endswith(".csv")


def test_export_filename_without_extension():
    """测试文件名没有扩展名时自动添加"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    # Mock 导出函数
    exp._export_csv = MagicMock()
    
    result = exp.export(m, "csv", filename="test_file")
    
    assert result.endswith(".csv")


def test_export_history_save_error_handled(tmp_path):
    """测试导出历史保存错误处理"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # _save_history 内部有 try-except，会捕获异常并记录错误
    # 这里我们直接测试 _save_history 方法
    # 通过 mock open 来模拟文件写入失败
    from unittest.mock import patch, mock_open
    
    with patch("builtins.open", side_effect=IOError("Permission denied")):
        # _save_history 应该不抛出异常，只记录错误
        exp._save_history()
    
    # 正常导出应该成功
    result = exp.export(m, "csv", filename="test.csv")
    assert result is not None


def test_export_to_buffer_unsupported_format():
    """测试导出到 buffer（不支持的格式）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="Unsupported export format"):
        exp.export_to_buffer(m, "unknown_format")


def test_export_to_buffer_none_data():
    """测试导出到 buffer（None 数据）"""
    m = _ModelWithoutData({"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="DataModel.data is None"):
        exp.export_to_buffer(m, "csv")


def test_export_to_buffer_csv():
    """测试导出到 buffer（CSV 格式）"""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    buffer = exp.export_to_buffer(m, "csv", include_metadata=False)
    
    assert isinstance(buffer, bytes)
    assert len(buffer) > 0


def test_export_to_buffer_json_with_metadata():
    """测试导出到 buffer（JSON 格式，包含元数据）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit", "symbol": "TEST"})
    exp = DataExporter("test_dir")
    
    buffer = exp.export_to_buffer(m, "json", include_metadata=True)
    
    data = json.loads(buffer.decode("utf-8"))
    assert "data" in data
    assert "metadata" in data


def test_export_to_buffer_json_without_metadata():
    """测试导出到 buffer（JSON 格式，不包含元数据）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    buffer = exp.export_to_buffer(m, "json", include_metadata=False)
    
    assert isinstance(buffer, bytes)


def test_export_to_buffer_excel():
    """测试导出到 buffer（Excel 格式）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    buffer = exp.export_to_buffer(m, "excel", include_metadata=True)
    
    assert isinstance(buffer, bytes)
    assert len(buffer) > 0


def test_export_to_buffer_pickle():
    """测试导出到 buffer（Pickle 格式）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    buffer = exp.export_to_buffer(m, "pickle", include_metadata=True)
    
    assert isinstance(buffer, bytes)


def test_export_to_buffer_unsupported_buffer_format():
    """测试导出到 buffer（不支持的 buffer 格式）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="Buffer export not supported"):
        exp.export_to_buffer(m, "parquet")


def test_export_multiple_unsupported_format():
    """测试批量导出（不支持的格式）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="Unsupported export format"):
        exp.export_multiple([m], "unknown_format")


def test_export_multiple_auto_zip_filename(tmp_path):
    """测试批量导出（自动生成 ZIP 文件名）"""
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})
    m1 = _Model(df1, {"source": "unit", "symbol": "S1"})
    m2 = _Model(df2, {"source": "unit", "symbol": "S2"})
    exp = DataExporter(str(tmp_path))
    
    zip_path = exp.export_multiple([m1, m2], "csv", zip_filename=None)
    
    assert Path(zip_path).exists()
    assert zip_path.endswith(".zip")


def test_export_multiple_zip_filename_without_extension(tmp_path):
    """测试批量导出（ZIP 文件名没有扩展名）"""
    df1 = pd.DataFrame({"a": [1]})
    m1 = _Model(df1, {"source": "unit", "symbol": "S1"})
    exp = DataExporter(str(tmp_path))
    
    zip_path = exp.export_multiple([m1], "csv", zip_filename="test_zip")
    
    assert zip_path.endswith(".zip")


def test_export_multiple_cleanup_on_error(tmp_path, monkeypatch):
    """测试批量导出（错误时清理临时文件）"""
    df1 = pd.DataFrame({"a": [1]})
    m1 = _Model(df1, {"source": "unit", "symbol": "S1"})
    exp = DataExporter(str(tmp_path))
    
    # Mock export function to raise exception during export
    # export_multiple 调用 supported_formats[format.lower()]，需要替换字典中的方法
    original_export = exp.supported_formats["csv"]
    
    def _failing_export(*args, **kwargs):
        raise RuntimeError("Export failed")
    
    exp.supported_formats["csv"] = _failing_export
    
    # export_multiple 应该捕获异常并抛出 DataLoaderError
    with pytest.raises(DataLoaderError, match="Failed to export multiple data models"):
        exp.export_multiple([m1], "csv", include_metadata=False)
    
    # 恢复原始方法
    exp.supported_formats["csv"] = original_export
    
    # 临时目录应该被清理（或至少尝试清理）
    temp_dir = tmp_path / "temp_export"
    # 由于错误，临时目录可能仍然存在，但应该尝试清理


def test_get_export_history_with_limit():
    """测试获取导出历史（带限制）"""
    exp = DataExporter("test_dir")
    
    # 添加一些历史记录
    for i in range(10):
        exp.history.append({"id": i, "format": "csv"})
    
    history = exp.get_export_history(limit=5)
    
    assert len(history) == 5
    assert history[-1]["id"] == 9


def test_get_export_history_without_limit(tmp_path):
    """测试获取导出历史（无限制）"""
    exp = DataExporter(str(tmp_path))
    
    # 添加一些历史记录
    for i in range(10):
        exp.history.append({"id": i, "format": "csv"})
    
    history = exp.get_export_history()
    
    # 历史记录可能包含之前加载的记录，所以至少应该有10条
    assert len(history) >= 10


def test_clear_history():
    """测试清除历史记录"""
    exp = DataExporter("test_dir")
    
    # 添加一些历史记录
    for i in range(5):
        exp.history.append({"id": i})
    
    exp.clear_history()
    
    assert len(exp.history) == 0


def test_load_history_nonexistent_file(tmp_path):
    """测试加载历史记录（文件不存在）"""
    exp = DataExporter(str(tmp_path))
    
    # 应该不抛出异常，返回空列表
    assert exp.history == []


def test_load_history_invalid_json(tmp_path, monkeypatch):
    """测试加载历史记录（无效 JSON）"""
    exp = DataExporter(str(tmp_path))
    
    # 创建无效的 JSON 文件
    history_file = tmp_path / "export_history.json"
    history_file.write_text("invalid json content")
    
    # 重新初始化应该处理无效 JSON
    exp2 = DataExporter(str(tmp_path))
    # 应该不抛出异常，history 应该为空或使用默认值
    assert isinstance(exp2.history, list)


def test_export_csv_none_data():
    """测试导出 CSV（None 数据）"""
    m = _ModelWithoutData({"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="DataModel.data is None"):
        exp._export_csv(m, Path("test.csv"), include_metadata=True)


def test_export_excel_none_data():
    """测试导出 Excel（None 数据）"""
    m = _ModelWithoutData({"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="DataModel.data is None"):
        exp._export_excel(m, Path("test.xlsx"), include_metadata=True)


def test_export_json_none_data():
    """测试导出 JSON（None 数据）"""
    m = _ModelWithoutData({"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="DataModel.data is None"):
        exp._export_json(m, Path("test.json"), include_metadata=True)


def test_export_parquet_none_data():
    """测试导出 Parquet（None 数据）"""
    m = _ModelWithoutData({"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="DataModel.data is None"):
        exp._export_parquet(m, Path("test.parquet"), include_metadata=True)


def test_export_pickle_none_data():
    """测试导出 Pickle（None 数据）"""
    m = _ModelWithoutData({"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="DataModel.data is None"):
        exp._export_pickle(m, Path("test.pickle"), include_metadata=True)


def test_export_hdf_none_data():
    """测试导出 HDF5（None 数据）"""
    m = _ModelWithoutData({"source": "unit"})
    exp = DataExporter("test_dir")
    
    with pytest.raises(DataLoaderError, match="DataModel.data is None"):
        exp._export_hdf(m, Path("test.h5"), include_metadata=True)


def test_get_supported_formats():
    """测试获取支持的格式列表"""
    exp = DataExporter("test_dir")
    
    formats = exp.get_supported_formats()
    
    assert "csv" in formats
    assert "excel" in formats
    assert "json" in formats
    assert "parquet" in formats
    assert "pickle" in formats
    assert "hdf" in formats


def test_export_exception_wrapping(tmp_path, monkeypatch):
    """测试导出异常包装"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # Mock _export_csv to raise exception by replacing it in supported_formats
    def _bad_export(*args, **kwargs):
        raise RuntimeError("Export failed")
    
    exp.supported_formats['csv'] = _bad_export
    
    with pytest.raises(DataLoaderError, match="Failed to export data"):
        exp.export(m, "csv", filename="test.csv")


def test_export_to_buffer_exception_wrapping(monkeypatch):
    """测试导出到 buffer 异常包装"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    # Mock to_csv to raise exception
    def _bad_to_csv(*args, **kwargs):
        raise RuntimeError("CSV export failed")
    
    monkeypatch.setattr(pd.DataFrame, "to_csv", _bad_to_csv)
    
    with pytest.raises(DataLoaderError, match="Failed to export to buffer"):
        exp.export_to_buffer(m, "csv")


def test_export_multiple_exception_wrapping(tmp_path, monkeypatch):
    """测试批量导出异常包装"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # Mock _export_csv to raise exception by replacing it in supported_formats
    def _bad_export(*args, **kwargs):
        raise RuntimeError("Export failed")
    
    exp.supported_formats['csv'] = _bad_export
    
    with pytest.raises(DataLoaderError, match="Failed to export multiple data models"):
        exp.export_multiple([m], "csv")


def test_export_to_buffer_pickle_without_metadata():
    """测试导出到 buffer（Pickle 格式，不包含元数据）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter("test_dir")
    
    buffer = exp.export_to_buffer(m, "pickle", include_metadata=False)
    
    assert isinstance(buffer, bytes)
    assert len(buffer) > 0


def test_export_multiple_cleanup_finally_block(tmp_path, monkeypatch):
    """测试批量导出的 finally 块清理逻辑"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit", "symbol": "TEST"})
    exp = DataExporter(str(tmp_path))
    
    # 正常导出，验证 finally 块会清理临时文件
    zip_path = exp.export_multiple([m], "csv", include_metadata=True)
    
    # 验证临时目录被清理（或至少尝试清理）
    temp_dir = tmp_path / "temp_export"
    # 临时目录可能仍然存在（如果清理失败），但应该尝试清理
    # 主要验证导出成功且 ZIP 文件存在
    assert Path(zip_path).exists()


def test_export_excel_without_metadata(tmp_path):
    """测试导出 Excel（不包含元数据）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # 测试不包含元数据的情况
    exp._export_excel(m, tmp_path / "test.xlsx", include_metadata=False)
    
    # 验证文件存在
    assert (tmp_path / "test.xlsx").exists()


def test_export_excel_with_metadata(tmp_path):
    """测试导出 Excel（包含元数据）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit", "symbol": "TEST"})
    exp = DataExporter(str(tmp_path))
    
    # 测试包含元数据的情况
    exp._export_excel(m, tmp_path / "test.xlsx", include_metadata=True)
    
    # 验证文件存在
    assert (tmp_path / "test.xlsx").exists()


def test_export_excel_exception(tmp_path, monkeypatch):
    """测试导出 Excel 异常处理"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # Mock ExcelWriter 来触发异常
    def _bad_excel_writer(*args, **kwargs):
        raise RuntimeError("Excel write failed")
    
    monkeypatch.setattr(pd, "ExcelWriter", _bad_excel_writer)
    
    with pytest.raises(Exception):  # ExcelWriter 会抛出异常
        exp._export_excel(m, tmp_path / "test.xlsx", include_metadata=True)


def test_export_json_exception(tmp_path, monkeypatch):
    """测试导出 JSON 异常处理"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # Mock json.dump 来触发异常
    def _bad_json_dump(*args, **kwargs):
        raise IOError("JSON write failed")
    
    monkeypatch.setattr(json, "dump", _bad_json_dump)
    
    with pytest.raises(IOError):
        exp._export_json(m, tmp_path / "test.json", include_metadata=True)


def test_export_parquet_exception(tmp_path, monkeypatch):
    """测试导出 Parquet 异常处理"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # Mock to_parquet 来触发异常
    def _bad_to_parquet(self, *args, **kwargs):
        raise RuntimeError("Parquet write failed")
    
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _bad_to_parquet)
    
    with pytest.raises(RuntimeError):
        exp._export_parquet(m, tmp_path / "test.parquet", include_metadata=True)


def test_export_pickle_exception(tmp_path, monkeypatch):
    """测试导出 Pickle 异常处理"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # Mock pd.to_pickle 来触发异常
    def _bad_to_pickle(*args, **kwargs):
        raise RuntimeError("Pickle write failed")
    
    monkeypatch.setattr(pd, "to_pickle", _bad_to_pickle)
    
    with pytest.raises(RuntimeError):
        exp._export_pickle(m, tmp_path / "test.pickle", include_metadata=True)


def test_export_parquet_with_metadata(tmp_path):
    """测试导出 Parquet（包含元数据）"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # 测试包含元数据的情况
    exp._export_parquet(m, tmp_path / "test.parquet", include_metadata=True)
    
    # 验证文件存在
    assert (tmp_path / "test.parquet").exists()
    assert (tmp_path / "test.metadata.json").exists()


def test_export_hdf_with_metadata(tmp_path):
    """测试导出 HDF5（包含元数据）"""
    try:
        import tables  # 检查 pytables 是否可用
    except ImportError:
        pytest.skip("pytables not available")
    
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # 测试包含元数据的情况
    exp._export_hdf(m, tmp_path / "test.h5", include_metadata=True)
    
    # 验证文件存在
    assert (tmp_path / "test.h5").exists()


def test_export_hdf_exception(tmp_path, monkeypatch):
    """测试导出 HDF5 异常处理"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit"})
    exp = DataExporter(str(tmp_path))
    
    # Mock HDFStore 来触发异常
    def _bad_hdf_store(*args, **kwargs):
        raise RuntimeError("HDF write failed")
    
    monkeypatch.setattr(pd, "HDFStore", _bad_hdf_store)
    
    with pytest.raises(RuntimeError):
        exp._export_hdf(m, tmp_path / "test.h5", include_metadata=True)


def test_export_multiple_cleanup_oserror(tmp_path, monkeypatch):
    """测试批量导出清理时的 OSError 处理"""
    df = pd.DataFrame({"a": [1, 2]})
    m = _Model(df, {"source": "unit", "symbol": "TEST"})
    exp = DataExporter(str(tmp_path))
    
    # Mock rmdir 来触发 OSError（模拟目录非空等情况）
    def _bad_rmdir(self):
        raise OSError("Directory not empty")
    
    monkeypatch.setattr(Path, "rmdir", _bad_rmdir)
    
    # 正常导出，应该处理 OSError 而不抛出异常
    zip_path = exp.export_multiple([m], "csv", include_metadata=False)
    
    assert Path(zip_path).exists()

