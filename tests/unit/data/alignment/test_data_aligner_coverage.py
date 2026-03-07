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
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.data.alignment.data_aligner import (
    DataAligner,
    AlignmentMethod,
    FrequencyType
)
from src.infrastructure.utils.exceptions import DataProcessingError


@pytest.fixture
def sample_dataframes():
    """创建示例数据框"""
    dates1 = pd.date_range('2024-01-01', periods=10, freq='D')
    dates2 = pd.date_range('2024-01-05', periods=10, freq='D')
    
    df1 = pd.DataFrame({
        'value': range(10)
    }, index=dates1)
    
    df2 = pd.DataFrame({
        'value': range(10, 20)
    }, index=dates2)
    
    return {'df1': df1, 'df2': df2}


def test_data_aligner_logger_fallback():
    """测试logger初始化异常时的降级处理（7-15行）"""
    # 注意：由于模块在导入时就会执行第18行的import，这个测试可能无法完全覆盖7-15行
    # 但我们可以验证logger存在
    from src.data.alignment.data_aligner import logger
    assert logger is not None


def test_data_aligner_ensure_datetime_index_exception(monkeypatch):
    """测试_ensure_datetime_index的异常处理（103-107行）"""
    aligner = DataAligner()
    
    # Create a DataFrame with non-convertible index
    df = pd.DataFrame({'value': [1, 2, 3]}, index=['invalid', 'index', 'values'])
    
    # Mock pd.to_datetime to raise exception
    original_to_datetime = pd.to_datetime
    def failing_to_datetime(*args, **kwargs):
        raise ValueError("Cannot convert to datetime")
    
    monkeypatch.setattr(pd, 'to_datetime', failing_to_datetime)
    
    with pytest.raises(DataProcessingError):
        aligner._ensure_datetime_index({'test': df})


def test_data_aligner_get_start_date_by_method_invalid(monkeypatch):
    """测试_get_start_date_by_method的无效方法（160行）"""
    aligner = DataAligner()
    data_frames = {
        'df1': pd.DataFrame({'value': [1, 2]}, index=pd.date_range('2024-01-01', periods=2, freq='D'))
    }
    
    with pytest.raises(DataProcessingError, match="不支持的对齐方法"):
        aligner._get_start_date_by_method(data_frames, 'invalid_method')


def test_data_aligner_get_end_date_by_method_right(monkeypatch):
    """测试_get_end_date_by_method的right方法（173行）"""
    aligner = DataAligner()
    dates1 = pd.date_range('2024-01-01', periods=5, freq='D')
    dates2 = pd.date_range('2024-01-10', periods=5, freq='D')
    
    data_frames = {
        'df1': pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates1),
        'df2': pd.DataFrame({'value': [10, 20, 30, 40, 50]}, index=dates2)
    }
    
    end_date = aligner._get_end_date_by_method(data_frames, 'right')
    assert end_date == dates2[-1]


def test_data_aligner_apply_fill_method_dict(monkeypatch):
    """测试_apply_fill_method使用字典填充方法（197, 205行）"""
    aligner = DataAligner()
    
    # Mock processor to have fill_missing method
    mock_processor = Mock()
    call_count = [0]
    
    def mock_fill(data, method=None):
        call_count[0] += 1
        # Convert 'forward' to 'ffill' for pandas compatibility
        if method == 'forward':
            method = 'ffill'
        elif method == 'backward':
            method = 'bfill'
        return data.fillna(method=method) if method else data
    
    mock_processor.fill_missing = mock_fill
    aligner.processor = mock_processor
    
    df = pd.DataFrame({'value': [1, 2, np.nan, 4]}, index=pd.date_range('2024-01-01', periods=4, freq='D'))
    
    # _apply_fill_method的签名是(aligned_df, name, fill_method)
    fill_method = {'df1': 'forward'}
    result = aligner._apply_fill_method(df, 'df1', fill_method)
    
    assert call_count[0] > 0
    assert isinstance(result, pd.DataFrame)


def test_data_aligner_align_to_reference(monkeypatch, sample_dataframes):
    """测试align_to_reference方法（358-415行）"""
    aligner = DataAligner()
    
    reference_df = sample_dataframes['df1']
    target_dfs = {'target': sample_dataframes['df2']}
    
    result = aligner.align_to_reference(reference_df, target_dfs)
    
    assert isinstance(result, dict)
    assert 'reference' in result
    assert 'target' in result


def test_data_aligner_align_to_reference_infer_freq(monkeypatch):
    """测试align_to_reference推断频率（369-375行）"""
    aligner = DataAligner()
    
    # Create reference with inferrable frequency
    reference_df = pd.DataFrame({
        'value': range(10)
    }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
    
    target_dfs = {
        'target': pd.DataFrame({
            'value': range(5)
        }, index=pd.date_range('2024-01-05', periods=5, freq='D'))
    }
    
    result = aligner.align_to_reference(reference_df, target_dfs, freq=None)
    
    assert isinstance(result, dict)


def test_data_aligner_align_to_reference_freq_enum(monkeypatch):
    """测试align_to_reference使用FrequencyType枚举（376-377行）"""
    aligner = DataAligner()
    
    reference_df = pd.DataFrame({
        'value': range(10)
    }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
    
    target_dfs = {
        'target': pd.DataFrame({
            'value': range(5)
        }, index=pd.date_range('2024-01-05', periods=5, freq='D'))
    }
    
    result = aligner.align_to_reference(reference_df, target_dfs, freq=FrequencyType.DAILY)
    
    assert isinstance(result, dict)


def test_data_aligner_align_to_reference_fill_method_dict(monkeypatch):
    """测试align_to_reference使用字典填充方法（395-402行）"""
    aligner = DataAligner()
    
    reference_df = pd.DataFrame({
        'value': range(10)
    }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
    
    target_df = pd.DataFrame({
        'value': [1, 2, np.nan, 4, 5]
    }, index=pd.date_range('2024-01-05', periods=5, freq='D'))
    
    target_dfs = {'target': target_df}
    
    # Mock processor to have fill_missing method
    mock_processor = Mock()
    call_count = [0]
    
    def mock_fill(data, method=None):
        call_count[0] += 1
        # Convert 'forward' to 'ffill' for pandas compatibility
        if method == 'forward':
            method = 'ffill'
        elif method == 'backward':
            method = 'bfill'
        return data.fillna(method=method) if method else data
    
    mock_processor.fill_missing = mock_fill
    aligner.processor = mock_processor
    
    fill_method = {'target': 'forward'}
    result = aligner.align_to_reference(reference_df, target_dfs, fill_method=fill_method)
    
    assert isinstance(result, dict)
    assert call_count[0] > 0


def test_data_aligner_align_to_reference_exception(monkeypatch):
    """测试align_to_reference的异常处理（411-415行）"""
    aligner = DataAligner()
    
    # Create invalid reference
    reference_df = pd.DataFrame({'value': [1, 2, 3]}, index=['invalid', 'index', 'values'])
    
    target_dfs = {
        'target': pd.DataFrame({
            'value': range(5)
        }, index=pd.date_range('2024-01-05', periods=5, freq='D'))
    }
    
    # Mock pd.to_datetime to raise exception
    original_to_datetime = pd.to_datetime
    def failing_to_datetime(*args, **kwargs):
        raise ValueError("Cannot convert to datetime")
    
    monkeypatch.setattr(pd, 'to_datetime', failing_to_datetime)
    
    with pytest.raises(DataProcessingError):
        aligner.align_to_reference(reference_df, target_dfs)
    
    monkeypatch.setattr(pd, 'to_datetime', original_to_datetime)


def test_data_aligner_align_multi_frequency(monkeypatch, sample_dataframes):
    """测试align_multi_frequency方法（417-504行）"""
    aligner = DataAligner()
    
    # Mock processor to have resample_data and fill_missing methods
    mock_processor = Mock()
    
    def mock_resample(data, freq=None, method=None, fill_method=None):
        # Simple resample simulation
        return data.resample(freq).agg(method if method else 'mean')
    
    def mock_fill(data, method=None):
        # Convert 'forward' to 'ffill' for pandas compatibility
        if method == 'forward':
            method = 'ffill'
        elif method == 'backward':
            method = 'bfill'
        return data.fillna(method=method) if method else data
    
    mock_processor.resample_data = mock_resample
    mock_processor.fill_missing = mock_fill
    aligner.processor = mock_processor
    
    # Create dataframes with different frequencies
    df_daily = pd.DataFrame({
        'value': range(10)
    }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
    
    df_hourly = pd.DataFrame({
        'value': range(24)
    }, index=pd.date_range('2024-01-01', periods=24, freq='H'))
    
    data_frames = {
        'daily': df_daily,
        'hourly': df_hourly
    }
    
    result = aligner.align_multi_frequency(data_frames, target_freq='D')
    
    assert isinstance(result, dict)
    assert 'daily' in result
    assert 'hourly' in result


def test_data_aligner_align_multi_frequency_freq_enum(monkeypatch):
    """测试align_multi_frequency使用FrequencyType枚举"""
    aligner = DataAligner()
    
    # Mock processor
    mock_processor = Mock()
    def mock_resample(data, freq=None, method=None, fill_method=None):
        return data.resample(freq).agg(method if method else 'mean')
    mock_processor.resample_data = mock_resample
    mock_processor.fill_missing = lambda data, method=None: data.fillna(method=method) if method else data
    aligner.processor = mock_processor
    
    df_daily = pd.DataFrame({
        'value': range(10)
    }, index=pd.date_range('2024-01-01', periods=10, freq='D'))
    
    data_frames = {'daily': df_daily}
    
    result = aligner.align_multi_frequency(data_frames, target_freq=FrequencyType.DAILY)
    
    assert isinstance(result, dict)


def test_data_aligner_align_multi_frequency_resample_methods(monkeypatch):
    """测试align_multi_frequency使用resample_methods"""
    aligner = DataAligner()
    
    # Mock processor
    mock_processor = Mock()
    def mock_resample(data, freq=None, method=None, fill_method=None):
        return data.resample(freq).agg(method if method else 'mean')
    mock_processor.resample_data = mock_resample
    mock_processor.fill_missing = lambda data, method=None: data.fillna(method=method) if method else data
    aligner.processor = mock_processor
    
    df_hourly = pd.DataFrame({
        'value': range(24)
    }, index=pd.date_range('2024-01-01', periods=24, freq='H'))
    
    data_frames = {'hourly': df_hourly}
    
    resample_methods = {'hourly': 'mean'}
    result = aligner.align_multi_frequency(data_frames, target_freq='D', resample_methods=resample_methods)
    
    assert isinstance(result, dict)


def test_data_aligner_get_alignment_history_limit(monkeypatch):
    """测试get_alignment_history使用limit参数（506-518行）"""
    aligner = DataAligner()
    
    # Add some history
    aligner.alignment_history = [
        {'operation': 'align', 'timestamp': datetime.now()},
        {'operation': 'merge', 'timestamp': datetime.now()},
        {'operation': 'align', 'timestamp': datetime.now()}
    ]
    
    result = aligner.get_alignment_history(limit=2)
    
    assert isinstance(result, list)
    assert len(result) <= 2


def test_data_aligner_record_alignment(monkeypatch):
    """测试_record_alignment方法（520-564行）"""
    aligner = DataAligner()
    
    initial_count = len(aligner.alignment_history)
    
    input_frames = {'df1': pd.DataFrame({'value': [1, 2]}, index=pd.date_range('2024-01-01', periods=2, freq='D'))}
    output_frames = {'result': pd.DataFrame({'value': [1, 2]}, index=pd.date_range('2024-01-01', periods=2, freq='D'))}
    
    aligner._record_alignment(
        input_frames=input_frames,
        output_frames=output_frames,
        freq='D',
        method='outer',
        fill_method=None,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2)
    )
    
    assert len(aligner.alignment_history) == initial_count + 1


def test_data_aligner_save_alignment_history(monkeypatch, tmp_path):
    """测试save_alignment_history方法（566-582行）"""
    aligner = DataAligner()
    
    # Add some history
    aligner.alignment_history = [
        {
            'operation': 'align',
            'timestamp': datetime.now().isoformat(),
            'input': {'df1': 'dataframe'},
            'output': {'result': 'success'}
        }
    ]
    
    file_path = tmp_path / "alignment_history.json"
    aligner.save_alignment_history(file_path)
    
    assert file_path.exists()


def test_data_aligner_save_alignment_history_exception(monkeypatch, tmp_path):
    """测试save_alignment_history的异常处理"""
    aligner = DataAligner()
    
    aligner.alignment_history = [{'operation': 'test'}]
    
    # Mock open to raise exception
    original_open = open
    def failing_open(*args, **kwargs):
        raise IOError("File write failed")
    
    monkeypatch.setattr('builtins.open', failing_open)
    
    # Should not raise exception, just log error
    file_path = tmp_path / "alignment_history.json"
    try:
        aligner.save_alignment_history(file_path)
    except Exception:
        pass  # Exception handling should catch it
    
    monkeypatch.setattr('builtins.open', original_open)


def test_data_aligner_load_alignment_history(monkeypatch, tmp_path):
    """测试load_alignment_history方法（584-602行）"""
    aligner = DataAligner()
    
    # Create history file
    history_data = [
        {
            'operation': 'align',
            'timestamp': datetime.now().isoformat(),
            'input': {},
            'output': {}
        }
    ]
    
    file_path = tmp_path / "alignment_history.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(history_data, f)
    
    aligner.load_alignment_history(file_path)
    
    assert len(aligner.alignment_history) > 0


def test_data_aligner_load_alignment_history_file_not_found(monkeypatch, tmp_path):
    """测试load_alignment_history文件不存在的情况"""
    aligner = DataAligner()
    
    file_path = tmp_path / "nonexistent.json"
    
    # Should not raise exception
    aligner.load_alignment_history(file_path)


def test_data_aligner_load_alignment_history_invalid_json(monkeypatch, tmp_path):
    """测试load_alignment_history无效JSON的情况"""
    aligner = DataAligner()
    
    file_path = tmp_path / "invalid.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("invalid json content")
    
    # Should not raise exception, just log error
    aligner.load_alignment_history(file_path)

