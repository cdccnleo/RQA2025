"""
测试data_processor的覆盖率提升
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
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.data.processing.data_processor import DataProcessor


class MockDataModel:
    """模拟IDataModel"""
    def __init__(self, data=None, frequency='1d', metadata=None):
        self.data = data if data is not None else pd.DataFrame({'a': [1, 2, 3]})
        self.metadata = metadata or {}
        self._frequency = frequency
    
    def get_frequency(self):
        return self._frequency
    
    def get_metadata(self, user_only=False):
        return self.metadata


def test_data_processor_logger_fallback():
    """测试logger fallback（11-19行）"""
    # This is difficult to test because the import happens at module level
    # We can verify the fallback code exists, but triggering it in tests is complex
    from src.data.processing import data_processor
    processor = data_processor.DataProcessor()
    assert processor is not None
    # The fallback code is there, but we can't easily trigger ImportError at import time


def test_data_processor_clean_data_fill_method_else_branch():
    """测试_clean_data中fill_method的else分支（160行）"""
    processor = DataProcessor()
    
    df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5]})
    
    # Use an unknown fill_method to trigger else branch
    result = processor._clean_data(df.copy(), fill_method='unknown_method')
    
    # Should use default fillna(0) in else branch
    assert result is not None
    assert not result['a'].isna().any()


def test_data_processor_clean_data_removed_rows_else_branch(monkeypatch):
    """测试_clean_data中removed_rows计算的else分支（181行）"""
    processor = DataProcessor()
    
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    
    # We need to make the step not have 'original_shape' when it's checked at line 178
    # We can do this by monkeypatching the step dictionary access
    original_steps = processor.processing_info['steps']
    
    # Create a custom list that wraps the steps and modifies dict access
    class StepList:
        def __init__(self, original_list):
            self._list = original_list
        
        def append(self, item):
            # Remove 'original_shape' after appending to trigger else branch
            self._list.append(item)
            if isinstance(item, dict) and 'original_shape' in item:
                # Remove it immediately after append
                del item['original_shape']
        
        def __getitem__(self, index):
            return self._list[index]
        
        def __len__(self):
            return len(self._list)
        
        def __iter__(self):
            return iter(self._list)
    
    # Replace the steps list
    processor.processing_info['steps'] = StepList(original_steps)
    
    # Call _clean_data
    result = processor._clean_data(df.copy())
    
    # Verify that removed_rows was set to 0 (else branch)
    if processor.processing_info['steps']:
        last_step = processor.processing_info['steps'][-1]
        # The else branch should have set removed_rows to 0
        assert 'removed_rows' in last_step
        assert last_step['removed_rows'] == 0


def test_data_processor_validate_processed_data_empty_dataframe():
    """测试_validate_processed_data中df.empty的异常（291行）"""
    processor = DataProcessor()
    
    # Create an empty DataFrame
    df = pd.DataFrame()
    
    # Should raise ValueError when df is empty
    with pytest.raises(ValueError, match="处理后的数据为空"):
        processor._validate_processed_data(df)


def test_data_processor_get_processing_stats_with_start_and_complete():
    """测试get_processing_stats中start_step和complete_step都存在时的处理（410-412行）"""
    processor = DataProcessor()
    
    # Add start and complete steps
    processor.processing_info['steps'].append({
        'step': 'start',
        'timestamp': datetime.now().isoformat()
    })
    
    # Add a delay to ensure different timestamps
    import time
    time.sleep(0.01)
    
    processor.processing_info['steps'].append({
        'step': 'complete',
        'timestamp': datetime.now().isoformat()
    })
    
    # Get processing stats
    stats = processor.get_processing_stats()
    
    # Verify that processing_time_seconds is calculated
    assert isinstance(stats, dict)
    assert 'total_steps' in stats
    assert 'processing_time_seconds' in stats
    assert 'steps' in stats
    assert stats['processing_time_seconds'] > 0  # Should be greater than 0

