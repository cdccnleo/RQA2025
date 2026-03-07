"""
测试quality_monitor的覆盖率提升 - 补充测试
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
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta
import json
import os

from src.data.monitoring.quality_monitor import (
    QualityMetrics,
    DataModel,
    DataQualityMonitor
)


def test_quality_metrics_calculate_completeness_empty_dataframe():
    """测试calculate_completeness中data.empty的返回（52行）"""
    metrics = QualityMetrics()
    df = pd.DataFrame()
    
    result = metrics.calculate_completeness(df)
    
    assert result == 1.0
    # When data is empty, the method returns early and doesn't set metrics
    # So we just verify the return value


def test_quality_metrics_calculate_accuracy_empty_dataframe():
    """测试calculate_accuracy中data.empty的返回（62行）"""
    metrics = QualityMetrics()
    df = pd.DataFrame()
    
    result = metrics.calculate_accuracy(df)
    
    assert result == 1.0
    # When data is empty, the method returns early and doesn't set metrics
    # So we just verify the return value


def test_quality_metrics_calculate_accuracy_with_numeric_data():
    """测试calculate_accuracy中numeric不为空时的计算（67-72行）"""
    metrics = QualityMetrics()
    # Create DataFrame with numeric data that will have outliers
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5, 100, 200]  # Contains outliers
    })
    
    result = metrics.calculate_accuracy(df)
    
    assert 0.0 <= result <= 1.0
    assert metrics.metrics["accuracy"] == result


def test_quality_metrics_calculate_consistency_empty_dataframe():
    """测试calculate_consistency中data.empty的返回（79行）"""
    metrics = QualityMetrics()
    df = pd.DataFrame()
    
    result = metrics.calculate_consistency(df)
    
    assert result == 1.0
    # When data is empty, the method returns early and doesn't set metrics
    # So we just verify the return value


def test_quality_metrics_calculate_consistency_with_numeric_data():
    """测试calculate_consistency中numeric不为空时的计算（83-85行）"""
    metrics = QualityMetrics()
    # Create DataFrame with numeric data
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    })
    
    result = metrics.calculate_consistency(df)
    
    assert 0.0 <= result <= 1.0
    assert metrics.metrics["consistency"] == result


def test_data_model_get_metadata():
    """测试get_metadata的返回（131行）"""
    model = DataModel()
    metadata = {'test': 'value'}
    model.set_metadata(metadata)
    
    result = model.get_metadata()
    
    assert result == metadata


def test_data_quality_monitor_evaluate_quality_timeliness_days_ago_greater_than_5():
    """测试evaluate_quality中timeliness的days_ago > 5分支（183行）"""
    monitor = DataQualityMonitor(report_dir='./tmp_test/')
    
    # Create data model with created_at > 5 days ago
    data_model = DataModel(data=pd.DataFrame({'a': [1, 2, 3]}))
    data_model.metadata = {
        'source': 'test_source',
        'created_at': (datetime.now() - timedelta(days=6)).isoformat()
    }
    
    metrics = monitor.evaluate_quality(data_model)
    
    assert metrics.timeliness == 0.0


def test_data_quality_monitor_evaluate_quality_timeliness_exception():
    """测试evaluate_quality中timeliness的异常处理（186-187行）"""
    monitor = DataQualityMonitor(report_dir='./tmp_test/')
    
    # Create data model with invalid created_at to trigger exception
    data_model = DataModel(data=pd.DataFrame({'a': [1, 2, 3]}))
    data_model.metadata = {
        'source': 'test_source',
        'created_at': 'invalid_date_format'
    }
    
    metrics = monitor.evaluate_quality(data_model)
    
    # Should handle exception and set timeliness to 1.0
    assert metrics.timeliness == 1.0


def test_data_quality_monitor_evaluate_quality_history_file_exception(tmp_path):
    """测试evaluate_quality中读取history文件的异常处理（205-206行）"""
    monitor = DataQualityMonitor(report_dir=str(tmp_path))
    
    # Create a corrupted history file
    history_file = os.path.join(tmp_path, 'quality_history.json')
    os.makedirs(tmp_path, exist_ok=True)
    with open(history_file, 'w', encoding='utf-8') as f:
        f.write('invalid json content')
    
    # Create data model
    data_model = DataModel(data=pd.DataFrame({'a': [1, 2, 3]}))
    data_model.metadata = {
        'source': 'test_source',
        'created_at': datetime.now().isoformat()
    }
    
    # Should handle exception and create new history
    metrics = monitor.evaluate_quality(data_model)
    
    assert metrics is not None
    # Verify history file was recreated
    assert os.path.exists(history_file)
    
    # Cleanup
    if os.path.exists(history_file):
        os.remove(history_file)

