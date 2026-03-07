"""
测试quality_monitor的覆盖率提升
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
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.data.monitoring.quality_monitor import (
    QualityMetrics,
    DataModel,
    DataQualityMonitor
)


@pytest.fixture
def sample_dataframe():
    """创建示例DataFrame"""
    return pd.DataFrame({
        'value': [1, 2, 3, 4, 5],
        'date': pd.date_range('2024-01-01', periods=5, freq='D')
    })


@pytest.fixture
def sample_data_model(sample_dataframe):
    """创建示例DataModel"""
    model = DataModel(data=sample_dataframe)
    model.set_metadata({'created_at': datetime.now().isoformat()})
    return model


def test_quality_metrics_post_init_with_none_overall_score():
    """测试QualityMetrics的__post_init__当overall_score为None时（44-46行）"""
    metrics = QualityMetrics(
        completeness=0.9,
        accuracy=0.8,
        consistency=0.85,
        timeliness=0.95,
        validity=0.88,
        overall_score=None
    )
    
    # overall_score应该被计算为其他指标的平均值
    assert metrics.overall_score is not None
    assert 0.0 <= metrics.overall_score <= 1.0


def test_quality_metrics_calculate_accuracy_empty_numeric(sample_dataframe):
    """测试calculate_accuracy当没有数值列时（64-65行）"""
    # 创建只有非数值列的DataFrame
    df = pd.DataFrame({
        'text': ['a', 'b', 'c'],
        'category': ['A', 'B', 'C']
    })
    
    metrics = QualityMetrics()
    result = metrics.calculate_accuracy(df)
    
    assert result == 1.0


def test_quality_metrics_calculate_consistency_empty_numeric(sample_dataframe):
    """测试calculate_consistency当没有数值列时（80-81行）"""
    # 创建只有非数值列的DataFrame
    df = pd.DataFrame({
        'text': ['a', 'b', 'c'],
        'category': ['A', 'B', 'C']
    })
    
    metrics = QualityMetrics()
    result = metrics.calculate_consistency(df)
    
    assert result == 1.0


def test_data_model_init_with_none_data():
    """测试DataModel初始化时data为None（119-120行）"""
    model = DataModel(data=None)
    
    assert model.data is not None
    assert isinstance(model.data, pd.DataFrame)
    assert model.data.empty


def test_data_model_init_with_non_dataframe():
    """测试DataModel初始化时data不是DataFrame（119-120行）"""
    model = DataModel(data="not a dataframe")
    
    assert model.data is not None
    assert isinstance(model.data, pd.DataFrame)
    assert model.data.empty


def test_data_quality_monitor_evaluate_quality_consistency_exception(sample_data_model):
    """测试evaluate_quality的consistency计算异常处理（171-172行）"""
    monitor = DataQualityMonitor()
    
    # 创建一个会导致异常的数据
    df = pd.DataFrame({
        'value': [1, 2, 3]
    })
    # 使用无效的索引来触发异常
    df.index = [None, None, None]  # 这可能导致异常
    
    model = DataModel(data=df)
    
    # 应该能处理异常
    result = monitor.evaluate_quality(model)
    assert result is not None


def test_data_quality_monitor_evaluate_quality_timeliness_exception(sample_data_model):
    """测试evaluate_quality的timeliness计算异常处理（186-187行）"""
    monitor = DataQualityMonitor()
    
    # 创建一个有无效created_at的模型
    model = DataModel(data=pd.DataFrame({'value': [1, 2, 3]}))
    model.set_metadata({'created_at': 'invalid_date_format'})
    
    # 应该能处理异常
    result = monitor.evaluate_quality(model)
    assert result is not None


def test_data_quality_monitor_evaluate_quality_timeliness_no_created_at(sample_data_model):
    """测试evaluate_quality的timeliness当没有created_at时（188-189行）"""
    monitor = DataQualityMonitor()
    
    # 创建没有created_at的模型
    model = DataModel(data=pd.DataFrame({'value': [1, 2, 3]}))
    model.set_metadata({})  # 没有created_at
    
    result = monitor.evaluate_quality(model)
    assert result is not None


def test_data_quality_monitor_evaluate_quality_timeliness_days_ago_0(sample_data_model):
    """测试evaluate_quality的timeliness当days_ago为0时（180-181行）"""
    monitor = DataQualityMonitor()
    
    # 创建created_at为今天的模型
    model = DataModel(data=pd.DataFrame({'value': [1, 2, 3]}))
    model.set_metadata({'created_at': datetime.now().isoformat()})
    
    result = monitor.evaluate_quality(model)
    assert result is not None
    # timeliness应该为1.0
    assert result.timeliness == 1.0




def test_data_quality_monitor_evaluate_quality_timeliness_days_ago_between(sample_data_model):
    """测试evaluate_quality的timeliness当days_ago在1-5之间时（184-185行）"""
    monitor = DataQualityMonitor()
    
    # 创建created_at为3天前的模型
    old_date = (datetime.now() - timedelta(days=3)).isoformat()
    model = DataModel(data=pd.DataFrame({'value': [1, 2, 3]}))
    model.set_metadata({'created_at': old_date})
    
    result = monitor.evaluate_quality(model)
    assert result is not None
    # timeliness应该在0.0和1.0之间
    assert 0.0 <= result.timeliness <= 1.0


def test_data_quality_monitor_evaluate_quality_consistency_equal_intervals(sample_data_model):
    """测试evaluate_quality的consistency当索引等间隔时（167-168行）"""
    monitor = DataQualityMonitor()
    
    # 创建等间隔索引的DataFrame
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
    
    model = DataModel(data=df)
    result = monitor.evaluate_quality(model)
    
    assert result is not None
    # consistency应该为1.0（等间隔）
    assert result.consistency == 1.0


def test_data_quality_monitor_evaluate_quality_consistency_unequal_intervals(sample_data_model):
    """测试evaluate_quality的consistency当索引不等间隔时（169-170行）"""
    monitor = DataQualityMonitor()
    
    # 创建不等间隔索引的DataFrame
    df = pd.DataFrame({
        'value': [1, 2, 3]
    }, index=[datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 5)])  # 不等间隔
    
    model = DataModel(data=df)
    result = monitor.evaluate_quality(model)
    
    assert result is not None
    # consistency应该为0.0（不等间隔）
    assert result.consistency == 0.0

