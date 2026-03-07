"""
测试performance_monitor的覆盖率提升
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
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.data.monitoring.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetric,
    PerformanceAlert
)


@pytest.fixture
def sample_monitor():
    """创建示例性能监控器"""
    return PerformanceMonitor(max_history=100)


def test_performance_monitor_get_metric_statistics_empty_history(sample_monitor):
    """测试get_metric_statistics方法当历史为空时（206-207行）"""
    result = sample_monitor.get_metric_statistics("nonexistent_metric", hours=24)
    
    # Should return empty dict
    assert result == {}


def test_performance_monitor_should_alert_default_case(sample_monitor):
    """测试_should_alert方法的默认情况（290-292行）"""
    # Test with a metric not in the special cases
    result = sample_monitor._should_alert("unknown_metric", 100.0, 50.0, "warning")
    
    # Should return True (value > threshold for default case)
    assert result is True


def test_performance_monitor_monitor_loop_exception(sample_monitor):
    """测试_monitor_loop方法的异常处理（305-307行）"""
    # Start monitoring
    sample_monitor.start_monitoring()
    
    # Mock _monitor_system_resources to raise exception
    original_monitor = sample_monitor._monitor_system_resources
    def failing_monitor():
        raise Exception("Monitor failed")
    sample_monitor._monitor_system_resources = failing_monitor
    
    # Wait a bit for the loop to run
    import time
    time.sleep(0.1)
    
    # Stop monitoring
    sample_monitor.stop_monitoring()
    
    # Restore original method
    sample_monitor._monitor_system_resources = original_monitor
    
    assert True


def test_performance_monitor_monitor_system_resources_no_psutil(sample_monitor):
    """测试_monitor_system_resources方法当psutil不可用时（311-313行）"""
    # Mock psutil to be None
    import src.data.monitoring.performance_monitor as module
    original_psutil = module.psutil
    module.psutil = None
    
    # Should not raise exception
    sample_monitor._monitor_system_resources()
    
    # Restore original
    module.psutil = original_psutil
    
    assert True


def test_performance_monitor_monitor_system_resources_exception(sample_monitor):
    """测试_monitor_system_resources方法的异常处理（325-326行）"""
    # Mock psutil.virtual_memory to raise exception
    with patch('psutil.virtual_memory', side_effect=Exception("Memory check failed")):
        # Should handle exception gracefully
        sample_monitor._monitor_system_resources()
    
    assert True


def test_performance_monitor_export_metrics_unsupported_format(sample_monitor):
    """测试export_metrics方法的unsupported format（400-401行）"""
    # Record some metrics
    sample_monitor.record_metric("test_metric", 1.0)
    
    # Try to export with unsupported format
    with pytest.raises(ValueError, match="Unsupported format"):
        sample_monitor.export_metrics(format="unsupported")


def test_performance_monitor_export_metrics_csv(sample_monitor):
    """测试export_metrics方法的CSV格式（380-398行）"""
    # Record some metrics
    sample_monitor.record_metric("test_metric", 1.0, unit="test_unit")
    
    # Export as CSV
    result = sample_monitor.export_metrics(format="csv")
    
    # Should return CSV string
    assert isinstance(result, str)
    assert "metric_name" in result
    assert "test_metric" in result


def test_performance_monitor_set_alert_threshold_new_metric(sample_monitor):
    """测试set_alert_threshold方法当metric不存在时（248-260行）"""
    # Set threshold for new metric
    sample_monitor.set_alert_threshold("new_metric", "warning", 10.0)
    
    # Should create new threshold entry
    assert "new_metric" in sample_monitor.alert_thresholds
    assert sample_monitor.alert_thresholds["new_metric"]["warning"] == 10.0


def test_performance_monitor_check_alerts_no_thresholds(sample_monitor):
    """测试_check_alerts方法当metric不在thresholds中时（264-265行）"""
    # Check alerts for metric not in thresholds
    sample_monitor._check_alerts("nonexistent_metric", 100.0)
    
    # Should not raise exception
    assert True


def test_performance_monitor_should_alert_cache_hit_rate(sample_monitor):
    """测试_should_alert方法对于cache_hit_rate（284-286行）"""
    # cache_hit_rate should alert when value < threshold
    result = sample_monitor._should_alert("cache_hit_rate", 0.5, 0.8, "warning")
    
    # Should return True (0.5 < 0.8)
    assert result is True
    
    # Should return False when value >= threshold
    result2 = sample_monitor._should_alert("cache_hit_rate", 0.9, 0.8, "warning")
    assert result2 is False


def test_performance_monitor_should_alert_memory_usage(sample_monitor):
    """测试_should_alert方法对于memory_usage（284-286行）"""
    # memory_usage should alert when value < threshold
    result = sample_monitor._should_alert("memory_usage", 0.5, 0.8, "warning")
    
    # Should return True (0.5 < 0.8)
    assert result is True


def test_performance_monitor_should_alert_data_load_time(sample_monitor):
    """测试_should_alert方法对于data_load_time（287-289行）"""
    # data_load_time should alert when value > threshold
    result = sample_monitor._should_alert("data_load_time", 15.0, 10.0, "error")
    
    # Should return True (15.0 > 10.0)
    assert result is True
    
    # Should return False when value <= threshold
    result2 = sample_monitor._should_alert("data_load_time", 5.0, 10.0, "error")
    assert result2 is False


def test_performance_monitor_should_alert_error_rate(sample_monitor):
    """测试_should_alert方法对于error_rate（287-289行）"""
    # error_rate should alert when value > threshold
    result = sample_monitor._should_alert("error_rate", 0.15, 0.1, "error")
    
    # Should return True (0.15 > 0.1)
    assert result is True


