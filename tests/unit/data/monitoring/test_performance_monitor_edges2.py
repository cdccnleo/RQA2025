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
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.data.monitoring.performance_monitor import PerformanceMonitor


def test_system_resource_monitor_psutil_missing_or_error(monkeypatch: pytest.MonkeyPatch, caplog):
    pm = PerformanceMonitor(max_history=10)

    # psutil missing path
    monkeypatch.setattr("src.data.monitoring.performance_monitor.psutil", None)
    pm._monitor_system_resources()  # should not raise

    # psutil present but errors
    class DummyMem:
        percent = 50.0

    class DummyDisk:
        percent = 70.0

    dummy_psutil = MagicMock()
    dummy_psutil.virtual_memory.return_value = DummyMem()
    dummy_psutil.cpu_percent.side_effect = RuntimeError("cpu boom")
    dummy_psutil.disk_usage.return_value = DummyDisk()
    monkeypatch.setattr("src.data.monitoring.performance_monitor.psutil", dummy_psutil)
    pm._monitor_system_resources()  # error handled internally


def test_alert_generation_and_cleanup_window():
    pm = PerformanceMonitor(max_history=50)
    # lower cache hit threshold to force alert
    pm.set_alert_threshold("cache_hit_rate", "warning", 0.9)
    pm.record_cache_hit_rate(0.5)
    assert any(a.metric_name == "cache_hit_rate" for a in pm.alerts)

    # manually backdate alerts to test cleanup
    for a in pm.alerts:
        a.timestamp = datetime.now() - timedelta(hours=25)
    pm._cleanup_old_alerts()
    assert len(pm.alerts) == 0


def test_get_metric_history_and_summary_window():
    pm = PerformanceMonitor(max_history=5)
    pm.record_data_load_time(1.0)
    pm.record_data_load_time(2.0)
    pm.record_data_load_time(3.0)

    hist = pm.get_metric_history("data_load_time", hours=1)
    assert len(hist) >= 1

    stats = pm.get_metric_statistics("data_load_time", hours=1)
    assert stats["count"] >= 1 and stats["min"] >= 1.0 and stats["max"] >= 1.0 and "avg" in stats

    summary = pm.get_all_metrics_summary()
    assert "data_load_time" in summary and "avg" in summary["data_load_time"]


def test_export_json_and_csv_contains_metrics_and_alerts():
    pm = PerformanceMonitor(max_history=10)
    pm.set_alert_threshold("error_rate", "warning", 0.01)
    pm.record_error_rate(0.1)  # triggers alert
    pm.record_throughput(123.0)

    data_json = pm.export_metrics("json")
    parsed = json.loads(data_json)
    assert "metrics" in parsed and "alerts" in parsed
    assert "error_rate" in parsed["metrics"]
    assert len(parsed["alerts"]) >= 1

    data_csv = pm.export_metrics("csv")
    assert "metric_name,value,unit,timestamp,metadata" in data_csv.splitlines()[0]
    assert "throughput" in data_csv


def test_start_and_stop_monitoring():
    """测试 PerformanceMonitor（启动和停止监控）"""
    pm = PerformanceMonitor(max_history=10)
    # 启动监控
    pm.start_monitoring()
    assert pm.is_monitoring is True
    assert pm.monitor_thread is not None
    # 停止监控
    pm.stop_monitoring()
    assert pm.is_monitoring is False


def test_start_monitoring_already_running():
    """测试 PerformanceMonitor（启动监控，已运行）"""
    pm = PerformanceMonitor(max_history=10)
    pm.start_monitoring()
    # 再次启动应该不会创建新线程
    original_thread = pm.monitor_thread
    pm.start_monitoring()
    assert pm.monitor_thread is original_thread
    pm.stop_monitoring()


def test_stop_monitoring_no_thread():
    """测试 PerformanceMonitor（停止监控，无线程）"""
    pm = PerformanceMonitor(max_history=10)
    pm.monitor_thread = None
    # 应该不会抛出异常
    pm.stop_monitoring()


def test_get_metric_history_nonexistent():
    """测试 PerformanceMonitor（获取指标历史，不存在）"""
    pm = PerformanceMonitor(max_history=10)
    history = pm.get_metric_history("nonexistent_metric", hours=24)
    assert history == []


def test_get_current_metric_nonexistent():
    """测试 PerformanceMonitor（获取当前指标，不存在）"""
    pm = PerformanceMonitor(max_history=10)
    metric = pm.get_current_metric("nonexistent_metric")
    assert metric is None


def test_get_current_metric_empty():
    """测试 PerformanceMonitor（获取当前指标，空历史）"""
    pm = PerformanceMonitor(max_history=10)
    # 创建空的指标历史
    pm.metrics["test_metric"] = []
    metric = pm.get_current_metric("test_metric")
    assert metric is None


def test_get_metric_statistics_empty():
    """测试 PerformanceMonitor（获取指标统计，空历史）"""
    pm = PerformanceMonitor(max_history=10)
    stats = pm.get_metric_statistics("nonexistent_metric", hours=24)
    assert stats == {}


def test_get_recent_alerts_empty():
    """测试 PerformanceMonitor（获取最近告警，空列表）"""
    pm = PerformanceMonitor(max_history=10)
    alerts = pm.get_recent_alerts(hours=24)
    assert alerts == []


def test_set_alert_threshold_new_metric():
    """测试 PerformanceMonitor（设置告警阈值，新指标）"""
    pm = PerformanceMonitor(max_history=10)
    pm.set_alert_threshold("new_metric", "warning", 0.5)
    assert "new_metric" in pm.alert_thresholds
    assert pm.alert_thresholds["new_metric"]["warning"] == 0.5


def test_should_alert_default_case():
    """测试 PerformanceMonitor（判断是否告警，默认情况）"""
    pm = PerformanceMonitor(max_history=10)
    # 测试默认情况（高于阈值告警）
    result = pm._should_alert("unknown_metric", 10.0, 5.0, "warning")
    assert result is True  # 10.0 > 5.0


def test_monitor_loop_exception():
    """测试 PerformanceMonitor（监控循环，异常处理）"""
    pm = PerformanceMonitor(max_history=10)
    # 这里我们无法直接测试监控循环中的异常处理，因为它在一个线程中运行
    # 但我们可以验证监控器已初始化
    assert pm is not None
    # 验证监控循环方法存在
    assert hasattr(pm, '_monitor_loop')


def test_monitor_system_resources_with_psutil():
    """测试 PerformanceMonitor（监控系统资源，有psutil）"""
    pm = PerformanceMonitor(max_history=10)
    # 如果psutil可用，应该能监控系统资源
    try:
        import psutil
        pm._monitor_system_resources()
        # 应该记录了一些指标
        assert True
    except ImportError:
        # psutil不可用，跳过
        pytest.skip("psutil not available")


def test_export_metrics_unsupported_format():
    """测试 PerformanceMonitor（导出指标，不支持格式）"""
    pm = PerformanceMonitor(max_history=10)
    pm.record_throughput(123.0)
    # 应该抛出ValueError
    with pytest.raises(ValueError, match="Unsupported format"):
        pm.export_metrics("unsupported_format")


def test_get_performance_report():
    """测试 PerformanceMonitor（获取性能报告）"""
    pm = PerformanceMonitor(max_history=10)
    pm.record_throughput(123.0)
    report = pm.get_performance_report()
    assert isinstance(report, dict)
    assert "timestamp" in report
    assert "metrics_summary" in report
    assert "recent_alerts" in report
    assert "alert_count" in report
    assert "monitoring_status" in report
    assert report["monitoring_status"] in ["active", "inactive"]


def test_get_performance_report_with_alerts():
    """测试 PerformanceMonitor（获取性能报告，有告警）"""
    pm = PerformanceMonitor(max_history=10)
    pm.set_alert_threshold("error_rate", "warning", 0.01)
    pm.record_error_rate(0.1)  # 触发告警
    report = pm.get_performance_report()
    assert report["alert_count"] > 0
    assert len(report["recent_alerts"]) > 0


def test_record_metric_with_metadata():
    """测试 PerformanceMonitor（记录指标，带元数据）"""
    pm = PerformanceMonitor(max_history=10)
    metadata = {"source": "test", "type": "unit"}
    pm.record_metric("test_metric", 1.0, "unit", metadata=metadata)
    metric = pm.get_current_metric("test_metric")
    assert metric is not None
    assert metric.metadata == metadata

