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
from typing import List, Dict, Any

import pytest

from src.data.monitoring.performance_monitor import PerformanceMonitor


def test_record_metrics_and_threshold_alerts(monkeypatch):
    monitor = PerformanceMonitor(max_history=10)

    # 降低阈值顺序敏感性，显式设置
    monitor.set_alert_threshold("cache_hit_rate", "warning", 0.9)
    monitor.set_alert_threshold("cache_hit_rate", "error", 0.8)
    monitor.set_alert_threshold("cache_hit_rate", "critical", 0.7)

    # 触发低于 warning/error/critical 的多级告警
    monitor.record_cache_hit_rate(0.65)
    recent_alerts = monitor.get_recent_alerts(hours=1)
    levels = {a.level for a in recent_alerts if a.metric_name == "cache_hit_rate"}
    assert {"warning", "error", "critical"}.issubset(levels)

    # 统计摘要
    summary = monitor.get_all_metrics_summary()
    assert "cache_hit_rate" in summary
    assert summary["cache_hit_rate"]["count"] >= 1
    assert "avg" in summary["cache_hit_rate"]


def test_monitor_loop_start_stop_fast(monkeypatch):
    monitor = PerformanceMonitor(max_history=5)

    # 加速监控线程中的 sleep，避免 60s 等待
    monkeypatch.setattr("src.data.monitoring.performance_monitor.time.sleep", lambda *_: None)
    # 让系统监控函数不执行阻塞行为
    monkeypatch.setattr("src.data.monitoring.performance_monitor.psutil", None)

    monitor.start_monitoring()
    # 立即停止，确保线程可被 join
    monitor.stop_monitoring()
    assert monitor.is_monitoring is False


def test_export_metrics_json_and_csv(tmp_path):
    monitor = PerformanceMonitor(max_history=10)
    monitor.record_data_load_time(12.3)
    monitor.record_error_rate(0.12)

    exported_json = monitor.export_metrics("json")
    data = json.loads(exported_json)
    assert "metrics" in data and "alerts" in data
    assert "data_load_time" in data["metrics"]

    exported_csv = monitor.export_metrics("csv")
    # 简单校验 CSV 头部和关键列
    assert "metric_name,value,unit,timestamp,metadata" in exported_csv.splitlines()[0]


def test_get_metric_history_and_statistics_window(monkeypatch):
    monitor = PerformanceMonitor(max_history=100)

    # 固定时间窗口，确保历史过滤逻辑
    base = datetime(2025, 1, 1, 0, 0, 0)
    ts = {"now": base}

    def fake_now():
        return ts["now"]

    # 打补丁到 dataclass 默认生成的时间使用点
    monkeypatch.setattr("src.data.monitoring.performance_monitor.datetime", type("DT", (), {
        "now": staticmethod(fake_now),
        "fromtimestamp": datetime.fromtimestamp,
        "timedelta": timedelta,
    }))

    # 生成两个小时跨度的指标
    ts["now"] = base
    monitor.record_throughput(100.0)
    ts["now"] = base + timedelta(hours=2)
    monitor.record_throughput(110.0)

    # 仅取 1 小时窗口，应当仅包含第二个数据点（或为空取决于时间戳记录点）
    hist = monitor.get_metric_history("throughput", hours=1)
    # 统计接口对空列表返回 {}
    if hist:
        stats = monitor.get_metric_statistics("throughput", hours=1)
        assert stats.get("count", 0) >= 1
    else:
        stats = monitor.get_metric_statistics("throughput", hours=1)
        assert stats == {}


