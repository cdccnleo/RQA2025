import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.infrastructure.resource.core.resource_manager import CoreResourceManager, ResourceMonitorConfig


@pytest.fixture
def manager(monkeypatch):
    # 防止后台线程
    dummy_thread = MagicMock()
    dummy_thread.is_alive.return_value = False
    monkeypatch.setattr(
        "threading.Thread",
        lambda target, daemon=True: dummy_thread,
    )
    config = ResourceMonitorConfig(
        enable_cpu_monitoring=True,
        enable_memory_monitoring=True,
        enable_disk_monitoring=True,
        thresholds={"cpu_warning": 70.0, "memory_warning": 75.0, "disk_warning": 80.0},
        alert_threshold={"cpu": 90.0, "memory": 85.0, "disk": 80.0},
    )
    mgr = CoreResourceManager(config=config)
    mgr._monitor_thread = dummy_thread
    return mgr, dummy_thread


def mock_psutil(monkeypatch, cpu=15.0, memory_percent=35.0, disk_percent=45.0):
    monkeypatch.setattr("psutil.cpu_percent", MagicMock(return_value=cpu))
    memory = MagicMock()
    memory.percent = memory_percent
    memory.used = 2 * 1024**3
    memory.total = 8 * 1024**3
    memory.available = 6 * 1024**3
    monkeypatch.setattr("psutil.virtual_memory", MagicMock(return_value=memory))
    disk = MagicMock()
    disk.percent = disk_percent
    disk.used = 100 * 1024**3
    disk.total = 200 * 1024**3
    disk.free = 100 * 1024**3
    monkeypatch.setattr("psutil.disk_usage", MagicMock(return_value=disk))
    monkeypatch.setattr("psutil.cpu_count", MagicMock(side_effect=[4, 8]))


def test_get_usage_history_filters_by_hours(manager):
    mgr, _ = manager
    now = datetime.now()
    mgr._resource_history = [
        {"timestamp": (now - timedelta(hours=3)).isoformat(), "cpu_percent": 10},
        {"timestamp": (now - timedelta(minutes=30)).isoformat(), "cpu_percent": 20},
    ]
    history = mgr.get_usage_history(hours=1)
    assert history["count"] == 1
    assert history["history"][0]["cpu_percent"] == 20


def test_get_resource_history_limit(manager):
    mgr, _ = manager
    mgr._resource_history = [{"timestamp": f"t{i}"} for i in range(5)]
    limited = mgr.get_resource_history(limit=2)
    assert len(limited) == 2
    assert limited[0]["timestamp"] == "t3"


def test_get_current_usage_includes_health(monkeypatch, manager):
    mock_psutil(monkeypatch, cpu=95.0, memory_percent=88.0, disk_percent=81.0)
    mgr, _ = manager
    usage = mgr.get_current_usage()
    assert usage["overall_health"] == "critical"
    assert any("CPU使用率过高" in msg for msg in usage["warnings"])
    assert any("内存使用率过高" in msg for msg in usage["alerts"])
    assert usage["disk_usage"]["percent"] == 81.0


def test_get_current_usage_error_path(manager, monkeypatch):
    mgr, _ = manager
    monkeypatch.setattr(
        mgr,
        "_collect_resource_info",
        MagicMock(side_effect=RuntimeError("collect fail")),
    )
    usage = mgr.get_current_usage()
    assert usage["overall_health"] == "unknown"
    assert usage["error"] == "无法获取资源信息"


def test_get_resource_limits(monkeypatch, manager):
    mock_psutil(monkeypatch, cpu=10.0, memory_percent=20.0, disk_percent=30.0)
    mgr, _ = manager
    limits = mgr.get_resource_limits()
    assert limits["cpu_cores"] == 4
    assert limits["cpu_logical"] == 8


def test_check_resource_health_warns(monkeypatch, manager):
    mgr, _ = manager
    monkeypatch.setattr(
        mgr,
        "get_current_usage",
        MagicMock(
            return_value={
                "cpu_percent": 100.0,
                "memory_percent": 95.0,
                "disk_percent": 92.0,
            }
        ),
    )
    monkeypatch.setattr(
        mgr,
        "get_resource_limits",
        MagicMock(return_value={"cpu_limit_percent": 95.0, "memory_limit_percent": 90.0}),
    )
    health = mgr.check_resource_health()
    assert health["overall_health"] == "warning"
    assert len(health["issues"]) == 3


def test_get_resource_summary_aggregates(monkeypatch, manager):
    mgr, _ = manager
    monkeypatch.setattr(mgr, "get_current_usage", MagicMock(return_value={"cpu_percent": 10}))
    summary = mgr.get_resource_summary()
    assert summary["history_count"] == len(mgr._resource_history)
    assert "alerts" in summary


def test_check_alerts_handles_error(monkeypatch, manager):
    mgr, _ = manager
    monkeypatch.setattr(mgr, "get_current_usage", MagicMock(side_effect=RuntimeError("boom")))
    alerts = mgr._check_alerts()
    assert alerts == []


def test_stop_monitoring_join_called(manager):
    mgr, monitor_thread = manager
    monitor_thread.is_alive.return_value = True
    mgr.stop_monitoring()
    monitor_thread.join.assert_called_once()


def test_collect_resource_info_handles_errors(monkeypatch, manager):
    mgr, _ = manager
    def raising_virtual_memory():
        raise RuntimeError("memory fail")

    monkeypatch.setattr("psutil.virtual_memory", raising_virtual_memory)
    monkeypatch.setattr("psutil.cpu_percent", MagicMock(return_value=10.0))
    info = mgr._collect_resource_info()
    assert "cpu_percent" in info

