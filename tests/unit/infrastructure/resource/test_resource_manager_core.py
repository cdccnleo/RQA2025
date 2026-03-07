import threading
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.resource.core.resource_manager import CoreResourceManager, ResourceMonitorConfig


class DummyThread:
    def __init__(self, target=None, daemon=None):
        self.target = target
        self.daemon = daemon
        self._alive = False

    def start(self):
        self._alive = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self._alive = False


@pytest.fixture
def patched_thread(monkeypatch):
    monkeypatch.setattr("threading.Thread", lambda target, daemon=True: DummyThread(target, daemon))


def make_config(**overrides):
    base = ResourceMonitorConfig()
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def mock_psutil(monkeypatch, cpu=10.5, memory_percent=20.2, disk_percent=30.3):
    monkeypatch.setattr("psutil.cpu_percent", MagicMock(return_value=cpu))
    memory = SimpleNamespace(
        percent=memory_percent,
        used=4 * 1024**3,
        total=8 * 1024**3,
        available=4 * 1024**3,
    )
    disk = SimpleNamespace(
        percent=disk_percent,
        used=200 * 1024**3,
        total=400 * 1024**3,
        free=200 * 1024**3,
    )
    monkeypatch.setattr("psutil.virtual_memory", MagicMock(return_value=memory))
    monkeypatch.setattr("psutil.disk_usage", MagicMock(return_value=disk))
    monkeypatch.setattr("psutil.cpu_count", MagicMock(side_effect=[8, 16]))


def test_collect_resource_info_respects_config(monkeypatch, patched_thread):
    mock_psutil(monkeypatch, cpu=55.5, memory_percent=66.6, disk_percent=77.7)
    config = make_config(
        enable_cpu_monitoring=True,
        enable_memory_monitoring=False,
        enable_disk_monitoring=True,
        precision=1,
    )
    manager = CoreResourceManager(config=config)

    info = manager._collect_resource_info()
    assert info["cpu_percent"] == pytest.approx(55.5, rel=1e-3)
    assert "memory_percent" not in info
    assert info["disk_percent"] == pytest.approx(77.7, rel=1e-3)


def test_get_current_usage_includes_health(monkeypatch, patched_thread):
    mock_psutil(monkeypatch, cpu=95.0, memory_percent=90.0, disk_percent=50.0)
    config = make_config(
        thresholds={"cpu_warning": 80.0, "memory_warning": 85.0, "disk_warning": 90.0},
        alert_threshold={"cpu": 90.0, "memory": 85.0, "disk": 80.0},
    )
    manager = CoreResourceManager(config=config)

    usage = manager.get_current_usage()
    assert usage["disk_usage"]["percent"] == 50.0
    assert usage["overall_health"] == "critical"
    assert any("CPU使用率过高" in msg for msg in usage["warnings"])
    assert any("内存使用率过高" in msg for msg in usage["alerts"])


def test_get_current_usage_handles_error(monkeypatch, patched_thread):
    monkeypatch.setattr(
        CoreResourceManager,
        "_collect_resource_info",
        MagicMock(side_effect=RuntimeError("collect fail")),
    )
    manager = CoreResourceManager()
    response = manager.get_current_usage()
    assert response["overall_health"] == "unknown"
    assert response["error"] == "无法获取资源信息"


def test_usage_history_filters(monkeypatch, patched_thread):
    mock_psutil(monkeypatch)
    manager = CoreResourceManager()
    now = datetime.now()
    with manager._lock:
        manager._resource_history = [
            {"timestamp": (now - timedelta(hours=2)).isoformat(), "value": 1},
            {"timestamp": (now - timedelta(minutes=30)).isoformat(), "value": 2},
        ]
    history = manager.get_usage_history(hours=1)
    assert history["count"] == 1
    assert history["history"][0]["value"] == 2


def test_get_resource_summary_triggers_alerts(monkeypatch, patched_thread):
    mock_psutil(monkeypatch, cpu=95.0, memory_percent=90.0, disk_percent=92.0)
    config = make_config(alert_threshold={"cpu": 90.0, "memory": 85.0, "disk": 80.0})
    manager = CoreResourceManager(config=config)

    summary = manager.get_resource_summary()
    assert summary["history_count"] == 0
    alerts = summary["alerts"]
    assert len(alerts) == 3
    assert any("CPU使用率过高" in alert for alert in alerts)


def test_get_resource_limits_and_health(monkeypatch, patched_thread):
    mock_psutil(monkeypatch, cpu=10.0, memory_percent=20.0, disk_percent=30.0)
    manager = CoreResourceManager()

    limits = manager.get_resource_limits()
    assert limits["cpu_cores"] == 8
    assert limits["cpu_logical"] == 16

    monkeypatch.setattr(
        CoreResourceManager,
        "get_current_usage",
        MagicMock(return_value={"cpu_percent": 96.0, "memory_percent": 91.0, "disk_percent": 95.0}),
    )
    monkeypatch.setattr(
        CoreResourceManager,
        "get_resource_limits",
        MagicMock(return_value={"cpu_limit_percent": 95.0, "memory_limit_percent": 90.0}),
    )
    health = manager.check_resource_health()
    assert health["overall_health"] == "warning"
    assert len(health["issues"]) == 3


def test_stop_monitoring_joins_thread(monkeypatch, patched_thread):
    mock_psutil(monkeypatch)
    manager = CoreResourceManager()
    manager._monitor_thread = DummyThread()
    manager._monitor_thread._alive = True

    manager.stop_monitoring()
    assert not manager._monitor_thread.is_alive()


