from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

import src.infrastructure.resource.monitoring.unified_monitor_adapter as adapter_module
from src.infrastructure.resource.monitoring.unified_monitor_adapter import UnifiedMonitorAdapter
from src.infrastructure.interfaces.standard_interfaces import ServiceStatus


@dataclass
class DummyMetric:
    value: float
    tags: Optional[Dict[str, str]]


class DummyMonitor:
    def __init__(self, config=None):
        self.config = config or {}
        self.is_running = False
        self.metrics: Dict[str, List[DummyMetric]] = {}
        self.alerts: List[Dict[str, Any]] = []

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        self.metrics.setdefault(name, []).append(DummyMetric(value, tags))

    def record_alert(self, level: str, message: str, tags: Optional[Dict[str, str]] = None):
        self.alerts.append({"level": level, "message": message, "tags": tags})

    def get_metrics(self, name: str, time_range=None):
        if name:
            return self.metrics.get(name, [])
        return [metric for items in self.metrics.values() for metric in items]

    def get_alerts(self, level: Optional[str] = None):
        if level:
            return [alert for alert in self.alerts if alert["level"] == level]
        return list(self.alerts)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(adapter_module, "UnifiedMonitor", DummyMonitor)
    return UnifiedMonitorAdapter(custom_option=True)


def test_start_stop_transitions(adapter):
    assert adapter.get_status() == ServiceStatus.STOPPED

    adapter.start()
    assert adapter.get_status() == ServiceStatus.RUNNING
    assert adapter.get_underlying_monitor().is_running is True

    adapter.stop()
    assert adapter.get_status() == ServiceStatus.STOPPED
    assert adapter.get_underlying_monitor().is_running is False


def test_record_metric_and_alert_routes_to_monitor(adapter):
    adapter.record_metric("cpu", 75.5, tags={"host": "agent-1"})
    adapter.record_alert("warning", "high usage", tags={"component": "cpu"})

    metrics = adapter.get_metrics("cpu")
    alerts = adapter.get_alerts()

    assert len(metrics) == 1
    assert metrics[0].value == 75.5
    assert metrics[0].tags == {"host": "agent-1"}

    assert len(alerts) == 1
    assert alerts[0]["level"] == "warning"


def test_collect_metrics_success(monkeypatch, adapter):
    fake_memory = type("Mem", (), {"percent": 41.2, "used": 1024, "total": 8192})
    fake_disk = type("Disk", (), {"percent": 73.5, "used": 2048, "total": 16384})
    fake_network = type("Net", (), {"bytes_sent": 111, "bytes_recv": 222})

    class FakePsutil:
        @staticmethod
        def cpu_percent(interval=0.1):
            return 12.7

        @staticmethod
        def virtual_memory():
            return fake_memory

        @staticmethod
        def disk_usage(_path):
            return fake_disk

        @staticmethod
        def net_io_counters():
            return fake_network

    monkeypatch.setattr(adapter_module, "psutil", FakePsutil)

    adapter.start()
    adapter.record_metric("cpu", 1.0)
    adapter.record_alert("info", "ok")

    result = adapter.collect_metrics()

    assert result["status"] == "running"
    assert result["metrics_count"] == 1
    assert result["alerts_count"] == 1
    assert result["cpu"]["percent"] == pytest.approx(12.7)
    assert result["memory"]["used"] == 1024
    assert result["disk"]["percent"] == pytest.approx(73.5)
    assert result["network"]["bytes_recv"] == 222


def test_collect_metrics_fallback_on_import_error(monkeypatch, adapter):
    class BrokenPsutil:
        @staticmethod
        def cpu_percent(*_args, **_kwargs):
            raise ImportError("cpu")

        @staticmethod
        def virtual_memory():
            raise ImportError("mem")

        @staticmethod
        def disk_usage(_path):
            raise ImportError("disk")

        @staticmethod
        def net_io_counters():
            raise ImportError("net")

    monkeypatch.setattr(adapter_module, "psutil", BrokenPsutil)

    result = adapter.collect_metrics()

    assert result["cpu"]["percent"] == 0.0
    assert result["memory"]["total"] == 0
    assert result["disk"]["used"] == 0
    assert result["network"]["bytes_sent"] == 0


def test_system_status_and_registration(adapter):
    adapter.start()
    adapter.register_monitor({"name": "aux"})

    status = adapter.get_system_status()
    registered = adapter.get_registered_monitors()

    assert status["status"] == ServiceStatus.RUNNING.value
    assert status["monitor_running"] is True
    assert status["overall_health"] == "healthy"
    assert registered == [{"name": "aux"}]
