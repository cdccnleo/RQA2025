import importlib
import time
import types

import pytest
import sys


@pytest.fixture(autouse=True)
def stub_psutil(monkeypatch):
    """替换 psutil 为可控的测试桩，避免真实系统调用与阻塞。"""
    module = importlib.import_module("src.infrastructure.monitoring.application.application_monitor")

    class StubPsutil:
        def __init__(self):
            self.cpu_percent_value = 15.0
            self.memory_percent_value = 25.0
            self.disk_percent_value = 35.0
            self.net = types.SimpleNamespace(bytes_sent=128, bytes_recv=256)

        def cpu_count(self, logical=False):
            return 8 if logical else 4

        def virtual_memory(self):
            return types.SimpleNamespace(
                percent=self.memory_percent_value,
                used=2048,
                available=4096,
                total=8192,
            )

        def disk_usage(self, path):
            return types.SimpleNamespace(
                percent=self.disk_percent_value,
                used=5120,
                free=10240,
                total=15360,
            )

        def cpu_percent(self, interval=0):
            return self.cpu_percent_value

        def net_io_counters(self):
            return self.net

    stub = StubPsutil()
    monkeypatch.setattr(module, "psutil", stub)
    return module, stub


@pytest.fixture
def app_module(stub_psutil):
    module, _ = stub_psutil
    return module


@pytest.fixture
def ApplicationMonitor(app_module):
    return app_module.ApplicationMonitor


@pytest.fixture
def psutil_stub(stub_psutil):
    _, stub = stub_psutil
    return stub


class StubLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)

    def debug(self, message):  # pragma: no cover - 调试路径未使用
        pass


def test_initialization_records_system_metrics(ApplicationMonitor):
    monitor = ApplicationMonitor(app_name="demo-app")

    assert monitor.metrics["system.cpu_count"]["value"] == 4
    assert monitor.metrics["system.cpu_count_logical"]["value"] == 8
    assert monitor.metrics["system.memory.total"]["value"] == 8192
    assert monitor.metrics["system.disk.total"]["value"] == 15360


def test_metric_lifecycle_and_recent_order(ApplicationMonitor):
    monitor = ApplicationMonitor()
    monitor.record_metric("custom.metric", 123, {"layer": "test"})

    assert monitor.get_metric("custom.metric")["value"] == 123
    assert monitor.get_all_metrics()["custom.metric"]["tags"]["layer"] == "test"

    baseline = max(
        value.get("timestamp", 0.0) for value in monitor.metrics.values()
    )
    monitor.metrics["custom.metric"]["timestamp"] = baseline + 10

    metrics = monitor.get_recent_metrics(limit=1)
    assert metrics[0]["name"] == "custom.metric"

    monitor.clear_metrics()
    assert monitor.get_all_metrics() == {}


def test_collect_performance_metrics_records_data_and_alerts(monkeypatch, ApplicationMonitor, psutil_stub):
    monitor = ApplicationMonitor()
    psutil_stub.cpu_percent_value = 92.5
    psutil_stub.memory_percent_value = 91.0
    psutil_stub.disk_percent_value = 93.0

    monitor.set_alert_threshold("cpu_percent", 50.0)
    monitor.set_alert_threshold("memory_percent", 60.0)
    monitor.set_alert_threshold("disk_usage_percent", 70.0)

    logger = StubLogger()
    monitor.logger = logger

    monitor.collect_performance_metrics()

    assert "performance.cpu_percent" in monitor.metrics
    assert "performance.memory.used" in monitor.metrics
    assert "performance.disk.free" in monitor.metrics
    assert "performance.network.bytes_sent" in monitor.metrics
    assert monitor.performance_history
    assert "alert.performance" in monitor.metrics
    assert logger.warnings


def test_get_performance_summary_handles_empty_history(ApplicationMonitor):
    monitor = ApplicationMonitor()

    summary = monitor.get_performance_summary()
    assert "error" in summary


def test_get_performance_summary_computes_statistics(ApplicationMonitor):
    monitor = ApplicationMonitor()
    now = time.time()
    monitor.performance_history.clear()
    monitor.performance_history.extend(
        [
            {"timestamp": now - 60, "cpu_percent": 10, "memory_percent": 20, "disk_percent": 30},
            {"timestamp": now - 30, "cpu_percent": 40, "memory_percent": 50, "disk_percent": 60},
        ]
    )

    summary = monitor.get_performance_summary(hours=1)

    assert summary["data_points"] == 2
    assert summary["cpu_percent"]["avg"] == pytest.approx(25.0)
    assert summary["memory_percent"]["max"] == 50
    assert summary["disk_percent"]["min"] == 30


def test_collect_system_info_failure_logs_warning(monkeypatch, ApplicationMonitor, psutil_stub):
    monitor = ApplicationMonitor()
    logger = StubLogger()
    monitor.logger = logger

    def failing_cpu_count(logical=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(psutil_stub, "cpu_count", failing_cpu_count)

    monitor._collect_system_info()
    assert logger.warnings


def test_collect_performance_metrics_failure_logs_error(monkeypatch, ApplicationMonitor, psutil_stub):
    monitor = ApplicationMonitor()
    logger = StubLogger()
    monitor.logger = logger

    def failing_cpu_percent(interval=0):
        raise RuntimeError("cpu boom")

    monkeypatch.setattr(psutil_stub, "cpu_percent", failing_cpu_percent)

    monitor.collect_performance_metrics()
    assert logger.errors


def test_health_check_reports_current_state(ApplicationMonitor):
    monitor = ApplicationMonitor()
    monitor.record_metric("custom.health", 1)

    health = monitor.health_check()
    assert health["status"] == "healthy"
    assert health["metrics_count"] >= 1
    assert "uptime" in health and health["uptime"] >= 0


def test_monitoring_loop_respects_active_flag(monkeypatch, ApplicationMonitor):
    monitor = ApplicationMonitor()
    calls = []

    def fake_collect():
        calls.append("collect")
        monitor.monitoring_active = False

    monitor.collect_performance_metrics = fake_collect  # type: ignore
    monitor.monitoring_active = True

    monkeypatch.setattr("src.infrastructure.monitoring.application.application_monitor.time.sleep", lambda _: None)
    monitor._monitoring_loop(interval=5)

    assert calls == ["collect"]


def test_start_and_stop_monitoring_manages_thread(monkeypatch, app_module, ApplicationMonitor):

    created_threads = []

    class DummyThread:
        def __init__(self, target=None, args=None):
            self.target = target
            self.args = args or ()
            self.daemon = False
            self.started = False
            self.join_called = False

        def start(self):
            self.started = True

        def join(self, timeout=None):
            self.join_called = True

    def fake_thread(target=None, args=None):
        thread = DummyThread(target, args)
        created_threads.append(thread)
        return thread

    monkeypatch.setattr(app_module.threading, "Thread", fake_thread)

    monitor = ApplicationMonitor()
    monitor.logger = StubLogger()

    monitor.start_monitoring(interval=2)
    assert monitor.monitoring_active is True
    assert created_threads[0].started is True

    monitor.start_monitoring(interval=1)
    assert len(created_threads) == 1  # 不应重复创建线程

    monitor.stop_monitoring()
    assert monitor.monitoring_active is False
    assert created_threads[0].join_called is True


def test_set_alert_threshold_updates_copy(ApplicationMonitor):
    monitor = ApplicationMonitor()
    monitor.set_alert_threshold("cpu_percent", 42.0)

    thresholds = monitor.get_alert_thresholds()
    assert thresholds["cpu_percent"] == 42.0
    thresholds["cpu_percent"] = 1.0

    # 确认返回的是副本
    assert monitor.alert_thresholds["cpu_percent"] == 42.0


def test_module_reload_keeps_exports(monkeypatch, psutil_stub):
    module = importlib.import_module("src.infrastructure.monitoring.application.application_monitor")
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)
    reloaded = importlib.reload(module)

    assert hasattr(reloaded, "ApplicationMonitor")
    # reload 后重新绑定测试桩以不影响后续用例
    monkeypatch.setattr(reloaded, "psutil", psutil_stub)

