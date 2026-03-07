import importlib
from collections import deque
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def application_monitor_module():
    return importlib.import_module("src.infrastructure.monitoring.application.application_monitor")


@pytest.fixture
def monitor_with_psutil(monkeypatch, application_monitor_module):
    module = application_monitor_module

    fake_psutil = MagicMock()

    def _cpu_count(logical=False):
        return 4 if not logical else 8

    fake_psutil.cpu_count.side_effect = _cpu_count

    fake_memory = MagicMock(total=16_000, percent=42.0, used=6_720, available=9_280)
    fake_psutil.virtual_memory.return_value = fake_memory

    fake_disk = MagicMock(total=1_000, percent=13.0, used=130, free=870)
    fake_psutil.disk_usage.return_value = fake_disk

    fake_psutil.cpu_percent.return_value = 21.0
    fake_psutil.net_io_counters.return_value = MagicMock(bytes_sent=123, bytes_recv=456)

    monkeypatch.setattr(module, "psutil", fake_psutil)

    monitor = module.ApplicationMonitor(app_name="demo")
    return monitor, fake_psutil, module


def test_initialization_collects_system_info(monitor_with_psutil):
    monitor, fake_psutil, _ = monitor_with_psutil

    assert monitor.get_metric("system.cpu_count")["value"] == 4
    assert monitor.get_metric("system.cpu_count_logical")["value"] == 8
    assert monitor.get_metric("system.memory.total")["value"] == 16_000
    assert monitor.get_metric("system.disk.total")["value"] == 1_000
    assert fake_psutil.cpu_count.call_count >= 2


def test_record_and_retrieve_metrics(monitor_with_psutil):
    monitor, _, module = monitor_with_psutil

    monitor.record_metric("requests.count", 10, tags={"route": "/api"})
    retrieved = monitor.get_metric("requests.count")

    assert retrieved["value"] == 10
    assert retrieved["tags"]["route"] == "/api"
    assert "requests.count" in monitor.get_all_metrics()

    # Clear and ensure metrics are removed
    monitor.clear_metrics()
    assert monitor.get_all_metrics() == {}

    health = monitor.health_check()
    assert health["status"] == "healthy"
    assert health["app_name"] == "demo"
    assert health["metrics_count"] == 0
    assert health["uptime"] >= 0


def test_get_recent_metrics_sorted_limit(monkeypatch, monitor_with_psutil):
    monitor, _, module = monitor_with_psutil

    monitor.clear_metrics()

    timestamps = iter([1000.0, 1001.0, 1002.0, 1003.0, 1004.0])

    def fake_time():
        try:
            return next(timestamps)
        except StopIteration:
            return 1004.0

    monkeypatch.setattr(module.time, "time", fake_time)

    monitor.record_metric("metric.first", 1)
    monitor.record_metric("metric.second", 2)
    monitor.record_metric("metric.third", 3)

    recent_two = monitor.get_recent_metrics(limit=2)
    names = [entry["name"] for entry in recent_two]

    assert names == ["metric.third", "metric.second"]


def test_collect_performance_metrics_records_history_and_alerts(caplog, monitor_with_psutil):
    monitor, fake_psutil, _ = monitor_with_psutil

    fake_memory = MagicMock(percent=95.0, used=12_000, available=4_000)
    fake_psutil.virtual_memory.return_value = fake_memory
    fake_disk = MagicMock(percent=92.0, used=920, free=80)
    fake_psutil.disk_usage.return_value = fake_disk
    fake_psutil.cpu_percent.return_value = 96.0
    fake_psutil.net_io_counters.return_value = MagicMock(bytes_sent=500, bytes_recv=900)

    monitor.collect_performance_metrics()

    assert monitor.performance_history
    last_data = monitor.performance_history[-1]
    assert last_data["cpu_percent"] == 96.0
    assert last_data["memory_percent"] == 95.0
    assert last_data["disk_percent"] == 92.0

    alert_metric = monitor.get_metric("alert.performance")
    assert alert_metric is not None
    assert "过高" in alert_metric["value"]


def test_collect_performance_metrics_handles_exception(caplog, monitor_with_psutil):
    monitor, fake_psutil, _ = monitor_with_psutil
    fake_psutil.cpu_percent.side_effect = RuntimeError("boom")

    with caplog.at_level("ERROR"):
        monitor.collect_performance_metrics()

    assert "Failed to collect performance metrics" in caplog.text


def test_start_and_stop_monitoring_creates_thread(monkeypatch, monitor_with_psutil):
    monitor, _, module = monitor_with_psutil

    class DummyThread:
        def __init__(self, target=None, args=()):
            self.target = target
            self.args = args
            self.started = False
            self.join_called = False
            self.daemon = False

        def start(self):
            self.started = True

        def join(self, timeout=None):
            self.join_called = True

    created_threads = []

    def fake_thread(target=None, args=(), daemon=None):
        thread = DummyThread(target, args)
        created_threads.append(thread)
        return thread

    monkeypatch.setattr(module.threading, "Thread", fake_thread)

    monitor.start_monitoring(interval=5)
    assert monitor.monitoring_active is True
    assert created_threads and created_threads[0].started is True

    monitor.stop_monitoring()
    assert monitor.monitoring_active is False
    assert created_threads[0].join_called is True


def test_start_monitoring_is_idempotent(monkeypatch, monitor_with_psutil):
    monitor, _, module = monitor_with_psutil

    thread_instances = []

    class DummyThread:
        def __init__(self, *args, **kwargs):
            thread_instances.append(self)
            self.started = False

        def start(self):
            self.started = True

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(module.threading, "Thread", lambda *a, **kw: DummyThread())

    monitor.start_monitoring()
    assert monitor.monitoring_active is True
    assert thread_instances and thread_instances[0].started is True

    monitor.start_monitoring()
    # second call should do nothing
    assert len(thread_instances) == 1

    monitor.stop_monitoring()


def test_get_performance_summary_with_data(monkeypatch, monitor_with_psutil):
    monitor, _, module = monitor_with_psutil

    now = 10_000.0
    monkeypatch.setattr(module.time, "time", lambda: now)

    monitor.performance_history = deque(
        [
            {"timestamp": now - 100, "cpu_percent": 40, "memory_percent": 60, "disk_percent": 55},
            {"timestamp": now - 50, "cpu_percent": 60, "memory_percent": 70, "disk_percent": 65},
        ]
    )

    summary = monitor.get_performance_summary(hours=1)

    assert summary["data_points"] == 2
    assert summary["cpu_percent"]["avg"] == pytest.approx(50)
    assert summary["memory_percent"]["max"] == 70
    assert summary["disk_percent"]["min"] == 55


def test_get_performance_summary_no_data(monkeypatch, monitor_with_psutil):
    monitor, _, module = monitor_with_psutil

    monkeypatch.setattr(module.time, "time", lambda: 1_000_000.0)
    monitor.performance_history.clear()

    summary = monitor.get_performance_summary(hours=1)
    assert summary == {"error": "No performance data available"}


def test_set_and_get_alert_thresholds(monitor_with_psutil):
    monitor, _, _ = monitor_with_psutil

    monitor.set_alert_threshold("cpu_percent", 75.0)
    thresholds = monitor.get_alert_thresholds()

    assert thresholds["cpu_percent"] == 75.0
    assert "memory_percent" in thresholds


def test_collect_system_info_failure(application_monitor_module, monkeypatch, caplog):
    module = application_monitor_module
    failing_psutil = MagicMock()
    failing_psutil.cpu_count.side_effect = RuntimeError("no cpu")
    monkeypatch.setattr(module, "psutil", failing_psutil)

    with caplog.at_level("WARNING"):
        module.ApplicationMonitor(app_name="fail")

    assert "Failed to collect system info" in caplog.text


def test_monitoring_loop_runs_until_inactive(monkeypatch, monitor_with_psutil):
    monitor, _, module = monitor_with_psutil

    calls = []

    def fake_collect():
        calls.append("called")
        monitor.monitoring_active = False

    monitor.monitoring_active = True
    monitor.collect_performance_metrics = fake_collect
    monkeypatch.setattr(module.time, "sleep", lambda interval: None)

    monitor._monitoring_loop(interval=1)

    assert calls == ["called"]


def test_get_all_metrics(monitor_with_psutil):
    """Test getting all metrics."""
    monitor, _, _ = monitor_with_psutil

    # Add some metrics
    monitor.record_metric("test.metric1", 100, {"tag": "value1"})
    monitor.record_metric("test.metric2", 200, {"tag": "value2"})

    all_metrics = monitor.get_all_metrics()

    assert isinstance(all_metrics, dict)
    assert "test.metric1" in all_metrics
    assert "test.metric2" in all_metrics
    assert all_metrics["test.metric1"]["value"] == 100
    assert all_metrics["test.metric2"]["value"] == 200


def test_clear_metrics(monitor_with_psutil):
    """Test clearing all metrics."""
    monitor, _, _ = monitor_with_psutil

    # Add some metrics
    monitor.record_metric("test.metric1", 100)
    monitor.record_metric("test.metric2", 200)

    # Verify metrics were added
    assert len(monitor.get_all_metrics()) >= 2  # At least our test metrics

    # Clear metrics
    monitor.clear_metrics()

    # Verify all metrics are cleared
    assert len(monitor.get_all_metrics()) == 0


def test_get_uptime(monitor_with_psutil):
    """Test getting application uptime."""
    monitor, _, _ = monitor_with_psutil

    uptime = monitor.get_uptime()

    assert isinstance(uptime, float)
    assert uptime >= 0


def test_health_check(monitor_with_psutil):
    """Test health check functionality."""
    monitor, fake_psutil, _ = monitor_with_psutil

    health = monitor.health_check()

    assert isinstance(health, dict)
    assert "status" in health
    assert "timestamp" in health
    assert "uptime" in health
    assert "metrics_count" in health

    # Test with some metrics
    monitor.record_metric("test.metric", 100)
    health_with_metrics = monitor.health_check()

    assert health_with_metrics["metrics_count"] > 0


def test_check_alerts_method(monitor_with_psutil):
    """Test internal _check_alerts method."""
    monitor, fake_psutil, module = monitor_with_psutil

    # Create test performance data that should trigger alerts
    performance_data = {
        "cpu_percent": 85.0,  # Above threshold
        "memory_percent": 90.0,  # Above threshold
        "disk_percent": 95.0  # Above threshold (note: key is 'disk_percent' not 'disk_usage_percent')
    }

    # Mock logger to capture alert messages
    from unittest.mock import patch
    with patch.object(monitor.logger, 'warning') as mock_warning:
        monitor._check_alerts(performance_data)

        # Verify alerts were triggered for all metrics above threshold
        assert mock_warning.call_count == 3
        call_args_list = [call[0][0] for call in mock_warning.call_args_list]

        assert any("CPU使用率过高" in msg for msg in call_args_list)
        assert any("内存使用率过高" in msg for msg in call_args_list)
        assert any("磁盘使用率过高" in msg for msg in call_args_list)

