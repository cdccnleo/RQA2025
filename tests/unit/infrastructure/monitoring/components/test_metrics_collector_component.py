import json
from datetime import datetime

import pytest

class _FakeIOCounters:
    def __init__(self, sent=123, recv=456):
        self.bytes_sent = sent
        self.bytes_recv = recv


def _make_fake_psutil():
    class FakeVM:
        total = 4 * 1024**3
        available = 2 * 1024**3
        percent = 50.0
        used = 2 * 1024**3

    class FakeDisk:
        total = 1 * 1024**4
        used = 512 * 1024**3
        free = 512 * 1024**3
        percent = 50.0

    class FakeFreq:
        current = 2400.0

    class FakePsutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 42.0

        @staticmethod
        def virtual_memory():
            return FakeVM()

        @staticmethod
        def disk_usage(path):
            assert path == "/"
            return FakeDisk()

        @staticmethod
        def net_connections():
            return [object(), object()]

        @staticmethod
        def net_if_addrs():
            return {"eth0": [], "lo": []}

        @staticmethod
        def net_io_counters():
            return _FakeIOCounters()

        @staticmethod
        def cpu_count():
            return 8

        @staticmethod
        def cpu_freq():
            return FakeFreq()

        @staticmethod
        def boot_time():
            return datetime.now().timestamp() - 3600

    return FakePsutil()


@pytest.fixture
def metrics_mod():
    import importlib
    import sys

    for name in (
        "src.infrastructure.monitoring.components.metrics_collector",
        "infrastructure.monitoring.components.metrics_collector",
    ):
        sys.modules.pop(name, None)

    module = importlib.import_module("src.infrastructure.monitoring.components.metrics_collector")
    return module


def test_psutil_import_failure(monkeypatch):
    import importlib
    import sys
    import builtins

    for name in (
        "src.infrastructure.monitoring.components.metrics_collector",
        "infrastructure.monitoring.components.metrics_collector",
    ):
        sys.modules.pop(name, None)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psutil":
            raise ImportError("psutil missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module = importlib.import_module("src.infrastructure.monitoring.components.metrics_collector")
    assert module.PSUTIL_AVAILABLE is False

    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(module)


def test_collect_system_metrics_with_psutil(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector()
    monkeypatch.setattr(metrics_mod, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(metrics_mod, "psutil", _make_fake_psutil())

    metrics = collector.collect_system_metrics()

    assert metrics["cpu_percent"] == pytest.approx(42.0)
    assert metrics["memory_percent"] == pytest.approx(50.0)
    assert metrics["disk_usage"] == pytest.approx(50.0)
    assert metrics["network_connections"] == 2, metrics


def test_collect_system_metrics_force_mock(metrics_mod):
    collector = metrics_mod.MetricsCollector()
    collector._force_mock = True

    metrics = collector.collect_system_metrics()
    assert metrics == collector._get_mock_system_metrics()


def test_collect_system_metrics_error_returns_zero(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector()
    monkeypatch.setattr(metrics_mod, "PSUTIL_AVAILABLE", True)

    class BrokenPsutil:
        @staticmethod
        def cpu_percent(interval=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(metrics_mod, "psutil", BrokenPsutil)

    metrics = collector.collect_system_metrics()
    assert metrics == collector._get_zero_system_metrics()


def test_collect_test_coverage_metrics_success(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector(project_root=".")

    class Result:
        returncode = 0
        stdout = json.dumps(
            {"totals": {"num_statements": 100, "num_covered": 90, "percent_covered": 90.0, "num_missing": 10}}
        )
        stderr = ""

    monkeypatch.setattr(metrics_mod.subprocess, "run", lambda *args, **kwargs: Result())

    data = collector.collect_test_coverage_metrics()
    assert data == {
        "total_lines": 100,
        "covered_lines": 90,
        "coverage_percent": 90.0,
        "missing_lines": 10,
    }, data


def test_collect_test_coverage_metrics_failure(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector(project_root=".")

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(metrics_mod.subprocess, "run", _raise)

    data = collector.collect_test_coverage_metrics()
    assert data["coverage_percent"] == 75.0


def test_collect_test_coverage_metrics_command_failure(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector(project_root=".")

    class Result:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr(metrics_mod.subprocess, "run", lambda *args, **kwargs: Result())

    data = collector.collect_test_coverage_metrics()
    assert data == collector._get_mock_coverage_data()


def test_collect_test_coverage_runs_pytest(tmp_path, metrics_mod, monkeypatch):
    coverage_json = tmp_path / "coverage_temp.json"
    coverage_json.write_text(json.dumps({"totals": {"percent_covered": 82.5}}), encoding="utf-8")

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(metrics_mod.subprocess, "run", lambda *args, **kwargs: Result())
    collector = metrics_mod.MetricsCollector(project_root=str(tmp_path))

    data = collector.collect_test_coverage()
    assert data["success"] is True
    assert data["coverage_percent"] == pytest.approx(82.5)
    assert "ok" in data["stdout"]


def test_collect_test_coverage_handles_exception(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector()

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(metrics_mod.subprocess, "run", _raise)

    data = collector.collect_test_coverage()
    assert data["success"] is False
    assert data["error"] == "boom"


def test_collect_performance_metrics_with_psutil(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector()
    fake_psutil = _make_fake_psutil()
    monkeypatch.setattr(metrics_mod, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(metrics_mod, "psutil", fake_psutil)

    data = collector.collect_performance_metrics()
    assert data["cpu_usage_percent"] == pytest.approx(42.0)
    assert data["memory_usage_mb"] > 0
    assert data["network_io"]["bytes_sent"] == 123


def test_collect_performance_metrics_force_mock(metrics_mod):
    collector = metrics_mod.MetricsCollector()
    collector._force_mock = True

    data = collector.collect_performance_metrics()
    assert data == collector._get_mock_performance_metrics()


def test_collect_performance_metrics_exception(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector()

    class BrokenPsutil:
        @staticmethod
        def virtual_memory():
            raise RuntimeError("memory error")

    monkeypatch.setattr(metrics_mod, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(metrics_mod, "psutil", BrokenPsutil)

    data = collector.collect_performance_metrics()
    assert data["error"] == "memory error"


def test_collect_resource_usage_with_psutil(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector()
    fake_psutil = _make_fake_psutil()
    monkeypatch.setattr(metrics_mod, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(metrics_mod, "psutil", fake_psutil)

    data = collector.collect_resource_usage()
    assert data["memory"]["total"] == fake_psutil.virtual_memory().total
    assert data["cpu"]["count"] == 8
    assert data["network"]["connections"] == 2


def test_collect_resource_usage_force_mock(metrics_mod):
    collector = metrics_mod.MetricsCollector()
    collector._force_mock = True

    data = collector.collect_resource_usage()
    assert data == collector._get_mock_resource_data()


def test_collect_resource_usage_exception(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector()

    class BrokenPsutil:
        @staticmethod
        def virtual_memory():
            raise RuntimeError("disk error")

    monkeypatch.setattr(metrics_mod, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(metrics_mod, "psutil", BrokenPsutil)

    data = collector.collect_resource_usage()
    assert data["error"] == "disk error"


def test_collect_health_status_without_psutil(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector()
    monkeypatch.setattr(metrics_mod, "PSUTIL_AVAILABLE", False)

    data = collector.collect_health_status()
    assert data["overall_status"] == "healthy"
    assert data["uptime_seconds"] >= 0


def test_collect_health_status_exception(metrics_mod, monkeypatch):
    collector = metrics_mod.MetricsCollector()

    class BrokenPsutil:
        @staticmethod
        def boot_time():
            raise RuntimeError("boot failure")

    monkeypatch.setattr(metrics_mod, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(metrics_mod, "psutil", BrokenPsutil)

    data = collector.collect_health_status()
    assert data["overall_status"] == "unknown"
    assert data["error"] == "boot failure"

