import builtins
import importlib
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from typing import List, Dict, Any

import pytest

import src.infrastructure.monitoring.services.continuous_monitoring_service as cms_module
import src.infrastructure.monitoring.services.continuous_monitoring_core as cms_core_module
from src.infrastructure.monitoring.services.continuous_monitoring_service import (
    ContinuousMonitoringSystem,
)
from src.infrastructure.monitoring.services import (
    monitoring_runtime,
    test_automation_optimizer as optimizer_module,
)


@pytest.fixture(autouse=True)
def restore_components(monkeypatch):
    """默认将可选组件替换为 None，便于针对性控制依赖。"""

    overrides: Dict[str, Any] = {}

    # 确保模块已经被导入，避免并行执行时的ImportError
    def _ensure_module(module):
        sys.modules[module.__name__] = module
        try:
            importlib.reload(module)
        except (ImportError, KeyError):
            if module.__name__ not in sys.modules:
                importlib.import_module(module.__name__)
            importlib.reload(module)

    _ensure_module(cms_module)
    _ensure_module(cms_core_module)

    from src.infrastructure.monitoring.services import optional_components

    globals()["ContinuousMonitoringSystem"] = cms_module.ContinuousMonitoringSystem
    globals()["TestAutomationOptimizer"] = optimizer_module.TestAutomationOptimizer

    original_get = optional_components.get_optional_component

    def fake_get(name: str):
        if name in overrides:
            return overrides[name]
        return original_get(name)

    fake_get.cache_clear = original_get.cache_clear  # type: ignore[attr-defined]
    fake_get.cache_info = original_get.cache_info  # type: ignore[attr-defined]

    monkeypatch.setattr(optional_components, "get_optional_component", fake_get)
    monkeypatch.setattr(cms_core_module, "get_optional_component", fake_get)
    monkeypatch.setattr(cms_module, "get_optional_component", fake_get, raising=False)
    monkeypatch.setattr(cms_core_module.psutil, "cpu_percent", lambda interval=1: 0.0)
    monkeypatch.setattr(
        cms_core_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=0.0, used=0.0),
    )
    monkeypatch.setattr(
        cms_core_module.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(percent=0.0, total=0.0, used=0.0, free=0.0),
    )
    monkeypatch.setattr(cms_core_module.psutil, "boot_time", lambda: time.time())
    monkeypatch.setattr(
        cms_core_module.psutil,
        "net_io_counters",
        lambda: SimpleNamespace(bytes_sent=0, bytes_recv=0),
    )
    monkeypatch.setattr(cms_core_module.psutil, "net_if_addrs", lambda: {"eth0": [], "lo": []})
    monkeypatch.setattr(cms_core_module.psutil, "cpu_count", lambda: 1)
    monkeypatch.setattr(cms_core_module.psutil, "cpu_freq", lambda: SimpleNamespace(current=0.0))
    monkeypatch.setattr(cms_core_module.psutil, "net_connections", lambda: [])

    return overrides


def test_collect_monitoring_data_with_metrics_collector(monkeypatch, restore_components):
    coverage = {"coverage_percent": 82.5}
    performance = {"latency_ms": 12}
    resources = {"cpu": 34}
    health = {"status": "ok"}

    class DummyCollector:
        def __init__(self, project_root):
            self.project_root = project_root

        def collect_test_coverage(self):
            return coverage

        def collect_performance_metrics(self):
            return performance

        def collect_resource_usage(self):
            return resources

        def collect_health_status(self):
            return health

    override_module_path = "src.infrastructure.monitoring.services.metrics_collector"
    monkeypatch.setitem(sys.modules, override_module_path, types.SimpleNamespace(MetricsCollector=DummyCollector))
    from src.infrastructure.monitoring.services import optional_components

    optional_components.get_optional_component.cache_clear()
    restore_components["MetricsCollector"] = DummyCollector

    cms = ContinuousMonitoringSystem(project_root=".")
    fallback = MagicMock()
    monkeypatch.setattr(cms, "_collect_test_coverage", fallback)

    result = cms._collect_monitoring_data()

    assert result["coverage"] is coverage
    assert result["performance"] is performance
    assert result["resources"] is resources
    assert result["health"] is health
    fallback.assert_not_called()


def test_collect_monitoring_data_without_metrics_collector(monkeypatch, restore_components):
    restore_components["MetricsCollector"] = None

    coverage = {"coverage_percent": 70.0}
    performance = {"response_time_ms": 5}
    resources = {"memory": {"percent": 60}}
    health = {"overall_status": "healthy"}

    monkeypatch.setattr(
        ContinuousMonitoringSystem,
        "_collect_test_coverage",
        MagicMock(return_value=coverage),
    )
    monkeypatch.setattr(
        ContinuousMonitoringSystem,
        "_collect_performance_metrics",
        MagicMock(return_value=performance),
    )
    monkeypatch.setattr(
        ContinuousMonitoringSystem,
        "_collect_resource_usage",
        MagicMock(return_value=resources),
    )
    monkeypatch.setattr(
        ContinuousMonitoringSystem,
        "_collect_health_status",
        MagicMock(return_value=health),
    )

    cms = ContinuousMonitoringSystem(project_root=".")
    result = cms._collect_monitoring_data()

    assert result == {
        "coverage": coverage,
        "performance": performance,
        "resources": resources,
        "health": health,
    }
    ContinuousMonitoringSystem._collect_test_coverage.assert_called_once()  # type: ignore[attr-defined]


def test_perform_monitoring_cycle_calls_handlers(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    monitoring_data = {
        "coverage": {},
        "performance": {},
        "resources": {},
        "health": {},
    }

    collect = MagicMock(return_value=monitoring_data)
    process_alerts = MagicMock()
    process_suggestions = MagicMock()
    persist = MagicMock()

    monkeypatch.setattr(cms, "_collect_monitoring_data", collect)
    monkeypatch.setattr(cms, "_process_alerts", process_alerts)
    monkeypatch.setattr(cms, "_process_optimization_suggestions", process_suggestions)
    monkeypatch.setattr(cms, "_persist_monitoring_results", persist)

    cms._perform_monitoring_cycle()

    collect.assert_called_once()
    process_alerts.assert_called_once_with(monitoring_data)
    process_suggestions.assert_called_once_with(monitoring_data)
    assert persist.call_count == 1
    args, _ = persist.call_args
    assert isinstance(args[0], datetime)
    assert args[1] == monitoring_data


def test_start_and_stop_monitoring(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    loop_calls = []

    def fake_loop():
        loop_calls.append(True)
        cms.monitoring_active = False

    monkeypatch.setattr(cms, "_monitoring_loop", fake_loop)

    class DummyThread:
        def __init__(self, target, *args, **kwargs):
            self.target = target
            self.daemon = False
            self.started = False

        def start(self):
            self.started = True
            self.target()

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(monitoring_runtime.threading, "Thread", DummyThread)

    assert cms.start_monitoring() is True
    assert loop_calls == [True]
    assert cms.monitoring_active is False


def test_start_monitoring_returns_false_when_active():
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.monitoring_active = True

    assert cms.start_monitoring() is False


def test_check_coverage_alerts_detects_drop():
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.test_coverage_trends = [{"coverage_percent": 90.0}]

    coverage = {"coverage_percent": 80.0}
    alerts = cms._check_coverage_alerts(coverage)

    assert alerts
    assert alerts[0]["type"] == "coverage_drop"
    assert "drop" in alerts[0]["data"]


def test_check_resource_alerts_thresholds():
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.monitoring_config["alert_thresholds"]["memory_usage_high"] = 70
    cms.monitoring_config["alert_thresholds"]["cpu_usage_high"] = 60

    resources = {
        "memory": {"percent": 75.0},
        "cpu": {"percent": 65.0},
    }
    alerts = cms._check_resource_alerts(resources)

    types = {alert["type"] for alert in alerts}
    assert {"high_memory_usage", "high_cpu_usage"} == types


def test_check_health_alerts_unhealthy_services():
    cms = ContinuousMonitoringSystem(project_root=".")
    health = {
        "services": {
            "config": {"status": "healthy"},
            "cache": {"status": "degraded"},
        }
    }

    alerts = cms._check_health_alerts(health)
    assert alerts and alerts[0]["data"]["unhealthy_services"] == ["cache"]


def test_process_alerts_appends_history(monkeypatch, capsys):
    cms = ContinuousMonitoringSystem(project_root=".")
    alerts = [
        {
            "type": "high_cpu_usage",
            "severity": "warning",
            "message": "CPU usage high",
            "timestamp": datetime.now(),
            "data": {},
        }
    ]

    cms._record_alerts(alerts)
    captured = capsys.readouterr()

    assert cms.alerts_history[-1]["type"] == "high_cpu_usage"
    assert "CPU usage high" in captured.out


def test_generate_optimization_suggestions(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.optimization_suggestions = []

    coverage = {"coverage_percent": 70.0}
    performance = {"response_time_ms": 15.0, "memory_usage_mb": 2048.0}

    cms._generate_optimization_suggestions(coverage, performance)

    suggestion_types = {s["type"] for s in cms.optimization_suggestions}
    assert {"coverage_improvement", "performance_optimization", "memory_optimization"} <= suggestion_types


def test_save_monitoring_data_trims_history(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.monitoring_config["max_history_items"] = 2
    cms.metrics_history = [
        {"timestamp": "old", "data": {"coverage": {}, "performance": {}, "health": {}}}
    ]

    persist_calls = []
    monkeypatch.setattr(cms, "_persist_monitoring_data", lambda: persist_calls.append(True))

    data = {
        "coverage": {"coverage_percent": 75.0},
        "performance": {"memory_usage_mb": 1024.0, "cpu_usage_percent": 10.0},
        "health": {"overall_status": "healthy"},
    }

    cms._save_monitoring_data(datetime.now(), data)
    cms._save_monitoring_data(datetime.now(), data)
    cms._save_monitoring_data(datetime.now(), data)

    assert len(cms.metrics_history) == 2
    assert persist_calls


def test_persist_monitoring_data_writes_json(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.metrics_history = [
        {
            "timestamp": datetime.now().isoformat(),
            "data": {
                "coverage": {"coverage_percent": 80.0},
                "performance": {"memory_usage_mb": 1024.0, "cpu_usage_percent": 20.0},
                "health": {"overall_status": "healthy"},
            },
        }
    ]
    cms.alerts_history = [{"type": "test"}]
    cms.optimization_suggestions = [{"type": "suggestion"}]

    class DummyFile:
        def __init__(self):
            self.contents = ""

        def write(self, data):
            self.contents += data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    opened = {}
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: opened.setdefault("file", DummyFile()))

    dumped = []

    def fake_dump(obj, fh, **kwargs):
        dumped.append(obj)

    monkeypatch.setattr(cms_core_module.json, "dump", fake_dump)

    cms._persist_monitoring_data()

    assert dumped
    assert dumped[0]["alerts_history"] == [{"type": "test"}]


def test_health_check_unhealthy_reports_issues(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.monitoring_active = False
    cms.metrics_history = []
    cms.alerts_history = []
    cms.optimization_suggestions = []
    cms.monitoring_thread = None

    monkeypatch.setattr(cms_core_module.psutil, "cpu_percent", lambda interval=1: 95.0)
    monkeypatch.setattr(
        cms_core_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=92.0),
    )
    monkeypatch.setattr(
        cms_core_module.psutil,
        "disk_usage",
        lambda path: SimpleNamespace(percent=93.0),
    )

    result = cms.health_check()

    assert result["healthy"] is False
    assert "issues" in result
    assert any("监控系统未激活" in issue for issue in result["issues"])


def test_health_check_error_path(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    monkeypatch.setattr(
        cms,
        "_check_monitoring_status",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    result = cms.health_check()

    assert result["status"] == "error"
    assert result["healthy"] is False


def test_collect_test_coverage_success(monkeypatch, tmp_path):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.project_root = str(tmp_path)

    coverage_json = {
        "totals": {
            "num_statements": 120,
            "num_covered": 96,
            "percent_covered": 80.0,
            "num_missing": 24,
        }
    }
    class DummyResult:
        returncode = 0
        stdout = json.dumps(coverage_json)
        stderr = ""

    def fake_run(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr(cms_core_module.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    data = cms._collect_test_coverage_metrics()
    assert data["coverage_percent"] == 80.0


def test_collect_test_coverage_failure(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    def fake_run(*args, **kwargs):
        raise RuntimeError("process failed")

    monkeypatch.setattr(cms_core_module.subprocess, "run", fake_run)

    data = cms._collect_test_coverage()
    assert data["coverage_percent"] == 0.0
    assert data["success"] is False


def test_collect_test_coverage_metrics_command_failure_returns_mock(monkeypatch, tmp_path):
    cms = ContinuousMonitoringSystem(project_root=str(tmp_path))

    class DummyResult:
        returncode = 1
        stdout = ""
        stderr = "failed"

    monkeypatch.setattr(cms_core_module.subprocess, "run", lambda *args, **kwargs: DummyResult())

    data = cms._collect_test_coverage_metrics()
    assert data == cms._get_mock_coverage_data()


def test_collect_test_coverage_metrics_handles_exception(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cms_core_module.subprocess, "run", raise_error)

    data = cms._collect_test_coverage_metrics()
    assert data == cms._get_mock_coverage_data()


def test_health_check_interface_fallback(monkeypatch):
    from src.infrastructure.monitoring.services import optional_components

    monkeypatch.setattr(optional_components, "get_optional_component", lambda name: None)

    reloaded = importlib.reload(cms_core_module)
    CMS = reloaded.ContinuousMonitoringSystem
    assert CMS.__mro__[1].__name__ == "_HealthCheckInterfaceBase"

    importlib.reload(cms_core_module)


def test_collect_performance_metrics_success(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    memory_info = SimpleNamespace(percent=55.0, used=4096.0)
    disk_info = SimpleNamespace(percent=33.0)
    net_info = SimpleNamespace(bytes_sent=1000, bytes_recv=2000)

    monkeypatch.setattr(cms_core_module.psutil, "cpu_percent", lambda interval=1: 25.0)
    monkeypatch.setattr(cms_core_module.psutil, "virtual_memory", lambda: memory_info)
    monkeypatch.setattr(cms_core_module.psutil, "disk_usage", lambda path: disk_info)
    monkeypatch.setattr(cms_core_module.psutil, "net_io_counters", lambda: net_info)

    data = cms._collect_performance_metrics()

    assert data["cpu_usage_percent"] == 25.0
    assert data["memory_usage_mb"] > 0
    assert data["disk_usage_percent"] == 33.0
    assert data["network_io"]["bytes_recv"] == 2000


def test_collect_performance_metrics_handles_exception(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    def raise_error(*args, **kwargs):
        raise RuntimeError("perf error")

    monkeypatch.setattr(cms_core_module.psutil, "virtual_memory", raise_error)

    data = cms._collect_performance_metrics()

    assert data["error"] == "perf error"
    assert data["cpu_usage_percent"] == 0.0


def test_collect_health_status_handles_psutil_error(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    def raise_error():
        raise RuntimeError("boot error")

    monkeypatch.setattr(cms_core_module.psutil, "boot_time", raise_error)

    data = cms._collect_health_status()
    assert data["overall_status"] == "unknown"
    assert data["error"] == "boot error"


def test_collect_health_status_success(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    monkeypatch.setattr(cms_core_module.psutil, "boot_time", lambda: datetime.now().timestamp() - 120.0)

    data = cms._collect_health_status()
    assert data["overall_status"] == "healthy"
    assert data["services"]["config_service"]["status"] == "healthy"
    assert data["uptime_seconds"] is not None


def test_collect_resource_usage_with_psutil(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    memory_info = SimpleNamespace(total=1024, available=256, percent=75.0, used=768)
    disk_info = SimpleNamespace(total=2048, used=1024, free=1024, percent=50.0)
    cpu_freq = SimpleNamespace(current=2400.0)

    monkeypatch.setattr(cms_core_module.psutil, "virtual_memory", lambda: memory_info)
    monkeypatch.setattr(cms_core_module.psutil, "cpu_percent", lambda interval=1: 55.0)
    monkeypatch.setattr(cms_core_module.psutil, "cpu_count", lambda: 8)
    monkeypatch.setattr(cms_core_module.psutil, "cpu_freq", lambda: cpu_freq)
    monkeypatch.setattr(cms_core_module.psutil, "disk_usage", lambda path: disk_info)
    monkeypatch.setattr(cms_core_module.psutil, "net_connections", lambda: [object(), object()])
    monkeypatch.setattr(cms_core_module.psutil, "net_if_addrs", lambda: {"eth0": [], "lo": []})

    data = cms._collect_resource_usage()

    assert data["memory"]["total"] == 1024
    assert data["cpu"]["percent"] == 55.0
    assert data["cpu"]["frequency"] == 2400.0
    assert data["disk"]["free"] == 1024
    assert data["network"]["connections"] == 2


def test_collect_resource_usage_handles_psutil_error(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    def raise_error(*args, **kwargs):
        raise RuntimeError("resource error")

    monkeypatch.setattr(cms_core_module.psutil, "virtual_memory", raise_error)

    data = cms._collect_resource_usage()
    assert data["error"] == "resource error"
    assert data["cpu"]["percent"] == 0.0
    assert data["network"]["connections"] == 0


def test_process_optimization_with_engine():
    suggestions = [{"type": "optimize"}]

    class DummyEngine:
        def __init__(self):
            self.optimization_suggestions = []
            self.called = False

        def generate_suggestions(self, coverage, performance):
            self.called = True
            self.optimization_suggestions.extend(suggestions)
            return suggestions

    cms = ContinuousMonitoringSystem(project_root=".")
    cms._optimization_engine = DummyEngine()
    cms.optimization_suggestions = []

    cms._process_optimization_suggestions(
        {"coverage": {"coverage_percent": 75.0}, "performance": {"response_time_ms": 12.0}}
    )

    assert cms._optimization_engine.called is True
    assert cms.optimization_suggestions == suggestions


def test_persist_results_with_datapersistence():
    saved = []
    persisted = []

    class DummyPersistence:
        def __init__(self):
            self.metrics_history = []

        def save_monitoring_data(self, timestamp, data):
            saved.append((timestamp, data))
            self.metrics_history.append({"timestamp": timestamp, "data": data})

        def persist_monitoring_data(self, config, alerts_history, suggestions):
            persisted.append((config, alerts_history, suggestions))

    cms = ContinuousMonitoringSystem(project_root=".")
    cms._data_persistence = DummyPersistence()
    cms.alerts_history = [{"type": "test"}]
    cms.optimization_suggestions = [{"type": "suggestion"}]

    timestamp = datetime.now()
    data = {"coverage": {}, "performance": {}, "resources": {}, "health": {}}
    cms._persist_monitoring_results(timestamp, data)

    assert saved and saved[0][1] is data
    assert persisted and persisted[0][0] is cms.monitoring_config
    assert cms.metrics_history == cms._data_persistence.metrics_history


def test_save_monitoring_data_without_datapersistence(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms._data_persistence = None
    cms.monitoring_config["max_history_items"] = 1
    cms.metrics_history = []

    persist_calls = []
    monkeypatch.setattr(cms, "_persist_monitoring_data", lambda: persist_calls.append(True))

    timestamp = datetime.now()
    data = {"coverage": {}, "performance": {}, "resources": {}, "health": {}}

    cms._persist_monitoring_results(timestamp, data)

    assert cms.metrics_history and cms.metrics_history[-1]["data"] == data
    assert persist_calls == [True]

def test_monitoring_loop_handles_exception(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.monitoring_active = True

    call_count = {"cycles": 0, "sleeps": []}

    def failing_cycle():
        call_count["cycles"] += 1
        raise RuntimeError("boom")

    def fake_sleep(seconds):
        call_count["sleeps"].append(seconds)
        cms.monitoring_active = False

    monkeypatch.setattr(cms, "_perform_monitoring_cycle", failing_cycle)
    monkeypatch.setattr(monitoring_runtime.time, "sleep", fake_sleep)

    cms._monitoring_loop()

    assert call_count["cycles"] == 1
    assert call_count["sleeps"] == [60]
    assert cms.monitoring_active is False


def test_monitoring_loop_runs_single_cycle(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.monitoring_active = True
    cms.monitoring_config["interval_seconds"] = 0.5

    cycles = []

    def successful_cycle():
        cycles.append("called")
        cms.monitoring_active = False

    sleeps = []

    def fake_sleep(interval):
        sleeps.append(interval)
        # monitoring_active already set to False; no action needed

    monkeypatch.setattr(cms, "_perform_monitoring_cycle", successful_cycle)
    monkeypatch.setattr(monitoring_runtime.time, "sleep", fake_sleep)

    cms._monitoring_loop()

    assert cycles == ["called"]
    assert sleeps == [0.5]
    assert cms.monitoring_active is False


def test_export_monitoring_report(tmp_path):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.metrics_history = [
        {"timestamp": datetime.now().isoformat(), "data": {"coverage": {}, "performance": {}, "health": {}}}
    ]
    cms.alerts_history = [{"type": "test"}]
    cms.optimization_suggestions = [{"type": "suggestion"}]

    target = tmp_path / "report.json"
    cms.export_monitoring_report(str(target))

    assert target.exists()
    content = json.loads(target.read_text(encoding="utf-8"))
    assert content["monitoring_system"]["total_metrics_collected"] == 1


def test_init_components_with_dependencies(monkeypatch, restore_components):
    created = {}

    class DummyCollector:
        def __init__(self, project_root):
            created["collector"] = project_root

    class DummyAlertManager:
        def __init__(self, thresholds):
            created["alert"] = thresholds

    class DummyDataPersistence:
        def __init__(self, max_items):
            created["data"] = max_items
            self.metrics_history = []

        def save_monitoring_data(self, timestamp, data):
            self.metrics_history.append({"timestamp": timestamp, "data": data})

        def persist_monitoring_data(self, config, alerts_history, suggestions):
            created["persist"] = (config, alerts_history, suggestions)

    class DummyOptimizationEngine:
        def __init__(self):
            created["optimization_init"] = True
            self.optimization_suggestions = []

        def generate_suggestions(self, coverage, performance):
            return []

    restore_components.update(
        {
            "MetricsCollector": DummyCollector,
            "AlertManager": DummyAlertManager,
            "DataPersistence": DummyDataPersistence,
            "OptimizationEngine": DummyOptimizationEngine,
        }
    )

    cms = ContinuousMonitoringSystem(project_root="/tmp/project")

    assert isinstance(cms._metrics_collector, DummyCollector)
    assert cms._alert_manager is not None
    assert isinstance(cms._data_persistence, DummyDataPersistence)
    assert isinstance(cms._optimization_engine, DummyOptimizationEngine)
    assert created["collector"] == "/tmp/project"
    assert created["alert"] == cms.monitoring_config["alert_thresholds"]
    assert created["data"] == cms.monitoring_config["max_history_items"]


def test_init_components_without_dependencies(restore_components):
    restore_components.update(
        {
            "MetricsCollector": None,
            "AlertManager": None,
            "DataPersistence": None,
            "OptimizationEngine": None,
        }
    )

    cms = ContinuousMonitoringSystem(project_root=".")

    assert cms._metrics_collector is None
    assert cms._alert_manager is None
    assert cms._data_persistence is None
    assert cms._optimization_engine is None


def test_collect_system_metrics_success(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    monkeypatch.setattr(cms_core_module.psutil, "cpu_percent", lambda interval=1: 33.3)
    monkeypatch.setattr(cms_core_module.psutil, "virtual_memory", lambda: SimpleNamespace(percent=44.4))
    monkeypatch.setattr(cms_core_module.psutil, "disk_usage", lambda path: SimpleNamespace(percent=55.5))
    monkeypatch.setattr(cms_core_module.psutil, "net_connections", lambda: [object(), object(), object()])

    data = cms._collect_system_metrics()

    assert data["cpu_percent"] == 33.3
    assert data["memory_percent"] == 44.4
    assert data["disk_usage"] == 55.5
    assert data["network_connections"] == 3


def test_collect_system_metrics_failure(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    def raise_error(*args, **kwargs):
        raise ImportError("psutil missing")

    monkeypatch.setattr(cms_core_module.psutil, "cpu_percent", raise_error)

    data = cms._collect_system_metrics()
    assert data["network_connections"] == 10
    assert data["cpu_percent"] == 45.5


def test_collect_test_coverage_metrics_success(monkeypatch, tmp_path):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.project_root = str(tmp_path)

    totals = {
        "num_statements": 200,
        "num_covered": 150,
        "percent_covered": 75.0,
        "num_missing": 50,
    }

    class DummyResult:
        returncode = 0
        stdout = json.dumps({"totals": totals})
        stderr = ""

    monkeypatch.setattr(cms_core_module.subprocess, "run", lambda *args, **kwargs: DummyResult())

    data = cms._collect_test_coverage_metrics()
    assert data["coverage_percent"] == 75.0
    assert data["missing_lines"] == 50


def test_process_alerts_without_alerts_prints_ok(capsys):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms._record_alerts([])
    captured = capsys.readouterr()
    assert "无告警" in captured.out


def test_analyze_and_alert_uses_internal_checks(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    coverage = {"coverage_percent": 70.0}
    resources = {"memory": {"percent": 90.0}, "cpu": {"percent": 95.0}}
    health = {"services": {"api": {"status": "degraded"}}}

    cms.test_coverage_trends = [{"coverage_percent": 90.0}]

    processed = []
    monkeypatch.setattr(cms, "_record_alerts", lambda alerts: processed.extend(alerts))

    cms._analyze_and_alert(coverage, {}, resources, health)

    alert_types = {alert["type"] for alert in processed}
    assert {"coverage_drop", "high_memory_usage", "high_cpu_usage", "service_unhealthy"} <= alert_types


def test_health_check_success(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.monitoring_active = True
    cms.metrics_history = [{"timestamp": datetime.now().isoformat(), "data": {}}]
    cms.alerts_history = []
    cms.optimization_suggestions = []

    class AliveThread:
        def is_alive(self):
            return True

    cms.monitoring_thread = AliveThread()

    monkeypatch.setattr(cms_core_module.psutil, "cpu_percent", lambda interval=1: 10.0)
    monkeypatch.setattr(cms_core_module.psutil, "virtual_memory", lambda: SimpleNamespace(percent=20.0))
    monkeypatch.setattr(cms_core_module.psutil, "disk_usage", lambda path: SimpleNamespace(percent=30.0))

    result = cms.health_check()

    assert result["healthy"] is True
    assert "issues" not in result


def test_create_error_health_result():
    cms = ContinuousMonitoringSystem(project_root=".")
    error = RuntimeError("boom")
    result = cms._create_error_health_result(error)

    assert result["status"] == "error"
    assert result["healthy"] is False
    assert result["error"] == "boom"


def test_export_monitoring_report_default_name(monkeypatch, tmp_path):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.metrics_history = []
    cms.alerts_history = []
    cms.optimization_suggestions = []

    class FakeDatetime(datetime):
        @classmethod
        def now(cls):
            return cls(2025, 1, 1, 12, 34, 56)

        def strftime(self, fmt):
            return "2025_01_01_123456"

    monkeypatch.setattr(cms_core_module, "datetime", FakeDatetime)
    monkeypatch.chdir(tmp_path)

    filename = cms.export_monitoring_report()

    assert Path(filename).exists()
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    assert data["report_title"].startswith("RQA2025")


def test_process_optimization_without_engine(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms._optimization_engine = None
    calls = {}

    def fake_generate(coverage, performance):
        calls["coverage"] = coverage
        calls["performance"] = performance

    monkeypatch.setattr(cms, "_generate_optimization_suggestions", fake_generate)

    monitoring_data = {
        "coverage": {"coverage_percent": 70.0},
        "performance": {"response_time_ms": 11.0},
    }

    cms._process_optimization_suggestions(monitoring_data)

    assert calls["coverage"] == monitoring_data["coverage"]
    assert calls["performance"] == monitoring_data["performance"]


def test_persist_monitoring_data_handles_exception(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.metrics_history = [
        {
            "timestamp": datetime.now().isoformat(),
            "data": {
                "coverage": {"coverage_percent": 85.0},
                "performance": {"memory_usage_mb": 512.0, "cpu_usage_percent": 10.0},
                "health": {"overall_status": "healthy"},
            },
        }
    ]

    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: (_ for _ in ()).throw(IOError("disk full")))

    cms._persist_monitoring_data()
    # 若抛出异常将导致测试失败；只需确认不会异常退出


def test_get_monitoring_report_fields():
    cms = ContinuousMonitoringSystem(project_root=".")
    cms.monitoring_active = True
    cms.metrics_history = [
        {"timestamp": datetime.now().isoformat(), "data": {"coverage": {}, "performance": {}, "health": {}}}
    ]
    cms.alerts_history = [{"type": "example"}]
    cms.optimization_suggestions = [{"type": "suggestion"}]

    report = cms.get_monitoring_report()

    assert report["monitoring_active"] is True
    assert report["total_alerts_generated"] == 1
    assert report["latest_suggestions"]


def test_test_automation_optimizer_optimize(monkeypatch):
    monkeypatch.setattr(optimizer_module.os, "cpu_count", lambda: 8)
    optimizer = optimizer_module.TestAutomationOptimizer()

    result = optimizer.optimize_test_execution()

    assert "parallel_execution" in result
    assert result["parallel_execution"]["max_workers"] == 4
    assert result["cache_strategy"]["strategy"] == "intelligent_caching"
    assert result["fixture_management"]["estimated_improvement"] == "30%"


def test_process_optimization_with_engine():
    cms = ContinuousMonitoringSystem(project_root=".")

    class DummyEngine:
        def __init__(self):
            self.optimization_suggestions = []

        def generate_suggestions(self, coverage, performance):
            suggestion = {"type": "engine", "coverage": coverage, "performance": performance}
            self.optimization_suggestions.append(suggestion)
            return self.optimization_suggestions

    engine = DummyEngine()
    cms._optimization_engine = engine

    monitoring_data = {
        "coverage": {"coverage_percent": 85.0},
        "performance": {"response_time_ms": 8.0},
    }

    cms._process_optimization_suggestions(monitoring_data)

    assert cms.optimization_suggestions == engine.optimization_suggestions


def test_process_alerts_with_manager(capsys):
    cms = ContinuousMonitoringSystem(project_root=".")

    class DummyManager:
        def __init__(self):
            self.alerts_history: List[Dict[str, Any]] = []
            self.test_coverage_trends: List[Dict[str, Any]] = []

        def analyze_and_alert(self, coverage, performance, resources, health):
            alert = {
                "type": "manager_alert",
                "severity": "info",
                "message": "manager active",
                "timestamp": datetime.now(),
                "data": {},
            }
            self.alerts_history.append(alert)
            return [alert]

        def update_coverage_trends(self, coverage):
            self.test_coverage_trends.append(coverage)

    manager = DummyManager()
    cms._alert_manager = manager

    monitoring_data = {
        "coverage": {"coverage_percent": 75.0},
        "performance": {},
        "resources": {},
        "health": {},
    }

    cms._process_alerts(monitoring_data)
    captured = capsys.readouterr()

    assert cms.alerts_history
    assert "INFO" in captured.out
    assert manager.test_coverage_trends


def test_process_alerts_without_alert_manager(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")
    cms._alert_manager = None

    expected_alerts = [{"type": "fallback"}]
    record_mock = MagicMock()

    monkeypatch.setattr(cms, "_analyze_and_alert", MagicMock(return_value=expected_alerts))
    monkeypatch.setattr(cms, "_record_alerts", record_mock)

    monitoring_data = {
        "coverage": {},
        "performance": {},
        "resources": {},
        "health": {},
    }

    alerts = cms._process_alerts(monitoring_data)

    assert alerts == expected_alerts
    record_mock.assert_called_once_with(expected_alerts)


def test_optional_import_returns_none_for_invalid_path():
    from src.infrastructure.monitoring.services import optional_components

    assert optional_components._optional_import("invalid.module.Class") is None


def test_get_optional_component_returns_real_class():
    from src.infrastructure.monitoring.services import optional_components

    component_cls = optional_components.get_optional_component("MetricsCollector")
    assert component_cls is not None


def test_service_properties():
    cms = ContinuousMonitoringSystem(project_root=".")
    assert cms.service_name == "continuous_monitoring_system"
    assert cms.service_version == "2.0.0"


def test_test_automation_optimizer_execution(monkeypatch):
    monkeypatch.setattr(optimizer_module.os, "cpu_count", lambda: 8)
    optimizer = optimizer_module.TestAutomationOptimizer()

    result = optimizer.optimize_test_execution()

    assert set(result.keys()) == {
        "parallel_execution",
        "selective_testing",
        "cache_strategy",
        "fixture_management",
    }
    assert result["parallel_execution"]["max_workers"] == 4
    assert result["selective_testing"]["strategy"] == "impact_based_selection"
    assert result["cache_strategy"]["cache_levels"] == ["memory", "disk", "distributed"]
    assert result["fixture_management"]["cleanup_policy"] == "automatic"


def test_test_automation_optimizer_parallel_minimum(monkeypatch):
    monkeypatch.setattr(optimizer_module.os, "cpu_count", lambda: None)
    optimizer = optimizer_module.TestAutomationOptimizer()

    data = optimizer._optimize_parallel_execution()

    assert data["max_workers"] == 2
    assert data["strategy"] == "dynamic_worker_scaling"


def test_stop_monitoring_delegates(monkeypatch):
    cms = ContinuousMonitoringSystem(project_root=".")

    called = {}

    def fake_stop(system):
        called["done"] = True
        return "stopped"

    monkeypatch.setattr(cms_core_module, "runtime_stop_monitoring", fake_stop)

    result = cms.stop_monitoring()
    assert result == "stopped"
    assert called.get("done") is True

