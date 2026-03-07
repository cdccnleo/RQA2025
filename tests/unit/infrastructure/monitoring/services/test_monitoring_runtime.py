import importlib
import json
import sys
import types
from pathlib import Path
from datetime import datetime

import pytest

from src.infrastructure.monitoring.services import monitoring_runtime


class _DummyThread:
    def __init__(self, target=None):
        self.target = target
        self.daemon = False
        self.started = False

    def start(self):
        self.started = True
        if self.target:
            self.target()

    def join(self, timeout=None):
        self.join_timeout = timeout


class _DummySystem:
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_config = {'interval_seconds': 1}
        self._monitoring_loop = lambda: None
        self._collect_monitoring_data = lambda: {}
        self._process_alerts = lambda data: None
        self._process_optimization_suggestions = lambda data: None
        self._persist_monitoring_results = lambda ts, data: None
        self.get_monitoring_report = lambda: {'ok': True}


@pytest.fixture
def dummy_system():
    return _DummySystem()


@pytest.fixture(autouse=True)
def _reload_runtime_module():
    """重新加载模块，确保测试隔离"""
    try:
        sys.modules[monitoring_runtime.__name__] = monitoring_runtime
        importlib.reload(monitoring_runtime)
    except (ImportError, KeyError):
        # 如果模块未导入，先导入它（并行执行时可能出现）
        if monitoring_runtime.__name__ not in sys.modules:
            importlib.import_module(monitoring_runtime.__name__)
        sys.modules[monitoring_runtime.__name__] = monitoring_runtime
        importlib.reload(monitoring_runtime)
    yield


def test_start_monitoring_starts_thread(monkeypatch, dummy_system):
    thread_holder = {}

    def fake_thread(target=None):
        thread = _DummyThread(target)
        thread_holder['thread'] = thread
        return thread

    monkeypatch.setattr(monitoring_runtime.threading, "Thread", fake_thread)

    result = monitoring_runtime.start_monitoring(dummy_system)

    assert result is True
    assert dummy_system.monitoring_active is True
    assert thread_holder['thread'].started is True


def test_start_monitoring_when_active(dummy_system):
    dummy_system.monitoring_active = True

    result = monitoring_runtime.start_monitoring(dummy_system)

    assert result is False


def test_stop_monitoring_with_thread(dummy_system):
    dummy_system.monitoring_active = True
    dummy_system.monitoring_thread = _DummyThread()

    result = monitoring_runtime.stop_monitoring(dummy_system)

    assert result is True
    assert dummy_system.monitoring_active is False
    assert getattr(dummy_system.monitoring_thread, "join_timeout", None) == 10


def test_stop_monitoring_without_thread(dummy_system):
    dummy_system.monitoring_thread = None
    dummy_system.monitoring_active = True

    assert monitoring_runtime.stop_monitoring(dummy_system) is True


def test_monitoring_loop_runs_once(monkeypatch):
    system = _DummySystem()
    calls = []

    def perform_cycle():
        calls.append("cycle")
        system.monitoring_active = False

    system.monitoring_active = True
    system._perform_monitoring_cycle = perform_cycle

    monkeypatch.setattr(monitoring_runtime, "time", types.SimpleNamespace(sleep=lambda interval: calls.append(f"sleep-{interval}")))

    monitoring_runtime.monitoring_loop(system)

    assert calls == ["cycle", "sleep-1"]


def test_monitoring_loop_handles_exception(monkeypatch):
    system = _DummySystem()
    calls = {
        'sleep': [],
    }

    def failing_cycle():
        system.monitoring_active = False
        raise RuntimeError("boom")

    system.monitoring_active = True
    system._perform_monitoring_cycle = failing_cycle

    def fake_sleep(interval):
        calls['sleep'].append(interval)

    monkeypatch.setattr(monitoring_runtime, "time", types.SimpleNamespace(sleep=fake_sleep))

    monitoring_runtime.monitoring_loop(system)

    assert calls['sleep'] == [60]


def test_perform_monitoring_cycle_invokes_subsystems(monkeypatch):
    system = _DummySystem()
    monitoring_called = {}

    def collect():
        monitoring_called['collect'] = True
        return {'data': True}

    def process_alerts(data):
        monitoring_called['alerts'] = data

    def process_opt(data):
        monitoring_called['opt'] = data

    def persist(ts, data):
        monitoring_called['persist'] = (ts, data)

    system._collect_monitoring_data = collect
    system._process_alerts = process_alerts
    system._process_optimization_suggestions = process_opt
    system._persist_monitoring_results = persist

    monitoring_runtime.perform_monitoring_cycle(system)

    assert monitoring_called['collect'] is True
    assert monitoring_called['alerts'] == {'data': True}
    assert monitoring_called['opt'] == {'data': True}
    assert monitoring_called['persist'][1] == {'data': True}


def test_collect_test_coverage_success(monkeypatch, tmp_path):
    system = _DummySystem()
    monkeypatch.chdir(tmp_path)

    coverage_result = {
        'totals': {'percent_covered': 87.5}
    }
    Path('coverage_temp.json').write_text(json.dumps(coverage_result), encoding='utf-8')

    class DummyResult:
        returncode = 0
        stdout = "{}"
        stderr = ""

    monkeypatch.setattr(monitoring_runtime.subprocess, "run", lambda *args, **kwargs: DummyResult())

    data = monitoring_runtime.collect_test_coverage(system)

    assert data['success'] is True
    assert data['coverage_percent'] == 87.5
    assert not Path('coverage_temp.json').exists()


def test_collect_test_coverage_failure(monkeypatch):
    system = _DummySystem()

    def raise_error(*args, **kwargs):
        raise RuntimeError("coverage")

    monkeypatch.setattr(monitoring_runtime.subprocess, "run", raise_error)

    data = monitoring_runtime.collect_test_coverage(system)

    assert data['success'] is False
    assert data['coverage_percent'] == 0.0
    assert data['error'] == "coverage"


def test_export_monitoring_report_default_filename(monkeypatch, tmp_path):
    system = _DummySystem()
    system.get_monitoring_report = lambda: {'status': 'ok'}

    class FakeDatetime(datetime):
        @classmethod
        def now(cls):
            return cls(2025, 1, 1, 8, 0, 0)

        def strftime(self, fmt):
            return "20250101_080000"

    monkeypatch.setattr(monitoring_runtime, "datetime", FakeDatetime)
    monkeypatch.chdir(tmp_path)

    filename = monitoring_runtime.export_monitoring_report(system)

    assert filename == "monitoring_report_20250101_080000.json"
    report_path = tmp_path / filename
    assert report_path.exists()

    content = json.loads(report_path.read_text(encoding='utf-8'))
    assert content['monitoring_system'] == {'status': 'ok'}


def test_export_monitoring_report_custom_filename(monkeypatch, tmp_path):
    system = _DummySystem()
    system.get_monitoring_report = lambda: {'status': 'ok'}

    target = tmp_path / "custom_report.json"
    monkeypatch.chdir(tmp_path)

    filename = monitoring_runtime.export_monitoring_report(system, str(target))

    assert filename == str(target)
    assert target.exists()
    assert json.loads(target.read_text(encoding='utf-8'))['monitoring_system'] == {'status': 'ok'}
