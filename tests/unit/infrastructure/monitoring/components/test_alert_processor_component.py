import builtins
import importlib
import sys
from datetime import datetime, timedelta

import pytest


@pytest.fixture
def component_module():
    module_name = "src.infrastructure.monitoring.components.alert_processor"
    for name in (
        module_name,
        "infrastructure.monitoring.components.alert_processor",
    ):
        sys.modules.pop(name, None)
    module = importlib.import_module(module_name)
    yield module
    importlib.reload(module)


def _make_rule(level="critical"):
    return type(
        "Rule",
        (),
        {
            "rule_id": "rule_cpu",
            "name": "CPU 超限",
            "description": "CPU 长时间超出阈值",
            "level": level,
        },
    )()


def test_create_and_manage_alerts(component_module):
    processor = component_module.AlertProcessor()
    rule = _make_rule()

    alert = processor.create_alert(rule, {"cpu": 95}, source="unit")
    assert alert is not None
    assert alert.alert_id in processor.alerts

    fetched = processor.get_alert(alert.alert_id)
    assert fetched is alert

    all_alerts = processor.get_all_alerts()
    assert alert.alert_id in all_alerts
    assert all_alerts is not processor.alerts  # 返回副本

    active = processor.get_active_alerts()
    assert alert.alert_id in active

    acknowledged = processor.acknowledge_alert(alert.alert_id, user="tester")
    assert acknowledged is True
    assert processor.alerts[alert.alert_id].status == component_module.AlertStatus.ACKNOWLEDGED

    resolved = processor.resolve_alert(alert.alert_id)
    assert resolved is True
    assert processor.alerts[alert.alert_id].status == component_module.AlertStatus.RESOLVED

    by_status = processor.get_alerts_by_status(component_module.AlertStatus.RESOLVED)
    assert list(by_status.values())[0].status == component_module.AlertStatus.RESOLVED

    by_rule = processor.get_alerts_by_rule(rule.rule_id)
    assert alert.alert_id in by_rule


def test_create_alert_failure_returns_none(component_module, monkeypatch):
    processor = component_module.AlertProcessor()
    rule = _make_rule()

    class BrokenAlert:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cannot init")

    monkeypatch.setattr(component_module, "Alert", BrokenAlert)

    result = processor.create_alert(rule, {"cpu": 90})
    assert result is None


def test_queue_put_failure_is_handled(component_module, monkeypatch):
    processor = component_module.AlertProcessor()
    rule = _make_rule()
    alert = processor.create_alert(rule, {"cpu": 88})

    def raise_error(*args, **kwargs):
        raise RuntimeError("queue error")

    monkeypatch.setattr(processor.alert_queue, "put", raise_error)

    processor.queue_alert_for_processing(alert)  # 不应抛出异常


def test_clear_resolved_alerts_and_statistics(component_module):
    processor = component_module.AlertProcessor()
    rule = _make_rule()

    recent = processor.create_alert(rule, {"cpu": 80})
    old = processor.create_alert(rule, {"cpu": 99})
    processor.resolve_alert(old.alert_id)
    processor.resolve_alert(recent.alert_id)

    now = datetime.now()
    processor.alerts[old.alert_id].resolved_at = now - timedelta(days=2)
    processor.alerts[recent.alert_id].resolved_at = now

    cleared = processor.clear_resolved_alerts(older_than_hours=24)
    assert cleared == 1
    assert old.alert_id not in processor.alerts
    assert recent.alert_id in processor.alerts

    stats = processor.get_alerts_statistics()
    assert stats["total_alerts"] == 1
    assert stats["queue_size"] == 0
    assert stats["resolved_alerts"] == 1
    assert stats["acknowledged_alerts"] == 0


def test_validate_rule_condition(component_module):
    processor = component_module.AlertProcessor()

    assert processor.validate_rule_condition("cpu_usage > 80") is True
    assert processor.validate_rule_condition("") is False
    assert processor.validate_rule_condition(123) is False
    assert processor.validate_rule_condition("invalid syntax &&&") is False


def test_start_and_stop_processing(component_module, monkeypatch):
    processor = component_module.AlertProcessor()

    created_threads = []

    class DummyThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self.started = False

        def start(self):
            self.started = True

        def join(self, timeout=None):
            pass

    def fake_thread(target, daemon=True):
        thread = DummyThread(target, daemon)
        created_threads.append(thread)
        return thread

    monkeypatch.setattr(component_module.threading, "Thread", fake_thread)

    processor.start_processing()
    assert processor.running is True
    assert created_threads and created_threads[0].started is True
    assert processor.worker_thread is created_threads[0]

    processor.start_processing()  # 二次启动应无副作用
    assert len(created_threads) == 1

    processor.stop_processing()
    assert processor.running is False


def test_queue_processing_loop(component_module, monkeypatch):
    processor = component_module.AlertProcessor()
    rule = _make_rule()
    alert = processor.create_alert(rule, {"cpu": 88})

    processor.queue_alert_for_processing(alert)
    assert processor.alert_queue.qsize() == 1

    processed = []

    def fake_print(*args, **kwargs):
        processed.append(args)
        processor.running = False

    monkeypatch.setattr(builtins, "print", fake_print)

    processor.running = True
    processor._process_alerts()

    assert processed


def test_process_alerts_handles_queue_empty(component_module, monkeypatch):
    processor = component_module.AlertProcessor()

    def raise_empty(timeout=None):
        processor.running = False
        raise component_module.queue.Empty()

    monkeypatch.setattr(processor.alert_queue, "get", raise_empty)

    processor.running = True
    processor._process_alerts()


def test_process_alerts_handles_generic_exception(component_module, monkeypatch):
    processor = component_module.AlertProcessor()

    def raise_error(timeout=None):
        processor.running = False
        raise RuntimeError("boom")

    monkeypatch.setattr(processor.alert_queue, "get", raise_error)

    processor.running = True
    processor._process_alerts()


def test_acknowledge_and_resolve_missing_alert(component_module):
    processor = component_module.AlertProcessor()

    assert processor.acknowledge_alert("missing", user="tester") is False
    assert processor.resolve_alert("missing") is False


def test_fallback_definitions_when_service_missing(monkeypatch):
    module_name = "src.infrastructure.monitoring.components.alert_processor"
    service_mod = "src.infrastructure.monitoring.services.alert_service"
    alt_mod = "src.infrastructure.monitoring.alert_system"

    for name in (module_name, "infrastructure.monitoring.components.alert_processor", service_mod, alt_mod):
        sys.modules.pop(name, None)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(service_mod) or name.startswith(alt_mod):
            raise ImportError("forced import error")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module = importlib.import_module(module_name)

    import inspect

    rule_kwargs = dict(
        rule_id="rule_fallback",
        name="降级规则",
        description="desc",
        condition={},
        level="warning",
    )
    for optional_key, value in {
        "channels": [],
        "enabled": True,
        "cooldown": 300,
        "metadata": None,
    }.items():
        if optional_key in inspect.signature(module.AlertRule).parameters:
            rule_kwargs.setdefault(optional_key, value)

    rule = module.AlertRule(**rule_kwargs)

    alert_kwargs = dict(
        alert_id="alert_fallback",
        rule_id=rule.rule_id,
        title="title",
        message="message",
        level="warning",
        data={"value": 1},
    )
    for optional_key, value in {
        "status": module.AlertStatus.ACTIVE if hasattr(module, "AlertStatus") else "active",
        "created_at": datetime.now(),
        "source": "system",
        "resolved_at": None,
        "acknowledged_at": None,
        "acknowledged_by": None,
    }.items():
        if optional_key in inspect.signature(module.Alert).parameters:
            alert_kwargs.setdefault(optional_key, value)

    alert = module.Alert(**alert_kwargs)

    assert alert.status == module.AlertStatus.ACTIVE
    assert rule.enabled is True

    # 重新加载真实模块，避免对其他测试造成影响
    monkeypatch.setattr(builtins, "__import__", real_import)
    module = importlib.import_module(module_name)
    importlib.reload(module)

