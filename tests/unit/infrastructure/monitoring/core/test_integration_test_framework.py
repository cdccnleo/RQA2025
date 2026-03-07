import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.infrastructure.monitoring.core.integration_test_framework as itf_module


class DummyComponent:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.config_updates = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def update_config(self, cfg):
        self.config_updates.append(cfg)


class DummyInfrastructureComponentRegistry:
    def __init__(self):
        self.components = {}
        self.started = {}
        self.system_health = {"total_components": 0, "running_components": 0}

    def register_component(self, name, component_class, config=None):
        self.components[name] = component_class()
        return True

    def start_component(self, name):
        inst = self.components[name]
        inst.start()
        self.started[name] = True
        return True

    def stop_component(self, name):
        inst = self.components.get(name)
        if inst:
            inst.stop()
            self.started[name] = False
        return True

    def get_component_instance(self, name):
        return self.components.get(name)

    def get_system_health(self):
        return self.system_health


class DummyComponentBus:
    def __init__(self):
        self.subscribers = {}
        self.published = []

    def subscribe(self, event_type, handler):
        self.subscribers.setdefault(event_type, []).append(handler)

    def publish_message(self, event_type, payload):
        self.published.append((event_type, payload))
        for handler in self.subscribers.get(event_type, []):
            handler(MagicMock(type=MagicMock(value=event_type), payload=payload))


class DummyPerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def record_metric(self, name, value):
        self.metrics[name] = value

    def get_recent_metrics(self):
        return self.metrics


class StubPerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def record_metric(self, name, value):
        self.metrics[name] = value

    def get_recent_metrics(self):
        return dict(self.metrics)


class StubInfrastructureComponentRegistry:
    def __init__(self):
        self.components = {}

    def get_component(self, name):
        return self.components.get(name)

    def get_system_health(self):
        running = sum(1 for comp in self.components.values() if getattr(comp, "is_running", False))
        return {"running_components": running, "dependency_satisfaction": 0.9}


class StubIntegrationEnv:
    def __init__(self, events=None):
        self.component_registry = StubInfrastructureComponentRegistry()
        self.performance_monitor = StubPerformanceMonitor()
        self.test_data = {}
        self.events = events or {}

    def register_test_component(self, name, component_factory, config=None):
        instance = component_factory() if callable(component_factory) else component_factory
        self.component_registry.components[name] = SimpleNamespace(instance=instance, is_running=False)
        return True

    def start_component(self, name):
        component = self.component_registry.components.get(name) or SimpleNamespace()
        component.is_running = True
        self.component_registry.components[name] = component
        if name == "data_persistor" and getattr(component, "instance", None) is None:
            component.instance = object()
        return True

    def get_component_instance(self, name):
        component = self.component_registry.components.get(name)
        if component is None:
            return None
        return getattr(component, "instance", component)

    def simulate_system_load(self, cpu_usage=50.0, memory_usage=60.0):
        self.performance_monitor.record_metric("cpu_usage", cpu_usage)
        if memory_usage is not None:
            self.performance_monitor.record_metric("memory_usage", memory_usage)
        self.events.setdefault("performance.metrics.updated", []).append({"cpu": cpu_usage})

    def wait_for_event(self, event_type, timeout=10):
        if event_type == "performance.metrics.updated":
            return {"cpu": 50.0}

        queue = self.events.get(event_type)
        if not queue:
            return None
        if isinstance(queue, list):
            return queue.pop(0)
        return queue

    def inject_test_data(self, key, value):
        self.test_data[key] = value

    def get_test_data(self, key):
        return self.test_data.get(key)


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    registry = DummyInfrastructureComponentRegistry()
    bus = DummyComponentBus()
    monitor = DummyPerformanceMonitor()

    monkeypatch.setattr(itf_module, "InfrastructureComponentRegistry", lambda: registry)
    monkeypatch.setattr(itf_module, "ComponentBus", lambda: bus)
    monkeypatch.setattr(itf_module, "PerformanceMonitor", lambda: monitor)

    env = itf_module.IntegrationTestEnvironment("unit_test_env")
    env.setup()
    yield env, registry, bus, monitor
    env.teardown()


def test_setup_and_teardown(isolated_env, monkeypatch):
    env, registry, bus, monitor = isolated_env
    assert env.is_setup is True
    assert env.temp_dir.exists()

    # Ensure global singletons replaced
    import src.infrastructure.monitoring.core.component_registry as reg_module
    assert reg_module.global_component_registry is registry

    env.teardown()
    assert env.is_setup is False
    assert not env.temp_dir.exists()


def test_register_and_start_component(isolated_env):
    env, registry, _, _ = isolated_env
    assert env.register_test_component("dummy", DummyComponent) is True
    assert env.start_component("dummy") is True
    instance = env.get_component_instance("dummy")
    assert isinstance(instance, DummyComponent)
    assert instance.started is True


def test_register_test_component_failure(monkeypatch):
    env = itf_module.IntegrationTestEnvironment("register_fail")
    env.component_registry = MagicMock()
    env.component_registry.register_component.return_value = False

    result = env.register_test_component("bad", DummyComponent)

    assert result is False
    assert env.components_started == []
    env.component_registry.register_component.assert_called_once()


def test_wait_for_event(isolated_env):
    env, _, bus, _ = isolated_env

    def delayed_publish():
        import time

        time.sleep(0.1)
        bus.publish_message("test.event", {"value": 1})

    import threading

    threading.Thread(target=delayed_publish, daemon=True).start()
    received = env.wait_for_event("test.event", timeout=2)
    assert received == {"value": 1}


def test_wait_for_event_timeout(isolated_env):
    env, _, _, _ = isolated_env
    result = env.wait_for_event("missing.event", timeout=1)
    assert result is None


def test_test_data_helpers(isolated_env):
    env, _, _, _ = isolated_env
    env.inject_test_data("key", {"v": 1})
    assert env.get_test_data("key") == {"v": 1}


def test_simulate_system_load(isolated_env):
    env, _, bus, monitor = isolated_env
    env.simulate_system_load(cpu_usage=70.0, memory_usage=80.0)

    assert monitor.metrics["cpu_usage"] == 70.0
    assert monitor.metrics["memory_usage"] == 80.0
    assert ("system.load.changed", bus.published[0][1]) in bus.published


def test_environment_status(isolated_env):
    env, registry, _, monitor = isolated_env
    registry.system_health = {"total_components": 1, "running_components": 1}
    monitor.record_metric("cpu_usage", 50.0)
    env.inject_test_data("scenario", {})

    status = env.get_environment_status()
    assert status["test_name"] == "unit_test_env"
    assert status["registry_health"]["total_components"] == 1
    assert "cpu_usage" in monitor.metrics
    assert "scenario" in status["test_data_keys"]


def test_teardown_stops_components(isolated_env):
    env, registry, _, _ = isolated_env
    env.register_test_component("dummy", DummyComponent)
    env.start_component("dummy")

    env.teardown()
    assert registry.started["dummy"] is False


def test_setup_idempotent(monkeypatch):
    env = itf_module.IntegrationTestEnvironment("idempotent")
    env.setup()
    temp_dir = env.temp_dir

    env.setup()  # second call should no-op

    assert env.is_setup is True
    assert env.temp_dir == temp_dir

    env.teardown()


def test_teardown_handles_failures(monkeypatch):
    env = itf_module.IntegrationTestEnvironment("teardown_fail")
    env.setup()

    failing_registry = MagicMock()
    failing_registry.stop_component.side_effect = RuntimeError("stop boom")
    env.component_registry = failing_registry
    env.components_started = ["bad_component"]

    monkeypatch.setattr(itf_module.shutil, "rmtree", MagicMock(side_effect=OSError("remove boom")))

    env.teardown()

    failing_registry.stop_component.assert_called_once_with("bad_component")
    assert env.is_setup is False


def test_end_to_end_full_monitoring_cycle(monkeypatch):
    events = {
        "monitoring.cycle.completed": [{"stats_collected": True, "alerts_checked": True}]
    }
    env = StubIntegrationEnv(events=events)

    test_case = itf_module.EndToEndTest("test_full_monitoring_cycle")
    test_case.env = env

    test_case.test_full_monitoring_cycle()

    assert env.performance_monitor.metrics["cpu_usage"] == 85.0
    assert env.performance_monitor.metrics["memory_usage"] == 90.0


def test_end_to_end_alert_and_adaptive(monkeypatch):
    events = {
        "alert.triggered": {"severity": "high", "message": "overload"},
        "config.adaptive_change": {"component": "stats_collector", "old_value": 1, "new_value": 2},
    }
    env = StubIntegrationEnv(events=events)

    monkeypatch.setattr(
        "src.infrastructure.monitoring.components.adaptive_configurator.create_adaptive_configurator",
        lambda: SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(itf_module.time, "sleep", lambda _: None)

    alert_case = itf_module.EndToEndTest("test_alert_triggering")
    alert_case.env = env
    alert_case.test_alert_triggering()

    adaptive_case = itf_module.EndToEndTest("test_adaptive_configuration")
    adaptive_case.env = env
    adaptive_case.test_adaptive_configuration()


def test_performance_benchmark_throughput(monkeypatch):
    env = StubIntegrationEnv()
    throughput_case = itf_module.PerformanceBenchmarkTest("test_monitoring_throughput")
    throughput_case.env = env

    times = iter([0.0, 5.0])
    monkeypatch.setattr(itf_module.time, "time", lambda: next(times))

    throughput_case.test_monitoring_throughput()


def test_performance_benchmark_memory_stability(monkeypatch):
    env = StubIntegrationEnv()
    memory_case = itf_module.PerformanceBenchmarkTest("test_memory_usage_stability")
    memory_case.env = env

    class FakeMemoryInfo:
        def __init__(self, rss):
            self.rss = rss

    class FakeProcess:
        def __init__(self, rss_sequence):
            self._rss = rss_sequence

        def memory_info(self):
            return FakeMemoryInfo(self._rss.pop(0))

    rss_values = [100 * 1024 * 1024, 110 * 1024 * 1024]
    process = FakeProcess(rss_values)

    import psutil

    monkeypatch.setattr(psutil, "Process", lambda _: process)
    monkeypatch.setattr(itf_module.time, "sleep", lambda _: None)

    memory_case.test_memory_usage_stability()


def test_performance_benchmark_response_times(monkeypatch):
    env = StubIntegrationEnv()
    response_case = itf_module.PerformanceBenchmarkTest("test_component_response_times")
    response_case.env = env

    timestamps = iter([i * 0.05 for i in range(60)])
    monkeypatch.setattr(itf_module.time, "time", lambda: next(timestamps))

    response_case.test_component_response_times()


def test_create_integration_test_suite_contents():
    suite = itf_module.create_integration_test_suite()
    assert suite.countTestCases() == 6


def test_integration_test_environment_context_manager(monkeypatch):
    created = []

    class StubEnv:
        def __init__(self, name):
            self.name = name
            self.setup_called = False
            self.teardown_called = False
            created.append(self)

        def setup(self):
            self.setup_called = True

        def teardown(self):
            self.teardown_called = True

    monkeypatch.setattr(itf_module, "IntegrationTestEnvironment", StubEnv)

    with itf_module.integration_test_environment("ctx") as env:
        assert env.setup_called is True

    assert created[0].teardown_called is True

    with pytest.raises(RuntimeError):
        with itf_module.integration_test_environment("ctx2"):
            raise RuntimeError("boom")

    assert created[1].teardown_called is True


def test_run_integration_tests_summary(monkeypatch):
    fake_suite = unittest.TestSuite()
    monkeypatch.setattr(itf_module, "create_integration_test_suite", lambda: fake_suite)

    class FakeResult:
        testsRun = 3
        failures = [1]
        errors = []
        skipped = [1]

        def wasSuccessful(self):
            return True

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def run(self, suite):
            assert suite is fake_suite
            return FakeResult()

    monkeypatch.setattr(itf_module.unittest, "TextTestRunner", lambda *a, **kw: FakeRunner(*a, **kw))
    run_times = iter([0.0, 5.0])
    monkeypatch.setattr(itf_module.time, "time", lambda: next(run_times))

    results = itf_module.run_integration_tests(verbose=False)

    assert results["tests_run"] == 3
    assert results["failures"] == 1
    assert results["duration"] == 5.0
    assert results["success"] is True


def test_component_integration_assertions(monkeypatch):
    class FakeComponent:
        def __init__(self):
            self.is_running = False

    class FakeRegistry:
        def __init__(self):
            self.components = {}

        def get_component(self, name):
            return self.components.get(name)

    class FakePerformanceMonitor:
        def __init__(self):
            self.metrics = {}

        def record_metric(self, name, value):
            self.metrics[name] = value

        def get_recent_metrics(self):
            return self.metrics

    class FakeEnv:
        def __init__(self, name):
            self.name = name
            self.component_registry = FakeRegistry()
            self.performance_monitor = FakePerformanceMonitor()
            self.test_data = {}
            self.components = {}
            self.setup_called = False
            self.teardown_called = False

        def setup(self):
            self.setup_called = True

        def teardown(self):
            self.teardown_called = True

        def register_test_component(self, name, component_class, config=None):
            comp = FakeComponent()
            self.component_registry.components[name] = comp
            self.components[name] = comp
            return True

        def start_component(self, name):
            comp = self.components[name]
            comp.is_running = True
            return True

        def get_component_instance(self, name):
            return self.components.get(name)

        def inject_test_data(self, key, value):
            self.test_data[key] = value

        def get_test_data(self, key):
            return self.test_data.get(key)

    def fake_env_factory(name):
        env = FakeEnv(name)
        return env

    monkeypatch.setattr(itf_module, "IntegrationTestEnvironment", fake_env_factory)

    class SimpleTest(itf_module.ComponentIntegrationTest):
        def _register_core_components(self):
            self.env.register_test_component("simple", DummyComponent)

        def test_body(self):
            self.env.start_component("simple")
            self.assertComponentRunning("simple")
            self.env.performance_monitor.record_metric("metric", 123.0)
            self.assertMetricsRecorded("metric", 123.0)
            self.env.inject_test_data("k", 1)
            self.assertEqual(self.env.get_test_data("k"), 1)

    suite = unittest.TestSuite()
    suite.addTest(SimpleTest("test_body"))
    log_dir = Path("test_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "integration_framework_tmp.txt"
    with log_file.open("w", encoding="utf-8") as buffer:
        result = unittest.TextTestRunner(stream=buffer, verbosity=0).run(suite)
    assert result.wasSuccessful()


def test_register_core_components_invokes_all(monkeypatch):
    recorded = []

    env_stub = type("EnvStub", (), {
        "register_test_component": lambda self, name, component_class, config=None: recorded.append((name, config)) or True,
        "get_temp_file_path": lambda self, filename: Path("test_logs") / filename,
    })()

    test_instance = itf_module.ComponentIntegrationTest()
    test_instance.env = env_stub

    monkeypatch.setattr(itf_module, "MonitoringCoordinator", object)
    monkeypatch.setattr(itf_module, "StatsCollector", object)
    monkeypatch.setattr(itf_module, "AlertManager", object)
    monkeypatch.setattr(itf_module, "MetricsExporter", object)
    monkeypatch.setattr(itf_module, "DataPersistor", object)

    test_instance._register_core_components()
    names = [name for name, _ in recorded]
    assert set(names) == {
        "monitoring_coordinator",
        "stats_collector",
        "alert_manager",
        "metrics_exporter",
        "data_persistor",
    }


def test_assert_system_healthy(monkeypatch):
    class DummyEnv:
        def __init__(self):
            self.component_registry = MagicMock(
                get_system_health=lambda: {
                    "running_components": 1,
                    "dependency_satisfaction": 0.8,
                }
            )

    test_instance = itf_module.ComponentIntegrationTest()
    test_instance.env = DummyEnv()
    test_instance.assertSystemHealthy()


def test_wait_for_condition(monkeypatch):
    instance = itf_module.ComponentIntegrationTest()
    instance.env = MagicMock()

    assert instance.waitForCondition(lambda: True, timeout=0.1) is True

    with pytest.raises(AssertionError):
        instance.waitForCondition(lambda: False, timeout=0.1, message="fail")


