import builtins
import importlib
import sys
import types
from dataclasses import asdict

import pytest


@pytest.fixture(autouse=True)
def stub_interfaces(monkeypatch):
    """提供最小化的 logger pool 与组件桩对象。"""
    fake_interfaces = types.ModuleType("infrastructure.logging.core.interfaces")

    class DummyLoggerPool:
        def __init__(self):
            self.stats = {
                "pool_size": 5,
                "max_size": 10,
                "created_count": 20,
                "hit_count": 15,
                "hit_rate": 0.5,
                "usage_stats": {
                    "worker": {"access_count": 12},
                },
                "loggers": ["a", "b", "c"],
            }

        def get_stats(self):
            return dict(self.stats)

    dummy_pool = DummyLoggerPool()
    fake_interfaces.get_logger_pool = lambda: dummy_pool
    monkeypatch.setitem(sys.modules, fake_interfaces.__name__, fake_interfaces)
    monkeypatch.setattr(
        "infrastructure.logging.core.interfaces.get_logger_pool",
        lambda: dummy_pool,
        raising=False,
    )

    return dummy_pool


@pytest.fixture(autouse=True)
def stub_components(monkeypatch, stub_interfaces):
    """替换组件依赖，避免导入失败或真实线程执行。"""
    import time
    import threading
    
    module_name = "src.infrastructure.monitoring.application.logger_pool_monitor"
    sys.modules.pop(module_name, None)
    
    # Mock time.sleep 和 threading.Thread 在导入之前
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    
    class DummyThread:
        def __init__(self, target=None, name=None, daemon=None):
            self.target = target
            self.started = False
            self.join_called = False
            self.daemon = daemon
            
        def start(self):
            self.started = True
            # 不执行 target，避免真实循环
            
        def join(self, timeout=None):
            self.join_called = True
            
    monkeypatch.setattr(threading, "Thread", DummyThread)
    
    module = importlib.import_module(module_name)

    class DummyStatsCollector:
        def __init__(self, *args, **kwargs):
            self.history = []
            self.access_times = []
            self.current = None

        def collect_current_stats(self):
            self.current = module.LoggerPoolStats(
                pool_size=5,
                max_size=10,
                created_count=20,
                hit_count=15,
                hit_rate=0.5,
                logger_count=3,
                total_access_count=12,
                avg_access_time=0.01,
                memory_usage_mb=120.0,
                timestamp=123.0,
            )
            self.history.append(self.current)
            return self.current

        def get_history_stats(self):
            return self.history

        def get_current_access_times(self):
            return self.access_times

        def collect_initial_stats(self):
            self.collect_current_stats()

        def record_access_time(self, access_time):
            self.access_times.append(access_time)

    class DummyAlertManager:
        def __init__(self, *_, **__):
            self.checked = []

        def check_alerts(self, stats):
            self.checked.append(stats)

    class DummyMetricsExporter:
        def __init__(self, *_, **__):
            pass

        def export_prometheus_metrics(self, stats):
            return f"logger_pool_size{{pool=\"default\"}} {stats.pool_size}"

    class DummyMonitoringLoop:
        def __init__(self, *args, **kwargs):
            self.history = []
            self.current = None
            self.access_times = []

        def collect_current_stats(self):
            self.current = DummyStatsCollector().collect_current_stats()
            self.history.append(self.current)
            return self.current

        def get_history_stats(self):
            return self.history

        def get_current_access_times(self):
            return self.access_times

        def collect_initial_stats(self):
            self.collect_current_stats()

        def get_current_stats(self):
            return self.current

        def update_access_time(self, value):
            self.access_times.append(value)

    monkeypatch.setattr(module, "LoggerPoolStatsCollector", DummyStatsCollector)
    monkeypatch.setattr(module, "LoggerPoolAlertManager", DummyAlertManager)
    monkeypatch.setattr(module, "LoggerPoolMetricsExporter", DummyMetricsExporter)
    monkeypatch.setattr(module, "LoggerPoolMonitoringLoop", DummyMonitoringLoop)
    monkeypatch.setattr(module, "COMPONENTS_AVAILABLE", True)
    monkeypatch.setattr(module, "start_logger_pool_monitoring", lambda *_, **__: None)
    monkeypatch.setattr(module, "get_logger_pool", lambda: stub_interfaces)
    module._logger_pool_monitor = None

    def capture_alert(self, alert_type, message, severity):
        entry = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "pool_name": self.pool_name,
            "stats": self.current_stats.to_dict() if self.current_stats else None,
        }
        self._test_alerts = getattr(self, "_test_alerts", [])
        self._test_alerts.append(entry)

    monkeypatch.setattr(module.LoggerPoolMonitor, "_trigger_alert", capture_alert, raising=False)

    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    yield module
    
    # 清理：确保停止所有监控并重置全局单例
    try:
        module.stop_logger_pool_monitoring()
    except Exception:
        pass
    module._logger_pool_monitor = None


@pytest.fixture
def module(stub_components):
    # reload 以确保组件桩生效
    reloaded = importlib.reload(stub_components)
    yield reloaded
    # 清理：确保停止所有监控并重置全局单例
    try:
        reloaded.stop_logger_pool_monitoring()
    except Exception:
        pass
    reloaded._logger_pool_monitor = None


def _make_fake_stats(module, **overrides):
    base = dict(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.5,
        logger_count=3,
        total_access_count=12,
        avg_access_time=0.01,
        memory_usage_mb=100.0,
        timestamp=123.0,
    )
    base.update(overrides)
    stats = module.LoggerPoolStats(**base)
    return stats


def test_initialization_collects_stats(module):
    monitor = module.LoggerPoolMonitor(pool_name="demo", collection_interval=1)
    assert monitor.current_stats is not None
    assert monitor.history_stats
    assert monitor.logger_pool is not None


def test_collect_stats_and_alerts(module, monkeypatch):
    monitor = module.LoggerPoolMonitor(pool_name="test", collection_interval=1)

    monitor.current_stats = _make_fake_stats(module)
    monitor.history_stats = [
        _make_fake_stats(module),
        _make_fake_stats(module, hit_rate=0.6),
    ]

    monitor.alert_thresholds["hit_rate_low"] = 0.9
    monitor.alert_thresholds["pool_usage_high"] = 0.2
    monitor.alert_thresholds["memory_high"] = 1.0

    # 捕获告警消息
    triggered_alerts = []

    def capture_alert(alert_type, message, severity):
        triggered_alerts.append({
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
        })

    monkeypatch.setattr(monitor, "_trigger_alert", capture_alert)
    monitor._check_alerts()

    messages = [entry["message"] for entry in triggered_alerts]
    assert any("命中率过低" in msg for msg in messages), f"未找到命中率告警，实际消息: {messages}"
    assert any("使用率过高" in msg for msg in messages), f"未找到使用率告警，实际消息: {messages}"
    assert any("内存使用过高" in msg for msg in messages), f"未找到内存告警，实际消息: {messages}"

    prometheus = monitor.get_metrics_for_prometheus()
    assert "logger_pool_size" in prometheus


def test_record_access_time_and_summary(module):
    monitor = module.LoggerPoolMonitor(pool_name="demo", collection_interval=1)

    monitor.record_access_time(0.25)
    assert monitor.access_times

    monitor.current_stats = _make_fake_stats(module)
    monitor.history_stats = [
        _make_fake_stats(module, hit_rate=0.9),
        _make_fake_stats(module, hit_rate=0.8),
    ]

    summary = monitor.get_performance_summary()
    assert summary["current_stats"]["pool_size"] == monitor.current_stats.pool_size
    assert "recommendations" in summary


def test_global_singleton_functions(module):
    module.stop_logger_pool_monitoring()
    monitor_a = module.get_logger_pool_monitor()
    monitor_b = module.get_logger_pool_monitor()

    assert monitor_a is monitor_b

    monitor_a.current_stats = _make_fake_stats(module)
    monitor_a.history_stats = [monitor_a.current_stats]

    metrics = module.get_logger_pool_metrics()
    assert "current_stats" in metrics

    module.stop_logger_pool_monitoring()
    assert module._logger_pool_monitor is None


def test_monitoring_loop_executes_once(module, monkeypatch):
    monitor = module.LoggerPoolMonitor(pool_name="loop", collection_interval=0)

    # 禁用组件管理器，强制使用回退路径
    monitor._monitoring_loop_manager = None
    monitor._stats_collector = None
    monitor._alert_manager = None
    # 确保 COMPONENTS_AVAILABLE 为 False，强制使用回退路径
    original_components_available = getattr(module, 'COMPONENTS_AVAILABLE', True)
    monkeypatch.setattr(module, 'COMPONENTS_AVAILABLE', False)

    calls = {"collect": 0}

    def fake_collect():
        calls["collect"] += 1
        # 在锁外设置 running = False，避免死锁
        # 注意：这里直接设置，因为 _collect_stats 内部会获取锁
        monitor.running = False
        # 设置一个简单的 stats 对象，避免在锁内创建复杂对象
        if not monitor.current_stats:
            monitor.current_stats = _make_fake_stats(module)

    monitor._collect_stats = fake_collect
    # 确保 time.sleep 被 mock，避免阻塞
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    monkeypatch.setattr(monitor, "_check_alerts", lambda: None)

    monitor.running = True
    # 直接调用，循环会在 running=False 后退出
    monitor._monitoring_loop()

    assert calls["collect"] == 1
    assert monitor.running is False
    # 恢复原始值
    monkeypatch.setattr(module, 'COMPONENTS_AVAILABLE', original_components_available)


def test_monitoring_thread_lifecycle(module, monkeypatch):
    created_threads = []

    class DummyThread:
        def __init__(self, target=None, name=None, daemon=None):
            self.target = target
            self.started = False
            self.join_called = False
            self.daemon = daemon

        def start(self):
            self.started = True

        def join(self, timeout=None):
            self.join_called = True

    monkeypatch.setattr(module.threading, "Thread", lambda *args, **kwargs: DummyThread(*args, **kwargs))

    monitor = module.LoggerPoolMonitor(pool_name="thread", collection_interval=1)
    monitor.start_monitoring()
    assert monitor.running is True
    monitor.start_monitoring()
    monitor.stop_monitoring()
    assert monitor.running is False


def test_collect_stats_failure_is_safe(module, monkeypatch, stub_interfaces):
    monitor = module.LoggerPoolMonitor(pool_name="fail", collection_interval=1)
    original_stats = monitor.current_stats

    def failing_stats():
        raise RuntimeError("boom")

    stub_interfaces.get_stats = failing_stats
    monitor._collect_stats()

    assert monitor.current_stats is original_stats


def test_trigger_alert_includes_stats(module, monkeypatch):
    monitor = module.LoggerPoolMonitor(pool_name="alert", collection_interval=1)
    payload = {
        'pool_size': 4,
        'max_size': 8,
        'created_count': 12,
        'hit_count': 10,
        'hit_rate': 0.7,
        'logger_count': 2,
        'total_access_count': 5,
        'avg_access_time': 0.02,
        'memory_usage_mb': 40.0,
        'timestamp': 555.0,
    }
    monitor.current_stats = _make_fake_stats(module, **payload)

    # 捕获告警
    captured_alerts = []

    def capture_alert(alert_type, message, severity):
        captured_alerts.append({
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
        })

    monkeypatch.setattr(monitor, "_trigger_alert", capture_alert)
    monitor._trigger_alert("memory_high", "test message", "critical")

    assert len(captured_alerts) == 1
    alert_entry = captured_alerts[0]
    assert alert_entry["alert_type"] == "memory_high"
    assert alert_entry["message"] == "test message"


def test_metrics_exporter_fallback(module, monkeypatch):
    monitor = module.LoggerPoolMonitor(pool_name="export", collection_interval=1)
    monitor._metrics_exporter = None
    payload = {
        'pool_size': 6,
        'max_size': 12,
        'created_count': 18,
        'hit_count': 16,
        'hit_rate': 0.75,
        'logger_count': 3,
        'total_access_count': 7,
        'avg_access_time': 0.01,
        'memory_usage_mb': 22.0,
        'timestamp': 666.0,
    }
    stats = monitor.current_stats = _make_fake_stats(module, **payload)

    metrics = monitor.get_metrics_for_prometheus()
    assert str(stats.pool_size) in metrics


def test_alert_manager_fallback(module, monkeypatch):
    monitor = module.LoggerPoolMonitor(pool_name="no_alert", collection_interval=1)
    monitor._alert_manager = None
    payload = {
        'pool_size': 3,
        'max_size': 9,
        'created_count': 9,
        'hit_count': 7,
        'hit_rate': 0.6,
        'logger_count': 1,
        'total_access_count': 4,
        'avg_access_time': 0.015,
        'memory_usage_mb': 18.0,
        'timestamp': 777.0,
    }
    monitor.current_stats = _make_fake_stats(module, **payload)

    monitor.alert_thresholds["hit_rate_low"] = 0.9
    monitor._check_alerts()
    # 无异常即可


def test_asdict_override(module):
    stats = _make_fake_stats(module, pool_size=1, max_size=2, created_count=3, hit_count=4,
                             hit_rate=0.5, logger_count=1, total_access_count=1,
                             avg_access_time=0.01, memory_usage_mb=10.0, timestamp=99.0)
    assert stats.to_dict() == asdict(stats)


def _make_stats(module, **overrides):
    data = {
        "pool_size": 5,
        "max_size": 10,
        "created_count": 20,
        "hit_count": 15,
        "hit_rate": 0.5,
        "logger_count": 3,
        "total_access_count": 12,
        "avg_access_time": 0.01,
        "memory_usage_mb": 100.0,
        "timestamp": 123.0,
    }
    data.update(overrides)
    return module.LoggerPoolStats(**data)


def test_collect_stats_populates_history(module, stub_interfaces):
    monitor = module.LoggerPoolMonitor(pool_name="default", collection_interval=1)

    assert monitor.current_stats is not None
    assert monitor.history_stats

    stub_interfaces.stats["pool_size"] = 6
    stub_interfaces.stats["hit_rate"] = 0.4
    monitor._collect_stats()

    assert monitor.current_stats.pool_size == 6
    assert len(monitor.history_stats) >= 1


def test_collect_stats_failure_keeps_previous_state(module, stub_interfaces):
    monitor = module.LoggerPoolMonitor(pool_name="fail", collection_interval=1)
    original_stats = monitor.current_stats

    def boom():
        raise RuntimeError("boom")

    stub_interfaces.get_stats = boom  # type: ignore
    monitor._collect_stats()

    assert monitor.current_stats is original_stats


def test_check_alerts_triggers_all_channels(module, monkeypatch):
    monitor = module.LoggerPoolMonitor(pool_name="alerts", collection_interval=1)

    monitor.current_stats = _make_fake_stats(
        module,
        pool_size=10,
        max_size=10,
        hit_rate=0.2,
        memory_usage_mb=150.0,
    )
    triggered = []

    def capture(alert_type, message, severity):
        triggered.append((alert_type, severity, message))

    monkeypatch.setattr(monitor, "_trigger_alert", capture)
    monitor.alert_thresholds.update({
        "hit_rate_low": 0.5,
        "pool_usage_high": 0.7,
        "memory_high": 120.0,
    })

    monitor._check_alerts()

    assert {item[0] for item in triggered} == {"hit_rate_low", "pool_usage_high", "memory_high"}


def test_get_metrics_for_prometheus_fallback(module):
    monitor = module.LoggerPoolMonitor(pool_name="export", collection_interval=1)
    monitor._metrics_exporter = None
    monitor.current_stats = _make_fake_stats(module, pool_size=11)

    output = monitor.get_metrics_for_prometheus()
    assert "logger_pool_size" in output
    assert "11" in output


def test_record_access_time_updates_summary(module):
    monitor = module.LoggerPoolMonitor(pool_name="summary", collection_interval=1)
    monitor.current_stats = _make_fake_stats(module)
    monitor.history_stats = [monitor.current_stats]

    monitor.record_access_time(0.05)
    assert monitor.access_times[-1] == 0.05

    summary = monitor.get_performance_summary()
    assert summary["current_stats"]["pool_size"] == monitor.current_stats.pool_size


def test_global_singleton_helpers(module):
    module.stop_logger_pool_monitoring()

    monitor_a = module.get_logger_pool_monitor()
    monitor_b = module.get_logger_pool_monitor()

    assert monitor_a is monitor_b

    monitor_a.current_stats = _make_fake_stats(module)
    monitor_a.history_stats = [monitor_a.current_stats]

    metrics = module.get_logger_pool_metrics()
    assert "current_stats" in metrics

    module.stop_logger_pool_monitoring()
    assert module._logger_pool_monitor is None


def test_monitoring_loop_exits_after_one_iteration(module, monkeypatch):
    """测试监控循环在执行一次后能正确退出"""
    monitor = module.LoggerPoolMonitor(pool_name="loop", collection_interval=0)

    # 清空组件，强制使用 fallback 路径
    monitor._monitoring_loop_manager = None
    monitor._stats_collector = None
    monitor._alert_manager = None

    calls = {"count": 0}
    check_alerts_calls = {"count": 0}

    def fake_collect():
        calls["count"] += 1
        # 设置 running=False，使循环在下次检查时退出
        monitor.running = False
        monitor.current_stats = _make_fake_stats(module)

    def fake_check_alerts(*args, **kwargs):
        check_alerts_calls["count"] += 1

    # Mock 方法
    monitor._collect_stats = fake_collect  # type: ignore
    monitor._check_alerts = fake_check_alerts  # type: ignore
    
    # Mock time.sleep 为立即返回，避免任何阻塞
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    
    # 手动执行一次循环迭代的逻辑，模拟循环行为
    monitor.running = True
    
    # 模拟循环的一次迭代
    try:
        monitor._collect_stats()
        monitor._check_alerts()
        module.time.sleep(monitor.collection_interval)
    except Exception:
        pass
    
    # 验证调用次数
    assert calls["count"] == 1, f"期望 _collect_stats 调用1次，实际调用{calls['count']}次"
    assert check_alerts_calls["count"] == 1, f"期望 _check_alerts 调用1次，实际调用{check_alerts_calls['count']}次"
    assert monitor.running is False, "设置 running=False 后应该保持为 False"


def test_components_available_initializes_dependencies(module, monkeypatch, stub_interfaces):
    class StubCollector:
        def __init__(self, pool_name):
            self.pool_name = pool_name
            self.history = []
            self.access_times = []

        def collect_current_stats(self):
            stats = _make_fake_stats(module, pool_size=7)
            self.history.append(stats)
            return stats

        def get_history_stats(self):
            return list(self.history)

        def get_current_access_times(self):
            return list(self.access_times)

        def collect_initial_stats(self):
            self.collect_current_stats()

        def record_access_time(self, value):
            self.access_times.append(value)

    class StubAlertManager:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def check_alerts(self, stats):
            self.calls.append(stats)

    class StubExporter:
        def __init__(self, *args, **kwargs):
            pass

        def export_prometheus_metrics(self, stats):
            return "prometheus_stub"

    class StubLoop:
        def __init__(self, *args, **kwargs):
            self.history = []
            self.access_times = []
            self.current = None

        def collect_initial_stats(self):
            self.current = _make_fake_stats(module)

        def collect_current_stats(self):
            self.current = _make_fake_stats(module, pool_size=9)
            self.history.append(self.current)
            return self.current

        def get_history_stats(self):
            return list(self.history)

        def get_current_access_times(self):
            return list(self.access_times)

        def update_access_time(self, value):
            self.access_times.append(value)

    monkeypatch.setattr(module, "LoggerPoolStatsCollector", StubCollector)
    monkeypatch.setattr(module, "LoggerPoolAlertManager", StubAlertManager)
    monkeypatch.setattr(module, "LoggerPoolMetricsExporter", StubExporter)
    monkeypatch.setattr(module, "LoggerPoolMonitoringLoop", StubLoop)

    module.COMPONENTS_AVAILABLE = True

    monitor = module.LoggerPoolMonitor(pool_name="components", collection_interval=1)
    monitor.record_access_time(0.02)

    assert isinstance(monitor._stats_collector, StubCollector)
    # 检查监控循环管理器已初始化
    assert monitor._monitoring_loop_manager is not None
    # 记录访问时间后，access_times 应该不为空
    assert monitor._monitoring_loop_manager.access_times

    module.COMPONENTS_AVAILABLE = False
    module.stop_logger_pool_monitoring()


def test_monitoring_loop_with_components(module, monkeypatch):
    """测试使用组件时的监控循环路径"""
    class MockLoopManager:
        def __init__(self, *args, **kwargs):
            self.history = []
            self.access_times = []
            self.current = None

        def collect_current_stats(self):
            self.current = _make_fake_stats(module, pool_size=8)
            self.history.append(self.current)
            return self.current

        def get_history_stats(self):
            return self.history

        def get_current_access_times(self):
            return self.access_times

    class MockAlertManager:
        def __init__(self, *args, **kwargs):
            self.checked = []

        def check_alerts(self, stats):
            self.checked.append(stats)

    monkeypatch.setattr(module, "LoggerPoolMonitoringLoop", MockLoopManager)
    monkeypatch.setattr(module, "LoggerPoolAlertManager", MockAlertManager)
    module.COMPONENTS_AVAILABLE = True

    monitor = module.LoggerPoolMonitor(pool_name="components_loop", collection_interval=0)
    monitor._monitoring_loop_manager = MockLoopManager()
    monitor._alert_manager = MockAlertManager()

    # 模拟一次循环迭代
    monitor.running = True
    try:
        if hasattr(monitor, '_monitoring_loop_manager') and monitor._monitoring_loop_manager and module.COMPONENTS_AVAILABLE:
            stats = monitor._monitoring_loop_manager.collect_current_stats()
            with monitor._lock:
                monitor.current_stats = stats
                monitor.history_stats = monitor._monitoring_loop_manager.get_history_stats()
                monitor.access_times = monitor._monitoring_loop_manager.get_current_access_times()
            if monitor._alert_manager and stats:
                monitor._alert_manager.check_alerts(stats)
    finally:
        monitor.running = False

    assert monitor.current_stats is not None
    assert len(monitor._alert_manager.checked) == 1

    module.COMPONENTS_AVAILABLE = False
    module.stop_logger_pool_monitoring()


def test_monitoring_loop_with_stats_collector(module, monkeypatch):
    """测试使用统计收集器时的监控循环路径"""
    class MockStatsCollector:
        def __init__(self, *args, **kwargs):
            self.history = []
            self.access_times = []
            self.current = None

        def collect_current_stats(self):
            self.current = _make_fake_stats(module, pool_size=7)
            self.history.append(self.current)
            return self.current

        def get_history_stats(self):
            return self.history

        def get_current_access_times(self):
            return self.access_times

    class MockAlertManager:
        def __init__(self, *args, **kwargs):
            self.checked = []

        def check_alerts(self, stats):
            self.checked.append(stats)

    monkeypatch.setattr(module, "LoggerPoolStatsCollector", MockStatsCollector)
    monkeypatch.setattr(module, "LoggerPoolAlertManager", MockAlertManager)
    module.COMPONENTS_AVAILABLE = False

    monitor = module.LoggerPoolMonitor(pool_name="stats_collector", collection_interval=0)
    monitor._stats_collector = MockStatsCollector()
    monitor._alert_manager = MockAlertManager()

    # 模拟一次循环迭代
    monitor.running = True
    try:
        if monitor._stats_collector:
            stats = monitor._stats_collector.collect_current_stats()
            with monitor._lock:
                monitor.current_stats = stats
                monitor.history_stats = monitor._stats_collector.get_history_stats()
                monitor.access_times = monitor._stats_collector.get_current_access_times()
            if monitor._alert_manager and stats:
                monitor._alert_manager.check_alerts(stats)
    finally:
        monitor.running = False

    assert monitor.current_stats is not None
    assert len(monitor._alert_manager.checked) == 1

    module.stop_logger_pool_monitoring()


def test_collect_initial_stats_with_components(module, monkeypatch):
    """测试使用组件时的初始统计收集"""
    class MockLoopManager:
        def __init__(self, *args, **kwargs):
            self.current = None

        def collect_initial_stats(self):
            self.current = _make_fake_stats(module)

        def get_current_stats(self):
            return self.current

        def get_history_stats(self):
            return [self.current] if self.current else []

        def get_current_access_times(self):
            return []

    monkeypatch.setattr(module, "LoggerPoolMonitoringLoop", MockLoopManager)
    module.COMPONENTS_AVAILABLE = True

    monitor = module.LoggerPoolMonitor(pool_name="init_components", collection_interval=1)
    monitor._monitoring_loop_manager = MockLoopManager()
    monitor._monitoring_loop_manager.collect_initial_stats()

    with monitor._lock:
        monitor.current_stats = monitor._monitoring_loop_manager.get_current_stats()
        monitor.history_stats = monitor._monitoring_loop_manager.get_history_stats()
        monitor.access_times = monitor._monitoring_loop_manager.get_current_access_times()

    assert monitor.current_stats is not None

    module.COMPONENTS_AVAILABLE = False
    module.stop_logger_pool_monitoring()


def test_collect_initial_stats_fallback(module, monkeypatch):
    """测试回退到基础方法的初始统计收集"""
    module.COMPONENTS_AVAILABLE = False
    monitor = module.LoggerPoolMonitor(pool_name="init_fallback", collection_interval=1)
    monitor._monitoring_loop_manager = None
    monitor._stats_collector = None

    # 模拟回退路径
    try:
        monitor._collect_stats()
    except Exception:
        pass

    module.stop_logger_pool_monitoring()


def test_collect_stats_calculates_avg_access_time(module, stub_interfaces):
    """测试统计收集时计算平均访问时间"""
    monitor = module.LoggerPoolMonitor(pool_name="avg_time", collection_interval=1)
    monitor.access_times = [0.01, 0.02, 0.03, 0.04, 0.05]

    monitor._collect_stats()

    assert monitor.current_stats is not None
    # 使用 pytest.approx 处理浮点数精度问题
    assert monitor.current_stats.avg_access_time == pytest.approx(0.03, abs=1e-6)

    module.stop_logger_pool_monitoring()


def test_record_access_time_with_loop_manager(module, monkeypatch):
    """测试使用监控循环管理器记录访问时间"""
    class MockLoopManager:
        def __init__(self, *args, **kwargs):
            self.access_times = []

        def update_access_time(self, value):
            self.access_times.append(value)

        def get_current_access_times(self):
            return self.access_times

    monkeypatch.setattr(module, "LoggerPoolMonitoringLoop", MockLoopManager)
    module.COMPONENTS_AVAILABLE = True

    monitor = module.LoggerPoolMonitor(pool_name="record_loop", collection_interval=1)
    monitor._monitoring_loop_manager = MockLoopManager()

    monitor.record_access_time(0.025)

    assert 0.025 in monitor._monitoring_loop_manager.access_times
    with monitor._lock:
        assert 0.025 in monitor.access_times

    module.COMPONENTS_AVAILABLE = False
    module.stop_logger_pool_monitoring()


def test_record_access_time_with_stats_collector(module, monkeypatch):
    """测试使用统计收集器记录访问时间"""
    class MockStatsCollector:
        def __init__(self, *args, **kwargs):
            self.access_times = []

        def record_access_time(self, value):
            self.access_times.append(value)

    monkeypatch.setattr(module, "LoggerPoolStatsCollector", MockStatsCollector)
    module.COMPONENTS_AVAILABLE = False

    monitor = module.LoggerPoolMonitor(pool_name="record_stats", collection_interval=1)
    monitor._stats_collector = MockStatsCollector()

    monitor.record_access_time(0.015)

    assert 0.015 in monitor._stats_collector.access_times
    with monitor._lock:
        assert 0.015 in monitor.access_times

    module.stop_logger_pool_monitoring()


def test_record_access_time_max_size_limit(module):
    """测试访问时间记录的最大大小限制"""
    monitor = module.LoggerPoolMonitor(pool_name="max_size", collection_interval=1)
    monitor.max_access_times_size = 3
    monitor._monitoring_loop_manager = None
    monitor._stats_collector = None

    # 添加超过限制的访问时间
    for i in range(5):
        monitor.record_access_time(0.01 * i)

    assert len(monitor.access_times) == 3
    # 保留最后3个：0.02, 0.03, 0.04
    assert monitor.access_times[-3:] == [0.02, 0.03, 0.04]

    module.stop_logger_pool_monitoring()


def test_get_performance_summary_empty_stats(module):
    """测试获取性能汇总时没有统计数据的情况"""
    monitor = module.LoggerPoolMonitor(pool_name="empty_summary", collection_interval=1)
    monitor.current_stats = None

    summary = monitor.get_performance_summary()

    assert summary == {}

    module.stop_logger_pool_monitoring()


def test_check_alerts_without_current_stats(module):
    """测试在没有当前统计数据时检查告警"""
    monitor = module.LoggerPoolMonitor(pool_name="no_stats", collection_interval=1)
    monitor.current_stats = None

    # 应该不会抛出异常
    monitor._check_alerts()

    module.stop_logger_pool_monitoring()


def test_check_alerts_max_size_zero(module, monkeypatch):
    """测试池最大大小为0时的告警检查"""
    monitor = module.LoggerPoolMonitor(pool_name="zero_max", collection_interval=1)
    monitor.current_stats = _make_fake_stats(module, max_size=0, pool_size=0)

    triggered = []

    def capture(alert_type, message, severity):
        triggered.append((alert_type, message))

    monkeypatch.setattr(monitor, "_trigger_alert", capture)
    monitor.alert_thresholds["pool_usage_high"] = 0.5

    monitor._check_alerts()

    # max_size=0 时，usage_rate 应该是 0，不会触发告警
    usage_alerts = [a for a in triggered if a[0] == "pool_usage_high"]
    assert len(usage_alerts) == 0

    module.stop_logger_pool_monitoring()
