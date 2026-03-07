import importlib
import threading
import time
from unittest.mock import MagicMock

import pytest

interfaces_module = importlib.import_module("infrastructure.logging.core.interfaces")

if not hasattr(interfaces_module, "get_logger_pool"):
    class _DefaultPool:
        def get_stats(self):
            return {}

    interfaces_module.get_logger_pool = lambda: _DefaultPool()  # type: ignore[attr-defined]

import src.infrastructure.monitoring.application.logger_pool_monitor as lpm_module
from src.infrastructure.monitoring.application.logger_pool_monitor import (
    LoggerPoolMonitor,
    LoggerPoolStats,
)


def _build_stats(**overrides):
    data = dict(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.75,
        logger_count=6,
        total_access_count=12,
        avg_access_time=0.2,
        memory_usage_mb=42.0,
        timestamp=time.time(),
    )
    data.update(overrides)
    return LoggerPoolStats(**data)


@pytest.fixture
def fake_logger_pool(monkeypatch):
    class DummyPool:
        def __init__(self):
            self.stats = {
                "pool_size": 4,
                "max_size": 8,
                "created_count": 12,
                "hit_count": 9,
                "hit_rate": 0.6,
                "loggers": [object(), object()],
                "usage_stats": {
                    "l1": {"access_count": 3},
                    "l2": {"access_count": 5},
                },
            }

        def get_stats(self):
            return self.stats

    pool = DummyPool()
    monkeypatch.setattr(lpm_module, "get_logger_pool", lambda: pool)
    return pool


def test_init_with_components(monkeypatch, fake_logger_pool):
    monkeypatch.setattr(lpm_module, "COMPONENTS_AVAILABLE", True)

    class DummyCollector:
        def __init__(self, pool_name):
            self.pool_name = pool_name

        def collect_current_stats(self):
            return _build_stats(pool_size=1)

        def get_history_stats(self):
            return []

        def get_current_access_times(self):
            return []

        def record_access_time(self, value):
            pass

    class DummyAlertManager:
        def __init__(self, pool_name, thresholds):
            self.calls = []

        def check_alerts(self, stats):
            self.calls.append(stats)

    class DummyExporter:
        def __init__(self, pool_name):
            self.pool_name = pool_name

        def export_prometheus_metrics(self, stats):
            return "dummy_metrics"

    class DummyLoopManager:
        def __init__(self, pool_name, interval):
            self.pool_name = pool_name
            self.interval = interval
            self.initial_called = False
            self.stats = _build_stats(pool_size=2)

        def collect_initial_stats(self):
            self.initial_called = True

        def get_current_stats(self):
            return self.stats

        def get_history_stats(self):
            return [self.stats]

        def get_current_access_times(self):
            return [0.1]

        def collect_current_stats(self):
            return self.stats

        def update_access_time(self, value):
            pass

    monkeypatch.setattr(lpm_module, "LoggerPoolStatsCollector", DummyCollector)
    monkeypatch.setattr(lpm_module, "LoggerPoolAlertManager", DummyAlertManager)
    monkeypatch.setattr(lpm_module, "LoggerPoolMetricsExporter", DummyExporter)
    monkeypatch.setattr(lpm_module, "LoggerPoolMonitoringLoop", DummyLoopManager)

    monitor = LoggerPoolMonitor(pool_name="test", collection_interval=1)

    assert isinstance(monitor._monitoring_loop_manager, DummyLoopManager)
    assert monitor.current_stats.pool_size == 2
    assert monitor.history_stats
    assert monitor.access_times == [0.1]

    metrics = monitor.get_metrics_for_prometheus()
    assert metrics == "dummy_metrics"


def test_collect_stats_fallback(monkeypatch, fake_logger_pool):
    monkeypatch.setattr(lpm_module, "COMPONENTS_AVAILABLE", False)

    monitor = LoggerPoolMonitor(pool_name="fallback", collection_interval=1)
    monitor.access_times = [0.2, 0.4]

    monitor._collect_stats()

    stats = monitor.current_stats
    assert stats.pool_size == fake_logger_pool.stats["pool_size"]
    assert stats.total_access_count == 8  # 3 + 5
    assert stats.avg_access_time == pytest.approx(0.3)
    assert len(monitor.history_stats) == 2  # initial + new


def test_check_alerts_triggers(monkeypatch, fake_logger_pool):
    monitor = LoggerPoolMonitor()
    monitor.current_stats = _build_stats(
        hit_rate=0.5, pool_size=10, max_size=10, memory_usage_mb=150.0
    )

    alerts = []
    monkeypatch.setattr(
        monitor,
        "_trigger_alert",
        lambda *args, **kwargs: alerts.append((args, kwargs)),
    )

    monitor._check_alerts()

    types = {args[0][0] for args in alerts}
    assert {"hit_rate_low", "pool_usage_high", "memory_high"} <= types


def test_record_access_time_without_components(monkeypatch, fake_logger_pool):
    monkeypatch.setattr(lpm_module, "COMPONENTS_AVAILABLE", False)
    monitor = LoggerPoolMonitor(collection_interval=1)
    monitor.access_times = [1.0] * (monitor.max_access_times_size - 1)

    monitor.record_access_time(0.5)

    assert monitor.access_times[-1] == 0.5
    assert len(monitor.access_times) == monitor.max_access_times_size


def test_start_and_stop_monitoring(monkeypatch, fake_logger_pool):
    monitor = LoggerPoolMonitor(collection_interval=1)

    class DummyThread:
        def __init__(self, target, name=None, daemon=None):
            self.target = target
            self.started = False

        def start(self):
            self.started = True
            monitor.running = False  # stop loop immediately

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(threading, "Thread", DummyThread)

    monitor.start_monitoring()
    assert monitor.running is False
    assert isinstance(monitor.monitoring_thread, DummyThread)

    monitor.start_monitoring()  # no effect when already stopped
    monitor.stop_monitoring()

