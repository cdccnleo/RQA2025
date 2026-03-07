import importlib
from datetime import datetime

import pytest


class _FakeStatsCollector:
    def __init__(self, pool_name, config):
        self.pool_name = pool_name
        self.config = config
        self.collected = []
        self.history = []
        self.current = None
        self.access_times = [10, 20, 30]
        self.next_stats = {"requests": 5}
        self.analyze_calls = []

    def collect_stats(self):
        self.collected.append("collect")
        result = self.next_stats
        if result is not None:
            self.current = result
            self.history.append(result)
        return result

    def get_history_stats(self, limit=None):
        if limit is None:
            return list(self.history)
        return self.history[:limit]

    def get_current_stats(self):
        return self.current

    def analyze_trends(self, metric_name, window_size):
        self.analyze_calls.append((metric_name, window_size))
        return {"metric": metric_name, "window": window_size}

    def calculate_percentiles(self, data, percentiles):
        return {str(p): data[0] if data else 0 for p in percentiles}

    def get_access_times(self):
        return self.access_times


class _FakeAlertManager:
    def __init__(self, pool_name, thresholds):
        self.pool_name = pool_name
        self.thresholds = thresholds
        self.history = [{"id": "a1"}]
        self.active = [{"id": "active"}]
        self.statistics = {"total": 1}
        self.rules = {}
        self.acknowledged = []

    def get_alert_history(self, limit):
        return self.history[:limit]

    def get_active_alerts(self):
        return list(self.active)

    def acknowledge_alert(self, alert_id):
        self.acknowledged.append(alert_id)
        return True

    def get_alert_statistics(self):
        return dict(self.statistics)

    def add_alert_rule(self, rule):
        self.rules[rule.rule_id] = rule

    def remove_alert_rule(self, rule_id):
        existed = rule_id in self.rules
        self.rules.pop(rule_id, None)
        return existed


class _FakeMetricsExporter:
    def __init__(self, pool_name, config):
        self.pool_name = pool_name
        self.config = config
        self.export_calls = []

    def get_prometheus_metrics(self):
        return "# TYPE logger_pool gauge"

    def get_json_metrics(self):
        return '{"metrics": 1}'

    def get_export_status(self):
        return {"status": "ok"}

    def export_to_file(self, format_type, file_path):
        self.export_calls.append((format_type, file_path))
        return True


class _FakeDataPersistor:
    def __init__(self, pool_name, config):
        self.pool_name = pool_name
        self.config = config
        self.persisted = []
        self.cleanup_calls = []

    def persist_data(self, stats):
        self.persisted.append(stats)

    def retrieve_data(self, start_time, end_time, limit):
        return [{"ts": start_time, "limit": limit}]

    def get_data_statistics(self):
        return {"records": len(self.persisted)}

    def export_data(self, file_path, format_type):
        return True

    def cleanup_old_data(self, days_to_keep):
        self.cleanup_calls.append(days_to_keep)
        return 3


class _FakeMonitoringCoordinator:
    def __init__(self, pool_name, config):
        self.pool_name = pool_name
        self.config = config
        self.started = False
        self.components = None

    def set_components(self, stats_collector, alert_manager, metrics_exporter):
        self.components = (stats_collector, alert_manager, metrics_exporter)

    def start_monitoring(self):
        self.started = True
        return True

    def stop_monitoring(self):
        was_started = self.started
        self.started = False
        return was_started

    def get_monitoring_status(self):
        return {"active": self.started}


@pytest.fixture
def monitor(monkeypatch):
    module = importlib.import_module(
        "src.infrastructure.monitoring.application.logger_pool_monitor_refactored"
    )

    monkeypatch.setattr(module, "StatsCollector", _FakeStatsCollector)
    monkeypatch.setattr(module, "AlertManager", _FakeAlertManager)
    monkeypatch.setattr(module, "MetricsExporter", _FakeMetricsExporter)
    monkeypatch.setattr(module, "DataPersistor", _FakeDataPersistor)
    monkeypatch.setattr(module, "MonitoringCoordinator", _FakeMonitoringCoordinator)

    instance = module.LoggerPoolMonitorRefactored(pool_name="pool-A")
    return instance, module


def test_initialization_creates_components(monitor):
    instance, module = monitor

    assert isinstance(instance.stats_collector, _FakeStatsCollector)
    assert isinstance(instance.alert_manager, _FakeAlertManager)
    assert isinstance(instance.metrics_exporter, _FakeMetricsExporter)
    assert isinstance(instance.data_persistor, _FakeDataPersistor)
    assert isinstance(instance.monitoring_coordinator, _FakeMonitoringCoordinator)
    assert instance.monitoring_coordinator.components == (
        instance.stats_collector,
        instance.alert_manager,
        instance.metrics_exporter,
    )


def test_start_stop_monitoring_and_status(monitor):
    instance, _ = monitor

    assert instance.start_monitoring() is True
    assert instance.get_monitoring_status() == {"active": True}
    assert instance.stop_monitoring() is True
    assert instance.get_monitoring_status() == {"active": False}


def test_collect_current_stats_persists_and_updates_state(monitor):
    instance, _ = monitor
    instance.stats_collector.next_stats = {"requests": 42}

    stats = instance.collect_current_stats()

    assert stats == {"requests": 42}
    assert instance.data_persistor.persisted == [{"requests": 42}]
    assert instance.current_stats == {"requests": 42}
    assert instance.history_stats == [{"requests": 42}]


def test_collect_current_stats_handles_none(monitor):
    instance, _ = monitor
    instance.stats_collector.next_stats = None

    stats = instance.collect_current_stats()

    assert stats is None
    assert instance.data_persistor.persisted == []
    assert instance.current_stats is None


def test_metric_and_alert_accessors(monitor):
    instance, _ = monitor
    instance.stats_collector.collect_stats()  # populate history

    assert instance.get_current_stats() == {"requests": 5}
    assert instance.get_history_stats(limit=1) == [{"requests": 5}]
    assert instance.get_metrics_for_prometheus().startswith("# TYPE")
    assert instance.get_metrics_for_json().startswith("{")
    assert instance.get_alert_history(limit=10) == [{"id": "a1"}]
    assert instance.get_active_alerts() == [{"id": "active"}]
    assert instance.acknowledge_alert("alert-1") is True
    assert "alert-1" in instance.alert_manager.acknowledged


def test_data_persistence_methods(monitor):
    instance, _ = monitor

    assert instance.retrieve_historical_data(None, None, 5) == [{"ts": None, "limit": 5}]
    assert instance.get_data_statistics() == {"records": 0}
    assert instance.export_data("path.json") is True
    assert instance.cleanup_old_data(days_to_keep=7) == 3
    assert instance.data_persistor.cleanup_calls == [7]


def test_alert_rule_management(monitor):
    instance, _ = monitor
    rule = type("Rule", (), {"rule_id": "rule-1"})()

    instance.add_custom_alert_rule(rule)
    assert "rule-1" in instance.alert_manager.rules
    assert instance.remove_alert_rule("rule-1") is True


def test_metrics_export_status_and_file(monitor):
    instance, _ = monitor

    assert instance.get_export_status() == {"status": "ok"}
    assert instance.export_metrics_to_file(format_type="json", file_path="/tmp/out.json") is True
    assert instance.metrics_exporter.export_calls == [("json", "/tmp/out.json")]


def test_analyze_performance_trends(monitor):
    instance, _ = monitor

    result = instance.analyze_performance_trends("latency", window_size=5)

    assert result == {"metric": "latency", "window": 5}
    assert instance.stats_collector.analyze_calls == [("latency", 5)]


def test_get_performance_summary_with_and_without_data(monkeypatch, monitor):
    instance, module = monitor

    # No data case
    instance.stats_collector.current = None
    assert instance.get_performance_summary() == {}

    # Populate current stats and dependencies
    instance.stats_collector.current = {"qps": 100}
    instance.alert_manager.statistics = {"total": 2}
    instance.data_persistor.persisted = [{"requests": 1}, {"requests": 2}]
    instance.stats_collector.access_times = [11, 22, 33]

    class _FixedDateTime:
        @staticmethod
        def now():
            return datetime(2025, 1, 1, 8, 0, 0)

    monkeypatch.setattr(module, "datetime", _FixedDateTime)

    summary = instance.get_performance_summary()

    assert summary["current_stats"] == {"qps": 100}
    assert summary["alert_statistics"] == {"total": 2}
    assert summary["data_statistics"] == {"records": 2}
    assert summary["access_time_percentiles"] == {"50.0": 11, "95.0": 11, "99.0": 11}
    assert summary["generated_at"] == "2025-01-01T08:00:00"


def test_context_manager_starts_and_stops(monitor):
    instance, _ = monitor

    with instance as ctx:
        assert ctx.monitoring_coordinator.started is True
    assert instance.monitoring_coordinator.started is False

