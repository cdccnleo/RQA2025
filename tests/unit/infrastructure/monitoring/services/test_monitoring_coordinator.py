import threading
import time
from unittest.mock import Mock, patch

from src.infrastructure.monitoring.services.monitoring_coordinator import MonitoringCoordinator


REAL_THREAD = threading.Thread


def test_start_and_stop_monitoring():
    coordinator = MonitoringCoordinator({'interval_seconds': 0.01})

    loop_started = threading.Event()

    def fake_loop():
        loop_started.set()
        while coordinator.monitoring_active:
            time.sleep(0.005)

    with patch('threading.Thread') as thread_factory, patch.object(coordinator, "_monitoring_loop", side_effect=fake_loop):
        def _spawn_thread(*args, **kwargs):
            thread = REAL_THREAD(target=fake_loop)
            thread.daemon = True
            return thread

        thread_factory.side_effect = _spawn_thread

        assert coordinator.start_monitoring() is True
        assert coordinator.monitoring_active is True
        loop_started.wait(timeout=1.0)
        assert loop_started.is_set() is True

        assert coordinator.start_monitoring() is False

        assert coordinator.stop_monitoring() is True
        assert coordinator.monitoring_active is False
        assert coordinator.stop_monitoring() is False


def test_perform_monitoring_cycle_with_components():
    coordinator = MonitoringCoordinator()

    metrics_collector = Mock()
    metrics_collector.collect_all_metrics.return_value = {'cpu': 10}

    alert_processor = Mock()
    alert_processor.process_alerts.return_value = [{'id': 'alert-1'}]

    optimization_suggester = Mock()
    optimization_suggester.generate_suggestions.return_value = [{'id': 'suggest-1'}]

    data_manager = Mock()

    coordinator.set_components(
        metrics_collector=metrics_collector,
        alert_processor=alert_processor,
        optimization_suggester=optimization_suggester,
        data_manager=data_manager,
    )

    coordinator.reset_stats()

    coordinator._perform_monitoring_cycle()

    metrics_collector.collect_all_metrics.assert_called_once()
    alert_processor.process_alerts.assert_called_once()
    optimization_suggester.generate_suggestions.assert_called_once()
    data_manager.save_monitoring_data.assert_called_once()

    saved_payload = data_manager.save_monitoring_data.call_args.args[0]
    assert saved_payload['metrics'] == {'cpu': 10}
    assert saved_payload['alerts'] == [{'id': 'alert-1'}]
    assert saved_payload['suggestions'] == [{'id': 'suggest-1'}]
    assert saved_payload['cycle_stats']['alerts_generated'] == 1
    assert saved_payload['cycle_stats']['suggestions_generated'] == 1


def test_perform_monitoring_cycle_without_components():
    coordinator = MonitoringCoordinator()
    coordinator.reset_stats()

    coordinator._perform_monitoring_cycle()

    assert coordinator.monitoring_stats['alerts_generated'] == 0
    assert coordinator.monitoring_stats['suggestions_generated'] == 0
    assert coordinator.monitoring_stats['errors_encountered'] == 0


def test_update_config_and_health_status():
    coordinator = MonitoringCoordinator()

    coordinator.update_config({'interval_seconds': 10, 'alert_thresholds': {'cpu_usage_high': 55}})

    assert coordinator.config['interval_seconds'] == 10
    assert coordinator.config['alert_thresholds']['cpu_usage_high'] == 55

    health = coordinator.get_health_status()
    assert '监控系统未运行' in health['issues']
    assert any('缺少组件' in issue for issue in health['issues'])
    assert health['status'] == 'warning'
    assert health['health_score'] == 60


def test_force_monitoring_cycle_success():
    coordinator = MonitoringCoordinator()

    metrics_collector = Mock()
    metrics_collector.collect_all_metrics.return_value = {}

    coordinator.set_components(metrics_collector=metrics_collector)
    coordinator.reset_stats()

    result = coordinator.force_monitoring_cycle()

    assert result['success'] is True
    assert coordinator.monitoring_stats['errors_encountered'] == 0


def test_force_monitoring_cycle_failure():
    coordinator = MonitoringCoordinator()

    with patch.object(coordinator, "_perform_monitoring_cycle", side_effect=RuntimeError("boom")):
        result = coordinator.force_monitoring_cycle()

    assert result['success'] is False
    assert 'boom' in result['error']


def test_start_monitoring_handles_exception(monkeypatch):
    coordinator = MonitoringCoordinator()

    def raise_error(*args, **kwargs):
        raise RuntimeError("thread failure")

    monkeypatch.setattr('src.infrastructure.monitoring.services.monitoring_coordinator.threading.Thread', raise_error)

    assert coordinator.start_monitoring() is False
    assert coordinator.monitoring_active is False


def test_stop_monitoring_handles_exception():
    coordinator = MonitoringCoordinator()
    coordinator.monitoring_active = True

    class BadThread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            raise RuntimeError("join error")

    coordinator.monitoring_thread = BadThread()

    assert coordinator.stop_monitoring() is False
    assert coordinator.monitoring_active is False


def test_monitoring_loop_error_path(monkeypatch):
    coordinator = MonitoringCoordinator({'interval_seconds': 0.01})

    calls = {'count': 0}

    def failing_cycle():
        calls['count'] += 1
        raise RuntimeError("cycle boom")

    coordinator._perform_monitoring_cycle = failing_cycle  # type: ignore[attr-defined]
    coordinator.monitoring_active = True

    def fake_sleep(interval):
        coordinator.monitoring_active = False

    monkeypatch.setattr('src.infrastructure.monitoring.services.monitoring_coordinator.time.sleep', fake_sleep)

    coordinator._monitoring_loop()

    assert coordinator.monitoring_stats['errors_encountered'] == 1
    assert calls['count'] == 1


def test_monitoring_loop_success_path(monkeypatch):
    coordinator = MonitoringCoordinator({'interval_seconds': 5})
    coordinator.monitoring_active = True

    cycle_calls = {'count': 0}

    def successful_cycle():
        cycle_calls['count'] += 1

    coordinator._perform_monitoring_cycle = successful_cycle  # type: ignore[attr-defined]

    sleep_calls = []

    def fake_sleep(interval):
        sleep_calls.append(interval)
        coordinator.monitoring_active = False

    monkeypatch.setattr('src.infrastructure.monitoring.services.monitoring_coordinator.time.sleep', fake_sleep)

    coordinator._monitoring_loop()

    assert coordinator.monitoring_stats['cycles_completed'] == 1
    assert coordinator.monitoring_stats['last_cycle_time'] is not None
    assert sleep_calls == [5]
    assert cycle_calls['count'] == 1


def test_perform_monitoring_cycle_exception_updates_errors():
    coordinator = MonitoringCoordinator()
    coordinator.reset_stats()

    class BadCollector:
        def collect_all_metrics(self):
            raise RuntimeError("collect fail")

    coordinator.set_components(metrics_collector=BadCollector())

    coordinator._perform_monitoring_cycle()

    assert coordinator.monitoring_stats['errors_encountered'] == 1


def test_health_status_when_errors_exceed_threshold():
    coordinator = MonitoringCoordinator()
    coordinator.set_components(Mock(), Mock(), Mock(), Mock())
    coordinator.monitoring_stats['errors_encountered'] = 15

    health = coordinator.get_health_status()

    assert "错误次数过多" in health['issues']
    assert health['status'] in {'warning', 'error'}


