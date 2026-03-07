import time

from src.infrastructure.error.core.performance_monitor import (
    AlertManager,
    MetricsCollector,
    PerformanceMetrics,
    PerformanceMonitor,
)


def test_metrics_collector_record_and_reset():
    collector = MetricsCollector(max_history_size=3)
    collector.record_request("handler", 0.5, True, None)
    collector.record_request("handler", 1.5, False, "ValueError")
    collector.record_request("handler", 0.7, True, None)
    collector.record_request("handler", 0.9, False, "RuntimeError")

    metrics = collector.get_metrics("handler")
    assert metrics.total_requests == 4
    assert metrics.failed_requests == 2
    assert "ValueError" in metrics.error_counts
    assert len(metrics.response_times) == 3  # 历史长度被限制

    collector.reset_metrics("handler")
    metrics_after_reset = collector.get_metrics("handler")
    assert metrics_after_reset.total_requests == 0

    collector.record_request("handler", 0.3, True, None)
    collector.reset_metrics()
    assert collector.get_metrics("handler").total_requests == 0


def test_alert_manager_triggers_on_thresholds():
    alert_manager = AlertManager(alert_check_interval=0)
    alerts = []
    alert_manager.add_alert_callback(lambda alert: alerts.append(alert.alert_type))
    alert_manager.set_alert_threshold("error_rate_threshold", 0.1)
    alert_manager.set_alert_threshold("response_time_threshold", 0.5)
    alert_manager.set_alert_threshold("throughput_drop_threshold", 0.9)

    metrics = PerformanceMetrics(
        total_requests=20,
        successful_requests=10,
        failed_requests=10,
        total_response_time=20.0,
        response_times=list(range(20)),
        error_counts={"ValueError": 5},
        throughput_history=[10.0, 1.0],
    )

    now = time.time()
    alert_manager._check_handler_alerts("handler", metrics, now)

    assert "high_error_rate" in alerts
    assert "high_response_time" in alerts
    assert "high_p95_response_time" in alerts
    assert "throughput_drop" in alerts


def test_performance_monitor_reports_and_suggestions():
    monitor = PerformanceMonitor(test_mode=True)
    captured_alerts = []
    monitor.add_alert_callback(lambda alert: captured_alerts.append(alert.alert_type))
    monitor.set_alert_threshold("error_rate_threshold", 0.0)
    monitor.set_alert_threshold("response_time_threshold", 0.01)

    monitor.record_request("handler", response_time=1.0, success=False, error_type="ValueError")
    monitor.record_request("handler", response_time=0.5, success=True, error_type=None)
    monitor.check_alerts()

    handler_report = monitor.get_performance_report("handler")
    assert handler_report["total_requests"] == 2
    assert handler_report["failed_requests"] == 1

    summary = monitor.get_performance_report()
    assert summary["total_handlers"] == 1
    assert summary["total_errors"] == 1

    suggestions = monitor.get_optimization_suggestions("handler")
    assert suggestions  # 至少给出一条建议

    assert captured_alerts  # 阈值足够低，应触发告警

    monitor.reset_metrics("handler")
    assert monitor.get_metrics("handler").total_requests == 0

