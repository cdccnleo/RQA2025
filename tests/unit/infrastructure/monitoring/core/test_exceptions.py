import pytest

from src.infrastructure.monitoring.core import exceptions as exc


def test_monitoring_exception_attributes():
    e = exc.MonitoringException(
        "error",
        monitor_type="system",
        component_name="collector",
        details={"code": 500},
    )
    assert e.monitor_type == "system"
    assert e.component_name == "collector"
    assert e.details["code"] == 500


def test_specialized_exceptions_store_fields():
    err = exc.MonitorConfigurationError(
        "config failed",
        config_key="interval",
        expected_value=10,
        actual_value=0,
        monitor_type="system",
    )
    assert err.config_key == "interval"
    assert err.expected_value == 10
    assert err.actual_value == 0
    assert err.monitor_type == "system"

    metric_err = exc.MetricCollectionError("metric", metric_name="cpu", collection_method="psutil")
    assert metric_err.metric_name == "cpu"
    assert metric_err.collection_method == "psutil"

    alert_err = exc.AlertProcessingError("alert", alert_id="a1", alert_rule="rule")
    assert alert_err.alert_id == "a1"
    assert alert_err.alert_rule == "rule"

    storage_err = exc.StorageError("storage", storage_type="db", operation="save")
    assert storage_err.storage_type == "db"
    assert storage_err.operation == "save"


def test_handle_monitoring_exception_decorator_wraps_errors(monkeypatch):
    class DummyMonitoringException(Exception):
        def __init__(self, message: str, **kwargs):
            super().__init__(message)
            self.payload = kwargs

    monkeypatch.setattr(exc, "MonitoringException", DummyMonitoringException)

    @exc.handle_monitoring_exception(operation="collect")
    def func():
        raise RuntimeError("boom")

    with pytest.raises(DummyMonitoringException) as oy:
        func()
    assert oy.value.payload["details"]["original_error"] == "boom"


def test_handle_metric_collection_exception_wraps():
    @exc.handle_metric_collection_exception(metric_name="cpu", collection_method="psutil")
    def func():
        raise RuntimeError("fail")

    with pytest.raises(exc.MetricCollectionError) as err:
        func()
    assert err.value.metric_name == "cpu"


def test_handle_alert_processing_exception_wraps():
    @exc.handle_alert_processing_exception(alert_id="alert1", alert_rule="rule1")
    def func():
        raise RuntimeError("fail")

    with pytest.raises(exc.AlertProcessingError) as err:
        func()
    assert err.value.alert_id == "alert1"


def test_handle_health_check_exception_wraps():
    @exc.handle_health_check_exception(check_type="system", check_target="db")
    def func():
        raise RuntimeError("fail")

    with pytest.raises(exc.HealthCheckError) as err:
        func()
    assert err.value.check_target == "db"


def test_handle_threshold_check_exception_wraps_generic(monkeypatch):
    class DummyMonitoringException(Exception):
        def __init__(self, message: str, **kwargs):
            super().__init__(message)
            self.payload = kwargs

    monkeypatch.setattr(exc, "MonitoringException", DummyMonitoringException)

    @exc.handle_threshold_check_exception(metric_name="latency", threshold=100)
    def func():
        raise RuntimeError("too slow")

    with pytest.raises(DummyMonitoringException) as err:
        func()
    assert err.value.payload["details"]["threshold"] == 100


@pytest.mark.parametrize(
    "cls, kwargs, attr, expected",
    [
        (exc.NotificationError, {"notification_type": "email", "recipient": "ops"}, "notification_type", "email"),
        (exc.HealthCheckError, {"check_type": "db", "check_target": "primary"}, "check_target", "primary"),
        (exc.ThresholdExceededError, {"metric_name": "cpu", "threshold_value": 80.0, "actual_value": 95.0}, "actual_value", 95.0),
        (exc.MonitorConnectionError, {"target_host": "localhost", "target_port": 8080}, "target_port", 8080),
        (exc.DataProcessingError, {"data_type": "json", "processing_step": "parse"}, "processing_step", "parse"),
        (exc.AlertRuleError, {"rule_id": "r1", "rule_condition": "cpu>90"}, "rule_condition", "cpu>90"),
        (exc.PerformanceMonitorError, {"performance_metric": "latency", "expected_performance": 200, "actual_performance": 450}, "expected_performance", 200),
        (exc.DisasterRecoveryError, {"recovery_phase": "failover", "failure_point": "db"}, "recovery_phase", "failover"),
        (exc.ComponentMonitorError, {"component_type": "worker", "component_id": "w1"}, "component_id", "w1"),
        (exc.ContinuousMonitoringError, {"monitoring_cycle": 10, "failure_cycle": 5}, "failure_cycle", 5),
        (exc.ExceptionMonitorError, {"exception_type": "RuntimeError", "exception_count": 3}, "exception_count", 3),
        (exc.ProductionMonitorError, {"environment": "prod", "production_metric": "throughput"}, "environment", "prod"),
        (exc.LoggerPoolMonitorError, {"pool_name": "main", "pool_size": 12}, "pool_size", 12),
        (exc.SystemMonitorError, {"system_component": "api", "system_metric": "requests"}, "system_metric", "requests"),
    ],
)
def test_additional_exception_classes(cls, kwargs, attr, expected):
    err = cls("message", **kwargs)
    assert getattr(err, attr) == expected
    assert err.message == "message"


def test_handle_monitoring_exception_pass_through():
    @exc.handle_monitoring_exception(operation="collect")
    def func():
        raise exc.MonitoringException("boom", monitor_type="sys")

    with pytest.raises(exc.MonitoringException) as err:
        func()
    assert err.value.monitor_type == "sys"


def test_threshold_check_exception_pass_through():
    @exc.handle_threshold_check_exception(metric_name="cpu", threshold=80)
    def func():
        raise exc.ThresholdExceededError("cpu high", metric_name="cpu")

    with pytest.raises(exc.ThresholdExceededError) as err:
        func()
    assert err.value.metric_name == "cpu"


def test_handle_metric_collection_exception_wraps_details():
    @exc.handle_metric_collection_exception(metric_name="cpu", collection_method="psutil")
    def func():
        raise ValueError("missing psutil")

    with pytest.raises(exc.MetricCollectionError) as err:
        func()
    assert err.value.details["original_error"] == "missing psutil"
