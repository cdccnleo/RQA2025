"""
基础设施层 - BusinessMetricsMonitor 单元测试

测试业务指标监控器的核心功能。
"""

from datetime import datetime
from typing import Any, Dict, List

import pytest

import src.infrastructure.resource.monitoring.metrics.business_metrics_monitor as monitor_module
from src.infrastructure.resource.monitoring.metrics.business_metrics_monitor import (
    BusinessMetricsMonitor,
    TradingMetricType,
    ModelMetricType,
    BusinessMetricType,
)


class DummyLogger:
    def __init__(self):
        self.errors: List[str] = []

    def error(self, message: str, *args, **kwargs):
        self.errors.append(message)

    def log_error(self, message: str, *args, **kwargs):
        self.error(message, *args, **kwargs)


@pytest.fixture
def monitor():
    instance = BusinessMetricsMonitor(config={"max_metrics": 2}, collection_interval=5.0, max_history_size=2)
    dummy_logger = DummyLogger()
    instance.logger = dummy_logger
    return instance, dummy_logger


def test_record_metric_and_retrieve(monitor):
    monitor_instance, _ = monitor

    monitor_instance.record_metric(TradingMetricType.VOLUME, 100.0, {"strategy": "alpha"})
    monitor_instance.record_metric(ModelMetricType.ACCURACY, 0.85, {"model": "bert"})
    monitor_instance.record_metric(BusinessMetricType.REVENUE.value, 12345.0, {"region": "apac"})

    trading = monitor_instance.get_metric(TradingMetricType.VOLUME, "alpha")
    assert len(trading) == 1 and trading[0]["value"] == 100.0

    model = monitor_instance.get_metric(ModelMetricType.ACCURACY, "bert")
    assert model[0]["value"] == pytest.approx(0.85)

    business = monitor_instance.get_metric(BusinessMetricType.REVENUE)
    assert business[0]["tags"]["region"] == "apac"

    stats = monitor_instance.get_metric_stats(TradingMetricType.VOLUME, "alpha")
    assert stats["count"] == 1 and stats["max"] == 100.0

    empty_stats = monitor_instance.get_metric_stats(TradingMetricType.PROFIT_LOSS, "alpha")
    assert empty_stats == {}


def test_alert_rule_triggers_and_error(monitor):
    monitor_instance, dummy_logger = monitor
    calls: List[Dict[str, Any]] = []

    def handler(metric, value, threshold, tags):
        calls.append({"metric": metric, "value": value, "threshold": threshold, "tags": tags})

    monitor_instance.set_alert_rule("custom_metric", threshold=10.0, condition="above", alert_handler=handler)
    monitor_instance.record_metric("custom_metric", 15.0, {"source": "test"})

    assert calls and calls[0]["value"] == 15.0

    def failing_handler(*_args, **_kwargs):
        raise RuntimeError("boom")

    monitor_instance.set_alert_rule("custom_metric", threshold=5.0, condition="above", alert_handler=failing_handler)
    monitor_instance.record_metric("custom_metric", 6.0)
    assert any("告警处理器执行失败" in msg for msg in dummy_logger.errors)


def test_max_metrics_limit_and_clear(monitor):
    monitor_instance, _ = monitor

    monitor_instance.record_metric(TradingMetricType.TRADE_COUNT, 1.0, {"strategy": "beta"})
    monitor_instance.record_metric(TradingMetricType.TRADE_COUNT, 2.0, {"strategy": "beta"})
    monitor_instance.record_metric(TradingMetricType.TRADE_COUNT, 3.0, {"strategy": "beta"})

    records = monitor_instance.get_metric(TradingMetricType.TRADE_COUNT, "beta")
    assert len(records) == 2
    assert [r["value"] for r in records] == [2.0, 3.0]

    monitor_instance.clear_metrics()
    assert monitor_instance.get_metric(TradingMetricType.TRADE_COUNT, "beta") == []
    assert monitor_instance.get_business_metrics() == {}


def test_get_monitor_stats_and_filters(monitor):
    monitor_instance, _ = monitor

    monitor_instance.record_metric(TradingMetricType.WIN_RATE, 0.55, {"strategy": "gamma"})
    monitor_instance.record_metric(ModelMetricType.LATENCY, 12.0, {"model": "xgb"})
    monitor_instance.record_metric(BusinessMetricType.USER_ACTIVE.value, 5000)

    trading_metrics = monitor_instance.get_trading_metrics("gamma")
    assert all(key.startswith("gamma.") for key in trading_metrics)

    model_metrics = monitor_instance.get_model_metrics("xgb")
    assert all(key.startswith("xgb.") for key in model_metrics)

    stats = monitor_instance.get_monitor_stats()
    assert stats["trading_metrics_count"] >= 1
    assert stats["total_business_records"] == 1
    assert stats["alert_rules_count"] >= 0