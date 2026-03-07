from prometheus_client import CollectorRegistry, REGISTRY

from src.infrastructure.cache.monitoring.business_metrics_plugin import (
    BusinessMetricsPlugin,
    update_active_users,
    update_strategy_return,
    inc_strategy_call,
    business_metrics,
)


def test_business_metrics_plugin_updates_and_prometheus():
    registry = CollectorRegistry()
    plugin = BusinessMetricsPlugin(registry=registry)

    plugin.update_strategy_return("alpha", 1.23)
    plugin.update_active_users(42)
    plugin.inc_strategy_call("alpha")

    metrics_dict = plugin.get_metrics_dict()
    assert metrics_dict["strategy_returns"]["alpha"] == 1.23
    assert metrics_dict["active_users"] == 42
    assert metrics_dict["strategy_calls"]["alpha"] == 1

    prometheus_output = plugin.get_metrics_prometheus().decode()
    assert "strategy_return" in prometheus_output
    assert "active_users" in prometheus_output


def test_business_metrics_global_functions_use_default():
    update_strategy_return("beta", 2.5)
    update_active_users(5)
    inc_strategy_call("beta")

    output = business_metrics().decode()
    assert "strategy_return" in output
    assert "strategy_calls_total" in output

