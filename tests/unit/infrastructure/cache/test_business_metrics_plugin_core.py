from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry

from src.infrastructure.cache.monitoring.business_metrics_plugin import (
    BusinessMetricsPlugin,
    update_active_users,
    update_strategy_return,
    inc_strategy_call,
)


def test_business_metrics_plugin_basic_flow():
    registry = CollectorRegistry()
    plugin = BusinessMetricsPlugin(registry=registry)

    plugin.update_strategy_return("alpha", 1.23)
    plugin.update_active_users(42)
    plugin.inc_strategy_call("alpha")
    plugin.inc_strategy_call("alpha")

    metrics = plugin.get_metrics_dict()
    assert metrics["strategy_returns"]["alpha"] == 1.23
    assert metrics["active_users"] == 42
    assert metrics["strategy_calls"]["alpha"] == 2

    assert plugin.get_strategy_return("missing") == 0.0
    assert plugin.get_strategy_calls("missing") == 0


def test_business_metrics_plugin_prometheus_output():
    plugin = BusinessMetricsPlugin()
    plugin.update_strategy_return("beta", 5.0)
    plugin.update_active_users(7)
    plugin.inc_strategy_call("beta")

    data = plugin.get_metrics_prometheus()
    assert b"strategy_return" in data
    assert b"beta" in data
    assert b"active_users" in data


def test_business_metrics_plugin_collect_aliases():
    plugin = BusinessMetricsPlugin()
    plugin.increment_strategy_calls("gamma")
    metrics = plugin.collect_metrics()
    assert metrics["strategy_calls"]["gamma"] == 1


def test_module_level_helpers_isolate_default_collector():
    update_strategy_return("delta", 2.0)
    update_active_users(10)
    inc_strategy_call("delta")

    from src.infrastructure.cache.monitoring import business_metrics_plugin as module

    if hasattr(module, "_default_collector"):
        prom_data = module._default_collector.get_metrics_prometheus()
        assert b"delta" in prom_data
        assert b"active_users" in prom_data


