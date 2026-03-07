#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充 src.infrastructure.utils.patterns.testing_tools 的测试覆盖。
"""

from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from src.infrastructure.utils.patterns.testing_tools import (
    InfrastructureIntegrationTest,
    InfrastructureTestHelper,
)


def test_create_test_environment_applies_overrides(tmp_path) -> None:
    overrides: Dict[str, Any] = {
        "cache": {"max_size": 10, "ttl": 5},
        "logging": {"level": "DEBUG"},
        "extra": {"feature": True},
    }
    env = InfrastructureIntegrationTest.create_test_environment(overrides)

    assert env["cache"]["max_size"] == 10
    assert env["cache"]["ttl"] == 5
    assert env["logging"]["level"] == "DEBUG"
    assert env["extra"]["feature"] is True
    assert os.path.isdir(env["temp_dir"])
    assert env["logging"]["file"].startswith(env["temp_dir"])


def test_run_performance_benchmark_counts_iterations() -> None:
    counter = {"calls": 0}

    def _work():
        counter["calls"] += 1

    result = InfrastructureIntegrationTest.run_performance_benchmark(
        _work, iterations=5, warmup_iterations=2
    )

    assert counter["calls"] == 7
    assert result["iterations"] == 5
    assert result["min_time"] <= result["max_time"]
    assert result["total_time"] >= result["avg_time"]


def test_assert_component_health_variants() -> None:
    class HealthyComponent:
        def __init__(self, healthy: bool = True):
            self._healthy = healthy

        def health_check(self) -> Dict[str, Any]:
            return {"healthy": self._healthy}

        def start(self) -> None:
            pass

    healthy = HealthyComponent()
    assert (
        InfrastructureIntegrationTest.assert_component_health(healthy, ["start"])
        is True
    )

    unhealthy = HealthyComponent(healthy=False)
    assert (
        InfrastructureIntegrationTest.assert_component_health(unhealthy, ["start"])
        is False
    )

    # Missing method
    assert (
        InfrastructureIntegrationTest.assert_component_health(healthy, ["missing"])
        is False
    )

    # Component without health_check
    class NoHealth:
        pass

    assert InfrastructureIntegrationTest.assert_component_health(NoHealth(), []) is False


def test_generate_test_template_contains_methods() -> None:
    template = InfrastructureTestHelper.generate_test_template(
        "Sample", ["run", "stop"]
    )
    assert "class TestSample" in template
    assert "def test_run" in template
    assert "def test_stop" in template


def test_create_mock_component_sets_attributes() -> None:
    mock_component = InfrastructureTestHelper.create_mock_component(
        "database", region="cn", version="1.0"
    )
    assert mock_component.component_type == "database"
    assert mock_component.is_healthy()
    status = mock_component.get_status()
    assert status["healthy"] is True
    assert mock_component.region == "cn"
    assert mock_component.version == "1.0"


@pytest.mark.parametrize(
    "actual,subset,expected",
    [
        ({"a": 1, "b": 2}, {"a": 1}, True),
        ({"a": {"b": 2}}, {"a": {"b": 2}}, True),
        ({"a": {"b": 2}}, {"a": {"c": 3}}, False),
        ({"a": 1}, {"a": 2}, False),
    ],
)
def test_assert_dict_contains(actual: Dict[str, Any], subset: Dict[str, Any], expected: bool) -> None:
    assert InfrastructureTestHelper.assert_dict_contains(actual, subset) is expected

