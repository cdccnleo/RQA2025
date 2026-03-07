#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对 src.infrastructure.utils.patterns.core_tools 的补充测试，覆盖日志、异常、初始化、配置与装饰器流程。
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import time

import pytest

from src.infrastructure.utils.patterns import core_tools


@pytest.fixture(autouse=True)
def reset_logger_handlers():
    """确保模块级 logger 存在 handler，避免无 handler 情况下记录失败。"""
    logger = logging.getLogger(core_tools.__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield


def test_infrastructure_logger_logs_messages(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=core_tools.__name__)

    core_tools.InfrastructureLogger.log_initialization_success("Cache")
    core_tools.InfrastructureLogger.log_operation_failure("写入", ValueError("bad"), "detail")
    core_tools.InfrastructureLogger.log_performance_metric("同步", duration=2.5, threshold=1.0)
    core_tools.InfrastructureLogger.log_cache_operation("命中", "key-1", hit=True, size=10)

    messages = [record.message for record in caplog.records]
    assert any("Cache 组件初始化成功" in msg for msg in messages)
    assert any("写入 操作失败: bad (detail)" in msg for msg in messages)
    assert any("性能警告: 同步 耗时 2.500s" in msg for msg in messages)
    assert any("缓存命中: key=key-1, hit=是, size=10" in msg for msg in messages)


def test_exception_handler_methods_raise_and_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger=core_tools.__name__)

    with pytest.raises(RuntimeError):
        core_tools.InfrastructureExceptionHandler.handle_initialization_error("Service", RuntimeError("boom"))

    with pytest.raises(ValueError):
        core_tools.InfrastructureExceptionHandler.handle_validation_error("field", 123, "str")

    with pytest.raises(ConnectionError):
        core_tools.InfrastructureExceptionHandler.handle_connection_error("redis", ConnectionError("down"), retry_count=2)

    assert any("初始化失败" in record.message for record in caplog.records)
    assert any("字段 'field' 验证失败" in record.message for record in caplog.records)
    assert any("服务 'redis' 连接失败 (已重试 2 次)" in record.message for record in caplog.records)


def test_safe_execute_returns_none_on_error(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger=core_tools.__name__)

    def _boom() -> None:
        raise ValueError("bad exec")

    assert core_tools.InfrastructureExceptionHandler.safe_execute(_boom) is None
    assert any("执行 _boom 时发生错误: bad exec" in record.message for record in caplog.records)


def test_initializer_success_and_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=core_tools.__name__)

    assert core_tools.InfrastructureInitializer.initialize_component("Adapter", lambda: "ok")
    assert any("Adapter 组件初始化成功" in record.message for record in caplog.records)

    def _primary():
        raise RuntimeError("primary fail")

    def _fallback():
        return "fallback"

    caplog.clear()
    assert core_tools.InfrastructureInitializer.initialize_with_fallback("Adapter", _primary, _fallback) == "fallback"
    assert any("主要初始化失败" in record.message for record in caplog.records)
    assert any("降级初始化成功" in record.message for record in caplog.records)

    with pytest.raises(RuntimeError):
        core_tools.InfrastructureInitializer.initialize_with_fallback("Adapter", _primary, None)


def test_config_helpers() -> None:
    config: Dict[str, Any] = {"database": {"host": "localhost", "port": 5432}, "cache": None}

    assert core_tools.InfrastructureConfig.get_nested_config(config, ["database", "host"]) == "localhost"
    assert core_tools.InfrastructureConfig.get_nested_config(config, ["database", "missing"], default="default") == "default"
    assert core_tools.InfrastructureConfig.get_nested_config(config, ["cache", "enabled"], default=False) is False

    assert core_tools.InfrastructureConfig.validate_required_config(config, ["database"])
    assert not core_tools.InfrastructureConfig.validate_required_config(config, ["database", "cache"])

    merged = core_tools.InfrastructureConfig.merge_configs(
        {"a": 1, "nested": {"x": 1}},
        {"b": 2, "nested": {"y": 3}},
    )
    assert merged == {"a": 1, "b": 2, "nested": {"x": 1, "y": 3}}


def test_performance_monitor_measure_and_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=core_tools.__name__)

    def _work(duration: float) -> str:
        time_spent = duration  # emulate workload
        return f"done-{time_spent}"

    result, elapsed = core_tools.InfrastructurePerformanceMonitor.measure_execution_time(_work, 0.01)
    assert result.startswith("done-")
    assert elapsed >= 0.0

    core_tools.InfrastructurePerformanceMonitor.log_performance("op", elapsed=2.0, threshold=1.0)
    assert any("操作耗时过长" in record.message for record in caplog.records)


def test_infrastructure_operation_decorator_success_and_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=core_tools.__name__)

    @core_tools.infrastructure_operation("run")
    def _success(x: int) -> int:
        return x * 2

    assert _success(3) == 6
    assert any("run 操作成功" in record.message for record in caplog.records)

    @core_tools.infrastructure_operation("boom")
    def _boom():
        raise RuntimeError("explode")

    with pytest.raises(RuntimeError):
        _boom()


def test_safe_infrastructure_operation_returns_default(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger=core_tools.__name__)

    @core_tools.safe_infrastructure_operation("safe-boom", default_return="fallback")
    def _safe_boom() -> None:
        raise RuntimeError("broken")

    assert _safe_boom() == "fallback"
    assert any("safe-boom 操作失败: broken" in record.message for record in caplog.records)

