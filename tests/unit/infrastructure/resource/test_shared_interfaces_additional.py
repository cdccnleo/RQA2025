import builtins
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.infrastructure.resource.core.shared_interfaces import (
    StandardLogger,
    BaseErrorHandler,
    ConfigValidator,
    DataValidator,
    ResourceManager,
)


@pytest.fixture
def patched_logging(monkeypatch):
    logger_mock = MagicMock()
    monkeypatch.setattr("logging.getLogger", lambda name=None: logger_mock)
    return logger_mock


def test_standard_logger_merges_extra(patched_logging):
    logger = StandardLogger("component")
    logger.log_info("hello", extra={"foo": 1}, bar=2)
    patched_logging.info.assert_called_once_with("hello", extra={"foo": 1, "bar": 2})

    patched_logging.reset_mock()
    exc = ValueError("boom")
    logger.log_error("oops", error=exc, extra={"key": "value"})
    patched_logging.error.assert_called_once()
    args, kwargs = patched_logging.error.call_args
    assert "oops" in args[0] and "boom" in args[0]
    assert kwargs["exc_info"] is True
    assert kwargs["extra"] == {"key": "value"}

    patched_logging.reset_mock()
    logger.warning("warn-message", extra={"a": 1})
    patched_logging.warning.assert_called_once_with("warn-message", extra={"a": 1})


def test_base_error_handler_records_and_resets(monkeypatch, patched_logging):
    handler = BaseErrorHandler(max_retries=2, retry_delay=0.1)
    error = RuntimeError("failure")
    handler.handle_error(error, context={"step": "test"})
    summary = handler.get_error_summary()
    assert summary["error_count"] == 1
    assert "failure" in summary["last_error"]
    assert handler.should_retry(error, attempt=1) is True
    assert handler.should_retry(error, attempt=2) is False

    handler.reset()
    summary_after_reset = handler.get_error_summary()
    assert summary_after_reset["error_count"] == 0
    assert summary_after_reset["last_error"] is None

    with pytest.raises(ValueError):
        handler.handle_error(ValueError("raise"), reraise=True)


def test_config_validator_validations():
    validator = ConfigValidator()
    assert validator.validate_config(None) is False
    assert "配置不能为空" in validator.get_validation_errors()

    valid = {"name": "cpu_pool", "size": 5}
    assert validator.validate_config(valid) is True
    assert validator.validate_required_fields(valid, ["name", "size"]) is True
    assert validator.validate_field_types(valid, {"size": int}) is True

    invalid_type = {"size": "large"}
    validator.validate_field_types(invalid_type, {"size": int})
    assert any("字段 size 类型错误" in err for err in validator.get_validation_errors())


def test_data_validator_schema_and_sanitize():
    validator = DataValidator()
    assert validator.validate_data(None) is False
    assert "数据不能为空" in validator.get_validation_errors()

    schema = {"age": {"type": int, "min": 0, "max": 120, "required": True}}
    assert validator.validate_data({"age": 30}, schema) is True
    assert validator.validate_data({"age": -1}, schema) is False

    sanitized = validator.sanitize_data({"a": 1, "b": None})
    assert sanitized == {"a": 1}


@pytest.fixture
def resource_manager(monkeypatch, patched_logging):
    manager = ResourceManager()
    # patch psutil usage for deterministic metrics
    psutil_stub = SimpleNamespace(
        cpu_percent=MagicMock(return_value=12.5),
        virtual_memory=MagicMock(return_value=SimpleNamespace(percent=34.5)),
        disk_usage=MagicMock(return_value=SimpleNamespace(percent=56.7)),
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil_stub)
    return manager, psutil_stub, patched_logging


def test_resource_manager_acquire_release(resource_manager):
    manager, _, logger_mock = resource_manager
    res = manager.acquire_resource("gpu-1")
    assert res["id"] == "gpu-1"
    assert manager.acquire_resource("gpu-1") is None
    logger_mock.warning.assert_called_once()

    assert manager.release_resource("cpu") is False
    assert manager.release_resource("gpu-1") is True


def test_resource_manager_allocation_flow(resource_manager):
    manager, _, _ = resource_manager
    assert manager.allocate_resource("cpu", 5) is True
    assert manager.allocate_resource("cpu", 6) is False
    assert manager.release_resource("cpu", amount=3) is True
    status = manager.get_resource_status()
    assert "cpu" in status and status["cpu"]["allocated"] == 2


def test_resource_manager_usage_metrics(resource_manager):
    manager, psutil_stub, _ = resource_manager
    usage = manager.get_current_usage()
    assert usage == {
        "cpu_percent": 12.5,
        "memory_percent": 34.5,
        "disk_percent": 56.7,
    }
    psutil_stub.cpu_percent.assert_called_once()


def test_resource_manager_optimize_and_summary(resource_manager):
    manager, _, _ = resource_manager
    manager.allocate_resource("memory", 1024)
    summary = manager.get_resource_status()
    assert summary["memory"]["allocated"] == 1024
    optimize_result = manager.optimize_resources()
    assert optimize_result["optimized"] is True
    assert optimize_result["recommendations"] == []

