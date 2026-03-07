import json
from unittest.mock import Mock

import pytest

import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.ml.core import error_handling


@pytest.fixture(autouse=True)
def reset_error_handler(monkeypatch):
    """确保每个用例都有干净的错误处理器实例和日志适配器。"""
    mock_logger = Mock()
    mock_adapter = Mock()
    mock_adapter.get_models_logger.return_value = mock_logger
    monkeypatch.setattr(error_handling, "_get_models_adapter", Mock(return_value=mock_adapter))
    monkeypatch.setattr(error_handling, "get_models_adapter", Mock(return_value=mock_adapter))
    monkeypatch.setattr(error_handling, "_ml_error_handler", None)
    yield
    monkeypatch.setattr(error_handling, "_ml_error_handler", None)


def test_handle_error_with_recovery_strategy():
    handler = error_handling.get_ml_error_handler()

    callback_hits = []
    system_hits = []

    handler.register_error_callback(
        error_handling.MLErrorCategory.DATA_ERROR,
        lambda err: callback_hits.append(err.error_id),
    )
    handler.register_error_callback(
        error_handling.MLErrorCategory.SYSTEM_ERROR,
        lambda err: system_hits.append(err.error_id),
    )

    def recovery_action(err: error_handling.MLError):
        err.context["recover_action"] = "fallback"
        return {"status": "ok"}

    handler.register_recovery_strategy(
        error_handling.ErrorRecoveryStrategy(
            strategy_id="data_recover",
            error_category=error_handling.MLErrorCategory.DATA_ERROR,
            condition=lambda e: True,
            recovery_action=recovery_action,
            max_attempts=2,
            cooldown_seconds=0,
            priority=5,
        )
    )

    ml_exception = error_handling.DataValidationError(
        "missing value", context={"source": "s3_bucket"}
    )
    error_handling.handle_ml_error(ml_exception, context={"batch": 42})

    assert callback_hits and system_hits
    error = handler.errors[callback_hits[0]]

    assert error.resolved is True
    assert error.context["source"] == "s3_bucket"
    assert error.context["batch"] == 42
    assert error.recovery_attempts == 1
    assert error.context["recover_action"] == "fallback"
    assert callback_hits == [error.error_id]
    assert system_hits == [error.error_id]


def test_handle_plain_exception_updates_statistics():
    handler = error_handling.get_ml_error_handler()

    handler.handle_error(ValueError("bad input"), context={"step": "load"})
    stored = handler.error_history[-1]

    assert stored.resolved is False
    assert stored.context["step"] == "load"
    assert stored.category is error_handling.MLErrorCategory.SYSTEM_ERROR

    stats = handler.get_error_statistics()
    assert stats["total_errors"] == 1
    assert stats["active_errors"] == 1
    assert stats["error_distribution"][error_handling.MLErrorCategory.SYSTEM_ERROR.value] == 1
    assert (
        stats["severity_distribution"][error_handling.MLErrorSeverity.MEDIUM.value] == 1
    )


def test_handle_existing_mLError_and_resolution_flow():
    handler = error_handling.get_ml_error_handler()

    ml_error = error_handling.MLError(
        error_id="custom_error",
        category=error_handling.MLErrorCategory.MODEL_ERROR,
        severity=error_handling.MLErrorSeverity.HIGH,
        message="model failed",
    )

    handler.handle_error(ml_error)
    assert handler.errors["custom_error"] is ml_error
    assert handler.errors["custom_error"] is ml_error

    assert handler.resolve_error("custom_error") is True
    assert handler.resolve_error("missing") is False

    recent = handler.get_recent_errors(limit=1)[0]
    assert recent["error_id"] == "custom_error"
    assert recent["resolved"] is True

    report = json.loads(handler.export_error_report())
    assert report["statistics"]["total_errors"] == 1
    assert report["recent_errors"][0]["error_id"] == "custom_error"

    with pytest.raises(ValueError):
        handler.export_error_report("yaml")


def test_recovery_strategy_priority_applies_highest_priority_first():
    handler = error_handling.get_ml_error_handler()

    executed = []

    def make_strategy(name, priority):
        return error_handling.ErrorRecoveryStrategy(
            strategy_id=name,
            error_category=error_handling.MLErrorCategory.DATA_ERROR,
            condition=lambda e: True,
            recovery_action=lambda e: executed.append(name) or {"status": name},
            max_attempts=2,
            cooldown_seconds=0,
            priority=priority,
        )

    handler.register_recovery_strategy(make_strategy("low", priority=1))
    handler.register_recovery_strategy(make_strategy("high", priority=10))

    ids = [strategy.strategy_id for strategy in handler.recovery_strategies]
    assert ids.index("high") < ids.index("low")

    error_handling.handle_ml_error(
        error_handling.DataValidationError("need strategy"),
        context={"attempt": "priority"},
    )

    assert executed == ["high"]


def test_default_recovery_strategy_recover_data_error():
    handler = error_handling.get_ml_error_handler()

    ml_exception = error_handling.DataValidationError("data load failed")
    result = error_handling.handle_ml_error(ml_exception)

    assert result.resolved is True
    assert result.recovery_attempts == 1
    assert result.resolution_time is not None


def test_default_recovery_strategy_recover_inference_error():
    handler = error_handling.get_ml_error_handler()

    inference_error = error_handling.InferenceError("predict failed")
    result = error_handling.handle_ml_error(inference_error)

    assert result.resolved is True
    assert result.recovery_attempts == 1


def test_resource_recovery_applies_for_non_critical(monkeypatch):
    handler = error_handling.get_ml_error_handler()

    calls = []
    monkeypatch.setattr("gc.collect", lambda: calls.append("gc"))

    resource_error = error_handling.MLError(
        error_id="resource_issue",
        category=error_handling.MLErrorCategory.RESOURCE_ERROR,
        severity=error_handling.MLErrorSeverity.MEDIUM,
        message="memory high",
    )

    result = handler.handle_error(resource_error)
    assert result.resolved is True
    assert result.recovery_attempts == 1
    assert calls == ["gc"]


def test_handle_error_returns_existing_error_and_respects_max_attempts():
    handler = error_handling.get_ml_error_handler()

    executed = []

    def failing_action(err):
        executed.append("run")
        raise RuntimeError("fail")

    handler.register_recovery_strategy(
        error_handling.ErrorRecoveryStrategy(
            strategy_id="always_fail",
            error_category=error_handling.MLErrorCategory.MODEL_ERROR,
            condition=lambda e: True,
            recovery_action=failing_action,
            max_attempts=1,
            cooldown_seconds=0,
            priority=5,
        )
    )

    ml_error = error_handling.MLError(
        error_id="model_issue",
        category=error_handling.MLErrorCategory.MODEL_ERROR,
        severity=error_handling.MLErrorSeverity.HIGH,
        message="model issue",
        max_recovery_attempts=1,
    )

    returned = handler.handle_error(ml_error)

    assert returned is ml_error
    assert returned.resolved is False
    assert returned.recovery_attempts == 1
    assert executed == ["run"]

    # 触发第二次处理，不应再次执行恢复策略
    handler.handle_error(ml_error)
    assert executed == ["run"]


def test_recovery_strategy_failure_keeps_error_unresolved():
    handler = error_handling.get_ml_error_handler()
    errors = []

    def failing_action(err):
        errors.append(err.error_id)
        raise RuntimeError("recover boom")

    handler.register_recovery_strategy(
        error_handling.ErrorRecoveryStrategy(
            strategy_id="fail_once",
            error_category=error_handling.MLErrorCategory.DATA_ERROR,
            condition=lambda e: True,
            recovery_action=failing_action,
            max_attempts=2,
            cooldown_seconds=0,
            priority=1,
        )
    )

    ml_error = error_handling.MLError(
        error_id="data_issue",
        category=error_handling.MLErrorCategory.DATA_ERROR,
        severity=error_handling.MLErrorSeverity.MEDIUM,
        message="bad data",
        max_recovery_attempts=2,
    )

    result = handler.handle_error(ml_error)
    assert result.recovery_attempts == 1
    assert result.resolved is False
    assert errors == ["data_issue"]

    # 第二次尝试应当增加 recovery_attempts，但仍未解决
    result_again = handler.handle_error(result)
    assert result_again.recovery_attempts == 2
    assert result_again.resolved is False


def test_recovery_strategy_condition_prevents_execution():
    handler = error_handling.get_ml_error_handler()
    executed = []

    handler.register_recovery_strategy(
        error_handling.ErrorRecoveryStrategy(
            strategy_id="never_run",
            error_category=error_handling.MLErrorCategory.MODEL_ERROR,
            condition=lambda e: False,
            recovery_action=lambda e: executed.append("run"),
            max_attempts=2,
            cooldown_seconds=0,
            priority=1,
        )
    )

    model_error = error_handling.MLError(
        error_id="model_fail",
        category=error_handling.MLErrorCategory.MODEL_ERROR,
        severity=error_handling.MLErrorSeverity.HIGH,
        message="fail",
    )

    handler.handle_error(model_error)
    assert executed == []


def test_export_error_report_includes_active_errors():
    handler = error_handling.get_ml_error_handler()

    unresolved = error_handling.MLError(
        error_id="active_1",
        category=error_handling.MLErrorCategory.SYSTEM_ERROR,
        severity=error_handling.MLErrorSeverity.HIGH,
        message="still bad",
    )

    handler.handle_error(unresolved)

    report = json.loads(handler.export_error_report())
    active_ids = [item["error_id"] for item in report["active_errors"]]
    assert "active_1" in active_ids


def test_get_error_statistics_recovery_success_rate(monkeypatch):
    handler = error_handling.get_ml_error_handler()

    class DummyError(error_handling.MLError):
        pass

    resolved = DummyError(
        error_id="resolved",
        category=error_handling.MLErrorCategory.DATA_ERROR,
        severity=error_handling.MLErrorSeverity.LOW,
        message="resolved",
        recovery_attempts=2,
    )
    resolved.resolved = True

    unresolved = DummyError(
        error_id="unresolved",
        category=error_handling.MLErrorCategory.MODEL_ERROR,
        severity=error_handling.MLErrorSeverity.HIGH,
        message="not fixed",
        recovery_attempts=1,
    )
    unresolved.resolved = False

    handler.error_history.extend([resolved, unresolved])
    handler.errors.update({resolved.error_id: resolved, unresolved.error_id: unresolved})
    handler.error_counters.update(
        {
            resolved.category: handler.error_counters[resolved.category] + 1,
            unresolved.category: handler.error_counters[unresolved.category] + 1,
        }
    )

    stats = handler.get_error_statistics()
    assert stats["total_errors"] >= 2
    assert stats["resolved_errors"] >= 1
    assert 0 <= stats["recovery_success_rate"] <= 1


def test_register_helpers_expose_handler(monkeypatch):
    handler = error_handling.get_ml_error_handler()

    calls = []

    def cb(err):
        calls.append(err.error_id)

    strategy = error_handling.ErrorRecoveryStrategy(
        strategy_id="helper_strategy",
        error_category=error_handling.MLErrorCategory.SYSTEM_ERROR,
        condition=lambda e: True,
        recovery_action=lambda e: {"status": "ok"},
        max_attempts=1,
        cooldown_seconds=0,
        priority=10,
    )

    error_handling.register_error_recovery_strategy(strategy)
    error_handling.register_error_callback(error_handling.MLErrorCategory.SYSTEM_ERROR, cb)

    ml_exception = error_handling.MLException("boom")
    error_handling.handle_ml_error(ml_exception)

    ids = [s.strategy_id for s in handler.recovery_strategies]
    assert "helper_strategy" in ids
    assert calls


def test_get_error_statistics_convenience_function():
    handler = error_handling.get_ml_error_handler()
    handler.handle_error(ValueError("another"))
    stats = error_handling.get_error_statistics()
    assert "total_errors" in stats


def test_ml_error_handler_decorator_handles_exception(monkeypatch):
    captured = []
    original_handle = error_handling.handle_ml_error

    def spying_handle(error, context=None):
        captured.append(error)
        return original_handle(error, context)

    monkeypatch.setattr(error_handling, "handle_ml_error", spying_handle)

    @error_handling.ml_error_handler(
        category=error_handling.MLErrorCategory.MODEL_ERROR,
        severity=error_handling.MLErrorSeverity.HIGH,
    )
    def faulty_function(x):
        raise RuntimeError(f"bad {x}")

    with pytest.raises(RuntimeError):
        faulty_function(123)

    assert captured
    error = captured[0]
    assert error.category == error_handling.MLErrorCategory.MODEL_ERROR
    assert error.severity == error_handling.MLErrorSeverity.HIGH
    assert "faulty_function" in error.context.get("function", "")


def test_model_load_error_includes_model_id():
    err = error_handling.ModelLoadError("load failed", model_id="model-123")
    assert err.context["model_id"] == "model-123"
    assert err.category == error_handling.MLErrorCategory.MODEL_ERROR


def test_resource_exhaustion_error_sets_context():
    err = error_handling.ResourceExhaustionError("oom", resource_type="gpu")
    assert err.context["resource_type"] == "gpu"
    assert err.severity == error_handling.MLErrorSeverity.CRITICAL


def test_error_history_trim_limits_size():
    handler = error_handling.MLErrorHandler()
    handler.recovery_strategies = []

    base_error = error_handling.MLError(
        error_id="base",
        category=error_handling.MLErrorCategory.SYSTEM_ERROR,
        severity=error_handling.MLErrorSeverity.LOW,
        message="base",
    )

    handler.error_history = [
        error_handling.MLError(
            error_id=f"old-{i}",
            category=error_handling.MLErrorCategory.SYSTEM_ERROR,
            severity=error_handling.MLErrorSeverity.LOW,
            message="old",
        )
        for i in range(1001)
    ]

    result = handler.handle_error(base_error)
    assert len(handler.error_history) == 1000
    assert handler.error_history[0].error_id != "old-0"
    assert result is base_error


def test_error_callbacks_failure_is_logged(monkeypatch):
    handler = error_handling.MLErrorHandler()
    handler.recovery_strategies = []

    def bad_callback(error):
        raise RuntimeError("callback boom")

    handler.register_error_callback(error_handling.MLErrorCategory.DATA_ERROR, bad_callback)
    handler.register_error_callback(error_handling.MLErrorCategory.SYSTEM_ERROR, bad_callback)

    err = error_handling.MLError(
        error_id="cb",
        category=error_handling.MLErrorCategory.DATA_ERROR,
        severity=error_handling.MLErrorSeverity.MEDIUM,
        message="cb",
    )

    result = handler.handle_error(err)
    assert result is err


def test_logger_fallback_when_adapter_raises(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    original_module = sys.modules.get("src.core.integration")

    failing_module = ModuleType("src.core.integration")

    def failing_adapter():
        raise RuntimeError("adapter failure")

    failing_module.get_models_adapter = failing_adapter
    monkeypatch.setitem(sys.modules, "src.core.integration", failing_module)

    reloaded = importlib.reload(error_handling)
    try:
        assert reloaded.logger.name == "src.ml.core.error_handling"
    finally:
        if original_module is None:
            sys.modules.pop("src.core.integration", None)
        else:
            sys.modules["src.core.integration"] = original_module
        globals()["error_handling"] = importlib.reload(reloaded)

