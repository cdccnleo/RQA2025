import pytest
from unittest.mock import MagicMock

from src.infrastructure.resource.core.optimization_memory_optimizer import MemoryOptimizer
from src.infrastructure.resource.core.optimization_config import MemoryOptimizationConfig


@pytest.fixture
def optimizer():
    logger = MagicMock()
    error_handler = MagicMock()
    optimizer = MemoryOptimizer(logger=logger, error_handler=error_handler)
    return optimizer, logger, error_handler


def test_optimize_memory_defaults_no_actions(optimizer):
    opt, logger, _ = optimizer
    result = opt.optimize_memory({}, {"memory_usage": 40})
    assert result["status"] == "applied"
    assert result["actions"] == []
    logger.log_warning.assert_not_called()


def test_optimize_memory_triggers_gc(optimizer, monkeypatch):
    opt, _, _ = optimizer
    monkeypatch.setattr(opt, "_perform_garbage_collection", MagicMock(return_value=123))
    config = {"gc_threshold": 50, "enable_pooling": True}
    resources = {"memory_usage": 75}

    result = opt.optimize_memory(config, resources)
    assert "执行垃圾回收，清理 123 个对象" in result["actions"]
    assert "启用对象池化以减少内存分配" in result["actions"]
    opt._perform_garbage_collection.assert_called_once()


def test_optimize_memory_large_object_monitoring(optimizer):
    opt, _, _ = optimizer
    config = {"monitor_large_objects": True}
    result = opt.optimize_memory(config, {"memory_usage": 10})
    assert "启用大对象监控" in result["actions"]


def test_optimize_memory_from_config_dataclass(optimizer, monkeypatch):
    opt, _, _ = optimizer
    monkeypatch.setattr(opt, "optimize_memory", MagicMock(return_value={"status": "applied"}))
    config = MemoryOptimizationConfig(
        enabled=True,
        gc_threshold=60,
        enable_pooling=True,
        monitor_large_objects=True,
    )

    result = opt.optimize_memory_from_config(config, {"memory_usage": 70})
    opt.optimize_memory.assert_called_once()
    assert result["status"] == "applied"


def test_optimize_memory_error_path(optimizer):
    opt, _, error_handler = optimizer

    class BadDict(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

    result = opt.optimize_memory(BadDict(), {"memory_usage": 90})
    error_handler.handle_error.assert_called_once()
    assert result["status"] == "failed"
    assert result["error"] == "boom"


def test_perform_garbage_collection_handles_exception(optimizer, monkeypatch):
    opt, logger, _ = optimizer
    def mock_collect():
        raise RuntimeError("gc fail")

    monkeypatch.setattr("gc.collect", mock_collect)
    collected = opt._perform_garbage_collection()
    assert collected == 0
    logger.log_warning.assert_called_once()

