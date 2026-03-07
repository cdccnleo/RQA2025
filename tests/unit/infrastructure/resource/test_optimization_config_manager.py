import pytest
from unittest.mock import MagicMock

from src.infrastructure.resource.core.optimization_config_manager import OptimizationConfigManager
from src.infrastructure.resource.core.optimization_config import ParallelizationConfig, CheckpointingConfig


@pytest.fixture
def manager():
    logger = MagicMock()
    error_handler = MagicMock()
    mgr = OptimizationConfigManager(logger=logger, error_handler=error_handler)
    return mgr, logger, error_handler


def test_configure_parallelization_defaults(manager):
    mgr, logger, _ = manager
    result = mgr.configure_parallelization({})

    assert result["status"] == "applied"
    assert "设置线程池大小为 4" in result["actions"]
    assert logger.log_info.called is False


def test_configure_parallelization_with_full_config(manager):
    mgr, _, _ = manager
    config = {
        "thread_pool_size": 16,
        "process_pool_size": 8,
        "async_enabled": True,
    }

    result = mgr.configure_parallelization(config)

    assert "设置线程池大小为 16" in result["actions"]
    assert "设置进程池大小为 8" in result["actions"]
    assert "启用异步处理" in result["actions"]


def test_configure_parallelization_from_dataclass(manager):
    mgr, _, _ = manager
    dataclass_config = ParallelizationConfig(thread_pool_size=6, process_pool_size=3, async_enabled=True)

    result = mgr.configure_parallelization_from_config(dataclass_config)

    assert "设置线程池大小为 6" in result["actions"]
    assert "设置进程池大小为 3" in result["actions"]
    assert "启用异步处理" in result["actions"]


def test_configure_parallelization_error_path(manager):
    mgr, _, error_handler = manager

    class BadDict(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("boom")

    result = mgr.configure_parallelization(BadDict())

    error_handler.handle_error.assert_called_once()
    assert result["status"] == "failed"
    assert result["type"] == "parallelization_config"
    assert result["error"] == "boom"


def test_configure_checkpointing_defaults(manager):
    mgr, _, _ = manager
    result = mgr.configure_checkpointing({})

    assert result["status"] == "applied"
    assert "设置检查点间隔为 300 秒" in result["actions"]


def test_configure_checkpointing_with_full_config(manager):
    mgr, _, _ = manager
    config = {
        "interval_seconds": 120,
        "storage_path": "/tmp/checkpoints",
        "compression": {"enabled": True},
    }

    result = mgr.configure_checkpointing(config)

    assert "设置检查点间隔为 120 秒" in result["actions"]
    assert "设置检查点存储路径: /tmp/checkpoints" in result["actions"]
    assert "启用检查点压缩" in result["actions"]


def test_configure_checkpointing_from_dataclass(manager):
    mgr, _, _ = manager
    dataclass_config = CheckpointingConfig(
        interval_seconds=90,
        storage_path="/data",
        compression={"enabled": True},
    )

    result = mgr.configure_checkpointing_from_config(dataclass_config)

    assert "设置检查点间隔为 90 秒" in result["actions"]
    assert "设置检查点存储路径: /data" in result["actions"]
    assert "启用检查点压缩" in result["actions"]


def test_configure_checkpointing_error_path(manager):
    mgr, _, error_handler = manager

    class BadDict(dict):
        def get(self, *args, **kwargs):
            raise ValueError("checkpoint fail")

    result = mgr.configure_checkpointing(BadDict())

    error_handler.handle_error.assert_called_once()
    assert result["status"] == "failed"
    assert result["type"] == "checkpointing_config"
    assert result["error"] == "checkpoint fail"

