import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


from concurrent.futures import ThreadPoolExecutor

import pytest

from src.data.integration.enhanced_data_integration_modules import components


def test_enhanced_parallel_loader_basic_flow():
    manager = components.create_enhanced_loader({"max_workers": 4})
    task = components.LoadTask(
        task_id="task_1",
        loader=lambda: "noop",
        start_date="2024-01-01",
        end_date="2024-01-31",
        priority=components.TaskPriority.HIGH,
        kwargs={"extra": True},
    )
    assert manager.submit_task(task) == "task_1"
    assert manager.execute_tasks(timeout=1) == {}


def test_dynamic_thread_pool_resize_and_utilization():
    pool = components.DynamicThreadPoolManager(initial_size=3, max_size=6, min_size=2)
    original_executor = pool.executor

    pool.resize(10)
    assert pool.get_current_size() == 6
    assert isinstance(pool.executor, ThreadPoolExecutor)
    assert pool.executor is not original_executor

    pool.resize(1)
    assert pool.get_current_size() == 2

    pool._utilization_history = [0.2, 0.4, 0.8]
    assert pool.get_utilization() == pytest.approx(14 / 30, rel=1e-6)

    pool._utilization_history = []
    assert pool.get_utilization() == 0.5
    assert pool.get_max_size() == 6

    pool.executor.shutdown(wait=False)


def test_connection_pool_manager_reuses_and_limits_connections():
    manager = components.ConnectionPoolManager(max_size=2, timeout=5)
    first = manager.get_connection()
    second = manager.get_connection()

    manager.return_connection(first)
    manager.return_connection(second)

    manager.return_connection("extra_connection")
    assert len(manager.connections) == 2

    reused = manager.get_connection()
    assert reused in {first, second, "extra_connection"}


def test_memory_and_financial_optimizers():
    optimizer = components.MemoryOptimizer(enable_compression=False, compression_level=3)
    optimizer.compress_cache_data(cache_strategy=object())

    compressing_optimizer = components.MemoryOptimizer(enable_compression=True, compression_level=5)
    compressing_optimizer.compress_cache_data(cache_strategy=object())

    financial_optimizer = components.FinancialDataOptimizer()
    result = financial_optimizer.optimize_financial_loading(["RQA", "ABC"], "2024-01-01", "2024-01-31")
    assert result["optimized_symbols"] == ["RQA", "ABC"]
    assert result["optimization_strategies"]["parallel_loading"] is True

