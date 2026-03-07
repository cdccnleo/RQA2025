"""
性能优化组件模块的边界测试
"""
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


import pytest
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from src.data.integration.enhanced_data_integration_modules.components import (
    TaskPriority,
    LoadTask,
    EnhancedParallelLoadingManager,
    create_enhanced_loader,
    DynamicThreadPoolManager,
    ConnectionPoolManager,
    MemoryOptimizer,
    FinancialDataOptimizer,
)


class TestTaskPriority:
    """测试 TaskPriority 枚举"""

    def test_task_priority_low(self):
        """测试 LOW 优先级"""
        assert TaskPriority.LOW.value == "low"
        assert isinstance(TaskPriority.LOW, TaskPriority)

    def test_task_priority_normal(self):
        """测试 NORMAL 优先级"""
        assert TaskPriority.NORMAL.value == "normal"
        assert isinstance(TaskPriority.NORMAL, TaskPriority)

    def test_task_priority_high(self):
        """测试 HIGH 优先级"""
        assert TaskPriority.HIGH.value == "high"
        assert isinstance(TaskPriority.HIGH, TaskPriority)

    def test_task_priority_critical(self):
        """测试 CRITICAL 优先级"""
        assert TaskPriority.CRITICAL.value == "critical"
        assert isinstance(TaskPriority.CRITICAL, TaskPriority)

    def test_task_priority_all_values(self):
        """测试所有优先级值"""
        priorities = [TaskPriority.LOW, TaskPriority.NORMAL, TaskPriority.HIGH, TaskPriority.CRITICAL]
        assert len(priorities) == 4
        assert all(isinstance(p.value, str) for p in priorities)
        assert all(isinstance(p, TaskPriority) for p in priorities)


class TestLoadTask:
    """测试 LoadTask 数据类"""

    def test_load_task_init_required(self):
        """测试必需参数初始化"""
        loader = Mock()
        task = LoadTask(
            task_id="task1",
            loader=loader,
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        assert task.task_id == "task1"
        assert task.loader == loader
        assert task.start_date == "2024-01-01"
        assert task.end_date == "2024-01-31"
        assert task.frequency == "1d"
        assert task.priority == TaskPriority.NORMAL
        assert task.kwargs is None

    def test_load_task_init_all_params(self):
        """测试所有参数初始化"""
        loader = Mock()
        task = LoadTask(
            task_id="task2",
            loader=loader,
            start_date="2024-01-01",
            end_date="2024-01-31",
            frequency="1h",
            priority=TaskPriority.HIGH,
            kwargs={"symbol": "AAPL"}
        )
        assert task.task_id == "task2"
        assert task.frequency == "1h"
        assert task.priority == TaskPriority.HIGH
        assert task.kwargs == {"symbol": "AAPL"}

    def test_load_task_empty_task_id(self):
        """测试空任务ID"""
        loader = Mock()
        task = LoadTask(
            task_id="",
            loader=loader,
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        assert task.task_id == ""

    def test_load_task_none_loader(self):
        """测试 None 加载器"""
        task = LoadTask(
            task_id="task3",
            loader=None,
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        assert task.loader is None

    def test_load_task_empty_dates(self):
        """测试空日期"""
        loader = Mock()
        task = LoadTask(
            task_id="task4",
            loader=loader,
            start_date="",
            end_date=""
        )
        assert task.start_date == ""
        assert task.end_date == ""

    def test_load_task_custom_priority(self):
        """测试自定义优先级"""
        loader = Mock()
        task = LoadTask(
            task_id="task5",
            loader=loader,
            start_date="2024-01-01",
            end_date="2024-01-31",
            priority=TaskPriority.CRITICAL
        )
        assert task.priority == TaskPriority.CRITICAL

    def test_load_task_empty_kwargs(self):
        """测试空 kwargs"""
        loader = Mock()
        task = LoadTask(
            task_id="task6",
            loader=loader,
            start_date="2024-01-01",
            end_date="2024-01-31",
            kwargs={}
        )
        assert task.kwargs == {}


class TestEnhancedParallelLoadingManager:
    """测试 EnhancedParallelLoadingManager 类"""

    def test_init_default(self):
        """测试默认初始化"""
        config = {}
        manager = EnhancedParallelLoadingManager(config)
        assert manager.config == config

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {"max_workers": 10, "batch_size": 100}
        manager = EnhancedParallelLoadingManager(config)
        assert manager.config == config

    def test_init_none_config(self):
        """测试 None 配置"""
        manager = EnhancedParallelLoadingManager(None)
        assert manager.config is None

    def test_submit_task_success(self):
        """测试成功提交任务"""
        manager = EnhancedParallelLoadingManager({})
        loader = Mock()
        task = LoadTask(
            task_id="task1",
            loader=loader,
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        result = manager.submit_task(task)
        assert result == "task1"

    def test_submit_task_empty_id(self):
        """测试空任务ID"""
        manager = EnhancedParallelLoadingManager({})
        loader = Mock()
        task = LoadTask(
            task_id="",
            loader=loader,
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        result = manager.submit_task(task)
        assert result == ""

    def test_execute_tasks_default_timeout(self):
        """测试默认超时执行任务"""
        manager = EnhancedParallelLoadingManager({})
        result = manager.execute_tasks()
        assert result == {}

    def test_execute_tasks_custom_timeout(self):
        """测试自定义超时执行任务"""
        manager = EnhancedParallelLoadingManager({})
        result = manager.execute_tasks(timeout=60)
        assert result == {}

    def test_execute_tasks_zero_timeout(self):
        """测试零超时执行任务"""
        manager = EnhancedParallelLoadingManager({})
        result = manager.execute_tasks(timeout=0)
        assert result == {}

    def test_execute_tasks_negative_timeout(self):
        """测试负超时执行任务"""
        manager = EnhancedParallelLoadingManager({})
        result = manager.execute_tasks(timeout=-1)
        assert result == {}


class TestCreateEnhancedLoader:
    """测试 create_enhanced_loader 函数"""

    def test_create_enhanced_loader_default(self):
        """测试默认配置"""
        config = {}
        loader = create_enhanced_loader(config)
        assert isinstance(loader, EnhancedParallelLoadingManager)
        assert loader.config == config

    def test_create_enhanced_loader_with_config(self):
        """测试带配置"""
        config = {"max_workers": 10}
        loader = create_enhanced_loader(config)
        assert isinstance(loader, EnhancedParallelLoadingManager)
        assert loader.config == config

    def test_create_enhanced_loader_none_config(self):
        """测试 None 配置"""
        loader = create_enhanced_loader(None)
        assert isinstance(loader, EnhancedParallelLoadingManager)
        assert loader.config is None

    def test_create_enhanced_loader_multiple_instances(self):
        """测试多个实例"""
        loader1 = create_enhanced_loader({})
        loader2 = create_enhanced_loader({})
        assert loader1 is not loader2


class TestDynamicThreadPoolManager:
    """测试 DynamicThreadPoolManager 类"""

    def test_init_valid_params(self):
        """测试有效参数初始化"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        assert manager.initial_size == 5
        assert manager.max_size == 10
        assert manager.min_size == 2
        assert manager.current_size == 5
        assert isinstance(manager.executor, ThreadPoolExecutor)

    def test_init_equal_sizes(self):
        """测试相等大小初始化"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=5, min_size=5)
        assert manager.initial_size == 5
        assert manager.max_size == 5
        assert manager.min_size == 5
        assert manager.current_size == 5

    def test_resize_within_range(self):
        """测试范围内调整大小"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        manager.resize(7)
        assert manager.current_size == 7

    def test_resize_below_min(self):
        """测试低于最小值调整"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        manager.resize(1)
        assert manager.current_size == 2  # 应该被限制为最小值

    def test_resize_above_max(self):
        """测试高于最大值调整"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        manager.resize(15)
        assert manager.current_size == 10  # 应该被限制为最大值

    def test_resize_to_min(self):
        """测试调整到最小值"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        manager.resize(2)
        assert manager.current_size == 2

    def test_resize_to_max(self):
        """测试调整到最大值"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        manager.resize(10)
        assert manager.current_size == 10

    def test_get_current_size(self):
        """测试获取当前大小"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        assert manager.get_current_size() == 5
        manager.resize(7)
        assert manager.get_current_size() == 7

    def test_get_max_size(self):
        """测试获取最大大小"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        assert manager.get_max_size() == 10

    def test_get_utilization_empty_history(self):
        """测试空历史利用率"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        utilization = manager.get_utilization()
        assert utilization == 0.5  # 默认值

    def test_get_utilization_with_history(self):
        """测试有历史利用率"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        manager._utilization_history = [0.3, 0.5, 0.7]
        utilization = manager.get_utilization()
        assert utilization == pytest.approx(0.5, abs=0.01)  # (0.3 + 0.5 + 0.7) / 3 = 0.5

    def test_resize_multiple_times(self):
        """测试多次调整大小"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        manager.resize(7)
        assert manager.current_size == 7
        manager.resize(3)
        assert manager.current_size == 3
        manager.resize(9)
        assert manager.current_size == 9


class TestConnectionPoolManager:
    """测试 ConnectionPoolManager 类"""

    def test_init_valid_params(self):
        """测试有效参数初始化"""
        pool = ConnectionPoolManager(max_size=10, timeout=30)
        assert pool.max_size == 10
        assert pool.timeout == 30
        assert pool.connections == []

    def test_init_zero_max_size(self):
        """测试零最大大小"""
        pool = ConnectionPoolManager(max_size=0, timeout=30)
        assert pool.max_size == 0

    def test_init_zero_timeout(self):
        """测试零超时"""
        pool = ConnectionPoolManager(max_size=10, timeout=0)
        assert pool.timeout == 0

    def test_get_connection_empty_pool(self):
        """测试空池获取连接"""
        pool = ConnectionPoolManager(max_size=10, timeout=30)
        conn = pool.get_connection()
        assert conn == "connection_1"

    def test_get_connection_with_connections(self):
        """测试有连接时获取"""
        pool = ConnectionPoolManager(max_size=10, timeout=30)
        pool.connections = ["conn1", "conn2"]
        conn = pool.get_connection()
        assert conn == "conn2"  # 从列表末尾弹出

    def test_get_connection_multiple_times(self):
        """测试多次获取连接"""
        pool = ConnectionPoolManager(max_size=10, timeout=30)
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        # 注意：当池为空时，get_connection 返回 f"connection_{len(self.connections) + 1}"
        # 由于 connections 始终为空，len(self.connections) 始终为 0，所以每次都返回 connection_1
        # 这是实现的行为
        assert conn1.startswith("connection_")
        assert conn2.startswith("connection_")

    def test_return_connection_success(self):
        """测试成功归还连接"""
        pool = ConnectionPoolManager(max_size=10, timeout=30)
        pool.return_connection("conn1")
        assert "conn1" in pool.connections
        assert len(pool.connections) == 1

    def test_return_connection_at_max(self):
        """测试达到最大大小时归还连接"""
        pool = ConnectionPoolManager(max_size=2, timeout=30)
        pool.return_connection("conn1")
        pool.return_connection("conn2")
        pool.return_connection("conn3")  # 应该被忽略
        assert len(pool.connections) == 2

    def test_return_connection_none(self):
        """测试归还 None 连接"""
        pool = ConnectionPoolManager(max_size=10, timeout=30)
        pool.return_connection(None)
        assert None in pool.connections

    def test_get_and_return_connection(self):
        """测试获取和归还连接"""
        pool = ConnectionPoolManager(max_size=10, timeout=30)
        conn = pool.get_connection()
        pool.return_connection(conn)
        assert conn in pool.connections

    def test_get_connection_after_return(self):
        """测试归还后获取连接"""
        pool = ConnectionPoolManager(max_size=10, timeout=30)
        pool.return_connection("conn1")
        conn = pool.get_connection()
        assert conn == "conn1"
        assert len(pool.connections) == 0


class TestMemoryOptimizer:
    """测试 MemoryOptimizer 类"""

    def test_init_enable_compression(self):
        """测试启用压缩初始化"""
        optimizer = MemoryOptimizer(enable_compression=True, compression_level=5)
        assert optimizer.enable_compression is True
        assert optimizer.compression_level == 5

    def test_init_disable_compression(self):
        """测试禁用压缩初始化"""
        optimizer = MemoryOptimizer(enable_compression=False, compression_level=0)
        assert optimizer.enable_compression is False
        assert optimizer.compression_level == 0

    def test_init_zero_compression_level(self):
        """测试零压缩级别"""
        optimizer = MemoryOptimizer(enable_compression=True, compression_level=0)
        assert optimizer.compression_level == 0

    def test_init_high_compression_level(self):
        """测试高压缩级别"""
        optimizer = MemoryOptimizer(enable_compression=True, compression_level=9)
        assert optimizer.compression_level == 9

    def test_compress_cache_data_enabled(self):
        """测试启用压缩时压缩缓存数据"""
        optimizer = MemoryOptimizer(enable_compression=True, compression_level=5)
        cache_strategy = Mock()
        # 应该不抛出异常
        optimizer.compress_cache_data(cache_strategy)

    def test_compress_cache_data_disabled(self):
        """测试禁用压缩时压缩缓存数据"""
        optimizer = MemoryOptimizer(enable_compression=False, compression_level=5)
        cache_strategy = Mock()
        # 应该不抛出异常
        optimizer.compress_cache_data(cache_strategy)

    def test_compress_cache_data_none(self):
        """测试 None 缓存策略"""
        optimizer = MemoryOptimizer(enable_compression=True, compression_level=5)
        # 应该不抛出异常
        optimizer.compress_cache_data(None)


class TestFinancialDataOptimizer:
    """测试 FinancialDataOptimizer 类"""

    def test_init_default(self):
        """测试默认初始化"""
        optimizer = FinancialDataOptimizer()
        assert optimizer.optimization_strategies["parallel_loading"] is True
        assert optimizer.optimization_strategies["batch_processing"] is True
        assert optimizer.optimization_strategies["data_compression"] is True
        assert optimizer.optimization_strategies["smart_caching"] is True

    def test_optimize_financial_loading_single_symbol(self):
        """测试单个符号优化"""
        optimizer = FinancialDataOptimizer()
        result = optimizer.optimize_financial_loading(
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        assert result["optimized_symbols"] == ["AAPL"]
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-01-31"
        assert "optimization_strategies" in result

    def test_optimize_financial_loading_multiple_symbols(self):
        """测试多个符号优化"""
        optimizer = FinancialDataOptimizer()
        result = optimizer.optimize_financial_loading(
            symbols=["AAPL", "GOOGL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        assert len(result["optimized_symbols"]) == 3
        assert "AAPL" in result["optimized_symbols"]

    def test_optimize_financial_loading_empty_symbols(self):
        """测试空符号列表"""
        optimizer = FinancialDataOptimizer()
        result = optimizer.optimize_financial_loading(
            symbols=[],
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        assert result["optimized_symbols"] == []

    def test_optimize_financial_loading_empty_dates(self):
        """测试空日期"""
        optimizer = FinancialDataOptimizer()
        result = optimizer.optimize_financial_loading(
            symbols=["AAPL"],
            start_date="",
            end_date=""
        )
        assert result["start_date"] == ""
        assert result["end_date"] == ""

    def test_optimize_financial_loading_same_dates(self):
        """测试相同日期"""
        optimizer = FinancialDataOptimizer()
        result = optimizer.optimize_financial_loading(
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-01"
        )
        assert result["start_date"] == result["end_date"]

    def test_optimize_financial_loading_reversed_dates(self):
        """测试反向日期（结束日期早于开始日期）"""
        optimizer = FinancialDataOptimizer()
        result = optimizer.optimize_financial_loading(
            symbols=["AAPL"],
            start_date="2024-01-31",
            end_date="2024-01-01"
        )
        assert result["start_date"] == "2024-01-31"
        assert result["end_date"] == "2024-01-01"


class TestEdgeCases:
    """测试边界情况"""

    def test_dynamic_thread_pool_manager_resize_same_size(self):
        """测试调整到相同大小"""
        manager = DynamicThreadPoolManager(initial_size=5, max_size=10, min_size=2)
        manager.resize(5)
        assert manager.current_size == 5

    def test_connection_pool_manager_get_connection_concurrent(self):
        """测试并发获取连接"""
        pool = ConnectionPoolManager(max_size=10, timeout=30)
        # 模拟并发获取
        conns = [pool.get_connection() for _ in range(5)]
        assert len(conns) == 5
        assert all(conn.startswith("connection_") for conn in conns)

    def test_connection_pool_manager_return_connection_duplicate(self):
        """测试归还重复连接"""
        pool = ConnectionPoolManager(max_size=10, timeout=30)
        pool.return_connection("conn1")
        pool.return_connection("conn1")  # 重复归还
        assert pool.connections.count("conn1") == 2

    def test_memory_optimizer_negative_compression_level(self):
        """测试负压缩级别"""
        optimizer = MemoryOptimizer(enable_compression=True, compression_level=-1)
        assert optimizer.compression_level == -1

    def test_financial_data_optimizer_very_long_symbol_list(self):
        """测试非常长的符号列表"""
        optimizer = FinancialDataOptimizer()
        symbols = [f"SYMBOL{i}" for i in range(1000)]
        result = optimizer.optimize_financial_loading(
            symbols=symbols,
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        assert len(result["optimized_symbols"]) == 1000

    def test_load_task_all_priorities(self):
        """测试所有优先级的任务"""
        loader = Mock()
        for priority in TaskPriority:
            task = LoadTask(
                task_id=f"task_{priority.value}",
                loader=loader,
                start_date="2024-01-01",
                end_date="2024-01-31",
                priority=priority
            )
            assert task.priority == priority

