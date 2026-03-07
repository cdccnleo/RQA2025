# -*- coding: utf-8 -*-
"""
ExecutionEngine边界条件和异常处理测试
补充Phase 31.3深度测试覆盖
"""

import pytest
from unittest.mock import Mock, patch

from src.trading.execution.execution_engine import ExecutionEngine, ExecutionMode


class TestExecutionEngineBoundaryConditions:
    """ExecutionEngine边界条件测试"""

    @pytest.fixture
    def execution_engine(self):
        """执行引擎实例"""
        return ExecutionEngine()

    def test_create_execution_quantity_too_large(self, execution_engine):
        """测试订单数量过大"""
        # 假设MAX_POSITION_SIZE = 1000000
        large_quantity = 2000000

        with pytest.raises(ValueError, match="订单数量过大"):
            execution_engine.create_execution(
                symbol="TEST",
                side="buy",
                quantity=large_quantity,
                price=100.0,
                mode=ExecutionMode.MARKET
            )

    def test_create_execution_price_too_large(self, execution_engine):
        """测试价格数值异常"""
        # 假设MAX_POSITION_SIZE = 1000000
        extreme_price = 2000000

        with pytest.raises(ValueError, match="价格数值异常"):
            execution_engine.create_execution(
                symbol="TEST",
                side="buy",
                quantity=100,
                price=extreme_price,
                mode=ExecutionMode.LIMIT
            )

    def test_create_execution_invalid_price_type(self, execution_engine):
        """测试无效价格类型"""
        with pytest.raises(ValueError, match="价格必须为正数"):
            execution_engine.create_execution(
                symbol="TEST",
                side="buy",
                quantity=100,
                price="invalid_price",
                mode=ExecutionMode.LIMIT
            )

    def test_create_execution_negative_price(self, execution_engine):
        """测试负价格"""
        with pytest.raises(ValueError, match="价格必须为正数"):
            execution_engine.create_execution(
                symbol="TEST",
                side="buy",
                quantity=100,
                price=-10.0,
                mode=ExecutionMode.LIMIT
            )

    def test_create_execution_zero_price(self, execution_engine):
        """测试零价格"""
        with pytest.raises(ValueError, match="价格必须为正数"):
            execution_engine.create_execution(
                symbol="TEST",
                side="buy",
                quantity=100,
                price=0.0,
                mode=ExecutionMode.LIMIT
            )

    def test_create_execution_limit_order_without_price(self, execution_engine):
        """测试限价单缺少价格"""
        with pytest.raises(ValueError, match="限价单必须指定价格"):
            execution_engine.create_execution(
                symbol="TEST",
                side="buy",
                quantity=100,
                price=None,
                mode=ExecutionMode.LIMIT
            )

    def test_create_execution_zero_quantity(self, execution_engine):
        """测试零数量"""
        with pytest.raises(ValueError, match="数量必须为正数"):
            execution_engine.create_execution(
                symbol="TEST",
                side="buy",
                quantity=0,
                price=100.0,
                mode=ExecutionMode.MARKET
            )

    def test_create_execution_negative_quantity(self, execution_engine):
        """测试负数量"""
        with pytest.raises(ValueError, match="数量必须为正数"):
            execution_engine.create_execution(
                symbol="TEST",
                side="buy",
                quantity=-100,
                price=100.0,
                mode=ExecutionMode.MARKET
            )

    def test_create_execution_empty_symbol(self, execution_engine):
        """测试空交易代码"""
        with pytest.raises(ValueError, match="交易标不能为空"):
            execution_engine.create_execution(
                symbol="",
                side="buy",
                quantity=100,
                price=100.0,
                mode=ExecutionMode.MARKET
            )

    def test_create_execution_valid_side(self, execution_engine):
        """测试有效交易方向"""
        # 测试买入方向
        execution_id_buy = execution_engine.create_execution(
            symbol="TEST",
            side="buy",
            quantity=100,
            price=100.0,
            mode=ExecutionMode.MARKET
        )
        assert execution_id_buy.startswith("exec_")

        # 测试卖出方向
        execution_id_sell = execution_engine.create_execution(
            symbol="TEST",
            side="sell",
            quantity=100,
            price=100.0,
            mode=ExecutionMode.MARKET
        )
        assert execution_id_sell.startswith("exec_")

    def test_get_execution_status_nonexistent(self, execution_engine):
        """测试获取不存在的执行状态"""
        result = execution_engine.get_execution_status("nonexistent_id")
        assert result is None

    def test_get_execution_status_dict_nonexistent(self, execution_engine):
        """测试获取不存在的执行状态字典"""
        result = execution_engine.get_execution_status_dict("nonexistent_id")
        assert result is None

    def test_cancel_execution_nonexistent(self, execution_engine):
        """测试取消不存在的执行"""
        result = execution_engine.cancel_execution("nonexistent_id")
        assert result is False

    def test_cancel_execution_already_completed(self, execution_engine):
        """测试取消已完成的执行"""
        # 先创建一个执行
        execution_id = execution_engine.create_execution(
            symbol="TEST",
            side="buy",
            quantity=100,
            price=100.0,
            mode=ExecutionMode.MARKET
        )

        # 模拟执行完成
        execution_engine.update_execution_status(execution_id, "completed")

        # 尝试取消已完成的执行
        result = execution_engine.cancel_execution(execution_id)
        assert result is False

    def test_update_execution_status_valid_status(self, execution_engine):
        """测试更新有效的执行状态"""
        # 先创建一个执行
        execution_id = execution_engine.create_execution(
            symbol="TEST",
            side="buy",
            quantity=100,
            price=100.0,
            mode=ExecutionMode.MARKET
        )

        # 更新为有效状态
        result = execution_engine.update_execution_status(execution_id, "completed")
        assert result is True

    def test_get_all_executions_empty(self, execution_engine):
        """测试获取所有执行（空列表）"""
        all_executions = execution_engine.get_all_executions()
        assert all_executions == []

    def test_get_executions_empty(self, execution_engine):
        """测试获取执行列表（空列表）"""
        executions = execution_engine.get_executions()
        assert executions == []

    def test_get_execution_statistics_empty(self, execution_engine):
        """测试获取执行统计（空统计）"""
        stats = execution_engine.get_execution_statistics()
        expected_stats = {
            'total_executions': 0,
            'completed_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'pending_executions': 0,
            'success_rate': 0.0,
            'symbol_performance': {}
        }
        assert stats == expected_stats

    def test_get_execution_queue_status_empty(self, execution_engine):
        """测试获取执行队列状态（空队列）"""
        queue_status = execution_engine.get_execution_queue_status()
        expected_status = {
            'total_orders': 0,
            'queued_orders': 0,
            'running_orders': 0,
            'active_executions': 0,
            'completed_orders': 0,
            'failed_orders': 0,
            'cancelled_orders': 0,
            'queue_utilization': 0.0,
            'avg_queue_time': 0.5,
            'max_queue_time': 2.1
        }
        assert queue_status == expected_status


class TestExecutionEngineConcurrency:
    """ExecutionEngine并发测试"""

    @pytest.fixture
    def execution_engine(self):
        """执行引擎实例"""
        return ExecutionEngine()

    def test_concurrent_execution_creation(self, execution_engine):
        """测试并发创建执行"""
        import threading
        import time

        results = []
        errors = []

        def create_execution_worker(worker_id):
            try:
                execution_id = execution_engine.create_execution(
                    symbol=f"TEST_{worker_id}",
                    side="buy",
                    quantity=100,
                    price=100.0,
                    mode=ExecutionMode.MARKET
                )
                results.append(execution_id)
            except Exception as e:
                errors.append(str(e))

        # 创建10个并发线程
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_execution_worker, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 10  # 所有执行都成功创建
        assert len(errors) == 0  # 没有错误
        assert len(set(results)) == 10  # 所有执行ID都唯一

    def test_execution_status_updates_concurrent(self, execution_engine):
        """测试并发执行状态更新"""
        import threading

        # 先创建多个执行
        execution_ids = []
        for i in range(5):
            execution_id = execution_engine.create_execution(
                symbol=f"TEST_{i}",
                side="buy",
                quantity=100,
                price=100.0,
                mode=ExecutionMode.MARKET
            )
            execution_ids.append(execution_id)

        results = []
        errors = []

        def update_status_worker(execution_id, status):
            try:
                result = execution_engine.update_execution_status(execution_id, status)
                results.append((execution_id, status, result))
            except Exception as e:
                errors.append(str(e))

        # 创建并发状态更新线程
        threads = []
        statuses = ["running", "completed", "failed", "cancelled", "rejected"]

        for i, execution_id in enumerate(execution_ids):
            thread = threading.Thread(
                target=update_status_worker,
                args=(execution_id, statuses[i])
            )
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 5  # 所有状态更新都成功
        assert len(errors) == 0  # 没有错误

        # 验证统计数据
        stats = execution_engine.get_execution_statistics()
        assert stats['total_executions'] == 5
        assert stats['completed_executions'] >= 1


class TestExecutionEnginePerformance:
    """ExecutionEngine性能测试"""

    @pytest.fixture
    def execution_engine(self):
        """执行引擎实例"""
        return ExecutionEngine()

    def test_bulk_execution_creation_performance(self, execution_engine):
        """测试批量创建执行的性能"""
        import time

        num_executions = 1000
        start_time = time.time()

        execution_ids = []
        for i in range(num_executions):
            execution_id = execution_engine.create_execution(
                symbol=f"BULK_TEST_{i}",
                side="buy",
                quantity=100,
                price=100.0,
                mode=ExecutionMode.MARKET
            )
            execution_ids.append(execution_id)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能要求：1000个执行创建应该在1秒内完成
        assert total_time < 1.0
        assert len(execution_ids) == num_executions
        assert len(set(execution_ids)) == num_executions  # 所有ID都唯一

        # 验证统计数据
        stats = execution_engine.get_execution_statistics()
        assert stats['total_executions'] == num_executions

    def test_execution_status_query_performance(self, execution_engine):
        """测试执行状态查询性能"""
        import time

        # 先创建100个执行
        execution_ids = []
        for i in range(100):
            execution_id = execution_engine.create_execution(
                symbol=f"PERF_TEST_{i}",
                side="buy",
                quantity=100,
                price=100.0,
                mode=ExecutionMode.MARKET
            )
            execution_ids.append(execution_id)

        # 测试查询性能
        start_time = time.time()

        for _ in range(1000):  # 1000次查询
            random_id = execution_ids[0]  # 查询第一个执行
            status = execution_engine.get_execution_status(random_id)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证查询性能：1000次查询应该在0.1秒内完成
        assert total_time < 0.1
        assert status is not None

    def test_memory_usage_with_many_executions(self, execution_engine):
        """测试大量执行时的内存使用"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 创建大量执行
        num_executions = 5000
        execution_ids = []

        for i in range(num_executions):
            execution_id = execution_engine.create_execution(
                symbol=f"MEM_TEST_{i}",
                side="buy",
                quantity=100,
                price=100.0,
                mode=ExecutionMode.MARKET
            )
            execution_ids.append(execution_id)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 验证内存使用：5000个执行不应导致过多的内存增长（< 50MB）
        assert memory_increase < 50 * 1024 * 1024  # 50MB in bytes
        assert len(execution_ids) == num_executions

        # 清理测试数据
        execution_engine.executions.clear()
        execution_engine.execution_id_counter = 0
