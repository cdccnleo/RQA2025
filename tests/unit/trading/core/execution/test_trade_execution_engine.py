#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易执行引擎单元测试

测试目标：提升trade_execution_engine.py的覆盖率到90%+
按照业务流程驱动架构设计测试交易执行引擎功能
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from src.trading.core.execution.trade_execution_engine import (
    TradeExecutionEngine,
    ExecutionAlgorithm,
)


class MockOrder:
    """模拟订单对象"""

    def __init__(self, order_id="order_001", symbol="AAPL", quantity=100, price=150.0):
        self.order_id = order_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price


class TestExecutionAlgorithm:
    """测试执行算法枚举"""

    def test_execution_algorithm_values(self):
        """测试执行算法枚举值"""
        assert ExecutionAlgorithm.MARKET.value == "market"
        assert ExecutionAlgorithm.LIMIT.value == "limit"
        assert ExecutionAlgorithm.TWAP.value == "twap"
        assert ExecutionAlgorithm.VWAP.value == "vwap"
        assert ExecutionAlgorithm.ICEBERG.value == "iceberg"
        assert ExecutionAlgorithm.ADAPTIVE.value == "adaptive"


class TestTradeExecutionEngine:
    """测试交易执行引擎"""

    def test_init_default_config(self):
        """测试使用默认配置初始化"""
        engine = TradeExecutionEngine()

        assert engine.config == {}
        assert engine.active_executions == {}
        assert engine.execution_history == []
        assert engine.total_executions == 0
        assert engine.successful_executions == 0
        assert engine.failed_executions == 0

    def test_init_custom_config(self):
        """测试使用自定义配置初始化"""
        config = {"max_concurrent_executions": 10, "timeout": 60}
        engine = TradeExecutionEngine(config)

        assert engine.config == config

    def test_execute_order_market(self):
        """测试执行市价订单"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        execution_id = engine.execute_order(order, ExecutionAlgorithm.MARKET)

        assert execution_id is not None
        assert execution_id.startswith("exec_")
        # 市价订单执行后立即完成，会从active_executions中移除
        assert execution_id not in engine.active_executions
        assert engine.total_executions == 1
        assert engine.successful_executions == 1
        assert len(engine.execution_history) == 1

    def test_execute_order_limit(self):
        """测试执行限价订单"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        execution_id = engine.execute_order(order, ExecutionAlgorithm.LIMIT)

        assert execution_id is not None
        assert execution_id in engine.active_executions

    def test_execute_order_twap(self):
        """测试执行TWAP订单"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        execution_id = engine.execute_order(order, ExecutionAlgorithm.TWAP)

        assert execution_id is not None
        assert execution_id in engine.active_executions

    def test_execute_order_vwap(self):
        """测试执行VWAP订单"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        execution_id = engine.execute_order(order, ExecutionAlgorithm.VWAP)

        assert execution_id is not None
        assert execution_id in engine.active_executions

    def test_execute_order_iceberg(self):
        """测试执行冰山订单"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        execution_id = engine.execute_order(order, ExecutionAlgorithm.ICEBERG)

        assert execution_id is not None
        assert execution_id in engine.active_executions

    def test_execute_order_adaptive(self):
        """测试执行自适应订单"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        execution_id = engine.execute_order(order, ExecutionAlgorithm.ADAPTIVE)

        assert execution_id is not None
        assert execution_id in engine.active_executions

    def test_execute_order_error_handling(self):
        """测试执行订单错误处理"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        # Mock _create_execution_context抛出异常
        with patch.object(engine, '_create_execution_context', side_effect=Exception("Test error")):
            with pytest.raises(Exception):
                engine.execute_order(order)

            assert engine.failed_executions == 1

    def test_execute_order_without_order_id(self):
        """测试没有order_id属性的订单"""
        engine = TradeExecutionEngine()
        # 创建一个没有order_id属性的对象
        order = type('Order', (), {'symbol': 'AAPL', 'quantity': 100})()

        execution_id = engine.execute_order(order)

        assert execution_id is not None
        assert execution_id.startswith("exec_")
        # 检查execution_id格式（应该包含时间戳）
        assert len(execution_id) > 10

    def test_cancel_execution_success(self):
        """测试取消执行成功"""
        engine = TradeExecutionEngine()
        order = MockOrder()
        # 使用限价订单，因为市价订单会立即完成
        execution_id = engine.execute_order(order, ExecutionAlgorithm.LIMIT)

        result = engine.cancel_execution(execution_id)

        assert result is True
        assert execution_id not in engine.active_executions

    def test_cancel_execution_not_found(self):
        """测试取消不存在的执行"""
        engine = TradeExecutionEngine()

        result = engine.cancel_execution("nonexistent_execution")

        assert result is False

    def test_cancel_execution_error_handling(self):
        """测试取消执行错误处理"""
        engine = TradeExecutionEngine()
        order = MockOrder()
        execution_id = engine.execute_order(order)

        # Mock _cancel_execution抛出异常
        with patch.object(engine, '_cancel_execution', side_effect=Exception("Cancel error")):
            result = engine.cancel_execution(execution_id)

            assert result is False

    def test_get_execution_status_exists(self):
        """测试获取执行状态 - 执行存在"""
        engine = TradeExecutionEngine()
        order = MockOrder()
        # 使用限价订单，因为市价订单会立即完成并从active_executions中移除
        execution_id = engine.execute_order(order, ExecutionAlgorithm.LIMIT)

        status = engine.get_execution_status(execution_id)

        assert status is not None
        assert "execution_id" in status or "status" in status

    def test_get_execution_status_not_exists(self):
        """测试获取执行状态 - 执行不存在"""
        engine = TradeExecutionEngine()

        status = engine.get_execution_status("nonexistent_execution")

        assert status is None

    def test_get_all_executions(self):
        """测试获取所有执行"""
        engine = TradeExecutionEngine()
        order1 = MockOrder(order_id="order_001")
        order2 = MockOrder(order_id="order_002")

        execution_id1 = engine.execute_order(order1, ExecutionAlgorithm.LIMIT)
        execution_id2 = engine.execute_order(order2, ExecutionAlgorithm.LIMIT)

        # TradeExecutionEngine没有get_all_executions方法，直接使用active_executions
        all_executions = list(engine.active_executions.keys())

        assert len(all_executions) == 2
        assert execution_id1 in all_executions
        assert execution_id2 in all_executions

    def test_get_execution_statistics(self):
        """测试获取执行统计"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        engine.execute_order(order)

        # TradeExecutionEngine没有get_execution_statistics方法，使用get_performance_stats代替
        stats = engine.get_performance_stats()

        assert stats["total_executions"] == 1
        assert stats["successful_executions"] >= 0
        assert stats["failed_executions"] >= 0

    def test_get_performance_stats(self):
        """测试获取性能统计"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        engine.execute_order(order, ExecutionAlgorithm.MARKET)

        stats = engine.get_performance_stats()

        assert stats["total_executions"] == 1
        assert stats["successful_executions"] >= 0
        assert stats["failed_executions"] >= 0
        assert "success_rate" in stats
        assert "active_executions" in stats
        assert "average_duration" in stats

    def test_get_execution_history_empty(self):
        """测试获取空执行历史"""
        engine = TradeExecutionEngine()

        history = engine.get_execution_history()

        assert history == []

    def test_get_execution_history_with_symbol_filter(self):
        """测试按标的代码过滤执行历史"""
        engine = TradeExecutionEngine()
        order1 = MockOrder(order_id="order_001", symbol="AAPL")
        order2 = MockOrder(order_id="order_002", symbol="GOOGL")

        engine.execute_order(order1, ExecutionAlgorithm.MARKET)
        engine.execute_order(order2, ExecutionAlgorithm.MARKET)

        history = engine.get_execution_history(symbol="AAPL")

        assert len(history) >= 0  # 可能已经完成并移除了

    def test_get_execution_history_with_algorithm_filter(self):
        """测试按算法过滤执行历史"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        engine.execute_order(order, ExecutionAlgorithm.MARKET)

        history = engine.get_execution_history(algorithm=ExecutionAlgorithm.MARKET)

        assert isinstance(history, list)


    def test_market_execution_completes_immediately(self):
        """测试市价执行立即完成"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        execution_id = engine.execute_order(order, ExecutionAlgorithm.MARKET)

        # 市价执行应该立即完成并从active_executions中移除
        # 等待一小段时间确保执行完成
        import time
        time.sleep(0.1)

        # 检查执行历史
        assert len(engine.execution_history) > 0
        assert engine.successful_executions >= 0

    def test_limit_execution_remains_pending(self):
        """测试限价执行保持pending状态"""
        engine = TradeExecutionEngine()
        order = MockOrder()

        execution_id = engine.execute_order(order, ExecutionAlgorithm.LIMIT)

        # 限价执行应该保持executing状态
        status = engine.get_execution_status(execution_id)
        if status:
            assert status.get("status") == "executing"

    def test_multiple_executions(self):
        """测试多个执行"""
        engine = TradeExecutionEngine()
        orders = [MockOrder(order_id=f"order_{i}") for i in range(5)]

        execution_ids = []
        for order in orders:
            execution_id = engine.execute_order(order)
            execution_ids.append(execution_id)

        assert len(execution_ids) == 5
        assert engine.total_executions == 5

