#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行上下文单元测试

测试目标：提升execution_context.py的覆盖率到90%+
按照业务流程驱动架构设计测试执行上下文功能
"""

import pytest
from datetime import datetime, timedelta

from src.trading.core.execution.execution_context import (
    ExecutionPhase,
    ExecutionContext,
)


class TestExecutionPhase:
    """测试执行阶段枚举"""

    def test_execution_phase_values(self):
        """测试执行阶段枚举值"""
        assert ExecutionPhase.PRE_EXECUTION.value == "pre_execution"
        assert ExecutionPhase.EXECUTING.value == "executing"
        assert ExecutionPhase.POST_EXECUTION.value == "post_execution"
        assert ExecutionPhase.COMPLETED.value == "completed"
        assert ExecutionPhase.FAILED.value == "failed"


class TestExecutionContext:
    """测试执行上下文"""

    def test_init_basic(self):
        """测试基本初始化"""
        context = ExecutionContext(
            execution_id="exec_001",
            symbol="AAPL"
        )

        assert context.execution_id == "exec_001"
        assert context.symbol == "AAPL"
        assert context.order_id is None
        assert context.quantity == 0.0
        assert context.price is None
        assert context.side == "buy"
        assert context.execution_strategy == "market"
        assert context.phase == ExecutionPhase.PRE_EXECUTION
        assert context.start_time is not None

    def test_init_with_all_fields(self):
        """测试包含所有字段的初始化"""
        context = ExecutionContext(
            execution_id="exec_002",
            symbol="AAPL",
            order_id="order_001",
            quantity=100.0,
            price=150.0,
            side="sell",
            execution_strategy="limit",
            time_limit=60,
            max_slippage=0.005
        )

        assert context.order_id == "order_001"
        assert context.quantity == 100.0
        assert context.price == 150.0
        assert context.side == "sell"
        assert context.execution_strategy == "limit"
        assert context.time_limit == 60
        assert context.max_slippage == 0.005

    def test_init_auto_start_time(self):
        """测试自动设置开始时间"""
        before = datetime.now()
        context = ExecutionContext(
            execution_id="exec_003",
            symbol="AAPL"
        )
        after = datetime.now()

        assert before <= context.start_time <= after

    def test_init_custom_start_time(self):
        """测试自定义开始时间"""
        custom_time = datetime(2024, 1, 1, 10, 0, 0)
        context = ExecutionContext(
            execution_id="exec_004",
            symbol="AAPL",
            start_time=custom_time
        )

        assert context.start_time == custom_time

    def test_update_progress(self):
        """测试更新执行进度"""
        context = ExecutionContext(
            execution_id="exec_005",
            symbol="AAPL",
            quantity=100.0
        )

        context.update_progress(50.0, 150.0)

        assert context.executed_quantity == 50.0
        assert context.executed_price == 150.0
        assert context.total_cost == 7500.0  # 50 * 150

    def test_update_progress_zero_price(self):
        """测试更新进度 - 零价格"""
        context = ExecutionContext(
            execution_id="exec_006",
            symbol="AAPL",
            quantity=100.0
        )

        context.update_progress(50.0, 0.0)

        assert context.executed_quantity == 50.0
        assert context.executed_price == 0.0
        assert context.total_cost == 0.0

    def test_update_progress_zero_quantity(self):
        """测试更新进度 - 零数量"""
        context = ExecutionContext(
            execution_id="exec_007",
            symbol="AAPL",
            quantity=100.0
        )

        context.update_progress(0.0, 150.0)

        assert context.executed_quantity == 0.0
        assert context.executed_price == 150.0
        assert context.total_cost == 0.0

    def test_mark_completed(self):
        """测试标记完成"""
        context = ExecutionContext(
            execution_id="exec_008",
            symbol="AAPL"
        )
        before = datetime.now()

        context.mark_completed()

        assert context.phase == ExecutionPhase.COMPLETED
        assert context.end_time is not None
        assert context.end_time >= before

    def test_mark_failed(self):
        """测试标记失败"""
        context = ExecutionContext(
            execution_id="exec_009",
            symbol="AAPL"
        )

        context.mark_failed("Test error")

        assert context.phase == ExecutionPhase.FAILED
        assert context.end_time is not None
        assert len(context.errors) == 1
        assert "Test error" in context.errors

    def test_mark_failed_multiple_errors(self):
        """测试标记失败 - 多个错误"""
        context = ExecutionContext(
            execution_id="exec_010",
            symbol="AAPL"
        )

        context.mark_failed("Error 1")
        context.mark_failed("Error 2")

        assert len(context.errors) == 2
        assert "Error 1" in context.errors
        assert "Error 2" in context.errors

    def test_is_completed_true(self):
        """测试检查完成 - 已完成"""
        context = ExecutionContext(
            execution_id="exec_011",
            symbol="AAPL"
        )
        context.mark_completed()

        assert context.is_completed() is True

    def test_is_completed_failed(self):
        """测试检查完成 - 已失败"""
        context = ExecutionContext(
            execution_id="exec_012",
            symbol="AAPL"
        )
        context.mark_failed("Error")

        assert context.is_completed() is True

    def test_is_completed_false(self):
        """测试检查完成 - 未完成"""
        context = ExecutionContext(
            execution_id="exec_013",
            symbol="AAPL"
        )

        assert context.is_completed() is False

    def test_get_execution_summary(self):
        """测试获取执行摘要"""
        context = ExecutionContext(
            execution_id="exec_014",
            symbol="AAPL",
            quantity=100.0
        )
        context.update_progress(80.0, 150.0)
        context.mark_completed()

        summary = context.get_execution_summary()

        assert summary["execution_id"] == "exec_014"
        assert summary["symbol"] == "AAPL"
        assert summary["total_quantity"] == 100.0
        assert summary["executed_quantity"] == 80.0
        assert summary["execution_rate"] == 0.8
        assert summary["average_price"] == 150.0
        assert summary["total_cost"] == 12000.0
        assert summary["phase"] == ExecutionPhase.COMPLETED.value
        assert summary["duration"] is not None
        assert summary["errors"] == []

    def test_get_execution_summary_zero_quantity(self):
        """测试获取执行摘要 - 零数量"""
        context = ExecutionContext(
            execution_id="exec_015",
            symbol="AAPL",
            quantity=0.0
        )

        summary = context.get_execution_summary()

        assert summary["execution_rate"] == 0.0

    def test_get_execution_summary_no_end_time(self):
        """测试获取执行摘要 - 无结束时间"""
        context = ExecutionContext(
            execution_id="exec_016",
            symbol="AAPL"
        )

        summary = context.get_execution_summary()

        assert summary["duration"] is None

    def test_metadata(self):
        """测试元数据"""
        context = ExecutionContext(
            execution_id="exec_017",
            symbol="AAPL"
        )

        context.metadata["key1"] = "value1"
        context.metadata["key2"] = 123

        assert context.metadata["key1"] == "value1"
        assert context.metadata["key2"] == 123

