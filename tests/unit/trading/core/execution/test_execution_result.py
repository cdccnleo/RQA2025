#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行结果单元测试

测试目标：提升execution_result.py的覆盖率到90%+
按照业务流程驱动架构设计测试执行结果功能
"""

import pytest
from datetime import datetime, timedelta

from src.trading.core.execution.execution_result import (
    ExecutionResultStatus,
    ExecutionResult,
)


class TestExecutionResultStatus:
    """测试执行结果状态枚举"""

    def test_execution_result_status_values(self):
        """测试执行结果状态枚举值"""
        assert ExecutionResultStatus.SUCCESS.value == "success"
        assert ExecutionResultStatus.PARTIAL.value == "partial"
        assert ExecutionResultStatus.FAILED.value == "failed"
        assert ExecutionResultStatus.CANCELLED.value == "cancelled"


class TestExecutionResult:
    """测试执行结果"""

    def test_init_basic(self):
        """测试基本初始化"""
        result = ExecutionResult(
            execution_id="exec_001",
            symbol="AAPL"
        )

        assert result.execution_id == "exec_001"
        assert result.symbol == "AAPL"
        assert result.order_id is None
        assert result.status == ExecutionResultStatus.SUCCESS
        assert result.requested_quantity == 0.0
        assert result.executed_quantity == 0.0
        assert result.start_time is not None

    def test_init_with_all_fields(self):
        """测试包含所有字段的初始化"""
        result = ExecutionResult(
            execution_id="exec_002",
            symbol="AAPL",
            order_id="order_001",
            status=ExecutionResultStatus.PARTIAL,
            requested_quantity=100.0,
            executed_quantity=80.0,
            average_price=150.0,
            total_cost=12000.0
        )

        assert result.order_id == "order_001"
        assert result.status == ExecutionResultStatus.PARTIAL
        assert result.requested_quantity == 100.0
        assert result.executed_quantity == 80.0
        assert result.average_price == 150.0
        assert result.total_cost == 12000.0

    def test_init_auto_start_time(self):
        """测试自动设置开始时间"""
        before = datetime.now()
        result = ExecutionResult(
            execution_id="exec_003",
            symbol="AAPL"
        )
        after = datetime.now()

        assert before <= result.start_time <= after

    def test_add_trade(self):
        """测试添加交易记录"""
        result = ExecutionResult(
            execution_id="exec_004",
            symbol="AAPL",
            requested_quantity=100.0
        )

        result.add_trade({"quantity": 50.0, "price": 150.0})
        result.add_trade({"quantity": 30.0, "price": 151.0})

        assert len(result.trades) == 2
        assert result.executed_quantity == 80.0
        assert result.total_cost == pytest.approx(50.0 * 150.0 + 30.0 * 151.0, rel=1e-6)
        assert result.average_price == pytest.approx(result.total_cost / 80.0, rel=1e-6)

    def test_add_trade_single(self):
        """测试添加单个交易记录"""
        result = ExecutionResult(
            execution_id="exec_005",
            symbol="AAPL",
            requested_quantity=100.0
        )

        result.add_trade({"quantity": 100.0, "price": 150.0})

        assert len(result.trades) == 1
        assert result.executed_quantity == 100.0
        assert result.average_price == 150.0
        assert result.total_cost == 15000.0

    def test_add_trade_zero_quantity(self):
        """测试添加零数量交易"""
        result = ExecutionResult(
            execution_id="exec_006",
            symbol="AAPL"
        )

        result.add_trade({"quantity": 0.0, "price": 150.0})

        assert result.executed_quantity == 0.0
        assert result.average_price is None

    def test_calculate_metrics_full_execution(self):
        """测试计算指标 - 完全执行"""
        result = ExecutionResult(
            execution_id="exec_007",
            symbol="AAPL",
            requested_quantity=100.0
        )
        result.add_trade({"quantity": 100.0, "price": 150.0})
        result.end_time = result.start_time + timedelta(seconds=5)

        metrics = result.calculate_metrics()

        assert metrics["execution_rate"] == 1.0
        assert metrics["execution_time"] == 5.0
        assert metrics["average_price"] == 150.0
        assert metrics["total_cost"] == 15000.0
        assert result.status == ExecutionResultStatus.SUCCESS

    def test_calculate_metrics_partial_execution(self):
        """测试计算指标 - 部分执行"""
        result = ExecutionResult(
            execution_id="exec_008",
            symbol="AAPL",
            requested_quantity=100.0
        )
        result.add_trade({"quantity": 80.0, "price": 150.0})
        result.end_time = result.start_time + timedelta(seconds=10)

        metrics = result.calculate_metrics()

        assert metrics["execution_rate"] == 0.8
        assert result.status == ExecutionResultStatus.PARTIAL

    def test_calculate_metrics_zero_requested(self):
        """测试计算指标 - 零请求数量"""
        result = ExecutionResult(
            execution_id="exec_009",
            symbol="AAPL",
            requested_quantity=0.0
        )

        metrics = result.calculate_metrics()

        assert metrics["execution_rate"] == 0.0

    def test_calculate_metrics_no_end_time(self):
        """测试计算指标 - 无结束时间"""
        result = ExecutionResult(
            execution_id="exec_010",
            symbol="AAPL",
            requested_quantity=100.0
        )
        result.add_trade({"quantity": 50.0, "price": 150.0})

        metrics = result.calculate_metrics()

        assert metrics["execution_time"] == 0.0

    def test_mark_completed_success(self):
        """测试标记完成 - 成功"""
        result = ExecutionResult(
            execution_id="exec_011",
            symbol="AAPL"
        )

        result.mark_completed(ExecutionResultStatus.SUCCESS)

        assert result.status == ExecutionResultStatus.SUCCESS
        assert result.end_time is not None

    def test_mark_completed_partial(self):
        """测试标记完成 - 部分执行"""
        result = ExecutionResult(
            execution_id="exec_012",
            symbol="AAPL"
        )

        result.mark_completed(ExecutionResultStatus.PARTIAL)

        assert result.status == ExecutionResultStatus.PARTIAL
        assert result.end_time is not None

    def test_add_error(self):
        """测试添加错误"""
        result = ExecutionResult(
            execution_id="exec_013",
            symbol="AAPL"
        )

        result.add_error("Test error")

        assert len(result.errors) == 1
        assert "Test error" in result.errors
        assert result.status == ExecutionResultStatus.FAILED

    def test_add_error_multiple(self):
        """测试添加多个错误"""
        result = ExecutionResult(
            execution_id="exec_014",
            symbol="AAPL"
        )

        result.add_error("Error 1")
        result.add_error("Error 2")

        assert len(result.errors) == 2
        assert result.status == ExecutionResultStatus.FAILED

    def test_add_warning(self):
        """测试添加警告"""
        result = ExecutionResult(
            execution_id="exec_015",
            symbol="AAPL"
        )

        result.add_warning("Test warning")

        assert len(result.warnings) == 1
        assert "Test warning" in result.warnings

    def test_add_warning_multiple(self):
        """测试添加多个警告"""
        result = ExecutionResult(
            execution_id="exec_016",
            symbol="AAPL"
        )

        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        assert len(result.warnings) == 2

    def test_get_summary(self):
        """测试获取结果摘要"""
        result = ExecutionResult(
            execution_id="exec_017",
            symbol="AAPL",
            requested_quantity=100.0
        )
        result.add_trade({"quantity": 80.0, "price": 150.0})
        result.end_time = result.start_time + timedelta(seconds=5)
        result.add_warning("Test warning")

        summary = result.get_summary()

        assert summary["execution_id"] == "exec_017"
        assert summary["symbol"] == "AAPL"
        assert summary["status"] == ExecutionResultStatus.PARTIAL.value
        assert summary["execution_rate"] == 0.8
        assert summary["average_price"] == 150.0
        assert summary["total_cost"] == 12000.0
        assert summary["execution_time"] == 5.0
        assert summary["trade_count"] == 1
        assert len(summary["warnings"]) == 1

    def test_is_successful_true(self):
        """测试检查是否成功 - 成功"""
        result = ExecutionResult(
            execution_id="exec_018",
            symbol="AAPL"
        )
        result.status = ExecutionResultStatus.SUCCESS

        assert result.is_successful() is True

    def test_is_successful_with_errors(self):
        """测试检查是否成功 - 有错误"""
        result = ExecutionResult(
            execution_id="exec_019",
            symbol="AAPL"
        )
        result.status = ExecutionResultStatus.SUCCESS
        result.add_error("Error")

        assert result.is_successful() is False

    def test_is_successful_failed_status(self):
        """测试检查是否成功 - 失败状态"""
        result = ExecutionResult(
            execution_id="exec_020",
            symbol="AAPL"
        )
        result.status = ExecutionResultStatus.FAILED

        assert result.is_successful() is False

    def test_is_partial_true(self):
        """测试检查是否部分执行 - 是"""
        result = ExecutionResult(
            execution_id="exec_021",
            symbol="AAPL"
        )
        result.status = ExecutionResultStatus.PARTIAL

        assert result.is_partial() is True

    def test_is_partial_false(self):
        """测试检查是否部分执行 - 否"""
        result = ExecutionResult(
            execution_id="exec_022",
            symbol="AAPL"
        )
        result.status = ExecutionResultStatus.SUCCESS

        assert result.is_partial() is False

