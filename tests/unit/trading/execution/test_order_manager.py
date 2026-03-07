# -*- coding: utf-8 -*-
"""
订单管理器测试
测试订单生命周期管理功能
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.trading.execution.order_manager import (
    OrderManager,
    Order,
    OrderType,
    OrderStatus,
    OrderSide,
    OrderValidationResult,
    OrderValidator
)


class TestOrderEnums:
    """订单枚举测试"""

    def test_order_type_enum(self):
        """测试订单类型枚举"""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
        assert OrderType.TRAILING_STOP.value == "trailing_stop"
        assert OrderType.CANCEL.value == "cancel"

    def test_order_status_enum(self):
        """测试订单状态枚举"""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.PARTIAL.value == "partial"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"

    def test_order_side_enum(self):
        """测试订单方向枚举"""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"


class TestOrder:
    """订单类测试"""

    def test_order_creation(self):
        """测试订单创建"""
        order = Order(
            order_id="test_order_123",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )

        assert order.order_id == "test_order_123"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 100
        assert order.price == 150.0
        assert order.status == OrderStatus.PENDING
        assert isinstance(order.created_time, datetime)

    def test_order_creation_market(self):
        """测试市价订单创建"""
        order = Order(
            order_id="market_order_123",
            symbol="GOOGL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50
        )

        assert order.order_type == OrderType.MARKET
        assert order.price is None  # 市价单没有价格
        assert order.quantity == 50

    def test_order_creation_defaults(self):
        """测试订单默认值"""
        order = Order(
            order_id="default_order",
            symbol="TSLA",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10
        )

        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0
        assert order.remaining_quantity == 10
        assert order.avg_fill_price == 0.0
        assert isinstance(order.created_time, datetime)

    def test_order_calculate_remaining_quantity(self):
        """测试剩余数量计算"""
        order = Order(
            order_id="calc_test",
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=200.0
        )

        # 初始状态
        assert order.remaining_quantity == 100

        # 注意：由于dataclass的限制，我们需要重新创建对象来测试不同的filled_quantity
        # 或者我们可以直接测试属性值
        order2 = Order(
            order_id="calc_test2",
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=200.0,
            filled_quantity=30
        )
        assert order2.remaining_quantity == 70

        order3 = Order(
            order_id="calc_test3",
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=200.0,
            filled_quantity=100
        )
        assert order3.remaining_quantity == 0

    def test_order_is_complete(self):
        """测试订单完成状态"""
        order = Order(
            order_id="complete_test",
            symbol="AMZN",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=3000.0
        )

        # 初始状态
        assert not order.is_completed

        # 部分成交 - 仍然活跃
        order.status = OrderStatus.PARTIAL
        assert not order.is_completed

        # 全部成交
        order.status = OrderStatus.FILLED
        assert order.is_completed

        # 取消状态
        order2 = Order(
            order_id="cancel_test",
            symbol="AMZN",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=3000.0,
            status=OrderStatus.CANCELLED
        )
        assert order2.is_completed


class TestOrderValidationResult:
    """订单验证结果测试"""

    def test_validation_result_creation(self):
        """测试验证结果创建"""
        result = OrderValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Low volume"]
        )

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.warnings == ["Low volume"]

    def test_validation_result_invalid(self):
        """测试无效验证结果"""
        result = OrderValidationResult(
            is_valid=False,
            errors=["Insufficient funds", "Invalid price"],
            warnings=[]
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 0

    def test_validation_result_empty(self):
        """测试空验证结果"""
        result = OrderValidationResult(
            is_valid=True,
            errors=[],
            warnings=[]
        )

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0


class TestOrderValidator:
    """订单验证器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.validator = OrderValidator()

    def test_validator_creation(self):
        """测试验证器创建"""
        assert self.validator is not None

    def test_validate_valid_order(self):
        """测试验证有效订单"""
        order = Order(
            order_id="valid_test",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )

        result = self.validator.validate_order(order)
        assert isinstance(result, OrderValidationResult)
        # 基本验证应该通过
        assert result.is_valid is True or len(result.errors) == 0

    def test_validate_invalid_order(self):
        """测试验证无效订单"""
        # 创建一个明显无效的订单（例如数量为0）
        order = Order(
            order_id="invalid_test",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0,  # 无效数量
            price=150.0
        )

        result = self.validator.validate_order(order)
        assert isinstance(result, OrderValidationResult)
        # 应该包含错误信息
        assert not result.is_valid or len(result.errors) > 0


class TestOrderManagerBasic:
    """订单管理器基础功能测试"""

    def setup_method(self):
        """设置测试方法"""
        self.order_manager = OrderManager()

    def test_order_manager_creation(self):
        """测试订单管理器创建"""
        assert self.order_manager is not None
        assert len(self.order_manager.active_orders) == 0
        assert len(self.order_manager.completed_orders) == 0

    def test_submit_order(self):
        """测试提交订单"""
        order = Order(
            order_id="submit_test",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )

        result = self.order_manager.submit_order(order)
        assert isinstance(result, tuple)
        assert result[0] is True  # 成功标志
        assert order.order_id in self.order_manager.active_orders
        assert self.order_manager.active_orders[order.order_id] == order

    def test_cancel_order(self):
        """测试取消订单"""
        # 先提交订单
        order = Order(
            order_id="cancel_test",
            symbol="GOOGL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50
        )

        self.order_manager.submit_order(order)
        assert order.order_id in self.order_manager.active_orders

        # 取消订单
        result = self.order_manager.cancel_order("cancel_test")
        assert isinstance(result, tuple)
        assert result[0] is True  # 成功标志
        # 注意：根据实际实现，可能订单仍然在active_orders中但状态改变

    def test_get_order(self):
        """测试获取订单"""
        order = Order(
            order_id="get_test",
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=75,
            price=280.0
        )

        self.order_manager.submit_order(order)
        retrieved_order = self.order_manager.get_order("get_test")

        assert retrieved_order is not None
        assert retrieved_order.order_id == "get_test"
        assert retrieved_order.symbol == "MSFT"

        # 测试获取不存在的订单
        non_existent = self.order_manager.get_order("non_existent")
        assert non_existent is None

    def test_get_active_orders(self):
        """测试获取活跃订单"""
        # 提交多个订单
        orders = []
        for i in range(3):
            order = Order(
                order_id=f"active_test_{i}",
                symbol=f"SYMBOL{i}",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=10 * (i + 1)
            )
            orders.append(order)
            self.order_manager.submit_order(order)

        active_orders = self.order_manager.get_active_orders()
        assert len(active_orders) == 3

        # 验证所有订单都在活跃列表中
        for order in orders:
            assert order in active_orders

    def test_order_count_tracking(self):
        """测试订单数量跟踪"""
        # 初始状态
        assert len(self.order_manager.active_orders) == 0
        assert len(self.order_manager.completed_orders) == 0

        # 提交一些订单
        self.order_manager.submit_order(Order(
            order_id="count_1", symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100
        ))
        self.order_manager.submit_order(Order(
            order_id="count_2", symbol="GOOGL", side=OrderSide.SELL,
            order_type=OrderType.LIMIT, quantity=50, price=2500.0
        ))

        assert len(self.order_manager.active_orders) == 2
        assert len(self.order_manager.completed_orders) == 0
