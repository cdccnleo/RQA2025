#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试交易引擎 - 简化版本

测试目标：提升trading_engine.py的核心功能覆盖率
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.trading.core.trading_engine import (
    OrderType,
    OrderDirection,
    OrderStatus,
    ChinaMarketAdapter
)


class TestOrderEnums:
    """测试订单枚举"""

    def test_order_type_values(self):
        """测试订单类型枚举值"""
        assert OrderType.MARKET.value == 1
        assert OrderType.LIMIT.value == 2
        assert OrderType.STOP.value == 3

    def test_order_direction_values(self):
        """测试订单方向枚举值"""
        assert OrderDirection.BUY.value == 1
        assert OrderDirection.SELL.value == -1

    def test_order_status_values(self):
        """测试订单状态枚举值"""
        assert OrderStatus.PENDING.value == 1
        assert OrderStatus.PARTIAL.value == 2
        assert OrderStatus.FILLED.value == 3
        assert OrderStatus.CANCELLED.value == 4
        assert OrderStatus.REJECTED.value == 5


class TestChinaMarketAdapter:
    """测试中国市场适配器"""

    def test_check_trade_restrictions_basic(self):
        """测试基本交易限制检查"""
        # 正常情况
        assert ChinaMarketAdapter.check_trade_restrictions("000001", 10.0, 9.9) == True

        # 价格波动过大
        assert ChinaMarketAdapter.check_trade_restrictions("000001", 15.0, 10.0) == False

    def test_check_t1_restriction(self):
        """测试T+1限制检查"""
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        t2 = today + timedelta(days=2)

        # 同一天不能卖出
        assert ChinaMarketAdapter.check_t1_restriction(today, today) == False
        # 次日可以卖出
        assert ChinaMarketAdapter.check_t1_restriction(today, tomorrow) == True
        # T+2及以后可以卖出
        assert ChinaMarketAdapter.check_t1_restriction(today, t2) == True

    def test_calculate_fees_buy_order(self):
        """测试买入订单费用计算"""
        order = {
            "quantity": 1000,
            "price": 10.0,
            "direction": OrderDirection.BUY
        }

        fees = ChinaMarketAdapter.calculate_fees(order, is_a_stock=True)
        # 买入订单：佣金5元 + 过户费0.1元 = 5.1元
        assert abs(fees - 5.1) < 0.01

    def test_calculate_fees_sell_order(self):
        """测试卖出订单费用计算"""
        order = {
            "quantity": 1000,
            "price": 10.0,
            "direction": OrderDirection.SELL
        }

        fees = ChinaMarketAdapter.calculate_fees(order, is_a_stock=True)
        # 卖出订单：印花税10元 + 佣金5元 + 过户费0.1元 = 15.1元
        assert abs(fees - 15.1) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
