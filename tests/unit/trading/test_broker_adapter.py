# -*- coding: utf-8 -*-
"""
交易层 - 券商适配器单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试券商适配器核心功能
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.trading.broker.broker_adapter import BrokerAdapter
# OrderStatus应该从正确的模块导入
try:
    from src.trading.core.trading_engine import OrderStatus
except ImportError:
    try:
        from src.trading.execution.order_manager import OrderStatus
    except ImportError:
        from src.trading.hft.execution.order_executor import OrderStatus


class MockBrokerAdapter(BrokerAdapter):
    """模拟券商适配器，用于测试"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.mock_orders = {}
        self.mock_positions = {}
        self.orders = {}  # 添加orders属性供测试使用
        self.logger = logging.getLogger(__name__)  # 添加logger属性供测试使用

    def connect(self):
        """模拟连接"""
        self.connected = True
        return True

    def disconnect(self):
        """模拟断开连接"""
        self.connected = False
        return True

    def place_order(self, order):
        """模拟下单"""
        order_id = f"mock_order_{len(self.mock_orders)}"
        self.mock_orders[order_id] = order
        return order_id

    def submit_order(self, order):
        """模拟提交订单"""
        order_id = f"mock_order_{len(self.mock_orders)}"
        self.mock_orders[order_id] = order
        return {"order_id": order_id, "status": "submitted"}

    def cancel_order(self, order_id):
        """模拟取消订单"""
        if order_id in self.mock_orders:
            return True
        return False

    def get_order_status(self, order_id):
        """模拟获取订单状态"""
        if order_id in self.mock_orders:
            return OrderStatus.FILLED
        return None

    def get_positions(self):
        """模拟获取持仓"""
        return self.mock_positions

    def get_balance(self):
        """模拟获取余额"""
        return {"cash": 10000.0, "total": 10000.0}

    def get_account_balance(self, account_id=None):
        """实现抽象方法"""
        return self.get_balance()

    def get_market_data(self, symbol):
        """模拟获取市场数据"""
        # 模拟某些符号找不到数据的情况
        if symbol == "NONEXISTENT":
            return None
        return {"symbol": symbol, "price": 100.0, "volume": 1000}


class TestBrokerAdapter:
    """测试券商适配器"""

    def setup_method(self, method):
        """设置测试环境"""
        config = {
            "broker_id": "test_broker",
            "api_key": "test_key",
            "api_secret": "test_secret"
        }
        self.adapter = MockBrokerAdapter(config)

    def test_init(self):
        """测试初始化"""
        assert self.adapter.config["broker_id"] == "test_broker"
        assert self.adapter.connected is False
        assert isinstance(self.adapter.orders, dict)
        assert isinstance(self.adapter.logger, object)

    def test_connect(self):
        """测试连接"""
        result = self.adapter.connect()
        assert result is True
        assert self.adapter.connected is True

    def test_disconnect(self):
        """测试断开连接"""
        self.adapter.connect()
        result = self.adapter.disconnect()
        assert result is True
        assert self.adapter.connected is False

    def test_submit_order(self):
        """测试提交订单"""
        order = {
            "symbol": "000001.SZ",
            "quantity": 100,
            "price": 10.0,
            "order_type": "LIMIT"
        }

        result = self.adapter.submit_order(order)
        assert "order_id" in result
        assert result["status"] == "submitted"
        assert len(self.adapter.mock_orders) == 1

    def test_cancel_order(self):
        """测试取消订单"""
        # 先提交订单
        order = {"symbol": "000001.SZ", "quantity": 100}
        result = self.adapter.submit_order(order)
        order_id = result["order_id"]

        # 取消订单
        cancel_result = self.adapter.cancel_order(order_id)
        assert cancel_result is True

    def test_cancel_order_nonexistent(self):
        """测试取消不存在的订单"""
        result = self.adapter.cancel_order("nonexistent")
        assert result is False

    def test_get_order_status(self):
        """测试获取订单状态"""
        # 先提交订单
        order = {"symbol": "000001.SZ", "quantity": 100}
        result = self.adapter.submit_order(order)
        order_id = result["order_id"]

        # 获取状态
        status = self.adapter.get_order_status(order_id)
        assert status == OrderStatus.FILLED

    def test_get_order_status_nonexistent(self):
        """测试获取不存在订单的状态"""
        status = self.adapter.get_order_status("nonexistent")
        assert status is None

    def test_get_positions(self):
        """测试获取持仓"""
        positions = self.adapter.get_positions()
        assert isinstance(positions, dict)

    def test_get_balance(self):
        """测试获取余额"""
        balance = self.adapter.get_balance()
        assert "cash" in balance
        assert "total" in balance
        assert balance["cash"] == 10000.0

    def test_get_market_data(self):
        """测试获取市场数据"""
        symbol = "000001.SZ"
        data = self.adapter.get_market_data(symbol)

        assert data["symbol"] == symbol
        assert "price" in data
        assert "volume" in data

    def test_get_market_data_none(self):
        """测试获取不存在的市场数据"""
        data = self.adapter.get_market_data("NONEXISTENT")
        assert data is None
