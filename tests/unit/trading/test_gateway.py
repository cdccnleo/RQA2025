# -*- coding: utf-8 -*-
"""
交易层 - 网关管理器单元测试
测试覆盖率目标: 95%+
按照业务流程驱动架构设计测试网关管理器核心功能
"""

import pytest
import random
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.trading.core.gateway import (
    GatewayManager, BaseGateway, FIXGateway, RESTGateway,
    GatewayStatus, ProtocolType, MarketType, AccountInfo
)


class MockGateway(BaseGateway):
    """模拟网关，用于测试"""

    def __init__(self, name: str):
        super().__init__(name)
        self.connected = False
        self.orders = {}
        self.order_counter = 0

    def connect(self, **kwargs):
        """模拟连接"""
        self.connected = True
        self.status = GatewayStatus.CONNECTED
        return True

    def disconnect(self):
        """模拟断开连接"""
        self.connected = False
        self.status = GatewayStatus.DISCONNECTED
        return True

    def send_order(self, order):
        """模拟发送订单"""
        self.order_counter += 1
        order_id = f"mock_order_{self.order_counter}"
        self.orders[order_id] = order
        return order_id

    def cancel_order(self, order_id: str):
        """模拟取消订单"""
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False

    def query_account(self):
        """模拟查询账户"""
        return AccountInfo(
            account_id="mock_account",
            balance=10000.0
        )

    def query_position(self, symbol: str = None):
        """模拟查询持仓"""
        if symbol:
            return {symbol: 100.0}
        return {"000001.SZ": 100.0, "000002.SZ": 50.0}

    def get_status(self):
        """模拟获取状态"""
        return self.status

    def get_active_orders(self):
        """模拟获取活跃订单"""
        return self.orders


class TestGatewayManager:
    """测试网关管理器"""

    def setup_method(self, method):
        """设置测试环境"""
        self.manager = GatewayManager()

    def test_init(self):
        """测试初始化"""
        assert isinstance(self.manager.gateways, dict)
        assert isinstance(self.manager.market_gateways, dict)
        assert isinstance(self.manager.gateway_loads, dict)
        assert isinstance(self.manager.gateway_health, dict)
        assert isinstance(self.manager.failover_history, list)

    def test_add_gateway(self):
        """测试添加网关"""
        gateway = MockGateway("test_gateway")

        # 添加网关到股票市场
        self.manager.add_gateway(gateway, [MarketType.STOCK])

        # 验证网关已被添加
        assert "test_gateway" in self.manager.gateways
        assert self.manager.gateways["test_gateway"] == gateway
        assert self.manager.gateway_loads["test_gateway"] == 0
        assert self.manager.gateway_health["test_gateway"] is True

        # 验证市场网关映射
        assert MarketType.STOCK in self.manager.market_gateways
        assert "test_gateway" in self.manager.market_gateways[MarketType.STOCK]

    def test_add_gateway_duplicate(self):
        """测试添加重复网关"""
        gateway1 = MockGateway("duplicate_gateway")
        gateway2 = MockGateway("duplicate_gateway")

        self.manager.add_gateway(gateway1, [MarketType.STOCK])

        # 尝试添加同名网关应该失败
        with pytest.raises(ValueError, match="already exists"):
            self.manager.add_gateway(gateway2, [MarketType.STOCK])

    def test_remove_gateway(self):
        """测试移除网关"""
        gateway = MockGateway("test_gateway")
        self.manager.add_gateway(gateway, [MarketType.STOCK])

        # 移除网关
        self.manager.remove_gateway("test_gateway")

        # 验证网关已被移除
        assert "test_gateway" not in self.manager.gateways
        assert "test_gateway" not in self.manager.gateway_loads
        assert "test_gateway" not in self.manager.gateway_health

    def test_remove_gateway_nonexistent(self):
        """测试移除不存在的网关"""
        # 移除不存在的网关应该不会出错
        self.manager.remove_gateway("nonexistent")
        assert True  # 如果没有异常抛出就算成功

    def test_get_gateway(self):
        """测试获取网关"""
        gateway = MockGateway("test_gateway")
        self.manager.add_gateway(gateway, [MarketType.STOCK])

        # 获取网关
        retrieved_gateway = self.manager.get_gateway("test_gateway")

        # 验证获取的网关正确
        assert retrieved_gateway == gateway

    def test_get_gateway_nonexistent(self):
        """测试获取不存在的网关"""
        retrieved_gateway = self.manager.get_gateway("nonexistent")
        assert retrieved_gateway is None

    def test_get_all_gateways(self):
        """测试获取所有网关"""
        gateway1 = MockGateway("gateway1")
        gateway2 = MockGateway("gateway2")

        self.manager.add_gateway(gateway1, [MarketType.STOCK])
        self.manager.add_gateway(gateway2, [MarketType.FUTURES])

        # 获取所有网关
        all_gateways = self.manager.get_all_gateways()

        # 验证所有网关都被返回
        assert len(all_gateways) == 2
        assert "gateway1" in all_gateways
        assert "gateway2" in all_gateways

    def test_send_order(self):
        """测试发送订单"""
        gateway = MockGateway("test_gateway")
        self.manager.add_gateway(gateway, [MarketType.STOCK])

        order = {
            "symbol": "000001.SZ",
            "quantity": 100,
            "price": 10.0
        }

        # 发送订单
        order_id = self.manager.send_order("test_gateway", order)

        # 验证订单已被发送
        assert order_id.startswith("mock_order_")

    def test_send_order_gateway_not_found(self):
        """测试通过不存在网关发送订单"""
        order = {"symbol": "000001.SZ", "quantity": 100}

        with pytest.raises(ValueError, match="not found"):
            self.manager.send_order("nonexistent_gateway", order)

    def test_cancel_order(self):
        """测试取消订单"""
        gateway = MockGateway("test_gateway")
        self.manager.add_gateway(gateway, [MarketType.STOCK])

        order = {"symbol": "000001.SZ", "quantity": 100}
        order_id = self.manager.send_order("test_gateway", order)

        # 取消订单
        result = self.manager.cancel_order("test_gateway", order_id)

        # 验证订单已被取消
        assert result is True

    def test_cancel_order_gateway_not_found(self):
        """测试通过不存在网关取消订单"""
        with pytest.raises(ValueError, match="not found"):
            self.manager.cancel_order("nonexistent_gateway", "order_123")

    def test_select_gateway_for_market(self):
        """测试为市场选择网关"""
        gateway1 = MockGateway("gateway1")
        gateway2 = MockGateway("gateway2")

        self.manager.add_gateway(gateway1, [MarketType.STOCK])
        self.manager.add_gateway(gateway2, [MarketType.STOCK])

        # 选择股票市场网关
        selected_gateway = self.manager.select_gateway_for_market(MarketType.STOCK)

        # 验证返回了有效的网关名称
        assert selected_gateway in ["gateway1", "gateway2"]

    def test_select_gateway_for_market_no_gateways(self):
        """测试为没有网关的市场选择网关"""
        selected_gateway = self.manager.select_gateway_for_market(MarketType.CRYPTO)
        assert selected_gateway is None

    def test_select_gateway_for_market_no_healthy_gateways(self):
        """测试为没有健康网关的市场选择网关"""
        gateway = MockGateway("gateway1")
        self.manager.add_gateway(gateway, [MarketType.STOCK])

        # 将网关标记为不健康
        self.manager.gateway_health["gateway1"] = False

        selected_gateway = self.manager.select_gateway_for_market(MarketType.STOCK)
        assert selected_gateway is None

    def test_select_gateway_by_load_balancing(self):
        """测试负载均衡网关选择"""
        gateway_names = ["gateway1", "gateway2", "gateway3"]

        # 设置不同的负载
        self.manager.gateway_loads["gateway1"] = 0  # 无负载
        self.manager.gateway_loads["gateway2"] = 5  # 中等负载
        self.manager.gateway_loads["gateway3"] = 10 # 高负载

        selected_gateway = self.manager._select_gateway_by_load_balancing(gateway_names)

        # 验证选择了有效的网关
        assert selected_gateway in gateway_names

    def test_select_gateway_by_load_balancing_single(self):
        """测试单个网关的负载均衡选择"""
        gateway_names = ["gateway1"]

        selected_gateway = self.manager._select_gateway_by_load_balancing(gateway_names)

        # 验证选择了唯一的网关
        assert selected_gateway == "gateway1"

    def test_get_gateways_for_market(self):
        """测试获取市场网关"""
        gateway1 = MockGateway("gateway1")
        gateway2 = MockGateway("gateway2")
        gateway3 = MockGateway("gateway3")

        self.manager.add_gateway(gateway1, [MarketType.STOCK])
        self.manager.add_gateway(gateway2, [MarketType.STOCK, MarketType.FUTURES])
        self.manager.add_gateway(gateway3, [MarketType.FUTURES])

        # 获取股票市场网关
        stock_gateways = self.manager.get_gateways_for_market(MarketType.STOCK)
        assert len(stock_gateways) == 2
        assert "gateway1" in stock_gateways
        assert "gateway2" in stock_gateways

        # 获取期货市场网关
        futures_gateways = self.manager.get_gateways_for_market(MarketType.FUTURES)
        assert len(futures_gateways) == 2
        assert "gateway2" in futures_gateways
        assert "gateway3" in futures_gateways

    def test_get_gateways_for_market_no_gateways(self):
        """测试获取没有网关的市场"""
        gateways = self.manager.get_gateways_for_market(MarketType.CRYPTO)
        assert len(gateways) == 0

    def test_update_gateway_load(self):
        """测试更新网关负载"""
        gateway = MockGateway("test_gateway")
        self.manager.add_gateway(gateway, [MarketType.STOCK])

        # 更新网关负载
        self.manager.update_gateway_load("test_gateway", 5)

        # 验证负载已被更新
        assert self.manager.gateway_loads["test_gateway"] == 5

    def test_update_gateway_load_nonexistent(self):
        """测试更新不存在网关的负载"""
        # 更新不存在网关的负载应该不会出错
        self.manager.update_gateway_load("nonexistent", 5)
        assert True  # 如果没有异常就算成功

    def test_get_gateway_health_status(self):
        """测试获取网关健康状态"""
        gateway = MockGateway("test_gateway")
        self.manager.add_gateway(gateway, [MarketType.STOCK])

        # 获取健康状态
        health_status = self.manager.get_gateway_health_status("test_gateway")

        # 验证健康状态
        assert health_status is True

    def test_get_gateway_health_status_nonexistent(self):
        """测试获取不存在网关的健康状态"""
        health_status = self.manager.get_gateway_health_status("nonexistent")
        assert health_status is None

    def test_get_gateway_load(self):
        """测试获取网关负载"""
        gateway = MockGateway("test_gateway")
        self.manager.add_gateway(gateway, [MarketType.STOCK])

        # 获取负载
        load = self.manager.get_gateway_load("test_gateway")

        # 验证负载
        assert load == 0

    def test_get_gateway_load_nonexistent(self):
        """测试获取不存在网关的负载"""
        load = self.manager.get_gateway_load("nonexistent")
        assert load is None

    def test_get_gateway_statistics(self):
        """测试获取网关统计信息"""
        gateway1 = MockGateway("gateway1")
        gateway2 = MockGateway("gateway2")

        self.manager.add_gateway(gateway1, [MarketType.STOCK])
        self.manager.add_gateway(gateway2, [MarketType.FUTURES])

        # 设置一些负载
        self.manager.update_gateway_load("gateway1", 3)
        self.manager.update_gateway_load("gateway2", 7)

        # 获取统计信息
        stats = self.manager.get_gateway_statistics()

        # 验证统计信息结构
        assert isinstance(stats, dict)
        assert "total_gateways" in stats
        assert "healthy_gateways" in stats
        assert "average_load" in stats
        assert stats["total_gateways"] == 2
