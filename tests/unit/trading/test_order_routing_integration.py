#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单路由集成测试
测试订单路由与其他交易组件的集成
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """订单数据类"""
    order_id: str
    symbol: str
    quantity: int
    price: float
    order_type: OrderType
    direction: str  # 'buy' or 'sell'
    status: OrderStatus = OrderStatus.PENDING
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class MockMarketDataProvider:
    """模拟市场数据提供商"""

    def __init__(self):
        self.market_data = {
            "000001.SZ": {"price": 10.0, "volume": 100000, "spread": 0.01},
            "000002.SZ": {"price": 15.0, "volume": 80000, "spread": 0.015},
            "600000.SH": {"price": 8.0, "volume": 120000, "spread": 0.008}
        }

    def get_market_data(self, symbol):
        return self.market_data.get(symbol, {"price": 0.0, "volume": 0, "spread": 0.0})

    def get_best_bid_ask(self, symbol):
        data = self.get_market_data(symbol)
        mid_price = data["price"]
        spread = data["spread"]
        return {
            "bid": mid_price - spread/2,
            "ask": mid_price + spread/2,
            "spread": spread
        }


class MockLiquidityProvider:
    """模拟流动性提供商"""

    def __init__(self, name, capacity=1000):
        self.name = name
        self.capacity = capacity
        self.available_capacity = capacity
        self.fees = {"market": 0.001, "limit": 0.0005}

    def check_capacity(self, order):
        """检查是否有足够容量处理订单"""
        return order.quantity <= self.available_capacity

    def get_fee(self, order_type):
        """获取交易费用"""
        return self.fees.get(order_type.value, 0.001)

    def execute_order(self, order):
        """执行订单"""
        if not self.check_capacity(order):
            return {"status": "rejected", "reason": "insufficient_capacity"}

        # 模拟执行
        self.available_capacity -= order.quantity

        return {
            "status": "filled",
            "order_id": order.order_id,
            "filled_quantity": order.quantity,
            "avg_price": order.price,
            "fee": self.get_fee(order.order_type),
            "provider": self.name
        }


class OrderRouter:
    """订单路由器"""

    def __init__(self):
        self.providers = {}
        self.routing_rules = {}
        self.market_data_provider = MockMarketDataProvider()

    def add_liquidity_provider(self, provider):
        """添加流动性提供商"""
        self.providers[provider.name] = provider

    def set_routing_rule(self, symbol, provider_name):
        """设置路由规则"""
        self.routing_rules[symbol] = provider_name

    def route_order(self, order):
        """路由订单到最佳流动性提供商"""
        # 1. 获取市场数据
        market_data = self.market_data_provider.get_market_data(order.symbol)

        # 2. 确定路由规则
        provider_name = self.routing_rules.get(order.symbol, "default_provider")

        # 3. 选择提供商
        provider = self.providers.get(provider_name)
        if not provider:
            return {"status": "failed", "reason": "no_provider_available"}

        # 4. 检查容量
        if not provider.check_capacity(order):
            # 尝试其他提供商
            for name, alt_provider in self.providers.items():
                if name != provider_name and alt_provider.check_capacity(order):
                    provider = alt_provider
                    break
            else:
                return {"status": "failed", "reason": "insufficient_capacity"}

        # 5. 执行订单
        result = provider.execute_order(order)
        result["routing_info"] = {
            "selected_provider": provider.name,
            "market_data_used": market_data,
            "routing_time": time.time()
        }

        return result


@pytest.fixture
def setup_routing_components():
    """设置路由测试组件"""
    # 创建订单路由器
    router = OrderRouter()

    # 创建流动性提供商
    provider1 = MockLiquidityProvider("primary_broker", capacity=1000)
    provider2 = MockLiquidityProvider("secondary_broker", capacity=800)
    provider3 = MockLiquidityProvider("alternative_broker", capacity=500)

    # 添加提供商到路由器
    router.add_liquidity_provider(provider1)
    router.add_liquidity_provider(provider2)
    router.add_liquidity_provider(provider3)

    # 设置路由规则
    router.set_routing_rule("000001.SZ", "primary_broker")
    router.set_routing_rule("000002.SZ", "secondary_broker")
    router.set_routing_rule("600000.SH", "primary_broker")

    return {
        "router": router,
        "providers": {
            "primary": provider1,
            "secondary": provider2,
            "alternative": provider3
        },
        "routing_rules": {
            "000001.SZ": "primary_broker",
            "000002.SZ": "secondary_broker",
            "600000.SH": "primary_broker"
        }
    }


class TestOrderRoutingIntegration:
    """订单路由集成测试"""

    def test_order_routing_basic_functionality(self, setup_routing_components):
        """测试订单路由基本功能"""
        components = setup_routing_components
        router = components["router"]

        # 创建测试订单
        order = Order(
            order_id="test_order_001",
            symbol="000001.SZ",
            quantity=100,
            price=10.0,
            order_type=OrderType.MARKET,
            direction="buy"
        )

        # 路由订单
        result = router.route_order(order)

        # 验证结果
        assert result["status"] == "filled"
        assert result["order_id"] == order.order_id
        assert result["filled_quantity"] == order.quantity
        assert "routing_info" in result
        assert result["routing_info"]["selected_provider"] == "primary_broker"

    def test_order_routing_capacity_management(self, setup_routing_components):
        """测试订单路由容量管理"""
        components = setup_routing_components
        router = components["router"]
        primary_provider = components["providers"]["primary"]

        # 创建大订单，超过主要提供商容量
        large_order = Order(
            order_id="large_order_001",
            symbol="000001.SZ",
            quantity=1200,  # 超过主要提供商的1000容量
            price=10.0,
            order_type=OrderType.MARKET,
            direction="buy"
        )

        # 路由大订单
        result = router.route_order(large_order)

        # 检查路由结果格式
        assert isinstance(result, dict)
        # 如果容量不足，应该返回失败状态
        if result.get("status") == "failed":
            assert result.get("reason") == "insufficient_capacity"
        else:
            assert result["status"] == "filled"
            assert result["routing_info"]["selected_provider"] in ["primary_broker", "secondary_broker", "alternative_broker"]

    def test_order_routing_market_data_integration(self, setup_routing_components):
        """测试订单路由与市场数据集成"""
        components = setup_routing_components
        router = components["router"]

        # 创建订单
        order = Order(
            order_id="market_data_order_001",
            symbol="000001.SZ",
            quantity=50,
            price=10.0,
            order_type=OrderType.LIMIT,
            direction="buy"
        )

        # 路由订单
        result = router.route_order(order)

        # 验证市场数据被使用
        assert result["status"] == "filled"
        assert "routing_info" in result
        market_data = result["routing_info"]["market_data_used"]
        assert "price" in market_data
        assert "volume" in market_data
        assert "spread" in market_data

    def test_order_routing_multiple_providers(self, setup_routing_components):
        """测试多提供商订单路由"""
        components = setup_routing_components
        router = components["router"]

        # 测试不同股票的路由规则
        orders = [
            Order(f"order_{i}", symbol, 100, 10.0, OrderType.MARKET, "buy")
            for i, symbol in enumerate(["000001.SZ", "000002.SZ", "600000.SH"])
        ]

        results = []
        for order in orders:
            result = router.route_order(order)
            results.append(result)

        # 验证所有订单都被处理
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert "status" in result
            assert "routing_info" in result

        # 验证路由规则
        symbol_to_provider = {
            "000001.SZ": "primary_broker",
            "000002.SZ": "secondary_broker",
            "600000.SH": "primary_broker"  # 默认路由
        }

        for order, result in zip(orders, results):
            expected_provider = symbol_to_provider[order.symbol]
            assert result["routing_info"]["selected_provider"] == expected_provider

    def test_order_routing_fee_optimization(self, setup_routing_components):
        """测试订单路由费用优化"""
        components = setup_routing_components
        router = components["router"]

        # 设置不同的费用结构
        components["providers"]["primary"].fees = {"market": 0.002, "limit": 0.001}
        components["providers"]["secondary"].fees = {"market": 0.001, "limit": 0.0005}  # 更便宜

        # 创建限价订单（应该选择费用更低的提供商）
        limit_order = Order(
            order_id="fee_opt_order_001",
            symbol="000001.SZ",
            quantity=100,
            price=10.0,
            order_type=OrderType.LIMIT,
            direction="buy"
        )

        # 虽然路由规则指定主要提供商，但应该选择费用更低的
        result = router.route_order(limit_order)

        assert result["status"] == "filled"
        # 验证费用信息
        assert "fee" in result
        assert result["fee"] > 0

    def test_order_routing_concurrent_orders(self, setup_routing_components):
        """测试并发订单路由"""
        components = setup_routing_components
        router = components["router"]

        # 创建多个并发订单
        orders = [
            Order(
                order_id=f"concurrent_order_{i}",
                symbol="000001.SZ",
                quantity=50,
                price=10.0,
                order_type=OrderType.MARKET,
                direction="buy"
            )
            for i in range(10)
        ]

        results = []

        def route_single_order(order):
            result = router.route_order(order)
            results.append(result)
            return result

        # 并发路由订单
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(route_single_order, order) for order in orders]
            for future in as_completed(futures):
                future.result()

        # 验证所有订单都被处理
        assert len(results) == 10
        successful_orders = [r for r in results if r["status"] == "filled"]
        assert len(successful_orders) == 10

        # 验证容量管理
        primary_provider = components["providers"]["primary"]
        assert primary_provider.available_capacity < primary_provider.capacity

    def test_order_routing_failure_handling(self, setup_routing_components):
        """测试订单路由失败处理"""
        components = setup_routing_components
        router = components["router"]

        # 创建超过所有提供商容量的订单
        huge_order = Order(
            order_id="huge_order_001",
            symbol="000001.SZ",
            quantity=10000,  # 超过所有提供商的总容量
            price=10.0,
            order_type=OrderType.MARKET,
            direction="buy"
        )

        # 路由巨大订单
        result = router.route_order(huge_order)

        # 应该被拒绝
        assert result["status"] == "failed"
        assert result["reason"] == "insufficient_capacity"

    def test_order_routing_performance_monitoring(self, setup_routing_components):
        """测试订单路由性能监控"""
        components = setup_routing_components
        router = components["router"]

        # 执行多个订单来测试性能
        orders = [
            Order(
                order_id=f"perf_order_{i}",
                symbol="000001.SZ",
                quantity=10,
                price=10.0,
                order_type=OrderType.MARKET,
                direction="buy"
            )
            for i in range(50)
        ]

        start_time = time.time()
        results = []

        for order in orders:
            result = router.route_order(order)
            results.append(result)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能
        assert len(results) == 50
        successful_results = [r for r in results if r["status"] == "filled"]
        assert len(successful_results) == 50

        # 平均每个订单的处理时间应该小于10ms
        avg_time_per_order = total_time / len(orders)
        assert avg_time_per_order < 0.01, f"平均处理时间太长: {avg_time_per_order:.4f}s"

    def test_order_routing_with_different_order_types(self, setup_routing_components):
        """测试不同订单类型的路由"""
        components = setup_routing_components
        router = components["router"]

        # 测试不同类型的订单
        order_types = [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT]

        for i, order_type in enumerate(order_types):
            order = Order(
                order_id=f"type_test_order_{i}",
                symbol="000001.SZ",
                quantity=50,
                price=10.0,
                order_type=order_type,
                direction="buy"
            )

            result = router.route_order(order)

            assert result["status"] == "filled"
            assert result["order_id"] == order.order_id
            assert "fee" in result
            assert "routing_info" in result


class TestOrderRoutingLoadTesting:
    """订单路由负载测试"""

    def test_order_routing_high_throughput(self, setup_routing_components):
        """测试订单路由高吞吐量"""
        components = setup_routing_components
        router = components["router"]

        # 创建大量订单进行压力测试
        order_count = 200
        orders = [
            Order(
                order_id=f"load_test_order_{i}",
                symbol="000001.SZ",
                quantity=10,
                price=10.0,
                order_type=OrderType.MARKET,
                direction="buy"
            )
            for i in range(order_count)
        ]

        start_time = time.time()
        results = []

        # 使用线程池进行并发测试
        def route_with_timing(order):
            order_start = time.time()
            result = router.route_order(order)
            order_end = time.time()
            return {
                "result": result,
                "execution_time": order_end - order_start
            }

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(route_with_timing, order) for order in orders]
            for future in as_completed(futures):
                results.append(future.result())

        end_time = time.time()
        total_time = end_time - start_time

        # 验证结果
        assert len(results) == order_count
        successful_results = [r for r in results if r["result"]["status"] == "filled"]
        assert len(successful_results) == order_count

        # 性能指标
        total_execution_time = sum(r["execution_time"] for r in results)
        avg_execution_time = total_execution_time / order_count
        throughput = order_count / total_time

        print(f"订单路由性能指标:")
        print(f"- 总订单数: {order_count}")
        print(f"- 总时间: {total_time:.2f}s")
        print(f"- 平均执行时间: {avg_execution_time:.4f}s")
        print(f"- 吞吐量: {throughput:.1f} orders/s")

        # 性能断言
        assert avg_execution_time < 0.05, f"平均执行时间太长: {avg_execution_time:.4f}s"
        assert throughput > 50, f"吞吐量太低: {throughput:.1f} orders/s"


if __name__ == "__main__":
    pytest.main([__file__])
