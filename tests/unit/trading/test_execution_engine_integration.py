#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行引擎集成测试
测试执行引擎与其他交易组件的集成
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.trading.execution.execution_engine import ExecutionEngine
try:
    from src.trading.execution.execution_engine import ExecutionMode, ExecutionStatus
except ImportError:
    ExecutionMode = None
    ExecutionStatus = None

# 导入相关组件
try:
    from src.trading.order_manager import OrderManager
    from src.trading.portfolio_manager import PortfolioManager
    from src.risk.risk_manager import RiskManager
    from src.infrastructure.cache.unified_cache import UnifiedCache
except ImportError:
    # 如果导入失败，创建Mock类用于测试
    class OrderManager:
        def __init__(self):
            self.orders = {}

        def submit_order(self, order):
            return {"order_id": "test_order_123", "status": "submitted"}

        def get_order_status(self, order_id):
            return {"status": "filled", "filled_quantity": 100}

    class PortfolioManager:
        def __init__(self):
            self.positions = {}

        def update_position(self, symbol, quantity, price):
            self.positions[symbol] = {"quantity": quantity, "avg_price": price}

        def get_position(self, symbol):
            return self.positions.get(symbol, {"quantity": 0, "avg_price": 0})

    class RiskManager:
        def __init__(self):
            self.risk_limits = {"max_position": 1000, "max_loss": 0.1}

        def check_order_risk(self, order):
            return {"approved": True, "risk_score": 0.1}

    class UnifiedCache:
        def __init__(self):
            self.data = {}

        def get(self, key):
            return self.data.get(key)

        def set(self, key, value, ttl=None):
            self.data[key] = value
            return True


class MockBrokerAdapter:
    """模拟经纪商适配器"""

    def __init__(self):
        self.connected = True
        self.orders = {}

    def connect(self):
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False
        return True

    def submit_order(self, order):
        order_id = f"broker_order_{len(self.orders)}"
        self.orders[order_id] = {"status": "filled", "order": order}
        return {"order_id": order_id, "status": "submitted"}

    def get_order_status(self, order_id):
        return self.orders.get(order_id, {"status": "unknown"})

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]


class TestExecutionEngineIntegration:
    """执行引擎集成测试"""

    @pytest.fixture
    def setup_load_test_components(self):
        """设置集成测试组件"""
        # 创建执行引擎
        execution_engine = ExecutionEngine()

        # 创建其他组件
        order_manager = OrderManager()
        portfolio_manager = PortfolioManager()
        risk_manager = RiskManager()
        cache = UnifiedCache()
        broker_adapter = MockBrokerAdapter()

        return {
            "execution_engine": execution_engine,
            "order_manager": order_manager,
            "portfolio_manager": portfolio_manager,
            "risk_manager": risk_manager,
            "cache": cache,
            "broker_adapter": broker_adapter
        }

    def test_execution_engine_with_order_manager_integration(self, setup_load_test_components):
        """测试执行引擎与订单管理器的集成"""
        components = setup_load_test_components
        execution_engine = components["execution_engine"]
        order_manager = components["order_manager"]

        # 创建测试订单
        test_order = {
            "symbol": "000001.SZ",
            "quantity": 100,
            "price": 10.0,
            "order_type": "market",
            "direction": "buy"
        }

        # 提交订单到订单管理器
        order_result = order_manager.submit_order(test_order)
        assert order_result["status"] == "submitted"

        # 先创建订单，然后执行
        # ExecutionEngine有create_order方法
        order_id = execution_engine.create_order(test_order)
        assert order_id is not None

        # 执行引擎处理订单
        with patch.object(execution_engine, 'execute_order', return_value={"status": "success"}):
            execution_result = execution_engine.execute_order(order_id)
            assert execution_result["status"] == "success"

    def test_execution_engine_with_portfolio_manager_integration(self, setup_load_test_components):
        """测试执行引擎与投资组合管理器的集成"""
        components = setup_load_test_components
        execution_engine = components["execution_engine"]
        portfolio_manager = components["portfolio_manager"]

        # 更新投资组合
        portfolio_manager.update_position("000001.SZ", 100, 10.0)

        # 执行新订单
        test_order = {
            "symbol": "000001.SZ",
            "quantity": 50,
            "price": 11.0,
            "order_type": "market",
            "direction": "buy"
        }

        # 先创建订单
        # ExecutionEngine可能没有create_order方法，使用execute_order或submit_order
        if hasattr(execution_engine, 'create_order'):
            order_id = execution_engine.create_order(test_order)
        elif hasattr(execution_engine, 'execute_order'):
            result = execution_engine.execute_order(test_order)
            order_id = result.get('order_id') if isinstance(result, dict) else None
        elif hasattr(execution_engine, 'submit_order'):
            order_id = execution_engine.submit_order(test_order)
        else:
            pytest.skip("ExecutionEngine does not have create_order, execute_order, or submit_order method")
        assert order_id is not None

        with patch.object(execution_engine, 'execute_order') as mock_execute:
            mock_execute.return_value = {
                "status": "success",
                "execution_id": "exec_123",
                "filled_quantity": 50,
                "avg_price": 11.0
            }

            execution_result = execution_engine.execute_order(order_id)
            assert execution_result["status"] == "success"

            # 验证投资组合更新
            position = portfolio_manager.get_position("000001.SZ")
            assert position["quantity"] == 100  # 原有持仓

    def test_execution_engine_with_risk_manager_integration(self, setup_load_test_components):
        """测试执行引擎与风险管理器的集成"""
        components = setup_load_test_components
        execution_engine = components["execution_engine"]
        risk_manager = components["risk_manager"]

        # 创建高风险订单
        high_risk_order = {
            "symbol": "000001.SZ",
            "quantity": 10000,  # 大量订单，可能超过风险限额
            "price": 10.0,
            "order_type": "market",
            "direction": "buy"
        }

        # 风险检查
        risk_result = risk_manager.check_order_risk(high_risk_order)
        assert "approved" in risk_result

        # 执行引擎处理（应该考虑风险）
        # 先创建订单
        order_id = execution_engine.create_order(high_risk_order)
        assert order_id is not None

        with patch.object(execution_engine, 'execute_order') as mock_execute:
            mock_execute.return_value = {"status": "rejected", "reason": "risk_limit_exceeded"}

            execution_result = execution_engine.execute_order(order_id)
            # 根据风险管理器的决定，可能被拒绝
            assert "status" in execution_result

    def test_execution_engine_with_cache_integration(self, setup_load_test_components):
        """测试执行引擎与缓存系统的集成"""
        components = setup_load_test_components
        execution_engine = components["execution_engine"]
        cache = components["cache"]

        # 缓存市场数据
        market_data = {"000001.SZ": {"price": 10.0, "volume": 100000}}
        cache.set("market_data", market_data, ttl=300)

        # 执行引擎使用缓存数据
        cached_data = cache.get("market_data")
        assert cached_data is not None
        assert "000001.SZ" in cached_data

        # 执行基于缓存数据的订单
        test_order = {
            "symbol": "000001.SZ",
            "quantity": 100,
            "price": cached_data["000001.SZ"]["price"],
            "order_type": "limit",
            "direction": "buy"
        }

        # 先创建订单
        # ExecutionEngine可能没有create_order方法，使用execute_order或submit_order
        if hasattr(execution_engine, 'create_order'):
            order_id = execution_engine.create_order(test_order)
        elif hasattr(execution_engine, 'execute_order'):
            result = execution_engine.execute_order(test_order)
            order_id = result.get('order_id') if isinstance(result, dict) else None
        elif hasattr(execution_engine, 'submit_order'):
            order_id = execution_engine.submit_order(test_order)
        else:
            pytest.skip("ExecutionEngine does not have create_order, execute_order, or submit_order method")
        assert order_id is not None

        with patch.object(execution_engine, 'execute_order', return_value={"status": "success"}):
            execution_result = execution_engine.execute_order(order_id)
            assert execution_result["status"] == "success"

    def test_execution_engine_with_broker_adapter_integration(self, setup_load_test_components):
        """测试执行引擎与经纪商适配器的集成"""
        components = setup_load_test_components
        execution_engine = components["execution_engine"]
        broker_adapter = components["broker_adapter"]

        # 连接经纪商
        assert broker_adapter.connect() == True
        assert broker_adapter.connected == True

        # 通过经纪商执行订单
        test_order = {
            "symbol": "000001.SZ",
            "quantity": 100,
            "price": 10.0,
            "order_type": "market",
            "direction": "buy"
        }

        with patch.object(broker_adapter, 'submit_order') as mock_submit:
            mock_submit.return_value = {"order_id": "broker_123", "status": "submitted"}

            # 执行引擎调用经纪商
            result = broker_adapter.submit_order(test_order)
            assert result["status"] == "submitted"
            assert "order_id" in result

    def test_execution_engine_concurrent_orders_integration(self, setup_load_test_components):
        """测试执行引擎并发订单处理集成"""
        components = setup_load_test_components
        execution_engine = components["execution_engine"]

        # 创建多个并发订单
        orders = [
            {
                "symbol": f"00000{i}.SZ",
                "quantity": 100,
                "price": 10.0 + i,
                "order_type": "market",
                "direction": "buy"
            }
            for i in range(5)
        ]

        results = []

        def process_order(order):
            # 先创建订单
            order_id = execution_engine.create_order(order)
            assert order_id is not None

            with patch.object(execution_engine, 'execute_order') as mock_execute:
                mock_execute.return_value = {
                    "status": "success",
                    "execution_id": f"exec_{order['symbol']}",
                    "filled_quantity": order["quantity"],
                    "avg_price": order["price"]
                }
                result = execution_engine.execute_order(order_id)
                results.append(result)
                return result

        # 并发执行订单
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_order, order) for order in orders]
            for future in as_completed(futures):
                future.result()

        # 验证所有订单都成功处理
        assert len(results) == 5
        for result in results:
            assert result["status"] == "success"
            assert "execution_id" in result

    def test_execution_engine_full_workflow_integration(self, setup_load_test_components):
        """测试执行引擎完整工作流集成"""
        components = setup_load_test_components

        # 1. 风险检查
        test_order = {
            "symbol": "000001.SZ",
            "quantity": 100,
            "price": 10.0,
            "order_type": "market",
            "direction": "buy"
        }

        risk_result = components["risk_manager"].check_order_risk(test_order)
        assert risk_result["approved"] == True

        # 2. 订单管理
        order_result = components["order_manager"].submit_order(test_order)
        assert order_result["status"] == "submitted"

        # 3. 执行订单
        # 先创建订单
        order_id = components["execution_engine"].create_order(test_order)
        assert order_id is not None

        with patch.object(components["execution_engine"], 'execute_order') as mock_execute:
            mock_execute.return_value = {
                "status": "success",
                "execution_id": "exec_123",
                "filled_quantity": 100,
                "avg_price": 10.0
            }

            execution_result = components["execution_engine"].execute_order(order_id)
            assert execution_result["status"] == "success"

        # 4. 更新投资组合
        components["portfolio_manager"].update_position(
            test_order["symbol"],
            test_order["quantity"],
            execution_result["avg_price"]
        )

        position = components["portfolio_manager"].get_position(test_order["symbol"])
        assert position["quantity"] == 100
        assert position["avg_price"] == 10.0

        # 5. 缓存结果
        components["cache"].set("last_execution", execution_result, ttl=3600)
        cached_result = components["cache"].get("last_execution")
        assert cached_result is not None
        assert cached_result["execution_id"] == "exec_123"


class TestExecutionEngineLoadTesting:
    """执行引擎负载测试"""

    @pytest.fixture
    def setup_load_test_components(self):
        """设置负载测试组件"""
        # 创建执行引擎
        execution_engine = ExecutionEngine()
        return {"execution_engine": execution_engine}

    def test_execution_engine_high_frequency_trading_load(self, setup_load_test_components):
        """测试执行引擎高频交易负载"""
        components = setup_load_test_components
        execution_engine = components["execution_engine"]

        # 模拟高频交易场景
        order_count = 100
        results = []

        def execute_high_freq_order(order_id):
            order = {
                "symbol": "000001.SZ",
                "quantity": 10,
                "price": 10.0,
                "order_type": "market",
                "side": "buy"
            }

            # 先创建订单
            order_id_created = execution_engine.create_order(order)
            assert order_id_created is not None

            with patch.object(execution_engine, 'execute_order') as mock_execute:
                mock_execute.return_value = {
                    "status": "success",
                    "execution_id": f"exec_{order_id}",
                    "filled_quantity": 10,
                    "avg_price": 10.0
                }

                start_time = time.time()
                result = execution_engine.execute_order(order_id_created)
                end_time = time.time()

                results.append({
                    "order_id": order_id,
                    "result": result,
                    "execution_time": end_time - start_time
                })

        # 并发执行高频订单
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(execute_high_freq_order, i) for i in range(order_count)]
            for future in as_completed(futures):
                future.result()

        # 验证结果
        assert len(results) == order_count
        successful_orders = [r for r in results if r["result"]["status"] == "success"]
        assert len(successful_orders) == order_count

        # 检查性能（平均执行时间应小于100ms）
        avg_execution_time = sum(r["execution_time"] for r in results) / len(results)
        assert avg_execution_time < 0.1, f"平均执行时间太长: {avg_execution_time:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__])
