#!/usr/bin/env python3
"""
RQA2025 测试配置和辅助工具
"""

import pytest
import time
from unittest.mock import Mock
from typing import Dict, Any, List
from decimal import Decimal


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """设置测试环境"""
    print("🧪 测试环境已初始化")
    yield
    print("🧹 测试环境已清理")


# Mock工厂类
class MockFactory:
    """Mock对象工厂"""

    @staticmethod
    def create_cache_manager():
        """创建缓存管理器Mock"""
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        mock_cache.delete.return_value = True
        mock_cache.exists.return_value = False
        mock_cache.clear.return_value = True
        return mock_cache

    @staticmethod
    def create_trading_service():
        """创建交易服务Mock"""
        mock_service = Mock()
        mock_service.submit_order.return_value = {
            "order_id": "mock_order_123",
            "status": "submitted",
            "timestamp": time.time()
        }
        mock_service.get_portfolio.return_value = {
            "cash": Decimal('50000'),
            "total_value": Decimal('60500')
        }
        return mock_service

    @staticmethod
    def create_risk_service():
        """创建风控服务Mock"""
        mock_service = Mock()
        mock_service.check_order_risk.return_value = {
            "approved": True,
            "risk_score": 0.1
        }
        return mock_service


# 测试数据生成器
class TestDataGenerator:
    """测试数据生成器"""

    @staticmethod
    def generate_sample_orders(count: int = 5) -> List[Dict[str, Any]]:
        """生成示例订单数据"""
        orders = []
        symbols = ["000001.SZ", "000002.SZ", "600000.SH", "000858.SZ"]

        for i in range(count):
            order = {
                "order_id": f"order_{i:04d}",
                "symbol": symbols[i % len(symbols)],
                "quantity": (i + 1) * 100,
                "price": Decimal('10.0') + i * Decimal('0.5'),
                "direction": "buy" if i % 2 == 0 else "sell",
                "timestamp": time.time() - i * 60
            }
            orders.append(order)
        return orders

    @staticmethod
    def generate_sample_portfolio() -> Dict[str, Any]:
        """生成示例投资组合数据"""
        return {
            "portfolio_id": "test_portfolio",
            "cash": Decimal('100000'),
            "total_value": Decimal('107400')
        }

    @staticmethod
    def generate_sample_signals(count: int = 10) -> List[Dict[str, Any]]:
        """生成示例交易信号"""
        signals = []
        signal_types = ["BUY", "SELL", "HOLD"]

        for i in range(count):
            signal = {
                "signal_id": f"signal_{i:06d}",
                "symbol": f"00000{i%4}.SZ",
                "signal_type": signal_types[i % len(signal_types)],
                "strength": 0.5 + (i % 5) * 0.1,
                "timestamp": time.time() - i * 300
            }
            signals.append(signal)
        return signals


# 断言辅助函数
class AssertHelper:
    """断言辅助函数"""

    @staticmethod
    def assert_order_valid(order: Dict[str, Any]):
        """断言订单数据有效"""
        required_fields = ["order_id", "symbol", "quantity", "price", "direction"]
        for field in required_fields:
            assert field in order, f"订单缺少必要字段: {field}"
        assert order["quantity"] > 0, "订单数量必须大于0"
        assert order["price"] > 0, "订单价格必须大于0"


# 全局实例
mock_factory = MockFactory()
test_data_generator = TestDataGenerator()
assert_helper = AssertHelper()


# pytest fixtures
@pytest.fixture
def mock_cache():
    """缓存Mock对象"""
    return mock_factory.create_cache_manager()


@pytest.fixture
def mock_trading_service():
    """交易服务Mock对象"""
    return mock_factory.create_trading_service()


@pytest.fixture
def mock_risk_service():
    """风控服务Mock对象"""
    return mock_factory.create_risk_service()


@pytest.fixture
def sample_orders():
    """示例订单数据"""
    return test_data_generator.generate_sample_orders()


@pytest.fixture
def sample_portfolio():
    """示例投资组合数据"""
    return test_data_generator.generate_sample_portfolio()


@pytest.fixture
def sample_signals():
    """示例交易信号"""
    return test_data_generator.generate_sample_signals()
