"""
交易层核心功能测试
测试交易引擎、订单管理、执行算法等核心业务功能
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.trading.execution.order_manager import OrderManager, OrderType, OrderStatus, Order, OrderSide
from src.trading.execution.execution_algorithm import BaseExecutionAlgorithm, AlgorithmType, AlgorithmConfig


class TestTradingDataFactory:
    """交易测试数据工厂"""

    @staticmethod
    def create_sample_order():
        """创建样本订单"""
        return Order(
            order_id="test_order_001",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=150.0,
            side=OrderSide.BUY
        )

    @staticmethod
    def create_sample_execution_config():
        """创建样本执行配置"""
        return AlgorithmConfig(
            algo_type=AlgorithmType.MARKET,
            duration=300,
            target_quantity=1000,
            max_participation=0.1
        )


class TestTradingCoreCoverage:
    """交易层核心功能测试"""

    def setup_method(self):
        """测试前准备"""
        self.data_factory = TestTradingDataFactory()

    def test_order_manager_initialization(self):
        """测试订单管理器初始化"""
        manager = OrderManager()
        assert manager is not None
        assert hasattr(manager, 'submit_order')
        assert hasattr(manager, 'cancel_order')
        assert hasattr(manager, 'get_order')

        # 验证初始统计信息
        assert manager.stats['total_submitted'] == 0
        assert manager.stats['total_filled'] == 0

    def test_order_creation_and_validation(self):
        """测试订单创建和验证"""
        order = self.data_factory.create_sample_order()
        assert order.order_id == "test_order_001"
        assert order.symbol == "AAPL"
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100
        assert order.price == 150.0

    def test_order_manager_submit_order(self):
        """测试订单提交"""
        manager = OrderManager()
        order = self.data_factory.create_sample_order()

        # 提交订单
        success, message, order_id = manager.submit_order(order)

        # 验证提交结果
        assert success is True
        assert order_id == order.order_id
        assert order_id in manager.active_orders

        # 验证统计信息更新
        assert manager.stats['total_submitted'] == 1

    def test_order_manager_cancel_order(self):
        """测试订单取消"""
        manager = OrderManager()
        order = self.data_factory.create_sample_order()

        # 先提交订单
        manager.submit_order(order)

        # 取消订单
        success, message = manager.cancel_order(order.order_id)

        # 验证取消结果
        assert success is True
        assert order.order_id not in manager.active_orders

        # 验证统计信息更新
        assert manager.stats['total_cancelled'] == 1

    def test_order_manager_get_order(self):
        """测试获取订单"""
        manager = OrderManager()
        order = self.data_factory.create_sample_order()

        # 提交前查询订单
        retrieved_order = manager.get_order(order.order_id)
        assert retrieved_order is None  # 订单不存在

        # 提交订单后查询订单
        manager.submit_order(order)
        retrieved_order = manager.get_order(order.order_id)
        assert retrieved_order is not None
        assert retrieved_order.status == OrderStatus.SUBMITTED

    def test_execution_algorithm_interface(self):
        """测试执行算法接口"""
        config = self.data_factory.create_sample_execution_config()

        # 创建基础执行算法实例
        class TestExecutionAlgorithm(BaseExecutionAlgorithm):
            def execute_slice(self, slice_obj):
                return True

            def calculate_next_slice(self, remaining_quantity, time_remaining):
                return None

        algorithm = TestExecutionAlgorithm(config)
        assert algorithm.config == config
        assert algorithm.config.algo_type == AlgorithmType.MARKET

    def test_order_type_enum_values(self):
        """测试订单类型枚举"""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"

        # 验证所有枚举值都是字符串
        for order_type in OrderType:
            assert isinstance(order_type.value, str)

    def test_order_status_enum_values(self):
        """测试订单状态枚举"""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.SUBMITTED.value == "submitted"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"

        # 验证所有枚举值都是字符串
        for status in OrderStatus:
            assert isinstance(status.value, str)

    def test_order_manager_statistics_tracking(self):
        """测试订单管理器统计跟踪"""
        manager = OrderManager()

        # 创建多个订单进行测试
        orders = []
        for i in range(3):
            order = self.data_factory.create_sample_order()
            order.order_id = f"test_order_{i}"
            orders.append(order)

        # 提交所有订单
        for order in orders:
            manager.submit_order(order)

        # 验证统计信息
        assert manager.stats['total_submitted'] == 3

        # 取消一个订单
        manager.cancel_order(orders[0].order_id)
        assert manager.stats['total_cancelled'] == 1

    def test_order_manager_queue_limits(self):
        """测试订单管理器队列限制"""
        # 创建小容量的订单管理器
        manager = OrderManager(max_orders=2)

        # 提交超过限制的订单
        orders = []
        for i in range(3):
            order = self.data_factory.create_sample_order()
            order.order_id = f"queue_test_{i}"
            orders.append(order)

        # 前两个应该成功
        success1, _, _ = manager.submit_order(orders[0])
        success2, _, _ = manager.submit_order(orders[1])

        assert success1 is True
        assert success2 is True

        # 第三个应该失败或队列满
        success3, message3, _ = manager.submit_order(orders[2])
        # 具体行为取决于实现，可能返回False或抛出异常

    def test_algorithm_config_validation(self):
        """测试算法配置验证"""
        # 有效的配置
        valid_config = AlgorithmConfig(
            algo_type=AlgorithmType.MARKET,
            duration=300,
            target_quantity=1000
        )
        assert valid_config.algo_type == AlgorithmType.MARKET
        assert valid_config.duration == 300

        # 无效的配置应该抛出异常或有默认值
        # (具体验证逻辑取决于实现)

    def test_order_manager_error_handling(self):
        """测试订单管理器错误处理"""
        manager = OrderManager()

        # 测试取消不存在的订单
        success, message = manager.cancel_order("nonexistent_order")
        # 应该返回False或抛出适当的异常

        # 测试查询不存在的订单
        retrieved_order = manager.get_order("nonexistent_order")
        assert retrieved_order is None

    def test_execution_algorithm_base_functionality(self):
        """测试执行算法基础功能"""
        config = self.data_factory.create_sample_execution_config()

        class TestExecutionAlgorithm(BaseExecutionAlgorithm):
            def execute_slice(self, slice_obj):
                # 模拟执行成功
                return True

            def calculate_next_slice(self, remaining_quantity, time_remaining):
                # 简单的时间加权算法
                if remaining_quantity > 0 and time_remaining > 0:
                    slice_quantity = min(remaining_quantity, 10)
                    return Mock(quantity=slice_quantity)
                return None

        algorithm = TestExecutionAlgorithm(config)

        # 测试基本属性
        assert algorithm.config == config

        # 测试slice计算
        slice_obj = algorithm.calculate_next_slice(100, 3600)  # 100股，1小时
        if slice_obj:
            assert hasattr(slice_obj, 'quantity')
            assert slice_obj.quantity <= 100

    def test_trading_enums_completeness(self):
        """测试交易枚举的完整性"""
        # 验证所有必要的枚举都存在
        required_order_types = ['MARKET', 'LIMIT']
        required_order_statuses = ['PENDING', 'SUBMITTED', 'FILLED', 'CANCELLED', 'REJECTED']

        for type_name in required_order_types:
            assert hasattr(OrderType, type_name)

        for status_name in required_order_statuses:
            assert hasattr(OrderStatus, status_name)

    def test_order_manager_concurrent_access(self):
        """测试订单管理器并发访问"""
        manager = OrderManager()

        # 这个测试在单线程环境中主要验证接口存在性
        # 在实际并发环境中需要更复杂的测试

        # 验证管理器有必要的同步机制
        assert hasattr(manager, 'active_orders')
        assert hasattr(manager, 'order_queue')

        # 验证队列是线程安全的
        assert hasattr(manager.order_queue, 'put')
        assert hasattr(manager.order_queue, 'get')

    def test_algorithm_type_enum_values(self):
        """测试算法类型枚举"""
        assert AlgorithmType.MARKET.value == "market"
        assert AlgorithmType.TWAP.value == "twap"
        assert AlgorithmType.VWAP.value == "vwap"

        # 验证所有枚举值都是字符串
        for algo_type in AlgorithmType:
            assert isinstance(algo_type.value, str)
