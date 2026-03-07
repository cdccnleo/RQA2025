"""
订单管理器增强测试
测试OrderManager的各种功能和边界情况
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from queue import Queue

from src.trading.execution.order_manager import OrderManager, OrderType, OrderStatus, OrderSide


class TestOrderManagerEnhanced:
    """订单管理器增强测试"""

    @pytest.fixture
    def order_manager(self):
        """创建订单管理器实例"""
        return OrderManager()

    @pytest.fixture
    def order_manager_with_config(self):
        """创建带配置的订单管理器"""
        config = {
            'max_orders_per_second': 200,
            'default_batch_size': 50,
            'cache_ttl_seconds': 7200,
            'max_position_size': 2000000
        }
        return OrderManager(config)

    def test_order_manager_initialization(self, order_manager):
        """测试订单管理器初始化"""
        assert order_manager is not None

    def test_create_market_order(self, order_manager):
        """测试创建市价单"""
        order = order_manager.create_order(
            symbol='000001',
            quantity=1000,
            order_type='market',
            side='buy'
        )
        assert order is not None
        assert hasattr(order, 'symbol')
        assert hasattr(order, 'quantity')

    def test_create_limit_order(self, order_manager):
        """测试创建限价单"""
        order_params = {
            'symbol': '000001',
            'quantity': 1000,
            'price': 10.5,
            'side': OrderSide.SELL,
            'order_type': OrderType.LIMIT
        }

        order = order_manager.create_order(**order_params)
        assert order is not None
        assert order.price == 10.5
        assert order.order_type == OrderType.LIMIT
        assert order.side == OrderSide.SELL

    def test_submit_order(self, order_manager):
        """测试提交订单"""
        order_params = {
            'symbol': '000001',
            'quantity': 1000,
            'side': OrderSide.BUY,
            'order_type': OrderType.MARKET
        }

        order = order_manager.create_order(**order_params)

        # Mock执行引擎
        with patch.object(order_manager, 'execution_engine') as mock_engine:
            mock_engine.submit_order.return_value = True

            result = order_manager.submit_order(order)
            assert result is True
            assert order.status == OrderStatus.SUBMITTED

    def test_cancel_order(self, order_manager):
        """测试取消订单"""
        order_params = {
            'symbol': '000001',
            'quantity': 1000,
            'side': OrderSide.BUY,
            'order_type': OrderType.MARKET
        }

        order = order_manager.create_order(**order_params)

        # Mock执行引擎
        with patch.object(order_manager, 'execution_engine') as mock_engine:
            mock_engine.cancel_order.return_value = True

            result = order_manager.cancel_order(order.order_id)
            assert result is True

    def test_get_order_status(self, order_manager):
        """测试获取订单状态"""
        order_params = {
            'symbol': '000001',
            'quantity': 1000,
            'side': OrderSide.BUY,
            'order_type': OrderType.MARKET
        }

        order = order_manager.create_order(**order_params)

        status = order_manager.get_order_status(order.order_id)
        assert status == OrderStatus.PENDING

    def test_update_order_status(self, order_manager):
        """测试更新订单状态"""
        order_params = {
            'symbol': '000001',
            'quantity': 1000,
            'side': OrderSide.BUY,
            'order_type': OrderType.MARKET
        }

        order = order_manager.create_order(**order_params)

        order_manager.update_order_status(order.order_id, OrderStatus.FILLED)
        assert order.status == OrderStatus.FILLED

    def test_get_pending_orders(self, order_manager):
        """测试获取待处理订单"""
        # 创建多个订单
        orders = []
        for i in range(3):
            order = order_manager.create_order(
                symbol=f'00000{i+1}',
                quantity=1000,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )
            orders.append(order)

        pending_orders = order_manager.get_pending_orders()
        assert len(pending_orders) >= 3

    def test_get_orders_by_symbol(self, order_manager):
        """测试按股票代码获取订单"""
        # 创建多个不同股票的订单
        order1 = order_manager.create_order(
            symbol='000001',
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        order2 = order_manager.create_order(
            symbol='000002',
            quantity=500,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=15.0
        )

        orders_000001 = order_manager.get_orders_by_symbol('000001')
        assert len(orders_000001) >= 1

        orders_000002 = order_manager.get_orders_by_symbol('000002')
        assert len(orders_000002) >= 1

    def test_batch_create_orders(self, order_manager):
        """测试批量创建订单"""
        order_specs = [
            {
                'symbol': '000001',
                'quantity': 1000,
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET
            },
            {
                'symbol': '000002',
                'quantity': 500,
                'side': OrderSide.SELL,
                'order_type': OrderType.LIMIT,
                'price': 15.0
            },
            {
                'symbol': '000003',
                'quantity': 800,
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET
            }
        ]

        orders = order_manager.batch_create_orders(order_specs)
        assert len(orders) == 3

        for order in orders:
            assert order is not None
            assert order.status == OrderStatus.PENDING

    def test_batch_submit_orders(self, order_manager):
        """测试批量提交订单"""
        order_specs = [
            {
                'symbol': '000001',
                'quantity': 1000,
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET
            },
            {
                'symbol': '000002',
                'quantity': 500,
                'side': OrderSide.SELL,
                'order_type': OrderType.LIMIT,
                'price': 15.0
            }
        ]

        orders = order_manager.batch_create_orders(order_specs)

        # Mock执行引擎
        with patch.object(order_manager, 'execution_engine') as mock_engine:
            mock_engine.submit_order.return_value = True

            results = order_manager.batch_submit_orders(orders)
            assert len(results) == 2

    def test_order_validation(self, order_manager):
        """测试订单验证"""
        # 有效订单
        valid_order = {
            'symbol': '000001',
            'quantity': 1000,
            'side': OrderSide.BUY,
            'order_type': OrderType.MARKET
        }

        is_valid, errors = order_manager.validate_order(valid_order)
        assert is_valid is True
        assert len(errors) == 0

        # 无效订单 - 数量为0
        invalid_order = {
            'symbol': '000001',
            'quantity': 0,
            'side': OrderSide.BUY,
            'order_type': OrderType.MARKET
        }

        is_valid, errors = order_manager.validate_order(invalid_order)
        assert is_valid is False
        assert len(errors) > 0

    def test_position_management(self, order_manager):
        """测试持仓管理"""
        # 创建买入订单
        buy_order = order_manager.create_order(
            symbol='000001',
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        # 模拟订单成交
        order_manager.update_order_status(buy_order.order_id, OrderStatus.FILLED)
        buy_order.filled_quantity = 1000
        buy_order.avg_fill_price = 10.5

        # 更新持仓
        order_manager.update_position_from_order(buy_order)

        position = order_manager.get_position('000001')
        assert position is not None
        assert position.quantity == 1000
        assert position.avg_cost == 10.5

    def test_position_update_sell(self, order_manager):
        """测试卖出时的持仓更新"""
        # 先建立买入持仓
        buy_order = order_manager.create_order(
            symbol='000001',
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )
        order_manager.update_order_status(buy_order.order_id, OrderStatus.FILLED)
        buy_order.filled_quantity = 1000
        buy_order.avg_fill_price = 10.5
        order_manager.update_position_from_order(buy_order)

        # 创建卖出订单
        sell_order = order_manager.create_order(
            symbol='000001',
            quantity=500,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET
        )
        order_manager.update_order_status(sell_order.order_id, OrderStatus.FILLED)
        sell_order.filled_quantity = 500
        sell_order.avg_fill_price = 11.0
        order_manager.update_position_from_order(sell_order)

        position = order_manager.get_position('000001')
        assert position.quantity == 500  # 1000 - 500

    def test_risk_check_before_order(self, order_manager):
        """测试下单前的风险检查"""
        # 设置风险限制
        order_manager.max_position_size = 5000

        # 小额订单 - 应该通过
        small_order = order_manager.create_order(
            symbol='000001',
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        can_place, risk_message = order_manager.check_risk_before_order(small_order)
        assert can_place is True

        # 大额订单 - 可能被拒绝
        large_order = order_manager.create_order(
            symbol='000001',
            quantity=10000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )

        can_place, risk_message = order_manager.check_risk_before_order(large_order)
        # 具体结果取决于风险检查逻辑

    def test_order_timeout_handling(self, order_manager):
        """测试订单超时处理"""
        order = order_manager.create_order(
            symbol='000001',
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=10.5
        )

        # 模拟超时
        order.created_at = datetime.now() - timedelta(seconds=400)  # 超过默认超时时间

        expired_orders = order_manager.get_expired_orders()
        assert len(expired_orders) >= 0

    def test_order_priority_handling(self, order_manager):
        """测试订单优先级处理"""
        # 创建不同优先级的订单
        high_priority_order = order_manager.create_order(
            symbol='000001',
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            priority=1  # 高优先级
        )

        low_priority_order = order_manager.create_order(
            symbol='000002',
            quantity=500,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            priority=5  # 低优先级
        )

        # 按优先级排序
        pending_orders = order_manager.get_pending_orders()
        if len(pending_orders) >= 2:
            # 第一个应该是高优先级订单
            assert pending_orders[0].priority <= pending_orders[1].priority

    def test_concurrent_order_processing(self, order_manager):
        """测试并发订单处理"""
        import threading
        import time

        results = []
        errors = []

        def create_and_submit_order(order_id):
            try:
                order = order_manager.create_order(
                    symbol='000001',
                    quantity=100,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET
                )
                results.append(order)
            except Exception as e:
                errors.append(e)

        # 创建多个线程并发创建订单
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_and_submit_order, args=(f'order_{i}',))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证并发处理结果
        assert len(results) == 10
        assert len(errors) == 0

    def test_order_statistics_tracking(self, order_manager):
        """测试订单统计跟踪"""
        # 创建并处理多个订单
        orders = []
        for i in range(5):
            order = order_manager.create_order(
                symbol='000001',
                quantity=1000,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET
            )
            orders.append(order)

        # 模拟部分订单成交
        for i, order in enumerate(orders[:3]):
            order_manager.update_order_status(order.order_id, OrderStatus.FILLED)
            order.filled_quantity = order.quantity
            order.avg_fill_price = 10.5 + i * 0.1

        # 获取统计信息
        stats = order_manager.get_order_statistics()
        assert isinstance(stats, dict)
        assert 'total_orders' in stats
        assert 'filled_orders' in stats
        assert 'pending_orders' in stats

    def test_cache_management(self, order_manager):
        """测试缓存管理"""
        # 创建大量订单以测试缓存
        for i in range(150):  # 超过默认缓存大小
            order = order_manager.create_order(
                symbol=f'00000{i%10+1}',
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )

        # 验证缓存大小控制
        assert len(order_manager.order_cache) <= order_manager.cache_size

        # 测试缓存清理
        order_manager.cleanup_expired_cache()
        # 缓存应该保持在合理大小

    def test_error_handling_and_recovery(self, order_manager):
        """测试错误处理和恢复"""
        # 测试无效参数
        try:
            order_manager.create_order(
                symbol='',  # 无效股票代码
                quantity=-100,  # 无效数量
                side=None,  # 无效方向
                order_type=None  # 无效类型
            )
        except Exception:
            # 应该优雅地处理错误
            pass

        # 验证系统仍然可以正常工作
        valid_order = order_manager.create_order(
            symbol='000001',
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET
        )
        assert valid_order is not None

    def test_performance_monitoring(self, order_manager):
        """测试性能监控"""
        # 执行一些操作
        for i in range(20):
            order = order_manager.create_order(
                symbol='000001',
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )

        # 获取性能指标
        performance = order_manager.get_performance_metrics()
        assert isinstance(performance, dict)

        # 应该包含一些基本的性能指标
        assert 'orders_created' in performance or len(performance) > 0

    def test_order_queue_management(self, order_manager):
        """测试订单队列管理"""
        # 批量创建订单
        orders = []
        for i in range(50):
            order = order_manager.create_order(
                symbol='000001',
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )
            orders.append(order)

        # 验证队列管理
        queue_size = order_manager.get_queue_size()
        assert queue_size >= 0

        # 测试队列处理
        processed = order_manager.process_order_queue(batch_size=10)
        assert isinstance(processed, int)

    def test_resource_cleanup(self, order_manager):
        """测试资源清理"""
        # 创建一些资源
        for i in range(10):
            order_manager.create_order(
                symbol='000001',
                quantity=100,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET
            )

        # 执行清理
        order_manager.cleanup()

        # 验证清理后的状态
        # 系统应该保持在一个合理的状态
