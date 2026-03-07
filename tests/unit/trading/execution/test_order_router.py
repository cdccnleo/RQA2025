# -*- coding: utf-8 -*-
"""
订单路由器单元测试
测试覆盖率目标: 90%+
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.trading.execution.order_router import (
    OrderRouter,
    RoutingStrategy,
    RoutingResult
)


class TestOrderRouter:
    """订单路由器测试类"""

    def setup_method(self):
        """测试前置设置"""
        self.config = Mock()
        self.metrics = Mock()
        self.router = OrderRouter(config=self.config, metrics=self.metrics)

    def test_init_default(self):
        """测试默认初始化"""
        router = OrderRouter()
        assert router.config is None
        assert router.metrics is None
        assert router.routing_config['strategy'] == RoutingStrategy.BALANCED
        assert router.routing_config['max_latency'] == 1000
        assert router.routing_config['max_cost'] == 0.01
        assert router.routing_config['min_confidence'] == 0.8
        assert len(router.destinations) == 4
        assert 'primary_exchange' in router.destinations

    def test_init_with_config(self):
        """测试使用配置初始化"""
        config = Mock()
        metrics = Mock()
        router = OrderRouter(config=config, metrics=metrics)
        assert router.config == config
        assert router.metrics == metrics

    def test_init_destination_metrics(self):
        """测试目的地指标初始化"""
        router = OrderRouter()
        assert len(router.destination_metrics) == 4
        for dest in router.destinations:
            assert dest in router.destination_metrics
            metrics = router.destination_metrics[dest]
            assert 'latency' in metrics
            assert 'cost' in metrics
            assert 'reliability' in metrics
            assert metrics['latency'] == 50.0
            assert metrics['cost'] == 0.005
            assert metrics['reliability'] == 0.95

    def test_route_order_high_urgency(self):
        """测试路由高优先级订单"""
        order = {
            'order_id': 'test_001',
            'quantity': 100,
            'urgency': 'high'
        }
        result = self.router.route_order(order)
        assert isinstance(result, RoutingResult)
        assert result.destination == 'primary_exchange'
        assert result.strategy == RoutingStrategy.BALANCED
        assert result.estimated_latency == 50.0
        assert result.estimated_cost == 0.005
        assert result.confidence == 0.95

    def test_route_order_large_size(self):
        """测试路由大单"""
        order = {
            'order_id': 'test_002',
            'quantity': 15000,
            'urgency': 'normal'
        }
        result = self.router.route_order(order)
        assert isinstance(result, RoutingResult)
        assert result.destination == 'dark_pool'
        assert result.estimated_latency == 50.0
        assert result.estimated_cost == 0.005

    def test_route_order_normal(self):
        """测试路由普通订单"""
        order = {
            'order_id': 'test_003',
            'quantity': 1000,
            'urgency': 'normal'
        }
        result = self.router.route_order(order)
        assert isinstance(result, RoutingResult)
        assert result.destination == 'secondary_exchange'
        assert result.estimated_latency == 50.0
        assert result.estimated_cost == 0.005

    def test_route_order_without_urgency(self):
        """测试路由没有urgency字段的订单"""
        order = {
            'order_id': 'test_004',
            'quantity': 1000
        }
        result = self.router.route_order(order)
        assert isinstance(result, RoutingResult)
        assert result.destination == 'secondary_exchange'

    def test_route_order_without_quantity(self):
        """测试路由没有quantity字段的订单"""
        order = {
            'order_id': 'test_005',
            'urgency': 'normal'
        }
        result = self.router.route_order(order)
        assert isinstance(result, RoutingResult)
        # 默认quantity为100，应该路由到secondary_exchange
        assert result.destination == 'secondary_exchange'

    def test_route_order_exception_handling(self):
        """测试路由订单异常处理"""
        # 模拟_select_best_destination抛出异常
        with patch.object(self.router, '_select_best_destination', side_effect=Exception("Test error")):
            order = {
                'order_id': 'test_006',
                'quantity': 100
            }
            result = self.router.route_order(order)
            # 应该返回默认路由
            assert isinstance(result, RoutingResult)
            assert result.destination == self.router.destinations[0]
            assert result.strategy == RoutingStrategy.BALANCED
            assert result.estimated_latency == 100.0
            assert result.estimated_cost == 0.01
            assert result.confidence == 0.5

    def test_select_best_destination_high_urgency(self):
        """测试选择最佳目的地 - 高优先级"""
        order = {
            'quantity': 100,
            'urgency': 'high'
        }
        destination = self.router._select_best_destination(order)
        assert destination == 'primary_exchange'

    def test_select_best_destination_large_order(self):
        """测试选择最佳目的地 - 大单"""
        order = {
            'quantity': 10001,
            'urgency': 'normal'
        }
        destination = self.router._select_best_destination(order)
        assert destination == 'dark_pool'

    def test_select_best_destination_normal_order(self):
        """测试选择最佳目的地 - 普通订单"""
        order = {
            'quantity': 5000,
            'urgency': 'normal'
        }
        destination = self.router._select_best_destination(order)
        assert destination == 'secondary_exchange'

    def test_select_best_destination_exact_threshold(self):
        """测试选择最佳目的地 - 刚好阈值"""
        order = {
            'quantity': 10000,
            'urgency': 'normal'
        }
        destination = self.router._select_best_destination(order)
        # 刚好10000，条件 order_size > 10000 为False，应该路由到secondary_exchange
        assert destination == 'secondary_exchange'

    def test_update_destination_metrics(self):
        """测试更新目的地指标"""
        destination = 'primary_exchange'
        new_metrics = {
            'latency': 30.0,
            'cost': 0.003,
            'reliability': 0.98
        }
        self.router.update_destination_metrics(destination, new_metrics)
        updated_metrics = self.router.destination_metrics[destination]
        assert updated_metrics['latency'] == 30.0
        assert updated_metrics['cost'] == 0.003
        assert updated_metrics['reliability'] == 0.98

    def test_update_destination_metrics_partial(self):
        """测试部分更新目的地指标"""
        destination = 'primary_exchange'
        original_latency = self.router.destination_metrics[destination]['latency']
        new_metrics = {
            'latency': 25.0
        }
        self.router.update_destination_metrics(destination, new_metrics)
        updated_metrics = self.router.destination_metrics[destination]
        assert updated_metrics['latency'] == 25.0
        # 其他指标应该保持不变
        assert updated_metrics['cost'] == 0.005
        assert updated_metrics['reliability'] == 0.95

    def test_update_destination_metrics_nonexistent(self):
        """测试更新不存在的目的地指标"""
        destination = 'nonexistent_destination'
        new_metrics = {
            'latency': 30.0
        }
        # 不应该抛出异常，但也不应该更新
        original_metrics = self.router.destination_metrics.copy()
        self.router.update_destination_metrics(destination, new_metrics)
        # 指标字典不应该改变
        assert self.router.destination_metrics == original_metrics

    def test_get_available_destinations(self):
        """测试获取可用目的地列表"""
        destinations = self.router.get_available_destinations()
        assert isinstance(destinations, list)
        assert len(destinations) == 4
        assert 'primary_exchange' in destinations
        assert 'secondary_exchange' in destinations
        assert 'dark_pool' in destinations
        assert 'internal_book' in destinations
        # 应该返回副本，不是原始列表
        assert destinations is not self.router.destinations

    def test_get_destination_metrics_existing(self):
        """测试获取存在的目的地指标"""
        destination = 'primary_exchange'
        metrics = self.router.get_destination_metrics(destination)
        assert metrics is not None
        assert 'latency' in metrics
        assert 'cost' in metrics
        assert 'reliability' in metrics
        assert metrics['latency'] == 50.0
        assert metrics['cost'] == 0.005
        assert metrics['reliability'] == 0.95

    def test_get_destination_metrics_nonexistent(self):
        """测试获取不存在的目的地指标"""
        destination = 'nonexistent_destination'
        metrics = self.router.get_destination_metrics(destination)
        assert metrics is None

    def test_routing_result_dataclass(self):
        """测试RoutingResult数据类"""
        result = RoutingResult(
            destination='test_dest',
            strategy=RoutingStrategy.BEST_PRICE,
            estimated_latency=100.0,
            estimated_cost=0.01,
            confidence=0.9
        )
        assert result.destination == 'test_dest'
        assert result.strategy == RoutingStrategy.BEST_PRICE
        assert result.estimated_latency == 100.0
        assert result.estimated_cost == 0.01
        assert result.confidence == 0.9

    def test_routing_strategy_enum(self):
        """测试RoutingStrategy枚举"""
        assert RoutingStrategy.BEST_PRICE.value == "best_price"
        assert RoutingStrategy.FASTEST_EXECUTION.value == "fastest_execution"
        assert RoutingStrategy.LOWEST_LATENCY.value == "lowest_latency"
        assert RoutingStrategy.BALANCED.value == "balanced"

    def test_route_order_logging(self):
        """测试路由订单日志记录"""
        with patch.object(self.router.logger, 'info') as mock_info:
            order = {
                'order_id': 'test_007',
                'quantity': 100,
                'urgency': 'normal'
            }
            result = self.router.route_order(order)
            # 应该记录路由完成日志
            assert mock_info.called
            log_calls = [str(call) for call in mock_info.call_args_list]
            assert any('订单路由完成' in str(call) for call in log_calls)

    def test_route_order_error_logging(self):
        """测试路由订单错误日志记录"""
        with patch.object(self.router.logger, 'error') as mock_error:
            with patch.object(self.router, '_select_best_destination', side_effect=Exception("Test error")):
                order = {
                    'order_id': 'test_008',
                    'quantity': 100
                }
                result = self.router.route_order(order)
                # 应该记录错误日志
                assert mock_error.called
                log_calls = [str(call) for call in mock_error.call_args_list]
                assert any('订单路由失败' in str(call) for call in log_calls)

    def test_update_destination_metrics_logging(self):
        """测试更新目的地指标日志记录"""
        with patch.object(self.router.logger, 'info') as mock_info:
            destination = 'primary_exchange'
            new_metrics = {'latency': 30.0}
            self.router.update_destination_metrics(destination, new_metrics)
            # 应该记录更新日志
            assert mock_info.called
            log_calls = [str(call) for call in mock_info.call_args_list]
            assert any('更新目的地指标' in str(call) for call in log_calls)

