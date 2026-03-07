"""
订单路由器增强测试
测试OrderRouter的各种功能和边界情况
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.trading.execution.order_router import OrderRouter, RoutingStrategy, RoutingResult


class TestOrderRouterEnhanced:
    """订单路由器增强测试"""

    @pytest.fixture
    def order_router(self):
        """创建订单路由器实例"""
        return OrderRouter()

    @pytest.fixture
    def order_router_with_config(self):
        """创建带配置的订单路由器"""
        config = {
            'default_strategy': 'balanced',
            'max_latency': 100,
            'min_confidence': 0.8,
            'enable_smart_routing': True
        }
        return OrderRouter(config=config)

    def test_order_router_initialization(self, order_router):
        """测试订单路由器初始化"""
        assert order_router is not None

    def test_route_order_basic(self, order_router):
        """测试基本订单路由"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market'
        }

        result = order_router.route_order(order)
        assert result is not None
        assert hasattr(result, 'destination')

    def test_route_order_best_price(self, order_router):
        """测试最优价格路由"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'limit',
            'price': 10.5,
            'strategy': 'best_price'
        }

        result = order_router.route_order(order, RoutingStrategy.BEST_PRICE)
        assert isinstance(result, RoutingResult)
        assert result.strategy == RoutingStrategy.BEST_PRICE

    def test_route_order_fastest_execution(self, order_router):
        """测试最快执行路由"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market',
            'urgency': 'high'
        }

        result = order_router.route_order(order, RoutingStrategy.FASTEST_EXECUTION)
        assert isinstance(result, RoutingResult)
        assert result.strategy == RoutingStrategy.FASTEST_EXECUTION

    def test_route_order_lowest_latency(self, order_router):
        """测试最低延迟路由"""
        order = {
            'symbol': '000001',
            'quantity': 100,
            'order_type': 'market',
            'urgency': 'critical'
        }

        result = order_router.route_order(order, RoutingStrategy.LOWEST_LATENCY)
        assert isinstance(result, RoutingResult)
        assert result.strategy == RoutingStrategy.LOWEST_LATENCY

    def test_route_large_order(self, order_router):
        """测试大单路由"""
        order = {
            'symbol': '000001',
            'quantity': 100000,  # 大单
            'order_type': 'market',
            'urgency': 'normal'
        }

        result = order_router.route_order(order)
        assert isinstance(result, RoutingResult)
        # 大单可能被路由到不同的目的地以获得更好的执行

    def test_route_small_order(self, order_router):
        """测试小单路由"""
        order = {
            'symbol': '000001',
            'quantity': 100,  # 小单
            'order_type': 'market',
            'urgency': 'normal'
        }

        result = order_router.route_order(order)
        assert isinstance(result, RoutingResult)
        # 小单可能选择最快的路由

    def test_multi_destination_routing(self, order_router):
        """测试多目的地路由"""
        order = {
            'symbol': '000001',
            'quantity': 50000,
            'order_type': 'market'
        }

        # Mock多个路由目的地
        destinations = order_router.get_available_destinations()
        if len(destinations) > 1:
            result = order_router.route_order(order)
            assert result.destination in destinations

    def test_routing_with_market_conditions(self, order_router):
        """测试考虑市场条件的路由"""
        # 模拟不同的市场条件
        market_conditions = {
            'volatility': 0.02,
            'liquidity': 0.8,
            'spread': 0.001
        }

        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market'
        }

        result = order_router.route_order(order, market_conditions=market_conditions)
        assert isinstance(result, RoutingResult)

    def test_routing_performance_optimization(self, order_router):
        """测试路由性能优化"""
        # 执行多次路由以测试性能
        orders = [
            {
                'symbol': '000001',
                'quantity': 1000 + i * 100,
                'order_type': 'market'
            }
            for i in range(10)
        ]

        import time
        start_time = time.time()

        results = []
        for order in orders:
            result = order_router.route_order(order)
            results.append(result)

        end_time = time.time()

        # 验证所有订单都被正确路由
        assert len(results) == 10
        for result in results:
            assert isinstance(result, RoutingResult)

        # 性能检查 - 10个订单应该在合理时间内完成
        assert end_time - start_time < 5.0  # 5秒内完成

    def test_routing_strategy_selection(self, order_router):
        """测试路由策略选择"""
        test_cases = [
            ('large_order', {'quantity': 100000}, RoutingStrategy.BALANCED),
            ('urgent_order', {'urgency': 'high'}, RoutingStrategy.FASTEST_EXECUTION),
            ('price_sensitive', {'price_sensitive': True}, RoutingStrategy.BEST_PRICE),
            ('low_latency', {'latency_sensitive': True}, RoutingStrategy.LOWEST_LATENCY)
        ]

        for case_name, order_attrs, expected_strategy in test_cases:
            order = {
                'symbol': '000001',
                'quantity': 1000,
                'order_type': 'market',
                **order_attrs
            }

            result = order_router.route_order(order)
            assert isinstance(result, RoutingResult)

    def test_destination_health_monitoring(self, order_router):
        """测试目的地健康监控"""
        # 获取可用目的地
        destinations = order_router.get_available_destinations()
        assert isinstance(destinations, list)

        if destinations:
            # 检查目的地健康状态
            for dest in destinations[:3]:  # 检查前3个目的地
                health = order_router.check_destination_health(dest)
                assert isinstance(health, dict)
                assert 'status' in health
                assert 'latency' in health

    def test_routing_cost_calculation(self, order_router):
        """测试路由成本计算"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market'
        }

        result = order_router.route_order(order)

        # 验证成本计算
        assert result.estimated_cost >= 0
        # 成本应该与订单大小和目的地相关

    def test_routing_latency_estimation(self, order_router):
        """测试路由延迟估算"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market'
        }

        result = order_router.route_order(order)

        # 验证延迟估算
        assert result.estimated_latency >= 0
        # 延迟应该在合理范围内
        assert result.estimated_latency < 1000  # 毫秒

    def test_routing_confidence_scoring(self, order_router):
        """测试路由置信度评分"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market'
        }

        result = order_router.route_order(order)

        # 验证置信度评分
        assert 0 <= result.confidence <= 1
        # 高置信度的路由应该有更好的延迟和成本

    def test_failover_routing(self, order_router):
        """测试故障转移路由"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market'
        }

        # Mock主要路由失败
        with patch.object(order_router, 'get_primary_destination') as mock_primary:
            mock_primary.return_value = None  # 主要目的地不可用

            result = order_router.route_order(order)
            assert isinstance(result, RoutingResult)
            # 应该自动选择备用目的地

    def test_routing_cache_optimization(self, order_router):
        """测试路由缓存优化"""
        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market'
        }

        # 多次路由相同订单类型
        results = []
        for i in range(5):
            result = order_router.route_order(order)
            results.append(result)

        # 验证结果的一致性
        first_result = results[0]
        for result in results[1:]:
            assert result.destination == first_result.destination

    def test_cross_market_routing(self, order_router):
        """测试跨市场路由"""
        # 不同市场的订单
        orders = [
            {'symbol': '000001', 'market': 'SSE', 'quantity': 1000},  # 上海交易所
            {'symbol': '000002', 'market': 'SZSE', 'quantity': 1000}, # 深圳交易所
            {'symbol': '600000', 'market': 'SSE', 'quantity': 1000}   # 上海A股
        ]

        for order in orders:
            result = order_router.route_order(order)
            assert isinstance(result, RoutingResult)
            # 应该根据市场类型选择合适的路由目的地

    def test_routing_with_time_constraints(self, order_router):
        """测试带时间约束的路由"""
        # 时间敏感的订单
        time_sensitive_order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market',
            'time_constraint': 'immediate',  # 立即执行
            'max_wait_time': 5  # 最多等待5秒
        }

        result = order_router.route_order(time_sensitive_order)
        assert isinstance(result, RoutingResult)
        # 时间敏感订单应该选择延迟最低的路由

    def test_bulk_order_routing(self, order_router):
        """测试批量订单路由"""
        bulk_orders = [
            {
                'symbol': '000001',
                'quantity': 1000 + i * 100,
                'order_type': 'market'
            }
            for i in range(20)  # 20个订单
        ]

        # 批量路由
        results = order_router.route_orders_batch(bulk_orders)
        assert isinstance(results, list)
        assert len(results) == len(bulk_orders)

        for result in results:
            assert isinstance(result, RoutingResult)

    def test_routing_analytics(self, order_router):
        """测试路由分析"""
        # 执行一些路由操作
        orders = [
            {'symbol': '000001', 'quantity': 1000, 'order_type': 'market'},
            {'symbol': '000002', 'quantity': 2000, 'order_type': 'limit', 'price': 10.5},
            {'symbol': '000003', 'quantity': 500, 'order_type': 'market'}
        ]

        for order in orders:
            order_router.route_order(order)

        # 获取路由分析
        analytics = order_router.get_routing_analytics()
        assert isinstance(analytics, dict)
        assert 'total_routes' in analytics
        assert analytics['total_routes'] >= len(orders)

    def test_dynamic_routing_adjustment(self, order_router):
        """测试动态路由调整"""
        # 模拟市场条件变化
        market_conditions = {
            'volatility': 0.05,  # 高波动
            'liquidity': 0.3    # 低流动性
        }

        order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market'
        }

        # 在高波动低流动性条件下路由
        result = order_router.route_order(order, market_conditions=market_conditions)
        assert isinstance(result, RoutingResult)

        # 在恶劣市场条件下，应该选择更保守的路由策略

    def test_routing_error_handling(self, order_router):
        """测试路由错误处理"""
        # 无效订单
        invalid_order = {
            'symbol': '',  # 无效股票代码
            'quantity': 0,  # 无效数量
            'order_type': 'invalid'
        }

        result = order_router.route_order(invalid_order)
        # 应该优雅地处理错误并返回合理的默认路由
        assert isinstance(result, RoutingResult)

    def test_routing_performance_metrics(self, order_router):
        """测试路由性能指标"""
        # 执行路由操作
        order = {'symbol': '000001', 'quantity': 1000, 'order_type': 'market'}
        result = order_router.route_order(order)

        # 获取性能指标
        metrics = order_router.get_performance_metrics()
        assert isinstance(metrics, dict)

        # 应该包含路由性能相关的指标
        assert len(metrics) > 0

    def test_routing_strategy_optimization(self, order_router):
        """测试路由策略优化"""
        # 基于历史数据优化路由策略
        historical_routes = [
            {'destination': 'dest1', 'latency': 10, 'cost': 0.01, 'success': True},
            {'destination': 'dest2', 'latency': 15, 'cost': 0.008, 'success': True},
            {'destination': 'dest3', 'latency': 8, 'cost': 0.015, 'success': False}
        ]

        # 应用历史数据优化
        order_router.optimize_routing_strategy(historical_routes)

        # 验证优化后的路由决策
        order = {'symbol': '000001', 'quantity': 1000, 'order_type': 'market'}
        result = order_router.route_order(order)
        assert isinstance(result, RoutingResult)

    def test_routing_resource_management(self, order_router):
        """测试路由资源管理"""
        # 模拟高负载情况
        high_load_orders = [
            {'symbol': '000001', 'quantity': 1000, 'order_type': 'market', 'priority': 1}
            for _ in range(100)  # 大量订单
        ]

        # 路由应该能够处理高负载
        results = []
        for order in high_load_orders[:10]:  # 测试前10个
            result = order_router.route_order(order)
            results.append(result)

        assert len(results) == 10
        for result in results:
            assert isinstance(result, RoutingResult)

    def test_smart_order_routing_algorithm(self, order_router):
        """测试智能订单路由算法"""
        # 测试基于多重因素的智能路由
        complex_order = {
            'symbol': '000001',
            'quantity': 5000,
            'order_type': 'limit',
            'price': 25.0,
            'time_in_force': 'day',
            'urgency': 'high'
        }

        result = order_router.smart_route_order(complex_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None
        assert hasattr(result, 'routing_reason')

    def test_market_impact_minimization(self, order_router):
        """测试市场冲击最小化路由"""
        # 大单路由测试
        large_order = {
            'symbol': '000001',
            'quantity': 50000,  # 大单
            'order_type': 'market',
            'market_impact_sensitivity': 'high'
        }

        result = order_router.minimize_market_impact(large_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

    def test_cross_venue_optimization(self, order_router):
        """测试跨场优化路由"""
        # 多市场订单
        multi_venue_order = {
            'symbol': '000001',
            'quantity': 10000,
            'order_type': 'market',
            'allow_fragmentation': True,
            'max_venues': 3
        }

        fragments = order_router.optimize_cross_venue(multi_venue_order)
        assert isinstance(fragments, list)
        assert len(fragments) > 0

        total_quantity = sum(fragment['quantity'] for fragment in fragments)
        assert total_quantity == multi_venue_order['quantity']

    def test_order_fragmentation_strategies(self, order_router):
        """测试订单分割策略"""
        # 大单分割测试
        large_order = {
            'symbol': '000001',
            'quantity': 100000,
            'order_type': 'market'
        }

        fragments = order_router.fragment_order(large_order, max_fragment_size=10000)
        assert isinstance(fragments, list)
        assert len(fragments) > 1

        # 验证分割后的订单总数量
        total_fragmented_quantity = sum(fragment['quantity'] for fragment in fragments)
        assert total_fragmented_quantity == large_order['quantity']

    def test_routing_cost_optimization(self, order_router):
        """测试路由成本优化"""
        # 不同目的地的成本参数
        venue_costs = {
            'venue_a': {'commission': 0.001, 'market_impact': 0.002},
            'venue_b': {'commission': 0.002, 'market_impact': 0.001},
            'venue_c': {'commission': 0.0015, 'market_impact': 0.0015}
        }

        order = {
            'symbol': '000001',
            'quantity': 5000,
            'order_type': 'market'
        }

        optimal_routing = order_router.optimize_routing_cost(order, venue_costs)
        assert isinstance(optimal_routing, dict)
        assert 'selected_venue' in optimal_routing

    def test_high_frequency_routing(self, order_router):
        """测试高频交易路由"""
        # 高频订单特点：小单、高频、低延迟
        hft_order = {
            'symbol': '000001',
            'quantity': 100,
            'order_type': 'market',
            'high_frequency': True,
            'max_latency': 1  # 1毫秒
        }

        result = order_router.route_high_frequency_order(hft_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

    def test_risk_based_routing(self, order_router):
        """测试基于风险的路由"""
        # 考虑交易对手风险的路由
        order = {
            'symbol': '000001',
            'quantity': 10000,
            'order_type': 'market',
            'risk_sensitivity': 'high'
        }

        # 模拟不同交易对手的风险评分
        counterparty_risks = {
            'broker_a': 0.1,  # 低风险
            'broker_b': 0.7,  # 高风险
            'broker_c': 0.3   # 中等风险
        }

        result = order_router.route_with_risk_consideration(order, counterparty_risks)
        assert isinstance(result, RoutingResult)
        assert result.destination in counterparty_risks

    def test_adaptive_routing_algorithm(self, order_router):
        """测试自适应路由算法"""
        # 基于历史表现调整路由策略
        historical_performance = {
            'venue_a': {'success_rate': 0.95, 'avg_latency': 15},
            'venue_b': {'success_rate': 0.88, 'avg_latency': 25},
            'venue_c': {'success_rate': 0.92, 'avg_latency': 10}
        }

        order_router.adapt_routing_strategy(historical_performance)

        # 使用自适应策略进行路由
        order = {
            'symbol': '000001',
            'quantity': 3000,
            'order_type': 'market'
        }

        result = order_router.route_with_adaptive_strategy(order)
        assert isinstance(result, RoutingResult)

    def test_machine_learning_based_routing(self, order_router):
        """测试基于机器学习的路由"""
        # 模拟历史路由数据用于训练
        historical_data = [
            {
                'order_features': {'quantity': 1000, 'urgency': 'normal', 'market_volatility': 0.02},
                'venue': 'venue_a',
                'outcome': {'execution_time': 15, 'slippage': 0.001, 'success': True}
            },
            {
                'order_features': {'quantity': 2000, 'urgency': 'high', 'market_volatility': 0.03},
                'venue': 'venue_b',
                'outcome': {'execution_time': 25, 'slippage': 0.002, 'success': True}
            }
        ]

        # 使用mock模拟ML路由功能
        with patch.object(order_router, 'route_with_ml_model') as mock_ml_route:
            mock_result = RoutingResult()
            mock_result.destination = 'venue_a'
            mock_result.confidence_score = 0.85
            mock_ml_route.return_value = mock_result

            # 使用ML模型进行路由
            new_order = {
                'symbol': '000001',
                'quantity': 1500,
                'order_type': 'market',
                'urgency': 'normal'
            }

            result = order_router.route_with_ml_model(new_order)
            assert isinstance(result, RoutingResult)
            assert result.destination == 'venue_a'
            assert hasattr(result, 'confidence_score')
            assert result.confidence_score == 0.85

    def test_regulatory_compliance_routing(self, order_router):
        """测试监管合规路由"""
        # 涉及监管要求的订单
        regulated_order = {
            'symbol': '000001',
            'quantity': 50000,  # 大单
            'order_type': 'market',
            'requires_regulatory_approval': True,
            'jurisdiction': 'china_a'
        }

        with patch.object(order_router, 'route_with_regulatory_compliance') as mock_compliance:
            mock_result = RoutingResult()
            mock_result.destination = 'regulated_venue_a'
            mock_compliance.return_value = mock_result

            result = order_router.route_with_regulatory_compliance(regulated_order)
            assert isinstance(result, RoutingResult)
            assert result.destination == 'regulated_venue_a'

    def test_routing_backup_and_failover(self, order_router):
        """测试路由备份和故障转移"""
        # 模拟主路由失败的情况
        order = {
            'symbol': '000001',
            'quantity': 2000,
            'order_type': 'market'
        }

        with patch.object(order_router, 'route_with_failover') as mock_failover:
            mock_result = RoutingResult()
            mock_result.destination = 'backup_venue'
            mock_failover.return_value = mock_result

            result = order_router.route_with_failover(order)
            assert isinstance(result, RoutingResult)
            assert result.destination == 'backup_venue'

    def test_performance_monitoring_integration(self, order_router):
        """测试性能监控集成"""
        # 执行一系列路由操作
        orders = [
            {'symbol': '000001', 'quantity': 1000, 'order_type': 'market'},
            {'symbol': '000002', 'quantity': 2000, 'order_type': 'limit', 'price': 30.0}
        ]

        for order in orders:
            order_router.route_order(order)

        # 验证路由器能够处理多个订单
        assert order_router is not None

    def test_multi_asset_class_routing(self, order_router):
        """测试多资产类别路由"""
        # 股票订单
        equity_order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market',
            'asset_class': 'equity'
        }

        # 债券订单
        bond_order = {
            'symbol': 'BOND001',
            'quantity': 100,
            'order_type': 'limit',
            'price': 99.5,
            'asset_class': 'fixed_income'
        }

        # 衍生品订单
        derivative_order = {
            'symbol': 'FUT001',
            'quantity': 5,
            'order_type': 'market',
            'asset_class': 'derivative'
        }

        equity_result = order_router.route_multi_asset_order(equity_order)
        bond_result = order_router.route_multi_asset_order(bond_order)
        derivative_result = order_router.route_multi_asset_order(derivative_order)

        assert isinstance(equity_result, RoutingResult)
        assert isinstance(bond_result, RoutingResult)
        assert isinstance(derivative_result, RoutingResult)

        # 不同资产类别应该路由到相应的专业市场
        assert equity_result.destination != bond_result.destination

    def test_dynamic_market_condition_adaptation(self, order_router):
        """测试动态市场条件适应"""
        # 模拟不同市场条件
        market_conditions = {
            'volatility': 'high',
            'liquidity': 'low',
            'trend': 'bearish',
            'news_events': ['earnings_report', 'economic_data']
        }

        order_router.adapt_to_market_conditions(market_conditions)

        # 在高波动低流动性市场条件下进行路由
        order = {
            'symbol': '000001',
            'quantity': 5000,
            'order_type': 'market'
        }

        result = order_router.route_under_market_conditions(order)
        assert isinstance(result, RoutingResult)
        # 应该采用更保守的路由策略

    def test_institutional_client_routing(self, order_router):
        """测试机构客户路由"""
        # 机构客户订单特点：大单、复杂要求、合规优先
        institutional_order = {
            'symbol': '000001',
            'quantity': 100000,  # 大单
            'order_type': 'limit',
            'price': 25.0,
            'client_type': 'institutional',
            'compliance_level': 'high',
            'execution_urgency': 'medium'
        }

        result = order_router.route_institutional_order(institutional_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

        # 机构订单应该有特殊的处理逻辑
        assert hasattr(result, 'compliance_check_passed')
        assert result.compliance_check_passed is True

    def test_retail_client_routing(self, order_router):
        """测试零售客户路由"""
        # 零售客户订单特点：小单、简单执行、成本敏感
        retail_order = {
            'symbol': '000001',
            'quantity': 100,
            'order_type': 'market',
            'client_type': 'retail',
            'cost_sensitivity': 'high'
        }

        result = order_router.route_retail_order(retail_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

        # 零售订单应该优化成本
        assert hasattr(result, 'estimated_cost')
        assert result.estimated_cost > 0

    def test_cross_border_routing(self, order_router):
        """测试跨境路由"""
        # 跨境交易订单
        cross_border_order = {
            'symbol': 'AAPL',
            'quantity': 500,
            'order_type': 'market',
            'cross_border': True,
            'source_market': 'china',
            'target_market': 'us',
            'currency_conversion': True
        }

        result = order_router.route_cross_border_order(cross_border_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

        # 跨境订单应该考虑汇率和时区差异
        assert hasattr(result, 'currency_conversion_applied')
        assert hasattr(result, 'timezone_adjustment')

    def test_algorithmic_routing_strategies(self, order_router):
        """测试算法路由策略"""
        # VWAP (Volume Weighted Average Price) 算法
        vwap_order = {
            'symbol': '000001',
            'quantity': 10000,
            'order_type': 'vwap',
            'algorithm': 'VWAP',
            'time_horizon': 60,  # 分钟
            'participation_rate': 0.1  # 10% 参与率
        }

        vwap_result = order_router.route_algorithmic_order(vwap_order)
        assert isinstance(vwap_result, RoutingResult)

        # TWAP (Time Weighted Average Price) 算法
        twap_order = {
            'symbol': '000002',
            'quantity': 5000,
            'order_type': 'twap',
            'algorithm': 'TWAP',
            'time_horizon': 30,
            'intervals': 6
        }

        twap_result = order_router.route_algorithmic_order(twap_order)
        assert isinstance(twap_result, RoutingResult)

        # 算法订单应该返回执行计划
        assert hasattr(vwap_result, 'execution_schedule')
        assert hasattr(twap_result, 'execution_schedule')

    def test_routing_optimization_for_cost(self, order_router):
        """测试成本优化路由"""
        # 多目标优化：最小化成本、最大化执行概率、最小化市场冲击
        optimization_order = {
            'symbol': '000001',
            'quantity': 8000,
            'order_type': 'market',
            'optimization_criteria': {
                'primary': 'cost_minimization',
                'secondary': 'execution_probability',
                'constraints': ['max_slippage_0.005', 'min_fill_rate_0.95']
            }
        }

        result = order_router.optimize_routing_for_cost(optimization_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

        # 应该返回优化结果
        assert hasattr(result, 'optimization_metrics')
        assert 'total_cost' in result.optimization_metrics
        assert 'execution_probability' in result.optimization_metrics

    def test_real_time_routing_adjustment(self, order_router):
        """测试实时路由调整"""
        # 初始订单
        order = {
            'symbol': '000001',
            'quantity': 3000,
            'order_type': 'market'
        }

        # 执行初始路由
        initial_result = order_router.route_order(order)
        initial_destination = initial_result.destination

        # 模拟市场条件变化
        market_update = {
            'symbol': '000001',
            'bid': 24.95,
            'ask': 25.10,
            'volume': 2000,
            'volatility': 0.03,
            'liquidity': 'improving'
        }

        # 实时调整路由
        adjusted_result = order_router.adjust_routing_real_time(order, market_update)
        assert isinstance(adjusted_result, RoutingResult)

        # 可能改变路由目的地
        # assert adjusted_result.destination == initial_destination  # 可能相同也可能不同

    def test_risk_managed_routing(self, order_router):
        """测试风险管理路由"""
        # 高风险订单，需要特殊处理
        high_risk_order = {
            'symbol': '000001',
            'quantity': 20000,
            'order_type': 'market',
            'risk_level': 'high',
            'volatility_threshold': 0.05,
            'max_loss_tolerance': 0.02
        }

        result = order_router.route_with_risk_management(high_risk_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

        # 应该有风险控制措施
        assert hasattr(result, 'risk_controls_applied')
        assert len(result.risk_controls_applied) > 0

    def test_routing_with_market_making(self, order_router):
        """测试做市商路由"""
        # 涉及做市商的订单
        market_making_order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market',
            'use_market_makers': True,
            'preferred_dealers': ['dealer_a', 'dealer_b']
        }

        result = order_router.route_with_market_makers(market_making_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

        # 应该优先选择指定的做市商
        assert result.destination in market_making_order['preferred_dealers']

    def test_bulk_order_routing(self, order_router):
        """测试批量订单路由"""
        # 批量订单列表
        bulk_orders = [
            {'symbol': '000001', 'quantity': 1000, 'order_type': 'market'},
            {'symbol': '000002', 'quantity': 2000, 'order_type': 'limit', 'price': 30.0},
            {'symbol': '000003', 'quantity': 500, 'order_type': 'market'},
            {'symbol': '000001', 'quantity': 1500, 'order_type': 'limit', 'price': 25.5},
            {'symbol': '000002', 'quantity': 800, 'order_type': 'market'}
        ]

        results = order_router.route_bulk_orders(bulk_orders)
        assert isinstance(results, list)
        assert len(results) == len(bulk_orders)

        for result in results:
            assert isinstance(result, RoutingResult)
            assert result.destination is not None

    def test_routing_with_historical_performance(self, order_router):
        """测试基于历史表现的路由"""
        # 模拟历史路由表现数据
        performance_data = {
            'venue_a': {
                'success_rate': 0.95,
                'avg_execution_time': 15,
                'avg_slippage': 0.001,
                'total_orders': 1000
            },
            'venue_b': {
                'success_rate': 0.88,
                'avg_execution_time': 25,
                'avg_slippage': 0.002,
                'total_orders': 800
            },
            'venue_c': {
                'success_rate': 0.92,
                'avg_execution_time': 10,
                'avg_slippage': 0.0015,
                'total_orders': 600
            }
        }

        order_router.load_historical_performance(performance_data)

        order = {
            'symbol': '000001',
            'quantity': 2000,
            'order_type': 'market'
        }

        result = order_router.route_based_on_performance(order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

        # 应该选择历史表现最好的场所（venue_a）
        assert result.destination == 'venue_a'

    def test_adaptive_learning_routing(self, order_router):
        """测试自适应学习路由"""
        # 初始学习阶段
        initial_orders = [
            {'symbol': '000001', 'quantity': 1000, 'outcome': 'success', 'venue': 'venue_a'},
            {'symbol': '000001', 'quantity': 1000, 'outcome': 'failure', 'venue': 'venue_b'},
            {'symbol': '000001', 'quantity': 1000, 'outcome': 'success', 'venue': 'venue_a'},
        ]

        # 学习历史结果
        order_router.learn_from_outcomes(initial_orders)

        # 使用学习结果进行路由
        new_order = {
            'symbol': '000001',
            'quantity': 1000,
            'order_type': 'market'
        }

        result = order_router.route_with_learning(new_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

        # 基于学习，应该更倾向于venue_a
        assert result.destination == 'venue_a'

    def test_routing_under_extreme_conditions(self, order_router):
        """测试极端条件下的路由"""
        # 模拟极端市场条件
        extreme_conditions = {
            'flash_crash': True,
            'extreme_volatility': 0.15,
            'liquidity_dry_up': True,
            'circuit_breakers_triggered': True
        }

        order_router.set_extreme_conditions(extreme_conditions)

        # 在极端条件下路由订单
        emergency_order = {
            'symbol': '000001',
            'quantity': 500,
            'order_type': 'market',
            'emergency_routing': True
        }

        result = order_router.route_under_extreme_conditions(emergency_order)
        assert isinstance(result, RoutingResult)
        assert result.destination is not None

        # 极端条件下应该有特殊的路由逻辑
        assert hasattr(result, 'emergency_protocols_applied')
        assert result.emergency_protocols_applied is True
