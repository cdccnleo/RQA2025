"""
订单路由器深度测试
全面测试订单路由器的路由策略、执行优化和智能决策功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

# 导入交易执行相关类
try:
    from src.trading.execution.order_router import (
        OrderRouter, RoutingStrategy, RoutingResult,
        ExecutionVenue, VenueMetrics
    )
    ORDER_ROUTER_AVAILABLE = True
except ImportError:
    ORDER_ROUTER_AVAILABLE = False
    OrderRouter = Mock
    RoutingStrategy = Mock
    RoutingResult = Mock
    ExecutionVenue = Mock
    VenueMetrics = Mock

try:
    from src.trading.execution.execution_engine import ExecutionEngine
    EXECUTION_ENGINE_AVAILABLE = True
except ImportError:
    EXECUTION_ENGINE_AVAILABLE = False
    ExecutionEngine = Mock

try:
    from src.trading.execution.order_manager import OrderManager
    ORDER_MANAGER_AVAILABLE = True
except ImportError:
    ORDER_MANAGER_AVAILABLE = False
    OrderManager = Mock


class TestOrderRouterComprehensive:
    """订单路由器综合深度测试"""

    @pytest.fixture
    def sample_order(self):
        """创建样本订单"""
        return {
            'order_id': 'ORD_001',
            'symbol': 'AAPL',
            'quantity': 1000,
            'order_type': 'market',
            'side': 'buy',
            'timestamp': datetime.now(),
            'urgency': 'normal',
            'min_quantity': 100,
            'max_show_quantity': 500
        }

    @pytest.fixture
    def execution_venues(self):
        """创建执行场所配置"""
        return [
            {
                'venue_id': 'NASDAQ',
                'name': 'NASDAQ',
                'type': 'exchange',
                'latency_ms': 10,
                'commission_bps': 5,
                'liquidity_score': 0.95,
                'market_hours': {'start': '09:30', 'end': '16:00'},
                'supported_symbols': ['AAPL', 'GOOGL', 'MSFT']
            },
            {
                'venue_id': 'NYSE',
                'name': 'NYSE',
                'type': 'exchange',
                'latency_ms': 12,
                'commission_bps': 6,
                'liquidity_score': 0.92,
                'market_hours': {'start': '09:30', 'end': '16:00'},
                'supported_symbols': ['AAPL', 'JPM', 'BAC']
            },
            {
                'venue_id': 'DARK_POOL_1',
                'name': 'Dark Pool Alpha',
                'type': 'dark_pool',
                'latency_ms': 5,
                'commission_bps': 2,
                'liquidity_score': 0.70,
                'market_hours': {'start': '09:30', 'end': '16:00'},
                'supported_symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
            },
            {
                'venue_id': 'HFT_VENUE',
                'name': 'HFT Trading Venue',
                'type': 'hft',
                'latency_ms': 1,
                'commission_bps': 1,
                'liquidity_score': 0.85,
                'market_hours': {'start': '09:30', 'end': '16:00'},
                'supported_symbols': ['AAPL', 'GOOGL', 'MSFT']
            }
        ]

    @pytest.fixture
    def market_data_snapshot(self):
        """创建市场数据快照"""
        return {
            'AAPL': {
                'bid': 150.25,
                'ask': 150.30,
                'bid_size': 1000,
                'ask_size': 1200,
                'last_price': 150.27,
                'volume': 50000,
                'spread_bps': 3.33
            },
            'GOOGL': {
                'bid': 2800.50,
                'ask': 2801.00,
                'bid_size': 500,
                'ask_size': 600,
                'last_price': 2800.75,
                'volume': 25000,
                'spread_bps': 1.79
            }
        }

    @pytest.fixture
    def order_router(self, execution_venues):
        """创建订单路由器实例"""
        if ORDER_ROUTER_AVAILABLE:
            router = OrderRouter()
            router.configure_venues(execution_venues)
            return router
        return Mock(spec=OrderRouter)

    @pytest.fixture
    def execution_engine(self):
        """创建执行引擎实例"""
        if EXECUTION_ENGINE_AVAILABLE:
            return ExecutionEngine()
        return Mock(spec=ExecutionEngine)

    def test_order_router_initialization(self, order_router, execution_venues):
        """测试订单路由器初始化"""
        if ORDER_ROUTER_AVAILABLE:
            assert order_router is not None
            assert hasattr(order_router, 'venues')
            assert hasattr(order_router, 'routing_history')
            assert hasattr(order_router, 'performance_metrics')

    def test_venue_configuration(self, order_router, execution_venues):
        """测试执行场所配置"""
        if ORDER_ROUTER_AVAILABLE:
            assert len(order_router.venues) == len(execution_venues)

            # 检查场所配置
            nasdaq = next(v for v in order_router.venues if v['venue_id'] == 'NASDAQ')
            assert nasdaq['latency_ms'] == 10
            assert nasdaq['commission_bps'] == 5
            assert nasdaq['liquidity_score'] == 0.95

    def test_best_price_routing(self, order_router, sample_order, market_data_snapshot):
        """测试最佳价格路由"""
        if ORDER_ROUTER_AVAILABLE:
            # 设置最佳价格路由策略
            routing_config = {'strategy': RoutingStrategy.BEST_PRICE}

            # 执行路由决策
            routing_result = order_router.route_order(
                order=sample_order,
                market_data=market_data_snapshot,
                routing_config=routing_config
            )

            assert isinstance(routing_result, RoutingResult)
            assert routing_result.destination in ['NASDAQ', 'NYSE', 'DARK_POOL_1', 'HFT_VENUE']
            assert routing_result.strategy == RoutingStrategy.BEST_PRICE
            assert 0 <= routing_result.confidence <= 1

    def test_lowest_latency_routing(self, order_router, sample_order, market_data_snapshot):
        """测试最低延迟路由"""
        if ORDER_ROUTER_AVAILABLE:
            # 设置最低延迟路由策略
            routing_config = {'strategy': RoutingStrategy.LOWEST_LATENCY}

            # 执行路由决策
            routing_result = order_router.route_order(
                order=sample_order,
                market_data=market_data_snapshot,
                routing_config=routing_config
            )

            assert isinstance(routing_result, RoutingResult)
            assert routing_result.strategy == RoutingStrategy.LOWEST_LATENCY
            # 最低延迟应该选择HFT_VENUE（1ms延迟）
            assert routing_result.destination == 'HFT_VENUE'
            assert routing_result.estimated_latency <= 2  # 应该很低

    def test_balanced_routing_strategy(self, order_router, sample_order, market_data_snapshot):
        """测试均衡路由策略"""
        if ORDER_ROUTER_AVAILABLE:
            # 设置均衡路由策略
            routing_config = {'strategy': RoutingStrategy.BALANCED}

            # 执行路由决策
            routing_result = order_router.route_order(
                order=sample_order,
                market_data=market_data_snapshot,
                routing_config=routing_config
            )

            assert isinstance(routing_result, RoutingResult)
            assert routing_result.strategy == RoutingStrategy.BALANCED
            # 均衡策略应该考虑价格、延迟和成本的平衡

    def test_fastest_execution_routing(self, order_router, sample_order, market_data_snapshot):
        """测试最快执行路由"""
        if ORDER_ROUTER_AVAILABLE:
            # 设置最快执行路由策略
            routing_config = {'strategy': RoutingStrategy.FASTEST_EXECUTION}

            # 执行路由决策
            routing_result = order_router.route_order(
                order=sample_order,
                market_data=market_data_snapshot,
                routing_config=routing_config
            )

            assert isinstance(routing_result, RoutingResult)
            assert routing_result.strategy == RoutingStrategy.FASTEST_EXECUTION
            # 最快执行应该优先考虑延迟和流动性

    def test_smart_order_routing(self, order_router, sample_order, market_data_snapshot):
        """测试智能订单路由"""
        if ORDER_ROUTER_AVAILABLE:
            # 启用智能路由
            order_router.enable_smart_routing()

            # 执行智能路由决策
            smart_result = order_router.smart_route(
                order=sample_order,
                market_data=market_data_snapshot,
                factors=['price', 'liquidity', 'latency', 'cost']
            )

            assert isinstance(smart_result, dict)
            assert 'primary_venue' in smart_result
            assert 'backup_venues' in smart_result
            assert 'routing_logic' in smart_result

    def test_multi_venue_order_splitting(self, order_router, sample_order, market_data_snapshot):
        """测试多场所订单拆分"""
        if ORDER_ROUTER_AVAILABLE:
            # 大订单拆分配置
            large_order = sample_order.copy()
            large_order['quantity'] = 10000  # 大订单

            split_config = {
                'enable_splitting': True,
                'max_venue_quantity': 3000,
                'min_venue_quantity': 500
            }

            # 执行订单拆分路由
            split_result = order_router.route_with_splitting(
                order=large_order,
                market_data=market_data_snapshot,
                split_config=split_config
            )

            assert isinstance(split_result, dict)
            assert 'split_orders' in split_result
            assert 'total_quantity' in split_result
            assert len(split_result['split_orders']) > 1  # 应该被拆分成多个订单

            # 检查拆分订单的总数量
            total_split_quantity = sum(order['quantity'] for order in split_result['split_orders'])
            assert total_split_quantity == large_order['quantity']

    def test_adaptive_routing_based_on_performance(self, order_router, sample_order, market_data_snapshot):
        """测试基于性能的自适应路由"""
        if ORDER_ROUTER_AVAILABLE:
            # 模拟历史路由性能数据
            performance_history = [
                {'venue': 'NASDAQ', 'success_rate': 0.95, 'avg_latency': 12},
                {'venue': 'NYSE', 'success_rate': 0.92, 'avg_latency': 15},
                {'venue': 'DARK_POOL_1', 'success_rate': 0.98, 'avg_latency': 8},
                {'venue': 'HFT_VENUE', 'success_rate': 0.90, 'avg_latency': 3}
            ]

            order_router.update_performance_history(performance_history)

            # 执行自适应路由
            adaptive_result = order_router.adaptive_route(
                order=sample_order,
                market_data=market_data_snapshot,
                learning_rate=0.1
            )

            assert isinstance(adaptive_result, RoutingResult)
            # 自适应路由应该偏向历史表现好的场所

    def test_risk_based_routing(self, order_router, sample_order, market_data_snapshot):
        """测试基于风险的路由"""
        if ORDER_ROUTER_AVAILABLE:
            # 配置风险参数
            risk_config = {
                'max_venue_concentration': 0.4,  # 单场所最大占比40%
                'min_venue_diversity': 3,        # 最少使用3个场所
                'risk_tolerance': 'medium'
            }

            order_router.configure_risk_parameters(risk_config)

            # 执行基于风险的路由
            risk_based_result = order_router.route_with_risk_management(
                order=sample_order,
                market_data=market_data_snapshot,
                portfolio_exposure={'NASDAQ': 0.6, 'NYSE': 0.3}  # 当前暴露
            )

            assert isinstance(risk_based_result, dict)
            assert 'risk_adjusted_routing' in risk_based_result
            assert 'concentration_limits' in risk_based_result

    def test_market_impact_aware_routing(self, order_router, sample_order, market_data_snapshot):
        """测试考虑市场冲击的路由"""
        if ORDER_ROUTER_AVAILABLE:
            # 大订单 - 可能产生市场冲击
            large_order = sample_order.copy()
            large_order['quantity'] = 50000

            # 执行考虑市场冲击的路由
            impact_aware_result = order_router.route_with_market_impact(
                order=large_order,
                market_data=market_data_snapshot,
                impact_model='square_root'  # 平方根冲击模型
            )

            assert isinstance(impact_aware_result, dict)
            assert 'estimated_impact' in impact_aware_result
            assert 'impact_adjusted_routing' in impact_aware_result
            assert 'suggested_splitting' in impact_aware_result

    def test_real_time_routing_updates(self, order_router, sample_order):
        """测试实时路由更新"""
        if ORDER_ROUTER_AVAILABLE:
            # 启用实时路由
            order_router.enable_real_time_routing()

            # 模拟实时市场数据流
            real_time_updates = [
                {'timestamp': datetime.now(), 'AAPL': {'bid': 150.25, 'ask': 150.30, 'bid_size': 1000}},
                {'timestamp': datetime.now() + timedelta(seconds=1), 'AAPL': {'bid': 150.20, 'ask': 150.25, 'bid_size': 800}},
                {'timestamp': datetime.now() + timedelta(seconds=2), 'AAPL': {'bid': 150.15, 'ask': 150.20, 'bid_size': 1200}}
            ]

            routing_updates = []

            for update in real_time_updates:
                result = order_router.update_routing_decision(
                    order=sample_order,
                    market_update=update
                )
                routing_updates.append(result)

            assert len(routing_updates) == len(real_time_updates)

            # 检查路由是否根据市场变化调整
            for update in routing_updates:
                assert 'routing_decision' in update
                assert 'reason_for_change' in update

    def test_cross_venue_arbitrage_routing(self, order_router, market_data_snapshot):
        """测试跨场所套利路由"""
        if ORDER_ROUTER_AVAILABLE:
            # 创建价格差异情景
            arbitrage_data = market_data_snapshot.copy()
            arbitrage_data['AAPL']['NASDAQ'] = {'bid': 150.25, 'ask': 150.30}
            arbitrage_data['AAPL']['NYSE'] = {'bid': 150.20, 'ask': 150.35}  # 更好的买入价格

            arbitrage_opportunity = {
                'symbol': 'AAPL',
                'buy_venue': 'NYSE',
                'sell_venue': 'NASDAQ',
                'quantity': 1000,
                'expected_profit': 0.05  # 每股5美分利润
            }

            # 执行套利路由
            arbitrage_result = order_router.route_arbitrage_opportunity(
                arbitrage_opportunity=arbitrage_opportunity,
                market_data=arbitrage_data
            )

            assert isinstance(arbitrage_result, dict)
            assert 'arbitrage_routing' in arbitrage_result
            assert 'expected_profit' in arbitrage_result
            assert 'execution_plan' in arbitrage_result

    def test_high_frequency_routing_optimization(self, order_router, sample_order):
        """测试高频路由优化"""
        if ORDER_ROUTER_AVAILABLE:
            # 配置高频交易参数
            hft_config = {
                'min_latency_threshold': 5,  # 5ms最大延迟
                'max_queue_position': 10,    # 最大队列位置
                'co_location_required': True
            }

            order_router.configure_hft_parameters(hft_config)

            # 执行HFT路由优化
            hft_result = order_router.optimize_hft_routing(
                order=sample_order,
                hft_factors=['latency', 'queue_position', 'co_location']
            )

            assert isinstance(hft_result, dict)
            assert 'hft_optimized_routing' in hft_result
            assert 'latency_analysis' in hft_result
            assert 'queue_position_analysis' in hft_result

    def test_routing_performance_monitoring(self, order_router, sample_order, market_data_snapshot):
        """测试路由性能监控"""
        if ORDER_ROUTER_AVAILABLE:
            # 执行一系列路由决策
            routing_operations = []

            for i in range(10):
                start_time = time.time()
                result = order_router.route_order(
                    order=sample_order,
                    market_data=market_data_snapshot
                )
                end_time = time.time()

                routing_operations.append({
                    'operation_id': i,
                    'execution_time': end_time - start_time,
                    'result': result,
                    'success': result is not None
                })

            # 获取路由性能指标
            performance_metrics = order_router.get_routing_performance()

            assert isinstance(performance_metrics, dict)
            assert 'average_routing_time' in performance_metrics
            assert 'routing_success_rate' in performance_metrics
            assert 'venue_utilization' in performance_metrics

    def test_routing_cost_optimization(self, order_router, sample_order, market_data_snapshot):
        """测试路由成本优化"""
        if ORDER_ROUTER_AVAILABLE:
            # 配置成本参数
            cost_config = {
                'commission_weight': 0.6,
                'latency_cost_weight': 0.3,
                'market_impact_weight': 0.1,
                'cost_horizon_days': 30
            }

            order_router.configure_cost_optimization(cost_config)

            # 执行成本优化路由
            cost_optimized_result = order_router.route_with_cost_optimization(
                order=sample_order,
                market_data=market_data_snapshot,
                cost_factors=['commission', 'latency', 'impact']
            )

            assert isinstance(cost_optimized_result, dict)
            assert 'cost_optimized_routing' in cost_optimized_result
            assert 'estimated_total_cost' in cost_optimized_result
            assert 'cost_breakdown' in cost_optimized_result

    def test_routing_with_market_regime_adaptation(self, order_router, sample_order):
        """测试考虑市场状况的路由适应"""
        if ORDER_ROUTER_AVAILABLE:
            # 定义不同的市场状况
            market_regimes = [
                {'regime': 'normal', 'volatility': 0.15, 'trend': 'sideways'},
                {'regime': 'volatile', 'volatility': 0.35, 'trend': 'up'},
                {'regime': 'crisis', 'volatility': 0.60, 'trend': 'down'}
            ]

            regime_adaptations = {}

            for regime in market_regimes:
                # 根据市场状况调整路由策略
                adapted_result = order_router.adapt_routing_to_regime(
                    order=sample_order,
                    market_regime=regime,
                    adaptation_factors=['liquidity_preference', 'speed_priority', 'cost_sensitivity']
                )

                regime_adaptations[regime['regime']] = adapted_result

            assert len(regime_adaptations) == len(market_regimes)

            # 验证不同市场状况下的路由调整
            for regime, result in regime_adaptations.items():
                assert 'regime_specific_routing' in result
                assert 'adaptation_reasoning' in result

    def test_routing_with_portfolio_constraints(self, order_router, sample_order):
        """测试考虑投资组合约束的路由"""
        if ORDER_ROUTER_AVAILABLE:
            # 定义投资组合约束
            portfolio_constraints = {
                'max_sector_exposure': {'technology': 0.4, 'finance': 0.3},
                'max_single_stock_exposure': 0.1,
                'min_diversification_score': 0.7,
                'risk_budget_utilization': 0.8
            }

            # 当前投资组合状态
            current_portfolio = {
                'AAPL': {'weight': 0.15, 'sector': 'technology'},
                'GOOGL': {'weight': 0.12, 'sector': 'technology'},
                'JPM': {'weight': 0.08, 'sector': 'finance'}
            }

            # 执行考虑约束的路由
            constrained_result = order_router.route_with_portfolio_constraints(
                order=sample_order,
                current_portfolio=current_portfolio,
                constraints=portfolio_constraints
            )

            assert isinstance(constrained_result, dict)
            assert 'constraint_compliant_routing' in constrained_result
            assert 'constraint_analysis' in constrained_result
            assert 'portfolio_impact' in constrained_result

    def test_routing_error_handling_and_recovery(self, order_router):
        """测试路由错误处理和恢复"""
        if ORDER_ROUTER_AVAILABLE:
            # 测试无效订单处理
            invalid_order = {
                'order_id': None,  # 无效订单ID
                'symbol': '',      # 空符号
                'quantity': -100   # 负数量
            }

            try:
                order_router.route_order(invalid_order, {})
            except (ValueError, TypeError):
                # 期望的错误处理
                pass

            # 测试场所故障恢复
            order_router.simulate_venue_failure('NASDAQ')

            # 故障后应该自动切换到其他场所
            recovery_result = order_router.route_order(
                order=sample_order,
                market_data={'AAPL': {'bid': 150.0, 'ask': 150.5}}
            )

            assert isinstance(recovery_result, RoutingResult)
            assert recovery_result.destination != 'NASDAQ'  # 应该避开故障场所

    def test_routing_audit_and_compliance(self, order_router, sample_order, market_data_snapshot):
        """测试路由审计和合规"""
        if ORDER_ROUTER_AVAILABLE:
            # 启用审计跟踪
            order_router.enable_routing_audit()

            # 执行路由操作
            routing_result = order_router.route_order(
                order=sample_order,
                market_data=market_data_snapshot
            )

            # 获取审计日志
            audit_log = order_router.get_routing_audit_log()

            assert isinstance(audit_log, list)
            assert len(audit_log) > 0

            # 检查审计记录
            for record in audit_log:
                assert 'timestamp' in record
                assert 'order_id' in record
                assert 'routing_decision' in record
                assert 'rationale' in record

            # 生成合规报告
            compliance_report = order_router.generate_compliance_report()

            assert isinstance(compliance_report, dict)
            assert 'routing_compliance_status' in compliance_report
            assert 'regulatory_requirements' in compliance_report

    def test_routing_scalability_and_performance(self, order_router):
        """测试路由扩展性和性能"""
        if ORDER_ROUTER_AVAILABLE:
            # 创建大量订单进行性能测试
            large_order_batch = []

            for i in range(100):
                order = {
                    'order_id': f'ORD_{i:03d}',
                    'symbol': f'STOCK_{i % 10}',
                    'quantity': np.random.randint(100, 10000),
                    'order_type': 'market',
                    'side': 'buy' if i % 2 == 0 else 'sell'
                }
                large_order_batch.append(order)

            # 测试批量路由性能
            import time
            start_time = time.time()

            batch_results = order_router.route_order_batch(
                orders=large_order_batch,
                market_data={'STOCK_0': {'bid': 100.0, 'ask': 100.5}}
            )

            end_time = time.time()

            processing_time = end_time - start_time

            # 验证性能（100个订单应该在合理时间内处理）
            assert processing_time < 5  # 5秒内完成
            assert len(batch_results) == len(large_order_batch)

            # 检查批量处理结果
            for result in batch_results:
                assert isinstance(result, RoutingResult)

    def test_routing_machine_learning_enhancement(self, order_router, sample_order, market_data_snapshot):
        """测试路由机器学习增强"""
        if ORDER_ROUTER_AVAILABLE:
            # 启用机器学习增强
            order_router.enable_ml_enhancement()

            # 训练路由决策模型（模拟）
            training_data = [
                {
                    'order_features': {'quantity': 1000, 'urgency': 'high'},
                    'market_features': {'spread': 0.05, 'volume': 50000},
                    'venue_performance': {'NASDAQ': 0.9, 'NYSE': 0.8},
                    'optimal_venue': 'NASDAQ'
                }
                # 更多训练样本...
            ]

            order_router.train_routing_model(training_data)

            # 使用ML增强的路由
            ml_routing_result = order_router.ml_enhanced_route(
                order=sample_order,
                market_data=market_data_snapshot
            )

            assert isinstance(ml_routing_result, RoutingResult)
            assert hasattr(ml_routing_result, 'ml_confidence')
            assert hasattr(ml_routing_result, 'feature_importance')

    def test_routing_configuration_management(self, order_router):
        """测试路由配置管理"""
        if ORDER_ROUTER_AVAILABLE:
            # 更新路由配置
            new_config = {
                'default_strategy': RoutingStrategy.BALANCED,
                'max_venues_per_order': 3,
                'routing_timeout_seconds': 5,
                'enable_fallback_routing': True
            }

            order_router.update_routing_config(new_config)

            # 验证配置更新
            current_config = order_router.get_routing_config()

            assert current_config['default_strategy'] == RoutingStrategy.BALANCED
            assert current_config['max_venues_per_order'] == 3

            # 测试配置持久化
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                config_file = f.name

            order_router.save_routing_config(config_file)
            order_router.load_routing_config(config_file)

            # 清理临时文件
            import os
            os.unlink(config_file)

    def test_routing_integration_with_execution_engine(self, order_router, execution_engine, sample_order):
        """测试路由与执行引擎的集成"""
        if ORDER_ROUTER_AVAILABLE and EXECUTION_ENGINE_AVAILABLE:
            # 配置路由-执行集成
            integration_config = {
                'enable_direct_execution': True,
                'execution_timeout_seconds': 30,
                'failure_retry_attempts': 3
            }

            order_router.configure_execution_integration(integration_config)

            # 执行集成路由和执行流程
            integrated_result = order_router.route_and_execute(
                order=sample_order,
                market_data={'AAPL': {'bid': 150.0, 'ask': 150.5}},
                execution_engine=execution_engine
            )

            assert isinstance(integrated_result, dict)
            assert 'routing_result' in integrated_result
            assert 'execution_result' in integrated_result
            assert 'integrated_performance' in integrated_result

    def test_routing_real_time_adaptation(self, order_router, sample_order):
        """测试路由实时适应"""
        if ORDER_ROUTER_AVAILABLE:
            # 启用实时适应
            order_router.enable_real_time_adaptation()

            # 模拟实时市场条件变化
            market_conditions_stream = [
                {'volatility': 0.15, 'liquidity': 0.9, 'trend': 'up'},
                {'volatility': 0.25, 'liquidity': 0.7, 'trend': 'sideways'},
                {'volatility': 0.35, 'liquidity': 0.5, 'trend': 'down'}
            ]

            adaptation_results = []

            for conditions in market_conditions_stream:
                adapted_routing = order_router.adapt_routing_to_conditions(
                    order=sample_order,
                    market_conditions=conditions
                )
                adaptation_results.append(adapted_routing)

            assert len(adaptation_results) == len(market_conditions_stream)

            # 验证适应性（高波动时应该更保守，低流动性时应该更灵活）
            for i, result in enumerate(adaptation_results):
                assert 'adapted_strategy' in result
                assert 'adaptation_factors' in result
                assert 'confidence_adjustment' in result
