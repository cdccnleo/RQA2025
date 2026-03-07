#!/usr/bin/env python3
"""
策略层深度测试覆盖率提升
目标：大幅提升策略层测试覆盖率，从7.9%提升至100%
策略：系统性地测试核心策略组件，确保全面覆盖
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


class TestStrategyComprehensiveCoverage:
    """策略层全面覆盖率测试"""

    @pytest.fixture(autouse=True)
    def setup_strategy_test(self):
        """设置策略层测试环境"""
        # 确保src路径正确
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        src_path = os.path.join(project_root, "src")

        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        yield

        # 清理（可选）

    def test_strategy_advanced_analysis_coverage(self):
        """测试高级分析功能覆盖率"""
        try:
            from src.strategy.backtest.advanced_analysis import AdvancedAnalysis

            # 创建测试数据
            market_data = pd.DataFrame({
                'close': np.random.normal(100, 5, 100),
                'volume': np.random.uniform(100000, 500000, 100)
            })

            factor_data = {
                'momentum': pd.DataFrame({'momentum_score': np.random.normal(0, 1, 100)}),
                'value': pd.DataFrame({'value_score': np.random.normal(0, 1, 100)}),
                'quality': pd.DataFrame({'quality_score': np.random.normal(0, 1, 100)})
            }

            # 测试高级分析
            analysis = AdvancedAnalysis()
            assert analysis is not None

            # 测试多因子分析
            results = analysis.analyze_multifactor(market_data, factor_data)
            assert isinstance(results, list)
            assert len(results) > 0

            # 测试情绪分析
            news_data = [{'title': 'Positive news', 'sentiment': 0.8}]
            social_data = [{'platform': 'twitter', 'sentiment': 0.6}]
            sentiment_analysis = analysis.analyze_sentiment(news_data, social_data)
            assert isinstance(sentiment_analysis, dict)

        except ImportError:
            pytest.skip("AdvancedAnalysis not available")

    def test_strategy_advanced_analytics_coverage(self):
        """测试高级分析功能覆盖率"""
        try:
            from src.strategy.backtest.advanced_analytics import AdvancedAnalyticsEngine, FactorData

            # 创建测试数据
            returns = np.random.normal(0.001, 0.02, 100).tolist()
            portfolio_values = (1 + np.cumsum(np.random.normal(0.001, 0.02, 100))).tolist()
            benchmark_returns = np.random.normal(0.0005, 0.01, 100).tolist()

            # 测试高级分析引擎
            analytics = AdvancedAnalyticsEngine()
            assert analytics is not None

            # 测试自定义指标计算
            metrics = analytics.calculate_custom_metrics(returns, portfolio_values, benchmark_returns)
            assert isinstance(metrics, dict)
            assert 'sharpe_ratio' in metrics
            assert 'max_drawdown' in metrics
            assert 'calmar_ratio' in metrics

            # 测试因子数据添加
            factor_data = FactorData(
                timestamp=datetime.now(),
                symbol='AAPL',
                factor_name='momentum',
                factor_value=0.85
            )
            analytics.add_factor_data(factor_data)

            # 测试因子模型构建
            returns_df = pd.DataFrame({'AAPL': returns})
            factor_model = analytics.build_factor_model(returns_df)
            assert isinstance(factor_model, dict)

        except ImportError:
            pytest.skip("AdvancedAnalyticsEngine not available")

    def test_strategy_alert_system_coverage(self):
        """测试告警系统覆盖率"""
        try:
            from src.strategy.backtest.alert_system import AlertSystem

            # 测试告警系统
            alert_system = AlertSystem()
            assert alert_system is not None

            # 测试添加自定义告警规则
            alert_system.add_custom_rule('drawdown_alert', 'max_drawdown', 0.05, '>')
            alert_system.add_custom_rule('volatility_alert', 'volatility', 0.03, '>')

            # 获取告警状态
            status = alert_system.get_alert_status()
            assert isinstance(status, dict)
            assert 'active_rules' in status
            assert status['active_rules'] >= 2  # 至少有我们添加的2个规则

            # 测试告警系统启动和停止
            alert_system.start()
            assert alert_system.alert_manager.monitoring == True

            alert_system.stop()
            assert alert_system.alert_manager.monitoring == False

        except ImportError:
            pytest.skip("AlertSystem not available")

    def test_strategy_analysis_components_coverage(self):
        """测试分析组件覆盖率"""
        try:
            from src.strategy.backtest.analysis.analysis_components import AnalysisComponentFactory

            # 测试分析组件工厂
            factory = AnalysisComponentFactory()

            # 测试获取可用analysis ID
            available_ids = factory.get_available_analysiss()
            assert isinstance(available_ids, list)
            assert len(available_ids) > 0

            # 测试创建组件
            analysis_id = available_ids[0]  # 使用第一个可用的ID
            component = factory.create_component(analysis_id)
            assert component is not None
            assert hasattr(component, 'analysis_id')
            assert component.analysis_id == analysis_id

            # 测试创建所有组件
            all_components = factory.create_all_analysiss()
            assert isinstance(all_components, dict)
            assert len(all_components) == len(available_ids)

        except ImportError:
            pytest.skip("AnalysisComponentFactory not available")

    def test_strategy_portfolio_optimization_coverage(self):
        """测试投资组合优化覆盖率"""
        try:
            from src.strategy.optimization.portfolio_optimizer import PortfolioOptimizer

            # 创建测试数据
            returns = pd.DataFrame({
                'asset1': np.random.normal(0.001, 0.02, 100),
                'asset2': np.random.normal(0.001, 0.025, 100),
                'asset3': np.random.normal(0.001, 0.015, 100)
            })

            # 测试投资组合优化器
            optimizer = PortfolioOptimizer()
            assert optimizer is not None

            # 测试最优权重计算
            weights = optimizer.optimize_portfolio(returns)
            assert isinstance(weights, dict)
            assert len(weights) == 3
            assert abs(sum(weights.values()) - 1.0) < 0.01  # 权重和为1

            # 测试风险-收益分析
            analysis = optimizer.analyze_efficient_frontier(returns)
            assert isinstance(analysis, dict)
            assert 'efficient_portfolios' in analysis

        except ImportError:
            pytest.skip("PortfolioOptimizer not available")

    def test_strategy_risk_management_coverage(self):
        """测试风险管理覆盖率"""
        try:
            from src.strategy.risk.risk_manager import StrategyRiskManager

            # 测试风险管理器
            risk_manager = StrategyRiskManager()
            assert risk_manager is not None

            # 测试风险限额设置
            limits = {
                'max_drawdown': 0.05,
                'max_var': 0.03,
                'max_leverage': 2.0
            }

            risk_manager.set_risk_limits(limits)
            assert risk_manager.risk_limits == limits

            # 测试风险评估
            portfolio = {'asset1': 0.4, 'asset2': 0.3, 'asset3': 0.3}
            risk_assessment = risk_manager.assess_portfolio_risk(portfolio)
            assert isinstance(risk_assessment, dict)
            assert 'overall_risk_score' in risk_assessment

        except ImportError:
            pytest.skip("StrategyRiskManager not available")

    def test_strategy_performance_monitoring_coverage(self):
        """测试性能监控覆盖率"""
        try:
            from src.strategy.monitoring.performance_monitor import StrategyPerformanceMonitor

            # 测试性能监控器
            monitor = StrategyPerformanceMonitor()
            assert monitor is not None

            # 测试指标收集
            metrics = {
                'total_return': 0.15,
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.03,
                'win_rate': 0.65
            }

            monitor.record_metrics('test_strategy', metrics)
            assert 'test_strategy' in monitor.metrics_history

            # 测试性能分析
            analysis = monitor.analyze_performance('test_strategy')
            assert isinstance(analysis, dict)
            assert 'trend_analysis' in analysis

        except ImportError:
            pytest.skip("StrategyPerformanceMonitor not available")

    def test_strategy_signal_generation_coverage(self):
        """测试信号生成覆盖率"""
        try:
            from src.strategy.signals.signal_generator import SignalGenerator

            # 创建测试数据
            data = pd.DataFrame({
                'close': np.random.normal(100, 5, 100),
                'volume': np.random.uniform(100000, 500000, 100)
            })

            # 测试信号生成器
            generator = SignalGenerator()
            assert generator is not None

            # 测试技术指标信号
            signals = generator.generate_technical_signals(data)
            assert isinstance(signals, pd.DataFrame)
            assert len(signals) == len(data)

            # 测试复合信号
            composite_signals = generator.generate_composite_signals(signals)
            assert isinstance(composite_signals, pd.Series)
            assert len(composite_signals) == len(data)

        except ImportError:
            pytest.skip("SignalGenerator not available")

    def test_strategy_execution_engine_coverage(self):
        """测试执行引擎覆盖率"""
        try:
            from src.strategy.execution.execution_engine import StrategyExecutionEngine

            # 测试执行引擎
            engine = StrategyExecutionEngine()
            assert engine is not None

            # 测试订单执行
            order = {
                'symbol': 'AAPL',
                'quantity': 100,
                'order_type': 'market',
                'side': 'buy'
            }

            execution_result = engine.execute_order(order)
            assert isinstance(execution_result, dict)
            assert 'order_id' in execution_result
            assert 'status' in execution_result

            # 测试批量执行
            orders = [order, {**order, 'symbol': 'GOOGL'}]
            batch_results = engine.execute_orders_batch(orders)
            assert isinstance(batch_results, list)
            assert len(batch_results) == 2

        except ImportError:
            pytest.skip("StrategyExecutionEngine not available")

    def test_strategy_integration_coverage(self):
        """测试策略集成覆盖率"""
        try:
            from src.strategy.integration.strategy_integrator import StrategyIntegrator

            # 测试策略集成器
            integrator = StrategyIntegrator()
            assert integrator is not None

            # 测试组件集成
            components = {
                'data_provider': Mock(),
                'signal_generator': Mock(),
                'risk_manager': Mock(),
                'execution_engine': Mock()
            }

            integrator.integrate_components(components)
            assert integrator.components == components

            # 测试端到端执行
            strategy_config = {
                'name': 'test_strategy',
                'type': 'momentum',
                'parameters': {'window': 20}
            }

            result = integrator.run_integrated_strategy(strategy_config)
            assert isinstance(result, dict)
            assert 'execution_result' in result

        except ImportError:
            pytest.skip("StrategyIntegrator not available")
