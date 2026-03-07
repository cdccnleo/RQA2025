#!/usr/bin/env python3
"""
策略层集成测试覆盖率提升
目标：大幅提升策略层集成测试覆盖率，从0%提升至>70%
策略：系统性地测试策略组件间的交互和端到端业务流程
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


class TestStrategyIntegrationCoverage:
    """策略层集成测试全面覆盖"""

    @pytest.fixture(autouse=True)
    def setup_integration_test(self):
        """设置集成测试环境"""
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        src_path = os.path.join(project_root, "src")

        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        yield

    def test_strategy_backtest_to_execution_flow(self):
        """测试策略回测到执行的完整流程"""
        try:
            from src.strategy.backtest.backtest_engine import BacktestEngine
            from src.strategy.execution.execution_engine import StrategyExecutionEngine
            from src.strategy.core.strategy_service import UnifiedStrategyService

            # 创建测试数据
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            data = pd.DataFrame({
                'close': np.random.normal(100, 5, 100),
                'volume': np.random.uniform(100000, 500000, 100)
            }, index=dates)

            # 1. 回测阶段
            backtest_engine = BacktestEngine()
            strategy_config = {
                'name': 'test_strategy',
                'type': 'momentum',
                'parameters': {'window': 20}
            }

            backtest_result = backtest_engine.run_single_backtest(strategy_config, data)
            assert isinstance(backtest_result, dict)
            assert 'performance_metrics' in backtest_result

            # 2. 策略服务阶段
            strategy_service = UnifiedStrategyService()
            strategy_id = strategy_service.create_strategy(strategy_config)
            assert strategy_id is not None

            # 3. 执行阶段
            execution_engine = StrategyExecutionEngine()
            execution_result = execution_engine.execute_strategy(strategy_id, data.iloc[-1:])
            assert isinstance(execution_result, dict)

        except ImportError as e:
            pytest.skip(f"Strategy integration components not available: {e}")

    def test_strategy_optimization_pipeline(self):
        """测试策略优化管道"""
        try:
            from src.strategy.optimization.strategy_optimizer import StrategyOptimizer
            from src.strategy.backtest.backtest_engine import BacktestEngine

            # 创建测试数据
            data = pd.DataFrame({
                'close': np.random.normal(100, 5, 100),
                'volume': np.random.uniform(100000, 500000, 100)
            })

            # 基础策略配置
            base_config = {
                'name': 'optimization_test',
                'type': 'mean_reversion',
                'parameters': {'window': 20, 'threshold': 2.0}
            }

            # 优化配置
            optimization_config = {
                'method': 'grid_search',
                'parameters': {
                    'window': [10, 20, 30],
                    'threshold': [1.5, 2.0, 2.5]
                },
                'metric': 'sharpe_ratio'
            }

            # 执行优化
            optimizer = StrategyOptimizer()
            backtest_engine = BacktestEngine()

            optimization_result = optimizer.optimize_strategy(
                base_config, optimization_config, data, backtest_engine
            )

            assert isinstance(optimization_result, dict)
            assert 'best_parameters' in optimization_result
            assert 'optimization_history' in optimization_result

        except ImportError as e:
            pytest.skip(f"Strategy optimization components not available: {e}")

    def test_strategy_risk_management_integration(self):
        """测试策略风险管理集成"""
        try:
            from src.strategy.risk.risk_manager import StrategyRiskManager
            from src.strategy.backtest.backtest_engine import BacktestEngine

            # 创建测试数据
            returns = pd.Series(np.random.normal(0.001, 0.02, 100))

            # 风险管理器
            risk_manager = StrategyRiskManager()
            risk_limits = {
                'max_drawdown': 0.05,
                'max_var': 0.03,
                'max_leverage': 2.0
            }
            risk_manager.set_risk_limits(risk_limits)

            # 回测引擎
            backtest_engine = BacktestEngine()

            # 集成风险检查的回测
            strategy_config = {
                'name': 'risk_managed_strategy',
                'type': 'momentum',
                'risk_limits': risk_limits
            }

            with patch.object(risk_manager, 'assess_portfolio_risk') as mock_risk:
                mock_risk.return_value = {'overall_risk_score': 0.02}

                backtest_result = backtest_engine.run_single_backtest(strategy_config, returns)

                assert isinstance(backtest_result, dict)
                # 验证风险检查被调用
                mock_risk.assert_called()

        except ImportError as e:
            pytest.skip(f"Strategy risk management components not available: {e}")

    def test_strategy_monitoring_and_alerting(self):
        """测试策略监控和告警"""
        try:
            from src.strategy.monitoring.performance_monitor import StrategyPerformanceMonitor
            from src.strategy.backtest.alert_system import AlertSystem

            # 创建监控器和告警系统
            monitor = StrategyPerformanceMonitor()
            alert_system = AlertSystem()

            # 配置告警规则
            alert_rules = [
                {'name': 'performance_decline', 'metric': 'total_return', 'threshold': -0.05, 'operator': '<'},
                {'name': 'high_volatility', 'metric': 'volatility', 'threshold': 0.04, 'operator': '>'}
            ]
            alert_system.add_rules(alert_rules)

            # 模拟策略性能数据
            strategy_id = 'test_strategy_001'
            performance_data = {
                'total_return': -0.03,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.02,
                'volatility': 0.035
            }

            # 记录性能
            monitor.record_metrics(strategy_id, performance_data)

            # 检查告警
            alerts = alert_system.check_alerts(performance_data)

            assert isinstance(alerts, list)
            # 验证性能数据被正确记录
            assert strategy_id in monitor.metrics_history

        except ImportError as e:
            pytest.skip(f"Strategy monitoring components not available: {e}")

    def test_strategy_end_to_end_workflow(self):
        """测试策略端到端工作流程"""
        try:
            from src.strategy.core.strategy_service import UnifiedStrategyService
            from src.strategy.backtest.backtest_engine import BacktestEngine
            from src.strategy.execution.execution_engine import StrategyExecutionEngine

            # 1. 创建策略
            strategy_service = UnifiedStrategyService()
            strategy_config = {
                'name': 'end_to_end_test',
                'type': 'trend_following',
                'parameters': {'fast_period': 12, 'slow_period': 26},
                'risk_limits': {'max_drawdown': 0.05}
            }

            strategy_id = strategy_service.create_strategy(strategy_config)
            assert strategy_id is not None

            # 2. 回测策略
            backtest_engine = BacktestEngine()
            market_data = pd.DataFrame({
                'close': np.random.normal(100, 5, 100),
                'volume': np.random.uniform(100000, 500000, 100)
            })

            backtest_result = backtest_engine.run_single_backtest(strategy_config, market_data)
            assert 'performance_metrics' in backtest_result

            # 3. 部署策略
            deployment_result = strategy_service.deploy_strategy(strategy_id)
            assert deployment_result is True

            # 4. 执行策略
            execution_engine = StrategyExecutionEngine()
            execution_result = execution_engine.execute_strategy(strategy_id, market_data.iloc[-1:])
            assert isinstance(execution_result, dict)

            # 5. 监控策略
            monitoring_result = strategy_service.monitor_strategy(strategy_id)
            assert isinstance(monitoring_result, dict)

        except ImportError as e:
            pytest.skip(f"Strategy end-to-end components not available: {e}")

    def test_strategy_multi_asset_portfolio(self):
        """测试多资产投资组合策略"""
        try:
            from src.strategy.portfolio.portfolio_manager import PortfolioStrategyManager

            # 创建多资产数据
            assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
            dates = pd.date_range('2024-01-01', periods=100, freq='D')

            portfolio_data = {}
            for asset in assets:
                portfolio_data[asset] = pd.DataFrame({
                    'close': np.random.normal(100, 5, 100),
                    'volume': np.random.uniform(100000, 500000, 100)
                }, index=dates)

            # 创建投资组合策略
            portfolio_config = {
                'name': 'multi_asset_portfolio',
                'assets': assets,
                'allocation_strategy': 'risk_parity',
                'rebalance_frequency': 'monthly'
            }

            portfolio_manager = PortfolioStrategyManager()
            portfolio = portfolio_manager.create_portfolio(portfolio_config)

            assert portfolio is not None
            assert len(portfolio.assets) == len(assets)

            # 测试投资组合优化
            optimized_weights = portfolio_manager.optimize_portfolio(portfolio_data)
            assert isinstance(optimized_weights, dict)
            assert len(optimized_weights) == len(assets)

            # 验证权重和为1
            total_weight = sum(optimized_weights.values())
            assert abs(total_weight - 1.0) < 0.01

        except ImportError as e:
            pytest.skip(f"Portfolio strategy components not available: {e}")

    def test_strategy_live_trading_simulation(self):
        """测试实盘交易模拟"""
        try:
            from src.strategy.execution.live_trading_engine import LiveTradingEngine
            from src.strategy.risk.real_time_risk_manager import RealTimeRiskManager

            # 创建实时交易引擎
            live_engine = LiveTradingEngine()
            risk_manager = RealTimeRiskManager()

            # 配置交易参数
            trading_config = {
                'max_position_size': 100000,
                'max_daily_loss': 0.02,
                'commission_rate': 0.0005
            }

            live_engine.configure(trading_config)

            # 模拟市场数据流
            market_data_stream = [
                {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000000, 'timestamp': datetime.now()},
                {'symbol': 'GOOGL', 'price': 2800.0, 'volume': 500000, 'timestamp': datetime.now()}
            ]

            # 模拟实时交易决策
            for market_data in market_data_stream:
                # 风险检查
                risk_check = risk_manager.check_real_time_risk(market_data)
                assert isinstance(risk_check, dict)

                # 交易决策
                if risk_check.get('approved', True):
                    trade_decision = live_engine.make_trading_decision(market_data)
                    assert isinstance(trade_decision, dict)

                    # 执行交易
                    if trade_decision.get('action') in ['buy', 'sell']:
                        execution_result = live_engine.execute_trade(trade_decision)
                        assert isinstance(execution_result, dict)

        except ImportError as e:
            pytest.skip(f"Live trading components not available: {e}")
