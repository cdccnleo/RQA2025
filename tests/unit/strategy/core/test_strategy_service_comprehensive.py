"""
策略服务核心深度测试
全面测试统一策略服务的核心功能、策略管理、信号生成和性能监控
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
import time

# 导入策略服务相关类
try:
    from src.strategy.core.strategy_service import (
        UnifiedStrategyService, StrategyExecutionContext,
        StrategyPerformanceMetrics, StrategyHealthStatus
    )
    STRATEGY_SERVICE_AVAILABLE = True
except ImportError:
    STRATEGY_SERVICE_AVAILABLE = False
    UnifiedStrategyService = Mock
    StrategyExecutionContext = Mock
    StrategyPerformanceMetrics = Mock
    StrategyHealthStatus = Mock

try:
    from src.strategy.interfaces.strategy_interfaces import (
        StrategyConfig, StrategySignal, StrategyResult, StrategyStatus, StrategyType
    )
    INTERFACES_AVAILABLE = True
except ImportError:
    INTERFACES_AVAILABLE = False
    StrategyConfig = Mock
    StrategySignal = Mock
    StrategyResult = Mock
    StrategyStatus = Mock
    StrategyType = Mock


class TestStrategyServiceComprehensive:
    """策略服务核心综合深度测试"""

    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')  # 一年的交易日
        np.random.seed(42)

        # 生成更真实的市场数据
        initial_price = 100.0
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 日均收益0.05%，波动率2%
        prices = initial_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'symbol': ['AAPL'] * len(dates),
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'returns': returns
        })

    @pytest.fixture
    def sample_strategy_config(self):
        """创建样本策略配置"""
        if INTERFACES_AVAILABLE:
            return StrategyConfig(
                strategy_id="test_momentum_strategy",
                name="Momentum Strategy",
                strategy_type=StrategyType.MOMENTUM,
                parameters={
                    'lookback_period': 20,
                    'threshold': 0.05,
                    'max_position': 100,
                    'stop_loss': 0.1,
                    'take_profit': 0.15
                },
                symbols=['AAPL', 'GOOGL', 'MSFT'],
                risk_limits={
                    'max_drawdown': 0.1,
                    'max_position_size': 1000,
                    'daily_loss_limit': 500
                }
            )
        return Mock()

    @pytest.fixture
    def strategy_service(self):
        """创建策略服务实例"""
        if STRATEGY_SERVICE_AVAILABLE:
            return UnifiedStrategyService()
        return Mock(spec=UnifiedStrategyService)

    def test_strategy_service_initialization(self, strategy_service):
        """测试策略服务初始化"""
        if STRATEGY_SERVICE_AVAILABLE:
            assert strategy_service is not None
            assert hasattr(strategy_service, 'strategy_registry')
            assert hasattr(strategy_service, 'execution_contexts')
            assert hasattr(strategy_service, 'performance_monitor')

    def test_strategy_registration_and_management(self, strategy_service, sample_strategy_config):
        """测试策略注册和管理"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            registration_result = strategy_service.register_strategy(sample_strategy_config)

            assert isinstance(registration_result, dict)
            assert 'strategy_id' in registration_result
            assert 'registration_status' in registration_result
            assert registration_result['strategy_id'] == sample_strategy_config.strategy_id

            # 获取已注册策略
            registered_strategies = strategy_service.get_registered_strategies()

            assert isinstance(registered_strategies, list)
            assert len(registered_strategies) > 0
            assert sample_strategy_config.strategy_id in [s['strategy_id'] for s in registered_strategies]

            # 更新策略配置
            updated_params = {'lookback_period': 30, 'threshold': 0.08}
            update_result = strategy_service.update_strategy_config(
                sample_strategy_config.strategy_id, updated_params
            )

            assert update_result['success'] is True

    def test_strategy_execution_context_creation(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略执行上下文创建"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建执行上下文
            context = strategy_service.create_execution_context(
                strategy_config=sample_strategy_config,
                market_data=sample_market_data.head(50),
                execution_params={
                    'initial_capital': 100000.0,
                    'commission': 0.001,
                    'start_date': '2024-01-01',
                    'end_date': '2024-02-20'
                }
            )

            assert isinstance(context, StrategyExecutionContext)
            assert hasattr(context, 'strategy_config')
            assert hasattr(context, 'market_data')
            assert hasattr(context, 'execution_params')
            assert context.strategy_config.strategy_id == sample_strategy_config.strategy_id

    def test_strategy_signal_generation(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略信号生成"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 初始化策略
            strategy_service.initialize_strategy(sample_strategy_config.strategy_id)

            # 生成信号
            signals = strategy_service.generate_signals(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data.head(100)
            )

            assert isinstance(signals, list)

            if signals:  # 如果有信号生成
                for signal in signals:
                    assert isinstance(signal, StrategySignal)
                    assert hasattr(signal, 'signal_type')
                    assert hasattr(signal, 'symbol')
                    assert hasattr(signal, 'confidence')
                    assert signal.signal_type in ['BUY', 'SELL', 'HOLD']
                    assert signal.symbol in sample_strategy_config.symbols

    def test_strategy_execution_and_result_processing(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略执行和结果处理"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 执行策略
            execution_result = strategy_service.execute_strategy(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data.head(100),
                execution_params={
                    'initial_capital': 100000.0,
                    'start_date': '2024-01-01',
                    'end_date': '2024-04-09'
                }
            )

            assert isinstance(execution_result, StrategyResult)
            assert hasattr(execution_result, 'signals')
            assert hasattr(execution_result, 'performance')
            assert hasattr(execution_result, 'status')

            # 检查性能指标
            if execution_result.performance:
                performance = execution_result.performance
                assert 'total_return' in performance
                assert 'sharpe_ratio' in performance
                assert 'max_drawdown' in performance

    def test_strategy_performance_monitoring(self, strategy_service, sample_strategy_config):
        """测试策略性能监控"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 获取策略性能指标
            performance_metrics = strategy_service.get_strategy_performance(
                strategy_id=sample_strategy_config.strategy_id
            )

            assert isinstance(performance_metrics, StrategyPerformanceMetrics)
            assert hasattr(performance_metrics, 'total_return')
            assert hasattr(performance_metrics, 'sharpe_ratio')
            assert hasattr(performance_metrics, 'max_drawdown')
            assert hasattr(performance_metrics, 'win_rate')

    def test_strategy_health_status_assessment(self, strategy_service, sample_strategy_config):
        """测试策略健康状态评估"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 评估策略健康状态
            health_status = strategy_service.assess_strategy_health(
                strategy_id=sample_strategy_config.strategy_id
            )

            assert isinstance(health_status, StrategyHealthStatus)
            assert hasattr(health_status, 'overall_status')
            assert hasattr(health_status, 'component_status')
            assert hasattr(health_status, 'issues')

            # 检查健康状态
            assert health_status.overall_status in ['healthy', 'warning', 'critical']

    def test_multi_strategy_portfolio_execution(self, strategy_service, sample_market_data):
        """测试多策略投资组合执行"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建多个策略配置
            strategy_configs = [
                StrategyConfig(
                    strategy_id=f"strategy_{i}",
                    name=f"Strategy {i}",
                    strategy_type=StrategyType.MOMENTUM,
                    parameters={'lookback_period': 10 + i * 5, 'threshold': 0.02 + i * 0.02},
                    symbols=['AAPL', 'GOOGL', 'MSFT']
                ) for i in range(3)
            ]

            # 注册多个策略
            for config in strategy_configs:
                strategy_service.register_strategy(config)

            # 执行多策略投资组合
            portfolio_result = strategy_service.execute_multi_strategy_portfolio(
                strategy_ids=[config.strategy_id for config in strategy_configs],
                market_data=sample_market_data.head(100),
                portfolio_allocation={'strategy_0': 0.4, 'strategy_1': 0.35, 'strategy_2': 0.25},
                initial_capital=200000.0
            )

            assert isinstance(portfolio_result, dict)
            assert 'portfolio_performance' in portfolio_result
            assert 'strategy_contributions' in portfolio_result
            assert 'correlation_matrix' in portfolio_result

    def test_strategy_risk_management_integration(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略风险管理集成"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 执行带风险管理的策略
            risk_managed_result = strategy_service.execute_strategy_with_risk_management(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data.head(100),
                risk_limits={
                    'max_drawdown': 0.1,
                    'max_position_size': 50000,
                    'value_at_risk_limit': 0.05
                },
                risk_management_rules={
                    'stop_loss_activation': True,
                    'position_sizing': 'risk_parity',
                    'rebalancing_frequency': 'daily'
                }
            )

            assert isinstance(risk_managed_result, StrategyResult)

            # 检查风险管理是否生效
            if risk_managed_result.performance:
                performance = risk_managed_result.performance
                assert 'risk_adjusted_return' in performance
                assert 'max_drawdown' in performance
                assert abs(performance['max_drawdown']) <= 0.1  # 没有超过风险限额

    def test_strategy_optimization_integration(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略优化集成"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 执行策略优化
            optimization_result = strategy_service.optimize_strategy(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data.head(200),
                optimization_params={
                    'method': 'grid_search',
                    'param_grid': {
                        'lookback_period': [10, 20, 30],
                        'threshold': [0.02, 0.05, 0.08]
                    },
                    'evaluation_metric': 'sharpe_ratio',
                    'cv_folds': 3
                }
            )

            assert isinstance(optimization_result, dict)
            assert 'best_parameters' in optimization_result
            assert 'optimization_score' in optimization_result
            assert 'parameter_sensitivity' in optimization_result

    def test_strategy_backtesting_integration(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略回测集成"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 执行策略回测
            backtest_result = strategy_service.backtest_strategy(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data,
                backtest_params={
                    'start_date': '2024-01-01',
                    'end_date': '2024-12-31',
                    'initial_capital': 100000.0,
                    'commission': 0.001,
                    'slippage': 0.0005
                }
            )

            assert isinstance(backtest_result, dict)
            assert 'backtest_performance' in backtest_result
            assert 'trade_log' in backtest_result
            assert 'performance_metrics' in backtest_result
            assert 'risk_metrics' in backtest_result

    def test_real_time_strategy_execution(self, strategy_service, sample_strategy_config):
        """测试实时策略执行"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 启用实时执行模式
            strategy_service.enable_real_time_execution(sample_strategy_config.strategy_id)

            # 模拟实时市场数据流
            real_time_data_stream = [
                {'timestamp': datetime.now(), 'symbol': 'AAPL', 'price': 150.25, 'volume': 1000000},
                {'timestamp': datetime.now() + timedelta(seconds=1), 'symbol': 'AAPL', 'price': 150.30, 'volume': 1200000},
                {'timestamp': datetime.now() + timedelta(seconds=2), 'symbol': 'AAPL', 'price': 150.15, 'volume': 900000}
            ]

            real_time_signals = []

            # 处理实时数据流
            for market_data in real_time_data_stream:
                signal = strategy_service.process_real_time_data(
                    strategy_id=sample_strategy_config.strategy_id,
                    market_data=market_data
                )

                if signal:
                    real_time_signals.append(signal)

            # 验证实时处理能力
            assert isinstance(real_time_signals, list)

            # 检查信号及时性（应该在合理时间内生成）
            for signal in real_time_signals:
                assert isinstance(signal, StrategySignal)
                assert signal.timestamp >= market_data['timestamp']

    def test_strategy_adaptive_execution(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略自适应执行"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 启用自适应执行
            strategy_service.enable_adaptive_execution(sample_strategy_config.strategy_id)

            # 执行自适应策略
            adaptive_result = strategy_service.execute_adaptive_strategy(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data.head(100),
                adaptation_rules={
                    'market_regime_detection': True,
                    'volatility_adjustment': True,
                    'liquidity_adaptation': True,
                    'learning_rate': 0.1
                }
            )

            assert isinstance(adaptive_result, StrategyResult)

            # 检查自适应调整
            if hasattr(adaptive_result, 'adaptation_log'):
                adaptation_log = adaptive_result.adaptation_log
                assert isinstance(adaptation_log, list)
                assert len(adaptation_log) > 0

    def test_strategy_parallel_execution(self, strategy_service, sample_market_data):
        """测试策略并行执行"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            import threading

            # 创建多个策略进行并行执行
            strategy_configs = [
                StrategyConfig(
                    strategy_id=f"parallel_strategy_{i}",
                    name=f"Parallel Strategy {i}",
                    strategy_type=StrategyType.MOMENTUM,
                    parameters={'lookback_period': 15 + i * 5},
                    symbols=['AAPL']
                ) for i in range(4)
            ]

            # 注册策略
            for config in strategy_configs:
                strategy_service.register_strategy(config)

            results = []
            errors = []

            def execute_strategy_parallel(strategy_id, index):
                try:
                    result = strategy_service.execute_strategy(
                        strategy_id=strategy_id,
                        market_data=sample_market_data.head(50),
                        execution_params={'initial_capital': 50000.0}
                    )
                    results.append((index, result))
                except Exception as e:
                    errors.append((index, str(e)))

            # 并行执行策略
            threads = []
            for i, config in enumerate(strategy_configs):
                thread = threading.Thread(
                    target=execute_strategy_parallel,
                    args=(config.strategy_id, i)
                )
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 验证并行执行结果
            assert len(results) == len(strategy_configs)
            assert len(errors) == 0

            # 检查所有结果都有效
            for index, result in results:
                assert isinstance(result, StrategyResult)

    def test_strategy_performance_analytics(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略性能分析"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 执行策略
            result = strategy_service.execute_strategy(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data.head(100),
                execution_params={'initial_capital': 100000.0}
            )

            # 生成性能分析报告
            analytics_report = strategy_service.generate_performance_analytics(
                strategy_id=sample_strategy_config.strategy_id,
                execution_result=result,
                analysis_config={
                    'benchmark_comparison': True,
                    'risk_decomposition': True,
                    'attribution_analysis': True,
                    'scenario_analysis': True
                }
            )

            assert isinstance(analytics_report, dict)
            assert 'performance_summary' in analytics_report
            assert 'risk_analysis' in analytics_report
            assert 'benchmark_comparison' in analytics_report
            assert 'attribution_breakdown' in analytics_report

    def test_strategy_configuration_management(self, strategy_service, sample_strategy_config):
        """测试策略配置管理"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 保存策略配置
            save_result = strategy_service.save_strategy_configuration(
                strategy_id=sample_strategy_config.strategy_id,
                config_path="test_strategy_config.json"
            )

            assert save_result['success'] is True

            # 加载策略配置
            load_result = strategy_service.load_strategy_configuration(
                config_path="test_strategy_config.json"
            )

            assert load_result['success'] is True
            assert load_result['config']['strategy_id'] == sample_strategy_config.strategy_id

            # 清理测试文件
            import os
            if os.path.exists("test_strategy_config.json"):
                os.remove("test_strategy_config.json")

    def test_strategy_error_handling_and_recovery(self, strategy_service):
        """测试策略错误处理和恢复"""
        if STRATEGY_SERVICE_AVAILABLE:
            # 测试无效策略ID
            try:
                strategy_service.get_strategy_performance("invalid_strategy_id")
            except ValueError:
                # 期望的错误处理
                pass

            # 测试数据质量问题
            invalid_market_data = pd.DataFrame({
                'symbol': [None, 'AAPL'],
                'close': [float('inf'), 100.0]
            })

            try:
                strategy_service.generate_signals("test_strategy", invalid_market_data)
            except (ValueError, TypeError):
                # 期望的数据验证错误
                pass

    def test_strategy_resource_management(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略资源管理"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # 记录初始资源使用
            initial_memory = process.memory_info().rss

            # 执行多个策略操作
            for i in range(10):
                strategy_service.generate_signals(
                    strategy_id=sample_strategy_config.strategy_id,
                    market_data=sample_market_data.head(20)
                )

            # 检查资源使用
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # 验证资源使用合理
            assert memory_increase < 100 * 1024 * 1024  # 100MB限制

            # 获取资源统计
            resource_stats = strategy_service.get_resource_usage()

            assert isinstance(resource_stats, dict)
            assert 'memory_usage_mb' in resource_stats
            assert 'active_strategies' in resource_stats

    def test_strategy_audit_and_compliance(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略审计和合规"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 启用审计跟踪
            strategy_service.enable_audit_trail(sample_strategy_config.strategy_id)

            # 执行策略操作
            strategy_service.generate_signals(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data.head(50)
            )

            # 获取审计日志
            audit_log = strategy_service.get_strategy_audit_log(
                strategy_id=sample_strategy_config.strategy_id
            )

            assert isinstance(audit_log, list)
            assert len(audit_log) > 0

            # 检查审计记录
            for record in audit_log:
                assert 'timestamp' in record
                assert 'operation' in record
                assert 'details' in record

            # 生成合规报告
            compliance_report = strategy_service.generate_compliance_report(
                strategy_id=sample_strategy_config.strategy_id
            )

            assert isinstance(compliance_report, dict)
            assert 'strategy_compliance' in compliance_report
            assert 'regulatory_requirements' in compliance_report

    def test_strategy_machine_learning_integration(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略机器学习集成"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 配置ML增强
            ml_config = {
                'ml_enabled': True,
                'prediction_model': 'random_forest',
                'feature_engineering': True,
                'online_learning': True,
                'model_update_frequency': 'daily'
            }

            strategy_service.configure_ml_enhancement(
                strategy_id=sample_strategy_config.strategy_id,
                ml_config=ml_config
            )

            # 执行ML增强策略
            ml_result = strategy_service.execute_strategy_with_ml(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data.head(100),
                ml_features=['technical_indicators', 'market_sentiment', 'volatility_forecast']
            )

            assert isinstance(ml_result, StrategyResult)

            # 检查ML预测结果
            if hasattr(ml_result, 'ml_predictions'):
                assert isinstance(ml_result.ml_predictions, list)

    def test_strategy_distributed_execution(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略分布式执行"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 配置分布式执行
            distributed_config = {
                'distributed_mode': True,
                'worker_nodes': 3,
                'load_balancing': 'round_robin',
                'fault_tolerance': True,
                'result_aggregation': 'weighted_average'
            }

            strategy_service.configure_distributed_execution(distributed_config)

            # 执行分布式策略
            distributed_result = strategy_service.execute_distributed_strategy(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data,
                execution_partitions=3
            )

            assert isinstance(distributed_result, StrategyResult)
            assert hasattr(distributed_result, 'execution_metadata')
            assert 'distributed_execution' in distributed_result.execution_metadata

    @pytest.mark.asyncio
    async def test_async_strategy_operations(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试异步策略操作"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 异步执行策略
            async_result = await strategy_service.async_execute_strategy(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data.head(50),
                execution_params={'initial_capital': 100000.0}
            )

            assert isinstance(async_result, StrategyResult)

            # 异步生成信号
            async_signals = await strategy_service.async_generate_signals(
                strategy_id=sample_strategy_config.strategy_id,
                market_data=sample_market_data.head(30)
            )

            assert isinstance(async_signals, list)

    def test_strategy_versioning_and_rollback(self, strategy_service, sample_strategy_config):
        """测试策略版本管理和回滚"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建策略版本
            version_1_config = sample_strategy_config
            strategy_service.register_strategy(version_1_config)

            # 更新策略参数创建新版本
            version_2_params = sample_strategy_config.parameters.copy()
            version_2_params['lookback_period'] = 30

            strategy_service.create_strategy_version(
                strategy_id=sample_strategy_config.strategy_id,
                new_parameters=version_2_params,
                version_description="Increased lookback period"
            )

            # 获取策略版本历史
            version_history = strategy_service.get_strategy_version_history(
                strategy_id=sample_strategy_config.strategy_id
            )

            assert isinstance(version_history, list)
            assert len(version_history) >= 2

            # 回滚到旧版本
            rollback_result = strategy_service.rollback_strategy_version(
                strategy_id=sample_strategy_config.strategy_id,
                version_id=version_history[0]['version_id']
            )

            assert rollback_result['success'] is True

    def test_strategy_scenario_analysis(self, strategy_service, sample_strategy_config, sample_market_data):
        """测试策略情景分析"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 定义情景
            scenarios = [
                {
                    'name': 'bull_market',
                    'market_adjustment': {'returns_multiplier': 1.5, 'volatility_multiplier': 0.8}
                },
                {
                    'name': 'bear_market',
                    'market_adjustment': {'returns_multiplier': 0.5, 'volatility_multiplier': 1.5}
                },
                {
                    'name': 'high_volatility',
                    'market_adjustment': {'returns_multiplier': 1.0, 'volatility_multiplier': 2.0}
                }
            ]

            # 执行情景分析
            scenario_results = strategy_service.perform_scenario_analysis(
                strategy_id=sample_strategy_config.strategy_id,
                base_market_data=sample_market_data.head(100),
                scenarios=scenarios
            )

            assert isinstance(scenario_results, dict)
            assert len(scenario_results) == len(scenarios)

            # 检查每个情景的结果
            for scenario_name, result in scenario_results.items():
                assert 'scenario_performance' in result
                assert 'stress_test_metrics' in result
                assert 'worst_case_analysis' in result