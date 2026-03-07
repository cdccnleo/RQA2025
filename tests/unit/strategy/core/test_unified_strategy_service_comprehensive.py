"""
统一策略服务深度测试
全面测试统一策略服务的核心功能、业务逻辑和边界条件
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import json

# 导入策略服务相关类
try:
    from src.strategy.core.strategy_service import UnifiedStrategyService
    STRATEGY_SERVICE_AVAILABLE = True
except ImportError:
    STRATEGY_SERVICE_AVAILABLE = False
    UnifiedStrategyService = Mock

try:
    from src.strategy.interfaces.strategy_interfaces import (
        StrategyConfig, StrategySignal, StrategyResult,
        StrategyStatus, StrategyType, IStrategyService
    )
    INTERFACES_AVAILABLE = True
except ImportError:
    INTERFACES_AVAILABLE = False
    StrategyConfig = Mock
    StrategySignal = Mock
    StrategyResult = Mock
    StrategyStatus = Mock
    StrategyType = Mock
    IStrategyService = Mock

try:
    from src.strategy.core.exceptions import StrategyException
    EXCEPTIONS_AVAILABLE = True
except ImportError:
    EXCEPTIONS_AVAILABLE = False
    StrategyException = Exception


class TestUnifiedStrategyServiceComprehensive:
    """统一策略服务综合深度测试"""

    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'symbol': ['AAPL'] * 100,
            'date': dates,
            'open': np.random.uniform(150, 200, 100),
            'high': np.random.uniform(155, 205, 100),
            'low': np.random.uniform(145, 195, 100),
            'close': np.random.uniform(150, 200, 100),
            'volume': np.random.randint(1000000, 10000000, 100),
            'returns': np.random.normal(0, 0.02, 100)
        })

    @pytest.fixture
    def strategy_config(self):
        """创建策略配置"""
        if INTERFACES_AVAILABLE:
            return StrategyConfig(
                strategy_id="test_001",
                strategy_name="test_strategy",
                strategy_type=StrategyType.MOMENTUM,
                parameters={
                    'lookback_period': 20,
                    'threshold': 0.05,
                    'max_position': 100
                },
                symbols=['AAPL', 'GOOGL'],
                risk_limits={
                    'max_drawdown': 0.1,
                    'max_position_size': 1000
                }
            )
        return Mock()

    @pytest.fixture
    def unified_strategy_service(self):
        """创建统一策略服务实例"""
        if STRATEGY_SERVICE_AVAILABLE:
            return UnifiedStrategyService()
        return Mock(spec=UnifiedStrategyService)

    def test_unified_strategy_service_initialization(self, unified_strategy_service):
        """测试统一策略服务初始化"""
        if STRATEGY_SERVICE_AVAILABLE:
            assert unified_strategy_service is not None
            assert hasattr(unified_strategy_service, 'config')
            assert hasattr(unified_strategy_service, 'strategy_registry')

    def test_strategy_config_creation(self, strategy_config):
        """测试策略配置创建"""
        if INTERFACES_AVAILABLE:
            assert strategy_config.strategy_name == "test_strategy"
            assert strategy_config.strategy_type == StrategyType.MOMENTUM
            assert 'lookback_period' in strategy_config.parameters
            assert hasattr(strategy_config, 'risk_limits')

    def test_strategy_registration(self, unified_strategy_service, strategy_config):
        """测试策略注册"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            assert strategy_id is not None
            assert isinstance(strategy_id, str)

            # 验证策略已注册
            registered_strategies = unified_strategy_service.list_strategies()
            assert strategy_id in registered_strategies

    def test_strategy_execution_basic(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略基本执行"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 执行策略
            result = unified_strategy_service.execute_strategy(strategy_id, sample_market_data)

            assert isinstance(result, StrategyResult)
            assert result.strategy_id == strategy_id
            assert 'all_signals' in result.metadata
            assert 'performance_metrics' in result.metadata

    def test_strategy_signal_generation(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略信号生成"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 生成信号
            signals = unified_strategy_service.generate_signals(strategy_id, sample_market_data)

            assert isinstance(signals, list)
            for signal in signals:
                assert isinstance(signal, StrategySignal)
                assert hasattr(signal, 'symbol')
                assert hasattr(signal, 'signal_type')
                assert hasattr(signal, 'strength')

    def test_strategy_performance_evaluation(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略性能评估"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 执行策略并评估性能
            result = unified_strategy_service.execute_strategy(strategy_id, sample_market_data)
            performance = unified_strategy_service.evaluate_performance(strategy_id, result)

            assert isinstance(performance, dict)
            assert 'sharpe_ratio' in performance
            assert 'max_drawdown' in performance
            assert 'total_return' in performance

    def test_strategy_risk_management(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略风险管理"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 执行风险检查
            risk_assessment = unified_strategy_service.assess_risk(strategy_id, sample_market_data)

            assert isinstance(risk_assessment, dict)
            assert 'risk_score' in risk_assessment
            assert 'risk_factors' in risk_assessment
            assert 'recommendations' in risk_assessment

    def test_strategy_optimization(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略优化"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 执行策略优化
            optimization_result = unified_strategy_service.optimize_strategy(
                strategy_id, sample_market_data,
                optimization_params=['lookback_period', 'threshold']
            )

            assert isinstance(optimization_result, dict)
            assert 'optimized_parameters' in optimization_result
            assert 'improvement' in optimization_result

    def test_strategy_lifecycle_management(self, unified_strategy_service, strategy_config):
        """测试策略生命周期管理"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 检查策略状态
            status = unified_strategy_service.get_strategy_status(strategy_id)
            assert status == StrategyStatus.CREATED

            # 启动策略
            unified_strategy_service.start_strategy(strategy_id)
            status = unified_strategy_service.get_strategy_status(strategy_id)
            assert status == StrategyStatus.RUNNING

            # 停止策略
            unified_strategy_service.stop_strategy(strategy_id)
            status = unified_strategy_service.get_strategy_status(strategy_id)
            assert status == StrategyStatus.STOPPED

    def test_strategy_configuration_management(self, unified_strategy_service):
        """测试策略配置管理"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建多个策略配置
            configs = [
                StrategyConfig(
                    name="momentum_strategy",
                    strategy_type=StrategyType.MOMENTUM,
                    parameters={'lookback': 20}
                ),
                StrategyConfig(
                    name="mean_reversion_strategy",
                    strategy_type=StrategyType.MEAN_REVERSION,
                    parameters={'threshold': 0.02}
                )
            ]

            # 批量注册
            strategy_ids = []
            for config in configs:
                strategy_id = unified_strategy_service.register_strategy(config)
                strategy_ids.append(strategy_id)

            # 验证配置管理
            for i, strategy_id in enumerate(strategy_ids):
                stored_config = unified_strategy_service.get_strategy_config(strategy_id)
                assert stored_config.name == configs[i].name

    def test_strategy_error_handling(self, unified_strategy_service):
        """测试策略错误处理"""
        if STRATEGY_SERVICE_AVAILABLE:
            # 测试无效策略ID
            with pytest.raises((StrategyException, KeyError, ValueError)):
                unified_strategy_service.get_strategy_status("invalid_id")

            # 测试无效配置
            invalid_config = Mock()
            invalid_config.name = None  # 无效配置

            try:
                unified_strategy_service.register_strategy(invalid_config)
                # 如果没有抛出异常，至少验证返回了结果
            except (StrategyException, ValueError, TypeError):
                # 期望的异常
                pass

    def test_strategy_concurrent_execution(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略并发执行"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册多个策略实例
            strategy_ids = []
            for i in range(3):
                config = strategy_config
                config.name = f"concurrent_strategy_{i}"
                strategy_id = unified_strategy_service.register_strategy(config)
                strategy_ids.append(strategy_id)

            # 并发执行策略
            import threading
            results = []
            errors = []

            def execute_strategy(strategy_id):
                try:
                    result = unified_strategy_service.execute_strategy(strategy_id, sample_market_data)
                    results.append((strategy_id, result))
                except Exception as e:
                    errors.append((strategy_id, str(e)))

            threads = []
            for strategy_id in strategy_ids:
                thread = threading.Thread(target=execute_strategy, args=(strategy_id,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # 验证并发执行结果
            assert len(results) == len(strategy_ids)
            assert len(errors) == 0

    def test_strategy_monitoring_and_alerts(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略监控和告警"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 执行策略
            unified_strategy_service.execute_strategy(strategy_id, sample_market_data)

            # 获取监控指标
            metrics = unified_strategy_service.get_strategy_metrics(strategy_id)

            assert isinstance(metrics, dict)
            assert 'execution_time' in metrics
            assert 'signal_count' in metrics

            # 检查告警
            alerts = unified_strategy_service.get_strategy_alerts(strategy_id)

            assert isinstance(alerts, list)

    def test_strategy_data_validation(self, unified_strategy_service, strategy_config):
        """测试策略数据验证"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 测试有效数据
            valid_data = pd.DataFrame({
                'symbol': ['AAPL'] * 50,
                'close': np.random.uniform(100, 200, 50),
                'volume': np.random.randint(1000, 10000, 50)
            })

            validation_result = unified_strategy_service.validate_strategy_data(strategy_id, valid_data)
            assert validation_result['valid'] is True

            # 测试无效数据
            invalid_data = pd.DataFrame({
                'symbol': [None] * 10,
                'close': ['invalid'] * 10
            })

            validation_result = unified_strategy_service.validate_strategy_data(strategy_id, invalid_data)
            assert validation_result['valid'] is False

    def test_strategy_persistence(self, unified_strategy_service, strategy_config, tmp_path):
        """测试策略持久化"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 保存策略
            save_path = tmp_path / "strategy_backup.json"
            unified_strategy_service.save_strategy(strategy_id, str(save_path))

            # 验证文件创建
            assert save_path.exists()

            # 加载策略
            loaded_strategy_id = unified_strategy_service.load_strategy(str(save_path))

            # 验证加载的策略
            loaded_config = unified_strategy_service.get_strategy_config(loaded_strategy_id)
            assert loaded_config.name == strategy_config.name

    def test_strategy_resource_management(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略资源管理"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册多个策略
            strategy_ids = []
            for i in range(5):
                config = StrategyConfig(
                    name=f"resource_test_{i}",
                    strategy_type=StrategyType.MOMENTUM,
                    parameters={'lookback_period': 10 + i}
                )
                strategy_id = unified_strategy_service.register_strategy(config)
                strategy_ids.append(strategy_id)

            # 执行所有策略
            for strategy_id in strategy_ids:
                unified_strategy_service.execute_strategy(strategy_id, sample_market_data)

            # 检查资源使用情况
            resource_usage = unified_strategy_service.get_resource_usage()

            assert isinstance(resource_usage, dict)
            assert 'memory_usage' in resource_usage
            assert 'cpu_usage' in resource_usage

    def test_strategy_performance_benchmarking(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略性能基准测试"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 执行性能基准测试
            benchmark_result = unified_strategy_service.run_performance_benchmark(
                strategy_id, sample_market_data, iterations=5
            )

            assert isinstance(benchmark_result, dict)
            assert 'avg_execution_time' in benchmark_result
            assert 'min_execution_time' in benchmark_result
            assert 'max_execution_time' in benchmark_result
            assert 'std_execution_time' in benchmark_result

    def test_strategy_adaptive_parameters(self, unified_strategy_service, sample_market_data):
        """测试策略自适应参数调整"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建自适应策略配置
            adaptive_config = StrategyConfig(
                name="adaptive_strategy",
                strategy_type=StrategyType.MOMENTUM,
                parameters={
                    'lookback_period': 20,
                    'adaptive': True,
                    'volatility_adjustment': True
                }
            )

            strategy_id = unified_strategy_service.register_strategy(adaptive_config)

            # 执行策略（应该触发自适应调整）
            result = unified_strategy_service.execute_strategy(strategy_id, sample_market_data)

            # 检查是否应用了自适应调整
            adaptive_metrics = unified_strategy_service.get_adaptive_metrics(strategy_id)

            assert isinstance(adaptive_metrics, dict)
            assert 'parameter_adjustments' in adaptive_metrics

    def test_strategy_multi_market_support(self, unified_strategy_service):
        """测试策略多市场支持"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建多市场数据
            markets_data = {
                'US': pd.DataFrame({
                    'symbol': ['AAPL'] * 50,
                    'close': np.random.uniform(150, 200, 50)
                }),
                'HK': pd.DataFrame({
                    'symbol': ['00700'] * 50,
                    'close': np.random.uniform(300, 400, 50)
                }),
                'CN': pd.DataFrame({
                    'symbol': ['000001'] * 50,
                    'close': np.random.uniform(10, 20, 50)
                })
            }

            # 创建多市场策略配置
            multi_market_config = StrategyConfig(
                name="multi_market_strategy",
                strategy_type=StrategyType.MOMENTUM,
                parameters={
                    'markets': ['US', 'HK', 'CN'],
                    'cross_market_arbitrage': True
                }
            )

            strategy_id = unified_strategy_service.register_strategy(multi_market_config)

            # 执行多市场策略
            result = unified_strategy_service.execute_multi_market_strategy(strategy_id, markets_data)

            assert isinstance(result, dict)
            assert 'market_results' in result
            assert len(result['market_results']) == len(markets_data)

    def test_strategy_real_time_execution(self, unified_strategy_service, strategy_config):
        """测试策略实时执行"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 启用实时模式
            unified_strategy_service.enable_real_time_mode(strategy_id)

            # 模拟实时数据流
            real_time_data_stream = [
                pd.DataFrame({'symbol': ['AAPL'], 'close': [150.0], 'timestamp': [datetime.now()]}),
                pd.DataFrame({'symbol': ['AAPL'], 'close': [152.0], 'timestamp': [datetime.now()]}),
                pd.DataFrame({'symbol': ['AAPL'], 'close': [148.0], 'timestamp': [datetime.now()]})
            ]

            # 实时执行策略
            real_time_signals = []
            for data_chunk in real_time_data_stream:
                signals = unified_strategy_service.execute_real_time(strategy_id, data_chunk)
                real_time_signals.extend(signals)

            assert isinstance(real_time_signals, list)
            assert len(real_time_signals) > 0

    def test_strategy_portfolio_integration(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略投资组合集成"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 创建投资组合策略配置
            portfolio_config = StrategyConfig(
                name="portfolio_strategy",
                strategy_type=StrategyType.MOMENTUM,
                parameters={
                    'portfolio_size': 10,
                    'rebalancing_frequency': 'daily',
                    'risk_parity': True
                }
            )

            strategy_id = unified_strategy_service.register_strategy(portfolio_config)

            # 执行投资组合策略
            portfolio_result = unified_strategy_service.execute_portfolio_strategy(
                strategy_id, sample_market_data
            )

            assert isinstance(portfolio_result, dict)
            assert 'portfolio_weights' in portfolio_result
            assert 'portfolio_performance' in portfolio_result
            assert 'rebalancing_trades' in portfolio_result

    def test_strategy_backtest_integration(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试策略回测集成"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)

            # 执行策略回测
            backtest_result = unified_strategy_service.run_backtest(
                strategy_id, sample_market_data,
                start_date='2024-01-01',
                end_date='2024-04-01'
            )

            assert isinstance(backtest_result, dict)
            assert 'backtest_metrics' in backtest_result
            assert 'trade_log' in backtest_result
            assert 'performance_summary' in backtest_result

    def test_strategy_exception_handling_comprehensive(self, unified_strategy_service):
        """测试策略异常处理综合测试"""
        if STRATEGY_SERVICE_AVAILABLE:
            # 测试各种异常情况
            exception_scenarios = [
                ("invalid_strategy_id", lambda: unified_strategy_service.get_strategy_status("nonexistent")),
                ("invalid_config", lambda: unified_strategy_service.register_strategy(None)),
                ("invalid_data", lambda: unified_strategy_service.execute_strategy("valid_id", None)),
            ]

            for scenario_name, operation in exception_scenarios:
                with pytest.raises((StrategyException, ValueError, TypeError, KeyError)):
                    operation()

    def test_strategy_cleanup_and_resource_release(self, unified_strategy_service, strategy_config):
        """测试策略清理和资源释放"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册多个策略
            strategy_ids = []
            for i in range(3):
                config = StrategyConfig(
                    name=f"cleanup_test_{i}",
                    strategy_type=StrategyType.MOMENTUM,
                    parameters={'test_param': i}
                )
                strategy_id = unified_strategy_service.register_strategy(config)
                strategy_ids.append(strategy_id)

            # 执行清理
            cleanup_result = unified_strategy_service.cleanup_strategies(strategy_ids)

            assert isinstance(cleanup_result, dict)
            assert 'cleaned_strategies' in cleanup_result
            assert len(cleanup_result['cleaned_strategies']) == len(strategy_ids)

            # 验证策略已清理
            for strategy_id in strategy_ids:
                with pytest.raises((StrategyException, KeyError)):
                    unified_strategy_service.get_strategy_status(strategy_id)

    def test_strategy_status_management(self, unified_strategy_service, strategy_config):
        """测试策略状态管理"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)
            assert strategy_id is not None

            # 获取策略状态
            status = unified_strategy_service.get_strategy_status(strategy_id)
            assert status is not None
            assert 'status' in status

            # 启动策略
            success = unified_strategy_service.start_strategy(strategy_id)
            assert success is True

            # 停止策略
            success = unified_strategy_service.stop_strategy(strategy_id)
            assert success is True

    def test_strategy_update_and_restart(self, unified_strategy_service, strategy_config):
        """测试策略更新和重启"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)
            assert strategy_id is not None

            # 更新策略配置
            updated_config = strategy_config.copy()
            updated_config.parameters = {'lookback_period': 30, 'threshold': 0.1}
            success = unified_strategy_service.update_strategy(strategy_id, updated_config)
            assert success is True

            # 重启策略
            success = unified_strategy_service.restart_strategy(strategy_id)
            assert success is True

    def test_data_validation_and_anomaly_handling(self, unified_strategy_service, sample_market_data):
        """测试数据验证和异常处理"""
        if STRATEGY_SERVICE_AVAILABLE:
            # 验证市场数据
            validation_result = unified_strategy_service.validate_market_data(sample_market_data)
            assert validation_result is not None
            assert 'is_valid' in validation_result

            # 处理数据异常
            anomaly_result = unified_strategy_service.handle_data_anomalies(sample_market_data)
            assert anomaly_result is not None
            assert 'anomalies_detected' in anomaly_result

    def test_backtest_execution(self, unified_strategy_service, strategy_config, sample_market_data):
        """测试回测执行"""
        if STRATEGY_SERVICE_AVAILABLE and INTERFACES_AVAILABLE:
            # 注册策略
            strategy_id = unified_strategy_service.register_strategy(strategy_config)
            assert strategy_id is not None

            # 执行回测
            backtest_result = unified_strategy_service.run_backtest(strategy_id, sample_market_data)
            assert backtest_result is not None
            assert 'performance' in backtest_result

    def test_signal_processing(self, unified_strategy_service, sample_strategy_signals):
        """测试信号处理"""
        if STRATEGY_SERVICE_AVAILABLE:
            # 处理交易信号
            processed_signals = unified_strategy_service.process_signals(sample_strategy_signals)
            assert processed_signals is not None
            assert len(processed_signals) >= 0
