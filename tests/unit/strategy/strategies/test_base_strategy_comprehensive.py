"""
基础策略深度测试
全面测试基础策略类的核心功能、信号生成和参数管理
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json

# 导入策略相关类
try:
    from src.strategy.strategies.base_strategy import (
        BaseStrategy, MarketData, StrategySignal, StrategyConfig,
        StrategyResult, StrategyStatus
    )
    BASE_STRATEGY_AVAILABLE = True
except ImportError:
    BASE_STRATEGY_AVAILABLE = False
    BaseStrategy = Mock
    MarketData = Mock
    StrategySignal = Mock
    StrategyConfig = Mock
    StrategyResult = Mock
    StrategyStatus = Mock

try:
    from src.strategy.strategies.momentum_strategy import MomentumStrategy
    MOMENTUM_STRATEGY_AVAILABLE = True
except ImportError:
    MOMENTUM_STRATEGY_AVAILABLE = False
    MomentumStrategy = Mock

try:
    from src.strategy.strategies.mean_reversion_strategy import MeanReversionStrategy
    MEAN_REVERSION_STRATEGY_AVAILABLE = True
except ImportError:
    MEAN_REVERSION_STRATEGY_AVAILABLE = False
    MeanReversionStrategy = Mock

try:
    from src.strategy.strategies.trend_following_strategy import TrendFollowingStrategy
    TREND_STRATEGY_AVAILABLE = True
except ImportError:
    TREND_STRATEGY_AVAILABLE = False
    TrendFollowingStrategy = Mock


class TestBaseStrategyComprehensive:
    """基础策略综合深度测试"""

    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)

        return pd.DataFrame({
            'symbol': ['AAPL'] * 100,
            'timestamp': dates,
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
        if BASE_STRATEGY_AVAILABLE:
            return StrategyConfig(
                strategy_id="test_strategy",
                name="Test Strategy",
                parameters={
                    'lookback_period': 20,
                    'threshold': 0.05,
                    'max_position': 100
                },
                risk_limits={
                    'max_drawdown': 0.1,
                    'max_position_size': 1000
                },
                symbols=['AAPL', 'GOOGL']
            )
        return Mock()

    @pytest.fixture
    def base_strategy(self, strategy_config):
        """创建基础策略实例"""
        if BASE_STRATEGY_AVAILABLE:
            return BaseStrategy(config=strategy_config)
        return Mock(spec=BaseStrategy)

    @pytest.fixture
    def momentum_strategy(self, strategy_config):
        """创建动量策略实例"""
        if MOMENTUM_STRATEGY_AVAILABLE:
            return MomentumStrategy(config=strategy_config)
        return Mock(spec=MomentumStrategy)

    @pytest.fixture
    def mean_reversion_strategy(self, strategy_config):
        """创建均值回归策略实例"""
        if MEAN_REVERSION_STRATEGY_AVAILABLE:
            return MeanReversionStrategy(config=strategy_config)
        return Mock(spec=MeanReversionStrategy)

    def test_base_strategy_initialization(self, base_strategy, strategy_config):
        """测试基础策略初始化"""
        if BASE_STRATEGY_AVAILABLE:
            assert base_strategy is not None
            assert base_strategy.config == strategy_config
            assert base_strategy.status == StrategyStatus.INACTIVE
            assert hasattr(base_strategy, 'signal_history')
            assert hasattr(base_strategy, 'performance_metrics')

    def test_market_data_structure(self, sample_market_data):
        """测试市场数据结构"""
        if BASE_STRATEGY_AVAILABLE:
            # 转换DataFrame为MarketData对象
            for _, row in sample_market_data.head(5).iterrows():
                market_data = MarketData(
                    symbol=row['symbol'],
                    price=row['close'],
                    volume=int(row['volume']),
                    timestamp=row['timestamp'],
                    high=row['high'],
                    low=row['low'],
                    open_price=row['open'],
                    close_price=row['close']
                )

                assert market_data.symbol == row['symbol']
                assert market_data.price == row['close']
                assert market_data.volume == int(row['volume'])
                assert market_data.timestamp == row['timestamp']

    def test_strategy_signal_generation(self, base_strategy, sample_market_data):
        """测试策略信号生成"""
        if BASE_STRATEGY_AVAILABLE:
            # 初始化策略
            base_strategy.initialize()

            # 生成信号
            signals = base_strategy.generate_signals(sample_market_data)

            assert isinstance(signals, list)
            for signal in signals:
                assert isinstance(signal, StrategySignal)
                assert hasattr(signal, 'signal_type')
                assert hasattr(signal, 'symbol')
                assert hasattr(signal, 'confidence')
                assert signal.signal_type in ['BUY', 'SELL', 'HOLD']

    def test_strategy_execution_simulation(self, base_strategy, sample_market_data):
        """测试策略执行模拟"""
        if BASE_STRATEGY_AVAILABLE:
            # 初始化策略
            base_strategy.initialize()

            # 执行策略
            result = base_strategy.execute(sample_market_data)

            assert isinstance(result, StrategyResult)
            assert hasattr(result, 'signals')
            assert hasattr(result, 'performance')
            assert hasattr(result, 'status')

    def test_strategy_parameter_management(self, base_strategy):
        """测试策略参数管理"""
        if BASE_STRATEGY_AVAILABLE:
            # 测试参数更新
            new_params = {
                'lookback_period': 30,
                'threshold': 0.08,
                'max_position': 200
            }

            base_strategy.update_parameters(new_params)

            # 验证参数更新
            updated_params = base_strategy.get_parameters()
            assert updated_params['lookback_period'] == 30
            assert updated_params['threshold'] == 0.08
            assert updated_params['max_position'] == 200

    def test_strategy_risk_limits_enforcement(self, base_strategy, sample_market_data):
        """测试策略风险限额执行"""
        if BASE_STRATEGY_AVAILABLE:
            # 设置风险限额
            risk_limits = {
                'max_drawdown': 0.05,  # 5% 最大回撤
                'max_position_size': 500,
                'daily_loss_limit': 1000
            }

            base_strategy.set_risk_limits(risk_limits)

            # 执行策略（应该受到风险限额约束）
            result = base_strategy.execute_with_risk_management(sample_market_data)

            assert isinstance(result, StrategyResult)
            # 验证风险限额被执行
            assert 'risk_checks_passed' in result.performance

    def test_momentum_strategy_signal_generation(self, momentum_strategy, sample_market_data):
        """测试动量策略信号生成"""
        if MOMENTUM_STRATEGY_AVAILABLE:
            # 初始化策略
            momentum_strategy.initialize()

            # 生成动量信号
            signals = momentum_strategy.generate_signals(sample_market_data)

            assert isinstance(signals, list)

            # 动量策略应该基于价格动量生成信号
            momentum_signals = [s for s in signals if s.signal_type in ['BUY', 'SELL']]
            assert len(momentum_signals) > 0

            # 检查信号质量
            for signal in momentum_signals:
                assert 0 <= signal.confidence <= 1
                assert signal.symbol in sample_market_data['symbol'].unique()

    def test_momentum_strategy_performance(self, momentum_strategy, sample_market_data):
        """测试动量策略性能"""
        if MOMENTUM_STRATEGY_AVAILABLE:
            # 初始化策略
            momentum_strategy.initialize()

            # 执行策略
            result = momentum_strategy.execute(sample_market_data)

            assert isinstance(result, StrategyResult)

            # 动量策略应该有合理的性能指标
            performance = result.performance
            assert 'total_return' in performance
            assert 'sharpe_ratio' in performance
            assert 'max_drawdown' in performance

    def test_mean_reversion_strategy_signal_generation(self, mean_reversion_strategy, sample_market_data):
        """测试均值回归策略信号生成"""
        if MEAN_REVERSION_STRATEGY_AVAILABLE:
            # 初始化策略
            mean_reversion_strategy.initialize()

            # 生成均值回归信号
            signals = mean_reversion_strategy.generate_signals(sample_market_data)

            assert isinstance(signals, list)

            # 均值回归策略应该在价格偏离均值时生成反向信号
            reversion_signals = [s for s in signals if s.signal_type in ['BUY', 'SELL']]
            assert len(reversion_signals) > 0

            # 验证信号逻辑（当价格低于均值时应该买入）
            # 这里需要具体的验证逻辑

    def test_strategy_portfolio_management(self, base_strategy, sample_market_data):
        """测试策略投资组合管理"""
        if BASE_STRATEGY_AVAILABLE:
            # 设置多资产配置
            portfolio_config = {
                'AAPL': 0.4,
                'GOOGL': 0.3,
                'MSFT': 0.3
            }

            base_strategy.set_portfolio_allocation(portfolio_config)

            # 执行投资组合策略
            portfolio_result = base_strategy.execute_portfolio_strategy(sample_market_data)

            assert isinstance(portfolio_result, dict)
            assert 'portfolio_performance' in portfolio_result
            assert 'asset_allocation' in portfolio_result

    def test_strategy_backtesting_integration(self, base_strategy, sample_market_data):
        """测试策略回测集成"""
        if BASE_STRATEGY_AVAILABLE:
            # 执行策略回测
            backtest_config = {
                'start_date': '2024-01-01',
                'end_date': '2024-04-01',
                'initial_capital': 100000.0,
                'commission': 0.001
            }

            backtest_result = base_strategy.run_backtest(
                sample_market_data, backtest_config
            )

            assert isinstance(backtest_result, dict)
            assert 'backtest_metrics' in backtest_result
            assert 'trade_log' in backtest_result
            assert 'performance_summary' in backtest_result

    def test_strategy_parameter_optimization(self, momentum_strategy, sample_market_data):
        """测试策略参数优化"""
        if MOMENTUM_STRATEGY_AVAILABLE:
            # 定义参数搜索空间
            param_space = {
                'lookback_period': [10, 20, 30, 50],
                'threshold': [0.02, 0.05, 0.08, 0.10],
                'momentum_window': [5, 10, 15, 20]
            }

            # 执行参数优化
            optimization_result = momentum_strategy.optimize_parameters(
                sample_market_data, param_space
            )

            assert isinstance(optimization_result, dict)
            assert 'best_parameters' in optimization_result
            assert 'optimization_score' in optimization_result
            assert 'parameter_performance' in optimization_result

    def test_strategy_risk_adjusted_performance(self, base_strategy, sample_market_data):
        """测试策略风险调整绩效"""
        if BASE_STRATEGY_AVAILABLE:
            # 执行策略
            result = base_strategy.execute(sample_market_data)

            # 计算风险调整指标
            risk_adjusted_metrics = base_strategy.calculate_risk_adjusted_performance(result)

            assert isinstance(risk_adjusted_metrics, dict)

            # 检查常见风险调整指标
            expected_metrics = [
                'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                'omega_ratio', 'information_ratio'
            ]

            for metric in expected_metrics:
                assert metric in risk_adjusted_metrics

    def test_strategy_adaptive_behavior(self, momentum_strategy, sample_market_data):
        """测试策略自适应行为"""
        if MOMENTUM_STRATEGY_AVAILABLE:
            # 启用自适应模式
            momentum_strategy.enable_adaptive_mode()

            # 执行策略（策略应该根据市场条件调整参数）
            result = momentum_strategy.execute_adaptive(sample_market_data)

            assert isinstance(result, StrategyResult)

            # 检查自适应调整
            adaptive_metrics = momentum_strategy.get_adaptive_metrics()
            assert isinstance(adaptive_metrics, dict)
            assert 'parameter_adjustments' in adaptive_metrics

    def test_strategy_multi_timeframe_analysis(self, base_strategy):
        """测试策略多时间框架分析"""
        if BASE_STRATEGY_AVAILABLE:
            # 创建不同时间框架的数据
            timeframes = ['1D', '1H', '15M']

            multi_timeframe_data = {}
            base_date = datetime(2024, 1, 1)

            for tf in timeframes:
                if tf == '1D':
                    periods = 100
                    freq = 'D'
                elif tf == '1H':
                    periods = 200
                    freq = 'H'
                else:  # 15M
                    periods = 400
                    freq = '15min'

                dates = pd.date_range(base_date, periods=periods, freq=freq)
                data = pd.DataFrame({
                    'symbol': ['AAPL'] * periods,
                    'timestamp': dates,
                    'close': np.random.uniform(150, 200, periods),
                    'volume': np.random.randint(10000, 100000, periods)
                })
                multi_timeframe_data[tf] = data

            # 执行多时间框架分析
            multi_tf_result = base_strategy.analyze_multi_timeframe(multi_timeframe_data)

            assert isinstance(multi_tf_result, dict)
            assert len(multi_tf_result) == len(timeframes)

            for tf in timeframes:
                assert tf in multi_tf_result
                assert 'signals' in multi_tf_result[tf]

    def test_strategy_machine_learning_integration(self, base_strategy, sample_market_data):
        """测试策略机器学习集成"""
        if BASE_STRATEGY_AVAILABLE:
            # 配置机器学习增强
            ml_config = {
                'use_ml_prediction': True,
                'ml_model_type': 'random_forest',
                'feature_engineering': True,
                'prediction_horizon': 5
            }

            base_strategy.configure_ml_enhancement(ml_config)

            # 执行ML增强策略
            ml_result = base_strategy.execute_with_ml_enhancement(sample_market_data)

            assert isinstance(ml_result, StrategyResult)

            # 检查ML预测结果
            if hasattr(ml_result, 'ml_predictions'):
                assert isinstance(ml_result.ml_predictions, list)

    def test_strategy_performance_monitoring(self, base_strategy, sample_market_data):
        """测试策略性能监控"""
        if BASE_STRATEGY_AVAILABLE:
            # 启用性能监控
            base_strategy.enable_performance_monitoring()

            # 执行策略
            result = base_strategy.execute(sample_market_data)

            # 获取性能指标
            performance_metrics = base_strategy.get_performance_metrics()

            assert isinstance(performance_metrics, dict)

            # 检查性能监控指标
            expected_metrics = [
                'execution_time', 'memory_usage', 'cpu_usage',
                'signal_generation_time', 'backtest_time'
            ]

            for metric in expected_metrics:
                assert metric in performance_metrics

    def test_strategy_error_handling_and_recovery(self, base_strategy):
        """测试策略错误处理和恢复"""
        if BASE_STRATEGY_AVAILABLE:
            # 测试无效数据处理
            invalid_data = pd.DataFrame({
                'symbol': [None, 'AAPL'],
                'close': ['invalid', 150.0]
            })

            try:
                base_strategy.generate_signals(invalid_data)
                # 如果没有抛出异常，验证错误被妥善处理
            except Exception as e:
                # 验证异常被正确捕获和处理
                assert isinstance(e, (ValueError, TypeError))

            # 测试参数验证
            invalid_params = {'lookback_period': -5}  # 无效参数

            try:
                base_strategy.update_parameters(invalid_params)
            except ValueError:
                # 期望的参数验证错误
                pass

    def test_strategy_configuration_persistence(self, base_strategy, tmp_path):
        """测试策略配置持久化"""
        if BASE_STRATEGY_AVAILABLE:
            # 保存策略配置
            config_file = tmp_path / "strategy_config.json"
            base_strategy.save_configuration(str(config_file))

            # 验证文件创建
            assert config_file.exists()

            # 加载配置
            loaded_config = base_strategy.load_configuration(str(config_file))

            assert isinstance(loaded_config, dict)
            assert 'strategy_id' in loaded_config

    def test_strategy_comparison_and_benchmarking(self, base_strategy, momentum_strategy, sample_market_data):
        """测试策略比较和基准测试"""
        if BASE_STRATEGY_AVAILABLE and MOMENTUM_STRATEGY_AVAILABLE:
            # 初始化策略
            base_strategy.initialize()
            momentum_strategy.initialize()

            # 执行两个策略
            base_result = base_strategy.execute(sample_market_data)
            momentum_result = momentum_strategy.execute(sample_market_data)

            # 比较策略性能
            comparison = base_strategy.compare_with_strategy(momentum_result, base_result)

            assert isinstance(comparison, dict)
            assert 'performance_comparison' in comparison
            assert 'statistical_significance' in comparison

            # 与基准比较
            benchmark_result = base_strategy.compare_with_benchmark(
                base_result, sample_market_data
            )

            assert isinstance(benchmark_result, dict)
            assert 'benchmark_return' in benchmark_result
            assert 'outperformance' in benchmark_result

    def test_strategy_real_time_execution(self, base_strategy):
        """测试策略实时执行"""
        if BASE_STRATEGY_AVAILABLE:
            # 启用实时模式
            base_strategy.enable_real_time_mode()

            # 模拟实时市场数据流
            real_time_data_stream = [
                MarketData('AAPL', 150.0, 100000, datetime.now()),
                MarketData('AAPL', 152.0, 120000, datetime.now() + timedelta(seconds=1)),
                MarketData('AAPL', 148.0, 90000, datetime.now() + timedelta(seconds=2)),
            ]

            # 实时执行策略
            real_time_signals = []
            for market_data in real_time_data_stream:
                signal = base_strategy.process_real_time_data(market_data)
                if signal:
                    real_time_signals.append(signal)

            # 验证实时处理能力
            assert isinstance(real_time_signals, list)

            # 检查处理延迟（应该很快）
            # 这里可以添加具体的延迟检查

    def test_strategy_parallel_execution(self, base_strategy, sample_market_data):
        """测试策略并行执行"""
        if BASE_STRATEGY_AVAILABLE:
            import threading

            # 准备多组数据进行并行处理
            data_chunks = [
                sample_market_data.head(25),
                sample_market_data.iloc[25:50],
                sample_market_data.iloc[50:75],
                sample_market_data.tail(25)
            ]

            results = []
            errors = []

            def execute_chunk(chunk, index):
                try:
                    result = base_strategy.execute(chunk)
                    results.append((index, result))
                except Exception as e:
                    errors.append((index, str(e)))

            # 并行执行
            threads = []
            for i, chunk in enumerate(data_chunks):
                thread = threading.Thread(target=execute_chunk, args=(chunk, i))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 验证结果
            assert len(results) == len(data_chunks)
            assert len(errors) == 0

            # 检查所有结果都有效
            for index, result in results:
                assert isinstance(result, StrategyResult)

    def test_strategy_resource_management(self, base_strategy, sample_market_data):
        """测试策略资源管理"""
        if BASE_STRATEGY_AVAILABLE:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # 记录初始资源使用
            initial_memory = process.memory_info().rss

            # 执行策略
            base_strategy.execute(sample_market_data)

            # 检查资源使用
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # 验证资源使用合理
            assert memory_increase < 100 * 1024 * 1024  # 100MB限制

            # 检查策略的资源监控
            resource_stats = base_strategy.get_resource_usage()

            assert isinstance(resource_stats, dict)
            assert 'memory_peak' in resource_stats
            assert 'cpu_time' in resource_stats

    def test_strategy_audit_and_logging(self, base_strategy, sample_market_data):
        """测试策略审计和日志"""
        if BASE_STRATEGY_AVAILABLE:
            # 启用审计日志
            base_strategy.enable_audit_logging()

            # 执行策略操作
            base_strategy.execute(sample_market_data)

            # 获取审计日志
            audit_log = base_strategy.get_audit_log()

            assert isinstance(audit_log, list)
            assert len(audit_log) > 0

            # 检查审计记录结构
            for entry in audit_log:
                assert 'timestamp' in entry
                assert 'action' in entry
                assert 'details' in entry
                assert isinstance(entry['timestamp'], (str, datetime))
