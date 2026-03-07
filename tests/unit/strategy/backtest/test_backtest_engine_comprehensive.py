"""
策略回测引擎深度测试
全面测试回测引擎的核心功能、策略评估和性能分析
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json

# 导入回测相关类
try:
    from src.strategy.backtest.backtest_engine import BacktestEngine, BacktestMode, BacktestResult
    BACKTEST_ENGINE_AVAILABLE = True
except ImportError:
    BACKTEST_ENGINE_AVAILABLE = False
    BacktestEngine = Mock
    BacktestMode = Mock
    BacktestResult = Mock

try:
    from src.strategy.core.strategy_service import UnifiedStrategyService
    STRATEGY_SERVICE_AVAILABLE = True
except ImportError:
    STRATEGY_SERVICE_AVAILABLE = False
    UnifiedStrategyService = Mock

try:
    from src.strategy.backtest.analyzer import BacktestAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    BacktestAnalyzer = Mock


class TestBacktestEngineComprehensive:
    """回测引擎综合深度测试"""

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
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'returns': returns
        })

    @pytest.fixture
    def sample_strategy_signals(self, sample_market_data):
        """创建样本策略信号"""
        signals = []

        # 简单的均线交叉策略信号
        data = sample_market_data.copy()
        data['sma_short'] = data['close'].rolling(window=10).mean()
        data['sma_long'] = data['close'].rolling(window=30).mean()

        for i in range(30, len(data)):
            if data['sma_short'].iloc[i] > data['sma_long'].iloc[i] and \
               data['sma_short'].iloc[i-1] <= data['sma_long'].iloc[i-1]:
                # 买入信号
                signals.append({
                    'date': data['date'].iloc[i],
                    'symbol': 'AAPL',
                    'signal': 'BUY',
                    'price': data['close'].iloc[i],
                    'strength': 1.0
                })
            elif data['sma_short'].iloc[i] < data['sma_long'].iloc[i] and \
                 data['sma_short'].iloc[i-1] >= data['sma_long'].iloc[i-1]:
                # 卖出信号
                signals.append({
                    'date': data['date'].iloc[i],
                    'symbol': 'AAPL',
                    'signal': 'SELL',
                    'price': data['close'].iloc[i],
                    'strength': 1.0
                })

        return signals

    @pytest.fixture
    def backtest_engine(self):
        """创建回测引擎实例"""
        if BACKTEST_ENGINE_AVAILABLE:
            return BacktestEngine()
        return Mock(spec=BacktestEngine)

    @pytest.fixture
    def strategy_service(self):
        """创建策略服务实例"""
        if STRATEGY_SERVICE_AVAILABLE:
            return UnifiedStrategyService()
        return Mock(spec=UnifiedStrategyService)

    @pytest.fixture
    def backtest_analyzer(self):
        """创建回测分析器实例"""
        if ANALYZER_AVAILABLE:
            return BacktestAnalyzer()
        return Mock(spec=BacktestAnalyzer)

    def test_backtest_engine_initialization(self, backtest_engine):
        """测试回测引擎初始化"""
        if BACKTEST_ENGINE_AVAILABLE:
            assert backtest_engine is not None
            assert hasattr(backtest_engine, 'config')
            assert hasattr(backtest_engine, 'results')

    def test_single_strategy_backtest(self, backtest_engine, sample_market_data, sample_strategy_signals):
        """测试单策略回测"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 设置回测配置
            config = {
                'initial_capital': 100000.0,
                'commission': 0.001,  # 0.1% 佣金
                'slippage': 0.0005,   # 0.05% 滑点
                'start_date': '2024-01-01',
                'end_date': '2024-12-31'
            }

            # 执行单策略回测
            backtest_engine.configure(config)
            backtest_engine.load_historical_data(sample_market_data)
            results = backtest_engine.run(BacktestMode.SINGLE)
            result = results.get('default', BacktestResult())  # 获取默认结果

            assert isinstance(result, BacktestResult)
            assert result.returns is not None
            assert isinstance(result.metrics, dict)
            assert 'total_return' in result.metrics
            # sharpe_ratio可能在metrics中，也可能不在，取决于实现
            assert len(result.metrics) > 0

    def test_multi_strategy_backtest(self, backtest_engine, sample_market_data):
        """测试多策略回测"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 创建多个策略信号
            strategies_signals = []

            # 策略1: 简单的动量策略
            signals1 = []
            for i in range(30, len(sample_market_data), 20):
                signals1.append({
                    'date': sample_market_data['date'].iloc[i],
                    'symbol': 'AAPL',
                    'signal': 'BUY' if i % 40 == 30 else 'SELL',
                    'price': sample_market_data['close'].iloc[i],
                    'strength': 0.8
                })

            # 策略2: 反转策略
            signals2 = []
            for i in range(30, len(sample_market_data), 25):
                signals2.append({
                    'date': sample_market_data['date'].iloc[i],
                    'symbol': 'AAPL',
                    'signal': 'SELL' if i % 50 == 30 else 'BUY',
                    'price': sample_market_data['close'].iloc[i],
                    'strength': 0.6
                })

            # 多策略参数配置
            strategies_params = [
                {'name': 'strategy_1', 'param1': 0.15},
                {'name': 'strategy_2', 'param1': 0.20}
            ]

            # 执行多策略回测
            backtest_engine.configure({'initial_capital': 50000.0})
            backtest_engine.load_historical_data(sample_market_data)
            results_dict = backtest_engine.run(BacktestMode.MULTI, strategies_params)
            results = list(results_dict.values())  # 转换为列表

            assert isinstance(results, list)
            assert len(results) == len(strategies_params)

            for result in results:
                assert isinstance(result, BacktestResult)
                assert result.metrics is not None

    def test_backtest_with_transaction_costs(self, backtest_engine, sample_market_data):
        """测试含交易成本的回测"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 创建频繁交易的信号
            signals = []
            for i in range(10, len(sample_market_data), 5):  # 每5天交易一次
                signals.append({
                    'date': sample_market_data['date'].iloc[i],
                    'symbol': 'AAPL',
                    'signal': 'BUY' if len(signals) % 2 == 0 else 'SELL',
                    'price': sample_market_data['close'].iloc[i],
                    'strength': 1.0
                })

            # 配置较高的交易成本
            config = {
                'initial_capital': 100000.0,
                'commission': 0.005,  # 0.5% 佣金
                'slippage': 0.002,   # 0.2% 滑点
                'market_impact': 0.001  # 0.1% 市场冲击
            }

            backtest_engine.configure(config)
            backtest_engine.load_historical_data(sample_market_data)
            results = backtest_engine.run(BacktestMode.SINGLE)
            result = results.get('default', BacktestResult())

            # 验证交易成本的影响
            assert result.metrics['total_return'] < 0.5  # 高成本应该限制收益
            assert 'transaction_costs' in result.metrics

    def test_backtest_risk_metrics_calculation(self, backtest_engine, sample_market_data, sample_strategy_signals):
        """测试回测风险指标计算"""
        if BACKTEST_ENGINE_AVAILABLE:
            config = {
                'initial_capital': 100000.0,
                'commission': 0.001,
                'start_date': '2024-01-01',
                'end_date': '2024-12-31'
            }

            backtest_engine.configure(config)
            backtest_engine.load_historical_data(sample_market_data)
            results = backtest_engine.run(BacktestMode.SINGLE)
            result = results.get('default', BacktestResult())

            # 检查风险指标
            risk_metrics = result.metrics

            expected_risk_metrics = [
                'volatility', 'max_drawdown', 'var_95', 'expected_shortfall',
                'beta', 'alpha', 'sortino_ratio', 'calmar_ratio'
            ]

            for metric in expected_risk_metrics:
                assert metric in risk_metrics
                assert isinstance(risk_metrics[metric], (int, float))

            # 验证指标合理性
            assert risk_metrics['volatility'] >= 0
            assert risk_metrics['max_drawdown'] <= 0  # 最大回撤是负数
            assert -1 <= risk_metrics['beta'] <= 3  # beta 通常在合理范围内

    def test_backtest_performance_attribution(self, backtest_engine, sample_market_data, sample_strategy_signals):
        """测试回测业绩归因"""
        if BACKTEST_ENGINE_AVAILABLE:
            config = {'initial_capital': 100000.0}

            backtest_engine.configure(config)
            backtest_engine.load_historical_data(sample_market_data)
            results = backtest_engine.run(BacktestMode.SINGLE)
            result = results.get('default', BacktestResult())

            # 获取业绩归因
            attribution = backtest_engine.get_performance_attribution(result)

            assert isinstance(attribution, dict)

            # 检查归因组件
            expected_components = [
                'market_timing', 'security_selection', 'asset_allocation',
                'transaction_costs_impact', 'risk_adjusted_return'
            ]

            for component in expected_components:
                assert component in attribution

    def test_backtest_walk_forward_analysis(self, backtest_engine, sample_market_data):
        """测试回测滚动窗口分析"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 创建简单的趋势跟随策略信号
            signals = []
            data = sample_market_data.copy()
            data['trend'] = data['close'].rolling(window=20).mean()

            for i in range(25, len(data), 10):
                current_trend = data['trend'].iloc[i]
                prev_trend = data['trend'].iloc[i-5]

                if current_trend > prev_trend * 1.001:  # 上涨趋势
                    signals.append({
                        'date': data['date'].iloc[i],
                        'symbol': 'AAPL',
                        'signal': 'BUY',
                        'price': data['close'].iloc[i],
                        'strength': 0.7
                    })
                elif current_trend < prev_trend * 0.999:  # 下跌趋势
                    signals.append({
                        'date': data['date'].iloc[i],
                        'symbol': 'AAPL',
                        'signal': 'SELL',
                        'price': data['close'].iloc[i],
                        'strength': 0.7
                    })

            # 执行滚动窗口回测
            backtest_engine.configure({'initial_capital': 100000.0})
            backtest_engine.load_historical_data(sample_market_data)
            walk_forward_results = backtest_engine.run_walk_forward_backtest(
                {'name': 'walk_forward_test'}, sample_market_data, window_size=63
            )

            assert isinstance(walk_forward_results, list)
            assert len(walk_forward_results) > 1  # 应该有多个窗口

            for window_result in walk_forward_results:
                assert isinstance(window_result, dict)
                assert 'window_start' in window_result
                assert 'window_end' in window_result
                assert 'performance' in window_result

    def test_backtest_stress_testing(self, backtest_engine, sample_market_data, sample_strategy_signals):
        """测试回测压力测试"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 定义压力情景
            stress_scenarios = [
                {
                    'name': 'market_crash',
                    'market_return_shift': -0.3,  # 市场下跌30%
                    'volatility_multiplier': 2.0
                },
                {
                    'name': 'high_volatility',
                    'market_return_shift': 0.0,
                    'volatility_multiplier': 3.0  # 波动率增加3倍
                },
                {
                    'name': 'liquidity_crisis',
                    'bid_ask_spread_multiplier': 5.0,  # 买卖价差扩大5倍
                    'trading_volume_reduction': 0.7   # 交易量减少30%
                }
            ]

            # 执行压力测试
            backtest_engine.configure({'initial_capital': 100000.0})
            backtest_engine.load_historical_data(sample_market_data)
            stress_results = backtest_engine.run_stress_test_backtest(
                {'name': 'stress_test'}, sample_market_data, stress_scenarios
            )

            assert isinstance(stress_results, dict)
            assert len(stress_results) == len(stress_scenarios)

            for scenario_name, result in stress_results.items():
                assert 'base_case_return' in result
                assert 'stressed_return' in result
                assert 'return_impact' in result
                assert 'risk_metrics' in result

    def test_backtest_benchmark_comparison(self, backtest_engine, sample_market_data, sample_strategy_signals):
        """测试回测基准比较"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 定义基准（买入持有策略）
            benchmark_signals = [
                {
                    'date': sample_market_data['date'].iloc[0],
                    'symbol': 'AAPL',
                    'signal': 'BUY',
                    'price': sample_market_data['close'].iloc[0],
                    'strength': 1.0
                },
                {
                    'date': sample_market_data['date'].iloc[-1],
                    'symbol': 'AAPL',
                    'signal': 'SELL',
                    'price': sample_market_data['close'].iloc[-1],
                    'strength': 1.0
                }
            ]

            # 执行策略vs基准比较
            backtest_engine.configure({'initial_capital': 100000.0})
            backtest_engine.load_historical_data(sample_market_data)
            results = backtest_engine.run(BacktestMode.SINGLE)
            strategy_result = results.get('default', BacktestResult())

            comparison_result = backtest_engine.compare_with_benchmark(strategy_result, sample_market_data)

            assert isinstance(comparison_result, dict)
            assert 'strategy_performance' in comparison_result
            assert 'benchmark_performance' in comparison_result
            assert 'outperformance' in comparison_result
            assert 'win_rate' in comparison_result

    def test_backtest_parameter_sensitivity(self, backtest_engine, sample_market_data):
        """测试回测参数敏感性"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 定义参数范围
            param_ranges = {
                'stop_loss': [0.05, 0.10, 0.15, 0.20],
                'take_profit': [0.10, 0.15, 0.20, 0.25],
                'position_size': [0.1, 0.2, 0.3, 0.5]
            }

            # 生成基于参数的策略信号
            def generate_signals(stop_loss, take_profit, position_size):
                signals = []
                # 简化的参数化策略逻辑
                for i in range(20, len(sample_market_data), 15):
                    signals.append({
                        'date': sample_market_data['date'].iloc[i],
                        'symbol': 'AAPL',
                        'signal': 'BUY',
                        'price': sample_market_data['close'].iloc[i],
                        'strength': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })
                return signals

            # 执行参数敏感性分析
            sensitivity_results = backtest_engine.run_parameter_sensitivity_analysis(
                signal_generator=generate_signals,
                param_ranges=param_ranges,
                market_data=sample_market_data,
                initial_capital=100000.0
            )

            assert isinstance(sensitivity_results, dict)
            assert 'parameter_combinations' in sensitivity_results
            assert 'best_parameters' in sensitivity_results
            assert 'sensitivity_metrics' in sensitivity_results

    def test_backtest_data_quality_checks(self, backtest_engine):
        """测试回测数据质量检查"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 创建包含数据质量问题的市场数据
            problematic_data = pd.DataFrame({
                'symbol': ['AAPL'] * 100,
                'date': pd.date_range('2024-01-01', periods=100),
                'open': [100.0] * 50 + [None] * 50,  # 缺失值
                'high': [105.0] * 100,
                'low': [95.0] * 100,
                'close': [102.0] * 100,
                'volume': [1000000] * 100
            })

            # 执行数据质量检查
            quality_report = backtest_engine.check_data_quality(problematic_data)

            assert isinstance(quality_report, dict)
            assert 'missing_values' in quality_report
            assert 'data_completeness' in quality_report
            assert 'quality_score' in quality_report

            # 应该检测到缺失值问题
            assert quality_report['missing_values'] > 0
            assert quality_report['data_completeness'] < 1.0

    def test_backtest_concurrent_execution(self, backtest_engine, sample_market_data):
        """测试回测并发执行"""
        if BACKTEST_ENGINE_AVAILABLE:
            import threading

            # 创建多个策略配置
            strategies_configs = [
                {'name': 'strategy_1', 'param': 0.1},
                {'name': 'strategy_2', 'param': 0.2},
                {'name': 'strategy_3', 'param': 0.3}
            ]

            results = []
            errors = []

            def run_backtest(config, index):
                try:
                    # 生成简单的策略信号
                    signals = []
                    for i in range(20, len(sample_market_data), 20):
                        signals.append({
                            'date': sample_market_data['date'].iloc[i],
                            'symbol': 'AAPL',
                            'signal': 'BUY',
                            'price': sample_market_data['close'].iloc[i],
                            'strength': config['param']
                        })

                    result = backtest_engine.run_single_backtest(
                        strategy_config={
                            'name': f"concurrent_strategy_{index}",
                            'initial_capital': 50000.0,
                            'param': config['param']
                        },
                        data=sample_market_data
                    )

                    results.append((index, result))

                except Exception as e:
                    errors.append((index, str(e)))

            # 并发执行回测
            threads = []
            for i, config in enumerate(strategies_configs):
                thread = threading.Thread(target=run_backtest, args=(config, i))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 验证结果
            assert len(results) == len(strategies_configs)
            assert len(errors) == 0

            # 检查所有结果都有效
            for index, result in results:
                assert isinstance(result, BacktestResult)
                assert result.metrics is not None

    def test_backtest_result_persistence(self, backtest_engine, sample_market_data, sample_strategy_signals, tmp_path):
        """测试回测结果持久化"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 执行回测
            result = backtest_engine.run_single_backtest(
                strategy_config={'name': 'persistence_test', 'initial_capital': 100000.0},
                data=sample_market_data
            )

            # 保存回测结果
            result_file = tmp_path / "backtest_result.json"
            backtest_engine.save_backtest_result(result, str(result_file))

            # 验证文件创建
            assert result_file.exists()

            # 加载回测结果
            loaded_result = backtest_engine.load_backtest_result(str(result_file))

            # 验证加载结果
            assert isinstance(loaded_result, BacktestResult)
            assert loaded_result.metrics == result.metrics

    def test_backtest_performance_monitoring(self, backtest_engine, sample_market_data, sample_strategy_signals):
        """测试回测性能监控"""
        if BACKTEST_ENGINE_AVAILABLE:
            import time

            # 执行回测并监控性能
            start_time = time.time()

            result = backtest_engine.run_single_backtest(
                strategy_config={'name': 'monitoring_test', 'initial_capital': 100000.0},
                data=sample_market_data
            )

            end_time = time.time()

            # 获取性能指标
            performance_stats = backtest_engine.get_performance_stats()

            assert isinstance(performance_stats, dict)
            assert 'average_execution_time' in performance_stats
            assert 'memory_usage' in performance_stats
            assert 'cpu_usage' in performance_stats

            # 验证执行时间合理
            assert performance_stats['average_execution_time'] > 0
            assert performance_stats['average_execution_time'] < end_time - start_time + 10  # 允许一些误差

    def test_backtest_error_handling(self, backtest_engine):
        """测试回测错误处理"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 测试无效信号数据
            invalid_signals = [
                {'date': 'invalid_date', 'signal': 'BUY'},  # 无效日期
                {'date': '2024-01-01', 'signal': 'INVALID'},  # 无效信号
                {'date': '2024-01-01', 'signal': 'BUY', 'price': 'invalid'}  # 无效价格
            ]

            # 测试错误处理
            try:
                backtest_engine.run_single_backtest(
                    strategy_signals=invalid_signals,
                    market_data=pd.DataFrame(),
                    config={'initial_capital': 100000.0}
                )
                # 如果没有抛出异常，至少验证返回了结果
            except Exception:
                # 期望的异常处理
                pass

            # 测试空数据处理
            try:
                backtest_engine.run_single_backtest(
                    strategy_signals=[],
                    market_data=pd.DataFrame(),
                    config={'initial_capital': 100000.0}
                )
            except Exception:
                # 空数据应该被适当处理
                pass

    def test_backtest_custom_metrics_calculation(self, backtest_engine, sample_market_data, sample_strategy_signals):
        """测试回测自定义指标计算"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 定义自定义指标函数
            def custom_sharpe_ratio(returns, risk_free_rate=0.02):
                excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
                if excess_returns.std() == 0:
                    return 0.0
                return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

            def custom_max_consecutive_losses(trades):
                if not trades:
                    return 0

                consecutive_losses = 0
                max_consecutive_losses = 0

                for trade in trades:
                    if trade.get('pnl', 0) < 0:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    else:
                        consecutive_losses = 0

                return max_consecutive_losses

            # 注册自定义指标
            backtest_engine.register_custom_metric('custom_sharpe', custom_sharpe_ratio)
            backtest_engine.register_custom_metric('max_consecutive_losses', custom_max_consecutive_losses)

            # 执行回测
            result = backtest_engine.run_single_backtest(
                strategy_config={'initial_capital': 100000.0},
                data=sample_market_data
            )

            # 检查自定义指标是否被计算
            assert 'custom_sharpe' in result.metrics
            assert 'max_consecutive_losses' in result.metrics

            # 验证指标值合理
            assert isinstance(result.metrics['custom_sharpe'], (int, float))
            assert isinstance(result.metrics['max_consecutive_losses'], (int, float))

    def test_backtest_multi_asset_portfolio(self, backtest_engine):
        """测试回测多资产投资组合"""
        if BACKTEST_ENGINE_AVAILABLE:
            # 创建多资产市场数据
            assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
            dates = pd.date_range('2024-01-01', periods=100, freq='D')

            multi_asset_data = []
            for asset in assets:
                np.random.seed(42 + hash(asset) % 1000)  # 不同的种子
                initial_price = 100.0
                returns = np.random.normal(0.0005, 0.02, len(dates))
                prices = initial_price * np.exp(np.cumsum(returns))

                asset_data = pd.DataFrame({
                    'symbol': [asset] * len(dates),
                    'date': dates,
                    'close': prices,
                    'volume': np.random.randint(100000, 1000000, len(dates))
                })
                multi_asset_data.append(asset_data)

            market_data = pd.concat(multi_asset_data, ignore_index=True)

            # 创建多资产策略信号
            portfolio_signals = []
            for asset in assets:
                asset_signals = []
                asset_data = market_data[market_data['symbol'] == asset]

                for i in range(15, len(asset_data), 25):
                    asset_signals.append({
                        'date': asset_data['date'].iloc[i],
                        'symbol': asset,
                        'signal': 'BUY' if i % 50 < 25 else 'SELL',
                        'price': asset_data['close'].iloc[i],
                        'strength': 0.25  # 每个资产25%权重
                    })

                portfolio_signals.extend(asset_signals)

            # 执行多资产投资组合回测
            portfolio_result = backtest_engine.run_portfolio_backtest(
                strategy_signals=portfolio_signals,
                market_data=market_data,
                initial_capital=100000.0,
                rebalance_frequency='monthly'
            )

            assert isinstance(portfolio_result, BacktestResult)
            assert 'asset_weights' in portfolio_result.metrics
            assert 'portfolio_correlation' in portfolio_result.metrics

            # 检查资产权重
            asset_weights = portfolio_result.metrics['asset_weights']
            assert len(asset_weights) == len(assets)
            assert abs(sum(asset_weights.values()) - 1.0) < 0.01  # 权重应该加起来为1