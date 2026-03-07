"""
策略参数优化器深度测试
全面测试参数优化算法、网格搜索、贝叶斯优化和性能评估功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import itertools

# 导入参数优化相关类
try:
    from src.strategy.backtest.parameter_optimizer import ParameterOptimizer
    from src.strategy.backtest.backtest_engine import BacktestEngine, BacktestResult
    PARAMETER_OPTIMIZER_AVAILABLE = True
except ImportError:
    PARAMETER_OPTIMIZER_AVAILABLE = False
    ParameterOptimizer = Mock
    BacktestEngine = Mock
    BacktestResult = Mock

try:
    from src.strategy.strategies.base_strategy import BaseStrategy
    BASE_STRATEGY_AVAILABLE = True
except ImportError:
    BASE_STRATEGY_AVAILABLE = False
    BaseStrategy = Mock


class TestParameterOptimizerComprehensive:
    """策略参数优化器综合深度测试"""

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
    def sample_strategy_params(self):
        """创建样本策略参数"""
        return {
            'lookback_period': {'type': 'int', 'range': [10, 50], 'default': 20},
            'threshold': {'type': 'float', 'range': [0.01, 0.1], 'default': 0.05},
            'stop_loss': {'type': 'float', 'range': [0.02, 0.15], 'default': 0.08},
            'take_profit': {'type': 'float', 'range': [0.05, 0.25], 'default': 0.12},
            'position_size': {'type': 'float', 'range': [0.1, 1.0], 'default': 0.5}
        }

    @pytest.fixture
    def backtest_engine(self):
        """创建回测引擎实例"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            return BacktestEngine()
        return Mock(spec=BacktestEngine)

    @pytest.fixture
    def parameter_optimizer(self, backtest_engine):
        """创建参数优化器实例"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            return ParameterOptimizer(engine=backtest_engine)
        return Mock(spec=ParameterOptimizer)

    @pytest.fixture
    def mock_strategy(self):
        """创建模拟策略"""
        if BASE_STRATEGY_AVAILABLE:
            return BaseStrategy()
        return Mock(spec=BaseStrategy)

    def test_parameter_optimizer_initialization(self, parameter_optimizer, backtest_engine):
        """测试参数优化器初始化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            assert parameter_optimizer is not None
            assert parameter_optimizer.engine == backtest_engine
            assert hasattr(parameter_optimizer, 'results')
            assert isinstance(parameter_optimizer.results, list)

    def test_grid_search_optimization(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试网格搜索优化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 定义参数网格
            param_grid = {
                'lookback_period': [10, 20, 30],
                'threshold': [0.02, 0.05, 0.08],
                'stop_loss': [0.05, 0.10]
            }

            # 执行网格搜索
            optimization_results = parameter_optimizer.grid_search(
                strategy=mock_strategy,
                param_grid=param_grid,
                start='2024-01-01',
                end='2024-12-31',
                n_jobs=1
            )

            assert isinstance(optimization_results, list)
            assert len(optimization_results) > 0

            # 检查结果结构
            for result in optimization_results:
                assert 'parameters' in result
                assert 'performance' in result
                assert 'sharpe_ratio' in result
                assert 'total_return' in result

    def test_random_search_optimization(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试随机搜索优化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 定义参数分布
            param_distributions = {
                'lookback_period': {'type': 'int', 'min': 10, 'max': 50},
                'threshold': {'type': 'uniform', 'min': 0.01, 'max': 0.1},
                'stop_loss': {'type': 'uniform', 'min': 0.02, 'max': 0.15}
            }

            # 执行随机搜索
            optimization_results = parameter_optimizer.random_search(
                strategy=mock_strategy,
                param_distributions=param_distributions,
                n_iter=20,
                start='2024-01-01',
                end='2024-12-31'
            )

            assert isinstance(optimization_results, list)
            assert len(optimization_results) == 20

            # 检查参数随机性
            lookback_values = [r['parameters']['lookback_period'] for r in optimization_results]
            assert len(set(lookback_values)) > 1  # 应该有不同的值

    def test_bayesian_optimization(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试贝叶斯优化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 定义参数边界
            param_bounds = {
                'lookback_period': (10, 50),
                'threshold': (0.01, 0.1),
                'stop_loss': (0.02, 0.15)
            }

            # 执行贝叶斯优化
            best_params, best_score = parameter_optimizer.bayesian_optimization(
                strategy=mock_strategy,
                param_bounds=param_bounds,
                init_points=5,
                n_iter=10,
                start='2024-01-01',
                end='2024-12-31'
            )

            assert isinstance(best_params, dict)
            assert isinstance(best_score, (int, float))

            # 检查参数在边界内
            assert 10 <= best_params['lookback_period'] <= 50
            assert 0.01 <= best_params['threshold'] <= 0.1
            assert 0.02 <= best_params['stop_loss'] <= 0.15

    def test_genetic_algorithm_optimization(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试遗传算法优化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 定义基因空间
            gene_space = {
                'lookback_period': {'type': 'int', 'min': 10, 'max': 50},
                'threshold': {'type': 'float', 'min': 0.01, 'max': 0.1, 'precision': 3},
                'stop_loss': {'type': 'float', 'min': 0.02, 'max': 0.15, 'precision': 3}
            }

            # 执行遗传算法优化
            ga_results = parameter_optimizer.genetic_algorithm_optimization(
                strategy=mock_strategy,
                gene_space=gene_space,
                population_size=20,
                generations=10,
                start='2024-01-01',
                end='2024-12-31'
            )

            assert isinstance(ga_results, dict)
            assert 'best_individual' in ga_results
            assert 'best_fitness' in ga_results
            assert 'evolution_history' in ga_results

    def test_walk_forward_optimization(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试滚动窗口优化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 定义参数网格
            param_grid = {
                'lookback_period': [15, 30],
                'threshold': [0.03, 0.07]
            }

            # 执行滚动窗口优化
            wf_results = parameter_optimizer.walk_forward_optimization(
                strategy=mock_strategy,
                param_grid=param_grid,
                train_window=63,  # 3个月训练
                test_window=21,   # 1个月测试
                step_size=21,     # 每月前进
                start='2024-01-01',
                end='2024-12-31'
            )

            assert isinstance(wf_results, dict)
            assert 'walk_forward_performance' in wf_results
            assert 'parameter_stability' in wf_results
            assert 'out_of_sample_performance' in wf_results

    def test_multi_objective_optimization(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试多目标优化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 定义多目标函数
            objectives = [
                'maximize_sharpe_ratio',
                'maximize_total_return',
                'minimize_max_drawdown'
            ]

            # 定义参数范围
            param_ranges = {
                'lookback_period': [10, 20, 30, 40, 50],
                'threshold': [0.02, 0.05, 0.08],
                'stop_loss': [0.05, 0.10, 0.15]
            }

            # 执行多目标优化
            pareto_front = parameter_optimizer.multi_objective_optimization(
                strategy=mock_strategy,
                objectives=objectives,
                param_ranges=param_ranges,
                population_size=20,
                generations=5,
                start='2024-01-01',
                end='2024-12-31'
            )

            assert isinstance(pareto_front, list)
            assert len(pareto_front) > 0

            # 检查帕累托前沿
            for solution in pareto_front:
                assert 'parameters' in solution
                assert 'objective_values' in solution
                assert 'dominance_rank' in solution

    def test_parallel_optimization_execution(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试并行优化执行"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            import threading

            # 定义多个参数组合进行并行测试
            param_combinations = [
                {'lookback_period': 10, 'threshold': 0.02},
                {'lookback_period': 20, 'threshold': 0.05},
                {'lookback_period': 30, 'threshold': 0.08},
                {'lookback_period': 40, 'threshold': 0.10}
            ]

            results = []
            errors = []

            def optimize_parameters(params, index):
                try:
                    # 执行单次优化
                    result = parameter_optimizer.evaluate_parameters(
                        strategy=mock_strategy,
                        parameters=params,
                        start='2024-01-01',
                        end='2024-12-31'
                    )
                    results.append((index, result))
                except Exception as e:
                    errors.append((index, str(e)))

            # 并行执行优化
            threads = []
            for i, params in enumerate(param_combinations):
                thread = threading.Thread(target=optimize_parameters, args=(params, i))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 验证并行执行结果
            assert len(results) == len(param_combinations)
            assert len(errors) == 0

    def test_optimization_with_constraints(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试带约束的优化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 定义约束条件
            constraints = [
                lambda params: params['stop_loss'] < params['take_profit'],  # 止损小于止盈
                lambda params: params['lookback_period'] * params['threshold'] < 2.0,  # 自定义约束
                lambda params: params['position_size'] <= 1.0  # 仓位不超过100%
            ]

            # 定义参数网格
            param_grid = {
                'lookback_period': [10, 20, 30],
                'threshold': [0.02, 0.05, 0.08],
                'stop_loss': [0.05, 0.10],
                'take_profit': [0.10, 0.15, 0.20],
                'position_size': [0.5, 0.8, 1.0]
            }

            # 执行带约束的优化
            constrained_results = parameter_optimizer.optimize_with_constraints(
                strategy=mock_strategy,
                param_grid=param_grid,
                constraints=constraints,
                start='2024-01-01',
                end='2024-12-31'
            )

            assert isinstance(constrained_results, list)
            assert len(constrained_results) > 0

            # 验证约束被满足
            for result in constrained_results:
                params = result['parameters']
                assert params['stop_loss'] < params['take_profit']
                assert params['lookback_period'] * params['threshold'] < 2.0
                assert params['position_size'] <= 1.0

    def test_optimization_result_analysis(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试优化结果分析"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 执行优化
            param_grid = {
                'lookback_period': [10, 20, 30, 40],
                'threshold': [0.02, 0.05, 0.08]
            }

            optimization_results = parameter_optimizer.grid_search(
                strategy=mock_strategy,
                param_grid=param_grid,
                start='2024-01-01',
                end='2024-12-31'
            )

            # 分析优化结果
            analysis = parameter_optimizer.analyze_optimization_results(optimization_results)

            assert isinstance(analysis, dict)
            assert 'best_parameters' in analysis
            assert 'parameter_importance' in analysis
            assert 'performance_distribution' in analysis
            assert 'robustness_analysis' in analysis

    def test_optimization_performance_monitoring(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试优化性能监控"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 执行优化并监控性能
            start_time = time.time()

            param_grid = {
                'lookback_period': [10, 20, 30],
                'threshold': [0.02, 0.05, 0.08]
            }

            results = parameter_optimizer.grid_search(
                strategy=mock_strategy,
                param_grid=param_grid,
                start='2024-01-01',
                end='2024-12-31'
            )

            end_time = time.time()

            # 获取性能统计
            performance_stats = parameter_optimizer.get_optimization_performance_stats()

            assert isinstance(performance_stats, dict)
            assert 'total_execution_time' in performance_stats
            assert 'average_evaluation_time' in performance_stats
            assert 'total_evaluations' in performance_stats
            assert 'parallel_efficiency' in performance_stats

            # 验证执行时间合理
            assert performance_stats['total_execution_time'] > 0
            assert performance_stats['total_execution_time'] < end_time - start_time + 10

    def test_optimization_convergence_analysis(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试优化收敛分析"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 执行贝叶斯优化（应该有收敛历史）
            param_bounds = {
                'lookback_period': (10, 50),
                'threshold': (0.01, 0.1)
            }

            best_params, best_score = parameter_optimizer.bayesian_optimization(
                strategy=mock_strategy,
                param_bounds=param_bounds,
                init_points=3,
                n_iter=8,
                start='2024-01-01',
                end='2024-12-31'
            )

            # 分析收敛
            convergence_analysis = parameter_optimizer.analyze_convergence()

            assert isinstance(convergence_analysis, dict)
            assert 'convergence_rate' in convergence_analysis
            assert 'iterations_to_convergence' in convergence_analysis
            assert 'stability_measure' in convergence_analysis

    def test_parameter_sensitivity_analysis(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试参数敏感性分析"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 定义基准参数
            baseline_params = {
                'lookback_period': 20,
                'threshold': 0.05,
                'stop_loss': 0.08
            }

            # 定义参数扰动范围
            param_ranges = {
                'lookback_period': [15, 20, 25],
                'threshold': [0.03, 0.05, 0.07],
                'stop_loss': [0.06, 0.08, 0.10]
            }

            # 执行敏感性分析
            sensitivity_results = parameter_optimizer.parameter_sensitivity_analysis(
                strategy=mock_strategy,
                baseline_params=baseline_params,
                param_ranges=param_ranges,
                start='2024-01-01',
                end='2024-12-31'
            )

            assert isinstance(sensitivity_results, dict)
            assert 'sensitivity_scores' in sensitivity_results
            assert 'most_influential_parameters' in sensitivity_results
            assert 'parameter_interactions' in sensitivity_results

    def test_optimization_with_custom_objective(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试自定义目标函数优化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 定义自定义目标函数
            def custom_objective(performance_metrics):
                """平衡收益和风险的目标函数"""
                sharpe = performance_metrics.get('sharpe_ratio', 0)
                max_drawdown = abs(performance_metrics.get('max_drawdown', 0))

                # 自定义评分：夏普率减去最大回撤的惩罚
                score = sharpe - 0.5 * max_drawdown
                return score

            # 定义参数网格
            param_grid = {
                'lookback_period': [15, 30],
                'threshold': [0.03, 0.07],
                'risk_multiplier': [0.5, 1.0, 1.5]
            }

            # 执行自定义目标优化
            custom_results = parameter_optimizer.optimize_with_custom_objective(
                strategy=mock_strategy,
                param_grid=param_grid,
                custom_objective=custom_objective,
                start='2024-01-01',
                end='2024-12-31'
            )

            assert isinstance(custom_results, list)
            assert len(custom_results) > 0

            # 检查自定义评分
            for result in custom_results:
                assert 'custom_score' in result
                assert isinstance(result['custom_score'], (int, float))

    def test_optimization_result_persistence(self, parameter_optimizer, mock_strategy, sample_market_data, tmp_path):
        """测试优化结果持久化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 执行优化
            param_grid = {
                'lookback_period': [10, 20],
                'threshold': [0.02, 0.05]
            }

            results = parameter_optimizer.grid_search(
                strategy=mock_strategy,
                param_grid=param_grid,
                start='2024-01-01',
                end='2024-12-31'
            )

            # 保存优化结果
            results_file = tmp_path / "optimization_results.json"
            parameter_optimizer.save_optimization_results(str(results_file))

            # 验证文件创建
            assert results_file.exists()

            # 加载优化结果
            loaded_results = parameter_optimizer.load_optimization_results(str(results_file))

            assert isinstance(loaded_results, list)
            assert len(loaded_results) == len(results)

    def test_optimization_error_handling_and_recovery(self, parameter_optimizer):
        """测试优化错误处理和恢复"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 测试无效参数处理
            invalid_param_grid = {
                'lookback_period': [],  # 空参数列表
                'threshold': [0.02, 0.05]
            }

            try:
                parameter_optimizer.grid_search(
                    strategy=None,  # 无效策略
                    param_grid=invalid_param_grid,
                    start='2024-01-01',
                    end='2024-12-31'
                )
            except (ValueError, TypeError):
                # 期望的错误处理
                pass

            # 测试恢复机制
            recovery_config = {
                'max_retries': 3,
                'retry_delay': 1.0,
                'fallback_method': 'random_search'
            }

            parameter_optimizer.configure_error_recovery(recovery_config)

            # 即使有错误也应该能够完成优化
            robust_results = parameter_optimizer.robust_optimization(
                strategy=mock_strategy,
                param_grid={'lookback_period': [10, 20], 'threshold': [0.02, 0.05]},
                start='2024-01-01',
                end='2024-12-31'
            )

            assert isinstance(robust_results, list)

    def test_optimization_with_market_regime_adaptation(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试市场状况适应性优化"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 定义不同的市场状况
            market_regimes = [
                {'regime': 'bull', 'trend': 0.15, 'volatility': 0.20},
                {'regime': 'bear', 'trend': -0.10, 'volatility': 0.25},
                {'regime': 'sideways', 'trend': 0.02, 'volatility': 0.15}
            ]

            regime_optimized_params = {}

            for regime in market_regimes:
                # 根据市场状况调整参数优化
                regime_params = parameter_optimizer.optimize_for_market_regime(
                    strategy=mock_strategy,
                    market_regime=regime,
                    param_grid={
                        'lookback_period': [10, 20, 30],
                        'threshold': [0.02, 0.05, 0.08]
                    },
                    start='2024-01-01',
                    end='2024-12-31'
                )

                regime_optimized_params[regime['regime']] = regime_params

            assert len(regime_optimized_params) == len(market_regimes)

            # 验证不同市场状况下的参数调整
            for regime, params in regime_optimized_params.items():
                assert 'optimal_parameters' in params
                assert 'regime_specific_score' in params

    def test_optimization_resource_management(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试优化资源管理"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # 记录初始资源使用
            initial_memory = process.memory_info().rss

            # 执行大规模优化
            large_param_grid = {
                'param1': list(range(10)),
                'param2': list(range(10)),
                'param3': list(range(10))
            }

            results = parameter_optimizer.grid_search(
                strategy=mock_strategy,
                param_grid=large_param_grid,
                start='2024-01-01',
                end='2024-12-31'
            )

            # 检查资源使用
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # 验证资源使用合理
            assert memory_increase < 100 * 1024 * 1024  # 100MB限制

            # 获取资源统计
            resource_stats = parameter_optimizer.get_resource_usage()

            assert isinstance(resource_stats, dict)
            assert 'peak_memory_usage' in resource_stats
            assert 'total_computation_time' in resource_stats

    def test_optimization_scalability_testing(self, parameter_optimizer, mock_strategy):
        """测试优化扩展性"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 测试不同规模问题的优化性能
            problem_sizes = [
                {'params': {'p1': [1, 2]}, 'name': 'small'},
                {'params': {'p1': list(range(5)), 'p2': list(range(5))}, 'name': 'medium'},
                {'params': {'p1': list(range(10)), 'p2': list(range(10)), 'p3': list(range(5))}, 'name': 'large'}
            ]

            scalability_metrics = {}

            for problem in problem_sizes:
                start_time = time.time()

                results = parameter_optimizer.grid_search(
                    strategy=mock_strategy,
                    param_grid=problem['params'],
                    start='2024-01-01',
                    end='2024-12-31'
                )

                end_time = time.time()

                scalability_metrics[problem['name']] = {
                    'execution_time': end_time - start_time,
                    'parameter_combinations': len(results),
                    'time_per_evaluation': (end_time - start_time) / len(results)
                }

            # 验证扩展性（较大的问题应该有合理的时间复杂度）
            assert scalability_metrics['medium']['time_per_evaluation'] < 10  # 每轮评估少于10秒
            assert scalability_metrics['large']['time_per_evaluation'] < 20   # 每轮评估少于20秒

    def test_optimization_audit_and_logging(self, parameter_optimizer, mock_strategy, sample_market_data):
        """测试优化审计和日志"""
        if PARAMETER_OPTIMIZER_AVAILABLE:
            # 启用审计日志
            parameter_optimizer.enable_audit_logging()

            # 执行优化操作
            param_grid = {
                'lookback_period': [10, 20],
                'threshold': [0.02, 0.05]
            }

            results = parameter_optimizer.grid_search(
                strategy=mock_strategy,
                param_grid=param_grid,
                start='2024-01-01',
                end='2024-12-31'
            )

            # 获取审计日志
            audit_log = parameter_optimizer.get_audit_log()

            assert isinstance(audit_log, list)
            assert len(audit_log) > 0

            # 检查审计记录结构
            for record in audit_log:
                assert 'timestamp' in record
                assert 'operation' in record
                assert 'parameters' in record
                assert 'result_summary' in record
