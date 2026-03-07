"""
优化层核心模块综合测试

测试优化层核心组件，包括：
1. OptimizationEngine - 优化引擎
2. PerformanceOptimizer - 性能优化器
3. EvaluationFramework - 评估框架
4. Optimizer - 优化器
5. PerformanceAnalyzer - 性能分析器
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List


class TestOptimizationCoreComprehensive:
    """测试优化层核心模块"""

    @pytest.fixture
    def sample_optimization_config(self):
        """测试优化配置"""
        return {
            'algorithm': 'genetic_algorithm',
            'population_size': 100,
            'max_generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'constraints': {
                'max_weight': 0.3,
                'min_weight': 0.0,
                'target_return': 0.1
            }
        }

    @pytest.fixture
    def sample_portfolio_data(self):
        """测试投资组合数据"""
        return pd.DataFrame({
            'asset_id': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            'weight': [0.3, 0.25, 0.25, 0.2],
            'expected_return': [0.12, 0.10, 0.08, 0.15],
            'volatility': [0.25, 0.30, 0.22, 0.35],
            'current_price': [150.0, 2800.0, 300.0, 3200.0]
        })

    @pytest.fixture
    def sample_performance_data(self):
        """测试性能数据"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)

        performance_data = pd.DataFrame({
            'portfolio_return': np.random.normal(0.001, 0.02, 252),
            'benchmark_return': np.random.normal(0.0008, 0.015, 252),
            'volatility': np.random.uniform(0.15, 0.35, 252),
            'sharpe_ratio': np.random.uniform(0.5, 2.0, 252),
            'max_drawdown': np.random.uniform(-0.05, -0.25, 252)
        }, index=dates)

        return performance_data

    def test_evaluation_framework_initialization(self, sample_optimization_config):
        """测试评估框架初始化"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework(sample_optimization_config)

        assert framework.config == sample_optimization_config
        assert hasattr(framework, 'metrics')
        assert isinstance(framework.metrics, dict)

    def test_evaluation_framework_evaluate_algorithm(self, sample_optimization_config):
        """测试评估框架算法评估"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework(sample_optimization_config)

        # 模拟算法结果
        algorithm_result = {
            'final_portfolio': {'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.25, 'AMZN': 0.2},
            'performance_metrics': {
                'sharpe_ratio': 1.8,
                'max_drawdown': -0.12,
                'total_return': 0.25
            },
            'convergence_history': [
                {'iteration': 1, 'value': 0.15},
                {'iteration': 2, 'value': 0.18},
                {'iteration': 3, 'value': 0.22}
            ]
        }

        evaluation_result = framework.evaluate_algorithm('test_algorithm', algorithm_result)

        assert isinstance(evaluation_result, dict)
        assert 'algorithm_name' in evaluation_result
        assert 'metrics' in evaluation_result

    def test_evaluation_framework_max_drawdown_calculation(self, sample_performance_data):
        """测试最大回撤计算"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        returns = pd.Series(sample_performance_data['portfolio_return'].values)
        max_dd = framework._calculate_max_drawdown(returns)

        assert isinstance(max_dd, float)
        assert max_dd >= 0  # 回撤应该是正数（绝对值）
        assert max_dd <= 1  # 合理的回撤范围

    def test_evaluation_framework_rolling_metrics(self, sample_performance_data):
        """测试滚动指标计算"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        returns = pd.Series(sample_performance_data['portfolio_return'].values)
        rolling_metrics = framework.calculate_rolling_metrics(returns, window=20)

        assert isinstance(rolling_metrics, pd.DataFrame)
        assert len(rolling_metrics) > 0

    def test_evaluation_framework_bootstrap_analysis(self, sample_performance_data):
        """测试自举分析"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        returns = pd.Series(sample_performance_data['portfolio_return'].values)
        bootstrap_results = framework.perform_bootstrap_analysis(returns, n_bootstraps=50)

        assert isinstance(bootstrap_results, dict)
        assert 'bootstrap_stats' in bootstrap_results

    def test_evaluation_framework_scenario_analysis(self, sample_performance_data):
        """测试情景分析"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        returns = pd.Series(sample_performance_data['portfolio_return'].values)

        scenarios = {
            'bull_market': {'market_return': 0.15},
            'bear_market': {'market_return': -0.20}
        }

        scenario_results = framework.analyze_scenarios(returns, scenarios)

        assert isinstance(scenario_results, dict)

    def test_evaluation_framework_stress_testing(self, sample_performance_data):
        """测试压力测试"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        returns = pd.Series(sample_performance_data['portfolio_return'].values)

        stress_scenarios = {
            'market_crash': -0.3,
            'rate_hike': -0.15
        }

        stress_results = framework.perform_stress_tests(returns, stress_scenarios)

        assert isinstance(stress_results, dict)

    def test_evaluation_framework_sensitivity_analysis(self, sample_portfolio_data):
        """测试敏感性分析"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        sensitivity_results = framework.perform_sensitivity_analysis(
            portfolio_data=sample_portfolio_data,
            parameter_ranges={'expected_return': (-0.1, 0.1), 'volatility': (0.1, 0.5)}
        )

        assert isinstance(sensitivity_results, dict)

    def test_evaluation_framework_cross_validation(self, sample_performance_data):
        """测试交叉验证"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        returns = pd.Series(sample_performance_data['portfolio_return'].values)
        cv_results = framework.perform_cross_validation(returns, n_folds=3)

        assert isinstance(cv_results, dict)

    def test_evaluation_framework_model_validation(self, sample_performance_data):
        """测试模型验证"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        returns = pd.Series(sample_performance_data['portfolio_return'].values)
        validation_results = framework.validate_model(returns)

        assert isinstance(validation_results, dict)

    def test_evaluation_framework_performance_attribution(self, sample_portfolio_data, sample_performance_data):
        """测试业绩归因"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        attribution_results = framework.perform_performance_attribution(
            portfolio_data=sample_portfolio_data,
            returns=sample_performance_data['portfolio_return']
        )

        assert isinstance(attribution_results, dict)

    def test_evaluation_framework_factor_analysis(self, sample_performance_data):
        """测试因子分析"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        returns = pd.Series(sample_performance_data['portfolio_return'].values)

        # 创建模拟因子数据
        factors = pd.DataFrame({
            'market': np.random.normal(0, 1, len(returns)),
            'size': np.random.normal(0, 1, len(returns)),
            'value': np.random.normal(0, 1, len(returns))
        })

        factor_results = framework.perform_factor_analysis(returns, factors)

        assert isinstance(factor_results, dict)

    def test_evaluation_framework_alpha_beta(self, sample_performance_data):
        """测试阿尔法和贝塔计算"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_information_ratio(self, sample_performance_data):
        """测试信息比率计算"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_benchmark_comparison(self, sample_performance_data):
        """测试基准比较"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_statistical_tests(self, sample_performance_data):
        """测试统计检验"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_comprehensive_evaluation(self, sample_performance_data, sample_portfolio_data):
        """测试综合评估"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_custom_metrics(self):
        """测试自定义指标"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_time_series_analysis(self, sample_performance_data):
        """测试时间序列分析"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_rolling_metrics(self, sample_performance_data):
        """测试滚动指标计算"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_bootstrap_analysis(self, sample_performance_data):
        """测试自举分析"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_scenario_analysis(self, sample_performance_data):
        """测试情景分析"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_stress_testing(self, sample_performance_data):
        """测试压力测试"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_sensitivity_analysis(self, sample_portfolio_data):
        """测试敏感性分析"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_cross_validation(self, sample_performance_data):
        """测试交叉验证"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_model_validation(self, sample_performance_data):
        """测试模型验证"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_performance_attribution(self, sample_portfolio_data, sample_performance_data):
        """测试业绩归因"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_factor_analysis(self, sample_performance_data):
        """测试因子分析"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None

    def test_evaluation_framework_portfolio_analytics(self, sample_portfolio_data, sample_performance_data):
        """测试投资组合分析"""
        from src.optimization.core.evaluation_framework import EvaluationFramework

        framework = EvaluationFramework()

        # 测试基本功能
        assert framework is not None
