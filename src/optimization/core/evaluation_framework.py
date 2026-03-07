"""
Evaluation Framework Module
评估框架模块

This module provides evaluation framework for optimization algorithms
此模块为优化算法提供评估框架

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class EvaluationMetric(ABC):

    """
    Evaluation Metric Base Class
    评估指标基类

    Abstract base class for evaluation metrics
    评估指标的抽象基类
    """

    def __init__(self, name: str, description: str):
        """
        Initialize evaluation metric
        初始化评估指标

        Args:
            name: Name of the metric
                指标名称
            description: Description of the metric
                        指标描述
        """
        self.name = name
        self.description = description

    @abstractmethod
    def calculate(self, **kwargs) -> Union[float, Dict[str, Any]]:
        """
        Calculate the metric value
        计算指标值

        Args:
            **kwargs: Arguments needed for calculation
                     计算所需的参数

        Returns:
            Metric value or detailed results
            指标值或详细结果
        """


class ConvergenceMetric(EvaluationMetric):

    """
    Convergence Metric Class
    收敛指标类

    Evaluates convergence properties of optimization algorithms
    评估优化算法的收敛特性
    """

    def __init__(self):

        super().__init__(
            "convergence",
            "Evaluates how well and how fast the algorithm converges to optimal solution"
        )

    def calculate(self,

                  convergence_history: List[Dict[str, Any]],
                  optimal_value: float,
                  tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Calculate convergence metrics
        计算收敛指标

        Args:
            convergence_history: History of convergence data
                                收敛历史数据
            optimal_value: Known optimal value (if available)
                          已知的最优值（如果可用）
            tolerance: Convergence tolerance
                      收敛容差

        Returns:
            dict: Convergence metrics
                  收敛指标
        """
        if not convergence_history:
            return {'error': 'No convergence history available'}

        results = {
            'total_iterations': len(convergence_history),
            'final_value': convergence_history[-1].get('value', 0),
            'best_value': min(h.get('value', float('inf')) for h in convergence_history),
            'convergence_speed': self._calculate_convergence_speed(convergence_history),
            'stability_score': self._calculate_stability_score(convergence_history)
        }

        # Check if converged to optimal
        if optimal_value is not None:
            results['convergence_accuracy'] = abs(results['final_value'] - optimal_value)
            results['converged_to_optimal'] = results['convergence_accuracy'] < tolerance

        # Calculate improvement rate
        if len(convergence_history) > 1:
            initial_value = convergence_history[0].get('value', 0)
            final_value = convergence_history[-1].get('value', 0)
        if initial_value != 0:
            results['improvement_rate'] = (initial_value - final_value) / abs(initial_value)

        return results

    def _calculate_convergence_speed(self, history: List[Dict[str, Any]]) -> float:
        """
        Calculate convergence speed
        计算收敛速度

        Args:
            history: Convergence history
                    收敛历史

        Returns:
            float: Convergence speed score
                   收敛速度评分
        """
        if len(history) < 2:
            return 0.0

        # Calculate rate of improvement
        improvements = []
        for i in range(1, len(history)):
            prev_value = history[i - 1].get('value', 0)
            curr_value = history[i].get('value', 0)
        if prev_value != 0:
            improvement = (prev_value - curr_value) / abs(prev_value)
            improvements.append(improvement)

        if not improvements:
            return 0.0

        # Average improvement per iteration
        return statistics.mean(improvements)

    def _calculate_stability_score(self, history: List[Dict[str, Any]]) -> float:
        """
        Calculate stability score
        计算稳定性评分

        Args:
            history: Convergence history
                    收敛历史

        Returns:
            float: Stability score (0.0 to 1.0)
                   稳定性评分（0.0到1.0）
        """
        if len(history) < 3:
            return 1.0

        values = [h.get('value', 0) for h in history]

        # Calculate variance in recent values
        recent_values = values[-min(10, len(values)):]
        if len(recent_values) > 1:
            variance = statistics.variance(recent_values)
            # Normalize variance to stability score
            max_expected_variance = (max(recent_values) - min(recent_values)) ** 2
        if max_expected_variance > 0:
            stability = 1.0 - min(variance / max_expected_variance, 1.0)
            return stability

        return 1.0


class EfficiencyMetric(EvaluationMetric):

    """
    Efficiency Metric Class
    效率指标类

    Evaluates computational efficiency of optimization algorithms
    评估优化算法的计算效率
    """

    def __init__(self):

        super().__init__(
            "efficiency",
            "Evaluates computational resources used by the algorithm"
        )

    def calculate(self,

                  execution_time: float,
                  iterations: int,
                  function_evaluations: Optional[int] = None,
                  gradient_evaluations: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate efficiency metrics
        计算效率指标

        Args:
            execution_time: Total execution time
                           总执行时间
            iterations: Number of iterations
                       迭代次数
            function_evaluations: Number of function evaluations
                                 函数评估次数
            gradient_evaluations: Number of gradient evaluations
                                 梯度评估次数

        Returns:
            dict: Efficiency metrics
                  效率指标
        """
        results = {
            'execution_time': execution_time,
            'iterations': iterations,
            'time_per_iteration': execution_time / max(iterations, 1)
        }

        if function_evaluations:
            results['function_evaluations'] = function_evaluations
            results['time_per_function_eval'] = execution_time / function_evaluations
            results['evaluations_per_iteration'] = function_evaluations / max(iterations, 1)

        if gradient_evaluations:
            results['gradient_evaluations'] = gradient_evaluations
            results['time_per_gradient_eval'] = execution_time / gradient_evaluations

        # Calculate efficiency score
        results['efficiency_score'] = self._calculate_efficiency_score(results)

        return results

    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate overall efficiency score
        计算整体效率评分

        Args:
            metrics: Efficiency metrics
                    效率指标

        Returns:
            float: Efficiency score (0.0 to 1.0)
                   效率评分（0.0到1.0）
        """
        score = 1.0

        # Penalize for long execution time
        if metrics['execution_time'] > 60:  # More than 1 minute
            time_penalty = min(metrics['execution_time'] / 300, 0.5)  # Max 50% penalty
            score *= (1.0 - time_penalty)

        # Penalize for many iterations
        if metrics['iterations'] > 1000:
            iteration_penalty = min(metrics['iterations'] / 5000, 0.3)  # Max 30% penalty
            score *= (1.0 - iteration_penalty)

        return max(0.0, score)


class SharpeRatio(EvaluationMetric):
    """夏普比率指标"""

    def __init__(self):
        super().__init__("Sharpe Ratio", "夏普比率 - 风险调整后的收益率指标")

    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate
        if len(excess_returns) < 2:
            return 0.0
        return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0

    def get_name(self) -> str:
        return "Sharpe Ratio"


class MaximumDrawdown(EvaluationMetric):
    """最大回撤指标"""

    def __init__(self):
        super().__init__("Maximum Drawdown", "最大回撤 - 投资组合的最大亏损幅度")

    def calculate(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    def get_name(self) -> str:
        return "Maximum Drawdown"


class WinRate(EvaluationMetric):
    """胜率指标"""

    def __init__(self):
        super().__init__("Win Rate", "胜率 - 盈利交易占总交易的比例")

    def calculate(self, returns: pd.Series) -> float:
        """计算胜率"""
        if len(returns) == 0:
            return 0.0
        winning_trades = (returns > 0).sum()
        return winning_trades / len(returns)

    def get_name(self) -> str:
        return "Win Rate"


class ProfitFactor(EvaluationMetric):
    """盈利因子指标"""

    def __init__(self):
        super().__init__("Profit Factor", "盈利因子 - 盈利总额与亏损总额的比率")

    def calculate(self, returns: pd.Series) -> float:
        """计算盈利因子"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return gains / losses if losses > 0 else float('inf')

    def get_name(self) -> str:
        return "Profit Factor"


class CalmarRatio(EvaluationMetric):
    """卡玛比率指标"""

    def __init__(self):
        super().__init__("Calmar Ratio", "卡玛比率 - 年化收益率与最大回撤的比率")

    def calculate(self, returns: pd.Series) -> float:
        """计算卡玛比率"""
        if len(returns) < 2:
            return 0.0

        # 计算年化收益率
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252  # 假设252个交易日
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

        # 计算最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        return annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    def get_name(self) -> str:
        return "Calmar Ratio"


class SortinoRatio(EvaluationMetric):
    """索提诺比率指标"""

    def __init__(self):
        super().__init__("Sortino Ratio", "索提诺比率 - 衡量下行风险调整后的收益率")

    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        return excess_returns.mean() / downside_std

    def get_name(self) -> str:
        return "Sortino Ratio"


class Alpha(EvaluationMetric):
    """阿尔法指标"""

    def __init__(self):
        super().__init__("Alpha", "阿尔法 - 投资组合相对于基准的超额收益")

    def calculate(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算阿尔法"""
        if len(returns) != len(benchmark_returns):
            return 0.0
        return returns.mean() - benchmark_returns.mean()

    def get_name(self) -> str:
        return "Alpha"


class Beta(EvaluationMetric):
    """贝塔指标"""

    def __init__(self):
        super().__init__("Beta", "贝塔 - 衡量投资组合相对于基准的波动性")

    def calculate(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算贝塔"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 1.0

        # 使用numpy的cov函数计算协方差矩阵
        try:
            cov_matrix = np.cov(returns, benchmark_returns)
            covariance = cov_matrix[0, 1]
            benchmark_variance = cov_matrix[1, 1]
        except:
            # 回退到pandas方法
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()

        if benchmark_variance == 0:
            return 1.0
        beta = covariance / benchmark_variance

        # 限制beta值在合理范围内
        return max(0.5, min(3.0, beta))

    def get_name(self) -> str:
        return "Beta"


class InformationRatio(EvaluationMetric):
    """信息比率指标"""

    def __init__(self):
        super().__init__("Information Ratio", "信息比率 - 阿尔法与跟踪误差的比率")

    def calculate(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算信息比率"""
        alpha = returns.mean() - benchmark_returns.mean()
        tracking_error = (returns - benchmark_returns).std()
        if tracking_error == 0:
            return 0.0
        return alpha / tracking_error

    def get_name(self) -> str:
        return "Information Ratio"


class BenchmarkComparison(EvaluationMetric):
    """基准比较指标"""

    def __init__(self):
        super().__init__("Benchmark Comparison", "基准比较 - 投资组合与基准的相对表现")

    def calculate(self, returns: pd.Series, benchmark_returns: pd.Series) -> dict:
        """计算基准比较指标"""
        alpha = returns.mean() - benchmark_returns.mean()
        beta = returns.cov(benchmark_returns) / \
            benchmark_returns.var() if benchmark_returns.var() > 0 else 1.0
        return {
            'alpha': alpha,
            'beta': beta,
            'outperformance': alpha > 0
        }

    def compare(self, returns: pd.Series, benchmark_returns: pd.Series) -> dict:
        """比较投资组合与基准表现"""
        alpha = returns.mean() - benchmark_returns.mean()
        beta = returns.cov(benchmark_returns) / \
            benchmark_returns.var() if benchmark_returns.var() > 0 else 1.0
        tracking_error = (returns - benchmark_returns).std()
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0

        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'outperformance': alpha > 0,
            'volatility_comparison': returns.std() / benchmark_returns.std() if benchmark_returns.std() > 0 else 1.0
        }

    def get_name(self) -> str:
        return "Benchmark Comparison"


class StatisticalTests(EvaluationMetric):
    """统计检验指标"""

    def __init__(self):
        super().__init__("Statistical Tests", "统计检验 - 投资表现的统计显著性")

    def calculate(self, returns: pd.Series) -> dict:
        """计算统计检验"""
        from scipy import stats
        try:
            # Shapiro-Wilk正态性检验
            shapiro_stat, shapiro_p = stats.shapiro(returns)
            # 均值t检验 (与0比较)
            t_stat, t_p = stats.ttest_1samp(returns, 0)
            return {
                'shapiro_test': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                't_test': {'statistic': t_stat, 'p_value': t_p},
                'is_normal': shapiro_p > 0.05,
                'is_significant': t_p < 0.05
            }
        except:
            return {'error': 'Statistical test failed'}

    def perform_tests(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> dict:
        """执行完整的统计检验"""
        from scipy import stats
        try:
            results = {}

            # Shapiro-Wilk正态性检验
            shapiro_stat, shapiro_p = stats.shapiro(returns)
            results['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }

            # 均值t检验 (与0比较)
            t_stat, t_p = stats.ttest_1samp(returns, 0)
            results['t_test'] = {
                'statistic': t_stat,
                'p_value': t_p,
                'is_significant': t_p < 0.05
            }

            # Jarque-Bera正态性检验
            try:
                jb_stat, jb_p = stats.jarque_bera(returns)
                results['jarque_bera'] = {
                    'statistic': jb_stat,
                    'p_value': jb_p,
                    'is_normal': jb_p > 0.05
                }
            except:
                results['jarque_bera'] = {'error': 'Jarque-Bera test failed'}

            # 自相关检验 (Durbin-Watson)
            try:
                dw_stat = sum((returns.iloc[1:] - returns.iloc[:-1])**2) / sum(returns**2)
                results['autocorrelation'] = {
                    'durbin_watson': dw_stat,
                    'has_autocorrelation': not (1.5 < dw_stat < 2.5)
                }
            except:
                results['autocorrelation'] = {'error': 'Autocorrelation test failed'}

            # 如果有基准收益，进行比较检验
            if benchmark_returns is not None:
                try:
                    # 两样本t检验
                    t_stat_2, t_p_2 = stats.ttest_ind(returns, benchmark_returns)
                    results['t_test_comparison'] = {
                        'statistic': t_stat_2,
                        'p_value': t_p_2,
                        'is_different': t_p_2 < 0.05
                    }
                except:
                    results['t_test_comparison'] = {'error': 'Comparison test failed'}

            return results
        except Exception as e:
            return {'error': f'Statistical tests failed: {str(e)}'}

    def get_name(self) -> str:
        return "Statistical Tests"


class RobustnessMetric(EvaluationMetric):

    """
    Robustness Metric Class
    鲁棒性指标类

    Evaluates robustness of optimization algorithms to different conditions
    评估优化算法对不同条件的鲁棒性
    """

    def __init__(self):

        super().__init__(
            "robustness",
            "Evaluates algorithm performance under different conditions"
        )

    def calculate(self,

                  test_results: List[Dict[str, Any]],
                  conditions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate robustness metrics
        计算鲁棒性指标

        Args:
            test_results: Results from different test conditions
                         不同测试条件的结果
            conditions: List of test conditions
                       测试条件列表

        Returns:
            dict: Robustness metrics
                  鲁棒性指标
        """
        if not test_results:
            return {'error': 'No test results available'}

        # Analyze success rates across conditions
        success_rates = []
        performance_variability = []

        for result in test_results:
            success = result.get('success', False)
            success_rates.append(1 if success else 0)

        if 'final_value' in result:
            performance_variability.append(result['final_value'])

        results = {
            'total_tests': len(test_results),
            'success_rate': statistics.mean(success_rates),
            'success_count': sum(success_rates),
            'failure_count': len(success_rates) - sum(success_rates)
        }

        # Calculate performance variability
        if len(performance_variability) > 1:
            results['performance_std'] = statistics.stdev(performance_variability)
            results['performance_cv'] = results['performance_std'] / \
                abs(statistics.mean(performance_variability))

        # Calculate robustness score
        results['robustness_score'] = self._calculate_robustness_score(results)

        return results

    def _calculate_robustness_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate robustness score
        计算鲁棒性评分

        Args:
            metrics: Robustness metrics
                    鲁棒性指标

        Returns:
            float: Robustness score (0.0 to 1.0)
                   鲁棒性评分（0.0到1.0）
        """
        success_rate = metrics.get('success_rate', 0)
        performance_cv = metrics.get('performance_cv', 0)

        # Base score from success rate
        score = success_rate

        # Penalize for high performance variability
        if performance_cv > 0.5:  # High variability
            variability_penalty = min(performance_cv - 0.5, 0.5)
            score *= (1.0 - variability_penalty)

        return score


class EvaluationFramework:

    """
    Evaluation Framework Class
    评估框架类

    Provides comprehensive evaluation capabilities for optimization algorithms
    为优化算法提供全面的评估能力
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluation framework
        初始化评估框架

        Args:
            config: Configuration dictionary for evaluation framework
        """
        self.config = config or {}
        self.metrics = {
            'convergence': ConvergenceMetric(),
            'efficiency': EfficiencyMetric(),
            'robustness': RobustnessMetric()
        }

        self.evaluation_results = defaultdict(list)
        self.baselines = {}
        self.benchmark_returns = None

        logger.info("Evaluation framework initialized")

    def register_metric(self, metric: EvaluationMetric) -> None:
        """
        Register a new evaluation metric
        注册新的评估指标

        Args:
            metric: Metric to register
                   要注册的指标
        """
        self.metrics[metric.name] = metric
        logger.info(f"Registered evaluation metric: {metric.name}")

    def evaluate_algorithm(self,

                           algorithm_name: str,
                           algorithm_result: Dict[str, Any],
                           **kwargs) -> Dict[str, Any]:
        """
        Evaluate an optimization algorithm
        评估优化算法

        Args:
            algorithm_name: Name of the algorithm
                          算法名称
            algorithm_result: Result from algorithm execution
                             算法执行结果
            **kwargs: Additional evaluation parameters
                     其他评估参数

        Returns:
            dict: Comprehensive evaluation results
                  全面的评估结果
        """
        evaluation = {
            'algorithm_name': algorithm_name,
            'evaluation_timestamp': datetime.now(),
            'metrics': {},
            'overall_score': 0.0,
            'recommendations': []
        }

        total_score = 0.0
        metric_count = 0

        # Evaluate using each metric
        for metric_name, metric in self.metrics.items():
            try:
                metric_result = self._evaluate_single_metric(
                    metric, algorithm_result, **kwargs
                )
                evaluation['metrics'][metric_name] = metric_result

                if isinstance(metric_result, dict) and 'error' not in metric_result:
                    # Extract score if available
                    score = metric_result.get(f'{metric_name}_score', 0.0)
                    if score == 0.0 and 'score' in metric_result:
                        score = metric_result['score']
                    if score == 0.0 and 'efficiency_score' in metric_result:
                        score = metric_result['efficiency_score']

                    total_score += score
                    metric_count += 1

            except Exception as e:
                logger.error(f"Failed to evaluate metric {metric_name}: {str(e)}")
                evaluation['metrics'][metric_name] = {'error': str(e)}

        # Calculate overall score
        if metric_count > 0:
            evaluation['overall_score'] = total_score / metric_count

        # Generate recommendations
        evaluation['recommendations'] = self._generate_recommendations(
            evaluation['metrics'], algorithm_result
        )

        # Store evaluation result
        self.evaluation_results[algorithm_name].append(evaluation)

        return evaluation

    def _evaluate_single_metric(self,

                                metric: EvaluationMetric,
                                algorithm_result: Dict[str, Any],
                                **kwargs) -> Any:
        """
        Evaluate using a single metric
        使用单个指标进行评估

        Args:
            metric: Evaluation metric
                   评估指标
            algorithm_result: Algorithm result
                             算法结果
            **kwargs: Additional parameters
                     其他参数

        Returns:
            Metric evaluation result
            指标评估结果
        """
        # Prepare arguments for the metric
        metric_args = {}

        if metric.name == 'convergence':
            metric_args.update({
                'convergence_history': algorithm_result.get('convergence_history', []),
                'optimal_value': algorithm_result.get('optimal_value'),
                'tolerance': kwargs.get('tolerance', 1e-6)
            })

        elif metric.name == 'efficiency':
            metric_args.update({
                'execution_time': algorithm_result.get('execution_time', 0),
                'iterations': algorithm_result.get('iterations', 0),
                'function_evaluations': algorithm_result.get('function_evaluations'),
                'gradient_evaluations': algorithm_result.get('gradient_evaluations')
            })

        elif metric.name == 'robustness':
            # For robustness, we need multiple test results
            metric_args.update({
                'test_results': kwargs.get('test_results', [algorithm_result]),
                'conditions': kwargs.get('conditions')
            })

        return metric.calculate(**metric_args)

    def _generate_recommendations(self,

                                  metrics: Dict[str, Any],
                                  algorithm_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on evaluation results
        基于评估结果生成建议

        Args:
            metrics: Evaluation metrics results
                    评估指标结果
            algorithm_result: Original algorithm result
                             原始算法结果

        Returns:
            list: List of recommendations
                  建议列表
        """
        recommendations = []

        # Check convergence
        convergence = metrics.get('convergence', {})
        if isinstance(convergence, dict):
            if not convergence.get('converged_to_optimal', True):
                recommendations.append(
                    "Consider adjusting convergence criteria or algorithm parameters")

        if convergence.get('total_iterations', 0) > 1000:
            recommendations.append(
                "Algorithm may be slow to converge; consider using more efficient method")

        # Check efficiency
        efficiency = metrics.get('efficiency', {})
        if isinstance(efficiency, dict):
            if efficiency.get('execution_time', 0) > 60:
                recommendations.append(
                    "Consider optimizing algorithm implementation for better performance")

            efficiency_score = efficiency.get('efficiency_score', 1.0)
            if efficiency_score < 0.5:
                recommendations.append(
                    "Algorithm efficiency is low; consider alternative optimization methods")

        # Check robustness
        robustness = metrics.get('robustness', {})
        if isinstance(robustness, dict):
            success_rate = robustness.get('success_rate', 1.0)
        if success_rate < 0.8:
            recommendations.append(
                "Algorithm may not be robust; consider adding regularization or constraints")

        return recommendations

    def compare_algorithms(self,

                           algorithm_names: List[str],
                           metric_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple algorithms
        比较多个算法

        Args:
            algorithm_names: Names of algorithms to compare
                            要比较的算法名称
            metric_name: Specific metric to compare (None for all)
                        要比较的特定指标（None表示全部）

        Returns:
            dict: Comparison results
                  比较结果
        """
        comparison = {
            'algorithms': algorithm_names,
            'comparison_timestamp': datetime.now(),
            'metric_focus': metric_name,
            'results': {}
        }

        for algorithm in algorithm_names:
            evaluations = self.evaluation_results.get(algorithm, [])
            if evaluations:
                latest_evaluation = evaluations[-1]

        if metric_name:
            comparison['results'][algorithm] = {
                'metric_value': latest_evaluation['metrics'].get(metric_name),
                'overall_score': latest_evaluation['overall_score']
            }
        else:
            comparison['results'][algorithm] = {
                'metrics': latest_evaluation['metrics'],
                'overall_score': latest_evaluation['overall_score']
            }

        # Generate ranking
        if comparison['results']:
            comparison['ranking'] = self._rank_algorithms(comparison['results'], metric_name)

        return comparison

    def _rank_algorithms(self,

                         results: Dict[str, Any],
                         metric_name: Optional[str] = None) -> List[str]:
        """
        Rank algorithms based on performance
        基于性能对算法进行排名

        Args:
            results: Comparison results
                    比较结果
            metric_name: Metric to rank by
                        排名依据的指标

        Returns:
            list: Ranked algorithm names
                  排名后的算法名称
        """

        def get_score(algorithm_result):

            if metric_name:
                metric_data = algorithm_result.get('metric_value', {})
                if isinstance(metric_data, dict):
                    # Try different score keys
                    for key in [f'{metric_name}_score', 'score', 'efficiency_score', 'robustness_score']:
                        if key in metric_data and isinstance(metric_data[key], (int, float)):
                            return metric_data[key]
                return 0.0
            else:
                return algorithm_result.get('overall_score', 0.0)

        return sorted(
            results.keys(),
            key=lambda x: get_score(results[x]),
            reverse=True
        )

    def set_baseline(self, algorithm_name: str, baseline_result: Dict[str, Any]) -> None:
        """
        Set baseline performance for comparison
        设置基准性能以进行比较

        Args:
            algorithm_name: Name of the algorithm
                          算法名称
            baseline_result: Baseline result
                           基准结果
        """
        self.baselines[algorithm_name] = baseline_result
        logger.info(f"Set baseline for algorithm: {algorithm_name}")

    def get_evaluation_history(self,

                               algorithm_name: Optional[str] = None,
                               limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Get evaluation history
        获取评估历史

        Args:
            algorithm_name: Specific algorithm (None for all)
                           特定算法（None表示全部）
            limit: Maximum number of records to return
                  返回的最大记录数

        Returns:
            dict: Evaluation history
                  评估历史
        """
        if algorithm_name:
            history = self.evaluation_results.get(algorithm_name, [])
            if limit:
                history = history[-limit:]
            return {
                'algorithm': algorithm_name,
                'evaluations': history,
                'total_evaluations': len(history)
            }
        else:
            result = {}
            for alg, evaluations in self.evaluation_results.items():
                if limit:
                    result[alg] = evaluations[-limit:]
                else:
                    result[alg] = evaluations
            return {
                'all_algorithms': result,
                'total_algorithms': len(result)
            }

    def generate_evaluation_report(self,

                                   algorithm_name: Optional[str] = None,
                                   include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        生成全面的评估报告

        Args:
            algorithm_name: Specific algorithm (None for all)
                           特定算法（None表示全部）
            include_recommendations: Whether to include recommendations
                                   是否包含建议

        Returns:
            dict: Evaluation report
                  评估报告
        """
        report = {
            'report_timestamp': datetime.now(),
            'framework_version': '1.0.0',
            'available_metrics': list(self.metrics.keys())
        }

        if algorithm_name:
            evaluations = self.evaluation_results.get(algorithm_name, [])
        if evaluations:
            latest_evaluation = evaluations[-1]
            report['algorithm_name'] = algorithm_name
            report['latest_evaluation'] = latest_evaluation
            report['evaluation_history'] = len(evaluations)

        if include_recommendations:
            report['recommendations'] = latest_evaluation.get('recommendations', [])
        else:
            report['error'] = f'No evaluations found for algorithm: {algorithm_name}'
            # Summary for all algorithms
            summary = {}
            for alg, evaluations in self.evaluation_results.items():
                if evaluations:
                    latest = evaluations[-1]
                    summary[alg] = {
                        'latest_score': latest['overall_score'],
                        'total_evaluations': len(evaluations),
                        'last_evaluation': latest['evaluation_timestamp']
                    }

            report['algorithm_summary'] = summary
            report['total_algorithms_evaluated'] = len(summary)

        if include_recommendations:
            report['overall_recommendations'] = self._generate_overall_recommendations(summary)

        return report

    def _generate_overall_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """
        Generate overall recommendations across all algorithms
        生成所有算法的总体建议

        Args:
            summary: Algorithm summary
                    算法摘要
        Returns:
            list: Overall recommendations
                  总体建议
        """
        recommendations = []

        if not summary:
            return recommendations

        # Find best and worst performing algorithms
        scores = {alg: data['latest_score'] for alg, data in summary.items()}
        best_algorithm = max(scores, key=scores.get)
        worst_algorithm = min(scores, key=scores.get)

        recommendations.append(
            f"Best performing algorithm: {best_algorithm} (score: {scores[best_algorithm]:.3f})")
        recommendations.append(
            f"Consider reviewing: {worst_algorithm} (score: {scores[worst_algorithm]:.3f})")

        # Check if any algorithms need improvement
        low_performers = [alg for alg, score in scores.items() if score < 0.5]
        if low_performers:
            recommendations.append(f"Algorithms needing improvement: {', '.join(low_performers)}")

        return recommendations

    def set_benchmark(self, benchmark_returns: pd.Series) -> None:
        """设置基准收益"""
        self.benchmark_returns = benchmark_returns

    def add_metric(self, metric: EvaluationMetric) -> None:
        """添加自定义指标"""
        self.metrics[metric.name.lower().replace(' ', '_')] = metric

    def evaluate_strategy(self, returns: pd.Series) -> Dict[str, Any]:
        """评估策略表现"""
        results = {'performance_metrics': {}, 'risk_metrics': {}}
        for name, metric in self.metrics.items():
            if hasattr(metric, 'calculate'):
                try:
                    if 'benchmark' in name.lower() and self.benchmark_returns is not None:
                        result = metric.calculate(returns, self.benchmark_returns)
                    else:
                        result = metric.calculate(returns)

                    # 分类存储指标
                    if any(keyword in name.lower() for keyword in ['sharpe', 'sortino', 'alpha', 'beta', 'information']):
                        results['performance_metrics'][name] = result
                    else:
                        results['risk_metrics'][name] = result

                except Exception as e:
                    results['performance_metrics'][name] = f"Error: {str(e)}"
        return results

    def analyze_time_series(self, returns: pd.Series) -> Dict[str, Any]:
        """时间序列分析"""
        # 计算自相关系数
        try:
            autocorr = returns.autocorr()
        except:
            autocorr = 0.0

        # 计算波动率聚集
        try:
            volatility_clustering = returns.rolling(5).std().autocorr()
        except:
            volatility_clustering = 0.0

        return {
            'mean': returns.mean(),
            'std': returns.std(),
            'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'trend': 'up' if returns.iloc[-1] > returns.iloc[0] else 'down',
            'autocorrelation': autocorr,
            'volatility_clustering': volatility_clustering
        }

    def calculate_rolling_metrics(self, returns: pd.Series, window: int = 20) -> pd.DataFrame:
        """计算滚动指标"""
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = rolling_mean / rolling_std
        rolling_volatility = rolling_std * np.sqrt(252)  # 年化波动率

        # 计算滚动最大回撤
        rolling_cumulative = (
            1 + returns).rolling(window=window).apply(lambda x: (1 + x).prod() - 1)
        rolling_running_max = rolling_cumulative.expanding().max()
        rolling_max_drawdown = (rolling_cumulative - rolling_running_max) / rolling_running_max
        rolling_max_drawdown = rolling_max_drawdown.abs()

        return pd.DataFrame({
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_sharpe': rolling_sharpe,
            'rolling_volatility': rolling_volatility,
            'rolling_max_drawdown': rolling_max_drawdown
        })

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        # 确保输入是pandas Series
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    def perform_bootstrap_analysis(self, returns: pd.Series, n_bootstraps: int = 1000) -> Dict[str, Any]:
        """执行自举分析"""
        bootstrapped_means = []
        for _ in range(n_bootstraps):
            sample = returns.sample(n=len(returns), replace=True)
            bootstrapped_means.append(sample.mean())

        return {
            'bootstrap_mean': np.mean(bootstrapped_means),
            'bootstrap_std': np.std(bootstrapped_means),
            'confidence_interval': (np.percentile(bootstrapped_means, 2.5), np.percentile(bootstrapped_means, 97.5))
        }

    def analyze_scenarios(self, returns: pd.Series, scenarios: Dict[str, pd.Series]) -> Dict[str, Any]:
        """情景分析"""
        results = {}
        for scenario_name, scenario_returns in scenarios.items():
            results[scenario_name] = {
                'mean': scenario_returns.mean(),
                'std': scenario_returns.std(),
                'sharpe': scenario_returns.mean() / scenario_returns.std() if scenario_returns.std() > 0 else 0
            }
        return results

    def perform_stress_tests(self, returns: pd.Series, stress_scenarios: Dict[str, float]) -> Dict[str, Any]:
        """压力测试"""
        results = {}
        for scenario_name, shock in stress_scenarios.items():
            stressed_returns = returns * (1 + shock)
            results[scenario_name] = {
                'stressed_mean': stressed_returns.mean(),
                'stressed_volatility': stressed_returns.std(),
                'max_drawdown': self._calculate_max_drawdown(stressed_returns)
            }
        return results

    def perform_sensitivity_analysis(self, returns: pd.Series, parameters: Dict[str, List[float]]) -> Dict[str, Any]:
        """敏感性分析"""
        results = {}
        for param_name, param_values in parameters.items():
            param_results = []
            for value in param_values:
                # 简化的敏感性计算
                adjusted_returns = returns * (1 + value * 0.1)
                param_results.append({
                    'parameter_value': value,
                    'mean': adjusted_returns.mean(),
                    'volatility': adjusted_returns.std()
                })
            results[param_name] = param_results
        return results

    def perform_cross_validation(self, returns: pd.Series, n_splits: int = 5) -> Dict[str, Any]:
        """交叉验证"""
        fold_size = len(returns) // n_splits
        cv_scores = []

        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(returns)

            test_fold = returns.iloc[start_idx:end_idx]
            train_fold = pd.concat([returns.iloc[:start_idx], returns.iloc[end_idx:]])

            test_score = test_fold.mean()
            cv_scores.append(test_score)

        return {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }

    def validate_model(self, returns: pd.Series, model_predictions: pd.Series) -> Dict[str, Any]:
        """模型验证"""
        return {
            'correlation': returns.corr(model_predictions),
            'mean_error': (returns - model_predictions).mean(),
            'rmse': np.sqrt(((returns - model_predictions) ** 2).mean())
        }

    def perform_performance_attribution(self, returns: pd.Series, factors: Dict[str, pd.Series]) -> Dict[str, Any]:
        """业绩归因分析"""
        attribution = {}
        for factor_name, factor_returns in factors.items():
            attribution[factor_name] = returns.corr(factor_returns)

        return {
            'factor_contributions': attribution,
            'total_attribution': sum(attribution.values())
        }

    def perform_factor_analysis(self, returns: pd.Series, factors: Dict[str, pd.Series]) -> Dict[str, Any]:
        """因子分析"""
        factor_loadings = {}
        factor_returns = {}

        for factor_name, factor_data in factors.items():
            # 计算因子载荷（相关系数）
            factor_loadings[factor_name] = returns.corr(factor_data)
            # 计算因子收益
            factor_returns[factor_name] = factor_data.mean()

        # 计算R平方
        r_squared = sum(abs(corr) for corr in factor_loadings.values()) / len(factor_loadings)

        return {
            'factor_loadings': factor_loadings,
            'factor_returns': factor_returns,
            'r_squared': r_squared,
            'analysis_timestamp': pd.Timestamp.now()
        }


# Global evaluation framework instance
# 全局评估框架实例
evaluation_framework = EvaluationFramework()

__all__ = [
    'EvaluationMetric',
    'ConvergenceMetric',
    'EfficiencyMetric',
    'RobustnessMetric',
    'EvaluationFramework',
    'evaluation_framework'
]
