# !/usr/bin/env python3
"""
策略调优器
自动优化交易策略参数，提高策略性能
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import itertools
import secrets

# 使用标准logging避免导入错误
import logging

from infrastructure.logging.utils.logger import get_logger

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):

    """优化方法"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"


class OptimizationTarget(Enum):

    """优化目标"""
    MAX_SHARPE_RATIO = "max_sharpe_ratio"
    MAX_TOTAL_RETURN = "max_total_return"
    MIN_MAX_DRAWDOWN = "min_max_drawdown"
    MAX_WIN_RATE = "max_win_rate"
    MIN_RISK_ADJUSTED_RETURN = "min_risk_adjusted_return"


@dataclass
class OptimizationResult:

    """优化结果"""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_iterations: int
    execution_time: timedelta
    convergence_score: float


@dataclass
class ParameterRange:

    """参数范围"""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    value_type: str = "float"  # float, int, choice
    choices: Optional[List[Any]] = None


class StrategyOptimizer:

    """策略调优器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 配置参数
        self.max_iterations = self.config.get('max_iterations', 100)
        self.early_stopping_rounds = self.config.get('early_stopping_rounds', 10)
        self.n_jobs = self.config.get('n_jobs', 4)
        self.random_seed = self.config.get('random_seed', 42)

        # 设置随机种子
        np.random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.logger = get_logger(__name__)

    def optimize_strategy(self, strategy_class: Any, data: pd.DataFrame,

                          parameter_ranges: List[ParameterRange],
                          method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
                          target: OptimizationTarget = OptimizationTarget.MAX_SHARPE_RATIO,
                          **kwargs) -> OptimizationResult:
        """
        优化策略参数

        Args:
            strategy_class: 策略类
            data: 历史数据
            parameter_ranges: 参数范围列表
            method: 优化方法
            target: 优化目标
            **kwargs: 其他参数

        Returns:
            优化结果
        """
        start_time = datetime.now()

        self.logger.info(f"开始策略优化: {strategy_class.__name__} 使用 {method.value}")

        try:
            if method == OptimizationMethod.GRID_SEARCH:
                result = self._grid_search(strategy_class, data, parameter_ranges, target, **kwargs)
            elif method == OptimizationMethod.RANDOM_SEARCH:
                result = self._random_search(
                    strategy_class, data, parameter_ranges, target, **kwargs)
            elif method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
                result = self._bayesian_optimization(
                    strategy_class, data, parameter_ranges, target, **kwargs)
            elif method == OptimizationMethod.GENETIC_ALGORITHM:
                result = self._genetic_algorithm(
                    strategy_class, data, parameter_ranges, target, **kwargs)
            else:
                raise ValueError(f"不支持的优化方法: {method}")

            execution_time = datetime.now() - start_time
            result.execution_time = execution_time

            self.logger.info(f"策略优化完成，耗时: {execution_time.total_seconds():.2f}秒")
            return result

        except Exception as e:
            self.logger.error(f"策略优化失败: {e}")
            raise

    def _grid_search(self, strategy_class: Any, data: pd.DataFrame,

                     parameter_ranges: List[ParameterRange],
                     target: OptimizationTarget, **kwargs) -> OptimizationResult:
        """网格搜索优化"""
        # 生成参数网格
        parameter_grid = self._generate_parameter_grid(parameter_ranges)

        best_score = float('-inf')
        best_parameters = None
        optimization_history = []

        self.logger.info(f"网格搜索参数组合数量: {len(parameter_grid)}")

        # 遍历所有参数组合
        for i, parameters in enumerate(parameter_grid):
            try:
                score = self._evaluate_parameters(strategy_class, data, parameters, target)

                optimization_history.append({
                    'iteration': i + 1,
                    'parameters': parameters,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_parameters = parameters

                # 进度日志
                if (i + 1) % 10 == 0:
                    self.logger.info(f"已完成 {i + 1}/{len(parameter_grid)} 个参数组合")

            except Exception as e:
                self.logger.warning(f"参数组合 {parameters} 评估失败: {e}")
                continue

        convergence_score = self._calculate_convergence_score(optimization_history)

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=optimization_history,
            total_iterations=len(parameter_grid),
            execution_time=timedelta(0),  # 会在上级函数中设置
            convergence_score=convergence_score
        )

    def _random_search(self, strategy_class: Any, data: pd.DataFrame,

                       parameter_ranges: List[ParameterRange],
                       target: OptimizationTarget, **kwargs) -> OptimizationResult:
        """随机搜索优化"""
        best_score = float('-inf')
        best_parameters = None
        optimization_history = []

        n_samples = kwargs.get('n_samples', min(self.max_iterations, 50))

        self.logger.info(f"随机搜索样本数量: {n_samples}")

        for i in range(n_samples):
            try:
                # 随机生成参数
                parameters = self._generate_random_parameters(parameter_ranges)
                score = self._evaluate_parameters(strategy_class, data, parameters, target)

                optimization_history.append({
                    'iteration': i + 1,
                    'parameters': parameters,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_parameters = parameters

                # 早停检查
                if self._check_early_stopping(optimization_history):
                    self.logger.info("触发早停条件，提前结束优化")
                    break

            except Exception as e:
                self.logger.warning(f"随机参数评估失败: {e}")
                continue

        convergence_score = self._calculate_convergence_score(optimization_history)

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=optimization_history,
            total_iterations=len(optimization_history),
            execution_time=timedelta(0),
            convergence_score=convergence_score
        )

    def _bayesian_optimization(self, strategy_class: Any, data: pd.DataFrame,

                               parameter_ranges: List[ParameterRange],
                               target: OptimizationTarget, **kwargs) -> OptimizationResult:
        """贝叶斯优化（简化实现）"""
        # 这里实现一个简化的贝叶斯优化
        # 实际实现应该使用更复杂的贝叶斯优化库如 scikit - optimize

        best_score = float('-inf')
        best_parameters = None
        optimization_history = []

        n_initial = kwargs.get('n_initial', 10)
        n_iterations = kwargs.get('n_iterations', 20)

        # 初始随机采样
        for i in range(n_initial):
            parameters = self._generate_random_parameters(parameter_ranges)
            score = self._evaluate_parameters(strategy_class, data, parameters, target)

            optimization_history.append({
                'iteration': i + 1,
                'parameters': parameters,
                'score': score
            })

        if score > best_score:
            best_score = score
            best_parameters = parameters

        # 基于历史结果进行指导性搜索
        for i in range(n_iterations):
            try:
                # 使用简单的高斯过程近似
                parameters = self._generate_guided_parameters(
                    parameter_ranges, optimization_history)
                score = self._evaluate_parameters(strategy_class, data, parameters, target)

                optimization_history.append({
                    'iteration': n_initial + i + 1,
                    'parameters': parameters,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_parameters = parameters

                if self._check_early_stopping(optimization_history):
                    break

            except Exception as e:
                self.logger.warning(f"贝叶斯优化迭代失败: {e}")
                continue

        convergence_score = self._calculate_convergence_score(optimization_history)

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=optimization_history,
            total_iterations=len(optimization_history),
            execution_time=timedelta(0),
            convergence_score=convergence_score
        )

    def _genetic_algorithm(self, strategy_class: Any, data: pd.DataFrame,


                           parameter_ranges: List[ParameterRange],
                           target: OptimizationTarget, **kwargs) -> OptimizationResult:
        """遗传算法优化"""
        population_size = kwargs.get('population_size', 20)
        generations = kwargs.get('generations', 10)
        mutation_rate = kwargs.get('mutation_rate', 0.1)

        # 初始化种群
        population = [self._generate_random_parameters(parameter_ranges)
                      for _ in range(population_size)]

        best_score = float('-inf')
        best_parameters = None
        optimization_history = []
        iteration = 0

        for generation in range(generations):
            # 评估种群
            scores = []
            for parameters in population:
                try:
                    score = self._evaluate_parameters(strategy_class, data, parameters, target)
                    scores.append((parameters, score))

                    iteration += 1
                    optimization_history.append({
                        'iteration': iteration,
                        'parameters': parameters,
                        'score': score
                    })

                    if score > best_score:
                        best_score = score
                        best_parameters = parameters

                except Exception:
                    scores.append((parameters, float('-inf')))
                    continue

            # 选择
            scores.sort(key=lambda x: x[1], reverse=True)
            selected = scores[:population_size // 2]

            # 交叉和变异
            new_population = []
            while len(new_population) < population_size:
                if secrets.random() < mutation_rate:
                    # 变异
                    parent = secrets.choice(selected)[0]
                    child = self._mutate_parameters(parent, parameter_ranges)
                    new_population.append(child)
                else:
                    # 交叉
                    parent1 = secrets.choice(selected)[0]
                    parent2 = secrets.choice(selected)[0]
                    child = self._crossover_parameters(parent1, parent2)
                    new_population.append(child)

            population = new_population

            self.logger.info(f"遗传算法第 {generation + 1} 代完成，最佳分数: {best_score:.4f}")

            if self._check_early_stopping(optimization_history):
                break

        convergence_score = self._calculate_convergence_score(optimization_history)

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=optimization_history,
            total_iterations=iteration,
            execution_time=timedelta(0),
            convergence_score=convergence_score
        )

    def _generate_parameter_grid(self, parameter_ranges: List[ParameterRange]) -> List[Dict[str, Any]]:
        """生成参数网格"""
        parameter_values = {}

        for param_range in parameter_ranges:
            if param_range.value_type == "choice":
                parameter_values[param_range.name] = param_range.choices or []
            elif param_range.value_type == "int":
                parameter_values[param_range.name] = list(range(
                    int(param_range.min_value),
                    int(param_range.max_value) + 1,
                    int(param_range.step or 1)
                ))
            else:  # float
                step = param_range.step or (param_range.max_value - param_range.min_value) / 10
                parameter_values[param_range.name] = list(np.arange(
                    param_range.min_value,
                    param_range.max_value + step,
                    step
                ))

        # 生成所有组合
        keys = list(parameter_values.keys())
        values = list(parameter_values.values())

        parameter_combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            parameter_combinations.append(param_dict)

        return parameter_combinations

    def _generate_random_parameters(self, parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """生成随机参数"""
        parameters = {}

        for param_range in parameter_ranges:
            if param_range.value_type == "choice":
                parameters[param_range.name] = secrets.choice(param_range.choices or [])
            elif param_range.value_type == "int":
                parameters[param_range.name] = secrets.randint(
                    int(param_range.min_value),
                    int(param_range.max_value)
                )
            else:  # float
                parameters[param_range.name] = secrets.uniform(
                    param_range.min_value,
                    param_range.max_value
                )

        return parameters

    def _generate_guided_parameters(self, parameter_ranges: List[ParameterRange],


                                    history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成引导参数（基于历史表现）"""
        if len(history) < 5:
            return self._generate_random_parameters(parameter_ranges)

        # 找到表现最好的参数组合
        best_results = sorted(history, key=lambda x: x['score'], reverse=True)[:5]

        # 在最佳参数附近进行搜索
        parameters = {}
        for param_range in parameter_ranges:
            param_name = param_range.name

            # 计算最佳参数的平均值
            best_values = [result['parameters'][param_name] for result in best_results]
            mean_value = np.mean(best_values)

            if param_range.value_type == "choice":
                parameters[param_name] = secrets.choice(param_range.choices or [])
            elif param_range.value_type == "int":
                # 在平均值附近的小范围内搜索
                search_range = max(1, int((param_range.max_value - param_range.min_value) * 0.1))
                min_val = max(param_range.min_value, mean_value - search_range)
                max_val = min(param_range.max_value, mean_value + search_range)
                parameters[param_name] = secrets.randint(int(min_val), int(max_val))
            else:  # float
                # 在平均值附近的小范围内搜索
                search_range = (param_range.max_value - param_range.min_value) * 0.1
                min_val = max(param_range.min_value, mean_value - search_range)
                max_val = min(param_range.max_value, mean_value + search_range)
                parameters[param_name] = secrets.uniform(min_val, max_val)

        return parameters

    def _mutate_parameters(self, parameters: Dict[str, Any],


                           parameter_ranges: List[ParameterRange]) -> Dict[str, Any]:
        """参数变异"""
        mutated = parameters.copy()

        # 随机选择一个参数进行变异
        param_range = secrets.choice(parameter_ranges)
        param_name = param_range.name

        if param_range.value_type == "choice":
            current_value = parameters[param_name]
            choices = [c for c in (param_range.choices or []) if c != current_value]
        if choices:
            mutated[param_name] = secrets.choice(choices)
        elif param_range.value_type == "int":
            # 小幅变异
            delta = secrets.randint(-2, 2)
            new_value = parameters[param_name] + delta
            mutated[param_name] = int(
                np.clip(new_value, param_range.min_value, param_range.max_value))
        else:  # float
            # 小幅变异
            delta = secrets.uniform(-0.1, 0.1) * (param_range.max_value - param_range.min_value)
            new_value = parameters[param_name] + delta
            mutated[param_name] = np.clip(new_value, param_range.min_value, param_range.max_value)

        return mutated

    def _crossover_parameters(self, parent1: Dict[str, Any],


                              parent2: Dict[str, Any]) -> Dict[str, Any]:
        """参数交叉"""
        child = {}

        for key in parent1.keys():
            # 随机选择父代的值
            child[key] = parent1[key] if secrets.random() < 0.5 else parent2[key]

        return child

    def _evaluate_parameters(self, strategy_class: Any, data: pd.DataFrame,


                             parameters: Dict[str, Any], target: OptimizationTarget) -> float:
        """评估参数表现"""
        try:
            # 创建策略实例
            strategy = strategy_class(config=parameters)

            # 运行回测
            from src.trading.backtesting.backtester import StrategyBacktester
            backtester = StrategyBacktester(initial_balance=100000)
            results = backtester.run_backtest(strategy, data, transaction_cost=0.001)

            # 计算目标指标
            metrics = results.calculate_metrics()

            if target == OptimizationTarget.MAX_SHARPE_RATIO:
                score = metrics.get('sharpe_ratio', -999)
            elif target == OptimizationTarget.MAX_TOTAL_RETURN:
                portfolio_value = 100000 + metrics.get('total_pnl', 0)
                score = (portfolio_value / 100000 - 1)
            elif target == OptimizationTarget.MIN_MAX_DRAWDOWN:
                score = -metrics.get('max_drawdown', 999)  # 取负数使最小化变最大化
            elif target == OptimizationTarget.MAX_WIN_RATE:
                score = metrics.get('win_rate', 0)
            elif target == OptimizationTarget.MIN_RISK_ADJUSTED_RETURN:
                sharpe = metrics.get('sharpe_ratio', 0)
                score = -abs(sharpe - 1.0)  # 目标夏普比率1.0
            else:
                score = metrics.get('sharpe_ratio', -999)

            return score

        except Exception as e:
            self.logger.warning(f"参数评估失败: {e}")
            return float('-inf')

    def _check_early_stopping(self, history: List[Dict[str, Any]]) -> bool:
        """检查早停条件"""
        if len(history) < self.early_stopping_rounds:
            return False

        # 检查最近几轮是否有显著改善
        recent_scores = [h['score'] for h in history[-self.early_stopping_rounds:]]
        best_recent = max(recent_scores)

        # 如果最近最好的分数比全局最好的分数差太多，则早停
        all_scores = [h['score'] for h in history]
        global_best = max(all_scores)

        improvement_threshold = 0.01  # 1 % 的改善阈值
        if global_best - best_recent > improvement_threshold:
            return True

        return False

    def _calculate_convergence_score(self, history: List[Dict[str, Any]]) -> float:
        """计算收敛分数"""
        if len(history) < 10:
            return 0.0

        # 计算分数标准差的减少程度
        recent_scores = [h['score'] for h in history[-10:]]
        older_scores = [h['score']
                        for h in history[-20:-10]] if len(history) >= 20 else recent_scores

        recent_std = np.std(recent_scores) if recent_scores else 1.0
        older_std = np.std(older_scores) if older_scores else 1.0

        if older_std == 0:
            return 1.0

        convergence = 1.0 - (recent_std / older_std)
        return max(0.0, min(1.0, convergence))

    def create_parameter_ranges(self, strategy_name: str) -> List[ParameterRange]:
        """为常见策略创建参数范围"""
        if "trend" in strategy_name.lower() or "ma" in strategy_name.lower():
            return [
                ParameterRange("short_period", 3, 20, step=1, value_type="int"),
                ParameterRange("long_period", 10, 50, step=2, value_type="int"),
            ]
        elif "rsi" in strategy_name.lower():
            return [
                ParameterRange("period", 5, 30, step=1, value_type="int"),
                ParameterRange("overbought", 65, 80, step=5, value_type="int"),
                ParameterRange("oversold", 20, 35, step=5, value_type="int"),
            ]
        elif "bollinger" in strategy_name.lower():
            return [
                ParameterRange("period", 10, 30, step=2, value_type="int"),
                ParameterRange("std_dev", 1.5, 3.0, step=0.1, value_type="float"),
            ]
        else:
            # 默认参数范围
            return [
                ParameterRange("param1", 0.1, 10.0, step=0.1, value_type="float"),
                ParameterRange("param2", 5, 50, step=5, value_type="int"),
            ]
