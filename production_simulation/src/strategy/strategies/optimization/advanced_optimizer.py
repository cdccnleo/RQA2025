#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级策略参数调优算法
实现多种优化方法、自适应参数调整和性能监控功能
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import warnings
from scipy.optimize import minimize, differential_evolution

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly not available, plotting features will be disabled")

# 条件导入optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("optuna not available, Bayesian optimization will be disabled")

try:
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available, some optimization features will be disabled")


logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:

    """优化配置"""
    optimization_id: str
    objective_function: Callable
    parameter_bounds: Dict[str, Tuple[float, float]]
    optimization_method: str = "bayesian"  # bayesian, genetic, particle_swarm, grid_search
    max_iterations: int = 100
    population_size: int = 50
    convergence_threshold: float = 1e-6
    timeout_seconds: int = 3600
    n_trials: int = 100
    cv_folds: int = 5
    random_state: int = 42


@dataclass
class OptimizationResult:

    """优化结果"""
    optimization_id: str
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_curve: List[float]
    optimization_time: float
    method_used: str
    status: str  # success, timeout, error
    metadata: Dict[str, Any]


@dataclass
class ParameterSpace:

    """参数空间定义"""
    name: str
    parameter_type: str  # continuous, discrete, categorical
    bounds: Tuple[float, float]

    default_value: Any
    constraints: Optional[Dict[str, Any]] = None


class AdvancedStrategyOptimizer:

    """高级策略参数优化器"""

    def __init__(self, config: OptimizationConfig):
        """
        初始化高级策略参数优化器

        Args:
            config: 优化配置
        """
        self.config = config
        self.optimization_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.parameter_spaces: Dict[str, ParameterSpace] = {}
        self.lock = threading.RLock()

        # 优化组件
        self.bayesian_optimizer = BayesianOptimizer()
        self.genetic_optimizer = GeneticOptimizer()
        self.particle_swarm_optimizer = ParticleSwarmOptimizer()
        self.grid_search_optimizer = GridSearchOptimizer()

        # 性能监控
        self.performance_monitor = OptimizationPerformanceMonitor()

        logger.info(f"高级策略参数优化器初始化完成: {config.optimization_id}")

    def add_parameter_space(self, parameter_space: ParameterSpace) -> bool:
        """
        添加参数空间

        Args:
            parameter_space: 参数空间定义

        Returns:
            bool: 是否添加成功
        """
        with self.lock:
            if parameter_space.name in self.parameter_spaces:
                logger.warning(f"参数空间 {parameter_space.name} 已存在")
                return False

            self.parameter_spaces[parameter_space.name] = parameter_space
            logger.info(f"参数空间 {parameter_space.name} 添加成功")
            return True

    def optimize_parameters(self, strategy_instance: Any,

                            training_data: pd.DataFrame,
                            validation_data: Optional[pd.DataFrame] = None) -> OptimizationResult:
        """
        优化策略参数

        Args:
            strategy_instance: 策略实例
            training_data: 训练数据
            validation_data: 验证数据（可选）

        Returns:
            OptimizationResult: 优化结果
        """
        with self.lock:
            start_time = time.time()

            # 准备优化目标函数
            objective_func = self._create_objective_function(
                strategy_instance, training_data, validation_data
            )

            # 根据配置选择优化方法
            if self.config.optimization_method == "bayesian":
                result = self.bayesian_optimizer.optimize(
                    objective_func, self.parameter_spaces, self.config
                )
            elif self.config.optimization_method == "genetic":
                result = self.genetic_optimizer.optimize(
                    objective_func, self.parameter_spaces, self.config
                )
            elif self.config.optimization_method == "particle_swarm":
                result = self.particle_swarm_optimizer.optimize(
                    objective_func, self.parameter_spaces, self.config
                )
            elif self.config.optimization_method == "grid_search":
                result = self.grid_search_optimizer.optimize(
                    objective_func, self.parameter_spaces, self.config
                )
            else:
                raise ValueError(f"不支持的优化方法: {self.config.optimization_method}")

            # 记录优化历史
            optimization_time = time.time() - start_time
            result.optimization_time = optimization_time
            result.optimization_id = self.config.optimization_id
            result.method_used = self.config.optimization_method

            self.optimization_history[self.config.optimization_id].append(result)

            logger.info(f"参数优化完成，最佳分数: {result.best_score:.4f}")
            return result

    def _create_objective_function(self, strategy_instance: Any,

                                   training_data: pd.DataFrame,
                                   validation_data: Optional[pd.DataFrame] = None) -> Callable:
        """
        创建优化目标函数

        Args:
            strategy_instance: 策略实例
            training_data: 训练数据
            validation_data: 验证数据

        Returns:
            Callable: 目标函数
        """

        def objective_function(trial_or_params):

            try:
                # 解析参数
                if hasattr(trial_or_params, 'suggest_float'):  # Optuna trial
                    params = {}
                    for param_name, param_space in self.parameter_spaces.items():
                        if param_space.parameter_type == "continuous":
                            params[param_name] = trial_or_params.suggest_float(
                                param_name, param_space.bounds[0], param_space.bounds[1]
                            )
                        elif param_space.parameter_type == "discrete":
                            params[param_name] = trial_or_params.suggest_int(
                                param_name, int(param_space.bounds[0]), int(param_space.bounds[1])
                            )
                        elif param_space.parameter_type == "categorical":
                            params[param_name] = trial_or_params.suggest_categorical(
                                param_name, param_space.bounds
                            )
                else:  # 直接参数
                    params = trial_or_params

                # 设置策略参数
                for param_name, param_value in params.items():
                    if hasattr(strategy_instance, param_name):
                        setattr(strategy_instance, param_name, param_value)

                # 评估策略性能
                score = self._evaluate_strategy(strategy_instance, training_data, validation_data)

                return -score  # 最小化负分数（最大化分数）

            except Exception as e:
                logger.error(f"目标函数评估失败: {e}")
                return float('inf')

        return objective_function

    def _evaluate_strategy(self, strategy_instance: Any,

                           training_data: pd.DataFrame,
                           validation_data: Optional[pd.DataFrame] = None) -> float:
        """
        评估策略性能

        Args:
            strategy_instance: 策略实例
            training_data: 训练数据
            validation_data: 验证数据

        Returns:
            float: 性能分数
        """
        try:
            # 使用训练数据训练策略
            if hasattr(strategy_instance, 'fit'):
                strategy_instance.fit(training_data)

            # 使用验证数据评估性能
            if validation_data is not None:
                if hasattr(strategy_instance, 'predict'):
                    predictions = strategy_instance.predict(validation_data)
                    score = self._calculate_performance_score(predictions, validation_data)
                elif hasattr(strategy_instance, 'generate_signals'):
                    signals = strategy_instance.generate_signals(validation_data)
                    score = self._calculate_signal_score(signals, validation_data)
                else:
                    score = 0.0
            else:
                # 使用训练数据评估
                if hasattr(strategy_instance, 'predict'):
                    predictions = strategy_instance.predict(training_data)
                    score = self._calculate_performance_score(predictions, training_data)
                elif hasattr(strategy_instance, 'generate_signals'):
                    signals = strategy_instance.generate_signals(training_data)
                    score = self._calculate_signal_score(signals, training_data)
                else:
                    score = 0.0

            return score

        except Exception as e:
            logger.error(f"策略评估失败: {e}")
            return 0.0

    def _calculate_performance_score(self, predictions: Any, data: pd.DataFrame) -> float:
        """
        计算性能分数

        Args:
            predictions: 预测结果
            data: 数据

        Returns:
            float: 性能分数
        """
        try:
            # 这里可以根据具体需求实现不同的评分方法
            # 例如：夏普比率、信息比率、最大回撤等

            if isinstance(predictions, pd.Series):
                # 计算夏普比率
                returns = predictions.pct_change().dropna()
                if len(returns) > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                    return max(0, sharpe_ratio)  # 确保非负
                else:
                    return 0.0
            else:
                return 0.0

        except Exception as e:
            logger.error(f"性能分数计算失败: {e}")
            return 0.0

    def _calculate_signal_score(self, signals: Any, data: pd.DataFrame) -> float:
        """
        计算信号分数

        Args:
            signals: 信号数据
            data: 数据

        Returns:
            float: 信号分数
        """
        try:
            if isinstance(signals, pd.DataFrame) and 'signal' in signals.columns:
                # 计算信号质量
                signal_quality = abs(signals['signal']).mean()
                return signal_quality
            else:
                return 0.0

        except Exception as e:
            logger.error(f"信号分数计算失败: {e}")
            return 0.0

    def get_optimization_history(self, optimization_id: str) -> List[OptimizationResult]:
        """
        获取优化历史

        Args:
            optimization_id: 优化ID

        Returns:
            List[OptimizationResult]: 优化历史
        """
        with self.lock:
            if optimization_id in self.optimization_history:
                return list(self.optimization_history[optimization_id])
            else:
                return []

    def plot_optimization_progress(self, optimization_id: str) -> Dict[str, Any]:
        """
        绘制优化进度

        Args:
            optimization_id: 优化ID

        Returns:
            Dict[str, Any]: 图表数据
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("plotly not available, cannot plot optimization progress")
            return {}

        history = self.get_optimization_history(optimization_id)
        if not history:
            return {}

        # 提取优化进度数据
        scores = [result.best_score for result in history]
        iterations = list(range(1, len(scores) + 1))

        # 创建进度图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iterations,
            y=scores,
            mode='lines+markers',
            name='优化进度',
            line=dict(color='blue'),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title=f'优化进度 - {optimization_id}',
            xaxis_title='迭代次数',
            yaxis_title='最佳分数',
            hovermode='x unified'
        )

        return {'optimization_progress': fig}


class BayesianOptimizer:

    """贝叶斯优化器"""

    def __init__(self):

        self.study = None

    def optimize(self, objective_function: Callable,


                 parameter_spaces: Dict[str, ParameterSpace],
                 config: OptimizationConfig) -> OptimizationResult:
        """
        贝叶斯优化

        Args:
            objective_function: 目标函数
            parameter_spaces: 参数空间
            config: 优化配置

        Returns:
            OptimizationResult: 优化结果
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("optuna not available, falling back to grid search")
            # 使用网格搜索作为fallback
            grid_optimizer = GridSearchOptimizer()
            return grid_optimizer.optimize(objective_function, parameter_spaces, config)

        # 创建Optuna研究
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=config.random_state)
        )

        # 运行优化
        study.optimize(
            objective_function,
            n_trials=config.n_trials,
            timeout=config.timeout_seconds
        )

        # 提取结果
        best_params = study.best_params
        best_score = -study.best_value  # 转换回正分数

        # 构建优化历史
        optimization_history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                optimization_history.append({
                    'iteration': trial.number,
                    'score': -trial.value,
                    'parameters': trial.params
                })

        # 构建收敛曲线
        convergence_curve = [-trial.value for trial in study.trials if trial.state ==
                             optuna.trial.TrialState.COMPLETE]

        result = OptimizationResult(
            optimization_id=config.optimization_id,
            best_parameters=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            convergence_curve=convergence_curve,
            optimization_time=0.0,  # 将在外部设置
            method_used="bayesian",
            status="success",
            metadata={'n_trials': len(study.trials)}
        )

        return result


class GeneticOptimizer:

    """遗传算法优化器"""

    def __init__(self):

        self.population = []
        self.fitness_history = []

    def optimize(self, objective_function: Callable,


                 parameter_spaces: Dict[str, ParameterSpace],
                 config: OptimizationConfig) -> OptimizationResult:
        """
        遗传算法优化

        Args:
            objective_function: 目标函数
            parameter_spaces: 参数空间
            config: 优化配置

        Returns:
            OptimizationResult: 优化结果
        """
        # 准备参数边界
        bounds = []
        param_names = []
        for param_name, param_space in parameter_spaces.items():
            bounds.append(param_space.bounds)
            param_names.append(param_name)

        # 运行差分进化
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=config.max_iterations,
            popsize=config.population_size,
            tol=config.convergence_threshold,
            seed=config.random_state
        )

        # 提取结果
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun  # 转换回正分数

        # 构建优化历史（简化版本）
        optimization_history = [{
            'iteration': i,
            'score': -result.fun,
            'parameters': best_params
        } for i in range(config.max_iterations)]

        # 构建收敛曲线
        convergence_curve = [-result.fun] * config.max_iterations

        result_obj = OptimizationResult(
            optimization_id=config.optimization_id,
            best_parameters=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            convergence_curve=convergence_curve,
            optimization_time=0.0,  # 将在外部设置
            method_used="genetic",
            status="success" if result.success else "error",
            metadata={'n_iterations': result.nit}
        )

        return result_obj


class ParticleSwarmOptimizer:

    """粒子群优化器"""

    def __init__(self):

        self.particles = []
        self.velocities = []
        self.best_positions = []
        self.best_scores = []

    def optimize(self, objective_function: Callable,


                 parameter_spaces: Dict[str, ParameterSpace],
                 config: OptimizationConfig) -> OptimizationResult:
        """
        粒子群优化

        Args:
            objective_function: 目标函数
            parameter_spaces: 参数空间
            config: 优化配置

        Returns:
            OptimizationResult: 优化结果
        """
        # 准备参数边界
        bounds = []
        param_names = []
        for param_name, param_space in parameter_spaces.items():
            bounds.append(param_space.bounds)
            param_names.append(param_name)

        # 运行粒子群优化
        result = minimize(
            objective_function,
            x0=[(b[0] + b[1]) / 2 for b in bounds],  # 初始位置
            bounds=bounds,
            method='L - BFGS - B',
            options={'maxiter': config.max_iterations}
        )

        # 提取结果
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun  # 转换回正分数

        # 构建优化历史（简化版本）
        optimization_history = [{
            'iteration': i,
            'score': -result.fun,
            'parameters': best_params
        } for i in range(config.max_iterations)]

        # 构建收敛曲线
        convergence_curve = [-result.fun] * config.max_iterations

        result_obj = OptimizationResult(
            optimization_id=config.optimization_id,
            best_parameters=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            convergence_curve=convergence_curve,
            optimization_time=0.0,  # 将在外部设置
            method_used="particle_swarm",
            status="success" if result.success else "error",
            metadata={'n_iterations': result.nit}
        )

        return result_obj


class GridSearchOptimizer:

    """网格搜索优化器"""

    def __init__(self):

        self.grid_points = []
        self.scores = []

    def optimize(self, objective_function: Callable,


                 parameter_spaces: Dict[str, ParameterSpace],
                 config: OptimizationConfig) -> OptimizationResult:
        """
        网格搜索优化

        Args:
            objective_function: 目标函数
            parameter_spaces: 参数空间
            config: 优化配置

        Returns:
            OptimizationResult: 优化结果
        """
        # 生成网格点
        param_combinations = self._generate_grid_points(parameter_spaces, config)

        best_score = float('-inf')
        best_params = {}
        optimization_history = []
        convergence_curve = []

        # 遍历网格点
        for i, params in enumerate(param_combinations):
            try:
                score = -objective_function(params)  # 转换回正分数
                optimization_history.append({
                    'iteration': i,
                    'score': score,
                    'parameters': params
                })
                convergence_curve.append(score)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                logger.error(f"网格搜索评估失败: {e}")
                continue

        result = OptimizationResult(
            optimization_id=config.optimization_id,
            best_parameters=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            convergence_curve=convergence_curve,
            optimization_time=0.0,  # 将在外部设置
            method_used="grid_search",
            status="success",
            metadata={'n_combinations': len(param_combinations)}
        )

        return result

    def _generate_grid_points(self, parameter_spaces: Dict[str, ParameterSpace],


                              config: OptimizationConfig) -> List[Dict[str, Any]]:
        """
        生成网格点

        Args:
            parameter_spaces: 参数空间
            config: 优化配置

        Returns:
            List[Dict[str, Any]]: 参数组合列表
        """
        param_values = {}
        for param_name, param_space in parameter_spaces.items():
            if param_space.parameter_type == "continuous":
                # 连续参数，生成等间距点
                n_points = min(10, int(np.sqrt(config.n_trials)))
                param_values[param_name] = np.linspace(
                    param_space.bounds[0], param_space.bounds[1], n_points
                )
            elif param_space.parameter_type == "discrete":
                # 离散参数，生成整数序列
                param_values[param_name] = range(
                    int(param_space.bounds[0]), int(param_space.bounds[1]) + 1
                )
            elif param_space.parameter_type == "categorical":
                # 分类参数，使用预定义值
                param_values[param_name] = param_space.bounds

        # 生成所有组合
        import itertools
        param_names = list(param_values.keys())
        param_value_lists = list(param_values.values())

        combinations = []
        for combination in itertools.product(*param_value_lists):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)

        return combinations


class OptimizationPerformanceMonitor:

    """优化性能监控器"""

    def __init__(self):

        self.metrics = defaultdict(list)
        self.alerts = []

    def record_metric(self, optimization_id: str, metric_name: str, value: float):
        """记录指标"""
        self.metrics[f"{optimization_id}_{metric_name}"].append({
            'timestamp': datetime.now(),
            'value': value
        })

    def get_metrics(self, optimization_id: str, metric_name: str,


                    lookback_days: int = 30) -> List[float]:
        """获取指标历史"""
        key = f"{optimization_id}_{metric_name}"
        if key not in self.metrics:
            return []

        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        recent_metrics = [
            m['value'] for m in self.metrics[key]
            if m['timestamp'] >= cutoff_time
        ]

        return recent_metrics
