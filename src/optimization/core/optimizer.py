import time
"""
Optimizer Module
优化器模块

This module provides the base optimizer implementation for quantitative trading strategies
此模块为量化交易策略提供基础优化器实现

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class OptimizationMethod(Enum):

    """Optimization methods"""
    GRADIENT_DESCENT = "gradient_descent"
    NEWTON_METHOD = "newton_method"
    QUASI_NEWTON = "quasi_newton"
    CONJUGATE_GRADIENT = "conjugate_gradient"
    STOCHASTIC_GRADIENT = "stochastic_gradient"
    ADAM = "adam"
    RMSPROP = "rmsprop"


class ConvergenceCriteria(Enum):

    """Convergence criteria"""
    MAX_ITERATIONS = "max_iterations"
    TOLERANCE = "tolerance"
    GRADIENT_NORM = "gradient_norm"
    FUNCTION_CHANGE = "function_change"
    PARAMETER_CHANGE = "parameter_change"


class OptimizationResult:

    """
    Optimization Result Class
    优化结果类

    Contains the results of an optimization run
    包含优化运行的结果
    """

    def __init__(self,


                 success: bool,
                 optimal_parameters: np.ndarray,
                 optimal_value: float,
                 iterations: int,
                 convergence_reason: str,
                 execution_time: float,
                 method_used: str,
                 convergence_info: Optional[Dict[str, Any]] = None):
        """
        Initialize optimization result
        初始化优化结果

        Args:
            success: Whether optimization was successful
                    优化是否成功
            optimal_parameters: Optimal parameter values
                               最优参数值
            optimal_value: Optimal objective function value
                          最优目标函数值
            iterations: Number of iterations performed
                       执行的迭代次数
            convergence_reason: Reason for convergence
                               收敛原因
            execution_time: Time taken for optimization
                           优化所用时间
            method_used: Optimization method used
                        使用的优化方法
            convergence_info: Additional convergence information
                             其他收敛信息
        """
        self.success = success
        self.optimal_parameters = optimal_parameters
        self.optimal_value = optimal_value
        self.iterations = iterations
        self.convergence_reason = convergence_reason
        self.execution_time = execution_time
        self.method_used = method_used
        self.convergence_info = convergence_info or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        转换为字典

        Returns:
            dict: Result data as dictionary
                  结果数据字典
        """
        return {
            'success': self.success,
            'optimal_parameters': self.optimal_parameters.tolist() if isinstance(self.optimal_parameters, np.ndarray) else self.optimal_parameters,
            'optimal_value': self.optimal_value,
            'iterations': self.iterations,
            'convergence_reason': self.convergence_reason,
            'execution_time': self.execution_time,
            'method_used': self.method_used,
            'convergence_info': self.convergence_info,
            'timestamp': self.timestamp.isoformat()
        }


class BaseOptimizer(ABC):

    """
    Base Optimizer Class
    基础优化器类

    Abstract base class for all optimizers
    所有优化器的抽象基类
    """

    def __init__(self,


                 method: OptimizationMethod,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 convergence_criteria: List[ConvergenceCriteria] = None):
        """
        Initialize base optimizer
        初始化基础优化器

        Args:
            method: Optimization method to use
                   要使用的优化方法
            max_iterations: Maximum number of iterations
                           最大迭代次数
            tolerance: Convergence tolerance
                      收敛容差
            convergence_criteria: List of convergence criteria
                                 收敛准则列表
        """
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.convergence_criteria = convergence_criteria or [
            ConvergenceCriteria.MAX_ITERATIONS,
            ConvergenceCriteria.TOLERANCE,
            ConvergenceCriteria.GRADIENT_NORM
        ]

        # Optimization state
        self.current_iteration = 0
        self.best_parameters = None
        self.best_value = float('inf')
        self.convergence_history = []

        # Statistics
        self.optimization_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'average_iterations': 0.0,
            'average_execution_time': 0.0
        }

        logger.info(f"Base optimizer initialized with method: {method.value}")

    @abstractmethod
    def optimize(self,

                 objective_function: Callable,
                 initial_parameters: np.ndarray,
                 bounds: Optional[List[tuple]] = None,
                 constraints: Optional[List[Callable]] = None) -> OptimizationResult:
        """
        Perform optimization
        执行优化

        Args:
            objective_function: Function to minimize
                               要最小化的函数
            initial_parameters: Initial parameter values
                              初始参数值
            bounds: Parameter bounds
                   参数界限
            constraints: Constraint functions
                        约束函数

        Returns:
            OptimizationResult: Optimization result
                               优化结果
        """

    def _check_convergence(self,

                           current_value: float,
                           previous_value: float,
                           gradient: Optional[np.ndarray] = None,
                           parameter_change: Optional[np.ndarray] = None) -> tuple:
        """
        Check convergence criteria
        检查收敛准则

        Args:
            current_value: Current objective function value
                          当前目标函数值
            previous_value: Previous objective function value
                          前一个目标函数值
            gradient: Current gradient
                     当前梯度
            parameter_change: Change in parameters
                             参数变化

        Returns:
            tuple: (converged, reason)
                  (是否收敛, 原因)
        """
        # Check maximum iterations
        if (ConvergenceCriteria.MAX_ITERATIONS in self.convergence_criteria
                and self.current_iteration >= self.max_iterations):
            return True, "Maximum iterations reached"

        # Check tolerance
        if ConvergenceCriteria.TOLERANCE in self.convergence_criteria:
            if abs(current_value - previous_value) < self.tolerance:
                return True, "Function value change below tolerance"

        # Check gradient norm
        if (ConvergenceCriteria.GRADIENT_NORM in self.convergence_criteria
                and gradient is not None):
            if np.linalg.norm(gradient) < self.tolerance:
                return True, "Gradient norm below tolerance"

        # Check parameter change
        if (ConvergenceCriteria.PARAMETER_CHANGE in self.convergence_criteria
                and parameter_change is not None):
            if np.linalg.norm(parameter_change) < self.tolerance:
                return True, "Parameter change below tolerance"

        return False, "Not converged"

    def _update_best_solution(self, parameters: np.ndarray, value: float) -> None:
        """
        Update best solution found
        更新找到的最优解

        Args:
            parameters: Parameter values
                       参数值
            value: Objective function value
                  目标函数值
        """
        if value < self.best_value:
            self.best_value = value
            self.best_parameters = parameters.copy()

    def _record_convergence_info(self,

                                 parameters: np.ndarray,
                                 value: float,
                                 gradient: Optional[np.ndarray] = None) -> None:
        """
        Record convergence information
        记录收敛信息

        Args:
            parameters: Current parameters
                       当前参数
            value: Current objective value
                  当前目标值
            gradient: Current gradient
                     当前梯度
        """
        info = {
            'iteration': self.current_iteration,
            'parameters': parameters.copy(),
            'value': value,
            'timestamp': datetime.now()
        }

        if gradient is not None:
            info['gradient_norm'] = np.linalg.norm(gradient)

        self.convergence_history.append(info)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics
        获取优化统计信息

        Returns:
            dict: Optimization statistics
                  优化统计信息
        """
        stats = self.optimization_stats.copy()

        if stats['total_runs'] > 0:
            stats['success_rate'] = stats['successful_runs'] / stats['total_runs'] * 100

        return stats

    def reset_optimizer(self) -> None:
        """
        Reset optimizer state
        重置优化器状态

        Returns:
            None
        """
        self.current_iteration = 0
        self.best_parameters = None
        self.best_value = float('inf')
        self.convergence_history.clear()


class GradientDescentOptimizer(BaseOptimizer):

    """
    Gradient Descent Optimizer Class
    梯度下降优化器类

    Implements gradient descent optimization method
    实现梯度下降优化方法
    """

    def __init__(self,


                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 **kwargs):
        """
        Initialize gradient descent optimizer
        初始化梯度下降优化器

        Args:
            learning_rate: Learning rate for gradient descent
                          梯度下降的学习率
            momentum: Momentum coefficient
                     动量系数
            nesterov: Whether to use Nesterov momentum
                     是否使用Nesterov动量
            **kwargs: Additional arguments for base optimizer
                     基础优化器的其他参数
        """
        super().__init__(OptimizationMethod.GRADIENT_DESCENT, **kwargs)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

        # Momentum state
        self.velocity = None

    def optimize(self,

                 objective_function: Callable,
                 initial_parameters: np.ndarray,
                 bounds: Optional[List[tuple]] = None,
                 constraints: Optional[List[Callable]] = None) -> OptimizationResult:
        """
        Perform gradient descent optimization
        执行梯度下降优化

        Args:
            objective_function: Function to minimize
                               要最小化的函数
            initial_parameters: Initial parameter values
                              初始参数值
            bounds: Parameter bounds (ignored in basic implementation)
                   参数界限（在基本实现中忽略）
            constraints: Constraint functions (ignored in basic implementation)
                        约束函数（在基本实现中忽略）

        Returns:
            OptimizationResult: Optimization result
                               优化结果
        """
        start_time = time.time()

        try:
            # Initialize parameters
            parameters = initial_parameters.copy()
            self.velocity = np.zeros_like(parameters) if self.momentum > 0 else None

            previous_value = float('inf')
            gradient = None

            for iteration in range(self.max_iterations):
                self.current_iteration = iteration + 1

                # Compute objective function and gradient
                value = objective_function(parameters)

                # Simple finite difference gradient approximation
                gradient = self._compute_gradient(objective_function, parameters)

                # Update best solution
                self._update_best_solution(parameters, value)

                # Record convergence info
                self._record_convergence_info(parameters, value, gradient)

                # Check convergence
                converged, reason = self._check_convergence(value, previous_value, gradient)
                if converged:
                    execution_time = time.time() - start_time
                    self._update_stats(True, self.current_iteration, execution_time)

                    return OptimizationResult(
                        success=True,
                        optimal_parameters=self.best_parameters,
                        optimal_value=self.best_value,
                        iterations=self.current_iteration,
                        convergence_reason=reason,
                        execution_time=execution_time,
                        method_used=self.method.value
                    )

                # Update parameters using gradient descent
                self._update_parameters(parameters, gradient)

                previous_value = value

            # Maximum iterations reached
            execution_time = time.time() - start_time
            self._update_stats(False, self.current_iteration, execution_time)

            return OptimizationResult(
                success=False,
                optimal_parameters=self.best_parameters,
                optimal_value=self.best_value,
                iterations=self.current_iteration,
                convergence_reason="Maximum iterations reached",
                execution_time=execution_time,
                method_used=self.method.value
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self._update_stats(False, self.current_iteration, execution_time)

            logger.error(f"Gradient descent optimization failed: {str(e)}")
            return OptimizationResult(
                success=False,
                optimal_parameters=initial_parameters,
                optimal_value=float('inf'),
                iterations=self.current_iteration,
                convergence_reason=f"Optimization error: {str(e)}",
                execution_time=execution_time,
                method_used=self.method.value
            )

    def _compute_gradient(self,

                          objective_function: Callable,
                          parameters: np.ndarray,
                          epsilon: float = 1e-8) -> np.ndarray:
        """
        Compute gradient using finite differences
        使用有限差分计算梯度

        Args:
            objective_function: Objective function
                               目标函数
            parameters: Parameter values
                       参数值
            epsilon: Finite difference step size
                    有限差分步长

        Returns:
            np.ndarray: Gradient vector
                       梯度向量
        """
        gradient = np.zeros_like(parameters)
        base_value = objective_function(parameters)

        for i in range(len(parameters)):
            parameters[i] += epsilon
            perturbed_value = objective_function(parameters)
            gradient[i] = (perturbed_value - base_value) / epsilon
            parameters[i] -= epsilon

        return gradient

    def _update_parameters(self, parameters: np.ndarray, gradient: np.ndarray) -> None:
        """
        Update parameters using gradient descent
        使用梯度下降更新参数

        Args:
            parameters: Current parameters
                       当前参数
            gradient: Gradient vector
                     梯度向量
        """
        if self.velocity is not None:
            # Apply momentum
            if self.nesterov:
                # Nesterov accelerated gradient
                # For Nesterov, we need to compute gradient at lookahead position
                # This requires access to objective_function, but we don't have it here
                # Use current gradient as approximation
                lookahead_gradient = gradient
                self.velocity = self.momentum * self.velocity + self.learning_rate * lookahead_gradient
            else:
                # Standard momentum
                self.velocity = self.momentum * self.velocity + self.learning_rate * gradient

            parameters -= self.velocity
        else:
            # Standard gradient descent
            parameters -= self.learning_rate * gradient

    def _update_stats(self, success: bool, iterations: int, execution_time: float) -> None:
        """
        Update optimization statistics
        更新优化统计信息

        Args:
            success: Whether optimization was successful
                    优化是否成功
            iterations: Number of iterations
                       迭代次数
            execution_time: Execution time
                           执行时间
        """
        self.optimization_stats['total_runs'] += 1

        if success:
            self.optimization_stats['successful_runs'] += 1

        # Update averages
        total_runs = self.optimization_stats['total_runs']
        current_avg_iter = self.optimization_stats['average_iterations']
        current_avg_time = self.optimization_stats['average_execution_time']

        self.optimization_stats['average_iterations'] = (
            (current_avg_iter * (total_runs - 1)) + iterations
        ) / total_runs

        self.optimization_stats['average_execution_time'] = (
            (current_avg_time * (total_runs - 1)) + execution_time
        ) / total_runs


# Global optimizer instance
# 全局优化器实例
gradient_descent_optimizer = GradientDescentOptimizer()

__all__ = [
    'OptimizationMethod',
    'ConvergenceCriteria',
    'OptimizationResult',
    'BaseOptimizer',
    'GradientDescentOptimizer',
    'gradient_descent_optimizer'
]
