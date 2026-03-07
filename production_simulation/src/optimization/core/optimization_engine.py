"""
Optimization Engine Module
优化引擎模块

This module provides the core optimization engine for quantitative trading strategies
此模块为量化交易策略提供核心优化引擎

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
import scipy.optimize as sco

# 尝试导入常量，如果失败则使用默认值
try:
    from src.core.constants import *
except ImportError:
    try:
        from src.constants import *
    except ImportError:
        # 定义默认常量
        DEFAULT_TIMEOUT = 30.0
        MAX_RETRIES = 3
        RETRY_DELAY = 1.0

# 尝试导入异常，如果失败则使用标准异常
try:
    from src.core.exceptions import *
except ImportError:
    try:
        from src.exceptions import *
    except ImportError:
        # 使用标准异常
        pass

logger = logging.getLogger(__name__)

T = TypeVar('T')


class OptimizationObjective(Enum):

    """Optimization objectives"""
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE_RATIO = "maximize_sharpe_ratio"
    MAXIMIZE_SORTINO_RATIO = "maximize_sortino_ratio"
    MAXIMIZE_CALMAR_RATIO = "maximize_calmar_ratio"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_PROFIT_FACTOR = "maximize_profit_factor"


class OptimizationConstraint(Enum):

    """Optimization constraints"""
    NO_SHORT_SELLING = "no_short_selling"
    MAX_WEIGHT = "max_weight"
    MIN_WEIGHT = "min_weight"
    SECTOR_LIMITS = "sector_limits"
    TURNOVER_LIMIT = "turnover_limit"
    RISK_BUDGET = "risk_budget"


class OptimizationAlgorithm(Enum):

    """Optimization algorithms"""
    SLSQP = "SLSQP"                    # Sequential Least Squares Programming
    COBYLA = "COBYLA"                  # Constrained Optimization BY Linear Approximation
    BFGS = "BFGS"                      # Broyden - Fletcher - Goldfarb - Shanno
    LBFGSB = "L - BFGS - B"                # Limited - memory BFGS with bounds
    TRUST_CONSTR = "trust - constr"      # Trust region constrained
    DIFFERENTIAL_EVOLUTION = "differential_evolution"  # Differential evolution
    SHGO = "shgo"                      # Simplicial homology global optimization


class OptimizationResult:

    """
    Optimization Result Class
    优化结果类

    Contains the results of an optimization run
    包含优化运行的结果
    """

    def __init__(self,


                 success: bool,
                 optimal_weights: np.ndarray,
                 objective_value: float,
                 convergence_info: Dict[str, Any],
                 execution_time: float,
                 algorithm_used: str,
                 constraints_satisfied: bool = True):
        """
        Initialize optimization result
        初始化优化结果

        Args:
            success: Whether optimization was successful
                    优化是否成功
            optimal_weights: Optimal portfolio weights
                           最优投资组合权重
            objective_value: Value of the objective function
                           目标函数的值
            convergence_info: Information about convergence
                            收敛信息
            execution_time: Time taken for optimization
                           优化所用时间
            algorithm_used: Algorithm that was used
                           使用的算法
            constraints_satisfied: Whether all constraints were satisfied
                                 是否满足所有约束
        """
        self.success = success
        self.optimal_weights = optimal_weights
        self.objective_value = objective_value
        self.convergence_info = convergence_info
        self.execution_time = execution_time
        self.algorithm_used = algorithm_used
        self.constraints_satisfied = constraints_satisfied
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
            'optimal_weights': self.optimal_weights.tolist() if isinstance(self.optimal_weights, np.ndarray) else self.optimal_weights,
            'objective_value': self.objective_value,
            'convergence_info': self.convergence_info,
            'execution_time': self.execution_time,
            'algorithm_used': self.algorithm_used,
            'constraints_satisfied': self.constraints_satisfied,
            'timestamp': self.timestamp.isoformat()
        }


class OptimizationEngine:

    """
    Optimization Engine Class
    优化引擎类

    Core engine for portfolio and strategy optimization
    投资组合和策略优化的核心引擎
    """

    def __init__(self, name: str = "default_optimization_engine"):
        """
        Initialize the optimization engine
        初始化优化引擎

        Args:
            name: Name of the optimization engine
                优化引擎的名称
        """
        self.name = name
        self.objective_functions = {}
        self.constraint_functions = {}
        self.optimization_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'average_execution_time': 0.0,
            'last_run_timestamp': None
        }

        # Register default objective functions
        self._register_default_objectives()

        # Register default constraint functions
        self._register_default_constraints()

        logger.info(f"Optimization engine {name} initialized")

    def _register_default_objectives(self) -> None:
        """Register default objective functions"""
        self.register_objective_function(
            OptimizationObjective.MAXIMIZE_RETURN, self._maximize_return)
        self.register_objective_function(OptimizationObjective.MINIMIZE_RISK, self._minimize_risk)
        self.register_objective_function(
            OptimizationObjective.MAXIMIZE_SHARPE_RATIO, self._maximize_sharpe_ratio)
        self.register_objective_function(
            OptimizationObjective.MAXIMIZE_SORTINO_RATIO, self._maximize_sortino_ratio)

    def _register_default_constraints(self) -> None:
        """Register default constraint functions"""
        self.register_constraint_function(
            OptimizationConstraint.NO_SHORT_SELLING, self._no_short_selling_constraint)
        self.register_constraint_function(
            OptimizationConstraint.MAX_WEIGHT, self._max_weight_constraint)
        self.register_constraint_function(
            OptimizationConstraint.MIN_WEIGHT, self._min_weight_constraint)

    def register_objective_function(self,

                                    objective_type: OptimizationObjective,
                                    func: Callable) -> None:
        """
        Register an objective function
        注册目标函数

        Args:
            objective_type: Type of objective function
                           目标函数类型
            func: Objective function to register
                 要注册的目标函数
        """
        self.objective_functions[objective_type] = func
        logger.debug(f"Registered objective function: {objective_type.value}")

    def register_constraint_function(self,

                                     constraint_type: OptimizationConstraint,
                                     func: Callable) -> None:
        """
        Register a constraint function
        注册约束函数

        Args:
            constraint_type: Type of constraint function
                            约束函数类型
            func: Constraint function to register
                 要注册的约束函数
        """
        self.constraint_functions[constraint_type] = func
        logger.debug(f"Registered constraint function: {constraint_type.value}")

    def optimize_portfolio(self,

                           returns: pd.DataFrame,
                           objective: OptimizationObjective,
                           constraints: List[OptimizationConstraint],
                           algorithm: OptimizationAlgorithm = OptimizationAlgorithm.SLSQP,
                           bounds: Optional[List[tuple]] = None,
                           **kwargs) -> OptimizationResult:
        """
        Optimize a portfolio
        优化投资组合

        Args:
            returns: Historical returns data
                    历史收益数据
            objective: Optimization objective
                      优化目标
            constraints: List of constraints to apply
                        要应用的约束列表
            algorithm: Optimization algorithm to use
                      要使用的优化算法
            bounds: Bounds for each asset weight
                   每个资产权重的界限
            **kwargs: Additional arguments for the optimization
                     优化的其他参数

        Returns:
            OptimizationResult: Result of the optimization
                               优化结果
        """
        start_time = datetime.now()

        try:
            # Validate inputs
            if returns.empty:
                raise ValueError("Returns data cannot be empty")

            n_assets = len(returns.columns)

            # Set default bounds if not provided
            if bounds is None:
                bounds = [(0, 1) for _ in range(n_assets)]

            # Initial guess (equal weights)
            initial_weights = np.array([1.0 / n_assets] * n_assets)

            # Get objective function
            if objective not in self.objective_functions:
                raise ValueError(f"Unknown objective: {objective}")

            objective_func = self.objective_functions[objective]

            # Prepare constraints
            opt_constraints = []
            for constraint in constraints:
                if constraint in self.constraint_functions:
                    constraint_func = self.constraint_functions[constraint]
                    opt_constraints.append(constraint_func)

            # Prepare arguments for objective function
            objective_args = self._prepare_objective_args(returns, objective, **kwargs)

            # Run optimization
            if algorithm.value in ['SLSQP', 'COBYLA', 'BFGS', 'L - BFGS - B', 'trust - constr']:
                # Local optimization methods
                result = sco.minimize(
                    # Minimize negative for maximization
                    fun=lambda w: -objective_func(w, *objective_args),
                    x0=initial_weights,
                    method=algorithm.value,
                    bounds=bounds,
                    constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
                )

                optimal_weights = result.x
                objective_value = -result.fun  # Negate back for maximization problems
                success = result.success

            elif algorithm.value == 'differential_evolution':
                # Global optimization
                bounds_array = np.array(bounds)
                result = sco.differential_evolution(
                    func=lambda w: -objective_func(w, *objective_args),
                    bounds=bounds_array,
                    **kwargs
                )

                optimal_weights = result.x
                objective_value = -result.fun
                success = result.success

            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # Check constraints satisfaction
            constraints_satisfied = self._check_constraints_satisfaction(
                optimal_weights, constraints, returns, **kwargs
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Update statistics
            self._update_stats(success, execution_time)

            convergence_info = {
                'success': success,
                'message': getattr(result, 'message', 'Optimization completed'),
                'nfev': getattr(result, 'nfev', None),
                'njev': getattr(result, 'njev', None),
                'nhev': getattr(result, 'nhev', None)
            }

            return OptimizationResult(
                success=success,
                optimal_weights=optimal_weights,
                objective_value=objective_value,
                convergence_info=convergence_info,
                execution_time=execution_time,
                algorithm_used=algorithm.value,
                constraints_satisfied=constraints_satisfied
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Optimization failed: {str(e)}")

            # Update statistics
            self._update_stats(False, execution_time)

            return OptimizationResult(
                success=False,
                optimal_weights=np.array([]),
                objective_value=0.0,
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                algorithm_used=algorithm.value,
                constraints_satisfied=False
            )

    def _prepare_objective_args(self, returns: pd.DataFrame, objective: OptimizationObjective, **kwargs) -> tuple:
        """
        Prepare arguments for objective function
        为目标函数准备参数
        Args:
            returns: Returns data
                    收益数据
            objective: Optimization objective
                      优化目标
            **kwargs: Additional arguments
                     其他参数

        Returns:
            tuple: Arguments for objective function
                  目标函数的参数
        """
        if objective in [OptimizationObjective.MAXIMIZE_RETURN, OptimizationObjective.MINIMIZE_RISK]:
            # Expected returns and covariance matrix
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            return (expected_returns, cov_matrix)

        elif objective in [OptimizationObjective.MAXIMIZE_SHARPE_RATIO, OptimizationObjective.MAXIMIZE_SORTINO_RATIO]:
            # Expected returns, covariance matrix, and risk - free rate
            expected_returns = returns.mean()
            cov_matrix = returns.cov()
            risk_free_rate = kwargs.get('risk_free_rate', 0.02)  # 默认无风险利率2%
            return (expected_returns, cov_matrix, risk_free_rate)

        else:
            return (returns,)

    def _maximize_return(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """Maximize portfolio return"""
        return np.dot(weights, expected_returns)

    def _minimize_risk(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """Minimize portfolio risk (variance)"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def _maximize_sharpe_ratio(self, weights: np.ndarray, expected_returns: pd.Series,


                               cov_matrix: pd.DataFrame, risk_free_rate: float) -> float:
        """Maximize Sharpe ratio"""
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

    def _maximize_sortino_ratio(self, weights: np.ndarray, expected_returns: pd.Series,


                                cov_matrix: pd.DataFrame, risk_free_rate: float) -> float:
        """Maximize Sortino ratio"""
        portfolio_return = np.dot(weights, expected_returns)
        # Simplified downside risk calculation
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

    def _no_short_selling_constraint(self) -> dict:
        """No short selling constraint"""
        return {'type': 'ineq', 'fun': lambda w: w}  # w >= 0

    def _max_weight_constraint(self, max_weight: float = 0.3) -> dict:
        """Maximum weight constraint"""
        return {'type': 'ineq', 'fun': lambda w: max_weight - w}

    def _min_weight_constraint(self, min_weight: float = 0.0) -> dict:
        """Minimum weight constraint"""
        return {'type': 'ineq', 'fun': lambda w: w - min_weight}

    def _check_constraints_satisfaction(self,


                                        weights: np.ndarray,
                                        constraints: List[OptimizationConstraint],
                                        returns: pd.DataFrame,
                                        **kwargs) -> bool:
        """
        Check if constraints are satisfied
        检查约束是否满足

        Args:
            weights: Portfolio weights
                    投资组合权重
            constraints: List of constraints
                        约束列表
            returns: Returns data
                    收益数据
            **kwargs: Additional arguments
                     其他参数

        Returns:
            bool: True if all constraints satisfied
                  如果所有约束都满足则返回True
        """
        try:
            # Check weight constraints
            if OptimizationConstraint.NO_SHORT_SELLING in constraints:
                if np.any(weights < 0):
                    return False

            if OptimizationConstraint.MAX_WEIGHT in constraints:
                max_weight = kwargs.get('max_weight', 0.3)
                if np.any(weights > max_weight):
                    return False

            # Check sum constraint
            if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
                return False

            return True

        except Exception as e:
            logger.error(f"Constraint check failed: {str(e)}")
            return False

    def _update_stats(self, success: bool, execution_time: float) -> None:
        """
        Update optimization statistics
        更新优化统计信息

        Args:
            success: Whether optimization was successful
                    优化是否成功
            execution_time: Time taken for optimization
                           优化所用时间
        """
        self.optimization_stats['total_runs'] += 1
        self.optimization_stats['last_run_timestamp'] = datetime.now()

        if success:
            self.optimization_stats['successful_runs'] += 1

        # Update average execution time
        total_runs = self.optimization_stats['total_runs']
        current_avg = self.optimization_stats['average_execution_time']
        self.optimization_stats['average_execution_time'] = (
            (current_avg * (total_runs - 1)) + execution_time
        ) / total_runs

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics
        获取优化统计信息

        Returns:
            dict: Optimization statistics
                  优化统计信息
        """
        stats = self.optimization_stats.copy()
        stats['success_rate'] = (
            stats['successful_runs'] / stats['total_runs'] * 100
            if stats['total_runs'] > 0 else 0
        )
        return stats

    def optimize_strategy_parameters(self,


                                     strategy_func: Callable,
                                     parameter_bounds: Dict[str, tuple],
                                     historical_data: pd.DataFrame,
                                     objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_RETURN,
                                     algorithm: OptimizationAlgorithm = OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION,
                                     **kwargs) -> OptimizationResult:
        """
        Optimize strategy parameters
        优化策略参数

        Args:
            strategy_func: Strategy function to optimize
                          要优化的策略函数
            parameter_bounds: Bounds for each parameter
                             每个参数的界限
            historical_data: Historical data for backtesting
                            用于回测的历史数据
            objective: Optimization objective
                      优化目标
            algorithm: Optimization algorithm
                      优化算法
            **kwargs: Additional arguments
                     其他参数

        Returns:
            OptimizationResult: Optimization result
                               优化结果
        """
        start_time = datetime.now()

        try:
            # Convert parameter bounds to scipy format
            bounds = list(parameter_bounds.values())
            param_names = list(parameter_bounds.keys())

            def objective_wrapper(params):
                """Wrapper for strategy objective evaluation"""
                param_dict = dict(zip(param_names, params))
                return -self._evaluate_strategy(strategy_func, param_dict, historical_data, objective)

            # Run optimization
            if algorithm.value == 'differential_evolution':
                result = sco.differential_evolution(
                    func=objective_wrapper,
                    bounds=bounds,
                    **kwargs
                )

            optimal_params = result.x
            objective_value = -result.fun
            success = result.success

            execution_time = (datetime.now() - start_time).total_seconds()

            # Update statistics
            self._update_stats(success, execution_time)

            return OptimizationResult(
                success=success,
                optimal_weights=np.array(optimal_params),  # Using weights field for parameters
                objective_value=objective_value,
                convergence_info={
                    'success': success,
                    'message': getattr(result, 'message', 'Parameter optimization completed'),
                    'nfev': getattr(result, 'nfev', None)
                },
                execution_time=execution_time,
                algorithm_used=algorithm.value
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Strategy parameter optimization failed: {str(e)}")

            self._update_stats(False, execution_time)

            return OptimizationResult(
                success=False,
                optimal_weights=np.array([]),
                objective_value=0.0,
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                algorithm_used=algorithm.value,
                constraints_satisfied=False
            )

    def _evaluate_strategy(self,


                           strategy_func: Callable,
                           params: Dict[str, Any],
                           data: pd.DataFrame,
                           objective: OptimizationObjective) -> float:
        """
        Evaluate strategy performance for optimization
        为优化评估策略性能

        Args:
            strategy_func: Strategy function
                          策略函数
            params: Strategy parameters
                   策略参数
            data: Historical data
                 历史数据
            objective: Optimization objective
                      优化目标

        Returns:
            float: Objective value
                  目标值
        """
        try:
            # This is a simplified implementation
            # In practice, you would run the strategy on historical data
            # and calculate the objective metric
            return 0.0  # Placeholder

        except Exception as e:
            logger.error(f"Strategy evaluation failed: {str(e)}")
            return float('-inf')


# Global optimization engine instance
# 全局优化引擎实例
optimization_engine = OptimizationEngine()

__all__ = [
    'OptimizationObjective',
    'OptimizationConstraint',
    'OptimizationAlgorithm',
    'OptimizationResult',
    'OptimizationEngine',
    'optimization_engine'
]
