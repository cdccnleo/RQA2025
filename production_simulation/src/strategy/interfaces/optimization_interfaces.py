#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略优化服务接口定义
Strategy Optimization Service Interfaces

定义统一的策略优化接口，支持参数优化、步进优化等多种优化方法。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationAlgorithm(Enum):

    """优化算法枚举"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"


class OptimizationTarget(Enum):

    """优化目标枚举"""
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"


@dataclass
class OptimizationConfig:

    """优化配置"""
    optimization_id: str
    strategy_id: str
    algorithm: OptimizationAlgorithm
    target: OptimizationTarget
    parameter_ranges: Dict[str, List[Any]]
    constraints: Dict[str, Any]
    max_iterations: int = 100
    early_stopping: bool = True
    early_stopping_patience: int = 10
    parallel_execution: bool = True
    n_jobs: int = -1
    created_at: datetime = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class OptimizationResult:

    """优化结果"""
    optimization_id: str
    strategy_id: str
    best_parameters: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    convergence_history: List[float]
    execution_time: float
    status: str
    timestamp: datetime = None
    error_message: Optional[str] = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class WalkForwardConfig:

    """步进优化配置"""
    optimization_id: str
    strategy_id: str
    train_window: int  # 训练窗口大小（天数）
    test_window: int   # 测试窗口大小（天数）
    step_size: int     # 步进大小（天数）
    min_train_period: int = 252  # 最小的训练周期（天数）
    parameter_ranges: Dict[str, List[Any]] = None
    target: OptimizationTarget = OptimizationTarget.SHARPE_RATIO

    def __post_init__(self):

        if self.parameter_ranges is None:
            self.parameter_ranges = {}


@dataclass
class WalkForwardResult:

    """步进优化结果"""
    optimization_id: str
    strategy_id: str
    periods: List[Dict[str, Any]]  # 每个周期的优化结果
    overall_performance: Dict[str, float]
    robustness_score: float  # 稳健性评分
    execution_time: float
    timestamp: datetime = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now()


class IOptimizationService(ABC):

    """
    优化服务接口
    Optimization Service Interface

    定义策略优化的核心功能接口。
    """

    @abstractmethod
    def create_optimization(self, config: OptimizationConfig) -> str:
        """
        创建优化任务

        Args:
            config: 优化配置

        Returns:
            str: 优化任务ID
        """

    @abstractmethod
    def run_optimization(self, optimization_id: str) -> OptimizationResult:
        """
        运行优化

        Args:
            optimization_id: 优化ID

        Returns:
            OptimizationResult: 优化结果
        """

    @abstractmethod
    def get_optimization_result(self, optimization_id: str) -> Optional[OptimizationResult]:
        """
        获取优化结果

        Args:
            optimization_id: 优化ID

        Returns:
            Optional[OptimizationResult]: 优化结果
        """

    @abstractmethod
    def cancel_optimization(self, optimization_id: str) -> bool:
        """
        取消优化

        Args:
            optimization_id: 优化ID

        Returns:
            bool: 取消是否成功
        """

    @abstractmethod
    def list_optimizations(self, strategy_id: Optional[str] = None) -> List[OptimizationConfig]:
        """
        列出优化任务

        Args:
            strategy_id: 策略ID过滤器

        Returns:
            List[OptimizationConfig]: 优化配置列表
        """


class IParameterOptimizer(ABC):

    """
    参数优化器接口
    Parameter Optimizer Interface

    定义参数优化的具体实现接口。
    """

    @abstractmethod
    def optimize(self, strategy_id: str, parameter_ranges: Dict[str, List[Any]],


                 target_function: callable, algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GRID_SEARCH,
                 **kwargs) -> OptimizationResult:
        """
        执行参数优化

        Args:
            strategy_id: 策略ID
            parameter_ranges: 参数范围字典
            target_function: 目标函数
            algorithm: 优化算法
            **kwargs: 其他参数

        Returns:
            OptimizationResult: 优化结果
        """

    @abstractmethod
    def grid_search(self, parameter_ranges: Dict[str, List[Any]],


                    target_function: callable) -> OptimizationResult:
        """
        网格搜索优化

        Args:
            parameter_ranges: 参数范围字典
            target_function: 目标函数

        Returns:
            OptimizationResult: 优化结果
        """

    @abstractmethod
    def random_search(self, parameter_ranges: Dict[str, List[Any]],


                      target_function: callable, n_iterations: int = 100) -> OptimizationResult:
        """
        随机搜索优化

        Args:
            parameter_ranges: 参数范围字典
            target_function: 目标函数
            n_iterations: 迭代次数

        Returns:
            OptimizationResult: 优化结果
        """

    @abstractmethod
    def bayesian_optimization(self, parameter_ranges: Dict[str, List[Any]],


                              target_function: callable, n_iterations: int = 50) -> OptimizationResult:
        """
        贝叶斯优化

        Args:
            parameter_ranges: 参数范围字典
            target_function: 目标函数
            n_iterations: 迭代次数

        Returns:
            OptimizationResult: 优化结果
        """


class IWalkForwardOptimizer(ABC):

    """
    步进优化器接口
    Walk - Forward Optimizer Interface

    定义步进优化的具体实现接口。
    """

    @abstractmethod
    def walk_forward_optimization(self, config: WalkForwardConfig) -> WalkForwardResult:
        """
        执行步进优化

        Args:
            config: 步进优化配置

        Returns:
            WalkForwardResult: 步进优化结果
        """

    @abstractmethod
    def anchored_walk_forward(self, strategy_id: str, train_window: int,


                              test_window: int, step_size: int) -> WalkForwardResult:
        """
        锚定步进优化

        Args:
            strategy_id: 策略ID
            train_window: 训练窗口大小
            test_window: 测试窗口大小
            step_size: 步进大小

        Returns:
            WalkForwardResult: 步进优化结果
        """

    @abstractmethod
    def rolling_walk_forward(self, strategy_id: str, train_window: int,


                             test_window: int, step_size: int) -> WalkForwardResult:
        """
        滚动步进优化

        Args:
            strategy_id: 策略ID
            train_window: 训练窗口大小
            test_window: 测试窗口大小
            step_size: 步进大小

        Returns:
            WalkForwardResult: 步进优化结果
        """

    @abstractmethod
    def evaluate_robustness(self, walk_forward_result: WalkForwardResult) -> float:
        """
        评估优化稳健性

        Args:
            walk_forward_result: 步进优化结果

        Returns:
            float: 稳健性评分 (0 - 1)
        """


class IOptimizationPersistence(ABC):

    """
    优化持久化接口
    Optimization Persistence Interface

    处理优化结果和配置的持久化存储。
    """

    @abstractmethod
    def save_optimization_result(self, result: OptimizationResult) -> bool:
        """
        保存优化结果

        Args:
            result: 优化结果

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def load_optimization_result(self, optimization_id: str) -> Optional[OptimizationResult]:
        """
        加载优化结果

        Args:
            optimization_id: 优化ID

        Returns:
            Optional[OptimizationResult]: 优化结果
        """

    @abstractmethod
    def save_walk_forward_result(self, result: WalkForwardResult) -> bool:
        """
        保存步进优化结果

        Args:
            result: 步进优化结果

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def load_walk_forward_result(self, optimization_id: str) -> Optional[WalkForwardResult]:
        """
        加载步进优化结果

        Args:
            optimization_id: 优化ID

        Returns:
            Optional[WalkForwardResult]: 步进优化结果
        """

    @abstractmethod
    def delete_optimization_data(self, optimization_id: str) -> bool:
        """
        删除优化数据

        Args:
            optimization_id: 优化ID

        Returns:
            bool: 删除是否成功
        """


# 导出所有接口
__all__ = [
    'OptimizationAlgorithm',
    'OptimizationTarget',
    'OptimizationConfig',
    'OptimizationResult',
    'WalkForwardConfig',
    'WalkForwardResult',
    'IOptimizationService',
    'IParameterOptimizer',
    'IWalkForwardOptimizer',
    'IOptimizationPersistence'
]
