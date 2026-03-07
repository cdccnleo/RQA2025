#!/usr/bin/env python3
"""
Optimization Interfaces Module
优化接口模块

定义优化算法的接口和数据结构
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np


class OptimizationAlgorithm(Enum):
    """优化算法枚举"""

    SLSQP = "SLSQP"  # Sequential Least Squares Programming
    COBYLA = "COBYLA"  # Constrained Optimization BY Linear Approximation
    BFGS = "BFGS"  # Broyden-Fletcher-Goldfarb-Shanno
    LBFGS = "L-BFGS"  # Limited-memory BFGS
    TNC = "TNC"  # Truncated Newton algorithm
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    TRUST_REGION = "trust-region"
    GENETIC = "genetic"
    PARTICLE_SWARM = "particle_swarm"
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"


class OptimizationResult:
    """优化结果类"""

    def __init__(self,
                 success: bool,
                 x: np.ndarray,
                 fun: float,
                 message: str = "",
                 nfev: int = 0,
                 nit: int = 0):
        self.success = success
        self.x = x  # 优化参数
        self.fun = fun  # 目标函数值
        self.message = message
        self.nfev = nfev  # 函数评估次数
        self.nit = nit  # 迭代次数

    def __str__(self):
        return f"OptimizationResult(success={self.success}, fun={self.fun:.6f})"


class IParameterOptimizer(ABC):
    """参数优化器接口"""

    @abstractmethod
    def optimize(self,
                 objective_func: callable,
                 initial_params: np.ndarray,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 constraints: Optional[List[Dict]] = None,
                 algorithm: OptimizationAlgorithm = OptimizationAlgorithm.SLSQP,
                 max_iter: int = 1000,
                 tolerance: float = 1e-6) -> OptimizationResult:
        """
        执行参数优化

        Args:
            objective_func: 目标函数
            initial_params: 初始参数
            bounds: 参数边界
            constraints: 约束条件
            algorithm: 优化算法
            max_iter: 最大迭代次数
            tolerance: 收敛容差

        Returns:
            优化结果
        """

    @abstractmethod
    def get_algorithm_info(self, algorithm: OptimizationAlgorithm) -> Dict[str, Any]:
        """
        获取算法信息

        Args:
            algorithm: 优化算法

        Returns:
            算法信息字典
        """

    @abstractmethod
    def validate_parameters(self,
                            params: np.ndarray,
                            bounds: Optional[List[Tuple[float, float]]] = None) -> bool:
        """
        验证参数是否在有效范围内

        Args:
            params: 参数数组
            bounds: 参数边界

        Returns:
            是否有效
        """
