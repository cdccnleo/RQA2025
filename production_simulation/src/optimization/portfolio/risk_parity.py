"""
Risk Parity Optimization Module
风险平价优化模块

This module provides risk parity portfolio optimization functionality
此模块提供风险平价投资组合优化功能

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.optimize as sco

logger = logging.getLogger(__name__)


class RiskParityOptimizer:

    """
    Risk Parity Optimizer Class
    风险平价优化器类

    Implements risk parity portfolio optimization where each asset contributes
    equally to the total portfolio risk
    实现风险平价投资组合优化，其中每个资产对总投资组合风险的贡献相等

    Risk parity aims to allocate capital so that each asset contributes equally
    to the overall portfolio volatility, rather than allocating by expected returns
    风险平价旨在分配资本，使得每个资产对整体投资组合波动率的贡献相等，而不是按预期收益分配
    """

    def __init__(self, name: str = "risk_parity_optimizer"):
        """
        Initialize the risk parity optimizer
        初始化风险平价优化器

        Args:
            name: Name of the optimizer
                优化器的名称
        """
        self.name = name
        self.optimization_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'average_execution_time': 0.0,
            'last_run_timestamp': None
        }

        logger.info(f"Risk parity optimizer {name} initialized")

    def optimize_portfolio(self,


                           returns: pd.DataFrame,
                           target_risk_contribution: Optional[np.ndarray] = None,
                           initial_weights: Optional[np.ndarray] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Optimize portfolio using risk parity approach
        使用风险平价方法优化投资组合

        Args:
            returns: Historical returns data (n_periods x n_assets)
                    历史收益数据（n_periods x n_assets）
            target_risk_contribution: Target risk contribution for each asset
                                    每个资产的目标风险贡献
            initial_weights: Initial portfolio weights
                           初始投资组合权重
            **kwargs: Additional optimization parameters
                     其他优化参数

        Returns:
            dict: Optimization results containing weights and risk metrics
                  包含权重和风险指标的优化结果
        """
        start_time = datetime.now()

        try:
            # Validate inputs
            if returns.empty:
                raise ValueError("Returns data cannot be empty")

            n_assets = len(returns.columns)

            # Calculate covariance matrix
            cov_matrix = returns.cov().values

            # Set target risk contribution
            if target_risk_contribution is None:
                target_risk_contribution = np.ones(n_assets) / n_assets
            else:
                target_risk_contribution = np.array(target_risk_contribution)
            if len(target_risk_contribution) != n_assets:
                raise ValueError("Target risk contribution length must match number of assets")

            # Set initial weights
            if initial_weights is None:
                initial_weights = np.ones(n_assets) / n_assets
            else:
                initial_weights = np.array(initial_weights)
            if len(initial_weights) != n_assets:
                raise ValueError("Initial weights length must match number of assets")

            # Run risk parity optimization
            optimal_weights = self._risk_parity_optimization(
                cov_matrix, target_risk_contribution, initial_weights, **kwargs
            )

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(optimal_weights, cov_matrix, returns)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Update statistics
            self._update_stats(True, execution_time)

            result = {
                'success': True,
                'optimal_weights': optimal_weights,
                'risk_contributions': risk_metrics['risk_contributions'],
                'portfolio_volatility': risk_metrics['portfolio_volatility'],
                'diversification_ratio': risk_metrics['diversification_ratio'],
                'asset_volatilities': risk_metrics['asset_volatilities'],
                'correlation_matrix': risk_metrics['correlation_matrix'],
                'execution_time': execution_time,
                'timestamp': datetime.now(),
                'target_risk_contribution': target_risk_contribution
            }

            logger.info(f"Risk parity optimization completed successfully for {n_assets} assets")
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Risk parity optimization failed: {str(e)}")

            self._update_stats(False, execution_time)

            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'timestamp': datetime.now()
            }

    def _risk_parity_optimization(self,


                                  cov_matrix: np.ndarray,
                                  target_risk_contribution: np.ndarray,
                                  initial_weights: np.ndarray,
                                  max_iter: int = 1000,
                                  tolerance: float = 1e-8) -> np.ndarray:
        """
        Core risk parity optimization algorithm
        核心风险平价优化算法

        Args:
            cov_matrix: Covariance matrix of asset returns
                       资产收益的协方差矩阵
            target_risk_contribution: Target risk contribution vector
                                    目标风险贡献向量
            initial_weights: Initial portfolio weights
                           初始投资组合权重
            max_iter: Maximum number of iterations
                     最大迭代次数
            tolerance: Convergence tolerance
                      收敛容差

        Returns:
            np.ndarray: Optimal portfolio weights
                       最优投资组合权重
        """
        weights = initial_weights.copy()

        for iteration in range(max_iter):
            # Calculate portfolio volatility
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)

            # Calculate marginal risk contributions
            marginal_risk_contributions = cov_matrix @ weights / portfolio_volatility

            # Calculate current risk contributions
            risk_contributions = weights * marginal_risk_contributions

            # Calculate difference from target
            risk_difference = risk_contributions - target_risk_contribution * portfolio_volatility

            # Check convergence
            if np.max(np.abs(risk_difference)) < tolerance:
                logger.debug(f"Risk parity optimization converged after {iteration + 1} iterations")
                break

            # Update weights using gradient descent
            gradient = marginal_risk_contributions
            learning_rate = 0.01

            # Project gradient onto simplex (weights sum to 1, non - negative)
            weights = self._project_to_simplex(weights - learning_rate * gradient)

        # Ensure weights sum to 1
        weights = weights / np.sum(weights)

        return weights

    def _project_to_simplex(self, v: np.ndarray, z: float = 1.0) -> np.ndarray:
        """
        Project vector onto simplex (non - negative, sum to z)
        将向量投影到单纯形（非负，和为z）

        Args:
            v: Input vector
               输入向量
            z: Target sum
               目标和

        Returns:
            np.ndarray: Projected vector
                       投影向量
        """
        n = len(v)
        u = np.sort(v)[::-1]  # Sort in descending order
        cssv = np.cumsum(u) - z
        ind = np.arange(n) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1] if np.any(cond) else n
        theta = cssv[cond][-1] / rho if np.any(cond) else 0
        w = np.maximum(v - theta, 0)
        return w

    def _calculate_risk_metrics(self,


                                weights: np.ndarray,
                                cov_matrix: np.ndarray,
                                returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for the portfolio
        为投资组合计算全面的风险指标

        Args:
            weights: Portfolio weights
                    投资组合权重
            cov_matrix: Covariance matrix
                       协方差矩阵
            returns: Historical returns data
                    历史收益数据

        Returns:
            dict: Risk metrics
                  风险指标
        """
        try:
            # Portfolio volatility
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)

            # Individual asset volatilities
            asset_volatilities = np.sqrt(np.diag(cov_matrix))

            # Risk contributions
            marginal_risk_contributions = cov_matrix @ weights / portfolio_volatility
            risk_contributions = weights * marginal_risk_contributions

            # Diversification ratio (Choueifaty & Coignard)
            weighted_volatility_sum = weights @ asset_volatilities
            diversification_ratio = weighted_volatility_sum / portfolio_volatility

            # Correlation matrix
            std_matrix = np.sqrt(np.diag(cov_matrix))
            correlation_matrix = cov_matrix / np.outer(std_matrix, std_matrix)

            return {
                'portfolio_volatility': portfolio_volatility,
                'asset_volatilities': asset_volatilities,
                'risk_contributions': risk_contributions,
                'marginal_risk_contributions': marginal_risk_contributions,
                'diversification_ratio': diversification_ratio,
                'correlation_matrix': correlation_matrix,
                'risk_contribution_percentage': risk_contributions / portfolio_volatility
            }

        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {str(e)}")
            return {}

    def optimize_with_risk_budgeting(self,


                                     returns: pd.DataFrame,
                                     risk_budgets: np.ndarray,
                                     **kwargs) -> Dict[str, Any]:
        """
        Optimize portfolio with specific risk budgets for each asset
        为每个资产优化具有特定风险预算的投资组合

        Args:
            returns: Historical returns data
                    历史收益数据
            risk_budgets: Risk budget for each asset (should sum to 1)
                         每个资产的风险预算（应总和为1）
            **kwargs: Additional optimization parameters
                     其他优化参数

        Returns:
            dict: Optimization results
                  优化结果
        """
        try:
            # Validate risk budgets
            risk_budgets = np.array(risk_budgets)
            if not np.isclose(np.sum(risk_budgets), 1.0):
                raise ValueError("Risk budgets must sum to 1")

            cov_matrix = returns.cov().values
            portfolio_volatility = np.sqrt(np.sum(cov_matrix))

            # Convert risk budgets to target risk contributions
            target_risk_contribution = risk_budgets * portfolio_volatility

            return self.optimize_portfolio(
                returns,
                target_risk_contribution=target_risk_contribution,
                **kwargs
            )

        except Exception as e:
            logger.error(f"Risk budgeting optimization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }

    def equal_risk_contribution_portfolio(self,


                                          returns: pd.DataFrame,
                                          **kwargs) -> Dict[str, Any]:
        """
        Create an Equal Risk Contribution (ERC) portfolio
        创建等风险贡献(ERC)投资组合

        Args:
            returns: Historical returns data
                    历史收益数据
            **kwargs: Additional optimization parameters
                     其他优化参数

        Returns:
            dict: ERC portfolio results
                  ERC投资组合结果
        """
        n_assets = len(returns.columns)
        target_risk_contribution = np.ones(n_assets) / n_assets

        return self.optimize_portfolio(
            returns,
            target_risk_contribution=target_risk_contribution,
            **kwargs
        )

    def maximum_diversification_portfolio(self,


                                          returns: pd.DataFrame,
                                          **kwargs) -> Dict[str, Any]:
        """
        Create a Maximum Diversification portfolio
        创建最大分散化投资组合

        Args:
            returns: Historical returns data
                    历史收益数据
            **kwargs: Additional optimization parameters
                     其他优化参数

        Returns:
            dict: Maximum diversification portfolio results
                  最大分散化投资组合结果
        """
        try:
            cov_matrix = returns.cov().values
            asset_volatilities = np.sqrt(np.diag(cov_matrix))

            # Calculate diversification ratio for different weight combinations
            n_assets = len(returns.columns)

            def diversification_objective(weights):

                portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
                weighted_volatility_sum = weights @ asset_volatilities
                diversification_ratio = weighted_volatility_sum / portfolio_volatility
                return -diversification_ratio  # Maximize diversification ratio

            # Constraints: weights sum to 1, non - negative
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
            bounds = [(0, 1) for _ in range(n_assets)]

            # Optimize
            initial_weights = np.ones(n_assets) / n_assets
            result = sco.minimize(
                diversification_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                optimal_weights = result.x
                risk_metrics = self._calculate_risk_metrics(optimal_weights, cov_matrix, returns)

                return {
                    'success': True,
                    'optimal_weights': optimal_weights,
                    'diversification_ratio': -result.fun,  # Negate back to positive
                    'portfolio_volatility': risk_metrics['portfolio_volatility'],
                    'risk_contributions': risk_metrics['risk_contributions'],
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'success': False,
                    'error': 'Maximum diversification optimization failed to converge',
                    'timestamp': datetime.now()
                }

        except Exception as e:
            logger.error(f"Maximum diversification optimization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }

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

    def compare_portfolios(self,


                           returns: pd.DataFrame,
                           portfolios: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compare multiple portfolios based on risk metrics
        基于风险指标比较多个投资组合

        Args:
            returns: Historical returns data
                    历史收益数据
            portfolios: Dictionary of portfolio names to weights
                       投资组合名称到权重的字典

        Returns:
            dict: Comparison results
                  比较结果
        """
        try:
            cov_matrix = returns.cov().values
            comparison_results = {}

            for name, weights in portfolios.items():
                risk_metrics = self._calculate_risk_metrics(weights, cov_matrix, returns)

                comparison_results[name] = {
                    'weights': weights,
                    'portfolio_volatility': risk_metrics['portfolio_volatility'],
                    'diversification_ratio': risk_metrics['diversification_ratio'],
                    'risk_contributions': risk_metrics['risk_contributions'],
                    'max_risk_contribution': np.max(risk_metrics['risk_contributions']),
                    'min_risk_contribution': np.min(risk_metrics['risk_contributions']),
                    'risk_concentration': np.max(risk_metrics['risk_contributions']) / np.sum(risk_metrics['risk_contributions'])
                }

            return {
                'success': True,
                'comparison_results': comparison_results,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Portfolio comparison failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }


# Global risk parity optimizer instance
# 全局风险平价优化器实例
risk_parity_optimizer = RiskParityOptimizer()

__all__ = [
    'RiskParityOptimizer',
    'risk_parity_optimizer'
]
