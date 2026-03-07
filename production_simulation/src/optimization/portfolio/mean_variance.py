"""
Mean - Variance Portfolio Optimization Module
均方差投资组合优化模块

This module provides classical mean - variance portfolio optimization
此模块提供经典的均方差投资组合优化

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from ...core.constants import *

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer:

    """
    Mean - Variance Optimizer Class
    均方差优化器类

    Implements the classical Markowitz mean - variance portfolio optimization
    实现经典的Markowitz均方差投资组合优化

    The mean - variance optimization finds the optimal portfolio weights that
    maximize expected return for a given level of risk, or minimize risk
    for a given level of expected return.
    均方差优化找到在给定风险水平下最大化预期收益的最优投资组合权重，
    或者在给定预期收益水平下最小化风险。
    """

    def __init__(self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE):
        """
        Initialize mean - variance optimizer
        初始化均方差优化器

        Args:
            risk_free_rate: Risk - free rate for Sharpe ratio calculations
                           用于夏普比率计算的无风险利率
        """
        self.risk_free_rate = risk_free_rate
        self.expected_returns = None
        self.covariance_matrix = None
        self.asset_names = None

        logger.info("Mean - Variance optimizer initialized")

    def set_asset_data(self,

                       returns: pd.DataFrame,
                       covariance_matrix: Optional[np.ndarray] = None,
                       frequency: str = 'daily') -> None:
        """
        Set asset return data and compute parameters
        设置资产收益数据并计算参数

        Args:
            returns: Historical returns DataFrame
                    历史收益DataFrame
            covariance_matrix: Pre - computed covariance matrix (optional)
                             预先计算的协方差矩阵（可选）
            frequency: Return frequency ('daily', 'monthly', 'yearly')
                      收益频率（'daily', 'monthly', 'yearly'）
        """
        self.asset_names = returns.columns.tolist()

        # Calculate expected returns
        if frequency == 'daily':
            # Annualize daily returns
            self.expected_returns = returns.mean() * 252
        elif frequency == 'monthly':
            # Annualize monthly returns
            self.expected_returns = returns.mean() * 12
        elif frequency == 'yearly':
            self.expected_returns = returns.mean()
        else:
            self.expected_returns = returns.mean()

        # Calculate covariance matrix
        if covariance_matrix is not None:
            self.covariance_matrix = covariance_matrix
        else:
            if frequency == 'daily':
                # Annualize daily covariance
                self.covariance_matrix = returns.cov() * 252
            elif frequency == 'monthly':
                # Annualize monthly covariance
                self.covariance_matrix = returns.cov() * 12
            elif frequency == 'yearly':
                self.covariance_matrix = returns.cov()
            else:
                self.covariance_matrix = returns.cov()

        logger.info(f"Asset data set for {len(self.asset_names)} assets")

    def optimize_portfolio(self,

                           objective: str = 'sharpe',
                           target_return: Optional[float] = None,
                           target_risk: Optional[float] = None,
                           constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize portfolio using mean - variance framework
        使用均方差框架优化投资组合

        Args:
            objective: Optimization objective ('sharpe', 'min_risk', 'max_return', 'efficient')
                      优化目标（'sharpe', 'min_risk', 'max_return', 'efficient'）
            target_return: Target portfolio return (for efficient frontier)
                         目标投资组合收益（用于有效前沿）
            target_risk: Target portfolio risk (for efficient frontier)
                       目标投资组合风险（用于有效前沿）
            constraints: Additional constraints dictionary
                        其他约束字典

        Returns:
            dict: Optimization results
                  优化结果
        """
        try:
            # Validate inputs
            if self.expected_returns is None or self.covariance_matrix is None:
                raise ValueError("Asset data not set. Call set_asset_data() first.")

            # Set default constraints
            if constraints is None:
                constraints = {}

            # Apply optimization based on objective
            if objective == 'sharpe':
                result = self._maximize_sharpe_ratio(constraints)
            elif objective == 'min_risk':
                result = self._minimize_risk(constraints)
            elif objective == 'max_return':
                result = self._maximize_return(constraints)
            elif objective == 'efficient':
                if target_return is not None:
                    result = self._efficient_portfolio_return(target_return, constraints)
                elif target_risk is not None:
                    result = self._efficient_portfolio_risk(target_risk, constraints)
                else:
                    result = self._efficient_portfolio_return(
                        target_return or self.expected_returns.mean(), constraints)
            else:
                raise ValueError(f"Unknown objective: {objective}")

            # Calculate additional metrics
            result.update(self._calculate_portfolio_metrics(result['weights']))

            result['objective'] = objective
            result['timestamp'] = datetime.now()

            logger.info(f"Mean - variance optimization completed: {objective}")
            return result

        except Exception as e:
            logger.error(f"Mean - variance optimization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }

    def _maximize_sharpe_ratio(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maximize Sharpe ratio
        最大化夏普比率

        Args:
            constraints: Constraints dictionary
                        约束字典

        Returns:
            dict: Optimization result
                  优化结果
        """
        n_assets = len(self.expected_returns)

        # Objective function: maximize Sharpe ratio = (return - rf) / risk

        def objective(weights):

            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / \
                portfolio_risk if portfolio_risk > 0 else 0
            return -sharpe_ratio  # Minimize negative Sharpe ratio

        # Constraints
        opt_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Add user constraints
        opt_constraints.extend(self._build_constraints(constraints))

        # Bounds
        bounds = self._build_bounds(constraints)

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints
        )

        return {
            'success': result.success,
            'weights': result.x,
            'message': result.message,
            'fun_value': -result.fun  # Negate back to positive Sharpe ratio
        }

    def _minimize_risk(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimize portfolio risk (variance)
        最小化投资组合风险（方差）

        Args:
            constraints: Constraints dictionary
                        约束字典

        Returns:
            dict: Optimization result
                  优化结果
        """
        n_assets = len(self.expected_returns)

        # Objective function: minimize variance

        def objective(weights):

            return np.dot(weights.T, np.dot(self.covariance_matrix, weights))

        # Constraints
        opt_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Add minimum return constraint if specified
        if 'min_return' in constraints:
            min_return = constraints['min_return']
            opt_constraints.append({
                'type': 'ineq',
                'fun': lambda w: np.dot(w, self.expected_returns) - min_return
            })

        # Add user constraints
        opt_constraints.extend(self._build_constraints(constraints))

        # Bounds
        bounds = self._build_bounds(constraints)

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints
        )

        return {
            'success': result.success,
            'weights': result.x,
            'message': result.message,
            'fun_value': result.fun
        }

    def _maximize_return(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maximize portfolio return
        最大化投资组合收益

        Args:
            constraints: Constraints dictionary
                        约束字典

        Returns:
            dict: Optimization result
                  优化结果
        """
        n_assets = len(self.expected_returns)

        # Objective function: maximize return

        def objective(weights):

            return -np.dot(weights, self.expected_returns)  # Minimize negative return

        # Constraints
        opt_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Add maximum risk constraint if specified
        if 'max_risk' in constraints:
            max_risk = constraints['max_risk']
            opt_constraints.append({
                'type': 'ineq',
                'fun': lambda w: max_risk - np.sqrt(np.dot(w.T, np.dot(self.covariance_matrix, w)))
            })

        # Add user constraints
        opt_constraints.extend(self._build_constraints(constraints))

        # Bounds
        bounds = self._build_bounds(constraints)

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints
        )

        return {
            'success': result.success,
            'weights': result.x,
            'message': result.message,
            'fun_value': -result.fun  # Negate back to positive return
        }

    def _efficient_portfolio_return(self, target_return: float, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find minimum risk portfolio for given return
        为给定收益找到最小风险投资组合

        Args:
            target_return: Target portfolio return
                          目标投资组合收益
            constraints: Constraints dictionary
                        约束字典

        Returns:
            dict: Optimization result
                  优化结果
        """
        n_assets = len(self.expected_returns)

        # Objective function: minimize variance

        def objective(weights):

            return np.dot(weights.T, np.dot(self.covariance_matrix, weights))

        # Constraints
        opt_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.dot(
                w, self.expected_returns) - target_return},  # Target return
        ]

        # Add user constraints
        opt_constraints.extend(self._build_constraints(constraints))

        # Bounds
        bounds = self._build_bounds(constraints)

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints
        )

        return {
            'success': result.success,
            'weights': result.x,
            'message': result.message,
            'target_return': target_return,
            'fun_value': result.fun
        }

    def _efficient_portfolio_risk(self, target_risk: float, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find maximum return portfolio for given risk
        为给定风险找到最大收益投资组合

        Args:
            target_risk: Target portfolio risk (standard deviation)
                        目标投资组合风险（标准差）
            constraints: Constraints dictionary
                        约束字典

        Returns:
            dict: Optimization result
                  优化结果
        """
        n_assets = len(self.expected_returns)

        # Objective function: maximize return

        def objective(weights):

            return -np.dot(weights, self.expected_returns)  # Minimize negative return

        # Constraints
        opt_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.sqrt(
                np.dot(w.T, np.dot(self.covariance_matrix, w))) - target_risk},  # Target risk
        ]

        # Add user constraints
        opt_constraints.extend(self._build_constraints(constraints))

        # Bounds
        bounds = self._build_bounds(constraints)

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints
        )

        return {
            'success': result.success,
            'weights': result.x,
            'message': result.message,
            'target_risk': target_risk,
            'fun_value': -result.fun  # Negate back to positive return
        }

    def _build_constraints(self, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build optimization constraints from dictionary
        从字典构建优化约束

        Args:
            constraints: Constraints dictionary
                        约束字典

        Returns:
            list: List of constraint dictionaries
                  约束字典列表
        """
        opt_constraints = []

        # No short selling constraint
        if constraints.get('no_short_selling', False):
            opt_constraints.append({
                'type': 'ineq',
                'fun': lambda w: w  # w >= 0
            })

        # Sector constraints
        if 'sector_constraints' in constraints:
            # This would require sector mapping - simplified implementation
            pass

        # Turnover constraint
        if 'max_turnover' in constraints:
            # This would require previous weights - simplified implementation
            pass

        return opt_constraints

    def _build_bounds(self, constraints: Dict[str, Any]) -> List[tuple]:
        """
        Build optimization bounds
        构建优化边界

        Args:
            constraints: Constraints dictionary
                        约束字典

        Returns:
            list: List of bound tuples
                  边界元组列表
        """
        n_assets = len(self.expected_returns)

        # Default bounds: 0 to 1 (no short selling, full investment)
        if constraints.get('no_short_selling', True):
            bounds = [(0, 1) for _ in range(n_assets)]
        else:
            bounds = [(-1, 1) for _ in range(n_assets)]  # Allow short selling

        # Apply individual asset bounds
        if 'asset_bounds' in constraints:
            asset_bounds = constraints['asset_bounds']
        for i, bound in enumerate(asset_bounds):
            if i < len(bounds):
                bounds[i] = bound

        # Apply maximum weight constraint
        if 'max_weight' in constraints:
            max_weight = constraints['max_weight']
            bounds = [(max(bound[0], -max_weight), min(bound[1], max_weight)) for bound in bounds]

        return bounds

    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics
        计算投资组合绩效指标

        Args:
            weights: Portfolio weights
                    投资组合权重

        Returns:
            dict: Portfolio metrics
                  投资组合指标
        """
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)

        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / \
            portfolio_risk if portfolio_risk > 0 else 0

        # Calculate diversification metrics
        asset_volatilities = np.sqrt(np.diag(self.covariance_matrix))
        weighted_volatility = np.dot(weights, asset_volatilities)
        diversification_ratio = weighted_volatility / portfolio_risk if portfolio_risk > 0 else 0

        # Calculate risk contributions
        marginal_risk_contributions = self.covariance_matrix @ weights / portfolio_risk
        risk_contributions = weights * marginal_risk_contributions

        return {
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio,
            'risk_contributions': risk_contributions,
            'asset_volatilities': asset_volatilities,
            'expected_returns': self.expected_returns
        }

    def plot_efficient_frontier(self,


                                n_portfolios: int = 100,
                                save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot the efficient frontier
        绘制有效前沿

        Args:
            n_portfolios: Number of portfolios to generate
                         要生成的投资组合数量
            save_path: Path to save the plot (optional)
                      保存图表的路径（可选）

        Returns:
            dict: Efficient frontier data
                  有效前沿数据
        """
        try:
            n_assets = len(self.expected_returns)

            # Generate random portfolios
            np.random.seed(42)
            weights = np.secrets.random((n_portfolios, n_assets))
            weights = weights / np.sum(weights, axis=1, keepdims=True)  # Normalize

            # Calculate portfolio metrics
            portfolio_returns = weights @ self.expected_returns
            portfolio_risks = np.sqrt(np.sum(weights * (weights @ self.covariance_matrix), axis=1))

            # Find efficient frontier
            target_returns = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 50)

            efficient_returns = []
            efficient_risks = []

            for target_return in target_returns:
                try:
                    result = self._efficient_portfolio_return(target_return, {})
                    if result['success']:
                        efficient_returns.append(np.dot(result['weights'], self.expected_returns))
                        efficient_risks.append(np.sqrt(np.dot(result['weights'].T,
                                                              np.dot(self.covariance_matrix, result['weights']))))
                except Exception:
                    continue

            # Plot if matplotlib is available
            try:
                plt.figure(figsize=(10, 6))
                plt.scatter(portfolio_risks, portfolio_returns, c='blue',
                            marker='o', alpha=0.3, label='Random Portfolios')
                plt.plot(efficient_risks, efficient_returns, 'r-',
                         linewidth=2, label='Efficient Frontier')
                plt.xlabel('Portfolio Risk (Standard Deviation)')
                plt.ylabel('Portfolio Return')
                plt.title('Efficient Frontier')
                plt.legend()
                plt.grid(True, alpha=0.3)

                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Efficient frontier plot saved to {save_path}")
                else:
                    plt.show()

            except ImportError:
                logger.warning("Matplotlib not available for plotting")

            return {
                'random_portfolios': {
                    'returns': portfolio_returns,
                    'risks': portfolio_risks,
                    'weights': weights
                },
                'efficient_frontier': {
                    'returns': efficient_returns,
                    'risks': efficient_risks
                },
                'target_returns': target_returns
            }

        except Exception as e:
            logger.error(f"Efficient frontier plotting failed: {str(e)}")
            return {'error': str(e)}

    def get_portfolio_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio statistics
        获取全面的投资组合统计信息

        Returns:
            dict: Portfolio statistics
                  投资组合统计信息
        """
        try:
            n_assets = len(self.expected_returns)

            # Individual asset statistics
            asset_stats = {
                'expected_returns': self.expected_returns,
                'volatilities': np.sqrt(np.diag(self.covariance_matrix)),
                'sharpe_ratios': (self.expected_returns - self.risk_free_rate) / np.sqrt(np.diag(self.covariance_matrix))
            }

            # Correlation matrix
            std_matrix = np.sqrt(np.diag(self.covariance_matrix))
            correlation_matrix = self.covariance_matrix / np.outer(std_matrix, std_matrix)

            # Portfolio possibilities
            equal_weights = np.ones(n_assets) / n_assets
            equal_weight_stats = self._calculate_portfolio_metrics(equal_weights)

            return {
                'n_assets': n_assets,
                'asset_names': self.asset_names,
                'asset_statistics': asset_stats,
                'correlation_matrix': correlation_matrix,
                'equal_weight_portfolio': equal_weight_stats,
                'market_statistics': {
                    'average_return': np.mean(self.expected_returns),
                    'average_volatility': np.mean(np.sqrt(np.diag(self.covariance_matrix))),
                    'maximum_return': np.max(self.expected_returns),
                    'minimum_volatility': np.min(np.sqrt(np.diag(self.covariance_matrix)))
                },
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Portfolio statistics calculation failed: {str(e)}")
            return {'error': str(e)}

    def validate_portfolio(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        Validate portfolio weights and constraints
        验证投资组合权重和约束

        Args:
            weights: Portfolio weights to validate
                    要验证的投资组合权重

        Returns:
            dict: Validation results
                  验证结果
        """
        try:
            issues = []

            # Check weight sum
            weight_sum = np.sum(weights)
            if not np.isclose(weight_sum, 1.0, atol=1e-6):
                issues.append(f"Weights sum to {weight_sum:.6f}, should sum to 1.0")

            # Check for negative weights (if no short selling)
            negative_weights = np.sum(weights < 0)
            if negative_weights > 0:
                issues.append(f"{negative_weights} negative weights found")

            # Check for weights outside bounds
            out_of_bounds = np.sum((weights < 0) | (weights > 1))
            if out_of_bounds > 0:
                issues.append(f"{out_of_bounds} weights outside [0,1] bounds")

            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(weights)

            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'metrics': metrics,
                'weights': weights,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Portfolio validation failed: {str(e)}")
            return {
                'valid': False,
                'error': str(e),
                'weights': weights,
                'timestamp': datetime.now()
            }


# Global mean - variance optimizer instance
# 全局均方差优化器实例
mean_variance_optimizer = MeanVarianceOptimizer()

__all__ = [
    'MeanVarianceOptimizer',
    'mean_variance_optimizer'
]
