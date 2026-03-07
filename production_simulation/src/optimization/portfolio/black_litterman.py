"""
Black - Litterman Portfolio Optimization Module
Black - Litterman投资组合优化模块

This module provides Black - Litterman portfolio optimization implementation
此模块提供Black - Litterman投资组合优化实现

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class BlackLittermanOptimizer:

    """
    Black - Litterman Optimizer Class
    Black - Litterman优化器类

    Implements the Black - Litterman portfolio optimization model
    实现Black - Litterman投资组合优化模型

    The Black - Litterman model combines market equilibrium with investor views
    to produce improved portfolio weights that reflect both market consensus
    and investor beliefs.
    Black - Litterman模型结合市场均衡与投资者观点，
    以产生反映市场共识和投资者信念的改进投资组合权重。
    """

    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        """
        Initialize Black - Litterman optimizer
        初始化Black - Litterman优化器

        Args:
            risk_aversion: Investor's risk aversion coefficient
                          投资者的风险厌恶系数
            tau: Uncertainty in the prior estimate (typically 0.05)
                先验估计的不确定性（通常为0.05）
        """
        self.risk_aversion = risk_aversion
        self.tau = tau

        # Market parameters
        self.market_cap_weights = None
        self.market_returns = None
        self.covariance_matrix = None

        # Investor views
        self.views = []
        self.view_confidences = []

        logger.info("Black - Litterman optimizer initialized")

    def set_market_parameters(self,

                              market_cap_weights: np.ndarray,
                              expected_returns: np.ndarray,
                              covariance_matrix: np.ndarray) -> None:
        """
        Set market equilibrium parameters
        设置市场均衡参数

        Args:
            market_cap_weights: Market capitalization weights
                               市值权重
            expected_returns: Expected returns vector
                             预期收益向量
            covariance_matrix: Asset covariance matrix
                             资产协方差矩阵
        """
        self.market_cap_weights = np.array(market_cap_weights)
        self.market_returns = np.array(expected_returns)
        self.covariance_matrix = np.array(covariance_matrix)

        # Validate dimensions
        n_assets = len(market_cap_weights)
        assert len(expected_returns) == n_assets, "Expected returns dimension mismatch"
        assert covariance_matrix.shape == (
            n_assets, n_assets), "Covariance matrix dimension mismatch"

        logger.info(f"Market parameters set for {n_assets} assets")

    def add_view(self,


                 view_vector: np.ndarray,
                 expected_return: float,
                 confidence: float) -> None:
        """
        Add an investor view
        添加投资者观点

        Args:
            view_vector: View on asset returns (e.g., [0, 0, 1, -1, 0] for asset 3 > asset 4)
                        对资产收益的观点（例如[0, 0, 1, -1, 0]表示资产3 > 资产4）
            expected_return: Expected return for this view
                           此观点的预期收益
            confidence: Confidence in this view (0 - 1)
                      对此观点的信心（0 - 1）
        """
        self.views.append({
            'vector': np.array(view_vector),
            'return': expected_return,
            'confidence': confidence
        })

        logger.info(f"Added view with confidence {confidence}")

    def clear_views(self) -> None:
        """
        Clear all investor views
        清除所有投资者观点
        """
        self.views.clear()
        logger.info("Cleared all investor views")

    def optimize_portfolio(self,

                           target_return: Optional[float] = None,
                           risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Optimize portfolio using Black - Litterman model
        使用Black - Litterman模型优化投资组合

        Args:
            target_return: Target portfolio return (optional)
                         目标投资组合收益（可选）
            risk_free_rate: Risk - free rate
                           无风险利率

        Returns:
            dict: Optimization results
                  优化结果
        """
        try:
            # Validate inputs
            if self.market_cap_weights is None:
                raise ValueError("Market parameters not set")

            if not self.views:
                logger.warning("No investor views provided, falling back to market equilibrium")
                return self._market_equilibrium_portfolio()

            # Compute posterior distribution
            posterior_returns, posterior_covariance = self._compute_posterior_distribution()

            # Optimize portfolio
            optimal_weights = self._optimize_portfolio_weights(
                posterior_returns, posterior_covariance, target_return, risk_free_rate
            )

            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, posterior_returns)
            portfolio_risk = np.sqrt(
                np.dot(optimal_weights.T, np.dot(posterior_covariance, optimal_weights)))

            # Calculate view impact
            view_impact = self._calculate_view_impact(optimal_weights)

            result = {
                'success': True,
                'optimal_weights': optimal_weights,
                'portfolio_return': portfolio_return,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0,
                'posterior_returns': posterior_returns,
                'posterior_covariance': posterior_covariance,
                'view_impact': view_impact,
                'market_weights': self.market_cap_weights,
                'views_count': len(self.views),
                'timestamp': datetime.now()
            }

            logger.info("Black - Litterman optimization completed successfully")
            return result

        except Exception as e:
            logger.error(f"Black - Litterman optimization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }

    def _compute_posterior_distribution(self) -> tuple:
        """
        Compute posterior expected returns and covariance matrix
        计算后验预期收益和协方差矩阵

        Returns:
            tuple: (posterior_returns, posterior_covariance)
                  (后验收益，后验协方差)
        """
        # Prior distribution
        prior_returns = self.market_returns
        prior_covariance = self.tau * self.covariance_matrix

        if not self.views:
            return prior_returns, prior_covariance

        # Construct views matrix and vector
        n_assets = len(self.market_cap_weights)
        n_views = len(self.views)

        P = np.zeros((n_views, n_assets))  # View matrix
        Q = np.zeros(n_views)             # View returns
        Omega = np.zeros((n_views, n_views))  # View uncertainty

        for i, view in enumerate(self.views):
            P[i] = view['vector']
            Q[i] = view['return']

            # View uncertainty based on confidence
            view_variance = (1 - view['confidence']) / \
                view['confidence'] if view['confidence'] > 0 else 1.0
            Omega[i, i] = view_variance

        # Compute posterior parameters
        # posterior_returns = prior_returns + tau * Σ * P^T * (P * tau * Σ * P^T + Ω)^(-1) * (Q - P * prior_returns)
        # posterior_covariance = tau * Σ - tau * Σ * P^T * (P * tau * Σ * P^T + Ω)^(-1) * P * tau * Σ

        try:
            tau_sigma = prior_covariance
            P_tau_sigma = np.dot(P, tau_sigma)
            temp_matrix = np.dot(P_tau_sigma, P.T) + Omega

            # Check for matrix singularity
            if np.linalg.cond(temp_matrix) > 1e12:
                logger.warning("View matrix is near singular, using regularization")
                temp_matrix += np.eye(n_views) * 1e-6

            temp_inverse = np.linalg.inv(temp_matrix)
            view_adjustment = np.dot(P_tau_sigma.T, np.dot(
                temp_inverse, (Q - np.dot(P, prior_returns))))

            posterior_returns = prior_returns + view_adjustment
            posterior_covariance = tau_sigma - \
                np.dot(np.dot(P_tau_sigma.T, temp_inverse), P_tau_sigma)

            return posterior_returns, posterior_covariance

        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed in posterior computation: {str(e)}")
            # Fallback to prior
            return prior_returns, prior_covariance

    def _optimize_portfolio_weights(self,

                                    expected_returns: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    target_return: Optional[float],
                                    risk_free_rate: float) -> np.ndarray:
        """
        Optimize portfolio weights
        优化投资组合权重

        Args:
            expected_returns: Expected returns vector
                             预期收益向量
            covariance_matrix: Covariance matrix
                             协方差矩阵
            target_return: Target return
                         目标收益
            risk_free_rate: Risk - free rate
                           无风险利率

        Returns:
            np.ndarray: Optimal portfolio weights
                       最优投资组合权重
        """
        n_assets = len(expected_returns)

        # Objective function: minimize portfolio variance

        def objective(weights):

            return np.dot(weights.T, np.dot(covariance_matrix, weights))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - target_return
            })

        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                return result.x
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return initial_weights

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}")
            return initial_weights

    def _market_equilibrium_portfolio(self) -> Dict[str, Any]:
        """
        Return market equilibrium portfolio (no views)
        返回市场均衡投资组合（无观点）

        Returns:
            dict: Market equilibrium portfolio
                  市场均衡投资组合
        """
        portfolio_return = np.dot(self.market_cap_weights, self.market_returns)
        portfolio_risk = np.sqrt(np.dot(self.market_cap_weights.T,
                                        np.dot(self.covariance_matrix, self.market_cap_weights)))

        return {
            'success': True,
            'optimal_weights': self.market_cap_weights,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': (portfolio_return - 0.02) / portfolio_risk if portfolio_risk > 0 else 0,
            'note': 'No investor views provided, using market equilibrium',
            'timestamp': datetime.now()
        }

    def _calculate_view_impact(self, optimal_weights: np.ndarray) -> Dict[str, Any]:
        """
        Calculate the impact of investor views on portfolio weights
        计算投资者观点对投资组合权重的影响

        Args:
            optimal_weights: Optimal portfolio weights
                           最优投资组合权重

        Returns:
            dict: View impact analysis
                  观点影响分析
        """
        if not self.views or self.market_cap_weights is None:
            return {'error': 'Insufficient data for view impact analysis'}

        # Calculate weight differences
        weight_difference = optimal_weights - self.market_cap_weights

        # Calculate absolute impact
        absolute_impact = np.abs(weight_difference)

        # Find most impacted assets
        most_impacted_idx = np.argmax(absolute_impact)
        least_impacted_idx = np.argmin(absolute_impact)

        return {
            'weight_difference': weight_difference,
            'absolute_impact': absolute_impact,
            'max_impact_asset': most_impacted_idx,
            'max_impact_value': absolute_impact[most_impacted_idx],
            'min_impact_asset': least_impacted_idx,
            'min_impact_value': absolute_impact[least_impacted_idx],
            'average_impact': np.mean(absolute_impact),
            'impact_std': np.std(absolute_impact)
        }

    def sensitivity_analysis(self,

                             parameter_ranges: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on Black - Litterman parameters
        对Black - Litterman参数执行敏感性分析

        Args:
            parameter_ranges: Ranges for parameters to test
                             要测试的参数范围

        Returns:
            dict: Sensitivity analysis results
                  敏感性分析结果
        """
        results = {
            'parameter_sensitivity': {},
            'optimal_weights_range': {},
            'portfolio_return_range': {},
            'portfolio_risk_range': {},
            'timestamp': datetime.now()
        }

        # Default parameter values
        base_tau = self.tau
        base_risk_aversion = self.risk_aversion

        for param_name, param_values in parameter_ranges.items():
            param_results = []

            for param_value in param_values:
                # Set parameter
                if param_name == 'tau':
                    self.tau = param_value
                elif param_name == 'risk_aversion':
                    self.risk_aversion = param_value

                # Run optimization
                opt_result = self.optimize_portfolio()
                if opt_result['success']:
                    param_results.append({
                        'parameter_value': param_value,
                        'optimal_weights': opt_result['optimal_weights'],
                        'portfolio_return': opt_result['portfolio_return'],
                        'portfolio_risk': opt_result['portfolio_risk']
                    })

            results['parameter_sensitivity'][param_name] = param_results

            # Calculate ranges
            if param_results:
                weights_array = np.array([r['optimal_weights'] for r in param_results])
                returns_array = np.array([r['portfolio_return'] for r in param_results])
                risks_array = np.array([r['portfolio_risk'] for r in param_results])

                results['optimal_weights_range'][param_name] = {
                    'min': np.min(weights_array, axis=0),
                    'max': np.max(weights_array, axis=0),
                    'std': np.std(weights_array, axis=0)
                }

                results['portfolio_return_range'][param_name] = {
                    'min': np.min(returns_array),
                    'max': np.max(returns_array),
                    'std': np.std(returns_array)
                }

                results['portfolio_risk_range'][param_name] = {
                    'min': np.min(risks_array),
                    'max': np.max(risks_array),
                    'std': np.std(risks_array)
                }

        # Restore original parameters
        self.tau = base_tau
        self.risk_aversion = base_risk_aversion

        return results

    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics and validation metrics
        获取模型诊断和验证指标

        Returns:
            dict: Model diagnostics
                  模型诊断
        """
        diagnostics = {
            'views_summary': {
                'total_views': len(self.views),
                'average_confidence': np.mean([v['confidence'] for v in self.views]) if self.views else 0,
                'confidence_std': np.std([v['confidence'] for v in self.views]) if self.views else 0
            },
            'market_parameters': {
                'assets_count': len(self.market_cap_weights) if self.market_cap_weights is not None else 0,
                'tau_value': self.tau,
                'risk_aversion': self.risk_aversion
            },
            'timestamp': datetime.now()
        }

        if self.market_cap_weights is not None:
            diagnostics['market_parameters'].update({
                'market_weights_sum': np.sum(self.market_cap_weights),
                'market_weights_std': np.std(self.market_cap_weights),
                'market_weights_range': {
                    'min': np.min(self.market_cap_weights),
                    'max': np.max(self.market_cap_weights)
                }
            })

        return diagnostics

    def export_views_to_dataframe(self) -> pd.DataFrame:
        """
        Export investor views to pandas DataFrame
        将投资者观点导出到pandas DataFrame

        Returns:
            pd.DataFrame: Views data
                         观点数据
        """
        if not self.views:
            return pd.DataFrame()

        views_data = []
        for i, view in enumerate(self.views):
            view_dict = {
                'view_id': i + 1,
                'expected_return': view['return'],
                'confidence': view['confidence']
            }

            # Add view vector elements
        for j, weight in enumerate(view['vector']):
            view_dict[f'asset_{j + 1}_weight'] = weight

            views_data.append(view_dict)

        return pd.DataFrame(views_data)

    def validate_inputs(self) -> List[str]:
        """
        Validate model inputs
        验证模型输入

        Returns:
            list: List of validation errors
                  验证错误列表
        """
        errors = []

        # Check market parameters
        if self.market_cap_weights is None:
            errors.append("Market capitalization weights not set")
        elif not np.isclose(np.sum(self.market_cap_weights), 1.0, atol=1e-6):
            errors.append("Market weights do not sum to 1")

        if self.market_returns is None:
            errors.append("Market expected returns not set")

        if self.covariance_matrix is None:
            errors.append("Covariance matrix not set")
        elif not np.allclose(self.covariance_matrix, self.covariance_matrix.T, atol=1e-10):
            errors.append("Covariance matrix is not symmetric")

        # Check views
        for i, view in enumerate(self.views):
            if not np.isclose(np.sum(view['vector']), 0):
                errors.append(f"View {i + 1} vector does not sum to 0 (relative view)")

            if not (0 <= view['confidence'] <= 1):
                errors.append(f"View {i + 1} confidence must be between 0 and 1")

        # Check dimensions consistency
        if (self.market_cap_weights is not None and
            self.market_returns is not None and
                self.covariance_matrix is not None):

            n_assets = len(self.market_cap_weights)
        if len(self.market_returns) != n_assets:
            errors.append("Market returns dimension mismatch")

        if self.covariance_matrix.shape != (n_assets, n_assets):
            errors.append("Covariance matrix dimension mismatch")

        return errors


# Global Black - Litterman optimizer instance
# 全局Black - Litterman优化器实例
black_litterman_optimizer = BlackLittermanOptimizer()

__all__ = [
    'BlackLittermanOptimizer',
    'black_litterman_optimizer'
]
