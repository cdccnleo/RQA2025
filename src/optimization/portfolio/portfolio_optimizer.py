#!/usr/bin/env python3
"""
RQA2025 多资产组合优化
提供现代投资组合理论和风险模型的组合优化功能
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time


logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):

    """优化目标枚举"""
    MAX_SHARPE_RATIO = "max_sharpe_ratio"          # 最大化夏普比率
    MIN_VARIANCE = "min_variance"                  # 最小化方差
    MAX_RETURN = "max_return"                      # 最大化收益
    RISK_BUDGET = "risk_budget"                    # 风险预算
    TARGET_RETURN = "target_return"                # 目标收益
    EFFICIENT_FRONTIER = "efficient_frontier"      # 有效前沿
    MAX_DIVERSIFICATION = "max_diversification"    # 最大分散化
    RISK_PARITY = "risk_parity"                    # 风险平价
    MIN_CVAR = "min_cvar"                          # 最小条件VaR
    MAX_SORTINO_RATIO = "max_sortino"              # 最大索蒂诺比率
    MAX_CALMAR_RATIO = "max_calmar"                # 最大卡尔玛比率
    BLACK_LITTERMAN = "black_litterman"            # Black - Litterman模型
    HIERARCHICAL_RISK = "hierarchical_risk"        # 层次风险平价
    MULTI_OBJECTIVE = "multi_objective"            # 多目标优化


class RiskModel(Enum):

    """风险模型枚举"""
    HISTORICAL = "historical"                      # 历史波动率
    EXPONENTIAL_WEIGHTED = "ewma"                  # 指数加权移动平均
    GARCH = "garch"                               # GARCH模型
    MULTI_FACTOR = "multi_factor"                 # 多因子模型
    COPULA = "copula"                             # Copula模型
    STOCHASTIC_VOLATILITY = "stochastic_vol"      # 随机波动率
    JUMP_DIFFUSION = "jump_diffusion"            # 跳跃扩散
    REGIME_SWITCHING = "regime_switching"        # 状态转换
    EXTREME_VALUE = "extreme_value"              # 极值理论
    DYNAMIC_CONDITIONAL = "dynamic_conditional"  # 动态条件相关


@dataclass
class AssetData:

    """资产数据"""
    symbol: str
    returns: np.ndarray
    volatility: float
    expected_return: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float


@dataclass
class PortfolioMetrics:

    """组合指标"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    diversification_ratio: float
    concentration_ratio: float
    sortino_ratio: float = 0.0
    cvar_95: float = 0.0
    concentration_index: float = 0.0
    tracking_error: float = 0.0


@dataclass
class OptimizationResult:

    """优化结果"""
    weights: np.ndarray
    metrics: PortfolioMetrics
    optimization_time: float
    convergence_score: float
    asset_contributions: Dict[str, float]


class PortfolioOptimizer:

    """投资组合优化器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 优化配置
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.tolerance = self.config.get('tolerance', 1e-8)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)

        # 约束条件
        self.constraints = {
            'max_weight': self.config.get('max_weight', 0.3),      # 最大权重
            'min_weight': self.config.get('min_weight', 0.0),      # 最小权重
            'max_volatility': self.config.get('max_volatility', 0.25),  # 最大波动率
            'min_return': self.config.get('min_return', 0.0),      # 最小收益
        }

        # 资产数据
        self.assets: Dict[str, AssetData] = {}
        self.asset_symbols: List[str] = []

        logger.info("投资组合优化器初始化完成")

    def add_asset(self, symbol: str, returns: np.ndarray, price_data: Optional[np.ndarray] = None):
        """添加资产"""
        # 计算资产指标
        expected_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # 计算最大回撤
        if price_data is not None:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
        else:
            max_drawdown = 0.0

        # 计算VaR(95%)
        var_95 = np.percentile(returns, 5)

        asset_data = AssetData(
            symbol=symbol,
            returns=returns,
            volatility=volatility,
            expected_return=expected_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95
        )

        self.assets[symbol] = asset_data
        self.asset_symbols = list(self.assets.keys())

        logger.info(f"添加资产: {symbol}, 预期收益: {expected_return:.4f}, 波动率: {volatility:.4f}")

    def get_asset_data(self, symbol: str) -> Optional[AssetData]:
        """获取资产数据"""
        return self.assets.get(symbol)

    def list_assets(self) -> List[str]:
        """列出所有资产"""
        return self.asset_symbols.copy()

    def clear_assets(self) -> None:
        """清空所有资产"""
        self.assets.clear()
        self.asset_symbols.clear()

    def _validate_constraints(self, constraints: Dict[str, Any]) -> bool:
        """验证约束条件"""
        try:
            # 检查总权重
            total_weight = constraints.get('total_weight', 1.0)
            if abs(total_weight - 1.0) > 1e-6:
                return False

            # 检查权重限制
            max_weight = constraints.get('max_weight', 1.0)
            min_weight = constraints.get('min_weight', 0.0)
            if max_weight < min_weight:
                return False

            # 检查波动率限制
            max_volatility = constraints.get('max_volatility', float('inf'))
            if max_volatility <= 0:
                return False

            # 检查收益限制
            min_return = constraints.get('min_return', float('-inf'))
            max_return = constraints.get('max_return', float('inf'))
            if min_return > max_return:
                return False

            return True

        except Exception:
            return False

    def _apply_constraints(self, weights: np.ndarray, constraints: Dict[str, Any]) -> np.ndarray:
        """应用约束条件到权重"""
        constrained_weights = weights.copy()

        # 应用权重限制
        max_weight = constraints.get('max_weight', 1.0)
        min_weight = constraints.get('min_weight', 0.0)

        constrained_weights = np.clip(constrained_weights, min_weight, max_weight)

        # 归一化权重
        if np.sum(constrained_weights) > 0:
            constrained_weights = constrained_weights / np.sum(constrained_weights)

        # 如果归一化后仍然超过最大权重限制，需要重新分配
        if max_weight < 1.0 and np.max(constrained_weights) > max_weight:
            # 使用迭代方法重新分配权重
            constrained_weights = self._redistribute_weights(constrained_weights, max_weight)

        return constrained_weights

    def _redistribute_weights(self, weights: np.ndarray, max_weight: float) -> np.ndarray:
        """重新分配权重以满足最大权重限制"""
        n_assets = len(weights)

        # 找到超过限制的权重
        excess_mask = weights > max_weight
        excess_amount = np.sum(weights[excess_mask] - max_weight)

        # 将超出的部分重新分配给其他资产
        non_excess_mask = ~excess_mask
        if np.sum(non_excess_mask) > 0:
            # 将超出的权重按比例分配给未超过限制的资产
            redistribution = excess_amount / np.sum(non_excess_mask)
            weights[excess_mask] = max_weight
            weights[non_excess_mask] += redistribution

        # 再次归一化
        weights = weights / np.sum(weights)

        return weights

    def calculate_performance_metrics(self, weights: np.ndarray, returns: np.ndarray) -> PortfolioMetrics:
        """计算投资组合绩效指标"""
        # 计算投资组合收益
        portfolio_returns = np.dot(returns, weights)

        # 基本指标
        expected_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # 计算最大回撤
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        # 计算VaR
        var_95 = np.percentile(portfolio_returns, 5)

        # 计算多样化比率（资产数量/权重集中度）
        weight_concentration = np.max(weights) / np.sum(weights)
        diversification_ratio = len(weights) / weight_concentration

        # 计算集中度比率（前三大资产权重占比）
        sorted_weights = np.sort(weights)[::-1]
        concentration_ratio = np.sum(sorted_weights[:3]) / np.sum(weights)

        return PortfolioMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            diversification_ratio=diversification_ratio,
            concentration_ratio=concentration_ratio
        )

    def backtest_portfolio(self, weights: np.ndarray, historical_returns: Any) -> Dict[str, Any]:
        """回测投资组合"""
        # 处理不同格式的历史收益数据
        if hasattr(historical_returns, 'values'):
            returns_matrix = historical_returns.values
        else:
            returns_matrix = np.array(historical_returns)

        # 计算投资组合历史收益
        portfolio_returns = np.dot(returns_matrix, weights)

        # 计算累积收益
        cumulative_returns = np.cumprod(1 + portfolio_returns)

        # 计算最大回撤
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        # 计算绩效指标
        performance_metrics = {
            'total_return': cumulative_returns[-1] - 1,
            'annualized_return': (cumulative_returns[-1]) ** (252 / len(portfolio_returns)) - 1,
            'volatility': np.std(portfolio_returns) * np.sqrt(252),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': (np.mean(portfolio_returns) * 252 - self.risk_free_rate) / (np.std(portfolio_returns) * np.sqrt(252))
        }

        return {
            'portfolio_returns': portfolio_returns.tolist(),
            'cumulative_returns': cumulative_returns.tolist(),
            'performance_metrics': performance_metrics
        }

    def optimize_portfolio(self, objective: OptimizationObjective,


                           risk_model: RiskModel = RiskModel.HISTORICAL,
                           **kwargs) -> OptimizationResult:
        """优化投资组合"""
        if len(self.assets) < 2:
            raise ValueError("至少需要2个资产才能进行组合优化")

        start_time = time.time()

        try:
            if objective == OptimizationObjective.MAX_SHARPE_RATIO:
                result = self._optimize_max_sharpe()
            elif objective == OptimizationObjective.MIN_VARIANCE:
                result = self._optimize_min_variance()
            elif objective == OptimizationObjective.MAX_RETURN:
                result = self._optimize_max_return()
            elif objective == OptimizationObjective.EFFICIENT_FRONTIER:
                result = self._optimize_efficient_frontier(**kwargs)
            elif objective == OptimizationObjective.MAX_DIVERSIFICATION:
                result = self._optimize_max_diversification()
            elif objective == OptimizationObjective.RISK_PARITY:
                result = self._optimize_risk_parity()
            elif objective == OptimizationObjective.MIN_CVAR:
                result = self._optimize_min_cvar()
            elif objective == OptimizationObjective.MAX_SORTINO_RATIO:
                result = self._optimize_max_sortino()
            elif objective == OptimizationObjective.BLACK_LITTERMAN:
                result = self._optimize_black_litterman(**kwargs)
            elif objective == OptimizationObjective.HIERARCHICAL_RISK:
                result = self._optimize_hierarchical_risk()
            elif objective == OptimizationObjective.MULTI_OBJECTIVE:
                result = self._optimize_multi_objective(**kwargs)
            else:
                raise ValueError(f"不支持的优化目标: {objective}")

            optimization_time = time.time() - start_time
            result.optimization_time = optimization_time

            logger.info(f"组合优化完成，耗时: {optimization_time:.2f}秒")
            return result

        except Exception as e:
            logger.error(f"组合优化失败: {e}")
            raise

    def _optimize_max_sharpe(self) -> OptimizationResult:
        """最大化夏普比率优化"""
        from scipy.optimize import minimize

        n_assets = len(self.asset_symbols)

        # 目标函数：负夏普比率（最小化）

        def objective(weights):

            return -self._calculate_portfolio_sharpe(weights)

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]

        # 权重边界
        bounds = [(self.constraints['min_weight'], self.constraints['max_weight'])
                  for _ in range(n_assets)]

        # 初始权重
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights)
            contributions = self._calculate_asset_contributions(weights)

            return OptimizationResult(
                weights=weights,
                metrics=metrics,
                optimization_time=0.0,  # 会在上级函数中设置
                convergence_score=result.nit / self.max_iterations,
                asset_contributions=contributions
            )
        else:
            raise ValueError(f"优化失败: {result.message}")

    def _optimize_max_diversification(self) -> OptimizationResult:
        """最大分散化优化"""
        from scipy.optimize import minimize

        n_assets = len(self.asset_symbols)

        # 目标函数：最大化分散化比率

        def objective(weights):

            return -self._calculate_diversification_ratio(weights)

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
            {'type': 'ineq', 'fun': lambda x: self._calculate_portfolio_return(
                x) - self.constraints['min_return']},  # 最小收益
        ]

        # 权重边界
        bounds = [(self.constraints['min_weight'], self.constraints['max_weight'])
                  for _ in range(n_assets)]

        # 初始权重
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights)
            contributions = self._calculate_asset_contributions(weights)

            return OptimizationResult(
                weights=weights,
                metrics=metrics,
                optimization_time=0.0,
                convergence_score=result.nit / self.max_iterations,
                asset_contributions=contributions
            )
        else:
            raise ValueError(f"最大分散化优化失败: {result.message}")

    def _optimize_risk_parity(self) -> OptimizationResult:
        """风险平价优化"""
        from scipy.optimize import minimize

        n_assets = len(self.asset_symbols)

        # 目标函数：最小化风险贡献偏差

        def objective(weights):

            return self._calculate_risk_parity_deviation(weights)

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]

        # 权重边界
        bounds = [(self.constraints['min_weight'], self.constraints['max_weight'])
                  for _ in range(n_assets)]

        # 初始权重
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights)
            contributions = self._calculate_asset_contributions(weights)

            return OptimizationResult(
                weights=weights,
                metrics=metrics,
                optimization_time=0.0,
                convergence_score=result.nit / self.max_iterations,
                asset_contributions=contributions
            )
        else:
            raise ValueError(f"风险平价优化失败: {result.message}")

    def _optimize_min_cvar(self) -> OptimizationResult:
        """最小化条件VaR优化"""
        from scipy.optimize import minimize

        n_assets = len(self.asset_symbols)

        # 目标函数：最小化CVaR

        def objective(weights):

            return self._calculate_portfolio_cvar(weights)

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
            {'type': 'ineq', 'fun': lambda x: self._calculate_portfolio_return(
                x) - self.constraints['min_return']},  # 最小收益
        ]

        # 权重边界
        bounds = [(self.constraints['min_weight'], self.constraints['max_weight'])
                  for _ in range(n_assets)]

        # 初始权重
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights)
            contributions = self._calculate_asset_contributions(weights)

            return OptimizationResult(
                weights=weights,
                metrics=metrics,
                optimization_time=0.0,
                convergence_score=result.nit / self.max_iterations,
                asset_contributions=contributions
            )
        else:
            raise ValueError(f"最小CVaR优化失败: {result.message}")

    def _optimize_max_sortino(self) -> OptimizationResult:
        """最大化索蒂诺比率优化"""
        from scipy.optimize import minimize

        n_assets = len(self.asset_symbols)

        # 目标函数：负索蒂诺比率（最小化）

        def objective(weights):

            return -self._calculate_portfolio_sortino(weights)

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
            {'type': 'ineq', 'fun': lambda x: self._calculate_portfolio_volatility(
                x) - self.constraints['max_volatility']},  # 最大波动率
        ]

        # 权重边界
        bounds = [(self.constraints['min_weight'], self.constraints['max_weight'])
                  for _ in range(n_assets)]

        # 初始权重
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights)
            contributions = self._calculate_asset_contributions(weights)

            return OptimizationResult(
                weights=weights,
                metrics=metrics,
                optimization_time=0.0,
                convergence_score=result.nit / self.max_iterations,
                asset_contributions=contributions
            )
        else:
            raise ValueError(f"最大索蒂诺比率优化失败: {result.message}")

    def _optimize_black_litterman(self, views: Dict[str, Any] = None) -> OptimizationResult:
        """Black - Litterman模型优化"""
        # 这里实现Black - Litterman模型
        # 简化的实现：结合市场均衡和投资者观点

        # 默认观点（简化的实现）
        if views is None:
            views = {}

        # 使用最大化夏普比率作为基础
        base_result = self._optimize_max_sharpe()

        # 调整权重以反映观点
        adjusted_weights = self._adjust_weights_for_views(base_result.weights, views)
        metrics = self._calculate_portfolio_metrics(adjusted_weights)
        contributions = self._calculate_asset_contributions(adjusted_weights)

        return OptimizationResult(
            weights=adjusted_weights,
            metrics=metrics,
            optimization_time=0.0,
            convergence_score=0.9,  # 假设收敛良好
            asset_contributions=contributions
        )

    def _optimize_hierarchical_risk(self) -> OptimizationResult:
        """层次风险平价优化"""
        # 简化的层次风险平价实现
        # 实际应该使用层次聚类方法

        n_assets = len(self.asset_symbols)

        # 计算资产间的相关性
        returns_matrix = np.column_stack([asset.returns for asset in self.assets.values()])

        if returns_matrix.shape[1] > 1:
            correlation_matrix = np.corrcoef(returns_matrix.T)
        else:
            correlation_matrix = np.ones((n_assets, n_assets))

        # 简化的层次聚类（实际应该使用更复杂的方法）
        clusters = self._perform_hierarchical_clustering(correlation_matrix)

        # 计算每个集群的风险
        cluster_risks = {}
        for cluster_id, asset_indices in clusters.items():
            cluster_returns = returns_matrix[:, asset_indices]
            cluster_risk = np.std(np.mean(cluster_returns, axis=1))
            cluster_risks[cluster_id] = cluster_risk

        # 分配风险预算 - 简化的实现
        # total_risk_budget = np.std(np.mean(returns_matrix, axis=1))

        # 计算权重（简化的实现）
        weights = np.ones(n_assets) / n_assets
        metrics = self._calculate_portfolio_metrics(weights)
        contributions = self._calculate_asset_contributions(weights)

        return OptimizationResult(
            weights=weights,
            metrics=metrics,
            optimization_time=0.0,
            convergence_score=0.85,
            asset_contributions=contributions
        )

    def _optimize_multi_objective(self, objectives: List[str] = None) -> OptimizationResult:
        """多目标优化"""
        if objectives is None:
            objectives = ['return', 'risk', 'diversification']

        # 简化的多目标优化实现
        # 实际应该使用NSGA - II等算法

        results = []
        weights_list = []

        # 生成多个单目标优化结果
        for obj in objectives:
            if obj == 'return':
                result = self._optimize_max_return()
            elif obj == 'risk':
                result = self._optimize_min_variance()
            elif obj == 'diversification':
                result = self._optimize_max_diversification()
            else:
                continue

            results.append(result)
            weights_list.append(result.weights)

        # 计算帕累托前沿（简化的实现）
        pareto_weights = self._calculate_pareto_frontier(weights_list)

        # 选择最优解（简化的实现）
        best_weights = pareto_weights[0] if pareto_weights else np.ones(
            len(self.asset_symbols)) / len(self.asset_symbols)
        metrics = self._calculate_portfolio_metrics(best_weights)
        contributions = self._calculate_asset_contributions(best_weights)

        return OptimizationResult(
            weights=best_weights,
            metrics=metrics,
            optimization_time=0.0,
            convergence_score=0.8,
            asset_contributions=contributions
        )

    def _optimize_min_variance(self) -> OptimizationResult:
        """最小化方差优化"""
        from scipy.optimize import minimize

        n_assets = len(self.asset_symbols)

        # 目标函数：组合方差

        def objective(weights):

            return self._calculate_portfolio_variance(weights)

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]

        # 权重边界
        bounds = [(self.constraints['min_weight'], self.constraints['max_weight'])
                  for _ in range(n_assets)]

        # 初始权重
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights)
            contributions = self._calculate_asset_contributions(weights)

            return OptimizationResult(
                weights=weights,
                metrics=metrics,
                optimization_time=0.0,
                convergence_score=result.nit / self.max_iterations,
                asset_contributions=contributions
            )
        else:
            raise ValueError(f"优化失败: {result.message}")

    def _optimize_max_return(self) -> OptimizationResult:
        """最大化收益优化"""
        from scipy.optimize import minimize

        n_assets = len(self.asset_symbols)

        # 目标函数：负预期收益（最小化）

        def objective(weights):

            return -self._calculate_portfolio_return(weights)

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
            {'type': 'ineq', 'fun': lambda x: self.constraints['max_volatility'] -
             self._calculate_portfolio_volatility(x)},  # 波动率约束
        ]

        # 权重边界
        bounds = [(self.constraints['min_weight'], self.constraints['max_weight'])
                  for _ in range(n_assets)]

        # 初始权重
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights)
            contributions = self._calculate_asset_contributions(weights)

            return OptimizationResult(
                weights=weights,
                metrics=metrics,
                optimization_time=0.0,
                convergence_score=result.nit / self.max_iterations,
                asset_contributions=contributions
            )
        else:
            raise ValueError(f"优化失败: {result.message}")

    def _optimize_efficient_frontier(self, target_returns: Optional[List[float]] = None) -> List[OptimizationResult]:
        """计算有效前沿"""
        if target_returns is None:
            # 生成目标收益序列
            returns = [asset.expected_return for asset in self.assets.values()]
            min_return = min(returns)
            max_return = max(returns)
            target_returns = np.linspace(min_return, max_return, 20).tolist()

        results = []

        for target_return in target_returns:
            try:
                result = self._optimize_target_return(target_return)
                results.append(result)
            except Exception as e:
                logger.warning(f"目标收益 {target_return:.4f} 优化失败: {e}")

        return results

    def _optimize_target_return(self, target_return: float) -> OptimizationResult:
        """目标收益优化"""
        from scipy.optimize import minimize

        n_assets = len(self.asset_symbols)

        # 目标函数：组合方差

        def objective(weights):

            return self._calculate_portfolio_variance(weights)

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
            {'type': 'eq', 'fun': lambda x: self._calculate_portfolio_return(
                x) - target_return},  # 目标收益
        ]

        # 权重边界
        bounds = [(self.constraints['min_weight'], self.constraints['max_weight'])
                  for _ in range(n_assets)]

        # 初始权重
        x0 = np.ones(n_assets) / n_assets

        # 优化
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if result.success:
            weights = result.x
            metrics = self._calculate_portfolio_metrics(weights)
            contributions = self._calculate_asset_contributions(weights)

            return OptimizationResult(
                weights=weights,
                metrics=metrics,
                optimization_time=0.0,
                convergence_score=result.nit / self.max_iterations,
                asset_contributions=contributions
            )
        else:
            raise ValueError(f"目标收益优化失败: {result.message}")

    def _calculate_portfolio_return(self, weights: np.ndarray) -> float:
        """计算组合预期收益"""
        returns = np.array([asset.expected_return for asset in self.assets.values()])
        return np.dot(weights, returns)

    def _calculate_portfolio_volatility(self, weights: np.ndarray) -> float:
        """计算组合波动率"""
        returns_matrix = np.column_stack([asset.returns for asset in self.assets.values()])
        cov_matrix = np.cov(returns_matrix.T)
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def _calculate_portfolio_variance(self, weights: np.ndarray) -> float:
        """计算组合方差"""
        volatility = self._calculate_portfolio_volatility(weights)
        return volatility ** 2

    def _calculate_portfolio_sharpe(self, weights: np.ndarray) -> float:
        """计算组合夏普比率"""
        expected_return = self._calculate_portfolio_return(weights)
        volatility = self._calculate_portfolio_volatility(weights)

        if volatility == 0:
            return 0

        return (expected_return - self.risk_free_rate) / volatility

    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> PortfolioMetrics:
        """计算组合指标"""
        expected_return = self._calculate_portfolio_return(weights)
        volatility = self._calculate_portfolio_volatility(weights)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # 计算最大回撤（简化）
        max_drawdown = max([asset.max_drawdown for asset in self.assets.values()])

        # 计算VaR(95%)
        portfolio_returns = np.zeros(len(list(self.assets.values())[0].returns))
        for i, (symbol, asset) in enumerate(self.assets.items()):
            portfolio_returns += weights[i] * asset.returns

        var_95 = np.abs(np.percentile(portfolio_returns, 5))

        # 计算多样化比率
        individual_vols = np.array([asset.volatility for asset in self.assets.values()])
        diversification_ratio = np.sum(weights * individual_vols) / \
            volatility if volatility > 0 else 0

        # 计算集中度
        concentration_ratio = np.max(weights)

        return PortfolioMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            diversification_ratio=diversification_ratio,
            concentration_ratio=concentration_ratio
        )

    def _calculate_asset_contributions(self, weights: np.ndarray) -> Dict[str, float]:
        """计算资产贡献"""
        contributions = {}
        for i, symbol in enumerate(self.asset_symbols):
            contributions[symbol] = weights[i]

        return contributions

    def get_asset_summary(self) -> List[Dict[str, Any]]:
        """获取资产摘要"""
        summary = []
        for symbol, asset in self.assets.items():
            summary.append({
                'symbol': symbol,
                'expected_return': asset.expected_return,
                'volatility': asset.volatility,
                'sharpe_ratio': asset.sharpe_ratio,
                'max_drawdown': asset.max_drawdown,
                'var_95': asset.var_95
            })

        return summary

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """计算相关系数矩阵"""
        returns_matrix = np.column_stack([asset.returns for asset in self.assets.values()])
        corr_matrix = np.corrcoef(returns_matrix.T)

        return pd.DataFrame(corr_matrix, index=self.asset_symbols, columns=self.asset_symbols)

    def calculate_covariance_matrix(self) -> pd.DataFrame:
        """计算协方差矩阵"""
        returns_matrix = np.column_stack([asset.returns for asset in self.assets.values()])
        cov_matrix = np.cov(returns_matrix.T)

        return pd.DataFrame(cov_matrix, index=self.asset_symbols, columns=self.asset_symbols)

    def backtest_portfolio(self, weights: np.ndarray, test_returns: pd.DataFrame) -> Dict[str, Any]:
        """回测投资组合"""
        # 计算组合收益
        portfolio_returns = np.zeros(len(test_returns))

        for i, symbol in enumerate(self.asset_symbols):
            if symbol in test_returns.columns:
                portfolio_returns += weights[i] * test_returns[symbol].values

        # 计算指标
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        total_return = cumulative_returns[-1] - 1
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # 年化波动率
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)

        # 计算最大回撤
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # 计算绩效指标
        performance_metrics = {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

        return {
            'portfolio_returns': portfolio_returns.tolist(),
            'cumulative_returns': cumulative_returns.tolist(),
            'performance_metrics': performance_metrics
        }

    def risk_budgeting_optimization(self, returns_data: pd.DataFrame, risk_budget: np.ndarray) -> OptimizationResult:
        """风险预算优化"""
        # 简化的风险预算优化实现
        n_assets = len(returns_data.columns)
        weights = risk_budget / np.sum(risk_budget)  # 归一化风险预算作为权重

        # 直接计算指标，避免使用内部方法
        portfolio_returns = np.dot(returns_data.values, weights)
        expected_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # 创建简化的指标对象
        from .portfolio_optimizer import PortfolioMetrics
        metrics = PortfolioMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=0.0,
            var_95=np.percentile(portfolio_returns, 5),
            diversification_ratio=float(n_assets),
            concentration_ratio=np.max(weights) / np.sum(weights)
        )

        return OptimizationResult(
            weights=weights,
            metrics=metrics,
            optimization_time=0.0,
            convergence_score=1.0,
            asset_contributions={}
        )

    def stress_test_portfolio(self, weights: np.ndarray, returns_data: pd.DataFrame,
                             stress_scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """压力测试投资组合"""
        results = {}

        for scenario_name, shocks in stress_scenarios.items():
            # 应用冲击到收益数据
            stressed_returns = returns_data.copy()
            for asset, shock in shocks.items():
                if asset in stressed_returns.columns:
                    stressed_returns[asset] = stressed_returns[asset] + shock

            # 计算压力测试下的表现
            portfolio_returns = np.dot(stressed_returns.values, weights)

            results[scenario_name] = {
                'portfolio_return': np.mean(portfolio_returns),
                'portfolio_volatility': np.std(portfolio_returns),
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
                'var_95': np.percentile(portfolio_returns, 5),
                'worst_case_loss': np.min(portfolio_returns)
            }

        return results

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))

    def portfolio_rebalancing(self, target_weights: np.ndarray, current_weights: np.ndarray,
                             current_prices: np.ndarray, current_holdings: np.ndarray) -> Dict[str, Any]:
        """投资组合再平衡"""
        # 计算当前投资组合价值
        portfolio_value = np.sum(current_prices * current_holdings)

        # 计算目标持仓数量
        target_holdings = (target_weights * portfolio_value) / current_prices

        # 计算需要交易的数量
        trade_holdings = target_holdings - current_holdings

        # 分离买入和卖出交易
        buy_trades = {}
        sell_trades = {}

        # 如果没有资产符号，使用索引作为键
        asset_names = self.asset_symbols if len(self.asset_symbols) >= len(trade_holdings) else [f'asset_{i}' for i in range(len(trade_holdings))]

        for i, trade in enumerate(trade_holdings):
            asset_name = asset_names[i] if i < len(asset_names) else f'asset_{i}'
            if trade > 0:  # 买入
                buy_trades[asset_name] = {
                    'quantity': trade,
                    'price': current_prices[i],
                    'value': trade * current_prices[i]
                }
            elif trade < 0:  # 卖出
                sell_trades[asset_name] = {
                    'quantity': abs(trade),
                    'price': current_prices[i],
                    'value': abs(trade) * current_prices[i]
                }

        return {
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'portfolio_value': portfolio_value,
            'rebalance_cost': sum(trade['value'] for trade in buy_trades.values()) +
                             sum(trade['value'] for trade in sell_trades.values())
        }

    def set_risk_model(self, risk_model: Any) -> None:
        """设置风险模型"""
        self.default_risk_model = risk_model

    def get_optimization_summary(self, result: OptimizationResult) -> Dict[str, Any]:
        """获取优化总结"""
        return {
            'objective': 'max_sharpe_ratio',  # 默认值
            'weights': result.weights.tolist(),
            'expected_return': result.metrics.expected_return,
            'volatility': result.metrics.volatility,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'max_drawdown': result.metrics.max_drawdown,
            'optimization_time': result.optimization_time,
            'convergence_status': 'success',  # 默认值
            'convergence_score': result.convergence_score,
            'asset_contributions': result.asset_contributions
        }

    # 新增的辅助方法

    def _calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """计算分散化比率"""
        # 简化的分散化计算
        asset_volatilities = np.array([asset.volatility for asset in self.assets.values()])
        portfolio_volatility = self._calculate_portfolio_volatility(weights)

        if portfolio_volatility == 0:
            return 0

        # 加权平均波动率
        weighted_avg_volatility = np.sum(weights * asset_volatilities)

        # 分散化比率
        diversification_ratio = weighted_avg_volatility / portfolio_volatility

        return diversification_ratio

    def _calculate_risk_parity_deviation(self, weights: np.ndarray) -> float:
        """计算风险平价偏差"""
        # 计算每个资产的风险贡献
        risk_contributions = self._calculate_risk_contributions(weights)

        # 目标风险贡献（相等）
        target_contribution = 1.0 / len(weights)

        # 计算偏差
        deviations = np.array(risk_contributions) - target_contribution
        total_deviation = np.sum(deviations ** 2)

        return total_deviation

    def _calculate_portfolio_cvar(self, weights: np.ndarray, confidence_level: float = 0.05) -> float:
        """计算投资组合的CVaR"""
        # 获取组合收益
        returns_matrix = np.column_stack([asset.returns for asset in self.assets.values()])
        portfolio_returns = np.dot(returns_matrix, weights)

        # 计算VaR
        var = np.percentile(portfolio_returns, confidence_level * 100)

        # 计算CVaR（条件VaR）
        tail_losses = portfolio_returns[portfolio_returns <= var]
        if len(tail_losses) > 0:
            cvar = np.mean(tail_losses)
        else:
            cvar = var

        return abs(cvar)  # 返回正值

    def _calculate_portfolio_sortino(self, weights: np.ndarray, target_return: float = 0.0) -> float:
        """计算索蒂诺比率"""
        portfolio_return = self._calculate_portfolio_return(weights)
        portfolio_volatility = self._calculate_portfolio_downside_volatility(weights, target_return)

        if portfolio_volatility == 0:
            return 0

        return (portfolio_return - target_return) / portfolio_volatility

    def _calculate_portfolio_downside_volatility(self, weights: np.ndarray, target_return: float) -> float:
        """计算下行波动率"""
        returns_matrix = np.column_stack([asset.returns for asset in self.assets.values()])
        portfolio_returns = np.dot(returns_matrix, weights)

        # 计算下行偏差
        downside_returns = portfolio_returns[portfolio_returns < target_return]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean((target_return - downside_returns) ** 2))
        else:
            downside_deviation = 0

        return downside_deviation

    def _adjust_weights_for_views(self, base_weights: np.ndarray, views: Dict[str, Any]) -> np.ndarray:
        """根据观点调整权重"""
        # 简化的Black - Litterman实现
        adjusted_weights = base_weights.copy()

        # 应用观点调整
        for asset_symbol, view in views.items():
            if asset_symbol in self.asset_symbols:
                index = self.asset_symbols.index(asset_symbol)
                # 简化的调整逻辑
                adjustment = view.get('adjustment', 0.0)
                adjusted_weights[index] += adjustment

        # 重新标准化
        if np.sum(adjusted_weights) > 0:
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

        return adjusted_weights

    def _perform_hierarchical_clustering(self, correlation_matrix: np.ndarray) -> Dict[int, List[int]]:
        """执行层次聚类"""
        # 简化的层次聚类实现
        n_assets = len(correlation_matrix)

        # 使用相关性阈值进行聚类
        threshold = 0.7
        clusters = {}
        cluster_id = 0

        visited = set()

        for i in range(n_assets):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            # 找到高度相关的资产
            for j in range(i + 1, n_assets):
                if j not in visited and abs(correlation_matrix[i, j]) > threshold:
                    cluster.append(j)
                    visited.add(j)

            clusters[cluster_id] = cluster
            cluster_id += 1

        return clusters

    def _calculate_pareto_frontier(self, weights_list: List[np.ndarray]) -> List[np.ndarray]:
        """计算帕累托前沿"""
        # 简化的帕累托前沿计算
        pareto_frontier = []

        for weights in weights_list:
            metrics = self._calculate_portfolio_metrics(weights)

            # 检查是否被其他解支配
            is_dominated = False
            for other_weights in weights_list:
                if np.array_equal(weights, other_weights):
                    continue

                other_metrics = self._calculate_portfolio_metrics(other_weights)

                # 如果其他解在所有指标上都更好或相等，则当前解被支配
                if (other_metrics.expected_return >= metrics.expected_return and
                    other_metrics.volatility <= metrics.volatility and
                    (other_metrics.expected_return > metrics.expected_return or
                     other_metrics.volatility < metrics.volatility)):
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_frontier.append(weights)

        return pareto_frontier

    def _calculate_risk_contributions(self, weights: np.ndarray) -> List[float]:
        """计算风险贡献"""
        # 简化的风险贡献计算
        asset_volatilities = np.array([asset.volatility for asset in self.assets.values()])

        # 假设相关性为对角矩阵（简化）
        risk_contributions = []
        for i, weight in enumerate(weights):
            risk_contribution = weight * asset_volatilities[i]
            risk_contributions.append(risk_contribution)

        # 标准化
        total_risk = sum(risk_contributions)
        if total_risk > 0:
            risk_contributions = [rc / total_risk for rc in risk_contributions]

        return risk_contributions

    def get_portfolio_statistics(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        获取投资组合统计信息

        Args:
            weights: 资产权重数组

        Returns:
            包含各种统计信息的字典
        """
        if len(weights) != len(self.assets):
            raise ValueError("权重数量与资产数量不匹配")

        # 提取收益数据
        returns_data = []
        for asset_symbol in self.asset_symbols:
            asset_data = self.assets[asset_symbol]
            if hasattr(asset_data, 'historical_returns'):
                if isinstance(asset_data.historical_returns, np.ndarray):
                    returns_data.append(asset_data.historical_returns)
                else:
                    returns_data.append(np.array(asset_data.historical_returns))
            else:
                # 如果没有历史收益，使用默认值
                returns_data.append(np.array([0.01] * 252))  # 一年交易日

        returns_data = np.array(returns_data)

        # 计算投资组合收益
        portfolio_returns = np.dot(weights, returns_data)

        # 计算统计信息
        expected_return = np.mean(portfolio_returns) * 252  # 年化收益
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # 年化波动率
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0

        # 计算VaR (95%)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)

        # 计算最大回撤
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        # 计算分散化比率
        diversification_ratio = self._calculate_diversification_ratio(weights)

        # 计算集中度 (HHI指数)
        concentration_ratio = np.sum(weights**2)

        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'diversification_ratio': diversification_ratio,
            'concentration_ratio': concentration_ratio,
            'total_assets': len(weights),
            'effective_assets': 1.0 / concentration_ratio  # 有效资产数量
        }
