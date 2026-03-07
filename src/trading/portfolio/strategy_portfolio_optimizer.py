#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略组合优化器

功能：
- 风险平价组合算法
- 均值方差优化
- 策略权重动态调整
- 组合回测和评估

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """优化方法"""
    RISK_PARITY = "risk_parity"           # 风险平价
    MEAN_VARIANCE = "mean_variance"       # 均值方差优化
    EQUAL_WEIGHT = "equal_weight"         # 等权重
    MINIMUM_VARIANCE = "minimum_variance" # 最小方差
    MAXIMUM_SHARPE = "maximum_sharpe"     # 最大夏普比率


@dataclass
class StrategyPerformance:
    """策略表现"""
    strategy_name: str
    returns: pd.Series                      # 收益率序列
    total_return: float                     # 总收益
    annualized_return: float                # 年化收益
    volatility: float                       # 波动率
    sharpe_ratio: float                     # 夏普比率
    max_drawdown: float                     # 最大回撤
    win_rate: float                         # 胜率
    var_95: float                          # 95% VaR
    cvar_95: float                         # 95% CVaR


@dataclass
class PortfolioAllocation:
    """组合配置"""
    strategy_weights: Dict[str, float]      # 策略权重
    method: OptimizationMethod
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    risk_contributions: Dict[str, float]    # 风险贡献
    timestamp: datetime


@dataclass
class PortfolioMetrics:
    """组合指标"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    diversification_ratio: float


class StrategyPortfolioOptimizer:
    """
    策略组合优化器
    
    实现多种组合优化方法：
    - 风险平价（Risk Parity）
    - 均值方差优化（Mean-Variance Optimization）
    - 最小方差（Minimum Variance）
    - 最大夏普比率（Maximum Sharpe Ratio）
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.03,
        max_weight: float = 0.5,
        min_weight: float = 0.0
    ):
        """
        初始化组合优化器
        
        Args:
            risk_free_rate: 无风险利率
            max_weight: 最大权重限制
            min_weight: 最小权重限制
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        
        # 历史配置
        self.allocation_history: List[PortfolioAllocation] = []
        
        logger.info(f"策略组合优化器初始化完成，无风险利率: {risk_free_rate:.2%}")
    
    async def optimize(
        self,
        strategy_performances: Dict[str, StrategyPerformance],
        method: OptimizationMethod = OptimizationMethod.RISK_PARITY,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None
    ) -> PortfolioAllocation:
        """
        优化组合配置
        
        Args:
            strategy_performances: 策略表现数据
            method: 优化方法
            target_return: 目标收益（可选）
            target_risk: 目标风险（可选）
            
        Returns:
            组合配置
        """
        if len(strategy_performances) == 0:
            raise ValueError("策略表现数据为空")
        
        # 构建收益矩阵
        returns_df = self._build_returns_matrix(strategy_performances)
        
        if method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity_optimization(returns_df)
        elif method == OptimizationMethod.MEAN_VARIANCE:
            weights = self._mean_variance_optimization(
                returns_df, target_return, target_risk
            )
        elif method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight_optimization(len(strategy_performances))
        elif method == OptimizationMethod.MINIMUM_VARIANCE:
            weights = self._minimum_variance_optimization(returns_df)
        elif method == OptimizationMethod.MAXIMUM_SHARPE:
            weights = self._maximum_sharpe_optimization(returns_df)
        else:
            raise ValueError(f"不支持的优化方法: {method}")
        
        # 计算组合预期指标
        expected_metrics = self._calculate_expected_metrics(
            weights, returns_df, strategy_performances
        )
        
        # 计算风险贡献
        risk_contributions = self._calculate_risk_contributions(weights, returns_df)
        
        allocation = PortfolioAllocation(
            strategy_weights=dict(zip(strategy_performances.keys(), weights)),
            method=method,
            expected_return=expected_metrics['return'],
            expected_volatility=expected_metrics['volatility'],
            expected_sharpe=expected_metrics['sharpe'],
            risk_contributions=risk_contributions,
            timestamp=datetime.now()
        )
        
        self.allocation_history.append(allocation)
        
        logger.info(f"组合优化完成: 方法={method.value}, "
                   f"预期收益={expected_metrics['return']:.2%}, "
                   f"预期夏普={expected_metrics['sharpe']:.2f}")
        
        return allocation
    
    def _build_returns_matrix(
        self,
        strategy_performances: Dict[str, StrategyPerformance]
    ) -> pd.DataFrame:
        """
        构建收益矩阵
        
        Args:
            strategy_performances: 策略表现数据
            
        Returns:
            收益矩阵DataFrame
        """
        returns_dict = {}
        
        for name, perf in strategy_performances.items():
            returns_dict[name] = perf.returns
        
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def _risk_parity_optimization(
        self,
        returns_df: pd.DataFrame,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        风险平价优化
        
        使每个策略对组合风险的贡献相等
        
        Args:
            returns_df: 收益矩阵
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        Returns:
            最优权重
        """
        n = len(returns_df.columns)
        
        # 计算协方差矩阵
        cov_matrix = returns_df.cov().values
        
        # 初始化等权重
        weights = np.ones(n) / n
        
        for _ in range(max_iter):
            # 计算组合风险
            portfolio_var = weights.T @ cov_matrix @ weights
            
            # 计算边际风险贡献
            marginal_risk = cov_matrix @ weights
            
            # 计算风险贡献
            risk_contrib = weights * marginal_risk
            
            # 目标：每个资产的风险贡献相等
            target_risk = portfolio_var / n
            
            # 更新权重
            new_weights = target_risk / marginal_risk
            new_weights = new_weights / new_weights.sum()
            
            # 应用权重限制
            new_weights = np.clip(new_weights, self.min_weight, self.max_weight)
            new_weights = new_weights / new_weights.sum()
            
            # 检查收敛
            if np.sum(np.abs(new_weights - weights)) < tol:
                break
            
            weights = new_weights
        
        return weights
    
    def _mean_variance_optimization(
        self,
        returns_df: pd.DataFrame,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None
    ) -> np.ndarray:
        """
        均值方差优化
        
        Args:
            returns_df: 收益矩阵
            target_return: 目标收益
            target_risk: 目标风险
            
        Returns:
            最优权重
        """
        try:
            from scipy.optimize import minimize
            
            n = len(returns_df.columns)
            
            # 计算期望收益和协方差
            expected_returns = returns_df.mean().values
            cov_matrix = returns_df.cov().values
            
            # 目标函数：最小化方差
            def portfolio_variance(weights):
                return weights.T @ cov_matrix @ weights
            
            # 约束条件
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            if target_return is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: x @ expected_returns - target_return
                })
            
            # 边界条件
            bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
            
            # 初始权重
            x0 = np.ones(n) / n
            
            # 优化
            result = minimize(
                portfolio_variance,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return result.x
            else:
                logger.warning("均值方差优化失败，使用等权重")
                return np.ones(n) / n
                
        except ImportError:
            logger.warning("scipy未安装，使用等权重")
            return np.ones(len(returns_df.columns)) / len(returns_df.columns)
    
    def _equal_weight_optimization(self, n: int) -> np.ndarray:
        """等权重优化"""
        return np.ones(n) / n
    
    def _minimum_variance_optimization(
        self,
        returns_df: pd.DataFrame
    ) -> np.ndarray:
        """最小方差优化"""
        return self._mean_variance_optimization(returns_df)
    
    def _maximum_sharpe_optimization(
        self,
        returns_df: pd.DataFrame
    ) -> np.ndarray:
        """
        最大夏普比率优化
        
        Args:
            returns_df: 收益矩阵
            
        Returns:
            最优权重
        """
        try:
            from scipy.optimize import minimize
            
            n = len(returns_df.columns)
            
            expected_returns = returns_df.mean().values
            cov_matrix = returns_df.cov().values
            
            # 目标函数：最大化夏普比率（最小化负夏普）
            def negative_sharpe(weights):
                port_return = weights @ expected_returns
                port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
                if port_volatility == 0:
                    return 0
                sharpe = (port_return - self.risk_free_rate) / port_volatility
                return -sharpe
            
            # 约束条件
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
            
            x0 = np.ones(n) / n
            
            result = minimize(
                negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return result.x
            else:
                return np.ones(n) / n
                
        except ImportError:
            return np.ones(len(returns_df.columns)) / len(returns_df.columns)
    
    def _calculate_expected_metrics(
        self,
        weights: np.ndarray,
        returns_df: pd.DataFrame,
        strategy_performances: Dict[str, StrategyPerformance]
    ) -> Dict[str, float]:
        """
        计算预期指标
        
        Args:
            weights: 权重
            returns_df: 收益矩阵
            strategy_performances: 策略表现
            
        Returns:
            预期指标
        """
        # 计算组合收益
        portfolio_returns = returns_df @ weights
        
        expected_return = portfolio_returns.mean()
        expected_volatility = portfolio_returns.std()
        
        # 年化
        annual_return = expected_return * 252
        annual_volatility = expected_volatility * np.sqrt(252)
        
        # 夏普比率
        sharpe = (annual_return - self.risk_free_rate) / annual_volatility \
                 if annual_volatility > 0 else 0
        
        return {
            'return': annual_return,
            'volatility': annual_volatility,
            'sharpe': sharpe
        }
    
    def _calculate_risk_contributions(
        self,
        weights: np.ndarray,
        returns_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        计算风险贡献
        
        Args:
            weights: 权重
            returns_df: 收益矩阵
            
        Returns:
            风险贡献字典
        """
        cov_matrix = returns_df.cov().values
        portfolio_var = weights.T @ cov_matrix @ weights
        
        if portfolio_var == 0:
            return {col: 0.0 for col in returns_df.columns}
        
        marginal_risk = cov_matrix @ weights
        risk_contrib = weights * marginal_risk
        risk_contrib_pct = risk_contrib / portfolio_var
        
        return {
            col: risk_contrib_pct[i]
            for i, col in enumerate(returns_df.columns)
        }
    
    def calculate_portfolio_metrics(
        self,
        allocation: PortfolioAllocation,
        strategy_performances: Dict[str, StrategyPerformance]
    ) -> PortfolioMetrics:
        """
        计算组合指标
        
        Args:
            allocation: 组合配置
            strategy_performances: 策略表现
            
        Returns:
            组合指标
        """
        # 构建组合收益序列
        portfolio_returns = pd.Series(0.0, index=list(strategy_performances.values())[0].returns.index)
        
        for strategy_name, perf in strategy_performances.items():
            weight = allocation.strategy_weights.get(strategy_name, 0)
            portfolio_returns += perf.returns * weight
        
        # 计算指标
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar比率
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino比率
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annualized_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # VaR和CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # 分散化比率
        individual_vols = np.array([perf.volatility for perf in strategy_performances.values()])
        weights = np.array(list(allocation.strategy_weights.values()))
        weighted_vol = np.sum(weights * individual_vols)
        diversification_ratio = weighted_vol / volatility if volatility > 0 else 1
        
        return PortfolioMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            var_95=var_95,
            cvar_95=cvar_95,
            diversification_ratio=diversification_ratio
        )
    
    def rebalance(
        self,
        current_allocation: PortfolioAllocation,
        strategy_performances: Dict[str, StrategyPerformance],
        threshold: float = 0.05
    ) -> bool:
        """
        判断是否需要再平衡
        
        Args:
            current_allocation: 当前配置
            strategy_performances: 最新策略表现
            threshold: 再平衡阈值
            
        Returns:
            是否需要再平衡
        """
        # 计算新配置
        new_allocation = asyncio.run(self.optimize(
            strategy_performances,
            current_allocation.method
        ))
        
        # 比较权重变化
        max_change = 0
        for strategy in current_allocation.strategy_weights.keys():
            old_weight = current_allocation.strategy_weights.get(strategy, 0)
            new_weight = new_allocation.strategy_weights.get(strategy, 0)
            change = abs(new_weight - old_weight)
            max_change = max(max_change, change)
        
        return max_change > threshold


import asyncio

# 全局优化器实例
_optimizer_instance: Optional[StrategyPortfolioOptimizer] = None


def get_portfolio_optimizer(
    risk_free_rate: float = 0.03
) -> StrategyPortfolioOptimizer:
    """
    获取组合优化器实例（单例模式）
    
    Args:
        risk_free_rate: 无风险利率
        
    Returns:
        StrategyPortfolioOptimizer实例
    """
    global _optimizer_instance
    
    if _optimizer_instance is None:
        _optimizer_instance = StrategyPortfolioOptimizer(risk_free_rate)
    
    return _optimizer_instance
