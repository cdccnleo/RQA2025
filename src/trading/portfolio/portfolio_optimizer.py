import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
import logging
import cvxpy as cp
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """优化方法枚举"""
    MEAN_VARIANCE = auto()   # 均值-方差
    RISK_PARITY = auto()     # 风险平价
    MAX_DIVERSITY = auto()   # 最大分散化
    BLACK_LITTERMAN = auto() # Black-Litterman

class ConstraintType(Enum):
    """约束类型枚举"""
    LEVERAGE = auto()        # 杠杆限制
    CONCENTRATION = auto()   # 集中度限制
    TURNOVER = auto()        # 换手率限制
    SHORTING = auto()        # 卖空限制

@dataclass
class PortfolioResult:
    """组合优化结果"""
    weights: pd.Series
    performance: Dict[str, float]
    risk_contributions: pd.Series
    constraints: Dict[ConstraintType, float]

class BaseOptimizer(ABC):
    """优化器基类"""

    @abstractmethod
    def optimize(self, returns: pd.DataFrame,
                method: OptimizationMethod,
                constraints: Dict[ConstraintType, float]) -> PortfolioResult:
        """执行组合优化"""
        pass

class MeanVarianceOptimizer(BaseOptimizer):
    """均值-方差优化器"""

    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion

    def optimize(self, returns: pd.DataFrame,
                method: OptimizationMethod,
                constraints: Dict[ConstraintType, float]) -> PortfolioResult:
        # 计算预期收益和协方差矩阵
        mu = returns.mean()
        Sigma = returns.cov()

        # 定义优化变量
        n = len(mu)
        w = cp.Variable(n)

        # 定义优化问题
        risk = cp.quad_form(w, Sigma)
        ret = mu @ w
        objective = cp.Maximize(ret - self.risk_aversion * risk)

        # 添加约束
        constraints_list = [cp.sum(w) == 1, w >= 0]
        if ConstraintType.LEVERAGE in constraints:
            constraints_list.append(cp.norm(w, 1) <= constraints[ConstraintType.LEVERAGE])

        # 求解
        prob = cp.Problem(objective, constraints_list)
        prob.solve()

        # 计算风险贡献
        weights = pd.Series(w.value, index=mu.index)
        risk_contrib = self._calculate_risk_contributions(weights, Sigma)

        return PortfolioResult(
            weights=weights,
            performance={
                'expected_return': ret.value,
                'volatility': np.sqrt(risk.value)
            },
            risk_contributions=risk_contrib,
            constraints=constraints
        )

    def _calculate_risk_contributions(self, weights: pd.Series,
                                    Sigma: pd.DataFrame) -> pd.Series:
        """计算风险贡献"""
        portfolio_var = weights.T @ Sigma @ weights
        marginal_contrib = Sigma @ weights
        risk_contrib = weights * marginal_contrib / portfolio_var
        return risk_contrib

class RiskParityOptimizer(BaseOptimizer):
    """风险平价优化器"""

    def optimize(self, returns: pd.DataFrame,
                method: OptimizationMethod,
                constraints: Dict[ConstraintType, float]) -> PortfolioResult:
        # 计算协方差矩阵
        Sigma = returns.cov()
        n = Sigma.shape[0]

        # 定义优化问题
        def objective(w):
            w = np.array(w)
            rc = w * (Sigma @ w)
            rc = rc / (w.T @ Sigma @ w)
            return np.sum((rc - 1/n)**2)

        # 初始权重
        x0 = np.ones(n) / n

        # 约束条件
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n)]

        # 优化求解
        res = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=cons
        )

        # 计算结果
        weights = pd.Series(res.x, index=Sigma.index)
        risk_contrib = self._calculate_risk_contributions(weights, Sigma)

        return PortfolioResult(
            weights=weights,
            performance={
                'expected_return': (returns.mean() @ weights),
                'volatility': np.sqrt(weights.T @ Sigma @ weights)
            },
            risk_contributions=risk_contrib,
            constraints=constraints
        )

    def _calculate_risk_contributions(self, weights: pd.Series,
                                    Sigma: pd.DataFrame) -> pd.Series:
        """计算风险贡献"""
        portfolio_var = weights.T @ Sigma @ weights
        marginal_contrib = Sigma @ weights
        risk_contrib = weights * marginal_contrib / portfolio_var
        return risk_contrib

class PortfolioVisualizer:
    """组合可视化工具"""

    @staticmethod
    def plot_weights(weights: pd.Series, top_n: int = 10) -> plt.Figure:
        """绘制权重分布图"""
        plt.figure(figsize=(12,6))
        weights.sort_values(ascending=False).head(top_n).plot.bar()
        plt.title("Portfolio Weights")
        plt.xlabel("Asset")
        plt.ylabel("Weight")
        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def plot_risk_contributions(risk_contrib: pd.Series) -> plt.Figure:
        """绘制风险贡献图"""
        plt.figure(figsize=(12,6))
        risk_contrib.sort_values(ascending=False).plot.bar(color='orange')
        plt.title("Risk Contributions")
        plt.xlabel("Asset")
        plt.ylabel("Contribution")
        plt.tight_layout()
        return plt.gcf()

    @staticmethod
    def plot_efficient_frontier(returns: pd.DataFrame) -> plt.Figure:
        """绘制有效前沿"""
        plt.figure(figsize=(12,6))

        # 计算预期收益和协方差
        mu = returns.mean()
        Sigma = returns.cov()
        n = len(mu)

        # 生成随机权重
        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            w = np.random.random(n)
            w /= np.sum(w)
            results[0,i] = np.sum(w * mu)
            results[1,i] = np.sqrt(w.T @ Sigma @ w)
            results[2,i] = results[0,i] / results[1,i]  # Sharpe ratio

        # 绘制散点图
        plt.scatter(results[1,:], results[0,:], c=results[2,:],
                   cmap='viridis', alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')
        plt.title("Efficient Frontier")
        plt.xlabel("Volatility")
        plt.ylabel("Return")
        plt.tight_layout()
        return plt.gcf()

class PortfolioManager:
    """组合管理器"""

    def __init__(self, optimizers: Dict[OptimizationMethod, BaseOptimizer]):
        self.optimizers = optimizers

    def rebalance(self, returns: pd.DataFrame,
                 method: OptimizationMethod,
                 constraints: Dict[ConstraintType, float]) -> PortfolioResult:
        """组合再平衡"""
        optimizer = self.optimizers.get(method)
        if not optimizer:
            raise ValueError(f"Optimizer for method {method} not found")

        return optimizer.optimize(returns, method, constraints)

    def analyze_performance(self, returns: pd.DataFrame,
                          weights: pd.Series) -> Dict[str, float]:
        """分析组合绩效"""
        # 计算基本指标
        portfolio_returns = (returns * weights).sum(axis=1)
        annual_return = (1 + portfolio_returns).prod() ** (252/len(returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol

        # 计算最大回撤
        cum_returns = (1 + portfolio_returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        max_dd = drawdown.min()

        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        }
