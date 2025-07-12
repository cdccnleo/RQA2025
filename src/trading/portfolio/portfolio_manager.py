import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod
import warnings
from scipy.optimize import minimize
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

class PortfolioMethod(Enum):
    """组合优化方法枚举"""
    EQUAL_WEIGHT = auto()      # 等权重
    MEAN_VARIANCE = auto()    # 均值方差
    RISK_PARITY = auto()      # 风险平价
    BLACK_LITTERMAN = auto()  # BL模型

class AttributionFactor(Enum):
    """归因因子枚举"""
    MARKET = auto()      # 市场因子
    SIZE = auto()       # 市值因子
    VALUE = auto()      # 价值因子
    MOMENTUM = auto()   # 动量因子
    VOLATILITY = auto() # 波动因子

@dataclass
class StrategyPerformance:
    """策略绩效数据结构"""
    returns: pd.Series
    sharpe: float
    max_drawdown: float
    turnover: float
    factor_exposure: Dict[AttributionFactor, float]

@dataclass
class PortfolioConstraints:
    """组合约束条件"""
    max_weight: float = 0.3
    min_weight: float = 0.05
    max_turnover: float = 0.5
    max_leverage: float = 1.0

class BasePortfolioOptimizer(ABC):
    """组合优化基类"""

    @abstractmethod
    def optimize(self,
                performances: Dict[str, StrategyPerformance],
                constraints: PortfolioConstraints) -> Dict[str, float]:
        """优化组合权重"""
        pass

class EqualWeightOptimizer(BasePortfolioOptimizer):
    """等权重优化"""

    def optimize(self, performances, constraints):
        n = len(performances)
        return {name: 1/n for name in performances.keys()}

class MeanVarianceOptimizer(BasePortfolioOptimizer):
    """均值方差优化"""

    def __init__(self, lookback: int = 252, risk_aversion: float = 1.0):
        self.lookback = lookback
        self.risk_aversion = risk_aversion

    def optimize(self, performances, constraints):
        # 准备输入数据
        returns = pd.DataFrame({name: perf.returns
                              for name, perf in performances.items()})
        recent_returns = returns.iloc[-self.lookback:]

        # 计算预期收益和协方差
        mu = recent_returns.mean()
        sigma = recent_returns.cov()

        # 优化问题
        n = len(performances)
        init_guess = np.repeat(1/n, n)
        bounds = [(constraints.min_weight, constraints.max_weight)] * n
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: constraints.max_leverage - np.sum(np.abs(x))}
        ]

        def objective(x):
            port_return = x @ mu
            port_risk = np.sqrt(x @ sigma @ x)
            return - (port_return - self.risk_aversion * port_risk)

        result = minimize(
            objective,
            init_guess,
            bounds=bounds,
            constraints=constraints
        )

        return dict(zip(performances.keys(), result.x))

class RiskParityOptimizer(BasePortfolioOptimizer):
    """风险平价优化"""

    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self.cov_estimator = LedoitWolf()

    def optimize(self, performances, constraints):
        # 估计协方差矩阵
        returns = pd.DataFrame({name: perf.returns
                              for name, perf in performances.items()})
        cov = self._estimate_covariance(returns)

        # 风险平价优化
        n = len(performances)
        init_guess = np.repeat(1/n, n)
        bounds = [(constraints.min_weight, constraints.max_weight)] * n
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        def objective(x):
            marginal_risk = cov.values @ x
            risk_contrib = x * marginal_risk
            target_contrib = np.ones(n) / n
            return np.sum((risk_contrib - target_contrib) ** 2)

        result = minimize(
            objective,
            init_guess,
            bounds=bounds,
            constraints=constraints
        )

        return dict(zip(performances.keys(), result.x))

    def _estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """估计协方差矩阵"""
        self.cov_estimator.fit(returns.iloc[-self.lookback:])
        return pd.DataFrame(
            self.cov_estimator.covariance_,
            index=returns.columns,
            columns=returns.columns
        )

class PortfolioManager:
    """组合管理核心类"""

    def __init__(self,
                optimizer: BasePortfolioOptimizer,
                rebalance_freq: str = 'M'):
        """
        Args:
            optimizer: 组合优化器
            rebalance_freq: 再平衡频率 ('D','W','M','Q','Y')
        """
        self.optimizer = optimizer
        self.rebalance_freq = rebalance_freq
        self.current_weights = {}

    def run_backtest(self,
                    strategy_performances: Dict[str, StrategyPerformance],
                    constraints: PortfolioConstraints,
                    start_date: str,
                    end_date: str) -> pd.DataFrame:
        """运行组合回测"""
        dates = pd.date_range(start_date, end_date, freq=self.rebalance_freq)
        weights_history = []

        for date in dates:
            # 获取历史绩效
            hist_perf = {
                name: StrategyPerformance(
                    returns=perf.returns.loc[:date],
                    sharpe=perf.sharpe,
                    max_drawdown=perf.max_drawdown,
                    turnover=perf.turnover,
                    factor_exposure=perf.factor_exposure
                )
                for name, perf in strategy_performances.items()
            }

            # 优化组合权重
            new_weights = self.optimizer.optimize(hist_perf, constraints)
            weights_history.append((date, new_weights))
            self.current_weights = new_weights

        return pd.DataFrame(
            dict(weights_history),
            index=dates
        ).T

    def calculate_attribution(self,
                            weights_df: pd.DataFrame,
                            strategy_performances: Dict[str, StrategyPerformance]) -> pd.DataFrame:
        """计算绩效归因"""
        # 准备因子数据
        factor_data = pd.DataFrame({
            name: perf.factor_exposure
            for name, perf in strategy_performances.items()
        }).T

        # 计算加权因子暴露
        weighted_exposure = pd.DataFrame(index=weights_df.columns)
        for factor in AttributionFactor:
            weighted_exposure[factor.name] = weights_df.T.dot(factor_data[factor])

        return weighted_exposure

class PortfolioVisualizer:
    """组合可视化工具"""

    @staticmethod
    def plot_weights(weights_df: pd.DataFrame):
        """绘制权重历史"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        weights_df.T.plot(kind='area', stacked=True, ax=ax)
        ax.set_title('Strategy Weights Over Time')
        ax.set_ylabel('Weight')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_attribution(attribution_df: pd.DataFrame):
        """绘制归因分析"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        attribution_df.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Factor Attribution Analysis')
        ax.set_ylabel('Exposure')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_performance(weights_df: pd.DataFrame,
                        strategy_performances: Dict[str, StrategyPerformance]):
        """绘制组合绩效"""
        import matplotlib.pyplot as plt

        # 计算组合收益
        portfolio_returns = pd.Series(0, index=strategy_performances[next(iter(strategy_performances))].returns.index)
        for name, perf in strategy_performances.items():
            portfolio_returns += weights_df.loc[name] * perf.returns

        # 绘制累计收益
        fig, ax = plt.subplots(figsize=(12, 6))
        portfolio_returns.cumsum().plot(ax=ax)
        ax.set_title('Cumulative Portfolio Returns')
        ax.set_ylabel('Return')
        plt.tight_layout()
        return fig
