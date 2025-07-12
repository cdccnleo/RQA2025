import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

class ReturnType(Enum):
    """收益类型枚举"""
    SIMPLE = auto()      # 简单收益
    LOG = auto()         # 对数收益
    EXCESS = auto()      # 超额收益

@dataclass
class PerformanceMetrics:
    """绩效指标数据结构"""
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    value_at_risk: float
    conditional_var: float

class BacktestAnalyzer:
    """回测分析器"""

    def __init__(self, returns: pd.Series, benchmark: Optional[pd.Series] = None):
        self.returns = returns
        self.benchmark = benchmark
        self._validate_inputs()

    def _validate_inputs(self):
        """验证输入数据"""
        if not isinstance(self.returns, pd.Series):
            raise ValueError("Returns must be a pandas Series")
        if self.benchmark is not None and not isinstance(self.benchmark, pd.Series):
            raise ValueError("Benchmark must be a pandas Series or None")

    def calculate_performance(self) -> PerformanceMetrics:
        """计算综合绩效指标"""
        return PerformanceMetrics(
            annualized_return=self._annualized_return(),
            annualized_volatility=self._annualized_volatility(),
            sharpe_ratio=self._sharpe_ratio(),
            max_drawdown=self._max_drawdown(),
            sortino_ratio=self._sortino_ratio(),
            calmar_ratio=self._calmar_ratio(),
            win_rate=self._win_rate(),
            profit_factor=self._profit_factor(),
            value_at_risk=self._value_at_risk(),
            conditional_var=self._conditional_var()
        )

    def _annualized_return(self) -> float:
        """计算年化收益率"""
        cum_return = (1 + self.returns).prod()
        years = len(self.returns) / 252  # 假设252个交易日
        return cum_return ** (1/years) - 1

    def _annualized_volatility(self) -> float:
        """计算年化波动率"""
        return self.returns.std() * np.sqrt(252)

    def _sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        excess_returns = self.returns - risk_free_rate/252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _max_drawdown(self) -> float:
        """计算最大回撤"""
        cum_returns = (1 + self.returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()

    def _sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        excess_returns = self.returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return np.nan
        downside_volatility = downside_returns.std()
        return excess_returns.mean() / downside_volatility * np.sqrt(252)

    def _calmar_ratio(self) -> float:
        """计算Calmar比率"""
        max_dd = abs(self._max_drawdown())
        if max_dd == 0:
            return np.nan
        return self._annualized_return() / max_dd

    def _win_rate(self) -> float:
        """计算胜率"""
        return (self.returns > 0).mean()

    def _profit_factor(self) -> float:
        """计算盈利因子"""
        gross_profit = self.returns[self.returns > 0].sum()
        gross_loss = abs(self.returns[self.returns < 0].sum())
        if gross_loss == 0:
            return np.inf
        return gross_profit / gross_loss

    def _value_at_risk(self, alpha: float = 0.05) -> float:
        """计算VaR"""
        return np.percentile(self.returns, alpha * 100)

    def _conditional_var(self, alpha: float = 0.05) -> float:
        """计算条件VaR"""
        var = self._value_at_risk(alpha)
        return self.returns[self.returns <= var].mean()

    def plot_returns(self) -> plt.Figure:
        """绘制收益曲线"""
        fig, ax = plt.subplots(figsize=(12,6))
        cum_returns = (1 + self.returns).cumprod()
        cum_returns.plot(ax=ax, label='Strategy')

        if self.benchmark is not None:
            cum_bench = (1 + self.benchmark).cumprod()
            cum_bench.plot(ax=ax, label='Benchmark', linestyle='--')

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(True)
        return fig

    def plot_drawdown(self) -> plt.Figure:
        """绘制回撤曲线"""
        fig, ax = plt.subplots(figsize=(12,6))
        cum_returns = (1 + self.returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        drawdown.plot(ax=ax)
        ax.set_title("Drawdown")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.grid(True)
        return fig

class PortfolioAnalyzer:
    """组合分析器"""

    def __init__(self, portfolio_returns: pd.DataFrame):
        self.portfolio = portfolio_returns

    def calculate_attribution(self) -> pd.DataFrame:
        """计算组合归因"""
        # 简单实现 - 实际应用中需要更复杂的模型
        return self.portfolio.corr()

    def plot_correlation(self) -> plt.Figure:
        """绘制相关性热力图"""
        fig, ax = plt.subplots(figsize=(10,8))
        corr = self.portfolio.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm',
                   vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title("Portfolio Correlation")
        return fig

class TransactionCostModel:
    """交易成本模型"""

    def __init__(self, commission: float = 0.0003,
                slippage: float = 0.0005,
                impact: float = 0.001):
        self.commission = commission
        self.slippage = slippage
        self.impact = impact

    def estimate_cost(self, trade_size: float,
                     price: float,
                     liquidity: float) -> float:
        """估算交易成本"""
        # 简单实现 - 实际应用中需要更复杂的模型
        commission_cost = trade_size * price * self.commission
        slippage_cost = trade_size * price * self.slippage
        impact_cost = (trade_size / liquidity) * price * self.impact
        return commission_cost + slippage_cost + impact_cost
