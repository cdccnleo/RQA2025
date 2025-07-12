import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """策略绩效分析器"""

    def __init__(self, returns: pd.Series, benchmark: Optional[pd.Series] = None):
        """
        初始化分析器

        Args:
            returns: 策略收益率序列 (日频)
            benchmark: 基准收益率序列 (可选)
        """
        self.returns = returns
        self.benchmark = benchmark
        self._results = None

    def analyze(self) -> Dict[str, float]:
        """执行全面绩效分析"""
        results = {}

        # 基础指标
        results['total_return'] = self._calculate_total_return()
        results['annual_return'] = self._calculate_annual_return()
        results['annual_volatility'] = self._calculate_annual_volatility()
        results['sharpe_ratio'] = self._calculate_sharpe_ratio()
        results['max_drawdown'] = self._calculate_max_drawdown()
        results['calmar_ratio'] = self._calculate_calmar_ratio()
        results['sortino_ratio'] = self._calculate_sortino_ratio()

        # 与基准比较
        if self.benchmark is not None:
            results['alpha'], results['beta'] = self._calculate_alpha_beta()
            results['information_ratio'] = self._calculate_information_ratio()
            results['tracking_error'] = self._calculate_tracking_error()
            results['outperformance'] = self._calculate_outperformance()

        # 高阶指标
        results['skewness'] = self._calculate_skewness()
        results['kurtosis'] = self._calculate_kurtosis()
        results['var_95'] = self._calculate_var(0.95)
        results['cvar_95'] = self._calculate_cvar(0.95)
        results['win_rate'] = self._calculate_win_rate()
        results['profit_factor'] = self._calculate_profit_factor()

        self._results = results
        return results

    def _calculate_total_return(self) -> float:
        """计算累计收益率"""
        return (1 + self.returns).prod() - 1

    def _calculate_annual_return(self) -> float:
        """计算年化收益率"""
        days = len(self.returns)
        return (1 + self._calculate_total_return()) ** (252 / days) - 1

    def _calculate_annual_volatility(self) -> float:
        """计算年化波动率"""
        return self.returns.std() * np.sqrt(252)

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """计算夏普比率"""
        excess_returns = self.returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        cum_returns = (1 + self.returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()

    def _calculate_calmar_ratio(self) -> float:
        """计算Calmar比率"""
        max_dd = abs(self._calculate_max_drawdown())
        if max_dd == 0:
            return np.nan
        return self._calculate_annual_return() / max_dd

    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """计算Sortino比率"""
        excess_returns = self.returns - risk_free_rate / 252
        downside = excess_returns[excess_returns < 0].std()
        if downside == 0:
            return np.nan
        return excess_returns.mean() / downside * np.sqrt(252)

    def _calculate_alpha_beta(self) -> Tuple[float, float]:
        """计算Alpha和Beta"""
        if self.benchmark is None:
            return np.nan, np.nan

        common_index = self.returns.index.intersection(self.benchmark.index)
        strategy_returns = self.returns[common_index]
        benchmark_returns = self.benchmark[common_index]

        beta, alpha, _, _, _ = stats.linregress(
            benchmark_returns,
            strategy_returns
        )
        annualized_alpha = (1 + alpha) ** 252 - 1
        return annualized_alpha, beta

    def _calculate_information_ratio(self) -> float:
        """计算信息比率"""
        if self.benchmark is None:
            return np.nan

        active_returns = self.returns - self.benchmark
        return active_returns.mean() / active_returns.std() * np.sqrt(252)

    def _calculate_tracking_error(self) -> float:
        """计算跟踪误差"""
        if self.benchmark is None:
            return np.nan

        active_returns = self.returns - self.benchmark
        return active_returns.std() * np.sqrt(252)

    def _calculate_outperformance(self) -> float:
        """计算超额收益率"""
        if self.benchmark is None:
            return np.nan

        total_return = self._calculate_total_return()
        benchmark_return = (1 + self.benchmark).prod() - 1
        return total_return - benchmark_return

    def _calculate_skewness(self) -> float:
        """计算收益偏度"""
        return self.returns.skew()

    def _calculate_kurtosis(self) -> float:
        """计算收益峰度"""
        return self.returns.kurtosis()

    def _calculate_var(self, confidence_level: float = 0.95) -> float:
        """计算VaR"""
        return np.percentile(self.returns, (1 - confidence_level) * 100)

    def _calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """计算CVaR"""
        var = self._calculate_var(confidence_level)
        return self.returns[self.returns <= var].mean()

    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        return (self.returns > 0).mean()

    def _calculate_profit_factor(self) -> float:
        """计算盈利因子"""
        gross_profit = self.returns[self.returns > 0].sum()
        gross_loss = abs(self.returns[self.returns < 0].sum())
        if gross_loss == 0:
            return np.inf
        return gross_profit / gross_loss

    def plot_performance(self, figsize: tuple = (12, 8)) -> plt.Figure:
        """绘制绩效分析图"""
        if self._results is None:
            self.analyze()

        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # 累计收益曲线
        cum_returns = (1 + self.returns).cumprod()
        axes[0].plot(cum_returns, label='Strategy')
        if self.benchmark is not None:
            cum_bench = (1 + self.benchmark).cumprod()
            axes[0].plot(cum_bench, label='Benchmark')
        axes[0].set_title('Cumulative Returns')
        axes[0].legend()

        # 回撤曲线
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns - peak) / peak
        axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.3)
        axes[1].set_title('Drawdown')

        # 月度收益热力图
        monthly_returns = self.returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly_returns = monthly_returns.unstack().T
        sns.heatmap(
            monthly_returns,
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            ax=axes[2]
        )
        axes[2].set_title('Monthly Returns Heatmap')

        plt.tight_layout()
        return fig

    def generate_report(self) -> Dict[str, Dict[str, float]]:
        """生成结构化报告"""
        if self._results is None:
            self.analyze()

        report = {
            'return_metrics': {
                'Total Return': self._results['total_return'],
                'Annual Return': self._results['annual_return'],
                'Win Rate': self._results['win_rate']
            },
            'risk_metrics': {
                'Annual Volatility': self._results['annual_volatility'],
                'Max Drawdown': self._results['max_drawdown'],
                'VaR (95%)': self._results['var_95'],
                'CVaR (95%)': self._results['cvar_95']
            },
            'ratio_metrics': {
                'Sharpe Ratio': self._results['sharpe_ratio'],
                'Sortino Ratio': self._results['sortino_ratio'],
                'Calmar Ratio': self._results['calmar_ratio']
            }
        }

        if self.benchmark is not None:
            report['benchmark_metrics'] = {
                'Alpha': self._results['alpha'],
                'Beta': self._results['beta'],
                'Information Ratio': self._results['information_ratio'],
                'Tracking Error': self._results['tracking_error'],
                'Outperformance': self._results['outperformance']
            }

        return report
