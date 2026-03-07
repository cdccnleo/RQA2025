import logging
#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
回测性能分析器
提供完整的绩效指标计算和风险分析功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:

    """绩效指标数据类"""
    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    cumulative_return: float = 0.0

    # 风险指标
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # 回撤指标
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    current_drawdown: float = 0.0

    # 交易指标
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0

    # 其他指标
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0


class PerformanceAnalyzer:

    """回测绩效分析器"""

    def __init__(self, risk_free_rate: float = 0.03, benchmark_returns: Optional[pd.Series] = None):
        """
        初始化绩效分析器

        Args:
            risk_free_rate: 无风险利率
            benchmark_returns: 基准收益率序列
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        self.metrics_history = []

    def analyze_performance(self, returns: pd.Series, trades: Optional[pd.DataFrame] = None) -> PerformanceMetrics:
        """
        分析绩效指标

        Args:
            returns: 收益率序列
            trades: 交易记录DataFrame

        Returns:
            PerformanceMetrics: 绩效指标
        """
        if returns.empty:
            logger.warning("收益率序列为空，返回默认指标")
            return PerformanceMetrics()

        try:
            metrics = PerformanceMetrics()

            # 计算基础收益指标
            metrics = self._calculate_return_metrics(returns, metrics)

            # 计算风险指标
            metrics = self._calculate_risk_metrics(returns, metrics)

            # 计算回撤指标
            metrics = self._calculate_drawdown_metrics(returns, metrics)

            # 计算交易指标
            if trades is not None and not trades.empty:
                metrics = self._calculate_trade_metrics(trades, metrics)

            # 计算相对指标
            if self.benchmark_returns is not None:
                metrics = self._calculate_relative_metrics(returns, metrics)

            # 记录指标历史
            self.metrics_history.append(metrics)

            logger.info(f"绩效分析完成，总收益率: {metrics.total_return:.2%}")
            return metrics

        except Exception as e:
            logger.error(f"绩效分析失败: {str(e)}")
            return PerformanceMetrics()

    def _calculate_return_metrics(self, returns: pd.Series, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """计算收益指标"""
        # 总收益率
        metrics.total_return = (1 + returns).prod() - 1

        # 年化收益率
        periods_per_year = 252  # 假设交易日
        metrics.annual_return = (1 + metrics.total_return) ** (periods_per_year / len(returns)) - 1

        # 累计收益率
        metrics.cumulative_return = (1 + returns).cumprod().iloc[-1] - 1

        return metrics

    def _calculate_risk_metrics(self, returns: pd.Series, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """计算风险指标"""
        # 波动率
        metrics.volatility = returns.std() * np.sqrt(252)

        # 夏普比率
        excess_returns = returns - self.risk_free_rate / 252
        if metrics.volatility > 0:
            metrics.sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)

        # 索提诺比率
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std()
        if downside_deviation > 0:
            metrics.sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252)

        # 卡玛比率
        if metrics.max_drawdown != 0:
            metrics.calmar_ratio = metrics.annual_return / abs(metrics.max_drawdown)

        return metrics

    def _calculate_drawdown_metrics(self, returns: pd.Series, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """计算回撤指标"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        # 最大回撤
        metrics.max_drawdown = drawdown.min()

        # 最大回撤持续时间
        max_dd_idx = drawdown.idxmin()
        if max_dd_idx is not None:
            recovery_idx = drawdown[max_dd_idx:].where(drawdown >= 0).first_valid_index()
            if recovery_idx is not None:
                # 处理索引类型，可能是整数或日期
                if hasattr(recovery_idx, 'days'):
                    metrics.max_drawdown_duration = (recovery_idx - max_dd_idx).days
                else:
                    metrics.max_drawdown_duration = recovery_idx - max_dd_idx
            else:
                # 处理索引类型，可能是整数或日期
                if hasattr(drawdown.index[-1], 'days'):
                    metrics.max_drawdown_duration = (drawdown.index[-1] - max_dd_idx).days
                else:
                    metrics.max_drawdown_duration = drawdown.index[-1] - max_dd_idx

        # 当前回撤
        metrics.current_drawdown = drawdown.iloc[-1]

        return metrics

    def _calculate_trade_metrics(self, trades: pd.DataFrame, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """计算交易指标"""
        if 'profit' not in trades.columns:
            return metrics

        profits = trades['profit']
        winning_trades = profits[profits > 0]
        losing_trades = profits[profits < 0]

        # 胜率
        if len(profits) > 0:
            metrics.win_rate = len(winning_trades) / len(profits)

        # 盈亏比
        if len(losing_trades) > 0 and losing_trades.sum() != 0:
            metrics.profit_factor = winning_trades.sum() / abs(losing_trades.sum())

        # 平均盈利
        if len(winning_trades) > 0:
            metrics.average_win = winning_trades.mean()

        # 平均亏损
        if len(losing_trades) > 0:
            metrics.average_loss = losing_trades.mean()

        return metrics

    def _calculate_relative_metrics(self, returns: pd.Series, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """计算相对指标"""
        if self.benchmark_returns is None:
            return metrics

        # 确保时间对齐
        aligned_returns = returns.align(self.benchmark_returns, join='inner')[0]
        aligned_benchmark = returns.align(self.benchmark_returns, join='inner')[1]

        if len(aligned_returns) == 0:
            return metrics

        # Beta系数
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        if benchmark_variance > 0:
            metrics.beta = covariance / benchmark_variance

        # Alpha系数
        benchmark_return = aligned_benchmark.mean() * 252
        if metrics.beta != 0:
            metrics.alpha = (aligned_returns.mean() * 252) - (self.risk_free_rate +
                                                              metrics.beta * (benchmark_return - self.risk_free_rate))

        # 信息比率
        tracking_error = (aligned_returns - aligned_benchmark).std()
        if tracking_error > 0:
            metrics.information_ratio = (aligned_returns.mean(
            ) - aligned_benchmark.mean()) / tracking_error * np.sqrt(252)

        # 特雷诺比率
        if metrics.beta > 0:
            metrics.treynor_ratio = (aligned_returns.mean() * 252 -
                                     self.risk_free_rate) / metrics.beta

        return metrics

    def get_summary_report(self) -> Dict[str, Any]:
        """生成绩效总结报告"""
        if not self.metrics_history:
            return {}

        latest_metrics = self.metrics_history[-1]

        return {
            'summary': {
                'total_return': f"{latest_metrics.total_return:.2%}",
                'annual_return': f"{latest_metrics.annual_return:.2%}",
                'sharpe_ratio': f"{latest_metrics.sharpe_ratio:.2f}",
                'max_drawdown': f"{latest_metrics.max_drawdown:.2%}",
                'win_rate': f"{latest_metrics.win_rate:.2%}"
            },
            'risk_metrics': {
                'volatility': f"{latest_metrics.volatility:.2%}",
                'sortino_ratio': f"{latest_metrics.sortino_ratio:.2f}",
                'calmar_ratio': f"{latest_metrics.calmar_ratio:.2f}"
            },
            'trade_metrics': {
                'profit_factor': f"{latest_metrics.profit_factor:.2f}",
                'average_win': f"{latest_metrics.average_win:.2f}",
                'average_loss': f"{latest_metrics.average_loss:.2f}"
            }
        }

    def analyze(self, returns):
        """向后兼容的简化接口"""
        if isinstance(returns, pd.Series):
            metrics = self.analyze_performance(returns)
            return {
                'total_return': metrics.total_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown
            }
        else:
            # 处理列表或其他格式
            returns_series = pd.Series(returns)
            return self.analyze(returns_series)
