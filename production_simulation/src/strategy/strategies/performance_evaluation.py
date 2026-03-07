#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略性能评估系统
提供多维度性能指标计算、基准对比、风险分析和报告生成功能
"""

import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import plotly.graph_objects as go


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:

    """性能指标"""
    # 收益指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # 风险指标
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    downside_deviation: float = 0.0

    # 交易指标
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # 基准对比指标
    alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    jensen_alpha: float = 0.0

    # 其他指标
    skewness: float = 0.0
    kurtosis: float = 0.0
    value_at_risk: float = 0.0
    conditional_var: float = 0.0


@dataclass
class EvaluationConfig:

    """评估配置"""
    evaluation_id: str
    benchmark_returns: Optional[pd.Series] = None
    risk_free_rate: float = 0.03
    trading_days_per_year: int = 252
    confidence_level: float = 0.95
    min_periods: int = 30
    performance_window: int = 252
    rebalance_frequency: str = "daily"
    transaction_costs: float = 0.001
    slippage: float = 0.0005


@dataclass
class EvaluationResult:

    """评估结果"""
    strategy_id: str
    evaluation_id: str
    timestamp: datetime
    metrics: PerformanceMetrics
    benchmark_comparison: Dict[str, float]
    risk_analysis: Dict[str, float]
    performance_attribution: Dict[str, float]
    recommendations: List[str]


class StrategyPerformanceEvaluator:

    """策略性能评估器"""

    def __init__(self, config: EvaluationConfig):
        """
        初始化策略性能评估器

        Args:
            config: 评估配置
        """
        self.config = config
        self.evaluation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_cache: Dict[str, PerformanceMetrics] = {}
        self.lock = threading.RLock()

        # 评估组件
        self.return_calculator = ReturnCalculator()
        self.risk_analyzer = RiskAnalyzer()
        self.benchmark_comparator = BenchmarkComparator()
        self.performance_attributor = PerformanceAttributor()

        logger.info(f"策略性能评估器初始化完成: {config.evaluation_id}")

    def evaluate_strategy(self, strategy_id: str, returns: pd.Series,


                          positions: Optional[pd.DataFrame] = None,
                          trades: Optional[pd.DataFrame] = None) -> EvaluationResult:
        """
        评估策略性能

        Args:
            strategy_id: 策略ID
            returns: 收益率序列
            positions: 持仓数据（可选）
            trades: 交易数据（可选）

        Returns:
            EvaluationResult: 评估结果
        """
        with self.lock:
            if returns.empty:
                raise ValueError("收益率数据为空")

            # 计算基础性能指标
            metrics = self._calculate_performance_metrics(returns)

            # 基准对比分析
            benchmark_comparison = {}
            if self.config.benchmark_returns is not None:
                benchmark_comparison = self.benchmark_comparator.compare(
                    returns, self.config.benchmark_returns
                )

            # 风险分析
            risk_analysis = self.risk_analyzer.analyze_risk(returns)

            # 性能归因分析
            performance_attribution = {}
            if positions is not None:
                performance_attribution = self.performance_attributor.attribute_performance(
                    returns, positions, self.config.benchmark_returns
                )

            # 生成建议
            recommendations = self._generate_recommendations(metrics, risk_analysis)

            # 创建评估结果
            result = EvaluationResult(
                strategy_id=strategy_id,
                evaluation_id=self.config.evaluation_id,
                timestamp=datetime.now(),
                metrics=metrics,
                benchmark_comparison=benchmark_comparison,
                risk_analysis=risk_analysis,
                performance_attribution=performance_attribution,
                recommendations=recommendations
            )

            # 缓存结果
            self.performance_cache[strategy_id] = metrics
            self.evaluation_history[strategy_id].append(result)

            logger.info(f"策略 {strategy_id} 性能评估完成")
            return result

    def _calculate_performance_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """
        计算性能指标

        Args:
            returns: 收益率序列

        Returns:
            PerformanceMetrics: 性能指标
        """
        metrics = PerformanceMetrics()

        # 基础收益指标
        metrics.total_return = self.return_calculator.calculate_total_return(returns)
        metrics.annualized_return = self.return_calculator.calculate_annualized_return(
            returns, self.config.trading_days_per_year
        )

        # 风险调整收益指标
        metrics.sharpe_ratio = self.return_calculator.calculate_sharpe_ratio(
            returns, self.config.risk_free_rate, self.config.trading_days_per_year
        )
        metrics.sortino_ratio = self.return_calculator.calculate_sortino_ratio(
            returns, self.config.risk_free_rate, self.config.trading_days_per_year
        )

        # 风险指标
        metrics.volatility = returns.std() * np.sqrt(self.config.trading_days_per_year)
        metrics.max_drawdown = self.risk_analyzer.calculate_max_drawdown(returns)
        metrics.var_95 = self.risk_analyzer.calculate_var(returns, 0.95)
        metrics.cvar_95 = self.risk_analyzer.calculate_cvar(returns, 0.95)
        metrics.downside_deviation = self.risk_analyzer.calculate_downside_deviation(returns)

        # 交易指标
        if len(returns) > 0:
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            metrics.win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
            metrics.profit_factor = abs(positive_returns.sum(
            ) / negative_returns.sum()) if len(negative_returns) > 0 else float('inf')
            metrics.average_win = positive_returns.mean() if len(positive_returns) > 0 else 0.0
            metrics.average_loss = negative_returns.mean() if len(negative_returns) > 0 else 0.0

        # 统计指标
        metrics.skewness = returns.skew()
        metrics.kurtosis = returns.kurtosis()

        return metrics

    def _generate_recommendations(self, metrics: PerformanceMetrics,


                                  risk_analysis: Dict[str, float]) -> List[str]:
        """
        生成建议

        Args:
            metrics: 性能指标
            risk_analysis: 风险分析结果

        Returns:
            List[str]: 建议列表
        """
        recommendations = []

        # 基于夏普比率的建议
        if metrics.sharpe_ratio < 1.0:
            recommendations.append("夏普比率较低，建议优化风险调整收益")
        elif metrics.sharpe_ratio > 2.0:
            recommendations.append("夏普比率优秀，可考虑增加杠杆")

        # 基于最大回撤的建议
        if metrics.max_drawdown > 0.2:
            recommendations.append("最大回撤过大，建议加强风险控制")

        # 基于胜率的建议
        if metrics.win_rate < 0.4:
            recommendations.append("胜率较低，建议优化交易策略")

        # 基于波动率的建议
        if metrics.volatility > 0.3:
            recommendations.append("波动率较高，建议分散投资")

        # 基于VaR的建议
        if metrics.var_95 < -0.05:
            recommendations.append("VaR过高，建议降低风险敞口")

        return recommendations

    def compare_strategies(self, strategy_results: Dict[str, EvaluationResult]) -> pd.DataFrame:
        """
        比较多个策略

        Args:
            strategy_results: 策略评估结果字典

        Returns:
            pd.DataFrame: 比较结果
        """
        comparison_data = []

        for strategy_id, result in strategy_results.items():
            metrics = result.metrics
            row = {
                'strategy_id': strategy_id,
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'volatility': metrics.volatility,
                'win_rate': metrics.win_rate,
                'var_95': metrics.var_95
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('sharpe_ratio', ascending=False)

        return df

    def generate_performance_report(self, strategy_id: str,


                                    result: EvaluationResult) -> str:
        """
        生成性能报告

        Args:
            strategy_id: 策略ID
            result: 评估结果

        Returns:
            str: 报告内容
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"策略性能评估报告 - {strategy_id}")
        report_lines.append("=" * 60)
        report_lines.append(f"评估时间: {result.timestamp}")
        report_lines.append(f"评估ID: {result.evaluation_id}")
        report_lines.append("")

        # 收益指标
        report_lines.append("收益指标:")
        report_lines.append("-" * 20)
        report_lines.append(f"总收益率: {result.metrics.total_return:.2%}")
        report_lines.append(f"年化收益率: {result.metrics.annualized_return:.2%}")
        report_lines.append(f"夏普比率: {result.metrics.sharpe_ratio:.3f}")
        report_lines.append(f"索提诺比率: {result.metrics.sortino_ratio:.3f}")
        report_lines.append("")

        # 风险指标
        report_lines.append("风险指标:")
        report_lines.append("-" * 20)
        report_lines.append(f"年化波动率: {result.metrics.volatility:.2%}")
        report_lines.append(f"最大回撤: {result.metrics.max_drawdown:.2%}")
        report_lines.append(f"VaR(95%): {result.metrics.var_95:.2%}")
        report_lines.append(f"CVaR(95%): {result.metrics.cvar_95:.2%}")
        report_lines.append("")

        # 交易指标
        report_lines.append("交易指标:")
        report_lines.append("-" * 20)
        report_lines.append(f"胜率: {result.metrics.win_rate:.2%}")
        report_lines.append(f"盈亏比: {result.metrics.profit_factor:.3f}")
        report_lines.append(f"平均盈利: {result.metrics.average_win:.2%}")
        report_lines.append(f"平均亏损: {result.metrics.average_loss:.2%}")
        report_lines.append("")

        # 基准对比
        if result.benchmark_comparison:
            report_lines.append("基准对比:")
            report_lines.append("-" * 20)
            for metric, value in result.benchmark_comparison.items():
                report_lines.append(f"{metric}: {value:.3f}")
            report_lines.append("")

        # 建议
        if result.recommendations:
            report_lines.append("建议:")
            report_lines.append("-" * 20)
            for i, recommendation in enumerate(result.recommendations, 1):
                report_lines.append(f"{i}. {recommendation}")

        return "\n".join(report_lines)

    def plot_performance_charts(self, strategy_id: str, returns: pd.Series,


                                benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        绘制性能图表

        Args:
            strategy_id: 策略ID
            returns: 收益率序列
            benchmark_returns: 基准收益率序列（可选）

        Returns:
            Dict[str, Any]: 图表数据
        """
        charts = {}

        # 累积收益曲线
        cumulative_returns = (1 + returns).cumprod()
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name=f'{strategy_id} 累积收益',
            line=dict(color='blue')
        ))

        if benchmark_returns is not None:
            cumulative_benchmark = (1 + benchmark_returns).cumprod()
            fig_cumulative.add_trace(go.Scatter(
                x=cumulative_benchmark.index,
                y=cumulative_benchmark.values,
                mode='lines',
                name='基准累积收益',
                line=dict(color='red', dash='dash')
            ))

        fig_cumulative.update_layout(
            title=f'{strategy_id} 累积收益曲线',
            xaxis_title='日期',
            yaxis_title='累积收益',
            hovermode='x unified'
        )

        charts['cumulative_returns'] = fig_cumulative

        # 回撤曲线
        drawdown = self.risk_analyzer.calculate_drawdown_series(returns)
        fig_drawdown = go.Figure()
        fig_drawdown.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            name='回撤',
            fill='tonexty',
            line=dict(color='red')
        ))

        fig_drawdown.update_layout(
            title=f'{strategy_id} 回撤曲线',
            xaxis_title='日期',
            yaxis_title='回撤 (%)',
            hovermode='x unified'
        )

        charts['drawdown'] = fig_drawdown

        # 收益率分布
        fig_distribution = go.Figure()
        fig_distribution.add_trace(go.Histogram(
            x=returns.values * 100,
            nbinsx=50,
            name='收益率分布',
            marker_color='lightblue'
        ))

        fig_distribution.update_layout(
            title=f'{strategy_id} 收益率分布',
            xaxis_title='收益率 (%)',
            yaxis_title='频次',
            showlegend=False
        )

        charts['returns_distribution'] = fig_distribution

        return charts


class ReturnCalculator:

    """收益率计算器"""

    def calculate_total_return(self, returns: pd.Series) -> float:
        """计算总收益率"""
        return (1 + returns).prod() - 1

    def calculate_annualized_return(self, returns: pd.Series, trading_days: int) -> float:
        """计算年化收益率"""
        total_return = self.calculate_total_return(returns)
        years = len(returns) / trading_days
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float,


                               trading_days: int) -> float:
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate / trading_days
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)

    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float,


                                trading_days: int) -> float:
        """计算索提诺比率"""
        excess_returns = returns - risk_free_rate / trading_days
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / downside_returns.std() * np.sqrt(trading_days)


class RiskAnalyzer:

    """风险分析器"""

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return abs(drawdown.min())

    def calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """计算回撤序列"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown

    def calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """计算VaR"""
        return np.percentile(returns, (1 - confidence_level) * 100)

    def calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """计算CVaR"""
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()

    def calculate_downside_deviation(self, returns: pd.Series) -> float:
        """计算下行偏差"""
        downside_returns = returns[returns < 0]
        return downside_returns.std() if len(downside_returns) > 0 else 0.0

    def analyze_risk(self, returns: pd.Series) -> Dict[str, float]:
        """分析风险"""
        risk_metrics = {
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'var_95': self.calculate_var(returns, 0.95),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'downside_deviation': self.calculate_downside_deviation(returns),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        return risk_metrics


class BenchmarkComparator:

    """基准比较器"""

    def compare(self, strategy_returns: pd.Series,


                benchmark_returns: pd.Series) -> Dict[str, float]:
        """比较策略与基准"""
        # 确保时间对齐
        aligned_returns = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        if len(aligned_returns) == 0:
            return {}

        strategy_ret = aligned_returns.iloc[:, 0]
        benchmark_ret = aligned_returns.iloc[:, 1]

        # 计算超额收益
        excess_returns = strategy_ret - benchmark_ret

        # 计算比较指标
        comparison = {
            'alpha': excess_returns.mean() * 252,
            'beta': self._calculate_beta(strategy_ret, benchmark_ret),
            'information_ratio': excess_returns.mean() / excess_returns.std() * np.sqrt(252),
            'tracking_error': excess_returns.std() * np.sqrt(252),
            'correlation': strategy_ret.corr(benchmark_ret)
        }

        return comparison

    def _calculate_beta(self, strategy_returns: pd.Series,


                        benchmark_returns: pd.Series) -> float:
        """计算贝塔系数"""
        covariance = strategy_returns.cov(benchmark_returns)
        variance = benchmark_returns.var()
        return covariance / variance if variance != 0 else 0.0


class PerformanceAttributor:

    """性能归因分析器"""

    def attribute_performance(self, returns: pd.Series, positions: pd.DataFrame,


                              benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """归因分析"""
        attribution = {}

        # 计算各因子贡献
        if not positions.empty:
            # 这里可以添加更复杂的归因分析逻辑
            attribution['factor_contribution'] = 0.0
            attribution['timing_contribution'] = 0.0
            attribution['selection_contribution'] = 0.0

        return attribution
