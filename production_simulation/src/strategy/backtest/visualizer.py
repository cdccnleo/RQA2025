import logging
"""回测结果可视化模块

提供回测结果的可视化功能，包括：
- 绩效曲线绘制
- 收益分布分析
- 风险指标可视化
- 交易记录展示
- 交互式图表
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


logger = logging.getLogger(__name__)


class Plotter:

    """回测结果可视化器"""

    def __init__(self,


                 output_dir: str = "output / backtest",
                 style: str = "default",
                 figsize: tuple = (12, 8),
                 use_plotly: bool = True):
        """
        初始化可视化器

        Args:
            output_dir: 输出目录
            style: 绘图样式
            figsize: 图形大小
            use_plotly: 是否使用plotly进行交互式绘图
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        self.figsize = figsize
        self.use_plotly = use_plotly

        # 设置绘图样式
        plt.style.use(style)
        sns.set_palette("husl")

        logger.info(f"可视化器初始化完成，输出目录: {self.output_dir}")

    def plot_performance(self,


                         results: Dict[str, Any],
                         save: bool = True,
                         filename: Optional[str] = None,
                         interactive: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        绘制绩效曲线

        Args:
            results: 回测结果字典
            save: 是否保存图片
            filename: 保存文件名
            interactive: 是否生成交互式图表

        Returns:
            matplotlib.Figure 或 plotly.graph_objects.Figure
        """
        if not results or 'returns' not in results:
            logger.warning("回测结果为空或缺少收益率数据")
            return None

        returns = results['returns']
        cumulative_returns = (1 + returns).cumprod()

        if interactive and self.use_plotly:
            return self._plot_performance_plotly(cumulative_returns, results, save, filename)
        else:
            return self._plot_performance_matplotlib(cumulative_returns, results, save, filename)

    def _plot_performance_plotly(self, cumulative_returns: pd.Series,


                                 results: Dict[str, Any],
                                 save: bool, filename: Optional[str]) -> go.Figure:
        """使用plotly绘制绩效曲线"""
        fig = go.Figure()

        # 添加绩效曲线
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='策略收益',
            line=dict(color='blue', width=2)
        ))

        # 添加基准线（如果有）
        if 'benchmark' in results:
            benchmark_returns = results['benchmark']
            benchmark_cum = (1 + benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=benchmark_cum.index,
                y=benchmark_cum.values,
                mode='lines',
                name='基准收益',
                line=dict(color='red', width=2, dash='dash')
            ))

        # 更新布局
        fig.update_layout(
            title='策略绩效曲线',
            xaxis_title='日期',
            yaxis_title='累计收益',
            hovermode='x unified',
            template='plotly_white'
        )

        if save:
            filename = filename or f"performance_plotly_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            logger.info(f"交互式绩效图表已保存到: {filepath}")

        return fig

    def _plot_performance_matplotlib(self, cumulative_returns: pd.Series,


                                     results: Dict[str, Any],
                                     save: bool, filename: Optional[str]) -> plt.Figure:
        """使用matplotlib绘制绩效曲线"""
        fig, ax = plt.subplots(figsize=self.figsize)

        # 绘制绩效曲线
        ax.plot(cumulative_returns.index, cumulative_returns.values,
                label='策略收益', linewidth=2, color='blue')

        # 添加基准线（如果有）
        if 'benchmark' in results:
            benchmark_returns = results['benchmark']
            benchmark_cum = (1 + benchmark_returns).cumprod()
            ax.plot(benchmark_cum.index, benchmark_cum.values,
                    label='基准收益', linewidth=2, color='red', linestyle='--')

        ax.set_title('策略绩效曲线', fontsize=16)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('累计收益', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save:
            filename = filename or f"performance_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.png"
            filepath = self.output_dir / filename
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
            logger.info(f"绩效图表已保存到: {filepath}")

        return fig

    def plot_returns_distribution(self,


                                  results: Dict[str, Any],
                                  save: bool = True,
                                  filename: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """绘制收益分布图"""
        if not results or 'returns' not in results:
            logger.warning("回测结果为空或缺少收益率数据")
            return None

        returns = results['returns']

        if self.use_plotly:
            return self._plot_returns_distribution_plotly(returns, save, filename)
        else:
            return self._plot_returns_distribution_matplotlib(returns, save, filename)

    def _plot_returns_distribution_plotly(self, returns: pd.Series,


                                          save: bool, filename: Optional[str]) -> go.Figure:
        """使用plotly绘制收益分布"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('收益分布直方图', '收益Q - Q图'),
            vertical_spacing=0.1
        )

        # 直方图
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='收益分布'),
            row=1, col=1
        )

        # Q - Q图
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        sample_quantiles = np.sort(returns)

        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                       mode='markers', name='Q - Q图'),
            row=2, col=1
        )

        # 添加对角线
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                       mode='lines', name='理论正态分布',
                       line=dict(dash='dash')),
            row=2, col=1
        )

        fig.update_layout(
            title='收益分布分析',
            height=800,
            template='plotly_white'
        )

        if save:
            filename = filename or f"returns_distribution_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            logger.info(f"收益分布图表已保存到: {filepath}")

        return fig

    def _plot_returns_distribution_matplotlib(self, returns: pd.Series,


                                              save: bool, filename: Optional[str]) -> plt.Figure:
        """使用matplotlib绘制收益分布"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 直方图
        ax1.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('收益分布直方图', fontsize=14)
        ax1.set_xlabel('收益率', fontsize=12)
        ax1.set_ylabel('频次', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Q - Q图
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('收益Q - Q图', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = filename or f"returns_distribution_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.png"
            filepath = self.output_dir / filename
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
            logger.info(f"收益分布图表已保存到: {filepath}")

        return fig

    def plot_trade_analysis(self,


                            results: Dict[str, Any],
                            save: bool = True,
                            filename: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """绘制交易分析图"""
        if not results or 'trades' not in results:
            logger.warning("回测结果为空或缺少交易数据")
            return None

        trades = results['trades']

        if self.use_plotly:
            return self._plot_trade_analysis_plotly(trades, save, filename)
        else:
            return self._plot_trade_analysis_matplotlib(trades, save, filename)

    def _plot_trade_analysis_plotly(self, trades: pd.DataFrame,


                                    save: bool, filename: Optional[str]) -> go.Figure:
        """使用plotly绘制交易分析"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('交易盈亏分布', '交易时间分布', '持仓时间分布', '交易规模分布'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )

        # 交易盈亏分布
        if 'profit' in trades.columns:
            fig.add_trace(
                go.Histogram(x=trades['profit'], nbinsx=30, name='交易盈亏'),
                row=1, col=1
            )

        # 交易时间分布
        if 'entry_time' in trades.columns:
            entry_times = pd.to_datetime(trades['entry_time'])
            fig.add_trace(
                go.Scatter(x=entry_times, y=trades.get('profit', range(len(trades))),
                           mode='markers', name='交易时间'),
                row=1, col=2
            )

        # 持仓时间分布
        if 'hold_days' in trades.columns:
            fig.add_trace(
                go.Histogram(x=trades['hold_days'], nbinsx=20, name='持仓时间'),
                row=2, col=1
            )

        # 交易规模分布
        if 'volume' in trades.columns:
            fig.add_trace(
                go.Scatter(x=trades['volume'], y=trades.get('profit', range(len(trades))),
                           mode='markers', name='交易规模'),
                row=2, col=2
            )

        fig.update_layout(
            title='交易分析',
            height=800,
            template='plotly_white'
        )

        if save:
            filename = filename or f"trade_analysis_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            logger.info(f"交易分析图表已保存到: {filepath}")

        return fig

    def _plot_trade_analysis_matplotlib(self, trades: pd.DataFrame,


                                        save: bool, filename: Optional[str]) -> plt.Figure:
        """使用matplotlib绘制交易分析"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 交易盈亏分布
        if 'profit' in trades.columns:
            ax1.hist(trades['profit'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            ax1.set_title('交易盈亏分布', fontsize=12)
            ax1.set_xlabel('盈亏', fontsize=10)
            ax1.set_ylabel('频次', fontsize=10)
            ax1.grid(True, alpha=0.3)

        # 交易时间分布
        if 'entry_time' in trades.columns:
            entry_times = pd.to_datetime(trades['entry_time'])
            ax2.scatter(entry_times, trades.get('profit', range(len(trades))), alpha=0.6)
            ax2.set_title('交易时间分布', fontsize=12)
            ax2.set_xlabel('时间', fontsize=10)
            ax2.set_ylabel('盈亏', fontsize=10)
            ax2.grid(True, alpha=0.3)

        # 持仓时间分布
        if 'hold_days' in trades.columns:
            ax3.hist(trades['hold_days'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax3.set_title('持仓时间分布', fontsize=12)
            ax3.set_xlabel('持仓天数', fontsize=10)
            ax3.set_ylabel('频次', fontsize=10)
            ax3.grid(True, alpha=0.3)

        # 交易规模分布
        if 'volume' in trades.columns:
            ax4.scatter(trades['volume'], trades.get('profit', range(len(trades))), alpha=0.6)
            ax4.set_title('交易规模分布', fontsize=12)
            ax4.set_xlabel('交易量', fontsize=10)
            ax4.set_ylabel('盈亏', fontsize=10)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = filename or f"trade_analysis_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.png"
            filepath = self.output_dir / filename
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
            logger.info(f"交易分析图表已保存到: {filepath}")

        return fig

    def plot_risk_metrics(self,


                          results: Dict[str, Any],
                          save: bool = True,
                          filename: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """绘制风险指标图"""
        if not results or 'returns' not in results:
            logger.warning("回测结果为空或缺少收益率数据")
            return None

        returns = results['returns']

        if self.use_plotly:
            return self._plot_risk_metrics_plotly(returns, save, filename)
        else:
            return self._plot_risk_metrics_matplotlib(returns, save, filename)

    def _plot_risk_metrics_plotly(self, returns: pd.Series,


                                  save: bool, filename: Optional[str]) -> go.Figure:
        """使用plotly绘制风险指标"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('滚动波动率', '滚动夏普比率', '回撤曲线', 'VaR分析'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )

        # 滚动波动率
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                       mode='lines', name='滚动波动率'),
            row=1, col=1
        )

        # 滚动夏普比率
        rolling_sharpe = returns.rolling(window=20).mean(
        ) / returns.rolling(window=20).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                       mode='lines', name='滚动夏普比率'),
            row=1, col=2
        )

        # 回撤曲线
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                       mode='lines', name='回撤', fill='tonexty'),
            row=2, col=1
        )

        # VaR分析
        var_95 = returns.rolling(window=20).quantile(0.05)
        fig.add_trace(
            go.Scatter(x=var_95.index, y=var_95.values,
                       mode='lines', name='95% VaR'),
            row=2, col=2
        )

        fig.update_layout(
            title='风险指标分析',
            height=800,
            template='plotly_white'
        )

        if save:
            filename = filename or f"risk_metrics_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            logger.info(f"风险指标图表已保存到: {filepath}")

        return fig

    def _plot_risk_metrics_matplotlib(self, returns: pd.Series,


                                      save: bool, filename: Optional[str]) -> plt.Figure:
        """使用matplotlib绘制风险指标"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 滚动波动率
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        ax1.plot(rolling_vol.index, rolling_vol.values, linewidth=2)
        ax1.set_title('滚动波动率', fontsize=12)
        ax1.set_ylabel('波动率', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 滚动夏普比率
        rolling_sharpe = returns.rolling(window=20).mean(
        ) / returns.rolling(window=20).std() * np.sqrt(252)
        ax2.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        ax2.set_title('滚动夏普比率', fontsize=12)
        ax2.set_ylabel('夏普比率', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 回撤曲线
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax3.plot(drawdown.index, drawdown.values, linewidth=2, color='red')
        ax3.set_title('回撤曲线', fontsize=12)
        ax3.set_ylabel('回撤', fontsize=10)
        ax3.grid(True, alpha=0.3)

        # VaR分析
        var_95 = returns.rolling(window=20).quantile(0.05)
        ax4.plot(var_95.index, var_95.values, linewidth=2, color='orange')
        ax4.set_title('95% VaR', fontsize=12)
        ax4.set_ylabel('VaR', fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = filename or f"risk_metrics_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.png"
            filepath = self.output_dir / filename
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
            logger.info(f"风险指标图表已保存到: {filepath}")

        return fig

    def generate_report(self,


                        results: Dict[str, Any],
                        save: bool = True,
                        filename: Optional[str] = None) -> str:
        """生成完整的可视化报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("回测可视化报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"输出目录: {self.output_dir}")

        # 生成各种图表
        charts_generated = []

        # 绩效曲线
        try:
            perf_fig = self.plot_performance(results, save=True,
                                             filename=f"performance_{datetime.now().strftime('%Y % m % d_ % H % M % S')}")
            if perf_fig:
                charts_generated.append("绩效曲线")
        except Exception as e:
            logger.error(f"生成绩效曲线失败: {str(e)}")

        # 收益分布
        try:
            dist_fig = self.plot_returns_distribution(results, save=True,
                                                      filename=f"distribution_{datetime.now().strftime('%Y % m % d_ % H % M % S')}")
            if dist_fig:
                charts_generated.append("收益分布")
        except Exception as e:
            logger.error(f"生成收益分布失败: {str(e)}")

        # 交易分析
        try:
            trade_fig = self.plot_trade_analysis(results, save=True,
                                                 filename=f"trade_analysis_{datetime.now().strftime('%Y % m % d_ % H % M % S')}")
            if trade_fig:
                charts_generated.append("交易分析")
        except Exception as e:
            logger.error(f"生成交易分析失败: {str(e)}")

        # 风险指标
        try:
            risk_fig = self.plot_risk_metrics(results, save=True,
                                              filename=f"risk_metrics_{datetime.now().strftime('%Y % m % d_ % H % M % S')}")
            if risk_fig:
                charts_generated.append("风险指标")
        except Exception as e:
            logger.error(f"生成风险指标失败: {str(e)}")

        # 报告总结
        report_lines.append(f"\n生成的图表:")
        for chart in charts_generated:
            report_lines.append(f"- {chart}")

        report_lines.append(f"\n图表文件保存在: {self.output_dir}")
        report_lines.append("=" * 60)

        report_content = "\n".join(report_lines)

        if save:
            filename = filename or f"visualization_report_{datetime.now().strftime('%Y % m % d_ % H % M % S')}.txt"
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf - 8') as f:
                f.write(report_content)
            logger.info(f"可视化报告已保存到: {filepath}")

        return report_content
