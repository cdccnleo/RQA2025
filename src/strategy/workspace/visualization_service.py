#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化服务
Visualization Service

提供策略数据可视化功能，包括图表生成、数据分析和交互式界面。
"""

import json
from typing import Dict, List, Any
from datetime import datetime
import logging

import pandas as pd
import numpy as np

from ..interfaces.strategy_interfaces import StrategyResult
from strategy.interfaces.backtest_interfaces import BacktestResult
from strategy.interfaces.optimization_interfaces import OptimizationResult
from strategy.interfaces.monitoring_interfaces import MetricData
from core.integration.business_adapters import get_unified_adapter_factory

logger = logging.getLogger(__name__)


class VisualizationService:

    """
    可视化服务
    Visualization Service

    提供各种数据可视化功能和图表生成服务。
    """

    def __init__(self):
        """初始化可视化服务"""
        self.adapter_factory = get_unified_adapter_factory()

        # 图表配置
        self.chart_configs = self._load_chart_configs()

        # 颜色主题
        self.color_schemes = self._load_color_schemes()

        logger.info("可视化服务初始化完成")

    def _load_chart_configs(self) -> Dict[str, Any]:
        """
        加载图表配置

        Returns:
            Dict[str, Any]: 图表配置字典
        """
        return {
            'performance_chart': {
                'type': 'line',
                'title': '策略表现',
                'x_axis': '时间',
                'y_axis': '收益率',
                'colors': ['#667eea', '#764ba2', '#f093fb', '#f5576c']
            },
            'returns_distribution': {
                'type': 'histogram',
                'title': '收益率分布',
                'bins': 50,
                'colors': ['#4ecdc4', '#44a08d']
            },
            'drawdown_chart': {
                'type': 'area',
                'title': '最大回撤',
                'fill_color': '#ff6b6b',
                'line_color': '#ee5a24'
            },
            'correlation_matrix': {
                'type': 'heatmap',
                'title': '相关性矩阵',
                'colorscale': 'RdBu'
            },
            'optimization_surface': {
                'type': 'surface',
                'title': '优化参数曲面',
                'colorscale': 'Viridis'
            }
        }

    def _load_color_schemes(self) -> Dict[str, List[str]]:
        """
        加载颜色主题

        Returns:
            Dict[str, List[str]]: 颜色主题字典
        """
        return {
            'default': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4ecdc4', '#44a08d'],
            'blues': ['#1e3a8a', '#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe'],
            'greens': ['#166534', '#16a34a', '#4ade80', '#86efac', '#dcfce7'],
            'reds': ['#991b1b', '#dc2626', '#ef4444', '#fca5a5', '#fee2e2'],
            'purples': ['#581c87', '#7c3aed', '#a78bfa', '#c4b5fd', '#ede9fe']
        }

    async def generate_strategy_performance_chart(self, strategy_result: StrategyResult,
                                                  chart_type: str = 'line') -> Dict[str, Any]:
        """
        生成策略表现图表

        Args:
            strategy_result: 策略执行结果
            chart_type: 图表类型 ('line', 'bar', 'area')

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            # 提取时间序列数据
            timestamps = []
            returns = []
            cumulative_returns = []

            current_cumulative = 0.0
            for i, signal in enumerate(strategy_result.signals):
                timestamp = signal.timestamp
                # 这里简化处理，实际应该从信号中提取收益率
                return_value = 0.02 if i % 2 == 0 else -0.01  # 模拟收益率

                timestamps.append(timestamp.isoformat())
                returns.append(return_value)
                current_cumulative += return_value
                cumulative_returns.append(current_cumulative)

            chart_data = {
                'type': chart_type,
                'title': f'策略表现 - {strategy_result.strategy_id}',
                'x_axis': {
                    'title': '时间',
                    'data': timestamps
                },
                'y_axis': {
                    'title': '收益率',
                    'data': {
                        'returns': returns,
                        'cumulative_returns': cumulative_returns
                    }
                },
                'config': self.chart_configs['performance_chart']
            }

            return chart_data

        except Exception as e:
            logger.error(f"生成策略表现图表失败: {e}")
            return {}

    async def generate_backtest_analysis_charts(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """
        生成回测分析图表

        Args:
            backtest_result: 回测结果

        Returns:
            Dict[str, Any]: 分析图表数据
        """
        try:
            charts = {}

            # 收益率曲线图
            if not backtest_result.returns.empty:
                charts['returns_curve'] = await self._generate_returns_curve_chart(backtest_result)

            # 收益分布直方图
            if not backtest_result.returns.empty:
                charts['returns_distribution'] = await self._generate_returns_distribution_chart(backtest_result)

            # 回撤分析图
            charts['drawdown_analysis'] = await self._generate_drawdown_chart(backtest_result)

            # 月度收益热力图
            if not backtest_result.returns.empty:
                charts['monthly_returns_heatmap'] = await self._generate_monthly_returns_heatmap(backtest_result)

            return charts

        except Exception as e:
            logger.error(f"生成回测分析图表失败: {e}")
            return {}

    async def _generate_returns_curve_chart(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """
        生成收益率曲线图

        Args:
            backtest_result: 回测结果

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            returns = backtest_result.returns
            cumulative_returns = (1 + returns).cumprod() - 1

            chart_data = {
                'type': 'line',
                'title': '收益率曲线',
                'x_axis': {
                    'title': '日期',
                    'data': returns.index.strftime('%Y-%m-%d').tolist()
                },
                'y_axis': {
                    'title': '收益率',
                    'data': {
                        'daily_returns': returns.values.tolist(),
                        'cumulative_returns': cumulative_returns.values.tolist()
                    }
                },
                'series': [
                    {
                        'name': '日收益率',
                        'data': returns.values.tolist(),
                        'color': self.color_schemes['default'][0]
                    },
                    {
                        'name': '累计收益率',
                        'data': cumulative_returns.values.tolist(),
                        'color': self.color_schemes['default'][1]
                    }
                ]
            }

            return chart_data

        except Exception as e:
            logger.error(f"生成收益率曲线图失败: {e}")
            return {}

    async def _generate_returns_distribution_chart(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """
        生成收益率分布图

        Args:
            backtest_result: 回测结果

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            returns = backtest_result.returns.values

            # 计算分布统计
            hist, bin_edges = np.histogram(returns, bins=50)

            chart_data = {
                'type': 'histogram',
                'title': '收益率分布',
                'x_axis': {
                    'title': '收益率',
                    'data': bin_edges[:-1].tolist()
                },
                'y_axis': {
                    'title': '频次',
                    'data': hist.tolist()
                },
                'config': self.chart_configs['returns_distribution']
            }

            return chart_data

        except Exception as e:
            logger.error(f"生成收益率分布图失败: {e}")
            return {}

    async def _generate_drawdown_chart(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """
        生成回撤分析图

        Args:
            backtest_result: 回测结果

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            returns = backtest_result.returns
            cumulative = (1 + returns).cumprod()

            # 计算回撤
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max

            chart_data = {
                'type': 'area',
                'title': '回撤分析',
                'x_axis': {
                    'title': '日期',
                    'data': returns.index.strftime('%Y-%m-%d').tolist()
                },
                'y_axis': {
                    'title': '回撤幅度',
                    'data': drawdown.values.tolist()
                },
                'config': self.chart_configs['drawdown_chart']
            }

            return chart_data

        except Exception as e:
            logger.error(f"生成回撤分析图失败: {e}")
            return {}

    async def _generate_monthly_returns_heatmap(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """
        生成月度收益热力图

        Args:
            backtest_result: 回测结果

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            returns = backtest_result.returns

            # 按月聚合收益
            monthly_returns = returns.groupby(pd.Grouper(
                freq='M')).apply(lambda x: (1 + x).prod() - 1)

            # 转换为热力图格式
            heatmap_data = []
            for date, value in monthly_returns.items():
                heatmap_data.append({
                    'year': date.year,
                    'month': date.month,
                    'return': value
                })

            chart_data = {
                'type': 'heatmap',
                'title': '月度收益热力图',
                'data': heatmap_data,
                'config': {
                    'colorscale': 'RdYlGn',
                    'showscale': True
                }
            }

            return chart_data

        except Exception as e:
            logger.error(f"生成月度收益热力图失败: {e}")
            return {}

    async def generate_optimization_visualization(self, optimization_result: OptimizationResult) -> Dict[str, Any]:
        """
        生成优化可视化

        Args:
            optimization_result: 优化结果

        Returns:
            Dict[str, Any]: 可视化数据
        """
        try:
            visualizations = {}

            # 收敛历史图
            if optimization_result.convergence_history:
                visualizations['convergence_history'] = await self._generate_convergence_history_chart(optimization_result)

            # 参数分布图
            if optimization_result.all_results:
                visualizations['parameter_distribution'] = await self._generate_parameter_distribution_chart(optimization_result)

            # 优化轨迹图（如果有足够的数据）
            if len(optimization_result.all_results) > 2:
                visualizations['optimization_trajectory'] = await self._generate_optimization_trajectory_chart(optimization_result)

            return visualizations

        except Exception as e:
            logger.error(f"生成优化可视化失败: {e}")
            return {}

    async def _generate_convergence_history_chart(self, optimization_result: OptimizationResult) -> Dict[str, Any]:
        """
        生成收敛历史图

        Args:
            optimization_result: 优化结果

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            chart_data = {
                'type': 'line',
                'title': '优化收敛历史',
                'x_axis': {
                    'title': '迭代次数',
                    'data': list(range(1, len(optimization_result.convergence_history) + 1))
                },
                'y_axis': {
                    'title': '目标值',
                    'data': optimization_result.convergence_history
                },
                'series': [{
                    'name': '收敛历史',
                    'data': optimization_result.convergence_history,
                    'color': self.color_schemes['default'][0]
                }]
            }

            return chart_data

        except Exception as e:
            logger.error(f"生成收敛历史图失败: {e}")
            return {}

    async def _generate_parameter_distribution_chart(self, optimization_result: OptimizationResult) -> Dict[str, Any]:
        """
        生成参数分布图

        Args:
            optimization_result: 优化结果

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            # 提取所有参数值
            param_names = list(optimization_result.best_parameters.keys())
            param_data = {name: [] for name in param_names}

            for result in optimization_result.all_results:
                for param_name in param_names:
                    if param_name in result['parameters']:
                        param_data[param_name].append(result['parameters'][param_name])

            chart_data = {
                'type': 'boxplot',
                'title': '参数分布分析',
                'data': param_data,
                'config': {
                    'showfliers': False,
                    'boxmean': True
                }
            }

            return chart_data

        except Exception as e:
            logger.error(f"生成参数分布图失败: {e}")
            return {}

    async def _generate_optimization_trajectory_chart(self, optimization_result: OptimizationResult) -> Dict[str, Any]:
        """
        生成优化轨迹图

        Args:
            optimization_result: 优化结果

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            # 这里简化实现，实际应该生成2D或3D轨迹图
            chart_data = {
                'type': 'scatter',
                'title': '优化轨迹',
                'data': optimization_result.all_results,
                'config': {
                    'mode': 'markers + ines',
                    'marker': {
                        'size': 8,
                        'color': self.color_schemes['default'][0]
                    }
                }
            }

            return chart_data

        except Exception as e:
            logger.error(f"生成优化轨迹图失败: {e}")
            return {}

    async def generate_monitoring_dashboard(self, metrics: Dict[str, MetricData],
                                            time_range: str = '1h') -> Dict[str, Any]:
        """
        生成监控仪表板

        Args:
            metrics: 指标数据字典
            time_range: 时间范围 ('1h', '6h', '1d', '7d')

        Returns:
            Dict[str, Any]: 仪表板数据
        """
        try:
            dashboard = {
                'summary': {},
                'charts': {},
                'alerts': [],
                'timestamp': datetime.now().isoformat()
            }

            # 生成指标汇总
            dashboard['summary'] = await self._generate_metrics_summary(metrics)

            # 生成时间序列图表
            dashboard['charts']['time_series'] = await self._generate_time_series_chart(metrics, time_range)

            # 生成系统状态图表
            dashboard['charts']['system_status'] = await self._generate_system_status_chart(metrics)

            # 生成性能指标图表
            dashboard['charts']['performance_metrics'] = await self._generate_performance_metrics_chart(metrics)

            return dashboard

        except Exception as e:
            logger.error(f"生成监控仪表板失败: {e}")
            return {}

    async def _generate_metrics_summary(self, metrics: Dict[str, MetricData]) -> Dict[str, Any]:
        """
        生成指标汇总

        Args:
            metrics: 指标数据字典

        Returns:
            Dict[str, Any]: 汇总数据
        """
        try:
            summary = {
                'total_metrics': len(metrics),
                'latest_update': None,
                'system_health': 'unknown',
                'alert_count': 0
            }

            if metrics:
                # 找到最新的更新时间
                latest_metric = max(metrics.values(), key=lambda m: m.timestamp)
                summary['latest_update'] = latest_metric.timestamp.isoformat()

                # 评估系统健康状态
                summary['system_health'] = self._evaluate_system_health(metrics)

            return summary

        except Exception as e:
            logger.error(f"生成指标汇总失败: {e}")
            return {}

    def _evaluate_system_health(self, metrics: Dict[str, MetricData]) -> str:
        """
        评估系统健康状态

        Args:
            metrics: 指标数据字典

        Returns:
            str: 健康状态 ('healthy', 'warning', 'critical')
        """
        try:
            # 简化的健康评估逻辑
            warning_count = 0
            critical_count = 0

            for metric_name, metric_data in metrics.items():
                value = metric_data.value

                # 检查CPU使用率
                if metric_name == 'cpu_usage' and value > 80:
                    if value > 90:
                        critical_count += 1
                    else:
                        warning_count += 1

                # 检查内存使用率
                elif metric_name == 'memory_usage' and value > 85:
                    if value > 95:
                        critical_count += 1
                    else:
                        warning_count += 1

                # 检查错误率
                elif metric_name == 'error_rate' and value > 0.05:
                    if value > 0.1:
                        critical_count += 1
                    else:
                        warning_count += 1

            if critical_count > 0:
                return 'critical'
            elif warning_count > 0:
                return 'warning'
            else:
                return 'healthy'

        except Exception as e:
            logger.error(f"评估系统健康状态失败: {e}")
            return 'unknown'

    async def _generate_time_series_chart(self, metrics: Dict[str, MetricData],
                                          time_range: str) -> Dict[str, Any]:
        """
        生成时间序列图表

        Args:
            metrics: 指标数据字典
            time_range: 时间范围

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            # 这里简化实现，实际应该按时间范围过滤数据
            chart_data = {
                'type': 'line',
                'title': f'系统指标趋势 ({time_range})',
                'series': []
            }

            for metric_name, metric_data in metrics.items():
                chart_data['series'].append({
                    'name': metric_name,
                    'data': [metric_data.value],  # 简化：只显示最新值
                    'timestamp': [metric_data.timestamp.isoformat()]
                })

            return chart_data

        except Exception as e:
            logger.error(f"生成时间序列图表失败: {e}")
            return {}

    async def _generate_system_status_chart(self, metrics: Dict[str, MetricData]) -> Dict[str, Any]:
        """
        生成系统状态图表

        Args:
            metrics: 指标数据字典

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            # 计算各项指标的状态
            status_data = {
                'CPU': 'normal',
                '内存': 'normal',
                '网络': 'normal',
                '磁盘': 'normal'
            }

            # 基于指标值更新状态
            for metric_name, metric_data in metrics.items():
                value = metric_data.value

                if metric_name == 'cpu_usage':
                    if value > 80:
                        status_data['CPU'] = 'warning' if value < 90 else 'critical'
                elif metric_name == 'memory_usage':
                    if value > 85:
                        status_data['内存'] = 'warning' if value < 95 else 'critical'

            chart_data = {
                'type': 'status_grid',
                'title': '系统状态概览',
                'data': status_data
            }

            return chart_data

        except Exception as e:
            logger.error(f"生成系统状态图表失败: {e}")
            return {}

    async def _generate_performance_metrics_chart(self, metrics: Dict[str, MetricData]) -> Dict[str, Any]:
        """
        生成性能指标图表

        Args:
            metrics: 指标数据字典

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            performance_metrics = {}

            # 提取性能相关指标
            for metric_name, metric_data in metrics.items():
                if metric_name in ['response_time', 'throughput', 'strategy_return', 'strategy_sharpe']:
                    performance_metrics[metric_name] = metric_data.value

            chart_data = {
                'type': 'bar',
                'title': '性能指标',
                'data': performance_metrics,
                'config': {
                    'colors': self.color_schemes['greens']
                }
            }

            return chart_data

        except Exception as e:
            logger.error(f"生成性能指标图表失败: {e}")
            return {}

    def export_chart_data(self, chart_data: Dict[str, Any], format: str = 'json') -> str:
        """
        导出图表数据

        Args:
            chart_data: 图表数据
            format: 导出格式 ('json', 'csv', 'png')

        Returns:
            str: 导出的数据
        """
        try:
            if format == 'json':
                return json.dumps(chart_data, indent=2, default=str)
            elif format == 'csv':
                # 这里可以实现CSV导出
                return "CSV export not implemented yet"
            elif format == 'png':
                # 这里可以实现PNG导出
                return "PNG export not implemented yet"
            else:
                raise ValueError(f"不支持的导出格式: {format}")

        except Exception as e:
            logger.error(f"导出图表数据失败: {e}")
            return ""

    async def generate_custom_chart(self, data: pd.DataFrame, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成自定义图表

        Args:
            data: 数据框
            chart_config: 图表配置

        Returns:
            Dict[str, Any]: 图表数据
        """
        try:
            chart_type = chart_config.get('type', 'line')
            title = chart_config.get('title', '自定义图表')

            base_chart = {
                'type': chart_type,
                'title': title,
                'config': chart_config
            }

            # 根据图表类型处理数据
            if chart_type == 'line':
                base_chart.update(await self._process_line_chart_data(data, chart_config))
            elif chart_type == 'bar':
                base_chart.update(await self._process_bar_chart_data(data, chart_config))
            elif chart_type == 'scatter':
                base_chart.update(await self._process_scatter_chart_data(data, chart_config))
            elif chart_type == 'heatmap':
                base_chart.update(await self._process_heatmap_data(data, chart_config))

            return base_chart

        except Exception as e:
            logger.error(f"生成自定义图表失败: {e}")
            return {}

    async def _process_line_chart_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理线图数据"""
        try:
            x_column = config.get('x_column', data.columns[0])
            y_columns = config.get('y_columns', [data.columns[1]])

            chart_data = {
                'x_axis': {
                    'title': x_column,
                    'data': data[x_column].tolist()
                },
                'series': []
            }

            colors = self.color_schemes.get(config.get(
                'color_scheme', 'default'), self.color_schemes['default'])

            for i, y_column in enumerate(y_columns):
                if y_column in data.columns:
                    chart_data['series'].append({
                        'name': y_column,
                        'data': data[y_column].tolist(),
                        'color': colors[i % len(colors)]
                    })

            return chart_data

        except Exception as e:
            logger.error(f"处理线图数据失败: {e}")
            return {}

    async def _process_bar_chart_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理柱状图数据"""
        # 类似实现
        return await self._process_line_chart_data(data, config)

    async def _process_scatter_chart_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理散点图数据"""
        try:
            x_column = config.get('x_column', data.columns[0])
            y_column = config.get('y_column', data.columns[1])

            chart_data = {
                'x_axis': {
                    'title': x_column,
                    'data': data[x_column].tolist()
                },
                'y_axis': {
                    'title': y_column,
                    'data': data[y_column].tolist()
                },
                'config': {
                    'mode': config.get('mode', 'markers'),
                    'marker': {
                        'size': config.get('marker_size', 6),
                        'color': config.get('marker_color', self.color_schemes['default'][0])
                    }
                }
            }

            return chart_data

        except Exception as e:
            logger.error(f"处理散点图数据失败: {e}")
            return {}

    async def _process_heatmap_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理热力图数据"""
        try:
            chart_data = {
                'data': data.values.tolist(),
                'x_labels': data.columns.tolist(),
                'y_labels': data.index.tolist(),
                'config': {
                    'colorscale': config.get('colorscale', 'Viridis'),
                    'showscale': config.get('showscale', True)
                }
            }

            return chart_data

        except Exception as e:
            logger.error(f"处理热力图数据失败: {e}")
            return {}


# 导出
__all__ = [
    'VisualizationService'
]

