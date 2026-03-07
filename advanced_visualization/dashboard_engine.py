#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 高级可视化仪表板引擎
提供实时数据可视化、交互式图表和智能监控面板

可视化能力:
1. 实时仪表板 - 系统状态、性能指标、业务KPI
2. 交互式图表 - 趋势图、热力图、网络图、3D可视化
3. 多维度分析 - 钻取分析、对比分析、相关性分析
4. 自定义视图 - 用户自定义仪表板和报告
5. 实时监控 - 异常检测、告警可视化、预测展示
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import random
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import base64
from io import BytesIO

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class VisualizationEngine:
    """可视化引擎"""

    def __init__(self):
        self.dashboards = {}
        self.visualizations = {}
        self.data_cache = {}
        self.color_schemes = self._load_color_schemes()

    def _load_color_schemes(self):
        """加载颜色方案"""
        return {
            'default': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
            'cool': ['#3498db', '#2980b9', '#34495e', '#2c3e50', '#95a5a6'],
            'warm': ['#e74c3c', '#c0392b', '#e67e22', '#f39c12', '#f1c40f'],
            'nature': ['#27ae60', '#2ecc71', '#3498db', '#9b59b6', '#34495e']
        }

    def create_system_dashboard(self, system_data):
        """创建系统仪表板"""
        dashboard = {
            'title': 'RQA2026 系统监控仪表板',
            'created_at': datetime.now().isoformat(),
            'panels': [],
            'layout': 'grid',
            'theme': 'default'
        }

        # 系统状态面板
        system_status_panel = self._create_system_status_panel(system_data)
        dashboard['panels'].append(system_status_panel)

        # 性能指标面板
        performance_panel = self._create_performance_panel(system_data)
        dashboard['panels'].append(performance_panel)

        # 引擎状态面板
        engine_panel = self._create_engine_status_panel(system_data)
        dashboard['panels'].append(engine_panel)

        # 告警面板
        alerts_panel = self._create_alerts_panel(system_data)
        dashboard['panels'].append(alerts_panel)

        return dashboard

    def _create_system_status_panel(self, system_data):
        """创建系统状态面板"""
        return {
            'id': 'system_status',
            'title': '系统状态概览',
            'type': 'status_cards',
            'position': {'x': 0, 'y': 0, 'width': 12, 'height': 2},
            'data': {
                'cpu_usage': system_data.get('system_status', {}).get('summary', {}).get('cpu_avg', 0),
                'memory_usage': system_data.get('system_status', {}).get('summary', {}).get('memory_avg', 0),
                'disk_usage': system_data.get('system_status', {}).get('summary', {}).get('disk_usage', 0),
                'services_online': system_data.get('service_health', {}).get('summary', {}).get('healthy_services', 0)
            },
            'visualization': 'cards'
        }

    def _create_performance_panel(self, system_data):
        """创建性能指标面板"""
        performance_data = system_data.get('performance_metrics', {})

        # 准备图表数据
        labels = list(performance_data.keys())
        response_times = [metrics.get('avg_response_time', 0) * 1000 for metrics in performance_data.values()]  # 转换为毫秒
        success_rates = [metrics.get('success_rate', 0) * 100 for metrics in performance_data.values()]

        chart_data = {
            'labels': labels,
            'datasets': [
                {
                    'label': '平均响应时间 (ms)',
                    'data': response_times,
                    'borderColor': self.color_schemes['default'][0],
                    'backgroundColor': self.color_schemes['default'][0] + '20',
                    'yAxisID': 'y'
                },
                {
                    'label': '成功率 (%)',
                    'data': success_rates,
                    'borderColor': self.color_schemes['default'][1],
                    'backgroundColor': self.color_schemes['default'][1] + '20',
                    'yAxisID': 'y1'
                }
            ]
        }

        return {
            'id': 'performance_metrics',
            'title': '性能指标监控',
            'type': 'chart',
            'position': {'x': 0, 'y': 2, 'width': 8, 'height': 4},
            'data': chart_data,
            'visualization': 'line_chart',
            'options': {
                'scales': {
                    'y': {'title': {'display': True, 'text': '响应时间 (ms)'}},
                    'y1': {'position': 'right', 'title': {'display': True, 'text': '成功率 (%)'}}
                }
            }
        }

    def _create_engine_status_panel(self, system_data):
        """创建引擎状态面板"""
        service_data = system_data.get('service_health', {}).get('services', {})

        # 创建网络图数据
        nodes = []
        links = []

        # 添加引擎节点
        engines = ['fusion_engine', 'quantum_engine', 'ai_engine', 'bci_engine', 'web_interface']
        for i, engine in enumerate(engines):
            status = service_data.get(engine, {}).get('healthy', False)
            nodes.append({
                'id': engine,
                'name': engine.replace('_', ' ').title(),
                'group': 'engine',
                'status': 'healthy' if status else 'unhealthy',
                'size': 20
            })

        # 添加连接
        connections = [
            ('fusion_engine', 'quantum_engine'),
            ('fusion_engine', 'ai_engine'),
            ('fusion_engine', 'bci_engine'),
            ('web_interface', 'fusion_engine')
        ]

        for source, target in connections:
            links.append({
                'source': source,
                'target': target,
                'value': 1
            })

        return {
            'id': 'engine_status',
            'title': '引擎状态网络',
            'type': 'network',
            'position': {'x': 8, 'y': 2, 'width': 4, 'height': 4},
            'data': {'nodes': nodes, 'links': links},
            'visualization': 'force_directed_graph'
        }

    def _create_alerts_panel(self, system_data):
        """创建告警面板"""
        alerts = system_data.get('system_status', {}).get('summary', {}).get('alerts', [])

        alert_counts = defaultdict(int)
        for alert in alerts:
            alert_counts[alert.get('level', 'unknown')] += 1

        chart_data = {
            'labels': list(alert_counts.keys()),
            'datasets': [{
                'label': '告警数量',
                'data': list(alert_counts.values()),
                'backgroundColor': [self.color_schemes['warm'][i % len(self.color_schemes['warm'])]
                                  for i in range(len(alert_counts))]
            }]
        }

        return {
            'id': 'alerts_overview',
            'title': '告警统计',
            'type': 'chart',
            'position': {'x': 0, 'y': 6, 'width': 6, 'height': 3},
            'data': chart_data,
            'visualization': 'doughnut_chart'
        }

    def create_business_intelligence_dashboard(self, bi_data):
        """创建商业智能仪表板"""
        dashboard = {
            'title': '商业智能分析仪表板',
            'created_at': datetime.now().isoformat(),
            'panels': [],
            'layout': 'grid',
            'theme': 'cool'
        }

        # KPI 指标面板
        kpi_panel = self._create_kpi_panel(bi_data)
        dashboard['panels'].append(kpi_panel)

        # 趋势分析面板
        trend_panel = self._create_trend_analysis_panel(bi_data)
        dashboard['panels'].append(trend_panel)

        # 预测面板
        forecast_panel = self._create_forecast_panel(bi_data)
        dashboard['panels'].append(forecast_panel)

        return dashboard

    def _create_kpi_panel(self, bi_data):
        """创建KPI指标面板"""
        kpis = bi_data.get('business_intelligence', {}).get('kpi_dashboard', {})

        return {
            'id': 'kpi_metrics',
            'title': '关键绩效指标',
            'type': 'kpi_cards',
            'position': {'x': 0, 'y': 0, 'width': 12, 'height': 2},
            'data': kpis,
            'visualization': 'metric_cards'
        }

    def _create_trend_analysis_panel(self, bi_data):
        """创建趋势分析面板"""
        insights = bi_data.get('data_analysis', {}).get('predictive_insights', [])

        # 按类型分组洞察
        insight_types = defaultdict(list)
        for insight in insights:
            insight_types[insight.get('type', 'unknown')].append(insight)

        return {
            'id': 'trend_analysis',
            'title': '趋势分析与洞察',
            'type': 'insights',
            'position': {'x': 0, 'y': 2, 'width': 8, 'height': 4},
            'data': dict(insight_types),
            'visualization': 'insight_cards'
        }

    def _create_forecast_panel(self, bi_data):
        """创建预测面板"""
        forecasts = bi_data.get('predictions', {})

        # 创建预测图表数据
        forecast_data = []
        for source, prediction in forecasts.items():
            if isinstance(prediction, list) and len(prediction) > 0:
                forecast_data.append({
                    'name': source,
                    'forecast': prediction
                })

        return {
            'id': 'forecast_analysis',
            'title': '预测分析',
            'type': 'forecast',
            'position': {'x': 8, 'y': 2, 'width': 4, 'height': 4},
            'data': forecast_data,
            'visualization': 'forecast_chart'
        }

    def create_analytics_dashboard(self, analytics_data):
        """创建分析仪表板"""
        dashboard = {
            'title': '高级分析仪表板',
            'created_at': datetime.now().isoformat(),
            'panels': [],
            'layout': 'grid',
            'theme': 'nature'
        }

        # 数据质量面板
        quality_panel = self._create_data_quality_panel(analytics_data)
        dashboard['panels'].append(quality_panel)

        # 相关性分析面板
        correlation_panel = self._create_correlation_panel(analytics_data)
        dashboard['panels'].append(correlation_panel)

        # 异常检测面板
        anomaly_panel = self._create_anomaly_detection_panel(analytics_data)
        dashboard['panels'].append(anomaly_panel)

        return dashboard

    def _create_data_quality_panel(self, analytics_data):
        """创建数据质量面板"""
        statistical_analysis = analytics_data.get('data_analysis', {}).get('statistical_analysis', {})

        quality_metrics = {}
        for source, stats in statistical_analysis.items():
            quality_score = self._calculate_data_quality_score(stats)
            quality_metrics[source] = {
                'completeness': 0.95,  # 假设数据完整性
                'accuracy': quality_score,
                'consistency': 0.88,
                'timeliness': 0.92
            }

        return {
            'id': 'data_quality',
            'title': '数据质量评估',
            'type': 'radar',
            'position': {'x': 0, 'y': 0, 'width': 6, 'height': 4},
            'data': quality_metrics,
            'visualization': 'radar_chart'
        }

    def _calculate_data_quality_score(self, stats):
        """计算数据质量评分"""
        if not stats or 'error' in stats:
            return 0.0

        # 基于统计特征计算质量评分
        score = 0.5  # 基础分

        # 标准差适中加分
        if 'std_dev' in stats and stats['std_dev'] > 0:
            if 0.1 < stats['std_dev'] / (stats.get('mean', 1) or 1) < 1.0:
                score += 0.2

        # 数据范围合理加分
        if 'min' in stats and 'max' in stats:
            range_ratio = abs(stats['max'] - stats['min']) / (abs(stats.get('mean', 1)) or 1)
            if 0.1 < range_ratio < 10:
                score += 0.2

        # 偏度峰度正常加分
        if 'skewness' in stats and 'kurtosis' in stats:
            if abs(stats['skewness']) < 2 and abs(stats['kurtosis']) < 5:
                score += 0.1

        return min(score, 1.0)

    def _create_correlation_panel(self, analytics_data):
        """创建相关性分析面板"""
        correlations = analytics_data.get('data_analysis', {}).get('correlation_analysis', {})

        # 转换为热力图数据
        sources = set()
        for corr_key in correlations.keys():
            sources.update(corr_key.split('_vs_'))

        sources = sorted(list(sources))
        correlation_matrix = np.eye(len(sources))

        # 填充相关系数
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                if i != j:
                    key1 = f"{source1}_vs_{source2}"
                    key2 = f"{source2}_vs_{source1}"
                    corr_value = correlations.get(key1, {}).get('correlation', 0) or \
                               correlations.get(key2, {}).get('correlation', 0)
                    correlation_matrix[i, j] = corr_value

        return {
            'id': 'correlation_analysis',
            'title': '数据相关性分析',
            'type': 'heatmap',
            'position': {'x': 6, 'y': 0, 'width': 6, 'height': 4},
            'data': {
                'matrix': correlation_matrix.tolist(),
                'labels': sources
            },
            'visualization': 'heatmap'
        }

    def _create_anomaly_detection_panel(self, analytics_data):
        """创建异常检测面板"""
        anomaly_data = analytics_data.get('data_analysis', {}).get('anomaly_detection', {})

        # 汇总异常统计
        anomaly_stats = {}
        for source, data in anomaly_data.items():
            if isinstance(data, dict) and 'anomalies_detected' in data:
                anomaly_stats[source] = {
                    'detected': data['anomalies_detected'],
                    'rate': data.get('anomaly_rate', 0),
                    'threshold': data.get('threshold', 0)
                }

        return {
            'id': 'anomaly_detection',
            'title': '异常检测结果',
            'type': 'anomaly_chart',
            'position': {'x': 0, 'y': 4, 'width': 12, 'height': 3},
            'data': anomaly_stats,
            'visualization': 'bar_chart'
        }

    def generate_chart_image(self, chart_data, chart_type='line'):
        """生成图表图片"""
        plt.figure(figsize=(10, 6))

        if chart_type == 'line':
            for dataset in chart_data.get('datasets', []):
                plt.plot(chart_data.get('labels', []), dataset.get('data', []),
                        label=dataset.get('label', ''),
                        color=dataset.get('borderColor', self.color_schemes['default'][0]))

        elif chart_type == 'bar':
            x = np.arange(len(chart_data.get('labels', [])))
            for i, dataset in enumerate(chart_data.get('datasets', [])):
                plt.bar(x + i * 0.25, dataset.get('data', []),
                       width=0.25, label=dataset.get('label', ''),
                       color=self.color_schemes['default'][i % len(self.color_schemes['default'])])

        elif chart_type == 'heatmap':
            matrix = np.array(chart_data.get('matrix', []))
            labels = chart_data.get('labels', [])
            sns.heatmap(matrix, annot=True, cmap='coolwarm',
                       xticklabels=labels, yticklabels=labels)
            plt.title('相关性热力图')

        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存为base64编码
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def create_custom_dashboard(self, user_preferences, available_data):
        """创建自定义仪表板"""
        dashboard = {
            'title': f'{user_preferences.get("user_name", "用户")}的自定义仪表板',
            'created_at': datetime.now().isoformat(),
            'panels': [],
            'layout': user_preferences.get('layout', 'grid'),
            'theme': user_preferences.get('theme', 'default')
        }

        # 根据用户偏好添加面板
        preferred_panels = user_preferences.get('preferred_panels', [])

        for panel_type in preferred_panels:
            if panel_type == 'system_status':
                panel = self._create_system_status_panel(available_data)
            elif panel_type == 'performance':
                panel = self._create_performance_panel(available_data)
            elif panel_type == 'alerts':
                panel = self._create_alerts_panel(available_data)
            else:
                continue

            dashboard['panels'].append(panel)

        return dashboard


class RealtimeVisualizationEngine:
    """实时可视化引擎"""

    def __init__(self):
        self.subscribers = defaultdict(list)
        self.data_streams = {}
        self.update_interval = 5  # 5秒更新一次

    def start_realtime_updates(self, dashboard_id, data_source):
        """启动实时更新"""
        self.data_streams[dashboard_id] = {
            'data_source': data_source,
            'last_update': datetime.now(),
            'active': True
        }

    def stop_realtime_updates(self, dashboard_id):
        """停止实时更新"""
        if dashboard_id in self.data_streams:
            self.data_streams[dashboard_id]['active'] = False

    def subscribe_to_updates(self, dashboard_id, callback):
        """订阅更新"""
        self.subscribers[dashboard_id].append(callback)

    def publish_update(self, dashboard_id, update_data):
        """发布更新"""
        if dashboard_id in self.subscribers:
            for callback in self.subscribers[dashboard_id]:
                try:
                    callback(update_data)
                except Exception as e:
                    print(f"更新回调失败: {e}")

    def get_realtime_data(self, dashboard_id):
        """获取实时数据"""
        if dashboard_id not in self.data_streams:
            return None

        stream = self.data_streams[dashboard_id]
        if not stream['active']:
            return None

        # 模拟实时数据更新
        current_time = datetime.now()
        time_diff = (current_time - stream['last_update']).total_seconds()

        if time_diff >= self.update_interval:
            # 生成新的实时数据
            realtime_data = self._generate_realtime_data(stream['data_source'])
            stream['last_update'] = current_time

            # 发布更新
            self.publish_update(dashboard_id, realtime_data)

            return realtime_data

        return None

    def _generate_realtime_data(self, data_source):
        """生成实时数据"""
        # 模拟实时数据生成
        current_time = datetime.now()

        if data_source == 'system_metrics':
            return {
                'timestamp': current_time.isoformat(),
                'cpu_usage': random.uniform(10, 90),
                'memory_usage': random.uniform(40, 85),
                'network_traffic': random.uniform(100, 1000),
                'active_connections': random.randint(10, 100)
            }

        elif data_source == 'performance_metrics':
            return {
                'timestamp': current_time.isoformat(),
                'response_time': random.uniform(0.01, 0.5),
                'throughput': random.uniform(50, 200),
                'error_rate': random.uniform(0, 0.05),
                'success_rate': random.uniform(0.95, 1.0)
            }

        return {
            'timestamp': current_time.isoformat(),
            'status': 'active',
            'value': random.uniform(0, 100)
        }


def create_comprehensive_dashboard():
    """创建综合仪表板"""
    viz_engine = VisualizationEngine()
    rt_engine = RealtimeVisualizationEngine()

    # 模拟数据源
    system_data = {
        'system_status': {
            'summary': {
                'cpu_avg': 35.2,
                'memory_avg': 62.8,
                'disk_usage': 34.1,
                'active_alerts': 2,
                'alerts': [
                    {'level': 'warning', 'message': 'CPU使用率较高', 'timestamp': datetime.now().isoformat()},
                    {'level': 'critical', 'message': '内存使用率临近阈值', 'timestamp': datetime.now().isoformat()}
                ]
            }
        },
        'service_health': {
            'summary': {
                'total_services': 5,
                'healthy_services': 4,
                'unhealthy_services': 1
            },
            'services': {
                'fusion_engine': {'healthy': True, 'response_time': 0.05},
                'quantum_engine': {'healthy': True, 'response_time': 0.08},
                'ai_engine': {'healthy': False, 'error': 'connection_failed'},
                'bci_engine': {'healthy': True, 'response_time': 0.03},
                'web_interface': {'healthy': True, 'response_time': 0.12}
            }
        },
        'performance_metrics': {
            'fusion_engine': {'avg_response_time': 0.045, 'success_rate': 0.98},
            'quantum_engine': {'avg_response_time': 0.082, 'success_rate': 0.95},
            'ai_engine': {'avg_response_time': 0.156, 'success_rate': 0.92},
            'bci_engine': {'avg_response_time': 0.028, 'success_rate': 0.99},
            'web_interface': {'avg_response_time': 0.118, 'success_rate': 0.97}
        }
    }

    # 创建系统仪表板
    system_dashboard = viz_engine.create_system_dashboard(system_data)

    # 创建商业智能仪表板
    bi_data = {
        'business_intelligence': {
            'kpi_dashboard': {
                'system_performance': {'value': 98.5, 'target': 99.0, 'status': 'near_target'},
                'data_quality': {'value': 95.2, 'target': 96.0, 'status': 'approaching_target'},
                'risk_coverage': {'value': 87.3, 'target': 90.0, 'status': 'below_target'},
                'user_satisfaction': {'value': 4.7, 'target': 4.8, 'status': 'at_target'}
            },
            'recommendations': [
                {'priority': 'high', 'recommendation': '优化系统性能'},
                {'priority': 'medium', 'recommendation': '加强数据质量监控'}
            ]
        },
        'data_analysis': {
            'predictive_insights': [
                {'type': 'trend_prediction', 'insight': '市场波动性呈上升趋势'},
                {'type': 'anomaly_alert', 'insight': '检测到异常交易模式'}
            ]
        },
        'predictions': {
            'market_volatility': [0.18, 0.19, 0.21],
            'portfolio_returns': [0.025, 0.028, 0.031]
        }
    }

    bi_dashboard = viz_engine.create_business_intelligence_dashboard(bi_data)

    # 创建分析仪表板
    analytics_data = {
        'data_analysis': {
            'statistical_analysis': {
                'market_data': {'mean': 0.15, 'std_dev': 0.03, 'min': 0.08, 'max': 0.25},
                'portfolio_data': {'mean': 0.02, 'std_dev': 0.015, 'min': -0.02, 'max': 0.06}
            },
            'correlation_analysis': {
                'market_vs_portfolio': {'correlation': 0.65, 'strength': 'moderate', 'direction': 'positive'}
            },
            'anomaly_detection': {
                'market_data': {'anomalies_detected': 3, 'anomaly_rate': 0.02},
                'portfolio_data': {'anomalies_detected': 1, 'anomaly_rate': 0.005}
            }
        }
    }

    analytics_dashboard = viz_engine.create_analytics_dashboard(analytics_data)

    return {
        'system_dashboard': system_dashboard,
        'bi_dashboard': bi_dashboard,
        'analytics_dashboard': analytics_dashboard,
        'realtime_engine': rt_engine
    }


def main():
    """主函数"""
    print("📊 启动 RQA2026 高级可视化仪表板引擎")
    print("=" * 80)

    # 创建综合仪表板
    dashboards = create_comprehensive_dashboard()

    print("✅ 系统仪表板创建完成")
    print(f"   面板数量: {len(dashboards['system_dashboard']['panels'])}")
    print(f"   主题: {dashboards['system_dashboard']['theme']}")

    print("\\n✅ 商业智能仪表板创建完成")
    print(f"   面板数量: {len(dashboards['bi_dashboard']['panels'])}")
    print(f"   主题: {dashboards['bi_dashboard']['theme']}")

    print("\\n✅ 分析仪表板创建完成")
    print(f"   面板数量: {len(dashboards['analytics_dashboard']['panels'])}")
    print(f"   主题: {dashboards['analytics_dashboard']['theme']}")

    # 启动实时可视化
    rt_engine = dashboards['realtime_engine']
    rt_engine.start_realtime_updates('system_dashboard', 'system_metrics')
    rt_engine.start_realtime_updates('performance_dashboard', 'performance_metrics')

    print("\\n🔄 实时可视化引擎启动")
    print("   支持实时数据更新和订阅推送")

    # 保存仪表板配置
    dashboard_config = {
        'created_at': datetime.now().isoformat(),
        'dashboards': {
            'system': dashboards['system_dashboard'],
            'business_intelligence': dashboards['bi_dashboard'],
            'analytics': dashboards['analytics_dashboard']
        },
        'realtime_enabled': True,
        'update_interval': 5
    }

    dashboard_file = Path('advanced_visualization/dashboard_config.json')
    dashboard_file.parent.mkdir(exist_ok=True)

    with open(dashboard_file, 'w', encoding='utf-8') as f:
        json.dump(dashboard_config, f, indent=2, ensure_ascii=False, default=str)

    print(f"\\n💾 仪表板配置已保存: {dashboard_file}")

    # 演示实时数据更新
    print("\\n📈 演示实时数据更新...")
    for i in range(3):
        import time
        time.sleep(2)

        # 获取实时数据
        system_data = rt_engine.get_realtime_data('system_dashboard')
        perf_data = rt_engine.get_realtime_data('performance_dashboard')

        if system_data:
            print(f"   系统数据更新: CPU {system_data['cpu_usage']:.1f}%, 内存 {system_data['memory_usage']:.1f}%")

        if perf_data:
            print(f"   性能数据更新: 响应时间 {perf_data['response_time']:.3f}s, 成功率 {perf_data['success_rate']:.1%}")

    print("\\n🎉 高级可视化仪表板引擎运行完成！")
    print("🌟 仪表板现已就绪，可用于实时监控和数据可视化")


if __name__ == "__main__":
    main()
