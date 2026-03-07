"""
监控面板管理器

提供监控数据的可视化展示和实时监控界面，支持多种图表类型，
包括实时指标展示、性能趋势分析、告警状态监控等功能。
"""

import json
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import webbrowser

from .features_monitor import get_monitor
from .metrics_persistence import get_persistence_manager


logger = logging.getLogger(__name__)


class ChartType(Enum):

    """图表类型枚举"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    GAUGE = "gauge"
    TABLE = "table"


@dataclass
class DashboardConfig:

    """面板配置"""
    title: str
    refresh_interval: float = 5.0
    auto_refresh: bool = True
    charts: List[Dict[str, Any]] = None
    layout: Dict[str, Any] = None


@dataclass
class ChartConfig:

    """图表配置"""
    id: str
    title: str
    chart_type: ChartType
    data_source: str
    metrics: List[str]
    time_range: Optional[Tuple[float, float]] = None
    options: Dict[str, Any] = None


class MonitoringDashboard:

    """
    监控面板管理器

    提供监控数据的可视化展示和实时监控界面。
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化监控面板

        Args:
            config: 面板配置
        """
        self.config = config or {}
        self.monitor = get_monitor(self.config.get('monitor_config', {}))
        self.persistence_manager = get_persistence_manager(
            self.config.get('persistence_config', {}))

        # 面板配置
        self.dashboard_config = DashboardConfig(
            title=self.config.get('title', '特征层监控面板'),
            refresh_interval=self.config.get('refresh_interval', 5.0),
            auto_refresh=self.config.get('auto_refresh', True)
        )

        # 图表配置
        self.charts: Dict[str, ChartConfig] = {}
        self._init_default_charts()

        # 新增：通用组件/仪表板/数据源注册表
        self.widgets: Dict[str, Dict[str, Any]] = {}
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        self.data_sources: Dict[str, Dict[str, Any]] = {}

        # 面板状态
        self.is_running = False
        self.dashboard_thread = None
        self.last_update = time.time()

        # 输出目录
        self.output_dir = Path(self.config.get('output_dir', './monitoring_dashboard'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_default_charts(self) -> None:
        """初始化默认图表"""

        default_charts = [
            {
                'id': 'performance_overview',
                'title': '性能概览',
                'chart_type': ChartType.LINE,
                'data_source': 'metrics',
                'metrics': ['feature_generation_time', 'indicator_calculation_time'],
                'options': {'height': 300, 'width': 600}
            },
            {
                'id': 'component_status',
                'title': '组件状态',
                'chart_type': ChartType.GAUGE,
                'data_source': 'status',
                'metrics': ['cpu_usage', 'memory_usage'],
                'options': {'height': 200, 'width': 300}
            },
            {
                'id': 'error_summary',
                'title': '错误统计',
                'chart_type': ChartType.BAR,
                'data_source': 'metrics',
                'metrics': ['feature_generation_errors', 'indicator_calculation_errors'],
                'options': {'height': 250, 'width': 400}
            },
            {
                'id': 'alert_summary',
                'title': '告警汇总',
                'chart_type': ChartType.TABLE,
                'data_source': 'alerts',
                'metrics': ['alert_count', 'alert_severity'],
                'options': {'height': 200, 'width': 500}
            }
        ]

        for chart_config in default_charts:
            self.add_chart(ChartConfig(**chart_config))

    def add_chart(self, chart_config: ChartConfig) -> None:
        """
        添加图表

        Args:
            chart_config: 图表配置
        """
        self.charts[chart_config.id] = chart_config
        logger.info(f"已添加图表: {chart_config.title}")

    def remove_chart(self, chart_id: str) -> None:
        """
        移除图表

        Args:
            chart_id: 图表ID
        """
        if chart_id in self.charts:
            del self.charts[chart_id]
            logger.info(f"已移除图表: {chart_id}")

    # ------------------------------------------------------------------
    #  Dashboard-friendly helper APIs (用于单测与业务统一调用)
    # ------------------------------------------------------------------
    def add_widget(self, widget_id: str, widget_type: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """注册一个监控组件"""
        self.widgets[widget_id] = {
            'id': widget_id,
            'type': widget_type,
            'config': config or {},
            'created_at': datetime.now(),
        }
        logger.info("已注册监控组件 %s 类型 %s", widget_id, widget_type)
        return True

    def update_widget_config(self, widget_id: str, new_config: Dict[str, Any]) -> bool:
        """更新组件配置"""
        widget = self.widgets.get(widget_id)
        if not widget:
            return False
        widget['config'].update(new_config)
        widget['updated_at'] = datetime.now()
        return True

    def create_dashboard(self, dashboard_id: str, title: str, widgets: List[str]) -> bool:
        """创建仪表板"""
        self.dashboards[dashboard_id] = {
            'title': title,
            'widgets': widgets,
            'created_at': datetime.now(),
            'layout': 'grid',
        }
        logger.info("仪表板 %s 创建完成，包含 %d 个组件", dashboard_id, len(widgets))
        return True

    def get_dashboard_list(self) -> List[Dict[str, Any]]:
        """列出所有仪表板"""
        results = []
        for dashboard_id, dashboard in self.dashboards.items():
            results.append({
                'id': dashboard_id,
                'title': dashboard['title'],
                'widget_count': len(dashboard['widgets']),
                'created_at': dashboard['created_at'],
            })
        return results

    def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """获取特定仪表板的可视化数据"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None

        widgets_data: Dict[str, Any] = {}
        for widget_id in dashboard['widgets']:
            widget = self.widgets.get(widget_id)
            if widget:
                widgets_data[widget_id] = self._generate_widget_data(widget)

        return {
            'dashboard_id': dashboard_id,
            'title': dashboard['title'],
            'widgets_data': widgets_data,
        }

    def add_data_source(self, source_id: str, source_type: str, connection_config: Dict[str, Any]) -> bool:
        """注册监控数据源"""
        self.data_sources[source_id] = {
            'type': source_type,
            'config': connection_config,
            'status': 'connected',
            'created_at': datetime.now(),
        }
        logger.info("数据源 %s (%s) 注册成功", source_id, source_type)
        return True

    def get_chart_data(self, chart_config: ChartConfig) -> Dict[str, Any]:
        """
        获取图表数据

        Args:
            chart_config: 图表配置

        Returns:
            图表数据
        """
        if chart_config.data_source == 'metrics':
            return self._get_metrics_data(chart_config)
        elif chart_config.data_source == 'status':
            return self._get_status_data(chart_config)
        elif chart_config.data_source == 'alerts':
            return self._get_alerts_data(chart_config)
        else:
            return {}

    def _get_metrics_data(self, chart_config: ChartConfig) -> Dict[str, Any]:
        """获取指标数据"""
        data = {
            'labels': [],
            'datasets': []
        }

        # 获取时间范围
        end_time = time.time()
        start_time = end_time - (24 * 3600)  # 默认24小时

        if chart_config.time_range:
            start_time, end_time = chart_config.time_range

        # 查询指标数据
        for metric in chart_config.metrics:
            df = self.persistence_manager.query_metrics(
                metric_name=metric,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )

            if not df.empty:
                # 按时间排序
                df = df.sort_values('timestamp')

                # 转换为图表数据格式
                labels = [datetime.fromtimestamp(ts).strftime('%H:%M')
                          for ts in df['timestamp']]
                values = df['metric_value'].tolist()

                data['labels'] = labels
                data['datasets'].append({
                    'label': metric,
                    'data': values,
                    'borderColor': self._get_chart_color(len(data['datasets'])),
                    'fill': False
                })

        return data

    def _get_status_data(self, chart_config: ChartConfig) -> Dict[str, Any]:
        """获取状态数据"""
        data = {
            'components': [],
            'values': []
        }

        # 获取组件状态
        status_data = self.monitor.get_all_status()

        for component_name, status in status_data.items():
            data['components'].append(component_name)

            # 获取状态值
            if 'metrics' in status:
                metrics = status['metrics']
                if chart_config.metrics:
                    # 取第一个指标的值
                    metric_name = chart_config.metrics[0]
                    value = metrics.get(metric_name, {}).get('value', 0)
                    data['values'].append(value)
                else:
                    data['values'].append(0)
            else:
                data['values'].append(0)

        return data

    def _get_alerts_data(self, chart_config: ChartConfig) -> Dict[str, Any]:
        """获取告警数据"""
        data = {
            'headers': ['时间', '组件', '类型', '消息', '严重程度'],
            'rows': []
        }

        # 获取最近的告警
        alerts = self.monitor.alert_manager.get_recent_alerts(limit=10)

        for alert in alerts:
            data['rows'].append([
                datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S'),
                alert.get('component', ''),
                alert.get('alert_type', ''),
                alert.get('message', '')[
                    :50] + '...' if len(alert.get('message', '')) > 50 else alert.get('message', ''),
                alert.get('severity', 'INFO')
            ])

        return data

    def _generate_widget_data(self, widget: Dict[str, Any]) -> Dict[str, Any]:
        """根据组件类型生成模拟数据"""
        widget_type = widget['type']
        if widget_type == 'line_chart':
            return {
                'type': 'line_chart',
                'data': {
                    'labels': ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
                    'datasets': [{
                        'label': widget['config'].get('title', 'Metric'),
                        'data': [45, 52, 78, 65, 58, 72],
                    }]
                }
            }
        if widget_type == 'bar_chart':
            return {
                'type': 'bar_chart',
                'data': {
                    'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
                    'datasets': [{
                        'label': widget['config'].get('title', 'Requests'),
                        'data': [120, 135, 98, 142, 156],
                    }]
                }
            }
        if widget_type == 'gauge':
            return {
                'type': 'gauge',
                'value': 75,
                'max': 100,
                'label': widget['config'].get('title', 'Usage'),
            }
        if widget_type == 'table':
            return {
                'type': 'table',
                'headers': ['时间', '组件', '状态'],
                'rows': [
                    ['12:00', 'feature_engine', 'OK'],
                    ['12:05', 'feature_processor', 'WARN'],
                ]
            }
        # 默认返回简洁结构
        return {
            'type': widget_type,
            'data': widget.get('config', {})
        }

    def _get_chart_color(self, index: int) -> str:
        """获取图表颜色"""
        colors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
            '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
        ]
        return colors[index % len(colors)]

    def generate_html_dashboard(self) -> str:
        """
        生成HTML监控面板

        Returns:
            HTML内容
        """
        html_template = """
            <!DOCTYPE html>
<html lang="zh - CN">
    <head>
    <meta charset="UTF - 8">
    <meta name="viewport" content="width=device - width, initial - scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net / npm / chart.js"></script>
    <style>
        body {{
            font - family: Arial, sans - serif;
            margin: 0;
            padding: 20px;
            background - color: #f5f5f5;
        }}
        .dashboard {{
            max - width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear - gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border - radius: 10px;
            margin - bottom: 20px;
            text - align: center;
        }}
        .charts - grid {{
            display: grid;
            grid - template - columns: repeat(auto - fit, minmax(400px, 1fr));
            gap: 20px;
            margin - bottom: 20px;
        }}
        .chart - container {{
            background: white;
            border - radius: 10px;
            padding: 20px;
            box - shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart - title {{
            font - size: 18px;
            font - weight: bold;
            margin - bottom: 15px;
            color: #333;
        }}
        .status - indicator {{
            display: inline - block;
            width: 12px;
            height: 12px;
            border - radius: 50%;
            margin - right: 8px;
        }}
        .status - online {{ background - color: #4CAF50; }}
        .status - offline {{ background - color: #f44336; }}
        .status - warning {{ background - color: #ff9800; }}
        .refresh - info {{
            text - align: center;
            color: #666;
            font - size: 12px;
            margin - top: 20px;
        }}
        .table - container {{
            overflow - x: auto;
        }}
        table {{
            width: 100%;
            border - collapse: collapse;
        }}
        th, td {{
            padding: 8px;
            text - align: left;
            border - bottom: 1px solid #ddd;
        }}
        th {{
            background - color: #f2f2f2;
            font - weight: bold;
        }}
        tr:hover {{
            background - color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>{title}</h1>
            <p>最后更新: <span id="last - update">{last_update}</span></p>
            <div>
                <span class="status - indicator status - online"></span>
                <span>监控系统运行中</span>
            </div>
        </div>

        <div class="charts - grid">
            {charts_html}
        </div>

        <div class="refresh - info">
            自动刷新间隔: {refresh_interval}秒
        </div>
    </div>

    <script>
        // 自动刷新
        setInterval(function() {{
            location.reload();
        }}, {refresh_interval_ms});

        // 更新时间显示
        function updateLastUpdate() {{
            const now = new Date();
            document.getElementById('last - update').textContent = now.toLocaleString();
        }}
        setInterval(updateLastUpdate, 1000);
    </script>
</body>
</html>
        """

        charts_html = ""
        for chart_id, chart_config in self.charts.items():
            chart_data = self.get_chart_data(chart_config)
            charts_html += self._generate_chart_html(chart_id, chart_config, chart_data)

        return html_template.format(
            title=self.dashboard_config.title,
            last_update=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            charts_html=charts_html,
            refresh_interval=self.dashboard_config.refresh_interval,
            refresh_interval_ms=int(self.dashboard_config.refresh_interval * 1000)
        )

    def _generate_chart_html(self, chart_id: str, chart_config: ChartConfig,


                             chart_data: Dict[str, Any]) -> str:
        """生成图表HTML"""
        if chart_config.chart_type == ChartType.TABLE:
            return self._generate_table_html(chart_id, chart_config, chart_data)
        else:
            return self._generate_chart_js_html(chart_id, chart_config, chart_data)

    def _generate_chart_js_html(self, chart_id: str, chart_config: ChartConfig,


                                chart_data: Dict[str, Any]) -> str:
        """生成Chart.js图表HTML"""
        chart_type = chart_config.chart_type.value

        if chart_type == 'gauge':
            # 仪表盘使用特殊处理
            return f"""
            <div class="chart - container">
                <div class="chart - title">{chart_config.title}</div>
                <canvas id="{chart_id}" width="300" height="200"></canvas>
                <script>
                    const ctx = document.getElementById('{chart_id}').getContext('2d');
                    new Chart(ctx, {{
                        type: 'doughnut',
                        data: {{
                            labels: {json.dumps(chart_data.get('components', []))},
                            datasets: [{{
                                data: {json.dumps(chart_data.get('values', []))},
                                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                legend: {{
                                    position: 'bottom'
                                }}
                            }}
                        }}
                    }});
                </script>
            </div>
            """
        else:
            # 其他图表类型
            return f"""
            <div class="chart - container">
                <div class="chart - title">{chart_config.title}</div>
                <canvas id="{chart_id}" width="400" height="300"></canvas>
                <script>
                    const ctx = document.getElementById('{chart_id}').getContext('2d');
                    new Chart(ctx, {{
                        type: '{chart_type}',
                        data: {{
                            labels: {json.dumps(chart_data.get('labels', []))},
                            datasets: {json.dumps(chart_data.get('datasets', []))}
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{
                                    beginAtZero: true
                                }}
                            }}
                        }}
                    }});
                </script>
            </div>
            """

    def _generate_table_html(self, chart_id: str, chart_config: ChartConfig,


                             chart_data: Dict[str, Any]) -> str:
        """生成表格HTML"""
        headers = chart_data.get('headers', [])
        rows = chart_data.get('rows', [])

        table_html = "<table>"
        if headers:
            table_html += "<thead><tr>"
            for header in headers:
                table_html += f"<th>{header}</th>"
            table_html += "</tr></thead>"

        table_html += "<tbody>"
        for row in rows:
            table_html += "<tr>"
            for cell in row:
                table_html += f"<td>{cell}</td>"
            table_html += "</tr>"
        table_html += "</tbody></table>"

        return f"""
        <div class="chart - container">
            <div class="chart - title">{chart_config.title}</div>
            <div class="table - container">
                {table_html}
            </div>
        </div>
        """

    def start_dashboard(self, auto_open: bool = True) -> None:
        """
        启动监控面板

        Args:
            auto_open: 是否自动打开浏览器
        """
        if self.is_running:
            logger.warning("监控面板已在运行")
            return

        self.is_running = True

        # 生成HTML文件
        html_content = self.generate_html_dashboard()
        html_file = self.output_dir / "dashboard.html"

        with open(html_file, 'w', encoding='utf - 8') as f:
            f.write(html_content)

        logger.info(f"监控面板已生成: {html_file}")

        # 自动打开浏览器
        if auto_open:
            try:
                webbrowser.open(f"file://{html_file.absolute()}")
                logger.info("已在浏览器中打开监控面板")
            except Exception as e:
                logger.warning(f"无法自动打开浏览器: {e}")

        # 启动自动刷新线程
        if self.dashboard_config.auto_refresh:
            self.dashboard_thread = threading.Thread(
                target=self._auto_refresh_dashboard, daemon=True)
            self.dashboard_thread.start()

    def _auto_refresh_dashboard(self) -> None:
        """自动刷新面板"""
        while self.is_running:
            time.sleep(self.dashboard_config.refresh_interval)

            try:
                # 重新生成HTML
                html_content = self.generate_html_dashboard()
                html_file = self.output_dir / "dashboard.html"

                with open(html_file, 'w', encoding='utf - 8') as f:
                    f.write(html_content)

                self.last_update = time.time()
                logger.debug("监控面板已刷新")
            except Exception as e:
                logger.error(f"刷新监控面板失败: {e}")

    def stop_dashboard(self) -> None:
        """停止监控面板"""
        self.is_running = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=5.0)
        logger.info("监控面板已停止")

    def export_dashboard_config(self, file_path: str) -> None:
        """
        导出面板配置

        Args:
            file_path: 配置文件路径
        """
        config = {
            'dashboard_config': asdict(self.dashboard_config),
            'charts': {chart_id: asdict(chart_config) for chart_id, chart_config in self.charts.items()},
            'exported_at': datetime.now().isoformat()
        }

        with open(file_path, 'w', encoding='utf - 8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"面板配置已导出到: {file_path}")

    def import_dashboard_config(self, file_path: str) -> None:
        """
        导入面板配置

        Args:
            file_path: 配置文件路径
        """
        with open(file_path, 'r', encoding='utf - 8') as f:
            config = json.load(f)

        # 更新面板配置
        if 'dashboard_config' in config:
            self.dashboard_config = DashboardConfig(**config['dashboard_config'])

        # 更新图表配置
        if 'charts' in config:
            self.charts.clear()
            for chart_id, chart_data in config['charts'].items():
                chart_config = ChartConfig(**chart_data)
                self.charts[chart_id] = chart_config

        logger.info(f"面板配置已从 {file_path} 导入")

    def get_dashboard_status(self) -> Dict[str, Any]:
        """
        获取面板状态

        Returns:
            面板状态信息
        """
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'charts_count': len(self.charts),
            'refresh_interval': self.dashboard_config.refresh_interval,
            'auto_refresh': self.dashboard_config.auto_refresh,
            'output_dir': str(self.output_dir)
        }


def get_dashboard(config: Optional[Dict] = None) -> MonitoringDashboard:
    """
    获取监控面板实例

    Args:
        config: 配置参数

    Returns:
        监控面板实例
    """
    return MonitoringDashboard(config)
