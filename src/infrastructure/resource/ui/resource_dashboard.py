"""
resource_dashboard 模块

提供 resource_dashboard 相关功能和接口。
"""

import logging
import requests

import dash
import threading
import time

from dash import html, dcc
from dash.dependencies import Input, Output
from dataclasses import dataclass
from typing import Dict, List
"""
基础设施层 - 资源监控仪表板组件

resource_dashboard 模块

资源监控仪表板相关的文件
提供资源监控的可视化功能实现。
"""

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """仪表板配置"""
    api_base_url: str = "http://localhost:8000/api/v1"
    update_interval: int = 5000  # 5秒
    max_data_points: int = 100
    theme: str = "light"


class ResourceDashboardUI:
    """资源仪表板UI管理器"""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def create_layout(self) -> html.Div:
        """创建仪表板布局"""
        return html.Div([
            html.H1("RQA2025 资源监控看板"),

            # 系统资源部分
            html.Div([
                html.H2("系统资源使用情况"),
                dcc.Graph(id="system-cpu-chart"),
                dcc.Graph(id="system-memory-chart"),
                dcc.Graph(id="system-disk-chart"),
                dcc.Interval(
                    id="system-update-interval",
                    interval=self.config.update_interval,
                    n_intervals=0
                )
            ]),

            # GPU资源部分
            html.Div([
                html.H2("GPU资源使用情况"),
                dcc.Graph(id="gpu-memory-chart"),
                dcc.Graph(id="gpu-utilization-chart"),
                dcc.Interval(
                    id="gpu-update-interval",
                    interval=self.config.update_interval,
                    n_intervals=0
                )
            ]),

            # 告警部分
            html.Div([
                html.H2("系统告警"),
                html.Div(id="alerts-container")
            ])
        ])


class ResourceDashboardData:
    """资源仪表板数据管理器"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.data = {
            "system": [],
            "gpu": []
        }
        self._running = False
        self._update_thread = None

    def start_data_collection(self) -> None:
        """启动数据收集"""
        if self._running:
            return

        self._running = True
        self._update_thread = threading.Thread(target=self._data_update_loop, daemon=True)
        self._update_thread.start()

    def stop_data_collection(self) -> None:
        """停止数据收集"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)

    def _data_update_loop(self) -> None:
        """数据更新循环"""
        while self._running:
            try:
                self._fetch_system_data()
                self._fetch_gpu_data()
                time.sleep(self.config.update_interval / 1000)
            except Exception as e:
                logger.error(f"Data update error: {e}")
                time.sleep(5)

    def _fetch_system_data(self) -> None:
        """获取系统数据"""
        try:
            response = requests.get(f"{self.config.api_base_url}/system/stats", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.data["system"].append(data)
                # 限制数据量
                if len(self.data["system"]) > self.config.max_data_points:
                    self.data["system"] = self.data["system"][-self.config.max_data_points:]
        except Exception as e:
            logger.error(f"Failed to fetch system data: {e}")

    def _fetch_gpu_data(self) -> None:
        """获取GPU数据"""
        try:
            response = requests.get(f"{self.config.api_base_url}/gpu/stats", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.data["gpu"].append(data)
                # 限制数据量
                if len(self.data["gpu"]) > self.config.max_data_points:
                    self.data["gpu"] = self.data["gpu"][-self.config.max_data_points:]
        except Exception as e:
            logger.error(f"Failed to fetch GPU data: {e}")

    def get_system_data(self) -> List[Dict]:
        """获取系统数据"""
        return self.data["system"]

    def get_gpu_data(self) -> List[Dict]:
        """获取GPU数据"""
        return self.data["gpu"]

    def get_alerts(self) -> List[Dict]:
        """获取告警数据"""
        try:
            response = requests.get(f"{self.config.api_base_url}/alerts", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch alerts: {e}")
        return []


class ResourceDashboardCallbacks:
    """资源仪表板回调管理器"""

    def __init__(self, app: dash.Dash, data_manager: ResourceDashboardData, ui_manager: ResourceDashboardUI):
        self.app = app
        self.data_manager = data_manager
        self.ui_manager = ui_manager
        self._registered = False

    def register_callbacks(self) -> None:
        """注册所有回调函数"""
        if self._registered:
            return

        self._register_system_callbacks()
        self._register_gpu_callbacks()
        self._register_alert_callbacks()
        self._registered = True

    def _register_system_callbacks(self) -> None:
        """注册系统资源回调"""
        @self.app.callback(
            [Output("system-cpu-chart", "figure"),
             Output("system-memory-chart", "figure"),
             Output("system-disk-chart", "figure")],
            [Input("system-update-interval", "n_intervals")]
        )
        def update_system_charts(n_intervals):
            system_data = self.data_manager.get_system_data()
            return (
                self.ui_manager.create_system_cpu_chart(system_data),
                self.ui_manager.create_system_memory_chart(system_data),
                self.ui_manager.create_system_disk_chart(system_data)
            )

    def _register_gpu_callbacks(self) -> None:
        """注册GPU资源回调"""
        @self.app.callback(
            [Output("gpu-memory-chart", "figure"),
             Output("gpu-utilization-chart", "figure")],
            [Input("gpu-update-interval", "n_intervals")]
        )
        def update_gpu_charts(n_intervals):
            gpu_data = self.data_manager.get_gpu_data()
            return (
                self.ui_manager.create_gpu_memory_chart(gpu_data),
                self.ui_manager.create_gpu_utilization_chart(gpu_data)
            )

    def _register_alert_callbacks(self) -> None:
        """注册告警回调"""
        @self.app.callback(
            Output("alerts-container", "children"),
            [Input("system-update-interval", "n_intervals")]
        )
        def update_alerts(n_intervals):
            alerts = self.data_manager.get_alerts()
            if not alerts:
                return "暂无告警"

            alert_items = []
            for alert in alerts[-10:]:  # 显示最近10个告警
                alert_items.append(
                    html.Div([
                        html.Span(f"[{alert.get('level', 'info').upper()}] ", style={
                                  'color': self._get_alert_color(alert.get('level', 'info'))}),
                        html.Span(alert.get('message', 'Unknown alert')),
                        html.Small(f" {alert.get('timestamp', '')}", style={'color': 'gray'})
                    ], style={'margin': '5px 0'})
                )

            return alert_items

    def _get_alert_color(self, level: str) -> str:
        """获取告警级别对应的颜色"""
        color_map = {
            'critical': 'red',
            'high': 'orange',
            'medium': 'yellow',
            'low': 'blue',
            'info': 'green'
        }
        return color_map.get(level.lower(), 'black')


class ResourceDashboardController:
    """资源仪表板控制器"""

    def __init__(self, config: DashboardConfig = None):
        if config is None:
            config = DashboardConfig()

        self.config = config
        self.app = dash.Dash(__name__)

        # 初始化各组件
        self.ui_manager = ResourceDashboardUI(config)
        self.data_manager = ResourceDashboardData(config)
        self.callbacks_manager = ResourceDashboardCallbacks(
            self.app, self.data_manager, self.ui_manager)

        # 设置应用布局
        self.app.layout = self.ui_manager.create_layout()

        # 注册回调
        self.callbacks_manager.register_callbacks()

        # 启动数据收集
        self.data_manager.start_data_collection()

    def run_server(self, host: str = "0.0.0.0", port: int = 8050, debug: bool = False) -> None:
        """运行服务器"""
        logger.info(f"Starting Resource Dashboard on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

    def shutdown(self) -> None:
        """关闭仪表板"""
        logger.info("Shutting down Resource Dashboard")
        self.data_manager.stop_data_collection()


# 向后兼容的别名
ResourceDashboard = ResourceDashboardController
