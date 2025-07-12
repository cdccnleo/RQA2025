import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import requests
import logging
from datetime import datetime, timedelta
import threading
import time

logger = logging.getLogger(__name__)

class ResourceDashboard:
    """资源监控仪表板"""
    
    def __init__(self, api_base_url="http://localhost:8000/api/v1"):
        self.app = dash.Dash(__name__)
        self.api_base_url = api_base_url
        self.data = {
            "system": [],
            "gpu": []
        }
        
        # 设置布局
        self.app.layout = self._create_layout()
        
        # 注册回调
        self._register_callbacks()
        
        # 启动数据更新线程
        self.update_thread = threading.Thread(
            target=self._update_data_thread,
            daemon=True
        )
        self.update_thread.start()
    
    def _create_layout(self):
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
                    interval=5*1000,  # 5秒
                    n_intervals=0
                )
            ], className="row"),
            
            # GPU资源部分
            html.Div([
                html.H2("GPU资源使用情况"),
                dcc.Graph(id="gpu-memory-chart"),
                dcc.Graph(id="gpu-utilization-chart"),
                dcc.Interval(
                    id="gpu-update-interval",
                    interval=5*1000,  # 5秒
                    n_intervals=0
                )
            ], className="row"),
            
            # 告警部分
            html.Div([
                html.H2("系统告警"),
                html.Div(id="alerts-container")
            ], className="row")
        ])
    
    def _register_callbacks(self):
        """注册Dash回调函数"""
        
        @self.app.callback(
            [Output("system-cpu-chart", "figure"),
             Output("system-memory-chart", "figure"),
             Output("system-disk-chart", "figure")],
            [Input("system-update-interval", "n_intervals")]
        )
        def update_system_charts(n):
            """更新系统资源图表"""
            cpu_fig = self._create_system_chart("CPU使用率", "%", "cpu")
            mem_fig = self._create_system_chart("内存使用率", "%", "memory")
            disk_fig = self._create_system_chart("磁盘使用率", "%", "disk")
            return cpu_fig, mem_fig, disk_fig
        
        @self.app.callback(
            [Output("gpu-memory-chart", "figure"),
             Output("gpu-utilization-chart", "figure")],
            [Input("gpu-update-interval", "n_intervals")]
        )
        def update_gpu_charts(n):
            """更新GPU图表"""
            mem_fig = self._create_gpu_chart("GPU显存使用", "MB", "memory")
            util_fig = self._create_gpu_chart("GPU利用率", "%", "utilization")
            return mem_fig, util_fig
        
        @self.app.callback(
            [Output("strategies-table", "children"),
             Output("strategy-filter", "options")],
            [Input("strategies-update-interval", "n_intervals"),
             Input("strategy-filter", "value")]
        )
        def update_strategies(n, selected_strategies):
            """更新策略资源显示"""
            try:
                response = requests.get(f"{self.api_base_url}/strategies")
                if response.status_code != 200:
                    return [], []
                
                data = response.json()
                strategies = data.get("strategies", [])
                
                # 过滤策略
                if selected_strategies:
                    strategies = [s for s in strategies if s["name"] in selected_strategies]
                
                # 生成策略表格
                rows = []
                for strategy in strategies:
                    cpu_usage = "N/A"
                    gpu_usage = "N/A"
                    
                    # 工作线程进度条
                    worker_percent = (strategy["workers"] / strategy["quota"]["max_workers"]) * 100 if strategy["quota"]["max_workers"] > 0 else 0
                    worker_bar = html.Div(
                        className="progress-bar",
                        children=[
                            html.Div(
                                className=f"progress-bar-fill {'quota-exceeded' if worker_percent > 100 else ''}",
                                style={"width": f"{min(worker_percent, 100)}%"},
                                children=f"{strategy['workers']}/{strategy['quota']['max_workers']}"
                            )
                        ]
                    )
                    
                    rows.append(html.Tr([
                        html.Td(strategy["name"]),
                        html.Td(worker_bar),
                        html.Td(f"{cpu_usage}%"),
                        html.Td(f"{gpu_usage}MB")
                    ]))
                
                table = html.Table(
                    className="table",
                    children=[
                        html.Thead(html.Tr([
                            html.Th("策略名称"),
                            html.Th("工作线程"),
                            html.Th("CPU使用"),
                            html.Th("GPU显存")
                        ])),
                        html.Tbody(rows)
                    ]
                )
                
                # 策略筛选选项
                options = [{"label": s["name"], "value": s["name"]} for s in data.get("strategies", [])]
                
                return table, options
                
            except Exception as e:
                logger.error(f"更新策略数据失败: {e}")
                return [], []
        
        @self.app.callback(
            Output("alerts-container", "children"),
            [Input("system-update-interval", "n_intervals")]
        )
        def update_alerts(n):
            """更新告警显示"""
            # TODO: 集成告警系统
            return html.Div("暂无告警")
    
    def _create_system_chart(self, title, y_title, metric):
        """创建系统资源图表"""
        data = []
        if self.data["system"]:
            x = [d["timestamp"] for d in self.data["system"]]
            y = [d[metric]["percent"] for d in self.data["system"]]
            
            data.append(
                go.Scatter(
                    x=x, y=y,
                    mode="lines+markers",
                    name=title
                )
            )
        
        return {
            "data": data,
            "layout": go.Layout(
                title=title,
                yaxis={"title": y_title},
                xaxis={"title": "时间"}
            )
        }
    
    def _create_gpu_chart(self, title, y_title, metric):
        """创建GPU图表"""
        fig = go.Figure()
        
        if self.data["gpu"]:
            for gpu in self.data["gpu"][0].get("gpus", []):
                x = [d["timestamp"] for d in self.data["gpu"]]
                y = []
                for d in self.data["gpu"]:
                    for g in d.get("gpus", []):
                        if g["index"] == gpu["index"]:
                            if metric == "memory":
                                y.append(g["memory"]["allocated"] / (1024 * 1024))  # 转换为MB
                            else:
                                y.append(g[metric])
                            break
                
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        mode="lines+markers",
                        name=f"GPU {gpu['index']}: {gpu['name']}"
                    )
                )
        
        fig.update_layout(
            title=title,
            yaxis={"title": y_title},
            xaxis={"title": "时间"}
        )
        
        return fig
    
    def _update_data_thread(self):
        """后台数据更新线程"""
        while True:
            try:
                # 获取最新系统数据
                response = requests.get(f"{self.api_base_url}/system")
                if response.status_code == 200:
                    current = response.json()
                    if len(self.data["system"]) >= 100:
                        self.data["system"] = self.data["system"][-99:]
                    self.data["system"].append({
                        "timestamp": current["timestamp"],
                        "cpu": {"percent": current["cpu"]["current"]},
                        "memory": {"percent": current["memory"]["current"]},
                        "disk": {"percent": current["disk"]["current"]}
                    })
                
                # 获取最新GPU数据
                response = requests.get(f"{self.api_base_url}/gpu")
                if response.status_code == 200:
                    current = response.json()
                    if len(self.data["gpu"]) >= 100:
                        self.data["gpu"] = self.data["gpu"][-99:]
                    self.data["gpu"].append(current)
                
            except Exception as e:
                logger.error(f"更新数据失败: {e}")
            
            time.sleep(5)  # 5秒更新一次
    
    def run(self, host="0.0.0.0", port=8050):
        """运行仪表板"""
        logger.info(f"启动资源监控仪表板: http://{host}:{port}")
        self.app.run_server(host=host, port=port)
