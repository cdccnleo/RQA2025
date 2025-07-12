#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 测试数据可视化工具
用于实时展示测试数据和结果
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from threading import Thread
import time
from datetime import datetime

class TestDataVisualizer:
    def __init__(self, port=8050):
        """
        初始化数据可视化工具
        :param port: 可视化面板端口
        """
        self.port = port
        self.data = {
            "market": pd.DataFrame(columns=["timestamp", "symbol", "price", "volume"]),
            "orders": pd.DataFrame(columns=["timestamp", "symbol", "price", "quantity", "status"]),
            "performance": pd.DataFrame(columns=["timestamp", "latency", "throughput"])
        }
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """设置可视化面板布局"""
        self.app.layout = html.Div([
            html.H1("RQA2025 测试数据监控面板", style={"textAlign": "center"}),

            # 市场数据图表
            html.Div([
                dcc.Graph(id="market-price-chart"),
                dcc.Interval(id="market-update", interval=1000)
            ], style={"width": "49%", "display": "inline-block"}),

            # 订单数据图表
            html.Div([
                dcc.Graph(id="order-volume-chart"),
                dcc.Interval(id="order-update", interval=1000)
            ], style={"width": "49%", "display": "inline-block", "float": "right"}),

            # 性能指标图表
            html.Div([
                dcc.Graph(id="performance-chart"),
                dcc.Interval(id="performance-update", interval=5000)
            ], style={"width": "100%"}),

            # 测试进度
            html.Div([
                html.H3("测试进度"),
                dcc.Markdown(id="test-progress"),
                dcc.Interval(id="progress-update", interval=3000)
            ])
        ])

    def _setup_callbacks(self):
        """设置回调函数"""
        # 市场数据更新
        @self.app.callback(
            Output("market-price-chart", "figure"),
            [Input("market-update", "n_intervals")]
        )
        def update_market_chart(_):
            df = self.data["market"]
            if df.empty:
                return go.Figure()

            traces = []
            for symbol in df["symbol"].unique():
                symbol_df = df[df["symbol"] == symbol]
                traces.append(go.Scatter(
                    x=symbol_df["timestamp"],
                    y=symbol_df["price"],
                    mode="lines+markers",
                    name=symbol,
                    line=dict(width=2)
                ))

            return {
                "data": traces,
                "layout": go.Layout(
                    title="实时行情价格",
                    xaxis={"title": "时间"},
                    yaxis={"title": "价格"},
                    hovermode="closest"
                )
            }

        # 订单数据更新
        @self.app.callback(
            Output("order-volume-chart", "figure"),
            [Input("order-update", "n_intervals")]
        )
        def update_order_chart(_):
            df = self.data["orders"]
            if df.empty:
                return go.Figure()

            status_counts = df.groupby(["symbol", "status"]).size().unstack().fillna(0)

            return {
                "data": [
                    go.Bar(
                        x=status_counts.index,
                        y=status_counts[status],
                        name=status
                    ) for status in status_counts.columns
                ],
                "layout": go.Layout(
                    title="订单状态分布",
                    xaxis={"title": "股票代码"},
                    yaxis={"title": "订单数量"},
                    barmode="stack"
                )
            }

        # 性能数据更新
        @self.app.callback(
            Output("performance-chart", "figure"),
            [Input("performance-update", "n_intervals")]
        )
        def update_performance_chart(_):
            df = self.data["performance"]
            if df.empty:
                return go.Figure()

            return {
                "data": [
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["latency"],
                        name="延迟(ms)",
                        yaxis="y1",
                        line=dict(color="blue")
                    ),
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["throughput"],
                        name="吞吐量(订单/秒)",
                        yaxis="y2",
                        line=dict(color="red")
                    )
                ],
                "layout": go.Layout(
                    title="系统性能指标",
                    xaxis={"title": "时间"},
                    yaxis={"title": "延迟(ms)", "side": "left", "color": "blue"},
                    yaxis2={"title": "吞吐量(订单/秒)", "side": "right", "overlaying": "y", "color": "red"},
                    hovermode="x unified"
                )
            }

        # 测试进度更新
        @self.app.callback(
            Output("test-progress", "children"),
            [Input("progress-update", "n_intervals")]
        )
        def update_progress(_):
            # 这里可以连接实际的测试进度数据
            return """
            **当前测试进度**  
            - 单元测试: 100% (125/125)  
            - 集成测试: 78% (39/50)  
            - 性能测试: 45% (9/20)  
            
            **最近事件**  
            - 2023-11-15 14:30:22 熔断机制测试通过  
            - 2023-11-15 14:28:10 FPGA一致性测试失败  
            - 2023-11-15 14:25:05 开始压力测试  
            """

    def add_market_data(self, symbol: str, price: float, volume: float):
        """添加市场数据"""
        timestamp = datetime.now().isoformat()
        new_row = {"timestamp": timestamp, "symbol": symbol, "price": price, "volume": volume}
        self.data["market"] = pd.concat([
            self.data["market"],
            pd.DataFrame([new_row])
        ], ignore_index=True)

    def add_order_data(self, symbol: str, price: float, quantity: float, status: str):
        """添加订单数据"""
        timestamp = datetime.now().isoformat()
        new_row = {"timestamp": timestamp, "symbol": symbol, "price": price, "quantity": quantity, "status": status}
        self.data["orders"] = pd.concat([
            self.data["orders"],
            pd.DataFrame([new_row])
        ], ignore_index=True)

    def add_performance_data(self, latency: float, throughput: float):
        """添加性能数据"""
        timestamp = datetime.now().isoformat()
        new_row = {"timestamp": timestamp, "latency": latency*1000, "throughput": throughput}
        self.data["performance"] = pd.concat([
            self.data["performance"],
            pd.DataFrame([new_row])
        ], ignore_index=True)

    def run_server(self):
        """启动可视化面板"""
        print(f"📊 数据可视化面板已启动: http://localhost:{self.port}")
        self.app.run_server(port=self.port)

    def start(self):
        """启动可视化面板(后台线程)"""
        thread = Thread(target=self.run_server, daemon=True)
        thread.start()
        return thread


if __name__ == "__main__":
    # 示例用法
    visualizer = TestDataVisualizer()
    visualizer.start()

    # 模拟数据更新
    import random
    symbols = ["600519.SH", "000001.SZ", "601318.SH"]

    try:
        while True:
            # 生成随机市场数据
            symbol = random.choice(symbols)
            price = round(100 + random.random() * 10, 2)
            volume = random.randint(1000, 10000)
            visualizer.add_market_data(symbol, price, volume)

            # 生成随机订单数据
            status = random.choice(["FILLED", "PARTIAL", "REJECTED"])
            visualizer.add_order_data(symbol, price, random.randint(1, 100), status)

            # 生成随机性能数据
            visualizer.add_performance_data(random.random(), random.randint(500, 1500))

            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止数据模拟...")
