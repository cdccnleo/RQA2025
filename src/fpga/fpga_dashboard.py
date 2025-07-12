#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FPGA性能可视化监控系统
提供FPGA性能数据的可视化展示界面
"""

import time
from datetime import datetime, timedelta
import json
from flask import Flask, render_template, jsonify
from .fpga_performance_monitor import FPGAPerformanceMonitor
from .fpga_manager import FPGAManager

app = Flask(__name__)

class FPGADashboard:
    def __init__(self, performance_monitor: FPGAPerformanceMonitor):
        self.monitor = performance_monitor
        self.app = app
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def dashboard():
            """主仪表板页面"""
            return render_template('dashboard.html')

        @self.app.route('/api/current_metrics')
        def current_metrics():
            """获取当前性能指标"""
            status = FPGAManager().get_device_status()
            report = self.monitor.generate_report()

            return jsonify({
                'status': status,
                'latency': report['latency_stats'],
                'utilization': report['utilization_stats'],
                'warnings': report['warning_count'],
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api/history/<metric>')
        def history_metrics(metric):
            """获取历史性能数据"""
            hours = int(request.args.get('hours', 24))
            cutoff = datetime.now() - timedelta(hours=hours)

            if metric == 'latency':
                data = [m for m in self.monitor.metrics['latency']
                       if m['timestamp'] >= cutoff]
            elif metric == 'utilization':
                data = [m for m in self.monitor.metrics['utilization']
                       if m['timestamp'] >= cutoff]
            else:
                return jsonify({'error': 'Invalid metric'}), 400

            return jsonify(data)

        @self.app.route('/api/alerts')
        def get_alerts():
            """获取告警信息"""
            report = self.monitor.generate_report()
            alerts = []

            if report['warning_count']['latency'] > 0:
                alerts.append({
                    'type': 'latency',
                    'count': report['warning_count']['latency'],
                    'message': '高延迟告警'
                })

            if report['warning_count']['utilization'] > 0:
                alerts.append({
                    'type': 'utilization',
                    'count': report['warning_count']['utilization'],
                    'message': '高资源利用率告警'
                })

            return jsonify(alerts)

    def run(self, host='0.0.0.0', port=5000):
        """启动仪表板服务"""
        self.app.run(host=host, port=port)

if __name__ == '__main__':
    # 初始化组件
    fpga_manager = FPGAManager()
    monitor = FPGAPerformanceMonitor(fpga_manager)

    # 创建并启动仪表板
    dashboard = FPGADashboard(monitor)
    dashboard.run()
