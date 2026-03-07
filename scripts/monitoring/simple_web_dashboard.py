#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单Web监控仪表板
提供实时监控数据展示和API接口
"""

import sqlite3
import time
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'monitoring_dashboard_2025'
socketio = SocketIO(app, cors_allowed_origins="*")


class DashboardManager:
    """仪表板管理器"""

    def __init__(self, db_path: str = "data/monitoring.db"):
        self.db_path = db_path

    def get_current_status(self) -> dict:
        """获取当前监控状态"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 获取最新的监控数据
                cursor.execute('''
                    SELECT environment, status, health_score, response_time, timestamp
                    FROM monitoring_data 
                    WHERE timestamp = (
                        SELECT MAX(timestamp) 
                        FROM monitoring_data m2 
                        WHERE m2.environment = monitoring_data.environment
                    )
                    ORDER BY environment
                ''')

                environments = {}
                for row in cursor.fetchall():
                    env_name, status, health_score, response_time, timestamp = row
                    environments[env_name] = {
                        "status": status,
                        "health_score": health_score,
                        "response_time": response_time,
                        "last_check": timestamp
                    }

                return {
                    "timestamp": time.time(),
                    "environments": environments
                }

        except Exception as e:
            return {
                "timestamp": time.time(),
                "environments": {},
                "error": str(e)
            }

    def get_trend_data(self, environment: str, hours: int = 24) -> dict:
        """获取趋势数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cutoff_time = time.time() - (hours * 3600)

                cursor.execute('''
                    SELECT timestamp, health_score, response_time, status
                    FROM monitoring_data 
                    WHERE environment = ? AND timestamp > ?
                    ORDER BY timestamp ASC
                ''', (environment, cutoff_time))

                data = cursor.fetchall()

                return {
                    "environment": environment,
                    "data": [
                        {
                            "timestamp": row[0],
                            "health_score": row[1],
                            "response_time": row[2],
                            "status": row[3]
                        }
                        for row in data
                    ]
                }

        except Exception as e:
            return {
                "environment": environment,
                "data": [],
                "error": str(e)
            }

    def get_alert_history(self, hours: int = 24) -> dict:
        """获取告警历史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cutoff_time = time.time() - (hours * 3600)

                cursor.execute('''
                    SELECT environment, alert_type, message, severity, timestamp
                    FROM alert_history 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT 50
                ''', (cutoff_time,))

                alerts = []
                for row in cursor.fetchall():
                    alerts.append({
                        "environment": row[0],
                        "alert_type": row[1],
                        "message": row[2],
                        "severity": row[3],
                        "timestamp": row[4]
                    })

                return {
                    "alerts": alerts,
                    "total": len(alerts)
                }

        except Exception as e:
            return {
                "alerts": [],
                "total": 0,
                "error": str(e)
            }


# 创建仪表板管理器
dashboard_manager = DashboardManager()


@app.route('/')
def index():
    """主页"""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """API: 获取当前状态"""
    return jsonify(dashboard_manager.get_current_status())


@app.route('/api/trend/<environment>')
def api_trend(environment):
    """API: 获取趋势数据"""
    hours = request.args.get('hours', 24, type=int)
    return jsonify(dashboard_manager.get_trend_data(environment, hours))


@app.route('/api/alerts')
def api_alerts():
    """API: 获取告警历史"""
    hours = request.args.get('hours', 24, type=int)
    return jsonify(dashboard_manager.get_alert_history(hours))


@socketio.on('connect')
def handle_connect():
    """WebSocket连接处理"""
    print('客户端已连接')
    emit('status', dashboard_manager.get_current_status())


@socketio.on('request_update')
def handle_request_update():
    """处理更新请求"""
    emit('status', dashboard_manager.get_current_status())


def create_templates():
    """创建HTML模板"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    dashboard_html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQA2025 监控仪表板</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .status-card:hover {
            transform: translateY(-2px);
        }
        .status-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .status-icon {
            font-size: 2em;
        }
        .status-running { color: #28a745; }
        .status-degraded { color: #ffc107; }
        .status-failed { color: #dc3545; }
        .status-unknown { color: #6c757d; }
        .health-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .health-fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        .health-good { background: #28a745; }
        .health-warning { background: #ffc107; }
        .health-danger { background: #dc3545; }
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }
        .metric {
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #495057;
        }
        .metric-label {
            font-size: 0.9em;
            color: #6c757d;
        }
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .chart-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .alerts-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .alert-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-warning { background: #fff3cd; border-color: #ffc107; }
        .alert-critical { background: #f8d7da; border-color: #dc3545; }
        .refresh-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 RQA2025 监控仪表板</h1>
            <p>实时监控系统状态和性能指标</p>
        </div>
        
        <button class="refresh-btn" onclick="refreshData()">🔄 刷新数据</button>
        
        <div class="status-grid" id="statusGrid">
            <!-- 状态卡片将通过JavaScript动态生成 -->
        </div>
        
        <div class="charts-container">
            <div class="chart-card">
                <h3>📈 健康度趋势</h3>
                <canvas id="healthChart"></canvas>
            </div>
            <div class="chart-card">
                <h3>⏱️ 响应时间趋势</h3>
                <canvas id="responseChart"></canvas>
            </div>
        </div>
        
        <div class="alerts-container">
            <h3>🚨 告警历史</h3>
            <div id="alertsList">
                <!-- 告警列表将通过JavaScript动态生成 -->
            </div>
        </div>
    </div>

    <script>
        // 初始化Socket.IO
        const socket = io();
        
        // 图表实例
        let healthChart = null;
        let responseChart = null;
        
        // 状态图标映射
        const statusIcons = {
            'running': '🟢',
            'degraded': '🟡', 
            'failed': '🔴',
            'unknown': '⚪'
        };
        
        // 初始化页面
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            loadInitialData();
            
            // 设置定时刷新
            setInterval(refreshData, 30000); // 每30秒刷新一次
        });
        
        // Socket.IO事件处理
        socket.on('status', function(data) {
            updateStatusDisplay(data);
        });
        
        function initializeCharts() {
            // 健康度趋势图
            const healthCtx = document.getElementById('healthChart').getContext('2d');
            healthChart = new Chart(healthCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '健康度',
                        data: [],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            ticks: {
                                callback: function(value) {
                                    return (value * 100).toFixed(0) + '%';
                                }
                            }
                        }
                    }
                }
            });
            
            // 响应时间趋势图
            const responseCtx = document.getElementById('responseChart').getContext('2d');
            responseChart = new Chart(responseCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '响应时间 (ms)',
                        data: [],
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        function loadInitialData() {
            refreshData();
            loadAlerts();
        }
        
        function refreshData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateStatusDisplay(data);
                    updateCharts(data);
                })
                .catch(error => {
                    console.error('获取状态数据失败:', error);
                });
        }
        
        function updateStatusDisplay(data) {
            const statusGrid = document.getElementById('statusGrid');
            statusGrid.innerHTML = '';
            
            if (data.environments) {
                Object.entries(data.environments).forEach(([env, info]) => {
                    const card = createStatusCard(env, info);
                    statusGrid.appendChild(card);
                });
            }
        }
        
        function createStatusCard(environment, info) {
            const card = document.createElement('div');
            card.className = 'status-card';
            
            const statusClass = `status-${info.status}`;
            const healthClass = info.health_score >= 0.8 ? 'health-good' : 
                              info.health_score >= 0.6 ? 'health-warning' : 'health-danger';
            
            card.innerHTML = `
                <div class="status-header">
                    <h3>${environment.toUpperCase()}</h3>
                    <span class="status-icon ${statusClass}">${statusIcons[info.status] || '⚪'}</span>
                </div>
                <div class="health-bar">
                    <div class="health-fill ${healthClass}" style="width: ${info.health_score * 100}%"></div>
                </div>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">${(info.health_score * 100).toFixed(1)}%</div>
                        <div class="metric-label">健康度</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${info.response_time.toFixed(1)}ms</div>
                        <div class="metric-label">响应时间</div>
                    </div>
                </div>
            `;
            
            return card;
        }
        
        function updateCharts(data) {
            if (data.environments) {
                const environments = Object.keys(data.environments);
                if (environments.length > 0) {
                    // 更新第一个环境的趋势数据
                    loadTrendData(environments[0]);
                }
            }
        }
        
        function loadTrendData(environment) {
            fetch(`/api/trend/${environment}?hours=24`)
                .then(response => response.json())
                .then(data => {
                    if (data.data && data.data.length > 0) {
                        updateHealthChart(data.data);
                        updateResponseChart(data.data);
                    }
                })
                .catch(error => {
                    console.error('获取趋势数据失败:', error);
                });
        }
        
        function updateHealthChart(data) {
            const labels = data.map(item => new Date(item.timestamp * 1000).toLocaleTimeString());
            const values = data.map(item => item.health_score);
            
            healthChart.data.labels = labels;
            healthChart.data.datasets[0].data = values;
            healthChart.update();
        }
        
        function updateResponseChart(data) {
            const labels = data.map(item => new Date(item.timestamp * 1000).toLocaleTimeString());
            const values = data.map(item => item.response_time);
            
            responseChart.data.labels = labels;
            responseChart.data.datasets[0].data = values;
            responseChart.update();
        }
        
        function loadAlerts() {
            fetch('/api/alerts?hours=24')
                .then(response => response.json())
                .then(data => {
                    updateAlertsDisplay(data.alerts);
                })
                .catch(error => {
                    console.error('获取告警数据失败:', error);
                });
        }
        
        function updateAlertsDisplay(alerts) {
            const alertsList = document.getElementById('alertsList');
            
            if (alerts.length === 0) {
                alertsList.innerHTML = '<p>暂无告警</p>';
                return;
            }
            
            alertsList.innerHTML = alerts.map(alert => `
                <div class="alert-item alert-${alert.severity}">
                    <strong>${alert.environment}</strong> - ${alert.message}
                    <br><small>${new Date(alert.timestamp * 1000).toLocaleString()}</small>
                </div>
            `).join('');
        }
    </script>
</body>
</html>'''

    with open(templates_dir / "dashboard.html", 'w', encoding='utf-8') as f:
        f.write(dashboard_html)


def main():
    """主函数"""
    print("🚀 启动Web监控仪表板...")

    # 创建模板文件
    create_templates()

    print("📊 仪表板功能:")
    print("  ✅ 实时状态监控")
    print("  ✅ 健康度趋势图")
    print("  ✅ 响应时间趋势图")
    print("  ✅ 告警历史显示")
    print("  ✅ WebSocket实时更新")
    print("  ✅ RESTful API接口")

    print(f"\n🌐 访问地址: http://localhost:5000")
    print(f"📡 WebSocket: ws://localhost:5000")
    print(f"📋 API接口:")
    print(f"  GET /api/status - 获取当前状态")
    print(f"  GET /api/trend/<environment> - 获取趋势数据")
    print(f"  GET /api/alerts - 获取告警历史")

    # 启动Flask应用
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


if __name__ == "__main__":
    main()
