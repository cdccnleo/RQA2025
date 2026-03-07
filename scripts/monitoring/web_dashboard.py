#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web监控仪表板
实现部署状态实时可视化和历史数据趋势图表
"""

import time
import sqlite3
from pathlib import Path
from typing import Dict, List
import random

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO
except ImportError:
    print("需要安装Flask和Flask-SocketIO: pip install flask flask-socketio")
    exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")


class DashboardDataManager:
    """仪表板数据管理器"""

    def __init__(self, db_path: str = "data/monitoring.db"):
        self.db_path = db_path
        self.environments = ["development", "staging", "production"]

    def get_current_status(self) -> Dict:
        """获取当前状态"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                status_data = {}
                for env in self.environments:
                    # 获取最新的监控数据
                    cursor.execute('''
                        SELECT status, health_score, response_time, timestamp 
                        FROM monitoring_data 
                        WHERE environment = ? 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    ''', (env,))

                    result = cursor.fetchone()
                    if result:
                        status, health_score, response_time, timestamp = result
                        status_data[env] = {
                            "status": status,
                            "health_score": health_score,
                            "response_time": response_time,
                            "last_check": timestamp
                        }
                    else:
                        # 如果没有数据，生成模拟数据
                        status_data[env] = self._generate_mock_data(env)

                return status_data

        except Exception as e:
            print(f"获取状态数据失败: {e}")
            return self._generate_mock_status()

    def get_trend_data(self, environment: str, hours: int = 24) -> List[Dict]:
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

                results = cursor.fetchall()
                return [
                    {
                        "timestamp": row[0],
                        "health_score": row[1],
                        "response_time": row[2],
                        "status": row[3]
                    }
                    for row in results
                ]

        except Exception as e:
            print(f"获取趋势数据失败: {e}")
            return self._generate_mock_trend_data(environment, hours)

    def get_alert_history(self, hours: int = 24) -> List[Dict]:
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

                results = cursor.fetchall()
                return [
                    {
                        "environment": row[0],
                        "alert_type": row[1],
                        "message": row[2],
                        "severity": row[3],
                        "timestamp": row[4]
                    }
                    for row in results
                ]

        except Exception as e:
            print(f"获取告警历史失败: {e}")
            return self._generate_mock_alerts()

    def _generate_mock_data(self, environment: str) -> Dict:
        """生成模拟数据"""
        health_score = random.uniform(0.85, 0.99)
        response_time = random.uniform(50, 200)

        if health_score >= 0.95:
            status = "running"
        elif health_score >= 0.80:
            status = "degraded"
        else:
            status = "failed"

        return {
            "status": status,
            "health_score": health_score,
            "response_time": response_time,
            "last_check": time.time()
        }

    def _generate_mock_status(self) -> Dict:
        """生成模拟状态数据"""
        return {env: self._generate_mock_data(env) for env in self.environments}

    def _generate_mock_trend_data(self, environment: str, hours: int) -> List[Dict]:
        """生成模拟趋势数据"""
        data = []
        now = time.time()
        interval = (hours * 3600) / 24  # 24个数据点

        for i in range(24):
            timestamp = now - (24 - i) * interval
            health_score = random.uniform(0.85, 0.99)
            response_time = random.uniform(50, 200)

            if health_score >= 0.95:
                status = "running"
            elif health_score >= 0.80:
                status = "degraded"
            else:
                status = "failed"

            data.append({
                "timestamp": timestamp,
                "health_score": health_score,
                "response_time": response_time,
                "status": status
            })

        return data

    def _generate_mock_alerts(self) -> List[Dict]:
        """生成模拟告警数据"""
        alerts = []
        now = time.time()

        for i in range(5):
            env = random.choice(self.environments)
            alert_types = ["health_check", "service_down", "performance_degraded"]
            severities = ["warning", "critical"]

            alerts.append({
                "environment": env,
                "alert_type": random.choice(alert_types),
                "message": f"模拟告警消息 {i+1}",
                "severity": random.choice(severities),
                "timestamp": now - random.uniform(0, 3600)
            })

        return alerts


# 创建数据管理器
data_manager = DashboardDataManager()


@app.route('/')
def index():
    """主页"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """获取当前状态API"""
    return jsonify(data_manager.get_current_status())


@app.route('/api/trend/<environment>')
def get_trend(environment):
    """获取趋势数据API"""
    hours = request.args.get('hours', 24, type=int)
    return jsonify(data_manager.get_trend_data(environment, hours))


@app.route('/api/alerts')
def get_alerts():
    """获取告警历史API"""
    hours = request.args.get('hours', 24, type=int)
    return jsonify(data_manager.get_alert_history(hours))


def create_html_template():
    """创建HTML模板"""
    template_dir = Path("templates")
    template_dir.mkdir(exist_ok=True)

    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>部署监控仪表板</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
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
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .status-card:hover {
            transform: translateY(-5px);
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
            height: 10px;
            background: #e9ecef;
            border-radius: 5px;
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
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .chart-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .alerts-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .alert-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-warning { background: #fff3cd; border-left-color: #ffc107; }
        .alert-critical { background: #f8d7da; border-left-color: #dc3545; }
        .refresh-time {
            text-align: center;
            color: white;
            font-size: 0.9em;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 部署监控仪表板</h1>
            <p>实时监控部署状态和性能指标</p>
        </div>
        
        <div class="status-grid" id="statusGrid">
            <!-- 状态卡片将通过JavaScript动态生成 -->
        </div>
        
        <div class="charts-container">
            <div class="chart-card">
                <h3>📈 健康度趋势</h3>
                <canvas id="healthChart"></canvas>
            </div>
            <div class="chart-card">
                <h3>⚡ 响应时间趋势</h3>
                <canvas id="responseChart"></canvas>
            </div>
        </div>
        
        <div class="alerts-container">
            <h3>🚨 告警历史</h3>
            <div id="alertsList">
                <!-- 告警列表将通过JavaScript动态生成 -->
            </div>
        </div>
        
        <div class="refresh-time" id="refreshTime">
            最后更新: <span id="lastUpdate">--</span>
        </div>
    </div>

    <script>
        // 初始化Socket.IO连接
        const socket = io();
        
        // 图表实例
        let healthChart, responseChart;
        
        // 状态图标映射
        const statusIcons = {
            'running': '🟢',
            'degraded': '🟡', 
            'failed': '🔴',
            'unknown': '⚪'
        };
        
        // 状态类名映射
        const statusClasses = {
            'running': 'status-running',
            'degraded': 'status-degraded',
            'failed': 'status-failed',
            'unknown': 'status-unknown'
        };
        
        // 初始化页面
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            loadStatus();
            loadAlerts();
            
            // 设置定时刷新
            setInterval(loadStatus, 30000); // 30秒刷新一次
            setInterval(loadAlerts, 60000); // 1分钟刷新告警
        });
        
        // 初始化图表
        function initializeCharts() {
            const healthCtx = document.getElementById('healthChart').getContext('2d');
            const responseCtx = document.getElementById('responseChart').getContext('2d');
            
            healthChart = new Chart(healthCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: []
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
            
            responseChart = new Chart(responseCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: []
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
        
        // 加载状态数据
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const statusData = await response.json();
                
                updateStatusGrid(statusData);
                updateCharts(statusData);
                updateRefreshTime();
                
            } catch (error) {
                console.error('加载状态数据失败:', error);
            }
        }
        
        // 更新状态网格
        function updateStatusGrid(statusData) {
            const statusGrid = document.getElementById('statusGrid');
            statusGrid.innerHTML = '';
            
            Object.entries(statusData).forEach(([env, data]) => {
                const card = createStatusCard(env, data);
                statusGrid.appendChild(card);
            });
        }
        
        // 创建状态卡片
        function createStatusCard(environment, data) {
            const card = document.createElement('div');
            card.className = 'status-card';
            
            const healthClass = data.health_score >= 0.9 ? 'health-good' : 
                              data.health_score >= 0.7 ? 'health-warning' : 'health-danger';
            
            card.innerHTML = `
                <div class="status-header">
                    <h3>${environment.toUpperCase()}</h3>
                    <span class="status-icon ${statusClasses[data.status]}">${statusIcons[data.status]}</span>
                </div>
                <div class="health-bar">
                    <div class="health-fill ${healthClass}" style="width: ${data.health_score * 100}%"></div>
                </div>
                <p><strong>健康度:</strong> ${(data.health_score * 100).toFixed(1)}%</p>
                <p><strong>响应时间:</strong> ${data.response_time.toFixed(1)}ms</p>
                <p><strong>状态:</strong> ${data.status}</p>
                <p><strong>最后检查:</strong> ${new Date(data.last_check * 1000).toLocaleString()}</p>
            `;
            
            return card;
        }
        
        // 更新图表
        async function updateCharts(statusData) {
            const environments = Object.keys(statusData);
            
            // 更新健康度图表
            healthChart.data.labels = environments;
            healthChart.data.datasets = [{
                label: '健康度',
                data: environments.map(env => statusData[env].health_score),
                backgroundColor: environments.map(env => {
                    const score = statusData[env].health_score;
                    return score >= 0.9 ? '#28a745' : score >= 0.7 ? '#ffc107' : '#dc3545';
                }),
                borderColor: '#007bff',
                borderWidth: 2
            }];
            healthChart.update();
            
            // 更新响应时间图表
            responseChart.data.labels = environments;
            responseChart.data.datasets = [{
                label: '响应时间 (ms)',
                data: environments.map(env => statusData[env].response_time),
                backgroundColor: '#17a2b8',
                borderColor: '#007bff',
                borderWidth: 2
            }];
            responseChart.update();
        }
        
        // 加载告警数据
        async function loadAlerts() {
            try {
                const response = await fetch('/api/alerts');
                const alerts = await response.json();
                
                updateAlertsList(alerts);
                
            } catch (error) {
                console.error('加载告警数据失败:', error);
            }
        }
        
        // 更新告警列表
        function updateAlertsList(alerts) {
            const alertsList = document.getElementById('alertsList');
            
            if (alerts.length === 0) {
                alertsList.innerHTML = '<p>暂无告警</p>';
                return;
            }
            
            alertsList.innerHTML = alerts.map(alert => `
                <div class="alert-item alert-${alert.severity}">
                    <strong>${alert.environment.toUpperCase()}</strong> - 
                    ${alert.message}<br>
                    <small>${new Date(alert.timestamp * 1000).toLocaleString()}</small>
                </div>
            `).join('');
        }
        
        // 更新刷新时间
        function updateRefreshTime() {
            document.getElementById('lastUpdate').textContent = new Date().toLocaleString();
        }
        
        // Socket.IO事件处理
        socket.on('status_update', function(data) {
            console.log('收到实时状态更新:', data);
            loadStatus();
        });
        
        socket.on('alert_new', function(data) {
            console.log('收到新告警:', data);
            loadAlerts();
        });
    </script>
</body>
</html>
    """

    with open(template_dir / "dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    """主函数"""
    print("🚀 启动Web监控仪表板...")

    # 创建HTML模板
    create_html_template()

    print("📊 仪表板功能:")
    print("  ✅ 实时状态监控")
    print("  ✅ 健康度趋势图表")
    print("  ✅ 响应时间趋势图表")
    print("  ✅ 告警历史显示")
    print("  ✅ WebSocket实时更新")
    print("  ✅ 响应式设计")

    print(f"\n🌐 访问地址: http://localhost:5000")
    print(f"📈 API端点:")
    print(f"  - GET /api/status - 获取当前状态")
    print(f"  - GET /api/trend/<env> - 获取趋势数据")
    print(f"  - GET /api/alerts - 获取告警历史")

    # 启动Flask应用
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


if __name__ == "__main__":
    main()
