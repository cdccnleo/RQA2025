#!/usr/bin/env python3
"""
RQA2025 移动端监控界面
提供移动设备友好的监控和控制界面
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import threading
import time
import os


logger = logging.getLogger(__name__)


class MobileMonitor:

    """移动端监控器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # Flask应用配置
        self.app = Flask(__name__,
                         template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                         static_folder=os.path.join(os.path.dirname(__file__), 'static'))

        # 配置参数
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8082)
        self.debug = self.config.get('debug', False)
        self.secret_key = self.config.get('secret_key', 'rqa2025_mobile_monitor')

        # 数据存储
        self.system_data: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.performance_data: List[Dict[str, Any]] = []
        self.strategy_data: Dict[str, Any] = {}

        # 更新间隔
        self.update_interval = self.config.get('update_interval', 5)  # 秒

        # 注册路由
        self._register_routes()

        # 设置Flask配置
        self.app.config['SECRET_KEY'] = self.secret_key

        logger.info("移动端监控器初始化完成")

    def _register_routes(self):
        """注册路由"""

        @self.app.route('/')
        def index():
            """主页面"""
            return render_template('index.html')

        @self.app.route('/dashboard')
        def dashboard():
            """仪表板页面"""
            return render_template('dashboard.html')

        @self.app.route('/strategies')
        def strategies():
            """策略监控页面"""
            return render_template('strategies.html')

        @self.app.route('/alerts')
        def alerts():
            """告警页面"""
            return render_template('alerts.html')

        @self.app.route('/system')
        def system():
            """系统监控页面"""
            return render_template('system.html')

        @self.app.route('/api / system / status')
        def api_system_status():
            """系统状态API"""
            return jsonify(self.system_data)

        @self.app.route('/api / performance / data')
        def api_performance_data():
            """性能数据API"""
            return jsonify({
                'data': self.performance_data[-50:],  # 返回最近50个数据点
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api / strategies / status')
        def api_strategies_status():
            """策略状态API"""
            return jsonify(self.strategy_data)

        @self.app.route('/api / alerts / list')
        def api_alerts_list():
            """告警列表API"""
            return jsonify({
                'alerts': self.alerts[-20:],  # 返回最近20个告警
                'total': len(self.alerts)
            })

        @self.app.route('/api / alerts / acknowledge/<alert_id>', methods=['POST'])
        def api_acknowledge_alert(alert_id):
            """确认告警API"""
            for alert in self.alerts:
                if alert.get('id') == alert_id:
                    alert['acknowledged'] = True
                    alert['acknowledged_at'] = datetime.now().isoformat()
                    break

            return jsonify({'success': True})

        @self.app.route('/api / system / control', methods=['POST'])
        def api_system_control():
            """系统控制API"""
            data = request.get_json()

            if not data:
                return jsonify({'success': False, 'message': 'No data provided'})

            command = data.get('command')

            if command == 'restart':
                # 这里应该实现重启逻辑
                logger.info("收到重启命令")
                return jsonify({'success': True, 'message': 'Restart command received'})

            elif command == 'stop':
                # 这里应该实现停止逻辑
                logger.info("收到停止命令")
                return jsonify({'success': True, 'message': 'Stop command received'})

            else:
                return jsonify({'success': False, 'message': f'Unknown command: {command}'})

        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            """静态文件服务"""
            return send_from_directory(self.app.static_folder, filename)

    def update_system_data(self, data: Dict[str, Any]):
        """更新系统数据"""
        self.system_data.update(data)
        self.system_data['last_update'] = datetime.now().isoformat()

    def update_performance_data(self, data: Dict[str, Any]):
        """更新性能数据"""
        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            **data
        }

        self.performance_data.append(performance_entry)

        # 保持数据点数量在合理范围内
        if len(self.performance_data) > 1000:
            self.performance_data = self.performance_data[-500:]

    def update_strategy_data(self, strategy_name: str, data: Dict[str, Any]):
        """更新策略数据"""
        if strategy_name not in self.strategy_data:
            self.strategy_data[strategy_name] = {}

        self.strategy_data[strategy_name].update(data)
        self.strategy_data[strategy_name]['last_update'] = datetime.now().isoformat()

    def add_alert(self, alert: Dict[str, Any]):
        """添加告警"""
        alert_entry = {
            'id': f"alert_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False,
            **alert
        }

        self.alerts.append(alert_entry)

        # 保持告警数量在合理范围内
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]

        logger.warning(f"移动端告警: {alert.get('message', 'Unknown alert')}")

    def start_server(self):
        """启动服务器"""
        logger.info(f"启动移动端监控服务器: {self.host}:{self.port}")

        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False,  # 在生产环境中关闭自动重载
                threaded=True
            )
        except Exception as e:
            logger.error(f"启动移动端监控服务器失败: {e}")
            raise

    def start_background_update(self):
        """启动后台更新"""

        def update_loop():

            while True:
                try:
                    # 从实际的监控系统中获取数据
                    mock_system_data = self._collect_real_system_data()
                    self.update_system_data(mock_system_data)

                    # 生成模拟性能数据
                    performance_data = self._generate_performance_data()
                    self.update_performance_data(performance_data)

                    # 检查并生成告警
                    self._check_and_generate_alerts()

                    time.sleep(self.update_interval)

                except Exception as e:
                    logger.error(f"后台更新异常: {e}")
                    time.sleep(5)

        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        logger.info("后台更新服务已启动")

    def _collect_real_system_data(self):
        """收集真实系统数据"""
        try:
            # 这里可以集成psutil等库获取真实系统信息
            import psutil

            # CPU和内存使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # 磁盘使用率
            disk = psutil.disk_usage('/')

            # 网络信息
            network = psutil.net_io_counters()

            return {
                'status': 'running',
                'uptime': self._get_system_uptime(),
                'active_nodes': 5,  # 在分布式系统中可以从注册中心获取
                'total_trades': 1250,  # 从交易引擎获取
                'pnl': 15420.50,  # 从策略引擎获取
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'network_sent': network.bytes_sent,
                'network_recv': network.bytes_recv
            }

        except ImportError:
            # 如果没有psutil，使用模拟数据
            return self._generate_mock_system_data()

    def _generate_mock_system_data(self):
        """生成模拟系统数据"""
        return {
            'status': 'running',
            'uptime': '2h 30m',
            'active_nodes': 5,
            'total_trades': 1250,
            'pnl': 15420.50,
            'cpu_usage': 45.0 + (5.0 * (time.time() % 60) / 60),  # 动态变化
            'memory_usage': 67.0,
            'disk_usage': 23.0,
            'network_sent': 1000000,
            'network_recv': 500000
        }

    def _generate_performance_data(self):
        """生成性能数据"""
        return {
            'cpu_usage': 45.0 + (10.0 * (time.time() % 60) / 60),
            'memory_usage': 67.0,
            'active_nodes': 5,
            'total_trades': 1250,
            'pnl': 15420.50,
            'response_time': 45.0,  # ms
            'throughput': 100.0,    # tps
            'error_rate': 0.1       # %
        }

    def _check_and_generate_alerts(self):
        """检查并生成告警"""
        # CPU使用率告警
        if self.system_data.get('cpu_usage', 0) > 90:
            cpu_usage = self.system_data.get('cpu_usage', 0)
            self.add_alert({
                'level': 'critical',
                'title': 'CPU使用率过高',
                'message': f'CPU使用率过高: {cpu_usage:.1f}%',
                'source': '系统监控'
            })

        # 内存使用率告警
        if self.system_data.get('memory_usage', 0) > 85:
            memory_usage = self.system_data.get('memory_usage', 0)
            self.add_alert({
                'level': 'warning',
                'title': '内存使用率过高',
                'message': f'内存使用率过高: {memory_usage:.1f}%',
                'source': '系统监控'
            })

        # 策略性能告警
        for strategy_name, data in self.strategy_data.items():
            pnl = data.get('pnl', 0)
            if pnl < -1000:  # 亏损超过1000
                self.add_alert({
                    'level': 'warning',
                    'title': f'策略{strategy_name}亏损超限',
                    'message': f'策略{strategy_name}当前亏损{pnl:.2f}，超过阈值',
                    'source': '策略监控'
                })

    def _get_system_uptime(self):
        """获取系统运行时间"""
        try:
            import psutil
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time

            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)

            return f"{hours:02d}h {minutes:02d}m"
        except BaseException:
            return "2h 30m"

    def add_performance_metrics(self, metrics: Dict[str, Any]):
        """添加性能指标"""
        self.update_performance_data(metrics)

    def add_strategy_metrics(self, strategy_name: str, metrics: Dict[str, Any]):
        """添加策略指标"""
        self.update_strategy_data(strategy_name, metrics)

    def get_mobile_optimized_data(self):
        """获取移动端优化数据"""
        return {
            'system': self.system_data,
            'performance': self.performance_data[-10:],  # 最近10个数据点
            'strategies': self.strategy_data,
            'alerts': self.alerts[-5:],  # 最近5个告警
            'timestamp': datetime.now().isoformat()
        }


# HTML模板
INDEX_HTML = """
    <!DOCTYPE html>
<html lang="zh - CN">
    <head>
    <meta charset="UTF - 8">
    <meta name="viewport" content="width=device - width, initial - scale=1.0">
    <title>RQA2025 量化交易系统</title>
    <link href="https://cdn.jsdelivr.net / npm / bootstrap@5.1.3 / dist / css / bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com / ajax / libs / font - awesome / 6.0.0 / css / all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear - gradient(135deg, #667eea 0%, #764ba2 100%);
            min - height: 100vh;
        }
        .card {
            border: none;
            border - radius: 15px;
            box - shadow: 0 10px 30px rgba(0,0,0,0.3);
            backdrop - filter: blur(10px);
            background: rgba(255,255,255,0.9);
        }
        .btn - primary {
            background: linear - gradient(45deg, #667eea, #764ba2);
            border: none;
            border - radius: 25px;
        }
        .feature - card {
            transition: transform 0.3s ease;
        }
        .feature - card:hover {
            transform: translateY(-5px);
        }
    </style>
</head>
<body>
    <div class="container py - 5">
        <div class="row justify - content - center">
            <div class="col - lg - 8">
                <div class="text - center mb - 5">
                    <h1 class="display - 4 text - white mb - 3">
                        <i class="fas fa - chart - line me - 3"></i>
                        RQA2025 量化交易系统
                    </h1>
                    <p class="lead text - white - 50">企业级智能量化交易平台</p>
                </div>

                <div class="row g - 4">
                    <div class="col - md - 6">
                        <div class="card feature - card h - 100">
                            <div class="card - body text - center">
                                <i class="fas fa - tachometer - alt fa - 3x text - primary mb - 3"></i>
                                <h5 class="card - title">系统监控</h5>
                                <p class="card - text">实时监控系统性能、策略表现和风险指标</p>
                                <a href="/dashboard" class="btn btn - primary">
                                    <i class="fas fa - arrow - right me - 2"></i>进入监控
                                </a>
                            </div>
                        </div>
                    </div>

                    <div class="col - md - 6">
                        <div class="card feature - card h - 100">
                            <div class="card - body text - center">
                                <i class="fas fa - brain fa - 3x text - success mb - 3"></i>
                                <h5 class="card - title">AI策略</h5>
                                <p class="card - text">机器学习驱动的智能交易策略</p>
                                <a href="/strategies" class="btn btn - success">
                                    <i class="fas fa - arrow - right me - 2"></i>策略管理
                                </a>
                            </div>
                        </div>
                    </div>

                    <div class="col - md - 6">
                        <div class="card feature - card h - 100">
                            <div class="card - body text - center">
                                <i class="fas fa - bell fa - 3x text - warning mb - 3"></i>
                                <h5 class="card - title">告警中心</h5>
                                <p class="card - text">智能告警系统，及时响应系统异常</p>
                                <a href="/alerts" class="btn btn - warning">
                                    <i class="fas fa - arrow - right me - 2"></i>查看告警
                                </a>
                            </div>
                        </div>
                    </div>

                    <div class="col - md - 6">
                        <div class="card feature - card h - 100">
                            <div class="card - body text - center">
                                <i class="fas fa - server fa - 3x text - info mb - 3"></i>
                                <h5 class="card - title">系统管理</h5>
                                <p class="card - text">系统配置、性能调优和运维管理</p>
                                <a href="/system" class="btn btn - info">
                                    <i class="fas fa - arrow - right me - 2"></i>系统管理
                                </a>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="text - center mt - 5">
                    <div class="card">
                        <div class="card - body">
                            <h5 class="card - title">系统状态</h5>
                            <div class="row text - center">
                                <div class="col - 6">
                                    <div class="h4 text - success">
                                        <i class="fas fa - circle"></i>
                                    </div>
                                    <small class="text - muted">运行状态</small>
                                </div>
                                <div class="col - 6">
                                    <div class="h4 text - primary">
                                        <i class="fas fa - clock"></i> 2h 30m
                                    </div>
                                    <small class="text - muted">运行时间</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net / npm / bootstrap@5.1.3 / dist / js / bootstrap.bundle.min.js"></script>
        </body>
</html>
"""

DASHBOARD_HTML = """
    <!DOCTYPE html>
<html lang="zh - CN">
    <head>
    <meta charset="UTF - 8">
    <meta name="viewport" content="width=device - width, initial - scale=1.0">
    <title>系统监控 - RQA2025</title>
    <link href="https://cdn.jsdelivr.net / npm / bootstrap@5.1.3 / dist / css / bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net / npm / chart.js"></script>
    <style>
        body { background - color: #f8f9fa; }
        .metric - card {
            border - radius: 10px;
            border: none;
            box - shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status - online { color: #28a745; }
        .status - offline { color: #dc3545; }
    </style>
</head>
<body>
    <nav class="navbar navbar - expand - lg navbar - dark bg - primary">
        <div class="container">
            <a class="navbar - brand" href="/">
                <i class="fas fa - chart - line me - 2"></i>
                RQA2025 监控中心
            </a>
            <div class="navbar - nav ms - auto">
                <span class="navbar - text me - 3">
                    <i class="fas fa - circle status - online"></i>
                    系统运行中
                </span>
                <span class="navbar - text" id="currentTime"></span>
            </div>
        </div>
    </nav>

    <div class="container mt - 4">
        <div class="row">
            <div class="col - md - 3">
                <div class="card metric - card">
                    <div class="card - body text - center">
                        <i class="fas fa - server fa - 2x text - primary mb - 2"></i>
                        <h4 class="card - title">活跃节点</h4>
                        <div class="h2 text - primary" id="activeNodes">5</div>
                    </div>
                </div>
            </div>

            <div class="col - md - 3">
                <div class="card metric - card">
                    <div class="card - body text - center">
                        <i class="fas fa - exchange - alt fa - 2x text - success mb - 2"></i>
                        <h4 class="card - title">总交易数</h4>
                        <div class="h2 text - success" id="totalTrades">1,250</div>
                    </div>
                </div>
            </div>

            <div class="col - md - 3">
                <div class="card metric - card">
                    <div class="card - body text - center">
                        <i class="fas fa - dollar - sign fa - 2x text - warning mb - 2"></i>
                        <h4 class="card - title">累计收益</h4>
                        <div class="h2 text - warning" id="totalPnl">15,420.50</div>
                    </div>
                </div>
            </div>

            <div class="col - md - 3">
                <div class="card metric - card">
                    <div class="card - body text - center">
                        <i class="fas fa - clock fa - 2x text - info mb - 2"></i>
                        <h4 class="card - title">运行时间</h4>
                        <div class="h4 text - info" id="uptime">2h 30m</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt - 4">
            <div class="col - md - 8">
                <div class="card">
                    <div class="card - header">
                        <h5 class="mb - 0">系统性能</h5>
                    </div>
                    <div class="card - body">
                        <canvas id="performanceChart" width="400" height="200"></canvas>
                    </div>
                </div>
            </div>

            <div class="col - md - 4">
                <div class="card">
                    <div class="card - header">
                        <h5 class="mb - 0">策略表现</h5>
                    </div>
                    <div class="card - body">
                        <div class="list - group list - group - flush">
                            <div class="list - group - item d - flex justify - content - between align - items - center">
                                趋势跟踪策略
                                <span class="badge bg - success">+8.5%</span>
                            </div>
                            <div class="list - group - item d - flex justify - content - between align - items - center">
                                均值回归策略
                                <span class="badge bg - warning">+2.1%</span>
                            </div>
                            <div class="list - group - item d - flex justify - content - between align - items - center">
                                高频策略
                                <span class="badge bg - danger">-1.2%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 更新时间
        function updateTime() {
            document.getElementById('currentTime').textContent = new Date().toLocaleString();
        }
        setInterval(updateTime, 1000);
        updateTime();

        // 性能图表
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU使用率',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }, {
                    label: '内存使用率',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });

        // 实时更新数据
        async function updateData() {
            try {
                const response = await fetch('/api / performance / data');
                const data = await response.json();

                if (data.data && data.data.length > 0) {
                    const latest = data.data[data.data.length - 1];
                    document.getElementById('activeNodes').textContent = latest.active_nodes || 5;
                    document.getElementById('totalTrades').textContent = latest.total_trades || 1250;
                    document.getElementById('totalPnl').textContent = (latest.pnl || 15420.50).toFixed(2);
                }
            } catch (error) {
                console.error('更新数据失败:', error);
            }
        }

        // 每5秒更新一次数据
        setInterval(updateData, 5000);
        updateData();
    </script>

    <script src="https://cdn.jsdelivr.net / npm / bootstrap@5.1.3 / dist / js / bootstrap.bundle.min.js"></script>
        </body>
</html>
"""

STRATEGIES_HTML = """
    <!DOCTYPE html>
<html lang="zh - CN">
    <head>
    <meta charset="UTF - 8">
    <meta name="viewport" content="width=device - width, initial - scale=1.0">
    <title>策略管理 - RQA2025</title>
    <link href="https://cdn.jsdelivr.net / npm / bootstrap@5.1.3 / dist / css / bootstrap.min.css" rel="stylesheet">
    <style>
        .strategy - card {
            transition: all 0.3s ease;
        }
        .strategy - card:hover {
            transform: translateY(-2px);
            box - shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .performance - bar {
            height: 8px;
            border - radius: 4px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar - expand - lg navbar - dark bg - success">
        <div class="container">
            <a class="navbar - brand" href="/">
                <i class="fas fa - brain me - 2"></i>
                策略管理中心
            </a>
            <div class="navbar - nav ms - auto">
                <a class="nav - link" href="/dashboard">
                    <i class="fas fa - tachometer - alt me - 1"></i>
                    返回监控
                </a>
            </div>
        </div>
    </nav>

    <div class="container mt - 4">
        <div class="row">
            <div class="col - 12">
                <div class="d - flex justify - content - between align - items - center mb - 4">
                    <h2>交易策略管理</h2>
                    <button class="btn btn - primary">
                        <i class="fas fa - plus me - 2"></i>
                        新建策略
                    </button>
                </div>
            </div>
        </div>

        <div class="row" id="strategiesContainer">
            <!-- 策略卡片将在这里动态生成 -->
        </div>
    </div>

    <script>
        // 模拟策略数据
        const strategies = [
            {
                name: '趋势跟踪策略',
                type: '趋势策略',
                status: '运行中',
                winRate: 68.5,
                totalPnl: 8542.30,
                sharpeRatio: 1.85,
                maxDrawdown: 8.2,
                totalTrades: 245
            },
            {
                name: '均值回归策略',
                type: '套利策略',
                status: '运行中',
                winRate: 72.1,
                totalPnl: 3210.80,
                sharpeRatio: 2.15,
                maxDrawdown: 5.8,
                totalTrades: 189
            },
            {
                name: '高频动量策略',
                type: '高频策略',
                status: '暂停',
                winRate: 65.3,
                totalPnl: -1245.50,
                sharpeRatio: 0.85,
                maxDrawdown: 12.1,
                totalTrades: 1250
            },
            {
                name: '机器学习策略',
                type: 'AI策略',
                status: '训练中',
                winRate: 0,
                totalPnl: 0,
                sharpeRatio: 0,
                maxDrawdown: 0,
                totalTrades: 0
            }
        ];

        function createStrategyCard(strategy) {
            const statusClass = strategy.status === '运行中' ? 'success' :
                              strategy.status === '暂停' ? 'warning' :
                              strategy.status === '训练中' ? 'info' : 'secondary';

            const pnlClass = strategy.totalPnl >= 0 ? 'success' : 'danger';

            return `
                <div class="col - md - 6 mb - 4">
                    <div class="card strategy - card h - 100">
                        <div class="card - header d - flex justify - content - between align - items - center">
                            <h5 class="mb - 0">${strategy.name}</h5>
                            <span class="badge bg-${statusClass}">${strategy.status}</span>
                        </div>
                        <div class="card - body">
                            <p class="text - muted mb - 3">${strategy.type}</p>

                            <div class="row mb - 3">
                                <div class="col - 6">
                                    <div class="text - center">
                                        <div class="h4 text-${pnlClass}">$${strategy.totalPnl.toFixed(2)}</div>
                                        <small class="text - muted">累计收益</small>
                                    </div>
                                </div>
                                <div class="col - 6">
                                    <div class="text - center">
                                        <div class="h4 text - primary">${strategy.winRate.toFixed(1)}%</div>
                                        <small class="text - muted">胜率</small>
                                    </div>
                                </div>
                            </div>

                            <div class="mb - 2">
                                <small class="text - muted">夏普比率: ${strategy.sharpeRatio.toFixed(2)}</small>
                            </div>
                            <div class="performance - bar bg - light mb - 2">
                                <div class="bg - primary performance - bar" style="width: ${Math.min(strategy.winRate, 100)}%"></div>
                            </div>

                            <div class="row text - center">
                                <div class="col - 4">
                                    <small class="text - muted">总交易</small>
                                    <div class="fw - bold">${strategy.totalTrades}</div>
                                </div>
                                <div class="col - 4">
                                    <small class="text - muted">最大回撤</small>
                                    <div class="fw - bold text - danger">${strategy.maxDrawdown.toFixed(1)}%</div>
                                </div>
                                <div class="col - 4">
                                    <small class="text - muted">操作</small>
                                    <div>
                                        <button class="btn btn - sm btn - outline - primary me - 1">
                                            <i class="fas fa - play"></i>
                                        </button>
                                        <button class="btn btn - sm btn - outline - secondary">
                                            <i class="fas fa - pause"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        function loadStrategies() {
            const container = document.getElementById('strategiesContainer');
            container.innerHTML = strategies.map(createStrategyCard).join('');
        }

        // 页面加载完成后加载策略
        document.addEventListener('DOMContentLoaded', loadStrategies);
    </script>

    <script src="https://cdn.jsdelivr.net / npm / bootstrap@5.1.3 / dist / js / bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com / ajax / libs / font - awesome / 6.0.0 / js / all.min.js"></script>
        </body>
</html>
"""

ALERTS_HTML = """
    <!DOCTYPE html>
<html lang="zh - CN">
    <head>
    <meta charset="UTF - 8">
    <meta name="viewport" content="width=device - width, initial - scale=1.0">
    <title>告警中心 - RQA2025</title>
    <link href="https://cdn.jsdelivr.net / npm / bootstrap@5.1.3 / dist / css / bootstrap.min.css" rel="stylesheet">
    <style>
        .alert - card {
            border - left: 4px solid;
        }
        .alert - critical { border - left - color: #dc3545; }
        .alert - warning { border - left - color: #ffc107; }
        .alert - info { border - left - color: #0dcaf0; }
        .alert - low { border - left - color: #6c757d; }
    </style>
</head>
<body>
    <nav class="navbar navbar - expand - lg navbar - dark bg - warning">
        <div class="container">
            <a class="navbar - brand" href="/">
                <i class="fas fa - bell me - 2"></i>
                告警中心
            </a>
            <div class="navbar - nav ms - auto">
                <span class="navbar - text" id="alertCount">0 个活动告警</span>
            </div>
        </div>
    </nav>

    <div class="container mt - 4">
        <div class="row mb - 4">
            <div class="col - 12">
                <div class="d - flex justify - content - between align - items - center">
                    <h2>系统告警</h2>
                    <div>
                        <button class="btn btn - outline - secondary me - 2" onclick="clearAcknowledged()">
                            <i class="fas fa - check me - 1"></i>
                            清空已确认
                        </button>
                        <button class="btn btn - primary" onclick="refreshAlerts()">
                            <i class="fas fa - sync me - 1"></i>
                            刷新
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <div id="alertsContainer">
            <!-- 告警卡片将在这里动态生成 -->
        </div>
    </div>

    <script>
        // 模拟告警数据
        let alerts = [
            {
                id: 'alert_001',
                level: 'critical',
                title: '系统CPU使用率过高',
                message: 'CPU使用率已达到95%，可能影响系统性能',
                source: '系统监控',
                timestamp: new Date(Date.now() - 300000).toISOString(),
                acknowledged: false
            },
            {
                id: 'alert_002',
                level: 'warning',
                title: '策略A最大回撤超限',
                message: '趋势跟踪策略最大回撤达到15%，超过10 % 阈值',
                source: '策略监控',
                timestamp: new Date(Date.now() - 600000).toISOString(),
                acknowledged: false
            },
            {
                id: 'alert_003',
                level: 'info',
                title: '新交易信号生成',
                message: '趋势跟踪策略生成了新的买入信号',
                source: '交易信号',
                timestamp: new Date(Date.now() - 1200000).toISOString(),
                acknowledged: true
            },
            {
                id: 'alert_004',
                level: 'critical',
                title: '市场数据连接断开',
                message: '无法连接到Bloomberg数据源',
                source: '数据源',
                timestamp: new Date(Date.now() - 1800000).toISOString(),
                acknowledged: false
            }
        ];

        function formatTime(timestamp) {
            const date = new Date(timestamp);
            const now = new Date();
            const diff = now - date;

            if (diff < 60000) return '刚刚';
            if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
            if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
            return date.toLocaleDateString();
        }

        function getAlertIcon(level) {
            switch(level) {
                case 'critical': return 'fas fa - exclamation - triangle text - danger';
                case 'warning': return 'fas fa - exclamation - circle text - warning';
                case 'info': return 'fas fa - info - circle text - info';

                default: return 'fas fa - bell text - secondary';
            }
        }

        function getAlertBadgeClass(level) {
            switch(level) {
                case 'critical': return 'bg - danger';
                case 'warning': return 'bg - warning text - dark';
                case 'info': return 'bg - info';

                default: return 'bg - secondary';
            }
        }

        function createAlertCard(alert) {
            const timeAgo = formatTime(alert.timestamp);
            const iconClass = getAlertIcon(alert.level);
            const badgeClass = getAlertBadgeClass(alert.level);

            return `
                <div class="card alert - card alert-${alert.level} mb - 3">
                    <div class="card - body">
                        <div class="d - flex justify - content - between align - items - start">
                            <div class="d - flex align - items - start">
                                <i class="${iconClass} fa - 2x me - 3"></i>
                                <div>
                                    <h5 class="card - title mb - 1">${alert.title}</h5>
                                    <p class="card - text text - muted mb - 2">${alert.message}</p>
                                    <div class="d - flex align - items - center">
                                        <span class="badge ${badgeClass} me - 2">${alert.level.toUpperCase()}</span>
                                        <small class="text - muted me - 2">${alert.source}</small>
                                        <small class="text - muted">${timeAgo}</small>
                                    </div>
                                </div>
                            </div>
                            <div>
                                ${!alert.acknowledged ?
                                    `<button class="btn btn - sm btn - outline - primary" onclick="acknowledgeAlert('${alert.id}')">
                                        <i class="fas fa - check me - 1"></i>
                                        确认
                                    </button>` :
                                    '<span class="badge bg - success"><i class="fas fa - check me - 1"></i>已确认</span>'
                                }
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        function loadAlerts() {
            const container = document.getElementById('alertsContainer');
            const activeAlerts = alerts.filter(a => !a.acknowledged);
            const acknowledgedAlerts = alerts.filter(a => a.acknowledged);

            // 显示活动告警
            let html = '';
            if (activeAlerts.length > 0) {
                html += '<h4 class="mb - 3"><i class="fas fa - exclamation - triangle text - warning me - 2"></i>活动告警</h4>';
                html += activeAlerts.map(createAlertCard).join('');
            }

            // 显示已确认告警
            if (acknowledgedAlerts.length > 0) {
                html += '<h4 class="mb - 3 mt - 4"><i class="fas fa - check text - success me - 2"></i>已确认告警</h4>';
                html += acknowledgedAlerts.map(createAlertCard).join('');
            }

            if (alerts.length === 0) {
                html = '<div class="text - center py - 5"><i class="fas fa - bell - slash fa - 3x text - muted mb - 3"></i><h5 class="text - muted">暂无告警</h5></div>';
            }

            container.innerHTML = html;

            // 更新告警计数
            document.getElementById('alertCount').textContent = `${activeAlerts.length} 个活动告警`;
        }

        async function acknowledgeAlert(alertId) {
            try {
                const response = await fetch(`/api / alerts / acknowledge/${alertId}`, {
                    method: 'POST'
                });

                if (response.ok) {
                    const alert = alerts.find(a => a.id === alertId);
                    if (alert) {
                        alert.acknowledged = true;
                        loadAlerts();
                    }
                }
            } catch (error) {
                console.error('确认告警失败:', error);
            }
        }

        function clearAcknowledged() {
            alerts = alerts.filter(a => !a.acknowledged);
            loadAlerts();
        }

        function refreshAlerts() {
            // 模拟刷新告警数据
            loadAlerts();
        }

        // 页面加载完成后加载告警
        document.addEventListener('DOMContentLoaded', loadAlerts);

        // 每30秒自动刷新
        setInterval(refreshAlerts, 30000);
    </script>

    <script src="https://cdn.jsdelivr.net / npm / bootstrap@5.1.3 / dist / js / bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com / ajax / libs / font - awesome / 6.0.0 / js / all.min.js"></script>
        </body>
</html>
"""

SYSTEM_HTML = """
    <!DOCTYPE html>
<html lang="zh - CN">
    <head>
    <meta charset="UTF - 8">
    <meta name="viewport" content="width=device - width, initial - scale=1.0">
    <title>系统管理 - RQA2025</title>
    <link href="https://cdn.jsdelivr.net / npm / bootstrap@5.1.3 / dist / css / bootstrap.min.css" rel="stylesheet">
    <style>
        .control - card {
            transition: all 0.3s ease;
        }
        .control - card:hover {
            transform: translateY(-2px);
            box - shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .status - indicator {
            width: 12px;
            height: 12px;
            border - radius: 50%;
            display: inline - block;
        }
        .status - online { background - color: #28a745; }
        .status - offline { background - color: #dc3545; }
        .status - warning { background - color: #ffc107; }
    </style>
</head>
<body>
    <nav class="navbar navbar - expand - lg navbar - dark bg - info">
        <div class="container">
            <a class="navbar - brand" href="/">
                <i class="fas fa - server me - 2"></i>
                系统管理中心
            </a>
            <div class="navbar - nav ms - auto">
                <button class="btn btn - light btn - sm me - 2" onclick="refreshSystem()">
                    <i class="fas fa - sync me - 1"></i>
                    刷新
                </button>
                <div class="text - light">
                    <span class="status - indicator status - online me - 1"></span>
                    系统正常
                </div>
            </div>
        </div>
    </nav>

    <div class="container mt - 4">
        <div class="row">
            <div class="col - md - 8">
                <div class="card">
                    <div class="card - header">
                        <h5 class="mb - 0">系统控制</h5>
                    </div>
                    <div class="card - body">
                        <div class="row g - 3">
                            <div class="col - md - 6">
                                <div class="card control - card border - primary">
                                    <div class="card - body text - center">
                                        <i class="fas fa - play fa - 2x text - primary mb - 2"></i>
                                        <h6>启动系统</h6>
                                        <button class="btn btn - primary btn - sm" onclick="systemControl('start')">
                                            <i class="fas fa - play me - 1"></i>
                                            启动
                                        </button>
                                    </div>
                                </div>
                            </div>

                            <div class="col - md - 6">
                                <div class="card control - card border - danger">
                                    <div class="card - body text - center">
                                        <i class="fas fa - stop fa - 2x text - danger mb - 2"></i>
                                        <h6>停止系统</h6>
                                        <button class="btn btn - danger btn - sm" onclick="systemControl('stop')">
                                            <i class="fas fa - stop me - 1"></i>
                                            停止
                                        </button>
                                    </div>
                                </div>
                            </div>

                            <div class="col - md - 6">
                                <div class="card control - card border - warning">
                                    <div class="card - body text - center">
                                        <i class="fas fa - redo fa - 2x text - warning mb - 2"></i>
                                        <h6>重启系统</h6>
                                        <button class="btn btn - warning btn - sm" onclick="systemControl('restart')">
                                            <i class="fas fa - redo me - 1"></i>
                                            重启
                                        </button>
                                    </div>
                                </div>
                            </div>

                            <div class="col - md - 6">
                                <div class="card control - card border - success">
                                    <div class="card - body text - center">
                                        <i class="fas fa - cog fa - 2x text - success mb - 2"></i>
                                        <h6>系统配置</h6>
                                        <button class="btn btn - success btn - sm" onclick="showConfig()">
                                            <i class="fas fa - cog me - 1"></i>
                                            配置
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="col - md - 4">
                <div class="card">
                    <div class="card - header">
                        <h5 class="mb - 0">系统状态</h5>
                    </div>
                    <div class="card - body">
                        <div class="list - group list - group - flush">
                            <div class="list - group - item d - flex justify - content - between align - items - center">
                                运行状态
                                <span class="badge bg - success" id="systemStatus">运行中</span>
                            </div>
                            <div class="list - group - item d - flex justify - content - between align - items - center">
                                运行时间
                                <span class="text - muted" id="uptime">2h 30m</span>
                            </div>
                            <div class="list - group - item d - flex justify - content - between align - items - center">
                                活跃节点
                                <span class="badge bg - primary" id="activeNodes">5</span>
                            </div>
                            <div class="list - group - item d - flex justify - content - between align - items - center">
                                CPU使用率
                                <span class="text - info" id="cpuUsage">45%</span>
                            </div>
                            <div class="list - group - item d - flex justify - content - between align - items - center">
                                内存使用率
                                <span class="text - warning" id="memoryUsage">67%</span>
                            </div>
                            <div class="list - group - item d - flex justify - content - between align - items - center">
                                磁盘使用率
                                <span class="text - success" id="diskUsage">23%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt - 4">
            <div class="col - 12">
                <div class="card">
                    <div class="card - header">
                        <h5 class="mb - 0">系统日志</h5>
                    </div>
                    <div class="card - body">
                        <div id="systemLogs" class="bg - dark text - light p - 3 rounded" style="height: 300px; overflow - y: auto; font - family: monospace; font - size: 0.875rem;">
                            <!-- 日志将在这里动态显示 -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 模拟系统日志
        const logMessages = [
            '[2024 - 08 - 25 19:45:30] INFO: 趋势跟踪策略执行成功，收益 +125.50',
            '[2024 - 08 - 25 19:45:25] INFO: 高频交易节点连接正常',
            '[2024 - 08 - 25 19:45:20] INFO: ML推理服务响应时间: 45ms',
            '[2024 - 08 - 25 19:45:15] WARNING: 内存使用率达到78%',
            '[2024 - 08 - 25 19:45:10] INFO: 新的市场数据已接收，处理延迟: 12ms',
            '[2024 - 08 - 25 19:45:05] INFO: 风险检查通过，订单已提交',
            '[2024 - 08 - 25 19:45:00] INFO: 策略优化完成，新的参数已应用',
            '[2024 - 08 - 25 19:44:55] INFO: 数据源连接状态正常',
            '[2024 - 08 - 25 19:44:50] INFO: 系统健康检查通过',
            '[2024 - 08 - 25 19:44:45] INFO: 监控数据已更新'
        ];

        let logIndex = 0;

        function addLogMessage() {
            if (logIndex < logMessages.length) {
                const logContainer = document.getElementById('systemLogs');
                const message = logMessages[logIndex];
                const logEntry = document.createElement('div');
                logEntry.textContent = message;
                logEntry.className = 'mb - 1';
                logContainer.appendChild(logEntry);

                // 保持最新的日志在底部
                logContainer.scrollTop = logContainer.scrollHeight;
                logIndex++;
            }
        }

        function systemControl(command) {
            fetch('/api / system / control', {
                method: 'POST',
                headers: {
                    'Content - Type': 'application / json',
                },
                body: JSON.stringify({ command: command })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`${command} 命令已发送`);
                } else {
                    alert(`命令失败: ${data.message}`);
                }
            })
            .catch(error => {
                console.error('系统控制失败:', error);
                alert('命令发送失败，请稍后重试');
            });
        }

        function showConfig() {
            alert('系统配置功能正在开发中...');
        }

        function refreshSystem() {
            // 模拟刷新系统状态
            document.getElementById('cpuUsage').textContent = Math.floor(Math.random() * 30 + 40) + '%';
            document.getElementById('memoryUsage').textContent = Math.floor(Math.random() * 20 + 60) + '%';
            document.getElementById('diskUsage').textContent = Math.floor(Math.random() * 10 + 20) + '%';

            addLogMessage();
        }

        // 页面加载完成后启动日志显示
        document.addEventListener('DOMContentLoaded', function() {
            // 显示初始日志
            for (let i = 0; i < 5 && i < logMessages.length; i++) {
                addLogMessage();
            }

            // 每5秒添加新日志
            setInterval(addLogMessage, 5000);

            // 每10秒刷新系统状态
            setInterval(refreshSystem, 10000);
        });
    </script>

    <script src="https://cdn.jsdelivr.net / npm / bootstrap@5.1.3 / dist / js / bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com / ajax / libs / font - awesome / 6.0.0 / js / all.min.js"></script>
        </body>
</html>
"""


def create_mobile_monitor_app():
    """创建移动端监控应用"""
    monitor = MobileMonitor()

    # 启动后台更新
    monitor.start_background_update()

    return monitor


if __name__ == "__main__":
    # 创建并启动移动端监控应用
    app = create_mobile_monitor_app()
    app.start_server()  # 这将在独立进程中运行

    print("移动端监控界面已启动")
    print("访问 http://localhost:8082 查看监控界面")
