#!/usr/bin/env python3
from src.infrastructure.web.dashboard import DashboardRequestHandler


from src.infrastructure.web.dashboard import DashboardRequestHandler


"""
风险监控仪表误

构建实时风险监控的可视化仪表板，支持多维度风险展示和交互操作
创建时间: 2025-08-24 10:13:48
"""

import sys
import os
import json
import time
import logging
import threading
from typing import Dict, List, Any
from datetime import datetime
import webbrowser
import http.server
import socketserver
from urllib.parse import urlparse

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from risk.realtime_risk_monitor import (
        RealtimeRiskMonitor, create_default_risk_monitor
    )
    from risk.alert_rule_engine import AlertRuleEngine, create_default_alert_rules
    print("误风险监控模块导入成功")
except ImportError as e:
    print(f"误风险监控模块导入失败: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DashboardDataProvider:

    """仪表板数据提供者"""

    def __init__(self, risk_monitor: RealtimeRiskMonitor, alert_engine: AlertRuleEngine):
        self.risk_monitor = risk_monitor
        self.alert_engine = alert_engine
        self.data_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 30  # 缓存30秒

    def get_dashboard_data(self) -> dict:
        """获取仪表板数据"""
        current_time = datetime.now()

        # 检查缓存是否有效
        if (self.cache_timestamp and
                (current_time - self.cache_timestamp).seconds < self.cache_duration):
            return self.data_cache

        # 获取实时数据
        monitor_status = self.risk_monitor.get_monitoring_status()
        rule_stats = self.alert_engine.get_rule_stats()
        rules_summary = self.alert_engine.get_rules_summary()
        risk_summary = monitor_status.get('risk_summary', {})

        # 构建仪表板数据
        dashboard_data = {
            'timestamp': current_time.isoformat(),
            'summary': {
                'total_indicators': risk_summary.get('total_indicators', 0),
                'active_alerts': risk_summary.get('active_alerts', 0),
                'total_rules': rules_summary.get('total_rules', 0),
                'enabled_rules': rules_summary.get('enabled_rules', 0),
                'monitoring_status': 'active' if monitor_status.get('is_monitoring') else 'inactive'
            },
            'risk_indicators': self._format_indicators(monitor_status.get('recent_indicators', [])),
            'active_alerts': monitor_status.get('active_alerts', []),
            'alert_trends': self._calculate_alert_trends(),
            'risk_distribution': risk_summary.get('risk_distribution', {}),
            'severity_distribution': rules_summary.get('severity_distribution', {}),
            'rule_performance': self._format_rule_performance(rule_stats),
            'system_health': self._calculate_system_health(monitor_status)
        }

        # 更新缓存
        self.data_cache = dashboard_data
        self.cache_timestamp = current_time

        return dashboard_data

    def _format_indicators(self, indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化指标数据"""
        formatted = []

        for indicator in indicators:
            formatted.append({
                'name': indicator.get('name', ''),
                'value': indicator.get('value', 0),
                'risk_level': indicator.get('risk_level', 'low'),
                'risk_type': indicator.get('risk_type', ''),
                'description': indicator.get('description', ''),
                'unit': indicator.get('unit', ''),
                'threshold_low': indicator.get('threshold_low', 0),
                'threshold_medium': indicator.get('threshold_medium', 0),
                'threshold_high': indicator.get('threshold_high', 0),
                'timestamp': indicator.get('timestamp', '')
            })

        return formatted

    def _calculate_alert_trends(self) -> Dict[str, Any]:
        """计算告警趋势"""
        # 这里应该是从历史数据中计算趋势
        # 简化实现，返回模拟数据
        return {
            'today_alerts': 12,
            'yesterday_alerts': 8,
            'week_trend': [5, 8, 12, 15, 10, 8, 12],
            'hourly_trend': [2, 1, 0, 1, 3, 2, 1, 2, 4, 3, 2, 1]
        }

    def _format_rule_performance(self, rule_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """格式化规则性能数据"""
        formatted = []

        for rule_id, stats in rule_stats.items():
            if rule_id in self.alert_engine.rules:
                rule = self.alert_engine.rules[rule_id]
            formatted.append({
                'rule_id': rule_id,
                'rule_name': rule.name,
                'trigger_count': stats.get('trigger_count', 0),
                'success_rate': (stats.get('success_count', 0)
                                 / max(stats.get('trigger_count', 1), 1)),
                'last_triggered': stats.get('last_triggered').isoformat()
                if stats.get('last_triggered') else None,
                'enabled': rule.enabled,
                'severity': rule.severity.value
            })

        return formatted

    def _calculate_system_health(self, monitor_status: Dict[str, Any]) -> Dict[str, Any]:
        """计算系统健康状态"""
        health_score = 100
        issues = []

        # 检查监控状态
        if not monitor_status.get('is_monitoring', False):
            health_score -= 20
        issues.append("风险监控未运行")

        # 检查活跃告警
        active_alerts = monitor_status.get('active_alerts', [])
        if len(active_alerts) > 5:
            health_score -= 15
        issues.append("告警数量过多")

        # 检查规则状态
        rule_summary = self.alert_engine.get_rules_summary()
        disabled_rules = rule_summary.get('disabled_rules', 0)
        if disabled_rules > 0:
            health_score -= 10
        issues.append(f"{disabled_rules}个规则被禁用")

        return {
            'health_score': max(0, health_score),
            'status': 'healthy' if health_score >= 80 else 'warning' if health_score >= 60 else 'critical',
            'issues': issues
        }


class DashboardWebServer:

    """仪表板Web服务器"""

    def __init__(self, data_provider: DashboardDataProvider, port: int = 8080):

        self.data_provider = data_provider
        self.port = port
        self.server = None
        self.is_running = False

        # HTML模板
        self.html_template = self._create_html_template()

    def start(self):
        """启动Web服务器"""
        if self.is_running:
            logger.warning("Web服务器已在运行")
        return

        try:
            # 创建请求处理器
            handler = self._create_request_handler()

            # 启动服务器
            self.server = socketserver.TCPServer(("", self.port), handler)
            self.is_running = True

            logger.info(f"风险监控仪表板已启动: http://localhost:{self.port}")

            # 在新线程中运行服务器
            server_thread = threading.Thread(target=self.server.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            # 自动打开浏览器
            try:
                webbrowser.open(f"http://localhost:{self.port}")
            except Exception as e:
                logger.warning(f"无法自动打开浏览器 {e}")

        except Exception as e:
            logger.error(f"启动Web服务器失败 {e}")
            self.is_running = False

    def stop(self):
        """停止Web服务器"""
        if self.server:
            self.server.shutdown()
        self.server.server_close()
        self.is_running = False
        logger.info("风险监控仪表板已停止")

    def _create_request_handler(self):
        """创建请求处理器"""
        data_provider = self.data_provider
        html_template = self.html_template

        class DashboardRequestHandler(http.server.BaseHTTPRequestHandler):

            def do_GET(self):

                try:
                    parsed_path = urlparse(self.path)

                    if parsed_path.path == '/':
                        # 主页面
                        self._send_html_response(html_template)
                    elif parsed_path.path == '/api/data':
                        # 数据API
                        data = data_provider.get_dashboard_data()
                        self._send_json_response(data)
                    elif parsed_path.path == '/api/health':
                        # 健康检查API
                        health_data = {
                            'status': 'healthy',
                            'timestamp': datetime.now().isoformat(),
                            'uptime': 'running'
                        }
                        self._send_json_response(health_data)
                    else:
                        # 静态文件或404
                        self._send_error_response(404, "Not Found")

                except Exception as e:
                    logger.error(f"请求处理错误: {e}")
                    self._send_error_response(500, "Internal Server Error")

            def do_POST(self):

                try:
                    parsed_path = urlparse(self.path)

                    if parsed_path.path == '/api/rule/enable':
                        # 启用规则
                        content_length = int(self.headers['Content-Length'])
                        post_data = self.rfile.read(content_length)
                        data = json.loads(post_data.decode('utf-8'))

                        rule_id = data.get('rule_id')
                        if rule_id and data_provider.alert_engine.enable_rule(rule_id):
                            self._send_json_response({'status': 'success'})
                        else:
                            self._send_json_response({'status': 'error'}, 400)

                    elif parsed_path.path == '/api/rule/disable':
                        # 禁用规则
                        content_length = int(self.headers['Content-Length'])
                        post_data = self.rfile.read(content_length)
                        data = json.loads(post_data.decode('utf-8'))

                        rule_id = data.get('rule_id')
                        if rule_id and data_provider.alert_engine.disable_rule(rule_id):
                            self._send_json_response({'status': 'success'})
                        else:
                            self._send_json_response({'status': 'error'}, 400)

                    else:
                        self._send_error_response(404, "Not Found")

                except Exception as e:
                    logger.error(f"POST请求处理错误: {e}")
                    self._send_error_response(500, "Internal Server Error")

            def _send_html_response(self, html_content: str):
                """发送HTML响应"""
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(html_content.encode('utf-8'))

            def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
                """发送JSON响应"""
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

            def _send_error_response(self, status_code: int, message: str):
                """发送错误响应"""
                self.send_response(status_code)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(message.encode('utf-8'))

            def log_message(self, format, *args):
                """重写日志方法以使用我们的logger"""
                logger.info(format % args)

    def _create_html_template(self) -> str:
        """创建HTML模板"""
        return """<!DOCTYPE html>
    <html lang="zh - CN">
    <head>
    <meta charset="UTF - 8">
    <meta name="viewport" content="width=device - width, initial - scale=1.0">
    <title>RQA2025 风险监控仪表误/title>
    <style>
    * {
    margin: 0;
    padding: 0;
    box - sizing: border - box;
    }

    body {
    font - family: -apple - system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans - serif;
    background: linear - gradient(135deg, #667eea 0%, #764ba2 100%);
    min - height: 100vh;
    color: #333;
    }

    .dashboard {
    max - width: 1200px;
    margin: 0 auto;
    padding: 20px;
    }

    .header {
    background: rgba(255, 255, 255, 0.95);
    backdrop - filter: blur(10px);
    border - radius: 15px;
    padding: 25px;
    margin - bottom: 25px;
    box - shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .header h1 {
    color: #2c3e50;
    font - size: 2.5em;
    text - align: center;
    margin - bottom: 10px;
    }

    .header .subtitle {
    text - align: center;
    color: #7f8c8d;
    font - size: 1.1em;
    }

    .summary - grid {
    display: grid;
    grid - template - columns: repeat(auto - fit, minmax(250px, 1fr));
    gap: 20px;
    margin - bottom: 25px;
    }

    .summary - card {
    background: rgba(255, 255, 255, 0.95);
    backdrop - filter: blur(10px);
    border - radius: 15px;
    padding: 20px;
    box - shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    text - align: center;
    }

    .summary - card h3 {
    color: #2c3e50;
    margin - bottom: 10px;
    font - size: 1.2em;
    }

    .metric - value {
    font - size: 2em;
    font - weight: bold;
    color: #3498db;
    margin - bottom: 5px;
    }

    .content - grid {
    display: grid;
    grid - template - columns: 2fr 1fr;
    gap: 25px;
    margin - bottom: 25px;
    }

    .main - content {
    background: rgba(255, 255, 255, 0.95);
    backdrop - filter: blur(10px);
    border - radius: 15px;
    padding: 25px;
    box - shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .sidebar {
    display: flex;
    flex - direction: column;
    gap: 20px;
    }

    .sidebar - card {
    background: rgba(255, 255, 255, 0.95);
    backdrop - filter: blur(10px);
    border - radius: 15px;
    padding: 20px;
    box - shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .section - title {
    color: #2c3e50;
    font - size: 1.5em;
    margin - bottom: 20px;
    border - bottom: 2px solid #3498db;
    padding - bottom: 10px;
    }

    .indicators - list {
    display: grid;
    gap: 15px;
    }

    .indicator - item {
    display: flex;
    justify - content: space - between;
    align - items: center;
    padding: 15px;
    background: #f8f9fa;
    border - radius: 10px;
    border - left: 4px solid #3498db;
    }

    .indicator - name {
    font - weight: bold;
    color: #2c3e50;
    }

    .indicator - value {
    font - size: 1.2em;
    font - weight: bold;
    }

    .risk - low { color: #27ae60; border - left - color: #27ae60; }
    .risk - medium { color: #f39c12; border - left - color: #f39c12; }
    .risk - high { color: #e74c3c; border - left - color: #e74c3c; }
    .risk - critical { color: #c0392b; border - left - color: #c0392b; }

    .alerts - list {
    max - height: 400px;
    overflow - y: auto;
    }

    .alert - item {
    padding: 15px;
    margin - bottom: 10px;
    border - radius: 10px;
    border - left: 4px solid #e74c3c;
    background: #fdf2f2;
    }

    .alert - message {
    font - weight: bold;
    color: #c0392b;
    margin - bottom: 5px;
    }

    .alert - details {
    font - size: 0.9em;
    color: #7f8c8d;
    }

    .rules - list {
    display: grid;
    gap: 10px;
    }

    .rule - item {
    display: flex;
    justify - content: space - between;
    align - items: center;
    padding: 10px;
    background: #f8f9fa;
    border - radius: 8px;
    }

    .rule - toggle {
    background: #3498db;
    color: white;
    border: none;
    padding: 5px 10px;
    border - radius: 5px;
    cursor: pointer;
    font - size: 0.8em;
    }

    .rule - toggle.disabled {
    background: #95a5a6;
    }

    .health - indicator {
    text - align: center;
    padding: 20px;
    }

    .health - score {
    font - size: 3em;
    font - weight: bold;
    margin - bottom: 10px;
    }

    .health - status {
    font - size: 1.2em;
    padding: 5px 15px;
    border - radius: 20px;
    display: inline - block;
    }

    .health - healthy { background: #d5f4e6; color: #27ae60; }
    .health - warning { background: #fdf2f2; color: #e74c3c; }
    .health - critical { background: #f8d7da; color: #c0392b; }

    .refresh - btn {
    background: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    border - radius: 8px;
    cursor: pointer;
    font - size: 1em;
    margin: 10px 0;
    }

    .refresh - btn:hover {
    background: #2980b9;
    }

    .footer {
    text - align: center;
    color: rgba(255, 255, 255, 0.8);
    margin - top: 40px;
    padding: 20px;
    }
    </style>
    </head>
    <body>
    <div class="dashboard">
    <div class="header">
    <h1>🛡误风险监控仪表误/h1>
    <div class="subtitle">RQA2025 实时风险监控与告警系误/div>
    </div>

    <div class="summary - grid" id="summaryGrid">
    <!-- 摘要卡片将通过JavaScript动态生误-->
    </div>

    <div class="content - grid">
    <div class="main - content">
    <h2 class="section - title">📊 风险指标</h2>
    <div class="indicators - list" id="indicatorsList">
    <!-- 指标列表将通过JavaScript动态生误-->
    </div>
    </div>

    <div class="sidebar">
    <div class="sidebar - card">
    <h3 class="section - title">🚨 活跃告警</h3>
    <div class="alerts - list" id="alertsList">
    <!-- 告警列表将通过JavaScript动态生误-->
    </div>
    </div>

    <div class="sidebar - card">
    <h3 class="section - title">📋 系统健康</h3>
    <div class="health - indicator" id="healthIndicator">
    <!-- 健康指标将通过JavaScript动态生误-->
    </div>
    </div>
    </div>
    </div>

    <div class="main - content">
    <h2 class="section - title">⚙️ 告警规则</h2>
    <div class="rules - list" id="rulesList">
    <!-- 规则列表将通过JavaScript动态生误-->
    </div>
    </div>

    <button class="refresh - btn" onclick="refreshData()">🔄 刷新数据</button>

    <div class="footer">
    <p>© 2025 RQA2025 量化交易人工智能系统 | 实时风险监控仪表误/p>
    </div>
    </div>

    <script>
    // 全局变量
    let dashboardData = {};

    // 页面加载时获取数误
    document.addEventListener('DOMContentLoaded', function() {
    refreshData();
    // 误0秒自动刷误
    setInterval(refreshData, 30000);
    });

    // 刷新数据
    async function refreshData() {
    try {
    const response = await fetch('/api / data');
    const data = await response.json();
    dashboardData = data;
    updateDashboard();
    } catch (error) {
    console.error('获取数据失败:', error);
    }
    }

    // 更新仪表误
    function updateDashboard() {
    updateSummary();
    updateIndicators();
    updateAlerts();
    updateHealth();
    updateRules();
    }

    // 更新摘要
    function updateSummary() {
    const summary = dashboardData.summary;
    const summaryGrid = document.getElementById('summaryGrid');

    summaryGrid.innerHTML = `
    <div class="summary - card">
    <h3>📈 风险指标</h3>
    <div class="metric - value">${summary.total_indicators}</div>
    <div>个活跃指误/div>
    </div>
    <div class="summary - card">
    <h3>🚨 活跃告警</h3>
    <div class="metric - value">${summary.active_alerts}</div>
    <div>个未处理告警</div>
    </div>
    <div class="summary - card">
    <h3>⚙️ 告警规则</h3>
    <div class="metric - value">${summary.total_rules}</div>
    <div>${summary.enabled_rules}个已启用</div>
    </div>
    <div class="summary - card">
    <h3>🔄 监控状误/h3>
    <div class="metric - value">
    <span style="color: ${summary.monitoring_status === 'active' ? '#27ae60' : '#e74c3c'}">
    ${summary.monitoring_status === 'active' ? '误 : '误}
    </span>
    </div>
    <div>${summary.monitoring_status === 'active' ? '运行误 : '已停误}</div>
    </div>
    `;
    }

    // 更新指标
    function updateIndicators() {
    const indicators = dashboardData.risk_indicators;
    const indicatorsList = document.getElementById('indicatorsList');

    indicatorsList.innerHTML = '';

    indicators.forEach(indicator => {
    const item = document.createElement('div');
    item.className = `indicator - item risk-${indicator.risk_level}`;

    item.innerHTML = `
    <div>
    <div class="indicator - name">${indicator.name}</div>
    <div style="font - size: 0.9em; color: #7f8c8d;">${indicator.description}</div>
    </div>
    <div class="indicator - value">
    ${typeof indicator.value === 'number' ? indicator.value.toFixed(4) : indicator.value}
    ${indicator.unit}
    </div>
    `;

    indicatorsList.appendChild(item);
    });
    }

    // 更新告警
    function updateAlerts() {
    const alerts = dashboardData.active_alerts;
    const alertsList = document.getElementById('alertsList');

    if (alerts.length === 0) {
    alertsList.innerHTML = '<div style="text - align: center; color: #7f8c8d; padding: 20px;">暂无活跃告警</div>';
    return;
    }

    alertsList.innerHTML = '';

    alerts.forEach(alert => {
    const item = document.createElement('div');
    item.className = 'alert - item';

    item.innerHTML = `
    <div class="alert - message">${alert.message}</div>
    <div class="alert - details">
    风险类型: ${alert.risk_type}<br>
    时间: ${new Date(alert.timestamp).toLocaleString()}
    </div>
    `;

    alertsList.appendChild(item);
    });
    }

    // 更新健康状误
    function updateHealth() {
    const health = dashboardData.system_health;
    const healthIndicator = document.getElementById('healthIndicator');

    healthIndicator.innerHTML = `
    <div class="health - score">${health.health_score}</div>
    <div class="health - status health-${health.status}">
    ${health.status === 'healthy' ? '健康' : health.status === 'warning' ? '警告' : '严重'}
    </div>
    ${health.issues.length > 0 ?
    '<div style="margin - top: 10px; font - size: 0.9em; color: #e74c3c;">' +
    health.issues.join('<br>') +
    '</div>' : ''}
    `;
    }

    // 更新规则
    function updateRules() {
    const rules = dashboardData.rule_performance;
    const rulesList = document.getElementById('rulesList');

    rulesList.innerHTML = '';

    rules.forEach(rule => {
    const item = document.createElement('div');
    item.className = 'rule - item';

    item.innerHTML = `
    <div>
    <div style="font - weight: bold;">${rule.rule_name}</div>
    <div style="font - size: 0.9em; color: #7f8c8d;">
    触发: ${rule.trigger_count}误| 成功误 ${(rule.success_rate * 100).toFixed(1)}%
    </div>
    </div>
    <button class="rule - toggle ${rule.enabled ? '' : 'disabled'}"
    onclick="toggleRule('${rule.rule_id}', ${!rule.enabled})">
    ${rule.enabled ? '禁用' : '启用'}
    </button>
    `;

    rulesList.appendChild(item);
    });
    }

    // 切换规则状误
    async function toggleRule(ruleId, enable) {
    try {
    const response = await fetch(enable ? '/api / rule / enable' : '/api / rule / disable', {
    method: 'POST',
    headers: {
    'Content - Type': 'application / json'
    },
    body: JSON.stringify({ rule_id: ruleId })
    });

    if (response.ok) {
    refreshData(); // 刷新数据
    } else {
    alert('操作失败');
    }
    } catch (error) {
    console.error('切换规则状态失误', error);
    alert('操作失败');
    }
    }
    </script>
    </body>
    </html>"""
        return DashboardRequestHandler


class RiskMonitoringDashboard:

    """风险监控仪表板"""

    def __init__(self):

        self.risk_monitor = create_default_risk_monitor()
        self.alert_engine = AlertRuleEngine()

        # 添加默认告警规则
        default_rules = create_default_alert_rules()
        for rule in default_rules:
            self.alert_engine.add_rule(rule)

        self.data_provider = DashboardDataProvider(self.risk_monitor, self.alert_engine)
        self.web_server = DashboardWebServer(self.data_provider)

        logger.info("风险监控仪表板初始化完成")

    def start(self, port: int = 8080):
        """启动仪表板"""
        logger.info("启动风险监控仪表板..")

        # 启动风险监控
        self.risk_monitor.start_monitoring(self.risk_monitor.simulate_market_data)

        # 启动Web服务器
        self.web_server.start()

        logger.info(f"仪表板已启动，请访问: http://localhost:{port}")

    def stop(self):
        """停止仪表板"""
        logger.info("停止风险监控仪表板..")

        # 停止风险监控
        self.risk_monitor.stop_monitoring()

        # 停止Web服务器
        self.web_server.stop()

        logger.info("仪表板已停止")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        return self.data_provider.get_dashboard_data()


def main():
    """主函数 - 风险监控仪表板演示"""
    print("🖥️RQA2025风险监控仪表板")
    print("="*50)

    # 创建仪表板
    dashboard = RiskMonitoringDashboard()

    print("风险监控仪表板创建完成")
    print("   包含以下组件:")
    print("   - 实时风险监控器")
    print("   - 告警规则引擎")
    print("   - Web仪表板服务器")
    print("   - 数据提供器")

    # 显示配置信息
    monitor_status = dashboard.risk_monitor.get_monitoring_status()
    risk_summary = monitor_status.get('risk_summary', {})

    print("\n📊 初始状态")
    print(f"   风险指标数量: {risk_summary.get('total_indicators', 0)}")
    print(f"   告警规则数量: {len(dashboard.alert_engine.rules)}")
    print(f"   监控状态: {'运行中' if monitor_status.get('is_monitoring') else '未运行'}")

    try:
        # 启动仪表板
        dashboard.start(port=8080)

        print("\n🚀 仪表板已启动")
        print("   访问地址: http://localhost:8080")
        print("   功能特性:")
        print("   - 实时风险指标监控")
        print("   - 告警规则管理")
        print("   - 系统健康状态")
        print("   - 自动数据刷新")
        print("   - 规则启用 / 禁用")

        print("\n按Ctrl + 停止服务...")

        # 保持运行
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号，正在停止仪表板...")
    except Exception as e:
        print(f"\n仪表板运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止仪表板
        dashboard.stop()
        print("风险监控仪表板已停止")

        # 显示最终统计
        final_data = dashboard.get_dashboard_data()
        summary = final_data.get('summary', {})

        print("\n📋 运行统计:")
        print(f"   处理指标数: {summary.get('total_indicators', 0)}")
        print(f"   活跃告警数: {summary.get('active_alerts', 0)}")
        print(f"   系统健康分: {final_data.get('system_health', {}).get('health_score', 0)}")

    return dashboard


if __name__ == "__main__":
    dashboard = main()
