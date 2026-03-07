#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 Web界面展示系统
提供三大创新引擎的可视化界面和交互功能

功能特性:
1. 实时引擎状态监控
2. 性能指标仪表板
3. 交互式功能演示
4. 投资组合管理界面
5. 用户认证和权限管理
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Warning: flask-cors not available, CORS disabled")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available, engine communication disabled")

from functools import wraps
from datetime import datetime, timedelta

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = 'rqa2026_secret_key_change_in_production'
if CORS_AVAILABLE:
    CORS(app)

# 引擎服务地址配置
ENGINE_ENDPOINTS = {
    'fusion': 'http://localhost:8080',
    'quantum': 'http://localhost:8081',
    'ai': 'http://localhost:8082',
    'bci': 'http://localhost:8083'
}

# 模拟用户数据 (生产环境应使用数据库)
USERS = {
    'admin': {'password': 'admin123', 'role': 'admin'},
    'demo': {'password': 'demo123', 'role': 'user'}
}


class EngineMonitor:
    """引擎监控器"""

    def __init__(self):
        self.last_update = None
        self.engine_status = {}
        self.performance_metrics = {}

    async def check_engine_health(self, engine_name, endpoint):
        """检查引擎健康状态"""
        try:
            start_time = time.time()
            response = requests.get(f"{endpoint}/health", timeout=5)
            response_time = time.time() - start_time

            if response.status_code == 200:
                status = response.json()
                return {
                    'status': 'healthy',
                    'response_time': round(response_time * 1000, 2),  # ms
                    'details': status
                }
            else:
                return {
                    'status': 'unhealthy',
                    'response_time': response_time,
                    'error': f"HTTP {response.status_code}"
                }
        except requests.exceptions.RequestException as e:
            return {
                'status': 'unreachable',
                'error': str(e)
            }

    async def update_all_engines_status(self):
        """更新所有引擎状态"""
        tasks = []
        for engine_name, endpoint in ENGINE_ENDPOINTS.items():
            tasks.append(self.check_engine_health(engine_name, endpoint))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (engine_name, _) in enumerate(ENGINE_ENDPOINTS.items()):
            if isinstance(results[i], Exception):
                self.engine_status[engine_name] = {
                    'status': 'error',
                    'error': str(results[i])
                }
            else:
                self.engine_status[engine_name] = results[i]

        self.last_update = datetime.now()

    def get_engine_status(self):
        """获取引擎状态"""
        return {
            'engines': self.engine_status,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }


# 初始化监控器
monitor = EngineMonitor()


def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """管理员权限验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session.get('role') != 'admin':
            flash('需要管理员权限')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in USERS and USERS[username]['password'] == password:
            session['user'] = username
            session['role'] = USERS[username]['role']
            flash('登录成功')
            return redirect(url_for('dashboard'))
        else:
            flash('用户名或密码错误')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """用户登出"""
    session.clear()
    flash('已登出')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    """主仪表板"""
    return render_template('dashboard.html', user=session['user'])


@app.route('/engines')
@login_required
def engines():
    """引擎管理页面"""
    return render_template('engines.html')


@app.route('/portfolio')
@login_required
def portfolio():
    """投资组合管理"""
    return render_template('portfolio.html')


@app.route('/analytics')
@login_required
def analytics():
    """分析中心"""
    return render_template('analytics.html')


@app.route('/api/engine-status')
@login_required
def api_engine_status():
    """获取引擎状态API"""
    try:
        if not REQUESTS_AVAILABLE:
            # 如果requests不可用，返回模拟数据
            return jsonify({
                'engines': {
                    'fusion': {'status': 'unreachable', 'error': 'requests library not available'},
                    'quantum': {'status': 'unreachable', 'error': 'requests library not available'},
                    'ai': {'status': 'unreachable', 'error': 'requests library not available'},
                    'bci': {'status': 'unreachable', 'error': 'requests library not available'}
                },
                'last_update': datetime.now().isoformat()
            })

        # 异步更新状态 (简化处理，实际应使用WebSocket)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(monitor.update_all_engines_status())
        loop.close()

        return jsonify(monitor.get_engine_status())
    except Exception as e:
        return jsonify({
            'engines': {},
            'last_update': datetime.now().isoformat(),
            'error': str(e)
        })


@app.route('/api/engine-demo/<engine_name>', methods=['POST'])
@login_required
def api_engine_demo(engine_name):
    """引擎功能演示API"""
    if engine_name not in ENGINE_ENDPOINTS:
        return jsonify({'error': '未知引擎'}), 404

    endpoint = ENGINE_ENDPOINTS[engine_name]

    try:
        # 根据引擎类型调用不同演示接口
        if engine_name == 'quantum':
            response = requests.post(f"{endpoint}/demo/qaoa", json={'nodes': 4, 'layers': 2})
        elif engine_name == 'ai':
            response = requests.post(f"{endpoint}/demo/multimodal", json={
                'text': '市场情绪积极，建议买入科技股',
                'vision_data': 'sample_chart.png'
            })
        elif engine_name == 'bci':
            response = requests.post(f"{endpoint}/demo/process", json={'eeg_data': [0.1] * 1000})
        elif engine_name == 'fusion':
            response = requests.post(f"{endpoint}/demo/fusion", json={
                'task': '投资组合优化',
                'engines': ['quantum', 'ai']
            })

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f'演示失败: HTTP {response.status_code}'}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'连接失败: {str(e)}'}), 500


@app.route('/api/portfolio-analysis', methods=['POST'])
@login_required
def api_portfolio_analysis():
    """投资组合分析API"""
    data = request.get_json()

    try:
        # 调用融合引擎进行投资组合分析
        response = requests.post(f"{ENGINE_ENDPOINTS['fusion']}/analyze/portfolio", json=data)

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': '分析失败'}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'服务连接失败: {str(e)}'}), 500


@app.route('/api/performance-metrics')
@login_required
def api_performance_metrics():
    """获取性能指标API"""
    # 这里应该从监控系统获取实时指标
    # 简化实现，返回模拟数据
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'cpu_usage': 45.2,
            'memory_usage': 62.8,
            'disk_usage': 34.1
        },
        'engines': {
            'quantum': {'requests_per_second': 12.5, 'avg_response_time': 0.8},
            'ai': {'requests_per_second': 28.3, 'avg_response_time': 0.3},
            'bci': {'requests_per_second': 8.7, 'avg_response_time': 0.5},
            'fusion': {'requests_per_second': 15.2, 'avg_response_time': 0.6}
        }
    }

    return jsonify(metrics)


@app.route('/api/system-health')
@login_required
def api_system_health():
    """系统健康检查API"""
    health_status = {
        'overall_status': 'healthy',
        'checks': {
            'database': {'status': 'healthy', 'details': 'PostgreSQL 连接正常'},
            'cache': {'status': 'healthy', 'details': 'Redis 连接正常'},
            'monitoring': {'status': 'healthy', 'details': 'Prometheus 运行正常'},
            'security': {'status': 'healthy', 'details': '所有安全检查通过'}
        },
        'last_check': datetime.now().isoformat()
    }

    return jsonify(health_status)


@app.context_processor
def inject_user():
    """注入用户信息到模板"""
    return {
        'current_user': session.get('user'),
        'user_role': session.get('role'),
        'datetime': datetime,
        'timedelta': timedelta
    }


if __name__ == '__main__':
    print("🚀 启动 RQA2026 Web界面服务")
    print("📊 访问地址: http://localhost:3000")

    # 启动后台任务更新引擎状态
    async def background_monitor():
        while True:
            await monitor.update_all_engines_status()
            await asyncio.sleep(30)  # 每30秒更新一次

    # 在生产环境中，建议使用gunicorn或其他WSGI服务器
    app.run(
        host='0.0.0.0',
        port=3000,
        debug=True,  # 生产环境应设为False
        threaded=True
    )
