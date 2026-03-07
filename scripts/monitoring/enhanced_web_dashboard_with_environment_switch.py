#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版Web监控仪表板 - 支持环境切换功能
提供多环境监控切换、移动端优化和更多图表类型
"""

import sqlite3
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced_monitoring_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")


class EnvironmentManager:
    """环境管理器"""

    def __init__(self):
        self.environments = {
            'development': {
                'name': '开发环境',
                'status': 'healthy',
                'health_score': 95,
                'response_time': 120,
                'cpu_usage': 45,
                'memory_usage': 60,
                'disk_usage': 55,
                'network_usage': 30
            },
            'staging': {
                'name': '测试环境',
                'status': 'warning',
                'health_score': 78,
                'response_time': 180,
                'cpu_usage': 65,
                'memory_usage': 75,
                'disk_usage': 70,
                'network_usage': 45
            },
            'production': {
                'name': '生产环境',
                'status': 'healthy',
                'health_score': 92,
                'response_time': 150,
                'cpu_usage': 58,
                'memory_usage': 68,
                'disk_usage': 62,
                'network_usage': 38
            }
        }
        self.current_environment = 'production'

    def get_environments(self):
        """获取所有环境信息"""
        return {
            'environments': list(self.environments.keys()),
            'current': self.current_environment,
            'environment_details': self.environments
        }

    def switch_environment(self, env_name):
        """切换环境"""
        if env_name in self.environments:
            self.current_environment = env_name
            return True
        return False

    def get_current_environment_data(self):
        """获取当前环境数据"""
        return self.environments.get(self.current_environment, {})

    def update_environment_metrics(self):
        """更新环境指标"""
        for env_name, env_data in self.environments.items():
            # 模拟指标变化
            env_data['health_score'] = max(
                50, min(100, env_data['health_score'] + random.randint(-5, 5)))
            env_data['response_time'] = max(
                50, min(300, env_data['response_time'] + random.randint(-20, 20)))
            env_data['cpu_usage'] = max(10, min(90, env_data['cpu_usage'] + random.randint(-3, 3)))
            env_data['memory_usage'] = max(
                20, min(85, env_data['memory_usage'] + random.randint(-2, 2)))
            env_data['disk_usage'] = max(
                30, min(80, env_data['disk_usage'] + random.randint(-1, 1)))
            env_data['network_usage'] = max(
                5, min(60, env_data['network_usage'] + random.randint(-2, 2)))

            # 更新状态
            if env_data['health_score'] >= 90:
                env_data['status'] = 'healthy'
            elif env_data['health_score'] >= 70:
                env_data['status'] = 'warning'
            else:
                env_data['status'] = 'critical'


class EnhancedDashboardManager:
    """增强版仪表板管理器"""

    def __init__(self):
        self.env_manager = EnvironmentManager()
        self.db_path = "data/monitoring.db"
        self.init_database()

    def init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 创建环境切换历史表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS environment_switches (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        from_env TEXT,
                        to_env TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT DEFAULT 'system'
                    )
                ''')

                # 创建环境指标历史表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS environment_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        environment TEXT,
                        health_score REAL,
                        response_time REAL,
                        cpu_usage REAL,
                        memory_usage REAL,
                        disk_usage REAL,
                        network_usage REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()
        except Exception as e:
            print(f"数据库初始化错误: {e}")

    def get_dashboard_data(self):
        """获取仪表板数据"""
        current_env = self.env_manager.get_current_environment_data()

        return {
            'current_environment': self.env_manager.current_environment,
            'environment_data': current_env,
            'all_environments': self.env_manager.get_environments(),
            'timestamp': datetime.now().isoformat()
        }

    def get_environment_comparison(self):
        """获取环境对比数据"""
        comparison_data = {
            'labels': [],
            'health_scores': [],
            'response_times': [],
            'cpu_usage': [],
            'memory_usage': []
        }

        for env_name, env_data in self.env_manager.environments.items():
            comparison_data['labels'].append(env_data['name'])
            comparison_data['health_scores'].append(env_data['health_score'])
            comparison_data['response_times'].append(env_data['response_time'])
            comparison_data['cpu_usage'].append(env_data['cpu_usage'])
            comparison_data['memory_usage'].append(env_data['memory_usage'])

        return comparison_data

    def get_performance_distribution(self):
        """获取性能分布数据"""
        current_env = self.env_manager.get_current_environment_data()

        return {
            'labels': ['CPU', '内存', '磁盘', '网络'],
            'data': [
                current_env.get('cpu_usage', 0),
                current_env.get('memory_usage', 0),
                current_env.get('disk_usage', 0),
                current_env.get('network_usage', 0)
            ]
        }

    def switch_environment(self, new_env):
        """切换环境"""
        old_env = self.env_manager.current_environment
        success = self.env_manager.switch_environment(new_env)

        if success:
            # 记录环境切换历史
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO environment_switches (from_env, to_env, user_id)
                        VALUES (?, ?, ?)
                    ''', (old_env, new_env, 'system'))
                    conn.commit()
            except Exception as e:
                print(f"记录环境切换历史失败: {e}")

        return success


# 全局仪表板管理器
dashboard_manager = EnhancedDashboardManager()


@app.route('/')
def index():
    """主页"""
    return render_template('enhanced_dashboard_mobile.html')


@app.route('/api/status')
def get_status():
    """获取状态信息"""
    return jsonify(dashboard_manager.get_dashboard_data())


@app.route('/api/environments')
def get_environments():
    """获取环境信息"""
    return jsonify(dashboard_manager.env_manager.get_environments())


@app.route('/api/switch_environment/<env>')
def switch_environment(env):
    """切换环境"""
    success = dashboard_manager.switch_environment(env)
    return jsonify({
        'success': success,
        'current_environment': dashboard_manager.env_manager.current_environment,
        'message': f"环境切换{'成功' if success else '失败'}"
    })


@app.route('/api/comparison')
def get_comparison():
    """获取环境对比数据"""
    return jsonify(dashboard_manager.get_environment_comparison())


@app.route('/api/distribution')
def get_distribution():
    """获取性能分布数据"""
    return jsonify(dashboard_manager.get_performance_distribution())


@app.route('/api/environment_history')
def get_environment_history():
    """获取环境切换历史"""
    try:
        with sqlite3.connect(dashboard_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT from_env, to_env, timestamp 
                FROM environment_switches 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            history = cursor.fetchall()

            return jsonify({
                'history': [
                    {
                        'from_env': row[0],
                        'to_env': row[1],
                        'timestamp': row[2]
                    }
                    for row in history
                ]
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def update_metrics():
    """更新指标数据"""
    while True:
        try:
            # 更新环境指标
            dashboard_manager.env_manager.update_environment_metrics()

            # 记录当前环境指标
            current_env = dashboard_manager.env_manager.get_current_environment_data()
            try:
                with sqlite3.connect(dashboard_manager.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO environment_metrics 
                        (environment, health_score, response_time, cpu_usage, memory_usage, disk_usage, network_usage)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        dashboard_manager.env_manager.current_environment,
                        current_env.get('health_score', 0),
                        current_env.get('response_time', 0),
                        current_env.get('cpu_usage', 0),
                        current_env.get('memory_usage', 0),
                        current_env.get('disk_usage', 0),
                        current_env.get('network_usage', 0)
                    ))
                    conn.commit()
            except Exception as e:
                print(f"记录环境指标失败: {e}")

            # 通过WebSocket推送更新
            socketio.emit('metrics_update', dashboard_manager.get_dashboard_data())

            time.sleep(30)  # 30秒更新一次
        except Exception as e:
            print(f"更新指标失败: {e}")
            time.sleep(30)


@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print(f"客户端连接: {request.sid}")
    emit('connected', {'message': '连接成功'})


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    print(f"客户端断开连接: {request.sid}")


@socketio.on('switch_environment')
def handle_environment_switch(data):
    """处理环境切换请求"""
    new_env = data.get('environment')
    success = dashboard_manager.switch_environment(new_env)
    emit('environment_switched', {
        'success': success,
        'current_environment': dashboard_manager.env_manager.current_environment,
        'message': f"环境切换{'成功' if success else '失败'}"
    })


def main():
    """主函数"""
    print("🚀 启动增强版Web监控仪表板 (支持环境切换)...")
    print("📍 访问地址: http://localhost:5003")
    print("🔧 功能特性:")
    print("   - 多环境监控切换")
    print("   - 移动端优化")
    print("   - 环境对比图表")
    print("   - 性能分布图表")
    print("   - 实时数据更新")
    print("   - 环境切换历史")

    # 启动指标更新线程
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    metrics_thread.start()

    # 启动Flask应用
    socketio.run(app, host='0.0.0.0', port=5003, debug=False)


if __name__ == '__main__':
    main()
