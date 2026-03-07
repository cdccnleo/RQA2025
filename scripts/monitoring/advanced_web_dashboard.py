#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级Web监控仪表板
提供多种图表类型和高级监控功能
"""

import sqlite3
import time
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'advanced_monitoring_dashboard_2025'
socketio = SocketIO(app, cors_allowed_origins="*")


class AdvancedDashboardManager:
    """高级仪表板管理器"""

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

    def get_status_distribution(self) -> dict:
        """获取状态分布数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT status, COUNT(*) as count
                    FROM monitoring_data 
                    WHERE timestamp > ?
                    GROUP BY status
                ''', (time.time() - 3600,))

                distribution = {}
                for row in cursor.fetchall():
                    status, count = row
                    distribution[status] = count

                return distribution

        except Exception as e:
            return {"error": str(e)}


# 创建仪表板管理器
dashboard_manager = AdvancedDashboardManager()


@app.route('/')
def index():
    """主页"""
    return render_template('advanced_dashboard.html')


@app.route('/api/status')
def api_status():
    """API: 获取当前状态"""
    return jsonify(dashboard_manager.get_current_status())


@app.route('/api/distribution')
def api_distribution():
    """API: 获取状态分布"""
    return jsonify(dashboard_manager.get_status_distribution())


@socketio.on('connect')
def handle_connect():
    """WebSocket连接处理"""
    print('客户端已连接')
    emit('status', dashboard_manager.get_current_status())


def main():
    """主函数"""
    print("🚀 启动高级Web监控仪表板...")

    print("📊 高级仪表板功能:")
    print("  ✅ 实时状态监控")
    print("  ✅ 状态分布图表")
    print("  ✅ WebSocket实时更新")
    print("  ✅ RESTful API接口")

    print(f"\n🌐 访问地址: http://localhost:5002")
    print(f"📡 WebSocket: ws://localhost:5002")

    # 启动Flask应用
    socketio.run(app, host='0.0.0.0', port=5002, debug=True)


if __name__ == "__main__":
    main()
