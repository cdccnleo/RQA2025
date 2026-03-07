#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版Web监控仪表板
提供多种图表类型和高级监控功能
"""

import sqlite3
import time
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced_monitoring_dashboard_2025'
socketio = SocketIO(app, cors_allowed_origins="*")


class EnhancedDashboardManager:
    """增强版仪表板管理器"""

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

    def get_performance_metrics(self) -> dict:
        """获取性能指标"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 获取平均响应时间
                cursor.execute('''
                    SELECT AVG(response_time) as avg_response_time,
                           MAX(response_time) as max_response_time,
                           MIN(response_time) as min_response_time
                    FROM monitoring_data 
                    WHERE timestamp > ?
                ''', (time.time() - 3600,))

                response_stats = cursor.fetchone()

                # 获取健康度统计
                cursor.execute('''
                    SELECT AVG(health_score) as avg_health,
                           COUNT(CASE WHEN status = 'running' THEN 1 END) as running_count,
                           COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count
                    FROM monitoring_data 
                    WHERE timestamp > ?
                ''', (time.time() - 3600,))

                health_stats = cursor.fetchone()

                return {
                    "response_time": {
                        "average": response_stats[0] or 0,
                        "maximum": response_stats[1] or 0,
                        "minimum": response_stats[2] or 0
                    },
                    "health_score": {
                        "average": health_stats[0] or 0,
                        "running_count": health_stats[1] or 0,
                        "failed_count": health_stats[2] or 0
                    }
                }

        except Exception as e:
            return {
                "response_time": {"average": 0, "maximum": 0, "minimum": 0},
                "health_score": {"average": 0, "running_count": 0, "failed_count": 0},
                "error": str(e)
            }


# 创建仪表板管理器
dashboard_manager = EnhancedDashboardManager()


@app.route('/')
def index():
    """主页"""
    return render_template('enhanced_dashboard.html')


@app.route('/api/status')
def api_status():
    """API: 获取当前状态"""
    return jsonify(dashboard_manager.get_current_status())


@app.route('/api/performance')
def api_performance():
    """API: 获取性能指标"""
    return jsonify(dashboard_manager.get_performance_metrics())


@socketio.on('connect')
def handle_connect():
    """WebSocket连接处理"""
    print('客户端已连接')
    emit('status', dashboard_manager.get_current_status())


def main():
    """主函数"""
    print("🚀 启动增强版Web监控仪表板...")

    print("📊 增强版仪表板功能:")
    print("  ✅ 实时状态监控")
    print("  ✅ 性能指标展示")
    print("  ✅ WebSocket实时更新")
    print("  ✅ RESTful API接口")

    print(f"\n🌐 访问地址: http://localhost:5001")
    print(f"📡 WebSocket: ws://localhost:5001")

    # 启动Flask应用
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)


if __name__ == "__main__":
    main()
