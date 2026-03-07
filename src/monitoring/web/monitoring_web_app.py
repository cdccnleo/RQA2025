# -*- coding: utf-8 -*-
"""
RQA2025 监控Web应用

提供监控面板Web界面和REST API
"""

import os
import json
from flask import Flask, render_template, jsonify, request
try:
    from flask_cors import CORS
except ImportError:
    # 如果flask_cors不可用，创建虚拟的CORS函数
    def CORS(app):
        pass
import logging
from datetime import datetime, timedelta

from ..core.real_time_monitor import get_monitor

logger = logging.getLogger(__name__)


class MonitoringWebApp:
    """监控Web应用"""

    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__,
                        template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
                        static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))

        CORS(self.app)  # 启用跨域支持

        # 获取监控实例
        self.monitor = get_monitor()

        # 注册路由
        self._register_routes()

        # 配置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _register_routes(self):
        """注册路由"""

        @self.app.route('/')
        def index():
            """监控面板主页"""
            return render_template('monitoring_dashboard.html')

        @self.app.route('/api/monitoring/metrics')
        def get_metrics():
            """获取监控指标API"""
            try:
                metrics = self.monitor.get_current_metrics()
                alerts = self.monitor.get_alerts_summary()
                system_status = self.monitor.get_system_status()

                # 转换为前端友好的格式
                metrics_data = {}
                for name, metric in metrics.items():
                    metrics_data[name.replace('_percent', '_percent').replace('_mb', '_mb').replace('_total', '_total').replace('_ms', '_ms')] = metric.value

                return jsonify({
                    'success': True,
                    'metrics': metrics_data,
                    'alerts': alerts,
                    'system_status': system_status,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to get metrics: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/api/monitoring/status')
        def get_status():
            """获取系统状态API"""
            try:
                status = self.monitor.get_system_status()
                return jsonify({
                    'success': True,
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to get status: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/api/monitoring/alerts')
        def get_alerts():
            """获取告警信息API"""
            try:
                alerts = self.monitor.get_alerts_summary()
                return jsonify({
                    'success': True,
                    'alerts': alerts,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to get alerts: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/api/monitoring/history')
        def get_history():
            """获取历史指标API"""
            try:
                hours = int(request.args.get('hours', 1))

                # 这里应该从历史存储中获取数据
                # 暂时返回当前数据
                metrics = self.monitor.get_current_metrics()

                return jsonify({
                    'success': True,
                    'history': [{
                        'timestamp': metric.timestamp.isoformat(),
                        'name': metric.name,
                        'value': metric.value
                    } for metric in metrics.values()],
                    'hours': hours,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to get history: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/api/monitoring/update', methods=['POST'])
        def update_metric():
            """更新业务指标API"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({
                        'success': False,
                        'error': 'No data provided'
                    }), 400

                name = data.get('name')
                value = data.get('value')

                if not name or value is None:
                    return jsonify({
                        'success': False,
                        'error': 'Missing name or value'
                    }), 400

                self.monitor.update_business_metric(name, float(value))

                return jsonify({
                    'success': True,
                    'message': f'Updated metric {name} to {value}',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                error_msg = str(e)
                # 检查是否是400类错误（Bad Request）
                if '400' in error_msg or 'Bad Request' in error_msg:
                    logger.warning(f"Bad request to update metric: {e}")
                    return jsonify({
                        'success': False,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    }), 400
                else:
                    logger.error(f"Failed to update metric: {e}")
                    return jsonify({
                        'success': False,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    }), 500

        @self.app.route('/health')
        def health_check():
            """健康检查端点"""
            try:
                status = self.monitor.get_system_status()
                return jsonify({
                    'status': 'healthy' if status['system_health'] == 'healthy' else 'unhealthy',
                    'timestamp': datetime.now().isoformat(),
                    'details': status
                })
            except Exception as e:
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500

    def start(self):
        """启动Web应用"""
        logger.info(f"Starting monitoring web app on {self.host}:{self.port}")

        # 启动监控系统
        self.monitor.start_monitoring()

        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,  # 生产环境禁用debug
                threaded=True
            )
        except KeyboardInterrupt:
            logger.info("Shutting down monitoring web app")
        finally:
            self.monitor.stop_monitoring()

    def stop(self):
        """停止Web应用"""
        logger.info("Stopping monitoring web app")
        self.monitor.stop_monitoring()


# 全局应用实例
_web_app_instance = None


def get_web_app(host='0.0.0.0', port=5000):
    """获取全局Web应用实例"""
    global _web_app_instance
    if _web_app_instance is None:
        _web_app_instance = MonitoringWebApp(host, port)
    return _web_app_instance


def start_web_app(host='0.0.0.0', port=5000):
    """启动监控Web应用"""
    app = get_web_app(host, port)
    app.start()


def stop_web_app():
    """停止监控Web应用"""
    global _web_app_instance
    if _web_app_instance:
        _web_app_instance.stop()
        _web_app_instance = None


if __name__ == "__main__":
    start_web_app()
