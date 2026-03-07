#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQA2025 监控系统启动脚本

启动完整的实时监控系统，包括：
- 实时指标收集
- 告警系统
- Web监控面板
- 通知服务
"""

import sys
import os
import logging
import argparse
import signal
import time
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.core.real_time_monitor import get_monitor, start_monitoring, stop_monitoring
from src.monitoring.web.monitoring_web_app import get_web_app, start_web_app, stop_web_app
from src.monitoring.alert.alert_notifier import get_notifier, start_alert_notifications, stop_alert_notifications, NotificationConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitoring.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class MonitoringSystem:
    """监控系统管理器"""

    def __init__(self,
                 web_host: str = '0.0.0.0',
                 web_port: int = 5000,
                 notification_config: Optional[NotificationConfig] = None):
        self.web_host = web_host
        self.web_port = web_port
        self.notification_config = notification_config or NotificationConfig()

        # 组件实例
        self.monitor = None
        self.web_app = None
        self.notifier = None

        # 运行状态
        self.running = False

    def start(self):
        """启动监控系统"""
        logger.info("Starting RQA2025 Monitoring System...")

        try:
            # 1. 启动实时监控
            self.monitor = get_monitor()
            self.monitor.start_monitoring()
            logger.info("Real-time monitor started")

            # 2. 启动告警通知器
            self.notifier = get_notifier(self.notification_config)
            self.notifier.start()
            logger.info("Alert notifier started")

            # 3. 将告警回调添加到监控系统
            self.monitor.add_alert_callback(self.notifier.notify_alert)

            # 4. 启动Web应用
            self.web_app = get_web_app(self.web_host, self.web_port)

            self.running = True
            logger.info(f"Monitoring system started successfully on {self.web_host}:{self.web_port}")
            logger.info("Access monitoring dashboard at: http://localhost:5000")

            # 启动Web服务器（阻塞调用）
            self.web_app.start()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            self.stop()
        except Exception as e:
            logger.error(f"Failed to start monitoring system: {e}")
            self.stop()
            raise

    def stop(self):
        """停止监控系统"""
        logger.info("Stopping RQA2025 Monitoring System...")

        try:
            # 停止各个组件
            if self.web_app:
                stop_web_app()

            if self.notifier:
                stop_alert_notifications()

            if self.monitor:
                stop_monitoring()

            self.running = False
            logger.info("Monitoring system stopped successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def add_custom_collector(self, name: str, collector_func):
        """添加自定义指标收集器"""
        if self.monitor:
            self.monitor.add_custom_collector(name, collector_func)
            logger.info(f"Added custom collector: {name}")

    def add_alert_rule(self, rule):
        """添加告警规则"""
        if self.monitor:
            self.monitor.add_alert_rule(rule)
            logger.info(f"Added alert rule: {rule.name}")


def create_notification_config_from_env() -> NotificationConfig:
    """从环境变量创建通知配置"""
    config = NotificationConfig()

    # 邮件配置
    config.email_enabled = os.getenv('MONITORING_EMAIL_ENABLED', 'false').lower() == 'true'
    config.email_smtp_server = os.getenv('MONITORING_EMAIL_SMTP_SERVER', '')
    config.email_smtp_port = int(os.getenv('MONITORING_EMAIL_SMTP_PORT', '587'))
    config.email_username = os.getenv('MONITORING_EMAIL_USERNAME', '')
    config.email_password = os.getenv('MONITORING_EMAIL_PASSWORD', '')
    config.email_from = os.getenv('MONITORING_EMAIL_FROM', '')
    config.email_to = os.getenv('MONITORING_EMAIL_TO', '').split(',') if os.getenv('MONITORING_EMAIL_TO') else []

    # 微信配置
    config.wechat_enabled = os.getenv('MONITORING_WECHAT_ENABLED', 'false').lower() == 'true'
    config.wechat_webhook_url = os.getenv('MONITORING_WECHAT_WEBHOOK_URL', '')
    config.wechat_corp_id = os.getenv('MONITORING_WECHAT_CORP_ID', '')
    config.wechat_corp_secret = os.getenv('MONITORING_WECHAT_CORP_SECRET', '')
    config.wechat_agent_id = int(os.getenv('MONITORING_WECHAT_AGENT_ID', '0'))

    # Slack配置
    config.slack_enabled = os.getenv('MONITORING_SLACK_ENABLED', 'false').lower() == 'true'
    config.slack_webhook_url = os.getenv('MONITORING_SLACK_WEBHOOK_URL', '')
    config.slack_channel = os.getenv('MONITORING_SLACK_CHANNEL', '#alerts')

    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 Monitoring System')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Web server port (default: 5000)')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 创建通知配置
    notification_config = create_notification_config_from_env()

    # 创建监控系统
    monitoring_system = MonitoringSystem(
        web_host=args.host,
        web_port=args.port,
        notification_config=notification_config
    )

    # 设置信号处理器
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        monitoring_system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动系统
    try:
        monitoring_system.start()
    except Exception as e:
        logger.error(f"Monitoring system failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()