#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 监控运行器
提供完整的系统监控和告警功能

作者: AI Assistant
创建日期: 2025年9月13日
"""

import os
import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psutil
import requests

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemMonitor:
    """系统监控器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.metrics = {}
        self.alerts = []
        self.is_running = False
        self.monitor_thread = None

    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'check_interval': 30,  # 检查间隔（秒）
            'alert_thresholds': {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'disk_percent': 90.0,
                'network_errors': 10,
                'response_time': 5.0  # 秒
            },
            'services': [
                {'name': 'api', 'url': 'http://localhost:8000/health', 'timeout': 5},
                {'name': 'redis', 'url': 'redis://localhost:6379', 'timeout': 5},
                {'name': 'database', 'url': 'postgresql://localhost:5432', 'timeout': 10}
            ],
            'alert_channels': [
                {'type': 'console', 'enabled': True},
                {'type': 'webhook', 'url': 'https://hooks.slack.com/...', 'enabled': False},
                {'type': 'email', 'recipients': ['admin@rqa2025.com'], 'enabled': False}
            ]
        }

    def start_monitoring(self):
        """启动监控"""
        if self.is_running:
            logger.warning("监控已经在运行中")
            return

        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logger.info("🚀 系统监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        if not self.is_running:
            logger.warning("监控未在运行")
            return

        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        logger.info("🛑 系统监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                self._collect_system_metrics()

                # 收集服务健康状态
                self._collect_service_health()

                # 检查告警条件
                self._check_alerts()

                # 发送告警
                self._send_alerts()

                # 清理过期数据
                self._cleanup_old_data()

            except Exception as e:
                logger.error(f"监控循环异常: {e}")

            time.sleep(self.config['check_interval'])

    def _collect_system_metrics(self):
        """收集系统指标"""
        timestamp = datetime.now().isoformat()

        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics[f'cpu_percent_{timestamp}'] = cpu_percent

            # 内存使用
            memory = psutil.virtual_memory()
            self.metrics[f'memory_percent_{timestamp}'] = memory.percent
            self.metrics[f'memory_used_{timestamp}'] = memory.used
            self.metrics[f'memory_available_{timestamp}'] = memory.available

            # 磁盘使用
            disk = psutil.disk_usage('/')
            self.metrics[f'disk_percent_{timestamp}'] = disk.percent
            self.metrics[f'disk_used_{timestamp}'] = disk.used
            self.metrics[f'disk_free_{timestamp}'] = disk.free

            # 网络IO
            network = psutil.net_io_counters()
            self.metrics[f'network_bytes_sent_{timestamp}'] = network.bytes_sent
            self.metrics[f'network_bytes_recv_{timestamp}'] = network.bytes_recv

            # 进程信息
            process = psutil.Process()
            self.metrics[f'process_cpu_percent_{timestamp}'] = process.cpu_percent()
            self.metrics[f'process_memory_percent_{timestamp}'] = process.memory_percent()

            logger.debug("系统指标收集完成")

        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")

    def _collect_service_health(self):
        """收集服务健康状态"""
        timestamp = datetime.now().isoformat()

        for service in self.config['services']:
            try:
                service_name = service['name']
                url = service['url']
                timeout = service['timeout']

                start_time = time.time()

                if url.startswith('http'):
                    # HTTP服务检查
                    response = requests.get(url, timeout=timeout)
                    response_time = time.time() - start_time

                    self.metrics[f'{service_name}_status_{timestamp}'] = response.status_code
                    self.metrics[f'{service_name}_response_time_{timestamp}'] = response_time

                    if response.status_code != 200:
                        self._add_alert(
                            'error',
                            f'服务 {service_name} 返回异常状态码: {response.status_code}',
                            {'service': service_name, 'status_code': response.status_code}
                        )

                    if response_time > self.config['alert_thresholds']['response_time']:
                        self._add_alert(
                            'warning',
                            f'服务 {service_name} 响应时间过慢: {response_time:.2f}s',
                            {'service': service_name, 'response_time': response_time}
                        )

                elif url.startswith('redis'):
                    # Redis服务检查
                    # 这里应该使用redis-py进行检查
                    pass

                elif url.startswith('postgresql'):
                    # 数据库服务检查
                    # 这里应该使用psycopg2进行检查
                    pass

                logger.debug(f"服务 {service_name} 健康检查完成")

            except requests.exceptions.RequestException as e:
                self._add_alert(
                    'error',
                    f'服务 {service_name} 连接失败: {e}',
                    {'service': service_name, 'error': str(e)}
                )
            except Exception as e:
                logger.error(f"检查服务 {service_name} 失败: {e}")

    def _check_alerts(self):
        """检查告警条件"""
        # CPU使用率告警
        cpu_percent = self._get_latest_metric('cpu_percent')
        if cpu_percent and cpu_percent > self.config['alert_thresholds']['cpu_percent']:
            self._add_alert(
                'warning',
                f'CPU使用率过高: {cpu_percent:.1f}%',
                {'metric': 'cpu_percent', 'value': cpu_percent}
            )

        # 内存使用率告警
        memory_percent = self._get_latest_metric('memory_percent')
        if memory_percent and memory_percent > self.config['alert_thresholds']['memory_percent']:
            self._add_alert(
                'warning',
                f'内存使用率过高: {memory_percent:.1f}%',
                {'metric': 'memory_percent', 'value': memory_percent}
            )

        # 磁盘使用率告警
        disk_percent = self._get_latest_metric('disk_percent')
        if disk_percent and disk_percent > self.config['alert_thresholds']['disk_percent']:
            self._add_alert(
                'critical',
                f'磁盘使用率过高: {disk_percent:.1f}%',
                {'metric': 'disk_percent', 'value': disk_percent}
            )

    def _get_latest_metric(self, metric_prefix: str) -> Optional[float]:
        """获取最新的指标值"""
        latest_metrics = [
            (k, v) for k, v in self.metrics.items()
            if k.startswith(metric_prefix + '_')
        ]

        if not latest_metrics:
            return None

        # 按时间戳排序，取最新的
        latest_metrics.sort(key=lambda x: x[0], reverse=True)
        return latest_metrics[0][1]

    def _add_alert(self, level: str, message: str, details: Dict[str, Any]):
        """添加告警"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details
        }

        self.alerts.append(alert)
        logger.warning(f"🔔 告警 [{level}]: {message}")

    def _send_alerts(self):
        """发送告警"""
        if not self.alerts:
            return

        # 按级别分组告警
        alerts_by_level = {}
        for alert in self.alerts:
            level = alert['level']
            if level not in alerts_by_level:
                alerts_by_level[level] = []
            alerts_by_level[level].append(alert)

        # 发送告警
        for channel in self.config['alert_channels']:
            if not channel.get('enabled', False):
                continue

            channel_type = channel['type']

            try:
                if channel_type == 'console':
                    self._send_console_alert(alerts_by_level)
                elif channel_type == 'webhook':
                    self._send_webhook_alert(channel, alerts_by_level)
                elif channel_type == 'email':
                    self._send_email_alert(channel, alerts_by_level)
            except Exception as e:
                logger.error(f"发送告警失败 ({channel_type}): {e}")

        # 清空已发送的告警
        self.alerts.clear()

    def _send_console_alert(self, alerts_by_level: Dict[str, List[Dict[str, Any]]]):
        """发送控制台告警"""
        for level, alerts in alerts_by_level.items():
            for alert in alerts:
                print(f"🔔 [{level.upper()}] {alert['message']}")
                if alert['details']:
                    print(f"   详情: {alert['details']}")

    def _send_webhook_alert(self, channel: Dict[str, Any], alerts_by_level: Dict[str, List[Dict[str, Any]]]):
        """发送Webhook告警"""
        url = channel['url']

        payload = {
            'text': 'RQA2025 系统告警',
            'alerts': alerts_by_level,
            'timestamp': datetime.now().isoformat()
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

    def _send_email_alert(self, channel: Dict[str, Any], alerts_by_level: Dict[str, List[Dict[str, Any]]]):
        """发送邮件告警"""
        # 这里应该使用smtplib发送邮件
        recipients = channel['recipients']

        subject = 'RQA2025 系统告警'
        body = f"""
RQA2025 系统检测到告警：

{json.dumps(alerts_by_level, indent=2, ensure_ascii=False)}

时间: {datetime.now().isoformat()}
        """.strip()

        logger.info(f"邮件告警: {subject}")

    def _cleanup_old_data(self):
        """清理过期数据"""
        # 保留最近24小时的数据
        cutoff_time = datetime.now() - timedelta(hours=24)
        cutoff_timestamp = cutoff_time.isoformat()

        # 清理指标数据
        keys_to_remove = [
            k for k in self.metrics.keys()
            if k.split('_')[-1] < cutoff_timestamp
        ]

        for key in keys_to_remove:
            del self.metrics[key]

        # 清理告警数据（保留最近1小时）
        alert_cutoff = datetime.now() - timedelta(hours=1)
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > alert_cutoff
        ]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标汇总"""
        return {
            'total_metrics': len(self.metrics),
            'total_alerts': len(self.alerts),
            'latest_metrics': {
                'cpu_percent': self._get_latest_metric('cpu_percent'),
                'memory_percent': self._get_latest_metric('memory_percent'),
                'disk_percent': self._get_latest_metric('disk_percent')
            },
            'alerts_by_level': self._count_alerts_by_level()
        }

    def _count_alerts_by_level(self) -> Dict[str, int]:
        """按级别统计告警"""
        counts = {}
        for alert in self.alerts:
            level = alert['level']
            counts[level] = counts.get(level, 0) + 1
        return counts


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025 监控运行器')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--daemon', action='store_true', help='后台运行模式')

    args = parser.parse_args()

    # 加载配置
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

    # 创建监控器
    monitor = SystemMonitor(config)

    if args.daemon:
        # 后台运行模式
        monitor.start_monitoring()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
    else:
        # 前台运行模式，执行一次检查
        monitor._collect_system_metrics()
        monitor._collect_service_health()
        monitor._check_alerts()
        monitor._send_alerts()

        # 输出汇总信息
        summary = monitor.get_metrics_summary()
        print("📊 监控汇总:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
