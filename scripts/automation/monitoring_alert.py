#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控告警集成脚本
支持多种告警方式、告警规则配置、告警历史管理等功能
"""
import os
import json
import requests
import time
import logging
import argparse
import smtplib
import threading
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric: str  # cpu, memory, response_time, error_rate
    threshold: float
    operator: str  # >, <, >=, <=, ==
    duration: int  # 持续时间(秒)
    severity: str  # info, warning, critical
    enabled: bool = True


@dataclass
class AlertEvent:
    """告警事件"""
    id: str
    rule_name: str
    metric: str
    value: float
    threshold: float
    severity: str
    timestamp: datetime
    message: str
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class AlertChannel:
    """告警通道"""
    name: str
    type: str  # email, webhook, slack, sms
    config: Dict
    enabled: bool = True


class MonitoringAlertSystem:
    """监控告警系统"""

    def __init__(self, config_file: str = "alert_config.json"):
        self.config_file = config_file
        self.rules = []
        self.channels = []
        self.alert_history = []
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = {}

        self.load_config()

    def load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                    # 加载告警规则
                    self.rules = [
                        AlertRule(**rule_config)
                        for rule_config in config.get("rules", [])
                    ]

                    # 加载告警通道
                    self.channels = [
                        AlertChannel(**channel_config)
                        for channel_config in config.get("channels", [])
                    ]
            else:
                # 默认配置
                self._create_default_config()

        except Exception as e:
            logger.error(f"加载告警配置失败: {e}")
            self._create_default_config()

    def _create_default_config(self):
        """创建默认配置"""
        # 默认告警规则
        self.rules = [
            AlertRule(
                name="高CPU使用率",
                metric="cpu",
                threshold=80.0,
                operator=">",
                duration=60,
                severity="warning"
            ),
            AlertRule(
                name="高内存使用率",
                metric="memory",
                threshold=85.0,
                operator=">",
                duration=60,
                severity="warning"
            ),
            AlertRule(
                name="服务响应时间过长",
                metric="response_time",
                threshold=5.0,
                operator=">",
                duration=30,
                severity="critical"
            ),
            AlertRule(
                name="错误率过高",
                metric="error_rate",
                threshold=0.1,
                operator=">",
                duration=60,
                severity="critical"
            )
        ]

        # 默认告警通道
        self.channels = [
            AlertChannel(
                name="邮件告警",
                type="email",
                config={
                    "smtp_server": "smtp.example.com",
                    "smtp_port": 587,
                    "username": "alert@example.com",
                    "password": "alert_pass",
                    "from_email": "alert@example.com",
                    "to_emails": ["admin@example.com"]
                }
            ),
            AlertChannel(
                name="Webhook告警",
                type="webhook",
                config={
                    "url": "https://webhook.example.com/alert",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"}
                }
            )
        ]

        self.save_config()

    def save_config(self):
        """保存配置"""
        try:
            config = {
                "rules": [asdict(rule) for rule in self.rules],
                "channels": [asdict(channel) for channel in self.channels]
            }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 告警配置已保存: {self.config_file}")

        except Exception as e:
            logger.error(f"保存告警配置失败: {e}")

    def start_monitoring(self, interval: int = 30):
        """开始监控"""
        logger.info("开始监控告警系统...")
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        logger.info("停止监控告警系统...")
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集指标
                metrics = self._collect_metrics()

                # 检查告警规则
                self._check_alert_rules(metrics)

                # 清理过期告警
                self._cleanup_expired_alerts()

                time.sleep(interval)

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(10)

    def _collect_metrics(self) -> Dict[str, float]:
        """收集指标"""
        metrics = {}

        try:
            # CPU使用率
            metrics["cpu"] = psutil.cpu_percent(interval=1)

            # 内存使用率
            memory = psutil.virtual_memory()
            metrics["memory"] = memory.percent

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            metrics["disk"] = (disk.used / disk.total) * 100

            # 网络IO
            network = psutil.net_io_counters()
            metrics["network_sent"] = network.bytes_sent
            metrics["network_recv"] = network.bytes_recv

            # 模拟服务指标
            metrics["response_time"] = self._simulate_response_time()
            metrics["error_rate"] = self._simulate_error_rate()

            # 记录指标历史
            timestamp = datetime.now()
            for metric, value in metrics.items():
                if metric not in self.metrics_history:
                    self.metrics_history[metric] = []
                self.metrics_history[metric].append({
                    "timestamp": timestamp,
                    "value": value
                })

                # 保留最近100个数据点
                if len(self.metrics_history[metric]) > 100:
                    self.metrics_history[metric] = self.metrics_history[metric][-100:]

            logger.debug(f"收集指标: {metrics}")

        except Exception as e:
            logger.error(f"收集指标失败: {e}")

        return metrics

    def _simulate_response_time(self) -> float:
        """模拟响应时间"""
        # 模拟配置管理服务的响应时间
        try:
            response = requests.get("http://localhost:8080/api/health", timeout=5)
            return response.elapsed.total_seconds()
        except Exception:
            return 0.0

    def _simulate_error_rate(self) -> float:
        """模拟错误率"""
        # 模拟错误率，实际应从日志或监控系统获取
        import random
        return random.uniform(0.0, 0.05)  # 0-5% 错误率

    def _check_alert_rules(self, metrics: Dict[str, float]):
        """检查告警规则"""
        for rule in self.rules:
            if not rule.enabled:
                continue

            if rule.metric not in metrics:
                continue

            value = metrics[rule.metric]
            triggered = self._evaluate_rule(rule, value)

            if triggered:
                # 检查是否持续触发
                if self._is_rule_continuously_triggered(rule, value):
                    self._trigger_alert(rule, value)

    def _evaluate_rule(self, rule: AlertRule, value: float) -> bool:
        """评估告警规则"""
        if rule.operator == ">":
            return value > rule.threshold
        elif rule.operator == "<":
            return value < rule.threshold
        elif rule.operator == ">=":
            return value >= rule.threshold
        elif rule.operator == "<=":
            return value <= rule.threshold
        elif rule.operator == "==":
            return value == rule.threshold
        else:
            return False

    def _is_rule_continuously_triggered(self, rule: AlertRule, current_value: float) -> bool:
        """检查规则是否持续触发"""
        if rule.metric not in self.metrics_history:
            return False

        # 获取最近的数据点
        recent_data = self.metrics_history[rule.metric]
        if not recent_data:
            return False

        # 检查持续时间内的数据点
        cutoff_time = datetime.now() - timedelta(seconds=rule.duration)
        relevant_data = [
            data for data in recent_data
            if data["timestamp"] >= cutoff_time
        ]

        if len(relevant_data) < 2:  # 至少需要2个数据点
            return False

        # 检查是否所有数据点都触发规则
        for data in relevant_data:
            if not self._evaluate_rule(rule, data["value"]):
                return False

        return True

    def _trigger_alert(self, rule: AlertRule, value: float):
        """触发告警"""
        # 检查是否已有相同告警
        for alert in self.alert_history:
            if (alert.rule_name == rule.name and
                not alert.resolved and
                    (datetime.now() - alert.timestamp).seconds < 300):  # 5分钟内不重复告警
                return

        # 创建告警事件
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rule.name}"
        message = f"{rule.name}: {rule.metric} = {value:.2f} {rule.operator} {rule.threshold}"

        alert = AlertEvent(
            id=alert_id,
            rule_name=rule.name,
            metric=rule.metric,
            value=value,
            threshold=rule.threshold,
            severity=rule.severity,
            timestamp=datetime.now(),
            message=message
        )

        self.alert_history.append(alert)

        # 发送告警
        self._send_alerts(alert)

        logger.warning(f"🚨 触发告警: {message}")

    def _send_alerts(self, alert: AlertEvent):
        """发送告警"""
        for channel in self.channels:
            if not channel.enabled:
                continue

            try:
                if channel.type == "email":
                    self._send_email_alert(channel, alert)
                elif channel.type == "webhook":
                    self._send_webhook_alert(channel, alert)
                elif channel.type == "slack":
                    self._send_slack_alert(channel, alert)
                else:
                    logger.warning(f"未知的告警通道类型: {channel.type}")

            except Exception as e:
                logger.error(f"发送告警到 {channel.name} 失败: {e}")

    def _send_email_alert(self, channel: AlertChannel, alert: AlertEvent):
        """发送邮件告警"""
        try:
            config = channel.config

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"配置管理告警 - {alert.severity.upper()}"

            # 邮件内容
            body = f"""
告警详情:
- 规则: {alert.rule_name}
- 指标: {alert.metric}
- 当前值: {alert.value:.2f}
- 阈值: {alert.threshold}
- 严重程度: {alert.severity}
- 时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- 消息: {alert.message}

请及时处理此告警。
            """

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # 发送邮件
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()

            logger.info(f"✅ 邮件告警已发送: {alert.rule_name}")

        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")

    def _send_webhook_alert(self, channel: AlertChannel, alert: AlertEvent):
        """发送Webhook告警"""
        try:
            config = channel.config

            # 构建告警数据
            alert_data = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat(),
                "message": alert.message
            }

            # 发送HTTP请求
            response = requests.post(
                config['url'],
                json=alert_data,
                headers=config.get('headers', {}),
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"✅ Webhook告警已发送: {alert.rule_name}")
            else:
                logger.error(f"Webhook告警发送失败: {response.status_code}")

        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")

    def _send_slack_alert(self, channel: AlertChannel, alert: AlertEvent):
        """发送Slack告警"""
        try:
            config = channel.config

            # 构建Slack消息
            slack_data = {
                "text": f"🚨 配置管理告警",
                "attachments": [{
                    "color": "danger" if alert.severity == "critical" else "warning",
                    "fields": [
                        {"title": "规则", "value": alert.rule_name, "short": True},
                        {"title": "指标", "value": alert.metric, "short": True},
                        {"title": "当前值", "value": f"{alert.value:.2f}", "short": True},
                        {"title": "阈值", "value": f"{alert.threshold}", "short": True},
                        {"title": "严重程度", "value": alert.severity, "short": True},
                        {"title": "时间", "value": alert.timestamp.strftime(
                            '%Y-%m-%d %H:%M:%S'), "short": True},
                        {"title": "消息", "value": alert.message, "short": False}
                    ]
                }]
            }

            # 发送到Slack
            response = requests.post(
                config['webhook_url'],
                json=slack_data,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"✅ Slack告警已发送: {alert.rule_name}")
            else:
                logger.error(f"Slack告警发送失败: {response.status_code}")

        except Exception as e:
            logger.error(f"发送Slack告警失败: {e}")

    def _cleanup_expired_alerts(self):
        """清理过期告警"""
        current_time = datetime.now()
        expired_alerts = []

        for alert in self.alert_history:
            # 清理7天前的告警
            if (current_time - alert.timestamp).days > 7:
                expired_alerts.append(alert)

        for alert in expired_alerts:
            self.alert_history.remove(alert)

        if expired_alerts:
            logger.info(f"清理了 {len(expired_alerts)} 个过期告警")

    def acknowledge_alert(self, alert_id: str):
        """确认告警"""
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"✅ 告警已确认: {alert_id}")
                return True
        return False

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"✅ 告警已解决: {alert_id}")
                return True
        return False

    def get_active_alerts(self) -> List[AlertEvent]:
        """获取活跃告警"""
        return [alert for alert in self.alert_history if not alert.resolved]

    def get_alert_history(self, days: int = 7) -> List[AlertEvent]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(days=days)
        return [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules.append(rule)
        self.save_config()
        logger.info(f"✅ 告警规则已添加: {rule.name}")

    def remove_rule(self, rule_name: str):
        """删除告警规则"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        self.save_config()
        logger.info(f"✅ 告警规则已删除: {rule_name}")

    def add_channel(self, channel: AlertChannel):
        """添加告警通道"""
        self.channels.append(channel)
        self.save_config()
        logger.info(f"✅ 告警通道已添加: {channel.name}")

    def remove_channel(self, channel_name: str):
        """删除告警通道"""
        self.channels = [channel for channel in self.channels if channel.name != channel_name]
        self.save_config()
        logger.info(f"✅ 告警通道已删除: {channel_name}")

    def generate_alert_report(self) -> Dict:
        """生成告警报告"""
        active_alerts = self.get_active_alerts()
        recent_alerts = self.get_alert_history(1)  # 最近1天

        # 按严重程度统计
        severity_stats = {}
        for alert in recent_alerts:
            severity = alert.severity
            if severity not in severity_stats:
                severity_stats[severity] = 0
            severity_stats[severity] += 1

        # 按规则统计
        rule_stats = {}
        for alert in recent_alerts:
            rule_name = alert.rule_name
            if rule_name not in rule_stats:
                rule_stats[rule_name] = 0
            rule_stats[rule_name] += 1

        return {
            "timestamp": datetime.now().isoformat(),
            "active_alerts_count": len(active_alerts),
            "recent_alerts_count": len(recent_alerts),
            "severity_stats": severity_stats,
            "rule_stats": rule_stats,
            "active_alerts": [
                {
                    "id": alert.id,
                    "rule_name": alert.rule_name,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp.isoformat(),
                    "message": alert.message,
                    "acknowledged": alert.acknowledged
                }
                for alert in active_alerts
            ]
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="监控告警系统")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 启动监控
    start_parser = subparsers.add_parser('start', help='启动监控')
    start_parser.add_argument('--interval', type=int, default=30, help='监控间隔(秒)')

    # 查看告警
    alerts_parser = subparsers.add_parser('alerts', help='查看告警')
    alerts_parser.add_argument('--active', action='store_true', help='只显示活跃告警')
    alerts_parser.add_argument('--days', type=int, default=7, help='显示最近几天的告警')

    # 确认告警
    ack_parser = subparsers.add_parser('ack', help='确认告警')
    ack_parser.add_argument('alert_id', help='告警ID')

    # 解决告警
    resolve_parser = subparsers.add_parser('resolve', help='解决告警')
    resolve_parser.add_argument('alert_id', help='告警ID')

    # 添加规则
    add_rule_parser = subparsers.add_parser('add-rule', help='添加告警规则')
    add_rule_parser.add_argument('name', help='规则名称')
    add_rule_parser.add_argument('metric', help='监控指标')
    add_rule_parser.add_argument('threshold', type=float, help='阈值')
    add_rule_parser.add_argument('operator', help='操作符(>, <, >=, <=, ==)')
    add_rule_parser.add_argument('duration', type=int, help='持续时间(秒)')
    add_rule_parser.add_argument('severity', help='严重程度(info, warning, critical)')

    # 删除规则
    remove_rule_parser = subparsers.add_parser('remove-rule', help='删除告警规则')
    remove_rule_parser.add_argument('name', help='规则名称')

    # 生成报告
    report_parser = subparsers.add_parser('report', help='生成告警报告')

    args = parser.parse_args()

    # 创建告警系统
    alert_system = MonitoringAlertSystem()

    try:
        if args.command == 'start':
            print("🚀 启动监控告警系统...")
            alert_system.start_monitoring(args.interval)

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 停止监控...")
                alert_system.stop_monitoring()

        elif args.command == 'alerts':
            if args.active:
                alerts = alert_system.get_active_alerts()
                print(f"\n📊 活跃告警 ({len(alerts)} 个):")
            else:
                alerts = alert_system.get_alert_history(args.days)
                print(f"\n📊 告警历史 (最近{args.days}天, {len(alerts)} 个):")

            print("="*80)
            for alert in alerts:
                status_icon = "🔴" if alert.severity == "critical" else "🟡" if alert.severity == "warning" else "🔵"
                ack_mark = " ✅" if alert.acknowledged else ""
                resolve_mark = " ✅" if alert.resolved else ""
                print(
                    f"{status_icon} [{alert.id}] {alert.rule_name} ({alert.severity}){ack_mark}{resolve_mark}")
                print(f"    {alert.message}")
                print(f"    时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print()

        elif args.command == 'ack':
            if alert_system.acknowledge_alert(args.alert_id):
                print(f"✅ 告警已确认: {args.alert_id}")
            else:
                print(f"❌ 告警不存在: {args.alert_id}")

        elif args.command == 'resolve':
            if alert_system.resolve_alert(args.alert_id):
                print(f"✅ 告警已解决: {args.alert_id}")
            else:
                print(f"❌ 告警不存在: {args.alert_id}")

        elif args.command == 'add-rule':
            rule = AlertRule(
                name=args.name,
                metric=args.metric,
                threshold=args.threshold,
                operator=args.operator,
                duration=args.duration,
                severity=args.severity
            )
            alert_system.add_rule(rule)
            print(f"✅ 告警规则已添加: {args.name}")

        elif args.command == 'remove-rule':
            alert_system.remove_rule(args.name)
            print(f"✅ 告警规则已删除: {args.name}")

        elif args.command == 'report':
            report = alert_system.generate_alert_report()
            print("\n📊 告警报告")
            print("="*60)
            print(f"⏰ 时间: {report['timestamp']}")
            print(f"🚨 活跃告警: {report['active_alerts_count']} 个")
            print(f"📈 最近告警: {report['recent_alerts_count']} 个")

            print("\n📊 严重程度统计:")
            for severity, count in report['severity_stats'].items():
                print(f"  {severity}: {count} 个")

            print("\n📊 规则统计:")
            for rule, count in report['rule_stats'].items():
                print(f"  {rule}: {count} 个")

            if report['active_alerts']:
                print("\n🚨 活跃告警详情:")
                for alert in report['active_alerts']:
                    print(f"  - {alert['rule_name']}: {alert['message']}")

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"操作失败: {e}")
        print(f"\n❌ 操作失败: {e}")


if __name__ == "__main__":
    main()
