#!/usr/bin/env python3
"""
告警系统自动化脚本

配置和管理系统告警规则、通知渠道和告警抑制机制
"""

import yaml
import json
import argparse
import smtplib
import requests
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """告警规则配置"""
    name: str
    description: str
    severity: str  # critical, warning, info
    condition: str
    duration: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    enabled: bool = True


@dataclass
class NotificationChannel:
    """通知渠道配置"""
    name: str
    type: str  # email, webhook, slack, telegram
    config: Dict[str, Any]
    enabled: bool = True


class AlertManager:
    """告警管理器"""

    def __init__(self, config_path: str = "config/alerts.yml"):
        self.config_path = Path(config_path)
        self.alert_rules: List[AlertRule] = []
        self.notification_channels: List[NotificationChannel] = []
        self.alert_history: List[Dict[str, Any]] = []

        # 创建配置目录
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化默认配置
        self._init_default_config()

    def _init_default_config(self):
        """初始化默认告警配置"""
        if not self.config_path.exists():
            default_config = {
                "alert_rules": [
                    {
                        "name": "high_cpu_usage",
                        "description": "CPU使用率过高",
                        "severity": "warning",
                        "condition": "cpu_usage > 80",
                        "duration": "5m",
                        "labels": {"service": "rqa2025", "component": "system"},
                        "annotations": {
                            "summary": "CPU使用率超过80%",
                            "description": "系统CPU使用率持续超过80%，可能影响性能"
                        }
                    },
                    {
                        "name": "high_memory_usage",
                        "description": "内存使用率过高",
                        "severity": "warning",
                        "condition": "memory_usage > 85",
                        "duration": "5m",
                        "labels": {"service": "rqa2025", "component": "system"},
                        "annotations": {
                            "summary": "内存使用率超过85%",
                            "description": "系统内存使用率持续超过85%，可能导致性能下降"
                        }
                    },
                    {
                        "name": "disk_space_low",
                        "description": "磁盘空间不足",
                        "severity": "critical",
                        "condition": "disk_usage > 90",
                        "duration": "2m",
                        "labels": {"service": "rqa2025", "component": "storage"},
                        "annotations": {
                            "summary": "磁盘空间不足",
                            "description": "磁盘使用率超过90%，需要立即处理"
                        }
                    },
                    {
                        "name": "service_down",
                        "description": "服务不可用",
                        "severity": "critical",
                        "condition": "service_status != 'running'",
                        "duration": "1m",
                        "labels": {"service": "rqa2025", "component": "service"},
                        "annotations": {
                            "summary": "服务不可用",
                            "description": "关键服务停止运行，需要立即检查"
                        }
                    },
                    {
                        "name": "high_error_rate",
                        "description": "错误率过高",
                        "severity": "warning",
                        "condition": "error_rate > 5",
                        "duration": "3m",
                        "labels": {"service": "rqa2025", "component": "application"},
                        "annotations": {
                            "summary": "错误率超过5%",
                            "description": "应用错误率持续超过5%，需要关注"
                        }
                    }
                ],
                "notification_channels": [
                    {
                        "name": "email_admin",
                        "type": "email",
                        "config": {
                            "smtp_server": "smtp.gmail.com",
                            "smtp_port": 587,
                            "username": "alerts@rqa2025.com",
                            "password": "${SMTP_PASSWORD}",
                            "recipients": ["admin@rqa2025.com"],
                            "subject_template": "[{severity}] {alert_name} - {summary}"
                        }
                    },
                    {
                        "name": "slack_alerts",
                        "type": "slack",
                        "config": {
                            "webhook_url": "${SLACK_WEBHOOK_URL}",
                            "channel": "#alerts",
                            "username": "RQA2025 Alert Bot",
                            "icon_emoji": ":warning:"
                        }
                    },
                    {
                        "name": "webhook_custom",
                        "type": "webhook",
                        "config": {
                            "url": "${WEBHOOK_URL}",
                            "method": "POST",
                            "headers": {"Content-Type": "application/json"},
                            "timeout": 30
                        }
                    }
                ]
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"✅ 已创建默认告警配置文件: {self.config_path}")

        # 加载配置
        self.load_config()

    def load_config(self):
        """加载告警配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 加载告警规则
            self.alert_rules = []
            for rule_data in config.get('alert_rules', []):
                self.alert_rules.append(AlertRule(**rule_data))

            # 加载通知渠道
            self.notification_channels = []
            for channel_data in config.get('notification_channels', []):
                self.notification_channels.append(NotificationChannel(**channel_data))

            logger.info(
                f"✅ 已加载 {len(self.alert_rules)} 个告警规则和 {len(self.notification_channels)} 个通知渠道")

        except Exception as e:
            logger.error(f"❌ 加载告警配置失败: {e}")
            raise

    def save_config(self):
        """保存告警配置"""
        try:
            config = {
                'alert_rules': [asdict(rule) for rule in self.alert_rules],
                'notification_channels': [asdict(channel) for channel in self.notification_channels]
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"✅ 告警配置已保存到: {self.config_path}")

        except Exception as e:
            logger.error(f"❌ 保存告警配置失败: {e}")
            raise

    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules.append(rule)
        self.save_config()
        logger.info(f"✅ 已添加告警规则: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """删除告警规则"""
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]
        self.save_config()
        logger.info(f"✅ 已删除告警规则: {rule_name}")

    def add_notification_channel(self, channel: NotificationChannel):
        """添加通知渠道"""
        self.notification_channels.append(channel)
        self.save_config()
        logger.info(f"✅ 已添加通知渠道: {channel.name}")

    def remove_notification_channel(self, channel_name: str):
        """删除通知渠道"""
        self.notification_channels = [
            ch for ch in self.notification_channels if ch.name != channel_name]
        self.save_config()
        logger.info(f"✅ 已删除通知渠道: {channel_name}")

    def send_email_alert(self, channel: NotificationChannel, alert_data: Dict[str, Any]):
        """发送邮件告警"""
        try:
            config = channel.config
            subject = config['subject_template'].format(**alert_data)

            # 构建邮件内容
            message = f"""
告警详情:
- 告警名称: {alert_data['alert_name']}
- 严重程度: {alert_data['severity']}
- 告警时间: {alert_data['timestamp']}
- 告警描述: {alert_data['description']}
- 触发条件: {alert_data['condition']}

请及时处理此告警。
            """.strip()

            # 发送邮件
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['username'], config['password'])

                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart

                msg = MIMEMultipart()
                msg['From'] = config['username']
                msg['To'] = ', '.join(config['recipients'])
                msg['Subject'] = subject
                msg.attach(MIMEText(message, 'plain', 'utf-8'))

                server.send_message(msg)

            logger.info(f"✅ 邮件告警已发送到: {config['recipients']}")
            return True

        except Exception as e:
            logger.error(f"❌ 发送邮件告警失败: {e}")
            return False

    def send_slack_alert(self, channel: NotificationChannel, alert_data: Dict[str, Any]):
        """发送Slack告警"""
        try:
            config = channel.config

            # 构建Slack消息
            slack_message = {
                "channel": config['channel'],
                "username": config['username'],
                "icon_emoji": config['icon_emoji'],
                "attachments": [{
                    "color": "danger" if alert_data['severity'] == 'critical' else "warning",
                    "title": f"🚨 {alert_data['alert_name']}",
                    "text": alert_data['description'],
                    "fields": [
                        {
                            "title": "严重程度",
                            "value": alert_data['severity'],
                            "short": True
                        },
                        {
                            "title": "触发时间",
                            "value": alert_data['timestamp'],
                            "short": True
                        },
                        {
                            "title": "触发条件",
                            "value": alert_data['condition'],
                            "short": False
                        }
                    ]
                }]
            }

            # 发送到Slack
            response = requests.post(
                config['webhook_url'],
                json=slack_message,
                timeout=config.get('timeout', 30)
            )

            if response.status_code == 200:
                logger.info(f"✅ Slack告警已发送到: {config['channel']}")
                return True
            else:
                logger.error(f"❌ Slack告警发送失败: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ 发送Slack告警失败: {e}")
            return False

    def send_webhook_alert(self, channel: NotificationChannel, alert_data: Dict[str, Any]):
        """发送Webhook告警"""
        try:
            config = channel.config

            # 发送Webhook请求
            response = requests.request(
                method=config.get('method', 'POST'),
                url=config['url'],
                json=alert_data,
                headers=config.get('headers', {}),
                timeout=config.get('timeout', 30)
            )

            if response.status_code in [200, 201, 202]:
                logger.info(f"✅ Webhook告警已发送到: {config['url']}")
                return True
            else:
                logger.error(f"❌ Webhook告警发送失败: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ 发送Webhook告警失败: {e}")
            return False

    def send_alert_notification(self, alert_data: Dict[str, Any]):
        """发送告警通知"""
        success_count = 0
        total_count = 0

        for channel in self.notification_channels:
            if not channel.enabled:
                continue

            total_count += 1
            success = False

            try:
                if channel.type == 'email':
                    success = self.send_email_alert(channel, alert_data)
                elif channel.type == 'slack':
                    success = self.send_slack_alert(channel, alert_data)
                elif channel.type == 'webhook':
                    success = self.send_webhook_alert(channel, alert_data)
                else:
                    logger.warning(f"⚠️ 不支持的通知渠道类型: {channel.type}")
                    continue

                if success:
                    success_count += 1

            except Exception as e:
                logger.error(f"❌ 发送告警到 {channel.name} 失败: {e}")

        # 记录告警历史
        alert_data['notification_success'] = success_count
        alert_data['notification_total'] = total_count
        self.alert_history.append(alert_data)

        logger.info(f"📊 告警通知发送完成: {success_count}/{total_count} 成功")
        return success_count > 0

    def check_alert_conditions(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查告警条件"""
        triggered_alerts = []

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            try:
                # 简单的条件检查（实际应用中需要更复杂的表达式解析）
                condition_met = self._evaluate_condition(rule.condition, metrics)

                if condition_met:
                    alert_data = {
                        'alert_name': rule.name,
                        'description': rule.description,
                        'severity': rule.severity,
                        'condition': rule.condition,
                        'timestamp': datetime.now().isoformat(),
                        'labels': rule.labels,
                        'annotations': rule.annotations,
                        'metrics': metrics
                    }

                    triggered_alerts.append(alert_data)
                    logger.warning(f"🚨 告警触发: {rule.name} - {rule.description}")

            except Exception as e:
                logger.error(f"❌ 检查告警条件失败 {rule.name}: {e}")

        return triggered_alerts

    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """评估告警条件（简化版本）"""
        try:
            # 简单的条件解析（实际应用中需要更复杂的表达式引擎）
            if 'cpu_usage' in condition and 'cpu_usage' in metrics:
                if '>' in condition:
                    threshold = float(condition.split('>')[1].strip())
                    return metrics['cpu_usage'] > threshold
                elif '<' in condition:
                    threshold = float(condition.split('<')[1].strip())
                    return metrics['cpu_usage'] < threshold

            elif 'memory_usage' in condition and 'memory_usage' in metrics:
                if '>' in condition:
                    threshold = float(condition.split('>')[1].strip())
                    return metrics['memory_usage'] > threshold
                elif '<' in condition:
                    threshold = float(condition.split('<')[1].strip())
                    return metrics['memory_usage'] < threshold

            elif 'disk_usage' in condition and 'disk_usage' in metrics:
                if '>' in condition:
                    threshold = float(condition.split('>')[1].strip())
                    return metrics['disk_usage'] > threshold
                elif '<' in condition:
                    threshold = float(condition.split('<')[1].strip())
                    return metrics['disk_usage'] < threshold

            elif 'service_status' in condition and 'service_status' in metrics:
                expected_status = condition.split('!=')[1].strip().strip("'")
                return metrics['service_status'] != expected_status

            elif 'error_rate' in condition and 'error_rate' in metrics:
                if '>' in condition:
                    threshold = float(condition.split('>')[1].strip())
                    return metrics['error_rate'] > threshold
                elif '<' in condition:
                    threshold = float(condition.split('<')[1].strip())
                    return metrics['error_rate'] < threshold

            return False

        except Exception as e:
            logger.error(f"❌ 条件评估失败: {condition} - {e}")
            return False

    def generate_alert_report(self, output_file: str = "reports/alert_report.json"):
        """生成告警报告"""
        try:
            report = {
                "report_info": {
                    "generated_at": datetime.now().isoformat(),
                    "total_alerts": len(self.alert_history),
                    "alert_rules_count": len(self.alert_rules),
                    "notification_channels_count": len(self.notification_channels)
                },
                "alert_rules": [asdict(rule) for rule in self.alert_rules],
                "notification_channels": [asdict(channel) for channel in self.notification_channels],
                "recent_alerts": self.alert_history[-50:] if self.alert_history else [],
                "statistics": {
                    "critical_alerts": len([a for a in self.alert_history if a.get('severity') == 'critical']),
                    "warning_alerts": len([a for a in self.alert_history if a.get('severity') == 'warning']),
                    "info_alerts": len([a for a in self.alert_history if a.get('severity') == 'info']),
                    "successful_notifications": len([a for a in self.alert_history if a.get('notification_success', 0) > 0])
                }
            }

            # 确保输出目录存在
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 告警报告已生成: {output_file}")
            return report

        except Exception as e:
            logger.error(f"❌ 生成告警报告失败: {e}")
            return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="告警系统自动化")
    parser.add_argument("--config", default="config/alerts.yml", help="配置文件路径")
    parser.add_argument("--action", choices=["setup", "test",
                        "report"], default="setup", help="执行动作")
    parser.add_argument("--output", default="reports/alert_report.json", help="报告输出文件")

    args = parser.parse_args()

    try:
        # 创建告警管理器
        alert_manager = AlertManager(args.config)

        if args.action == "setup":
            print("🔧 设置告警系统...")
            print(f"✅ 已配置 {len(alert_manager.alert_rules)} 个告警规则")
            print(f"✅ 已配置 {len(alert_manager.notification_channels)} 个通知渠道")
            print("🎯 告警系统设置完成！")

        elif args.action == "test":
            print("🧪 测试告警系统...")

            # 模拟一些指标数据
            test_metrics = {
                'cpu_usage': 85.5,
                'memory_usage': 78.2,
                'disk_usage': 92.1,
                'service_status': 'running',
                'error_rate': 3.2
            }

            # 检查告警条件
            triggered_alerts = alert_manager.check_alert_conditions(test_metrics)

            if triggered_alerts:
                print(f"🚨 触发 {len(triggered_alerts)} 个告警:")
                for alert in triggered_alerts:
                    print(f"  - {alert['alert_name']}: {alert['description']}")
                    alert_manager.send_alert_notification(alert)
            else:
                print("✅ 未触发任何告警")

        elif args.action == "report":
            print("📊 生成告警报告...")
            report = alert_manager.generate_alert_report(args.output)
            if report:
                print(f"✅ 告警报告已生成: {args.output}")
                print(f"📈 统计信息:")
                print(
                    f"  - 总告警数: {report['statistics']['critical_alerts'] + report['statistics']['warning_alerts'] + report['statistics']['info_alerts']}")
                print(f"  - 严重告警: {report['statistics']['critical_alerts']}")
                print(f"  - 警告告警: {report['statistics']['warning_alerts']}")
                print(f"  - 成功通知: {report['statistics']['successful_notifications']}")

        print("🎉 告警系统操作完成！")

    except Exception as e:
        logger.error(f"❌ 告警系统操作失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()
