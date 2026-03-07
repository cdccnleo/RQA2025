#!/usr/bin/env python3
"""
发送覆盖率告警脚本

通过各种渠道发送覆盖率告警通知
"""

import os
import sys
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class AlertSender:
    """告警发送器"""

    def __init__(self):
        self.alert_file = os.path.join(project_root, 'coverage_alert.json')
        self.report_file = os.path.join(project_root, 'coverage_alert_report.md')

    def send_alerts(self):
        """发送告警"""
        alert = self._load_alert()
        if not alert:
            print("未找到告警文件")
            return

        # 根据严重程度选择发送渠道
        severity = alert.get('severity', 'low')

        if severity == 'high':
            # 高严重程度：所有渠道都发送
            self._send_email_alert(alert)
            self._send_slack_alert(alert)
            self._send_teams_alert(alert)
        elif severity == 'medium':
            # 中等严重程度：发送邮件和Slack
            self._send_email_alert(alert)
            self._send_slack_alert(alert)
        else:
            # 低严重程度：只发送Slack
            self._send_slack_alert(alert)

        print(f"告警发送完成 (严重程度: {severity})")

    def _load_alert(self) -> Optional[Dict]:
        """加载告警数据"""
        try:
            with open(self.alert_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return None

    def _load_report(self) -> str:
        """加载告警报告"""
        try:
            with open(self.report_file, 'r', encoding='utf-8') as f:
                return f.read()
        except IOError:
            return "无法加载详细报告"

    def _send_email_alert(self, alert: Dict):
        """发送邮件告警"""
        try:
            # 邮件配置
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_user = os.getenv('SMTP_USER')
            smtp_password = os.getenv('SMTP_PASSWORD')
            to_emails = os.getenv('ALERT_EMAILS', '').split(',')

            if not all([smtp_user, smtp_password, to_emails]):
                print("邮件配置不完整，跳过邮件发送")
                return

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"🚨 RQA2025 覆盖率告警 - {alert.get('severity', 'unknown').upper()}"

            # 邮件正文
            body = f"""
RQA2025 系统覆盖率告警

严重程度: {alert.get('severity', 'unknown').upper()}
摘要: {alert.get('summary', 'N/A')}
时间: {alert.get('timestamp', 'N/A')}

详细报告请查看附件或CI/CD系统。

此为自动告警邮件，请勿回复。
            """

            msg.attach(MIMEText(body, 'plain'))

            # 发送邮件
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
            server.quit()

            print("✅ 邮件告警发送成功")

        except Exception as e:
            print(f"❌ 邮件告警发送失败: {e}")

    def _send_slack_alert(self, alert: Dict):
        """发送Slack告警"""
        try:
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if not webhook_url:
                print("Slack Webhook URL未配置，跳过Slack发送")
                return

            severity = alert.get('severity', 'low')
            color = {'high': 'danger', 'medium': 'warning', 'low': 'good'}.get(severity, 'good')

            # 构建Slack消息
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🚨 RQA2025 覆盖率告警"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*严重程度:* {severity.upper()}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*时间:* {alert.get('timestamp', 'N/A')}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*摘要:* {alert.get('summary', 'N/A')}"
                    }
                }
            ]

            # 添加告警详情
            if 'details' in alert and 'alerts' in alert['details']:
                alerts_text = "\n".join([f"• {a['message']}" for a in alert['details']['alerts']])
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*告警详情:*\n{alerts_text}"
                    }
                })

            # 添加建议
            if alert.get('recommendations'):
                recs_text = "\n".join([f"• {rec}" for rec in alert['recommendations'][:3]])  # 只显示前3条
                if len(alert['recommendations']) > 3:
                    recs_text += f"\n• ... 还有{len(alert['recommendations']) - 3}条建议"

                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*建议措施:*\n{recs_text}"
                    }
                })

            message = {
                "blocks": blocks,
                "attachments": [
                    {
                        "color": color,
                        "footer": "RQA2025 质量监控系统",
                        "ts": int(__import__('time').time())
                    }
                ]
            }

            # 发送到Slack
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()

            print("✅ Slack告警发送成功")

        except Exception as e:
            print(f"❌ Slack告警发送失败: {e}")

    def _send_teams_alert(self, alert: Dict):
        """发送Teams告警"""
        try:
            webhook_url = os.getenv('TEAMS_WEBHOOK_URL')
            if not webhook_url:
                print("Teams Webhook URL未配置，跳过Teams发送")
                return

            severity = alert.get('severity', 'low')
            color = {'high': 'FF0000', 'medium': 'FFA500', 'low': '00FF00'}.get(severity, '0000FF')

            # 构建Teams消息
            facts = [
                {
                    "name": "严重程度",
                    "value": severity.upper()
                },
                {
                    "name": "时间",
                    "value": alert.get('timestamp', 'N/A')
                },
                {
                    "name": "摘要",
                    "value": alert.get('summary', 'N/A')
                }
            ]

            # 添加告警详情
            if 'details' in alert and 'alerts' in alert['details']:
                for i, alert_item in enumerate(alert['details']['alerts'], 1):
                    facts.append({
                        "name": f"告警 {i}",
                        "value": alert_item['message']
                    })

            message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color,
                "summary": "RQA2025 覆盖率告警",
                "sections": [
                    {
                        "activityTitle": "🚨 RQA2025 覆盖率告警",
                        "activitySubtitle": f"严重程度: {severity.upper()}",
                        "facts": facts,
                        "markdown": True
                    }
                ],
                "potentialAction": [
                    {
                        "@type": "OpenUri",
                        "name": "查看详细报告",
                        "targets": [
                            {
                                "os": "default",
                                "uri": f"{os.getenv('CI_PROJECT_URL', '#')}/-/jobs"
                            }
                        ]
                    }
                ]
            }

            # 发送到Teams
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()

            print("✅ Teams告警发送成功")

        except Exception as e:
            print(f"❌ Teams告警发送失败: {e}")


def main():
    """主函数"""
    sender = AlertSender()
    sender.send_alerts()


if __name__ == "__main__":
    main()
