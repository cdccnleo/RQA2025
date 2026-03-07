#!/usr/bin/env python3
"""
RQA2025 系统监控和告警脚本
实时监控系统健康状态和性能指标
"""

import os
import sys
import time
import json
import psutil
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class SystemMonitor:
    """系统监控器"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "monitor-config.json"
        self.config = self._load_config()
        self.logger = self._setup_logger()
        self.alerts_history = []
        self.metrics_history = []

        # 监控指标
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_usage": [],
            "network_io": [],
            "system_load": []
        }

    def _load_config(self) -> Dict[str, Any]:
        """加载监控配置"""
        default_config = {
            "monitoring": {
                "interval": 30,  # 监控间隔(秒)
                "retention_days": 7,  # 数据保留天数
                "alert_thresholds": {
                    "cpu_usage": 85.0,
                    "memory_usage": 90.0,
                    "disk_usage": 95.0,
                    "response_time": 2.0
                }
            },
            "services": {
                "trading_engine": {
                    "url": "http://localhost:8001/health",
                    "timeout": 5
                },
                "risk_monitor": {
                    "url": "http://localhost:8002/health",
                    "timeout": 5
                },
                "data_processor": {
                    "url": "http://localhost:8003/health",
                    "timeout": 5
                }
            },
            "alerts": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "alerts@rqa2025.com",
                    "receiver_emails": ["admin@rqa2025.com"],
                    "password": "your_password"
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "https://hooks.slack.com/...",
                    "channel": "#alerts"
                }
            }
        }

        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 合并配置
                for section, values in user_config.items():
                    if section in default_config:
                        default_config[section].update(values)
                    else:
                        default_config[section] = values

        return default_config

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('RQA2025_Monitor')
        logger.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"monitor_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "used": psutil.virtual_memory().used,
                "usage_percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "usage_percent": psutil.disk_usage('/').percent
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
                "packets_sent": psutil.net_io_counters().packets_sent,
                "packets_recv": psutil.net_io_counters().packets_recv
            }
        }

        return metrics

    def check_service_health(self) -> Dict[str, Dict[str, Any]]:
        """检查服务健康状态"""
        results = {}

        for service_name, service_config in self.config["services"].items():
            try:
                url = service_config["url"]
                timeout = service_config.get("timeout", 5)

                start_time = time.time()
                response = requests.get(url, timeout=timeout)
                response_time = time.time() - start_time

                if response.status_code == 200:
                    health_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    results[service_name] = {
                        "status": "healthy",
                        "response_time": response_time,
                        "http_status": response.status_code,
                        "details": health_data
                    }
                else:
                    results[service_name] = {
                        "status": "unhealthy",
                        "response_time": response_time,
                        "http_status": response.status_code,
                        "error": f"HTTP {response.status_code}"
                    }

            except requests.exceptions.RequestException as e:
                results[service_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "response_time": None
                }
            except Exception as e:
                results[service_name] = {
                    "status": "error",
                    "error": f"检查失败: {str(e)}",
                    "response_time": None
                }

        return results

    def check_alert_conditions(self, system_metrics: Dict[str, Any],
                             service_health: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检查告警条件"""
        alerts = []
        thresholds = self.config["monitoring"]["alert_thresholds"]

        # 系统指标告警
        if system_metrics["cpu"]["usage_percent"] > thresholds["cpu_usage"]:
            alerts.append({
                "type": "system",
                "level": "warning",
                "metric": "cpu_usage",
                "value": system_metrics["cpu"]["usage_percent"],
                "threshold": thresholds["cpu_usage"],
                "message": ".1f"            })

        if system_metrics["memory"]["usage_percent"] > thresholds["memory_usage"]:
            alerts.append({
                "type": "system",
                "level": "critical",
                "metric": "memory_usage",
                "value": system_metrics["memory"]["usage_percent"],
                "threshold": thresholds["memory_usage"],
                "message": ".1f"            })

        if system_metrics["disk"]["usage_percent"] > thresholds["disk_usage"]:
            alerts.append({
                "type": "system",
                "level": "critical",
                "metric": "disk_usage",
                "value": system_metrics["disk"]["usage_percent"],
                "threshold": thresholds["disk_usage"],
                "message": ".1f"            })

        # 服务健康告警
        for service_name, health_info in service_health.items():
            if health_info["status"] != "healthy":
                alerts.append({
                    "type": "service",
                    "level": "critical",
                    "service": service_name,
                    "status": health_info["status"],
                    "error": health_info.get("error", "未知错误"),
                    "message": f"服务 {service_name} 状态异常: {health_info['status']}"
                })

            # 响应时间告警
            response_time = health_info.get("response_time")
            if response_time and response_time > thresholds["response_time"]:
                alerts.append({
                    "type": "performance",
                    "level": "warning",
                    "service": service_name,
                    "metric": "response_time",
                    "value": response_time,
                    "threshold": thresholds["response_time"],
                    "message": ".2f"                })

        return alerts

    def send_alerts(self, alerts: List[Dict[str, Any]]):
        """发送告警"""
        if not alerts:
            return

        # 记录告警历史
        for alert in alerts:
            alert["timestamp"] = datetime.now().isoformat()
            self.alerts_history.append(alert)

        # 发送邮件告警
        if self.config["alerts"]["email"]["enabled"]:
            self._send_email_alerts(alerts)

        # 发送Slack告警
        if self.config["alerts"]["slack"]["enabled"]:
            self._send_slack_alerts(alerts)

        # 记录到日志
        for alert in alerts:
            level = alert.get("level", "info").upper()
            message = alert.get("message", "未指定告警信息")
            self.logger.log(getattr(logging, level, logging.INFO), f"告警: {message}")

    def _send_email_alerts(self, alerts: List[Dict[str, Any]]):
        """发送邮件告警"""
        try:
            email_config = self.config["alerts"]["email"]

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = email_config["sender_email"]
            msg['To'] = ", ".join(email_config["receiver_emails"])
            msg['Subject'] = f"RQA2025 系统告警 - {len(alerts)} 个问题"

            # 邮件内容
            body = "RQA2025 系统检测到以下问题:\n\n"
            for i, alert in enumerate(alerts, 1):
                body += f"{i}. {alert.get('message', '未指定问题')}\n"
            body += f"\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            body += "\n\n请及时处理。"

            msg.attach(MIMEText(body, 'plain'))

            # 发送邮件
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["sender_email"], email_config["password"])
            text = msg.as_string()
            server.sendmail(email_config["sender_email"], email_config["receiver_emails"], text)
            server.quit()

            self.logger.info(f"邮件告警已发送至 {len(email_config['receiver_emails'])} 个收件人")

        except Exception as e:
            self.logger.error(f"发送邮件告警失败: {e}")

    def _send_slack_alerts(self, alerts: List[Dict[str, Any]]):
        """发送Slack告警"""
        try:
            slack_config = self.config["alerts"]["slack"]

            # 构建消息
            message = {
                "channel": slack_config["channel"],
                "text": f":warning: RQA2025 系统告警 - 检测到 {len(alerts)} 个问题",
                "attachments": []
            }

            for alert in alerts:
                attachment = {
                    "color": "danger" if alert.get("level") == "critical" else "warning",
                    "fields": [
                        {
                            "title": "问题类型",
                            "value": alert.get("type", "unknown"),
                            "short": True
                        },
                        {
                            "title": "严重程度",
                            "value": alert.get("level", "unknown"),
                            "short": True
                        },
                        {
                            "title": "详细信息",
                            "value": alert.get("message", "未指定"),
                            "short": False
                        }
                    ]
                }
                message["attachments"].append(attachment)

            # 发送到Slack
            response = requests.post(
                slack_config["webhook_url"],
                json=message,
                timeout=10
            )

            if response.status_code == 200:
                self.logger.info("Slack告警已发送")
            else:
                self.logger.error(f"Slack告警发送失败: HTTP {response.status_code}")

        except Exception as e:
            self.logger.error(f"发送Slack告警失败: {e}")

    def generate_report(self) -> str:
        """生成监控报告"""
        report = [f"# RQA2025 系统监控报告\n"]
        report.append(f"**报告时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**监控周期**: {self.config['monitoring']['interval']}秒")
        report.append("")

        # 系统状态
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        if latest_metrics:
            report.append("## 系统状态")
            report.append(".1f"            report.append(".1f"            report.append(".1f"            report.append("")

        # 服务健康状态
        report.append("## 服务健康状态")
        service_health = self.check_service_health()
        for service_name, health_info in service_health.items():
            status_emoji = "✅" if health_info["status"] == "healthy" else "❌"
            report.append(f"- **{service_name}**: {status_emoji} {health_info['status']}")

            if health_info.get("response_time"):
                report.append(".2f"        report.append("")

        # 近期告警
        if self.alerts_history:
            report.append("## 近期告警")
            recent_alerts = self.alerts_history[-10:]  # 最近10个告警
            for alert in recent_alerts:
                level_emoji = "🔴" if alert.get("level") == "critical" else "🟡"
                timestamp = alert.get("timestamp", "未知时间")
                message = alert.get("message", "未指定")
                report.append(f"- {level_emoji} {timestamp}: {message}")
            report.append("")

        return "\n".join(report)

    def save_metrics_to_file(self):
        """保存指标数据到文件"""
        try:
            metrics_dir = Path("metrics")
            metrics_dir.mkdir(exist_ok=True)

            # 保存最新指标
            if self.metrics_history:
                latest_file = metrics_dir / "latest_metrics.json"
                with open(latest_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metrics_history[-1], f, indent=2, ensure_ascii=False)

            # 保存告警历史
            if self.alerts_history:
                alerts_file = metrics_dir / "alerts_history.json"
                with open(alerts_file, 'w', encoding='utf-8') as f:
                    json.dump(self.alerts_history[-100:], f, indent=2, ensure_ascii=False)  # 保存最近100个告警

        except Exception as e:
            self.logger.error(f"保存指标数据失败: {e}")

    def run_monitoring_loop(self):
        """运行监控循环"""
        self.logger.info("启动RQA2025系统监控...")

        try:
            while True:
                # 收集系统指标
                system_metrics = self.collect_system_metrics()
                self.metrics_history.append(system_metrics)

                # 检查服务健康
                service_health = self.check_service_health()

                # 检查告警条件
                alerts = self.check_alert_conditions(system_metrics, service_health)

                # 发送告警
                if alerts:
                    self.send_alerts(alerts)

                # 保存数据
                self.save_metrics_to_file()

                # 等待下一个监控周期
                time.sleep(self.config["monitoring"]["interval"])

        except KeyboardInterrupt:
            self.logger.info("监控已停止")
        except Exception as e:
            self.logger.error(f"监控循环异常: {e}")
            self.send_alerts([{
                "type": "system",
                "level": "critical",
                "message": f"监控系统异常: {str(e)}"
            }])

    def run_once(self) -> Dict[str, Any]:
        """运行一次监控检查"""
        results = {}

        # 收集系统指标
        results["system_metrics"] = self.collect_system_metrics()

        # 检查服务健康
        results["service_health"] = self.check_service_health()

        # 检查告警条件
        results["alerts"] = self.check_alert_conditions(
            results["system_metrics"],
            results["service_health"]
        )

        # 生成报告
        results["report"] = self.generate_report()

        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 系统监控器')
    parser.add_argument('--config', help='监控配置文件')
    parser.add_argument('--once', action='store_true', help='只执行一次检查')
    parser.add_argument('--report', action='store_true', help='生成并显示报告')
    parser.add_argument('--alert-test', action='store_true', help='测试告警功能')

    args = parser.parse_args()

    # 初始化监控器
    monitor = SystemMonitor(args.config)

    if args.alert_test:
        # 测试告警功能
        test_alerts = [
            {
                "type": "system",
                "level": "warning",
                "metric": "cpu_usage",
                "value": 95.5,
                "threshold": 85.0,
                "message": "CPU使用率过高: 95.5% > 85.0%"
            },
            {
                "type": "service",
                "level": "critical",
                "service": "trading_engine",
                "status": "unhealthy",
                "error": "连接超时",
                "message": "交易引擎服务异常: 连接超时"
            }
        ]
        print("🔔 发送测试告警...")
        monitor.send_alerts(test_alerts)
        print("✅ 测试告警已发送")
        return

    if args.once or args.report:
        # 执行一次检查并显示报告
        print("🔍 执行系统检查...")
        results = monitor.run_once()

        if args.report:
            print("\n" + "="*60)
            print(results["report"])
            print("="*60)

        # 显示告警
        if results["alerts"]:
            print(f"\n⚠️ 发现 {len(results['alerts'])} 个告警:")
            for alert in results["alerts"]:
                level_emoji = "🔴" if alert.get("level") == "critical" else "🟡"
                print(f"  {level_emoji} {alert.get('message', '未指定')}")
        else:
            print("\n✅ 所有检查正常")

        return

    # 默认启动监控循环
    monitor.run_monitoring_loop()


if __name__ == "__main__":
    main()
