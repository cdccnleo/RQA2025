#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件告警系统
实现监控系统的邮件告警功能
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import time
from datetime import datetime
from pathlib import Path
import sqlite3
import logging


class EmailAlertSystem:
    """邮件告警系统"""

    def __init__(self, config_path="config/email_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.alert_history = []
        self.logger = self._setup_logging()

    def _load_config(self):
        """加载邮件配置"""
        default_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your-email@gmail.com",
            "password": "your-app-password",
            "from_email": "your-email@gmail.com",
            "to_emails": ["admin@example.com"],
            "alert_levels": {
                "critical": {"threshold": 90, "cooldown": 300},
                "warning": {"threshold": 70, "cooldown": 600},
                "info": {"threshold": 50, "cooldown": 1800}
            }
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return default_config
        else:
            # 创建默认配置文件
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            return default_config

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/email_alerts.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def send_alert(self, level, subject, message, metrics=None):
        """发送告警邮件"""
        try:
            # 检查告警级别和冷却时间
            if not self._should_send_alert(level):
                self.logger.info(f"告警 {level} 在冷却时间内，跳过发送")
                return False

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.config['from_email']
            msg['To'] = ', '.join(self.config['to_emails'])
            msg['Subject'] = f"[RQA2025] {level.upper()} - {subject}"

            # 生成邮件内容
            html_content = self._generate_email_template(level, subject, message, metrics)
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))

            # 发送邮件
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['username'], self.config['password'])
                server.send_message(msg)

            # 记录告警历史
            self._record_alert(level, subject, message, metrics)
            self.logger.info(f"告警邮件发送成功: {level} - {subject}")
            return True

        except Exception as e:
            self.logger.error(f"发送告警邮件失败: {e}")
            return False

    def _should_send_alert(self, level):
        """检查是否应该发送告警"""
        if level not in self.config['alert_levels']:
            return False

        cooldown = self.config['alert_levels'][level]['cooldown']
        current_time = time.time()

        # 检查最近是否有相同级别的告警
        for alert in self.alert_history[-10:]:  # 检查最近10条
            if alert['level'] == level:
                time_diff = current_time - alert['timestamp']
                if time_diff < cooldown:
                    return False

        return True

    def _generate_email_template(self, level, subject, message, metrics):
        """生成邮件模板"""
        level_colors = {
            'critical': '#e74c3c',
            'warning': '#f39c12',
            'info': '#3498db'
        }

        color = level_colors.get(level, '#7f8c8d')

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .header {{ background: {color}; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                .metrics {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid {color}; }}
                .footer {{ color: #7f8c8d; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚨 RQA2025 监控告警</h1>
                <p>级别: {level.upper()} | 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                <h2>{subject}</h2>
                <p>{message}</p>
        """

        if metrics:
            html += """
                <h3>📊 监控指标</h3>
                <div class="metrics">
            """
            for key, value in metrics.items():
                html += f"<p><strong>{key}:</strong> {value}</p>"
            html += "</div>"

        html += f"""
            </div>
            
            <div class="footer">
                <p>此邮件由 RQA2025 监控系统自动发送</p>
                <p>如需修改告警设置，请联系系统管理员</p>
            </div>
        </body>
        </html>
        """

        return html

    def _record_alert(self, level, subject, message, metrics):
        """记录告警历史"""
        alert = {
            'timestamp': time.time(),
            'level': level,
            'subject': subject,
            'message': message,
            'metrics': metrics or {}
        }
        self.alert_history.append(alert)

        # 保存到数据库
        self._save_alert_to_db(alert)

    def _save_alert_to_db(self, alert):
        """保存告警到数据库"""
        try:
            db_path = Path("data/monitoring.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 创建告警历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    level TEXT,
                    subject TEXT,
                    message TEXT,
                    metrics TEXT
                )
            ''')

            # 插入告警记录
            cursor.execute('''
                INSERT INTO email_alerts (timestamp, level, subject, message, metrics)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                alert['timestamp'],
                alert['level'],
                alert['subject'],
                alert['message'],
                json.dumps(alert['metrics'])
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"保存告警到数据库失败: {e}")

    def get_alert_history(self, limit=50):
        """获取告警历史"""
        try:
            db_path = Path("data/monitoring.db")
            if not db_path.exists():
                return []

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT timestamp, level, subject, message, metrics
                FROM email_alerts
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'timestamp': row[0],
                    'level': row[1],
                    'subject': row[2],
                    'message': row[3],
                    'metrics': json.loads(row[4]) if row[4] else {}
                })

            conn.close()
            return alerts

        except Exception as e:
            self.logger.error(f"获取告警历史失败: {e}")
            return []

    def test_connection(self):
        """测试邮件连接"""
        try:
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['username'], self.config['password'])
                self.logger.info("邮件服务器连接测试成功")
                return True
        except Exception as e:
            self.logger.error(f"邮件服务器连接测试失败: {e}")
            return False

    def update_config(self, new_config):
        """更新配置"""
        self.config.update(new_config)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        self.logger.info("邮件配置已更新")


def main():
    """主函数"""
    print("📧 RQA2025 邮件告警系统")
    print("="*50)

    # 创建邮件告警系统
    email_system = EmailAlertSystem()

    # 测试连接
    print("🔍 测试邮件服务器连接...")
    if email_system.test_connection():
        print("✅ 邮件服务器连接成功")
    else:
        print("❌ 邮件服务器连接失败，请检查配置")
        return

    # 发送测试告警
    print("\n📤 发送测试告警...")
    test_metrics = {
        "CPU使用率": "85%",
        "内存使用率": "78%",
        "磁盘使用率": "65%",
        "响应时间": "120ms"
    }

    success = email_system.send_alert(
        level="warning",
        subject="系统性能告警",
        message="检测到系统性能指标异常，请及时处理。",
        metrics=test_metrics
    )

    if success:
        print("✅ 测试告警发送成功")
    else:
        print("❌ 测试告警发送失败")

    # 显示告警历史
    print("\n📋 告警历史:")
    alerts = email_system.get_alert_history(5)
    for alert in alerts:
        time_str = datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {time_str} [{alert['level'].upper()}] {alert['subject']}")


if __name__ == "__main__":
    main()


class EmailAlertSystem:
    def __init__(self):
        print('邮件告警系统初始化完成')

    def send_alert(self, level, subject, message):
        print(f'发送{level}级别告警: {subject}')
        return True


if __name__ == '__main__':
    print('📧 邮件告警系统测试')
    system = EmailAlertSystem()
    system.send_alert('warning', '系统测试', '这是一条测试告警')
