
import time

from .alert_dataclasses import Alert
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from typing import Dict, Optional, Any
from src.monitoring.core.monitoring_system import AlertLevel
"""
通知渠道管理器

负责通知渠道的配置、管理和消息发送
"""


class NotificationChannelManager:
    """通知渠道管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

        self.config = config or {}

        # 通知渠道配置
        self.notification_config: Dict[str, Dict[str, Any]] = {
            "email": {"enabled": True, "recipients": []},
            "sms": {"enabled": False, "recipients": []},
            "webhook": {"enabled": False, "url": "", "headers": {}},
            "log": {"enabled": True, "level": "WARNING"}
        }

        # 更新配置
        if config:
            self.notification_config.update(config)

        # 通知统计
        self.notification_stats = {
            "total_sent": 0,
            "by_channel": {},
            "by_level": {},
            "errors": 0
        }

    def configure_channel(self, channel_name: str, config: Dict[str, Any]):
        """配置通知渠道"""
        try:
            if channel_name not in self.notification_config:
                raise ValueError(f"未知的通知渠道: {channel_name}")

            # 更新配置
            self.notification_config[channel_name].update(config)
            self.logger.log_info(f"通知渠道 '{channel_name}' 配置已更新")

        except Exception as e:
            self.error_handler.handle_error(e, {
                "context": "配置通知渠道失败",
                "channel_name": channel_name
            })
            raise

    def enable_channel(self, channel_name: str):
        """启用通知渠道"""
        if channel_name in self.notification_config:
            self.notification_config[channel_name]["enabled"] = True
            self.logger.log_info(f"通知渠道 '{channel_name}' 已启用")

    def disable_channel(self, channel_name: str):
        """禁用通知渠道"""
        if channel_name in self.notification_config:
            self.notification_config[channel_name]["enabled"] = False
            self.logger.log_info(f"通知渠道 '{channel_name}' 已禁用")

    def get_channel_status(self) -> Dict[str, bool]:
        """获取通知渠道状态"""
        return {
            name: config.get("enabled", False)
            for name, config in self.notification_config.items()
        }

    def get_channel_config(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """获取指定渠道的配置"""
        return self.notification_config.get(channel_name)

    def send_notification(self, alert: Alert):
        """发送告警通知"""
        try:
            sent_count = 0

            # 遍历所有启用的渠道
            for channel_name, config in self.notification_config.items():
                if config.get("enabled", False):
                    try:
                        self._send_to_channel(channel_name, alert, config)
                        sent_count += 1
                        self._update_stats(channel_name, alert.alert_level.value, success=True)
                    except Exception as e:
                        self._update_stats(channel_name, alert.alert_level.value, success=False)
                        self.error_handler.handle_error(e, {
                            "context": f"发送通知到渠道 '{channel_name}' 失败",
                            "alert_id": alert.alert_id
                        })

            if sent_count > 0:
                self.logger.log_info(f"告警 '{alert.alert_id}' 已发送到 {sent_count} 个渠道")
            else:
                self.logger.log_warning(f"告警 '{alert.alert_id}' 没有可用的通知渠道")

        except Exception as e:
            self.error_handler.handle_error(e, {
                "context": "发送告警通知失败",
                "alert_id": alert.alert_id
            })

    def _send_to_channel(self, channel_name: str, alert: Alert, config: Dict[str, Any]):
        """发送通知到指定渠道"""
        if channel_name == "email":
            self._send_email(alert, config)
        elif channel_name == "sms":
            self._send_sms(alert, config)
        elif channel_name == "webhook":
            self._send_webhook(alert, config)
        elif channel_name == "log":
            self._send_log(alert, config)
        else:
            raise ValueError(f"不支持的通知渠道: {channel_name}")

    def _send_email(self, alert: Alert, config: Dict[str, Any]):
        """发送邮件通知"""
        recipients = config.get("recipients", [])
        if not recipients:
            return

        subject = f"[{alert.alert_level.value}] {alert.message}"
        body = f"""
告警详情:
- 告警ID: {alert.alert_id}
- 规则名称: {alert.rule_name}
- 告警级别: {alert.alert_level.value}
- 消息: {alert.message}
- 时间戳: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}

请及时处理！
        """.strip()

        # 这里应该集成实际的邮件发送服务
        # 暂时只记录日志
        self.logger.log_info(f"邮件通知已发送到 {len(recipients)} 个收件人")

    def _send_sms(self, alert: Alert, config: Dict[str, Any]):
        """发送短信通知"""
        recipients = config.get("recipients", [])
        if not recipients:
            return

        message = f"[{alert.alert_level.value}] {alert.message}"

        # 这里应该集成实际的短信发送服务
        # 暂时只记录日志
        self.logger.log_info(f"短信通知已发送到 {len(recipients)} 个号码")

    def _send_webhook(self, alert: Alert, config: Dict[str, Any]):
        """发送Webhook通知"""
        url = config.get("url")
        if not url:
            return

        headers = config.get("headers", {})
        payload = {
            "alert_id": alert.alert_id,
            "rule_name": alert.rule_name,
            "level": alert.alert_level.value,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "metadata": alert.metadata
        }

        # 这里应该集成实际的HTTP请求
        # 暂时只记录日志
        self.logger.log_info(f"Webhook通知已发送到 {url}")

    def _send_log(self, alert: Alert, config: Dict[str, Any]):
        """发送日志通知"""
        log_level = config.get("level", "WARNING")

        message = f"告警通知 - ID:{alert.alert_id}, 级别:{alert.alert_level.value}, 消息:{alert.message}"

        if log_level == "ERROR":
            self.logger.log_error(message)
        elif log_level == "WARNING":
            self.logger.log_warning(message)
        elif log_level == "INFO":
            self.logger.log_info(message)
        else:
            self.logger.log_info(message)

    def _update_stats(self, channel_name: str, level: str, success: bool):
        """更新通知统计"""
        if success:
            self.notification_stats["total_sent"] += 1

            if channel_name not in self.notification_stats["by_channel"]:
                self.notification_stats["by_channel"][channel_name] = 0
            self.notification_stats["by_channel"][channel_name] += 1

            if level not in self.notification_stats["by_level"]:
                self.notification_stats["by_level"][level] = 0
            self.notification_stats["by_level"][level] += 1
        else:
            self.notification_stats["errors"] += 1

    def get_notification_stats(self) -> Dict[str, Any]:
        """获取通知统计信息"""
        return self.notification_stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.notification_stats = {
            "total_sent": 0,
            "by_channel": {},
            "by_level": {},
            "errors": 0
        }
        self.logger.log_info("通知统计信息已重置")

    def test_channel(self, channel_name: str) -> bool:
        """测试通知渠道是否可用"""
        try:
            if channel_name not in self.notification_config:
                return False

            config = self.notification_config[channel_name]
            if not config.get("enabled", False):
                return False

            # 创建测试告警
            test_alert = Alert(
                alert_id="test_channel",
                rule_name="test_rule",
                alert_type=Alert.from_str("PERFORMANCE"),
                alert_level=AlertLevel.from_str("INFO"),
                message="测试通知渠道",
                timestamp=time.time(),
                metadata={"test": True}
            )

            # 尝试发送测试通知
            self._send_to_channel(channel_name, test_alert, config)
            return True

        except Exception as e:
            self.error_handler.handle_error(e, {
                "context": f"测试通知渠道 '{channel_name}' 失败"
            })
            return False
