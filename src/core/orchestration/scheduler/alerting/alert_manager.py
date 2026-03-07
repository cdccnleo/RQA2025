"""
告警管理器

支持多种告警渠道：邮件、Webhook、日志、短信
"""

import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

 
class AlertLevel(Enum):
    """告警级别"""
    INFO = 0         # 信息
    WARNING = 1      # 警告
    ERROR = 2        # 错误
    CRITICAL = 3     # 严重


class AlertChannel(Enum):
    """告警渠道"""
    EMAIL = "email"       # 邮件
    WEBHOOK = "webhook"   # Webhook
    LOG = "log"           # 日志
    SMS = "sms"           # 短信


@dataclass
class AlertMessage:
    """告警消息"""
    title: str
    content: str
    level: AlertLevel
    channel: AlertChannel
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level.value,
            "channel": self.channel.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AlertConfig:
    """告警配置"""
    enabled: bool = True
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    level_threshold: AlertLevel = AlertLevel.WARNING
    rate_limit_seconds: int = 60  # 告警频率限制（秒）

    # 邮件配置
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)

    # Webhook配置
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # 短信配置
    sms_provider: str = ""  # twilio, aliyun, etc.
    sms_api_key: str = ""
    sms_api_secret: str = ""
    sms_to_numbers: List[str] = field(default_factory=list)


class AlertHandler(ABC):
    """告警处理器抽象基类"""

    @abstractmethod
    async def send(self, message: AlertMessage) -> bool:
        """
        发送告警

        Args:
            message: 告警消息

        Returns:
            bool: 发送是否成功
        """
        pass


class EmailAlertHandler(AlertHandler):
    """邮件告警处理器"""

    def __init__(self, config: AlertConfig):
        self._config = config

    async def send(self, message: AlertMessage) -> bool:
        """发送邮件告警"""
        try:
            if not all([self._config.email_username, self._config.email_password,
                       self._config.email_from, self._config.email_to]):
                logger.warning("邮件配置不完整，无法发送邮件告警")
                return False

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self._config.email_from
            msg['To'] = ', '.join(self._config.email_to)
            msg['Subject'] = f"[{message.level.value.upper()}] {message.title}"

            # 邮件内容
            body = self._format_email_body(message)
            msg.attach(MIMEText(body, 'html', 'utf-8'))

            # 发送邮件
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_email_sync, msg
            )

            logger.info(f"邮件告警已发送: {message.title}")
            return True

        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")
            return False

    def _send_email_sync(self, msg: MIMEMultipart):
        """同步发送邮件"""
        with smtplib.SMTP(self._config.email_smtp_host, self._config.email_smtp_port) as server:
            server.starttls()
            server.login(self._config.email_username, self._config.email_password)
            server.send_message(msg)

    # 级别名称映射
    LEVEL_NAMES = {
        AlertLevel.INFO: "INFO",
        AlertLevel.WARNING: "WARNING",
        AlertLevel.ERROR: "ERROR",
        AlertLevel.CRITICAL: "CRITICAL"
    }

    def _format_email_body(self, message: AlertMessage) -> str:
        """格式化邮件内容"""
        level_colors = {
            AlertLevel.INFO: "#3498db",
            AlertLevel.WARNING: "#f39c12",
            AlertLevel.ERROR: "#e74c3c",
            AlertLevel.CRITICAL: "#c0392b"
        }

        color = level_colors.get(message.level, "#333333")
        level_name = self.LEVEL_NAMES.get(message.level, "UNKNOWN")

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="border: 1px solid #ddd; padding: 20px; border-radius: 5px;">
                <h2 style="color: {color}; margin-top: 0;">
                    [{level_name}] {message.title}
                </h2>
                <p style="font-size: 14px; color: #666;">
                    时间: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                </p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 15px 0;">
                <div style="font-size: 14px; line-height: 1.6;">
                    {message.content}
                </div>
                {self._format_metadata(message.metadata)}
            </div>
        </body>
        </html>
        """

    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """格式化元数据"""
        if not metadata:
            return ""

        html = '<hr style="border: none; border-top: 1px solid #eee; margin: 15px 0;">'
        html += '<h3 style="font-size: 14px; color: #333;">附加信息:</h3>'
        html += '<table style="font-size: 12px; color: #666;">'

        for key, value in metadata.items():
            html += f'<tr><td style="padding: 5px 10px 5px 0; font-weight: bold;">{key}:</td>'
            html += f'<td>{value}</td></tr>'

        html += '</table>'
        return html


class WebhookAlertHandler(AlertHandler):
    """Webhook告警处理器"""

    def __init__(self, config: AlertConfig):
        self._config = config

    async def send(self, message: AlertMessage) -> bool:
        """发送Webhook告警"""
        try:
            if not self._config.webhook_url:
                logger.warning("Webhook URL未配置，无法发送Webhook告警")
                return False

            payload = {
                "title": message.title,
                "content": message.content,
                "level": message.level.value,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata
            }

            headers = {
                "Content-Type": "application/json",
                **self._config.webhook_headers
            }

            # 尝试使用aiohttp，如果不存在则使用同步请求
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self._config.webhook_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Webhook告警已发送: {message.title}")
                            return True
                        else:
                            logger.error(f"Webhook告警发送失败: HTTP {response.status}")
                            return False
            except ImportError:
                # 使用同步请求作为回退
                import urllib.request
                import urllib.error

                req = urllib.request.Request(
                    self._config.webhook_url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers=headers,
                    method='POST'
                )

                def send_sync():
                    with urllib.request.urlopen(req, timeout=30) as response:
                        return response.status == 200

                result = await asyncio.get_event_loop().run_in_executor(None, send_sync)
                if result:
                    logger.info(f"Webhook告警已发送: {message.title}")
                    return True
                else:
                    logger.error(f"Webhook告警发送失败")
                    return False

        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")
            return False


class LogAlertHandler(AlertHandler):
    """日志告警处理器"""

    # 级别名称映射
    LEVEL_NAMES = {
        AlertLevel.INFO: "INFO",
        AlertLevel.WARNING: "WARNING",
        AlertLevel.ERROR: "ERROR",
        AlertLevel.CRITICAL: "CRITICAL"
    }

    async def send(self, message: AlertMessage) -> bool:
        """发送日志告警"""
        try:
            level_name = self.LEVEL_NAMES.get(message.level, "UNKNOWN")
            log_message = f"[{level_name}] {message.title}: {message.content}"

            if message.metadata:
                log_message += f" | Metadata: {json.dumps(message.metadata, ensure_ascii=False)}"

            if message.level == AlertLevel.INFO:
                logger.info(log_message)
            elif message.level == AlertLevel.WARNING:
                logger.warning(log_message)
            elif message.level == AlertLevel.ERROR:
                logger.error(log_message)
            elif message.level == AlertLevel.CRITICAL:
                logger.critical(log_message)

            return True

        except Exception as e:
            logger.error(f"发送日志告警失败: {e}")
            return False


class SMSAlertHandler(AlertHandler):
    """短信告警处理器"""

    def __init__(self, config: AlertConfig):
        self._config = config

    async def send(self, message: AlertMessage) -> bool:
        """发送短信告警"""
        try:
            if not all([self._config.sms_provider, self._config.sms_api_key,
                       self._config.sms_to_numbers]):
                logger.warning("短信配置不完整，无法发送短信告警")
                return False

            # 根据提供商发送短信
            if self._config.sms_provider == "twilio":
                return await self._send_twilio(message)
            elif self._config.sms_provider == "aliyun":
                return await self._send_aliyun(message)
            else:
                logger.warning(f"不支持的短信提供商: {self._config.sms_provider}")
                return False

        except Exception as e:
            logger.error(f"发送短信告警失败: {e}")
            return False

    async def _send_twilio(self, message: AlertMessage) -> bool:
        """使用Twilio发送短信"""
        try:
            from twilio.rest import Client

            client = Client(self._config.sms_api_key, self._config.sms_api_secret)

            sms_body = f"[{message.level.value.upper()}] {message.title}: {message.content[:100]}"

            for to_number in self._config.sms_to_numbers:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.messages.create(
                        body=sms_body,
                        from_=self._config.sms_from_number,
                        to=to_number
                    )
                )

            logger.info(f"Twilio短信告警已发送: {message.title}")
            return True

        except Exception as e:
            logger.error(f"发送Twilio短信失败: {e}")
            return False

    async def _send_aliyun(self, message: AlertMessage) -> bool:
        """使用阿里云发送短信"""
        # 阿里云短信实现
        logger.info("阿里云短信功能待实现")
        return False


class AlertManager:
    """告警管理器"""

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        初始化告警管理器

        Args:
            config: 告警配置
        """
        self._config = config or AlertConfig()
        self._handlers: Dict[AlertChannel, AlertHandler] = {}
        self._last_alert_time: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

        # 初始化处理器
        self._init_handlers()

    def _init_handlers(self):
        """初始化告警处理器"""
        if AlertChannel.EMAIL in self._config.channels:
            self._handlers[AlertChannel.EMAIL] = EmailAlertHandler(self._config)

        if AlertChannel.WEBHOOK in self._config.channels:
            self._handlers[AlertChannel.WEBHOOK] = WebhookAlertHandler(self._config)

        if AlertChannel.LOG in self._config.channels:
            self._handlers[AlertChannel.LOG] = LogAlertHandler()

        if AlertChannel.SMS in self._config.channels:
            self._handlers[AlertChannel.SMS] = SMSAlertHandler(self._config)

    def update_config(self, config: AlertConfig):
        """
        更新配置

        Args:
            config: 新配置
        """
        self._config = config
        self._handlers.clear()
        self._init_handlers()

    async def send_alert(self, title: str, content: str, level: AlertLevel = AlertLevel.WARNING,
                        channel: Optional[AlertChannel] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        发送告警

        Args:
            title: 告警标题
            content: 告警内容
            level: 告警级别
            channel: 指定渠道，None则使用所有配置的渠道
            metadata: 附加元数据

        Returns:
            bool: 发送是否成功
        """
        if not self._config.enabled:
            return False

        # 检查告警级别阈值
        if level.value < self._config.level_threshold.value:
            return False

        # 检查频率限制
        alert_key = f"{title}:{level.value}"
        async with self._lock:
            now = datetime.now()
            last_time = self._last_alert_time.get(alert_key)

            if last_time and (now - last_time).total_seconds() < self._config.rate_limit_seconds:
                logger.debug(f"告警频率限制，跳过: {title}")
                return False

            self._last_alert_time[alert_key] = now

        # 创建告警消息
        message = AlertMessage(
            title=title,
            content=content,
            level=level,
            channel=channel or AlertChannel.LOG,
            metadata=metadata or {}
        )

        # 发送告警
        success = False
        channels_to_use = [channel] if channel else self._config.channels

        for ch in channels_to_use:
            handler = self._handlers.get(ch)
            if handler:
                try:
                    if await handler.send(message):
                        success = True
                except Exception as e:
                    logger.error(f"发送告警失败 [{ch.value}]: {e}")

        return success

    # 便捷方法
    async def info(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """发送信息告警"""
        return await self.send_alert(title, content, AlertLevel.INFO, metadata=metadata)

    async def warning(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """发送警告告警"""
        return await self.send_alert(title, content, AlertLevel.WARNING, metadata=metadata)

    async def error(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """发送错误告警"""
        return await self.send_alert(title, content, AlertLevel.ERROR, metadata=metadata)

    async def critical(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """发送严重告警"""
        return await self.send_alert(title, content, AlertLevel.CRITICAL, metadata=metadata)

    # 任务相关告警
    async def task_failed(self, task_id: str, task_type: str, error: str, retry_count: int = 0):
        """任务失败告警"""
        title = f"任务执行失败: {task_type}"
        content = f"任务ID: {task_id}\n错误信息: {error}\n重试次数: {retry_count}"

        level = AlertLevel.WARNING if retry_count > 0 else AlertLevel.ERROR

        return await self.send_alert(
            title=title,
            content=content,
            level=level,
            metadata={"task_id": task_id, "task_type": task_type, "retry_count": retry_count}
        )

    async def task_timeout(self, task_id: str, task_type: str, timeout_seconds: int):
        """任务超时告警"""
        title = f"任务执行超时: {task_type}"
        content = f"任务ID: {task_id}\n超时时间: {timeout_seconds}秒"

        return await self.send_alert(
            title=title,
            content=content,
            level=AlertLevel.WARNING,
            metadata={"task_id": task_id, "task_type": task_type, "timeout": timeout_seconds}
        )

    async def task_retry_exhausted(self, task_id: str, task_type: str, max_retries: int):
        """任务重试次数耗尽告警"""
        title = f"任务重试次数耗尽: {task_type}"
        content = f"任务ID: {task_id}\n最大重试次数: {max_retries}\n任务最终失败，需要人工介入"

        return await self.send_alert(
            title=title,
            content=content,
            level=AlertLevel.CRITICAL,
            metadata={"task_id": task_id, "task_type": task_type, "max_retries": max_retries}
        )

    async def scheduler_error(self, error: str, context: Optional[Dict[str, Any]] = None):
        """调度器错误告警"""
        title = "调度器异常"
        content = f"错误信息: {error}"

        return await self.send_alert(
            title=title,
            content=content,
            level=AlertLevel.ERROR,
            metadata=context or {}
        )

    async def worker_died(self, worker_id: str, task_id: Optional[str] = None):
        """工作进程异常退出告警"""
        title = f"工作进程异常退出: {worker_id}"
        content = f"工作进程ID: {worker_id}"
        if task_id:
            content += f"\n正在执行的任务: {task_id}"

        return await self.send_alert(
            title=title,
            content=content,
            level=AlertLevel.ERROR,
            metadata={"worker_id": worker_id, "task_id": task_id}
        )


# 全局告警管理器实例
_alert_manager_instance: Optional[AlertManager] = None


def get_alert_manager(config: Optional[AlertConfig] = None) -> AlertManager:
    """
    获取告警管理器实例（单例）

    Args:
        config: 告警配置

    Returns:
        AlertManager: 告警管理器实例
    """
    global _alert_manager_instance

    if _alert_manager_instance is None:
        _alert_manager_instance = AlertManager(config)

    return _alert_manager_instance


def reset_alert_manager():
    """重置告警管理器实例（用于测试）"""
    global _alert_manager_instance
    _alert_manager_instance = None
