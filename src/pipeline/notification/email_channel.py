"""
邮件通知通道模块

实现通过SMTP发送邮件通知的功能
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

try:
    from .notification_service import NotificationChannel, NotificationLevel, NotificationResult
except ImportError:
    from notification_service import NotificationChannel, NotificationLevel, NotificationResult


class EmailNotificationChannel(NotificationChannel):
    """
    邮件通知通道

    通过SMTP协议发送邮件通知

    Attributes:
        smtp_host: SMTP服务器地址
        smtp_port: SMTP服务器端口
        username: 登录用户名
        password: 登录密码
        sender: 发件人地址
        recipients: 收件人地址列表
        use_tls: 是否使用TLS加密
    """

    # 通知级别到邮件主题的映射
    LEVEL_SUBJECT_PREFIX = {
        NotificationLevel.DEBUG: "[调试]",
        NotificationLevel.INFO: "[信息]",
        NotificationLevel.WARNING: "[警告]",
        NotificationLevel.ERROR: "[错误]",
        NotificationLevel.CRITICAL: "[严重]",
    }

    def __init__(
        self,
        name: str = "email",
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        sender: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        use_tls: bool = True,
        enabled: bool = True,
        timeout: int = 30
    ):
        """
        初始化邮件通知通道

        Args:
            name: 通道名称
            smtp_host: SMTP服务器地址
            smtp_port: SMTP服务器端口
            username: 登录用户名
            password: 登录密码
            sender: 发件人地址
            recipients: 收件人地址列表
            use_tls: 是否使用TLS加密
            enabled: 是否启用
            timeout: 连接超时时间（秒）
        """
        super().__init__(name, enabled)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender or username
        self.recipients = recipients or []
        self.use_tls = use_tls
        self.timeout = timeout

    def send(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        **kwargs: Any
    ) -> NotificationResult:
        """
        发送邮件通知

        Args:
            message: 邮件内容
            level: 通知级别
            **kwargs: 额外参数
                - subject: 邮件主题
                - recipients: 覆盖默认收件人列表
                - html: 是否使用HTML格式
                - attachments: 附件列表

        Returns:
            发送结果
        """
        # 验证消息
        if not self.validate_message(message):
            return NotificationResult(
                channel_name=self.name,
                success=False,
                error="消息验证失败"
            )

        # 检查配置
        if not self._validate_config():
            return NotificationResult(
                channel_name=self.name,
                success=False,
                error="邮件配置不完整"
            )

        # 获取收件人
        recipients = kwargs.get("recipients", self.recipients)
        if not recipients:
            return NotificationResult(
                channel_name=self.name,
                success=False,
                error="收件人列表为空"
            )

        try:
            # 构建邮件
            msg = self._build_message(message, level, recipients, **kwargs)

            # 发送邮件
            message_id = self._send_email(msg, recipients)

            self.logger.info(f"邮件发送成功: {message_id}")
            return NotificationResult(
                channel_name=self.name,
                success=True,
                message_id=message_id,
                response_data={
                    "recipients": recipients,
                    "subject": msg["Subject"],
                    "timestamp": datetime.now().isoformat()
                }
            )

        except smtplib.SMTPException as e:
            error_msg = f"SMTP错误: {e}"
            self.logger.error(error_msg)
            return NotificationResult(
                channel_name=self.name,
                success=False,
                error=error_msg
            )
        except Exception as e:
            error_msg = f"发送邮件异常: {e}"
            self.logger.error(error_msg)
            return NotificationResult(
                channel_name=self.name,
                success=False,
                error=error_msg
            )

    def _validate_config(self) -> bool:
        """
        验证邮件配置

        Returns:
            配置是否有效
        """
        if not self.smtp_host or not self.smtp_port:
            self.logger.error("SMTP服务器地址和端口必须配置")
            return False
        if not self.sender:
            self.logger.error("发件人地址必须配置")
            return False
        return True

    def _build_message(
        self,
        message: str,
        level: NotificationLevel,
        recipients: List[str],
        **kwargs: Any
    ) -> MIMEMultipart:
        """
        构建邮件消息

        Args:
            message: 邮件内容
            level: 通知级别
            recipients: 收件人列表
            **kwargs: 额外参数

        Returns:
            邮件消息对象
        """
        msg = MIMEMultipart("alternative")

        # 设置邮件头
        msg["From"] = self.sender
        msg["To"] = ", ".join(recipients)
        msg["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
        msg["Message-ID"] = f"<{uuid.uuid4()}@{self.smtp_host}>"

        # 设置主题
        custom_subject = kwargs.get("subject", "")
        prefix = self.LEVEL_SUBJECT_PREFIX.get(level, "")
        msg["Subject"] = f"{prefix} {custom_subject}".strip()

        # 构建内容
        is_html = kwargs.get("html", False)

        if is_html:
            # HTML格式
            html_content = f"""
            <html>
                <body>
                    <div style="padding: 20px; font-family: Arial, sans-serif;">
                        <h2 style="color: {self._get_level_color(level)};">
                            {prefix} 系统通知
                        </h2>
                        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
                            {message}
                        </div>
                        <p style="color: #666; font-size: 12px; margin-top: 20px;">
                            发送时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        </p>
                    </div>
                </body>
            </html>
            """
            msg.attach(MIMEText(message, "plain", "utf-8"))
            msg.attach(MIMEText(html_content, "html", "utf-8"))
        else:
            # 纯文本格式
            text_content = f"""{prefix} 系统通知

{message}

---
发送时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
            msg.attach(MIMEText(text_content, "plain", "utf-8"))

        return msg

    def _send_email(self, msg: MIMEMultipart, recipients: List[str]) -> str:
        """
        发送邮件

        Args:
            msg: 邮件消息对象
            recipients: 收件人列表

        Returns:
            消息ID
        """
        context = ssl.create_default_context() if self.use_tls else None

        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=self.timeout) as server:
            if self.use_tls:
                server.starttls(context=context)

            if self.username and self.password:
                server.login(self.username, self.password)

            server.sendmail(
                self.sender,
                recipients,
                msg.as_string()
            )

        return msg["Message-ID"]

    def _get_level_color(self, level: NotificationLevel) -> str:
        """
        获取通知级别对应的颜色

        Args:
            level: 通知级别

        Returns:
            颜色代码
        """
        colors = {
            NotificationLevel.DEBUG: "#6c757d",
            NotificationLevel.INFO: "#17a2b8",
            NotificationLevel.WARNING: "#ffc107",
            NotificationLevel.ERROR: "#dc3545",
            NotificationLevel.CRITICAL: "#721c24",
        }
        return colors.get(level, "#333333")

    def test_connection(self) -> bool:
        """
        测试SMTP连接

        Returns:
            连接是否成功
        """
        try:
            context = ssl.create_default_context() if self.use_tls else None

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=self.timeout) as server:
                server.ehlo()
                if self.use_tls:
                    server.starttls(context=context)
                    server.ehlo()

                if self.username and self.password:
                    server.login(self.username, self.password)

            self.logger.info("SMTP连接测试成功")
            return True

        except Exception as e:
            self.logger.error(f"SMTP连接测试失败: {e}")
            return False

    def add_recipient(self, email: str) -> None:
        """
        添加收件人

        Args:
            email: 邮箱地址
        """
        if email not in self.recipients:
            self.recipients.append(email)
            self.logger.debug(f"添加收件人: {email}")

    def remove_recipient(self, email: str) -> bool:
        """
        移除收件人

        Args:
            email: 邮箱地址

        Returns:
            是否成功移除
        """
        if email in self.recipients:
            self.recipients.remove(email)
            self.logger.debug(f"移除收件人: {email}")
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            配置字典
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "username": self.username,
            "sender": self.sender,
            "recipients": self.recipients,
            "use_tls": self.use_tls,
            "timeout": self.timeout
        }
