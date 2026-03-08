"""
Webhook通知通道模块

实现HTTP回调通知功能，支持POST/PUT请求
"""

import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List
from datetime import datetime
import ssl
import hmac
import hashlib
import base64

try:
    from .notification_service import NotificationChannel, NotificationLevel, NotificationResult
except ImportError:
    from notification_service import NotificationChannel, NotificationLevel, NotificationResult


class WebhookNotificationChannel(NotificationChannel):
    """
    Webhook通知通道

    通过HTTP请求发送通知到指定的Webhook URL

    Attributes:
        webhook_url: Webhook地址
        method: HTTP方法(POST/PUT)
        headers: 自定义请求头
        timeout: 请求超时时间
        verify_ssl: 是否验证SSL证书
        secret: 用于签名验证的密钥
        retry_count: 重试次数
    """

    def __init__(
        self,
        name: str = "webhook",
        webhook_url: Optional[str] = None,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        secret: Optional[str] = None,
        retry_count: int = 3,
        enabled: bool = True
    ):
        """
        初始化Webhook通知通道

        Args:
            name: 通道名称
            webhook_url: Webhook地址
            method: HTTP方法
            headers: 自定义请求头
            timeout: 请求超时时间（秒）
            verify_ssl: 是否验证SSL证书
            secret: 用于签名验证的密钥
            retry_count: 重试次数
            enabled: 是否启用
        """
        super().__init__(name, enabled)
        self.webhook_url = webhook_url
        self.method = method.upper()
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.secret = secret
        self.retry_count = retry_count

    def send(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        **kwargs: Any
    ) -> NotificationResult:
        """
        发送Webhook通知

        Args:
            message: 通知内容
            level: 通知级别
            **kwargs: 额外参数
                - webhook_url: 覆盖默认Webhook地址
                - payload: 自定义请求体
                - headers: 覆盖默认请求头
                - metadata: 附加元数据

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

        # 获取Webhook地址
        webhook_url = kwargs.get("webhook_url", self.webhook_url)
        if not webhook_url:
            return NotificationResult(
                channel_name=self.name,
                success=False,
                error="Webhook地址未配置"
            )

        # 构建请求
        payload = self._build_payload(message, level, kwargs.get("metadata", {}))
        custom_payload = kwargs.get("payload")
        if custom_payload:
            payload = custom_payload

        headers = dict(self.headers)
        headers.update(kwargs.get("headers", {}))

        # 添加签名
        if self.secret:
            signature = self._generate_signature(payload)
            headers["X-Webhook-Signature"] = signature

        # 发送请求（带重试）
        last_error = None
        for attempt in range(self.retry_count):
            try:
                response_data = self._send_request(webhook_url, payload, headers)

                self.logger.info(f"Webhook发送成功: {webhook_url}")
                return NotificationResult(
                    channel_name=self.name,
                    success=True,
                    response_data={
                        "webhook_url": webhook_url,
                        "status_code": response_data.get("status_code"),
                        "response_body": response_data.get("body"),
                        "attempt": attempt + 1,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            except urllib.error.HTTPError as e:
                last_error = f"HTTP错误 {e.code}: {e.reason}"
                self.logger.warning(f"Webhook请求失败(尝试{attempt+1}/{self.retry_count}): {last_error}")

                # 4xx错误不重试
                if 400 <= e.code < 500:
                    break

            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Webhook请求异常(尝试{attempt+1}/{self.retry_count}): {last_error}")

        error_msg = f"Webhook发送失败(重试{self.retry_count}次): {last_error}"
        self.logger.error(error_msg)
        return NotificationResult(
            channel_name=self.name,
            success=False,
            error=error_msg
        )

    def _build_payload(
        self,
        message: str,
        level: NotificationLevel,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        构建请求体

        Args:
            message: 消息内容
            level: 通知级别
            metadata: 元数据

        Returns:
            请求体字典
        """
        payload = {
            "message": message,
            "level": level.name,
            "timestamp": datetime.now().isoformat(),
            "channel": self.name,
            "metadata": metadata
        }
        return payload

    def _generate_signature(self, payload: Dict[str, Any]) -> str:
        """
        生成请求签名

        Args:
            payload: 请求体

        Returns:
            签名字符串
        """
        if not self.secret:
            return ""

        payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        signature = hmac.new(
            self.secret.encode("utf-8"),
            payload_str.encode("utf-8"),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode("utf-8")

    def _send_request(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        发送HTTP请求

        Args:
            url: 请求地址
            payload: 请求体
            headers: 请求头

        Returns:
            响应数据
        """
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=self.method
        )

        # SSL上下文
        if not self.verify_ssl:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        else:
            ssl_context = None

        response = urllib.request.urlopen(
            request,
            timeout=self.timeout,
            context=ssl_context
        )

        status_code = response.getcode()
        body = response.read().decode("utf-8")

        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            pass

        return {
            "status_code": status_code,
            "body": body
        }

    def test_connection(self) -> bool:
        """
        测试Webhook连接

        Returns:
            连接是否成功
        """
        if not self.webhook_url:
            self.logger.error("Webhook地址未配置")
            return False

        try:
            test_payload = {"test": True, "timestamp": datetime.now().isoformat()}
            self._send_request(self.webhook_url, test_payload, self.headers)
            self.logger.info("Webhook连接测试成功")
            return True
        except Exception as e:
            self.logger.error(f"Webhook连接测试失败: {e}")
            return False

    def set_webhook_url(self, url: str) -> None:
        """
        设置Webhook地址

        Args:
            url: Webhook地址
        """
        self.webhook_url = url
        self.logger.debug(f"设置Webhook地址: {url}")

    def add_header(self, key: str, value: str) -> None:
        """
        添加请求头

        Args:
            key: 头名称
            value: 头值
        """
        self.headers[key] = value
        self.logger.debug(f"添加请求头: {key}")

    def remove_header(self, key: str) -> bool:
        """
        移除请求头

        Args:
            key: 头名称

        Returns:
            是否成功移除
        """
        if key in self.headers:
            del self.headers[key]
            self.logger.debug(f"移除请求头: {key}")
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
            "webhook_url": self.webhook_url,
            "method": self.method,
            "headers": self.headers,
            "timeout": self.timeout,
            "verify_ssl": self.verify_ssl,
            "retry_count": self.retry_count
        }
