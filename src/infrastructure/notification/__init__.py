#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 通知服务模块
提供报告通知功能的基础设施服务
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class NotificationService:
    """基础通知服务实现"""

    def __init__(self, smtp_server: str = "localhost", smtp_port: int = 25):
        """
        初始化通知服务

        Args:
            smtp_server: SMTP服务器地址
            smtp_port: SMTP服务器端口
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self._setup_logger()

    def send(self,
             subject: str,
             message: str,
             recipients: List[str]) -> bool:
        """
        发送通知

        Args:
            subject: 通知主题
            message: 通知内容
            recipients: 收件人列表

        Returns:
            bool: 是否发送成功
        """
        try:
            logger.info(f"发送通知到 {len(recipients)} 个收件人")
            logger.debug(f"主题: {subject}")
            logger.debug(f"内容: {message[:50]}...")

            # 实际发送逻辑应在此实现
            # 这里仅模拟发送成功
            return True

        except Exception as e:
            logger.error(f"发送通知失败: {str(e)}")
            return False

    def _setup_logger(self):
        """配置日志记录器"""
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

class MockNotificationService(NotificationService):
    """测试用模拟通知服务"""

    def __init__(self):
        super().__init__()
        self.sent_messages = []

    def send(self, subject: str, message: str, recipients: List[str]) -> bool:
        """模拟发送，仅记录不实际发送"""
        self.sent_messages.append({
            'subject': subject,
            'message': message,
            'recipients': recipients
        })
        return True
