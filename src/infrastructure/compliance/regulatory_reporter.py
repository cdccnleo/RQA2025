#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 合规报告服务
负责生成和分发合规报告
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from . import IReportGenerator
from ..notification import NotificationService

logger = logging.getLogger(__name__)

class RegulatoryReporter:
    """合规报告服务"""

    def __init__(self,
                 report_generator: IReportGenerator,
                 notification_service: Optional[NotificationService] = None):
        """
        初始化合规报告服务

        Args:
            report_generator: 报告生成器实现
            notification_service: 通知服务(可选)
        """
        self.report_generator = report_generator
        self.notification_service = notification_service or NotificationService()
        self.last_report_time = None

    def generate_and_send_report(self) -> bool:
        """生成并发送合规报告"""
        try:
            # 生成报告
            report_path = self.report_generator.generate_daily_report()
            self.last_report_time = datetime.now()

            # 发送通知
            if self.notification_service:
                self.notification_service.send(
                    subject="RQA2025合规报告",
                    message=f"合规报告已生成: {report_path}",
                    recipients=["compliance@rqa2025.com"]
                )

            logger.info(f"合规报告生成并发送成功: {report_path}")
            return True

        except Exception as e:
            logger.error(f"生成合规报告失败: {str(e)}")
            return False

    def get_last_report_status(self) -> Dict:
        """获取上次报告状态"""
        return {
            "last_report_time": self.last_report_time,
            "status": "SUCCESS" if self.last_report_time else "PENDING"
        }

    def get_today_trades(self) -> List[Dict]:
        """获取当日交易记录(代理方法)"""
        return self.report_generator.get_today_trades()

    def run_compliance_checks(self, trades: List[Dict]) -> List[Dict]:
        """执行合规检查(代理方法)"""
        return self.report_generator.run_compliance_checks(trades)
