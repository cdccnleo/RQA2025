#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 合规报告测试 - 重构后版本
"""

import unittest
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.infrastructure.compliance import IReportGenerator
from src.infrastructure.compliance.regulatory_reporter import RegulatoryReporter
from src.infrastructure.notification import MockNotificationService

class MockReportGenerator(IReportGenerator):
    """测试用mock报告生成器"""

    def __init__(self):
        self.trades = []
        self.violations = []

    def generate_daily_report(self) -> str:
        return "/tmp/mock_report.pdf"

    def get_today_trades(self) -> List[Dict]:
        return [
            {
                "trade_id": "TEST001",
                "symbol": "600519.SH",
                "price": 1800.50,
                "quantity": 100,
                "timestamp": datetime.now(),
                "account": "TEST_ACCT",
                "broker": "TEST_BROKER",
                "is_buy": True
            }
        ]

    def run_compliance_checks(self, trades: List[Dict]) -> List[Dict]:
        return [
            {
                "rule_id": "TEST_RULE",
                "description": "Test violation",
                "severity": "minor",
                "trade_ids": ["TEST001"],
                "timestamp": datetime.now()
            }
        ]

class TestRegulatoryReporter(unittest.TestCase):
    """合规报告生成器测试"""

    def setUp(self):
        """测试初始化"""
        self.mock_generator = MockReportGenerator()
        self.mock_notification = MockNotificationService()
        self.reporter = RegulatoryReporter(
            report_generator=self.mock_generator,
            notification_service=self.mock_notification
        )

    def test_generate_report(self):
        """测试报告生成"""
        result = self.reporter.generate_and_send_report()
        self.assertTrue(result)
        self.assertIsNotNone(self.reporter.last_report_time)
        self.mock_notification.send.assert_called_once()

    def test_get_trades(self):
        """测试获取交易记录"""
        trades = self.reporter.get_today_trades()
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["symbol"], "600519.SH")

    def test_run_checks(self):
        """测试合规检查"""
        trades = self.mock_generator.get_today_trades()
        violations = self.reporter.run_compliance_checks(trades)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["rule_id"], "TEST_RULE")

    def test_report_status(self):
        """测试报告状态"""
        status = self.reporter.get_last_report_status()
        self.assertEqual(status["status"], "PENDING")

        self.reporter.generate_and_send_report()
        status = self.reporter.get_last_report_status()
        self.assertEqual(status["status"], "SUCCESS")

if __name__ == '__main__':
    unittest.main()
