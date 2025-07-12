"""科创板规则检查单元测试"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, time
from src.trading.risk.china.star_market import STARMarketRuleChecker
from src.trading.execution.order import OrderStatus

class TestSTARMarketRuleChecker(unittest.TestCase):
    def setUp(self):
        self.metrics = MagicMock()
        self.checker = STARMarketRuleChecker(metrics_collector=self.metrics)

    def test_price_limit_check(self):
        """测试科创板20%涨跌停限制检查"""
        # 创建测试订单
        order = MagicMock()
        order.symbol = '688001'
        order.is_buy = True
        order.reference_price = 100.0

        # 测试买入价格超过涨停价
        order.price = 121.0
        passed, reason = self.checker._check_price_limit(order)
        self.assertFalse(passed)
        self.assertIn("超过涨停价120.00", reason)
        self.metrics.record_violation.assert_called_with('price_limit', 0.21)

        # 测试卖出价格低于跌停价
        order.is_buy = False
        order.price = 79.0
        passed, reason = self.checker._check_price_limit(order)
        self.assertFalse(passed)
        self.assertIn("低于跌停价80.00", reason)
        self.metrics.record_violation.assert_called_with('price_limit', 0.21)

        # 测试正常价格
        order.price = 110.0
        passed, reason = self.checker._check_price_limit(order)
        self.assertTrue(passed)
        self.metrics.record_metric.assert_called_with('price_change', 0.1)

        # 测试非科创板股票不检查
        order.symbol = '600000'
        passed, reason = self.checker._check_price_limit(order)
        self.assertTrue(passed)

    def test_after_hours_trading_check(self):
        """测试盘后固定价格交易规则检查"""
        # 创建测试订单
        order = MagicMock()
        order.symbol = '688001'
        order.is_after_hours = True
        order.order_type = OrderStatus.LIMIT
        order.closing_price = 100.0
        order.price = 100.0
        order.quantity = 200
        order.timestamp = datetime(2024, 4, 18, 15, 4, 0)  # 15:04

        # 测试非限价单
        order.order_type = OrderStatus.MARKET
        passed, reason = self.checker._check_after_hours_trading(order)
        self.assertFalse(passed)
        self.assertIn("必须使用限价单", reason)

        # 测试时间过早
        order.order_type = OrderStatus.LIMIT
        order.timestamp = datetime(2024, 4, 18, 14, 59, 0)  # 14:59
        passed, reason = self.checker._check_after_hours_trading(order)
        self.assertFalse(passed)
        self.assertIn("盘后交易时间应为15:00", reason)

        # 测试时间过晚
        order.timestamp = datetime(2024, 4, 18, 15, 6, 0)  # 15:06
        passed, reason = self.checker._check_after_hours_trading(order)
        self.assertFalse(passed)
        self.assertIn("超过盘后交易截止时间", reason)

        # 测试价格误差范围
        order.timestamp = datetime(2024, 4, 18, 15, 4, 0)
        order.price = 100.02  # 超过1分钱误差
        passed, reason = self.checker._check_after_hours_trading(order)
        self.assertFalse(passed)
        self.assertIn("必须等于收盘价100.00", reason)

        # 测试最小交易单位
        order.price = 100.0
        order.quantity = 199  # 不足200股
        passed, reason = self.checker._check_after_hours_trading(order)
        self.assertFalse(passed)
        self.assertIn("应为200股的整数倍", reason)

        # 测试正常盘后交易
        order.quantity = 200
        passed, reason = self.checker._check_after_hours_trading(order)
        self.assertTrue(passed)
        self.metrics.record_check.assert_called_with('after_hours', True)

    def test_check_star_market_rules(self):
        """测试综合规则检查"""
        # 创建测试订单
        order = MagicMock()
        order.symbol = '688001'
        order.is_after_hours = False
        order.is_buy = True
        order.reference_price = 100.0
        order.price = 110.0

        # 测试正常情况
        passed, reason = self.checker.check_star_market_rules(order)
        self.assertTrue(passed)

        # 测试涨跌停违规
        order.price = 121.0
        passed, reason = self.checker.check_star_market_rules(order)
        self.assertFalse(passed)
        self.metrics.record_check_failure.assert_called_with('price_limit')

        # 测试盘后交易违规
        order.is_after_hours = True
        order.price = 110.0
        order.order_type = OrderStatus.MARKET
        passed, reason = self.checker.check_star_market_rules(order)
        self.assertFalse(passed)
        self.metrics.record_check_failure.assert_called_with('after_hours')

        # 验证性能指标记录
        self.metrics.record_latency.assert_called()

if __name__ == '__main__':
    unittest.main()
