"""T+1限制检查单元测试"""

import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from src.trading.risk.china.t1_restriction import T1RestrictionChecker

class TestT1Restriction(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.checker = T1RestrictionChecker()
        self.checker.position_service = MagicMock()
        self.checker.metrics = MagicMock()

        # 模拟持仓数据
        self.sample_position = {
            'account_id': 'test_account',
            'symbol': '600000',
            'quantity': 1000,
            'trades': []
        }

    def test_normal_stock_no_violation(self):
        """测试普通A股不违规情况"""
        # 模拟没有当日买入记录
        self.checker.position_service.get_positions_and_trades.return_value = {
            'position': 1000,
            'trades': []
        }

        result = self.checker.check_sell_restriction(
            'test_account', '600000', date.today())
        self.assertFalse(result)

    def test_normal_stock_violation(self):
        """测试普通A股违规情况"""
        # 模拟当日买入量等于持仓量
        self.checker.position_service.get_positions_and_trades.return_value = {
            'position': 1000,
            'trades': [{'quantity': 1000}]
        }

        result = self.checker.check_sell_restriction(
            'test_account', '600000', date.today())
        self.assertTrue(result)

    def test_star_market_check(self):
        """测试科创板规则检查"""
        # 模拟科创板股票
        self.checker.star_market_checker.check_star_market_rules.return_value = (False, "测试违规")

        result = self.checker.check_sell_restriction(
            'test_account', '688000', date.today())
        self.assertTrue(result)
        self.checker.star_market_checker.check_star_market_rules.assert_called_once()

    def test_error_handling(self):
        """测试异常处理"""
        # 模拟抛出异常
        self.checker.position_service.get_positions_and_trades.side_effect = Exception("测试异常")

        result = self.checker.check_sell_restriction(
            'test_account', '600000', date.today())
        self.assertTrue(result)
        self.checker.metrics.record_error.assert_called_once()

    def test_performance_metrics(self):
        """测试性能指标记录"""
        self.checker.position_service.get_positions_and_trades.return_value = {
            'position': 1000,
            'trades': []
        }

        result = self.checker.check_sell_restriction(
            'test_account', '600000', date.today())
        self.assertFalse(result)
        self.checker.metrics.record_check.assert_called_once()

if __name__ == '__main__':
    unittest.main()
