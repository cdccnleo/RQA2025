"""涨跌停限制检查单元测试"""

import unittest
from unittest.mock import MagicMock, patch

from src.trading.risk.china.price_limit import PriceLimitChecker

class TestPriceLimit(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.checker = PriceLimitChecker()
        self.checker.market_data_service = MagicMock()
        self.checker.metrics = MagicMock()

        # 模拟前收盘价
        self.checker.market_data_service.get_prev_close.return_value = 10.0

    def test_normal_stock_buy_within_limit(self):
        """测试普通A股买入未超限"""
        # 设置普通股票(10%涨跌幅)
        self.checker._get_limit_rate = lambda _: 0.1

        result = self.checker.check_price_limit('600000', 11.0, 'buy')
        self.assertTrue(result)
        self.checker.metrics.record_check.assert_called_once()

    def test_normal_stock_buy_exceed_limit(self):
        """测试普通A股买入超限"""
        self.checker._get_limit_rate = lambda _: 0.1

        result = self.checker.check_price_limit('600000', 11.01, 'buy')
        self.assertFalse(result)
        self.checker.metrics.record_check.assert_called_once()

    def test_star_market_sell_within_limit(self):
        """测试科创板卖出未超限"""
        self.checker._get_limit_rate = lambda _: 0.2

        result = self.checker.check_price_limit('688000', 8.0, 'sell')
        self.assertTrue(result)
        self.checker.metrics.record_check.assert_called_once()

    def test_star_market_sell_exceed_limit(self):
        """测试科创板卖出超限"""
        self.checker._get_limit_rate = lambda _: 0.2

        result = self.checker.check_price_limit('688000', 7.99, 'sell')
        self.assertFalse(result)
        self.checker.metrics.record_check.assert_called_once()

    def test_missing_prev_close(self):
        """测试缺少前收盘价情况"""
        self.checker.market_data_service.get_prev_close.return_value = None

        result = self.checker.check_price_limit('600000', 11.0, 'buy')
        self.assertFalse(result)
        self.checker.metrics.record_error.assert_called_once()

    def test_error_handling(self):
        """测试异常处理"""
        self.checker.market_data_service.get_prev_close.side_effect = Exception("测试异常")

        result = self.checker.check_price_limit('600000', 11.0, 'buy')
        self.assertFalse(result)
        self.checker.metrics.record_error.assert_called_once()

if __name__ == '__main__':
    unittest.main()
