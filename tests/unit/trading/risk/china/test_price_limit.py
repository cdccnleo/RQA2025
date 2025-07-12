import pytest
from src.trading.risk.china.price_limit import PriceLimitChecker

class TestPriceLimit:
    @pytest.fixture
    def checker(self):
        return PriceLimitChecker()

    def test_normal_stock_limit(self, checker):
        # 测试普通股票涨跌停限制(10%)
        assert checker.check('600519.SH', 1050, 1000) == True  # +5%
        assert checker.check('600519.SH', 1101, 1000) == False  # +10.1%

    def test_st_stock_limit(self, checker):
        # 测试ST股票涨跌停限制(5%)
        assert checker.check('600***.SH', 525, 500) == True  # +5%
        assert checker.check('600***.SH', 526, 500) == False  # +5.2%

    def test_new_stock_limit(self, checker):
        # 测试新股首日涨跌停限制(44%)
        assert checker.check('603***.SH', 1440, 1000) == True  # +44%
        assert checker.check('603***.SH', 1441, 1000) == False  # +44.1%
