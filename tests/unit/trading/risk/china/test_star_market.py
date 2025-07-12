import pytest
from trading.risk.china.position_limits import STARMarketChecker

class TestSTARMarket:
    @pytest.fixture
    def checker(self):
        return STARMarketChecker()

    def test_stock_qualification(self, checker):
        # 测试科创板股票资格
        assert checker.check_qualification('688111.SH') == True
        assert checker.check_qualification('600519.SH') == False

    def test_price_limit(self, checker):
        # 测试科创板涨跌幅限制
        assert checker.check_price_limit('688111.SH', 0.21) == False  # 超过20%限制
        assert checker.check_price_limit('688111.SH', 0.15) == True

    def test_opening_auction(self, checker):
        # 测试科创板开盘集合竞价规则
        order = {
            'symbol': '688111.SH',
            'price': 100,
            'time': '09:15:00',
            'type': 'limit'
        }
        assert checker.check_opening_auction(order) == True

        order['price'] = 120  # 超过开盘价±2%限制
        assert checker.check_opening_auction(order) == False
