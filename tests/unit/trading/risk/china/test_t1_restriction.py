import pytest
from datetime import datetime, timedelta
from trading.risk.china.t1_restriction import T1RestrictionChecker

class TestT1Restriction:
    @pytest.fixture
    def checker(self):
        return T1RestrictionChecker()

    def test_t1_restriction_violation(self, checker):
        # 测试T+1限制违规
        order = {
            'symbol': '600519.SH',
            'quantity': 1000,
            'direction': 'sell'
        }
        position = {
            '600519.SH': {
                'buy_date': datetime.now().date(),
                'quantity': 500
            }
        }
        assert checker.check(order, position) == False

    def test_t1_restriction_pass(self, checker):
        # 测试T+1限制通过
        order = {
            'symbol': '600519.SH',
            'quantity': 1000,
            'direction': 'sell'
        }
        position = {
            '600519.SH': {
                'buy_date': (datetime.now() - timedelta(days=2)).date(),
                'quantity': 1500
            }
        }
        assert checker.check(order, position) == True
