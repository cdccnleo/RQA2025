import unittest
from datetime import datetime, timedelta
from src.trading.china.risk.circuit_breaker import CircuitBreaker

class TestCircuitBreaker(unittest.TestCase):
    def setUp(self):
        self.cb = CircuitBreaker()

    def test_normal_market(self):
        """测试正常市场状态"""
        status = self.cb.check_market_status({
            'current': 100,
            'prev_close': 100,
            'timestamp': datetime.now()
        })
        self.assertFalse(status['triggered'])

    def test_level1_breaker(self):
        """测试5%熔断触发"""
        status = self.cb.check_market_status({
            'current': 95,
            'prev_close': 100,
            'timestamp': datetime.now()
        })
        self.assertTrue(status['triggered'])
        self.assertEqual(status['level'], 0.05)

    def test_level2_breaker(self):
        """测试7%熔断触发"""
        status = self.cb.check_market_status({
            'current': 93,
            'prev_close': 100,
            'timestamp': datetime.now()
        })
        self.assertTrue(status['triggered'])
        self.assertEqual(status['level'], 0.07)

    def test_level3_breaker(self):
        """测试10%熔断触发"""
        status = self.cb.check_market_status({
            'current': 90,
            'prev_close': 100,
            'timestamp': datetime.now()
        })
        self.assertTrue(status['triggered'])
        self.assertEqual(status['level'], 0.10)

    def test_recovery(self):
        """测试熔断恢复"""
        now = datetime.now()
        self.cb.check_market_status({
            'current': 95,
            'prev_close': 100,
            'timestamp': now
        })

        # 15分钟后应恢复
        status = self.cb.check_market_status({
            'current': 95,
            'prev_close': 100,
            'timestamp': now + timedelta(minutes=16)
        })
        self.assertFalse(status['triggered'])

if __name__ == '__main__':
    unittest.main()
