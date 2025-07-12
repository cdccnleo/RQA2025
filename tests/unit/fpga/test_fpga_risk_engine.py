#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from src.fpga.fpga_manager import FPGAManager
from src.fpga.fpga_risk_engine import FPGARiskEngine

class TestFPGARiskEngine(unittest.TestCase):
    def setUp(self):
        self.fpga_manager = FPGAManager()
        self.risk_engine = FPGARiskEngine(self.fpga_manager)

    def test_initialization(self):
        # 测试初始化
        self.assertTrue(self.risk_engine.initialize())
        self.assertTrue(self.risk_engine.initialized)

    def test_circuit_breaker(self):
        # 测试熔断检查
        self.risk_engine.initialize()

        # 测试10%熔断
        self.assertTrue(self.risk_engine.check_circuit_breaker(90, 100))

        # 测试7%熔断
        self.assertTrue(self.risk_engine.check_circuit_breaker(93, 100))

        # 测试5%熔断
        self.assertTrue(self.risk_engine.check_circuit_breaker(95, 100))

        # 测试不熔断
        self.assertFalse(self.risk_engine.check_circuit_breaker(96, 100))

    def test_price_limit(self):
        # 测试涨跌停检查
        self.risk_engine.initialize()

        # 测试普通股票涨停
        self.assertTrue(self.risk_engine.check_price_limit(110, 100))

        # 测试普通股票跌停
        self.assertTrue(self.risk_engine.check_price_limit(90, 100))

        # 测试科创板涨停
        self.assertTrue(self.risk_engine.check_price_limit(120, 100, True))

        # 测试科创板跌停
        self.assertTrue(self.risk_engine.check_price_limit(80, 100, True))

        # 测试不触发涨跌停
        self.assertFalse(self.risk_engine.check_price_limit(105, 100))

    def test_batch_check(self):
        # 测试批量检查
        self.risk_engine.initialize()

        checks = [
            {'type': 'circuit_breaker', 'params': {'price': 90, 'ref_price': 100}},
            {'type': 'price_limit', 'params': {'price': 110, 'prev_close': 100}},
            {'type': 'invalid', 'params': {}}
        ]

        results = self.risk_engine.batch_check(checks)

        self.assertEqual(len(results), 3)
        self.assertTrue(results[0]['result'])  # circuit_breaker
        self.assertTrue(results[1]['result'])  # price_limit
        self.assertIn('error', results[2])     # invalid type

if __name__ == '__main__':
    unittest.main()
