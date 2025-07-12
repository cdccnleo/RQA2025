#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from src.fpga.fpga_manager import FPGAManager
from src.fpga.fpga_orderbook_optimizer import FPGAOrderbookOptimizer

class TestFPGAOrderbookOptimizer(unittest.TestCase):
    def setUp(self):
        self.fpga_manager = FPGAManager()
        self.optimizer = FPGAOrderbookOptimizer(self.fpga_manager)

    def test_initialization(self):
        # 测试初始化
        self.assertTrue(self.optimizer.initialize())
        self.assertTrue(self.optimizer.initialized)

    def test_vwap_calculation(self):
        # 测试VWAP计算
        self.optimizer.initialize()

        # 正常情况
        prices = [10.0, 10.1, 10.2]
        volumes = [100, 200, 300]
        expected = (10.0*100 + 10.1*200 + 10.2*300) / (100+200+300)
        self.assertAlmostEqual(self.optimizer.calculate_vwap(prices, volumes), expected)

        # 空数据
        self.assertEqual(self.optimizer.calculate_vwap([], []), 0.0)

    def test_twap_calculation(self):
        # 测试TWAP计算
        self.optimizer.initialize()

        # 正常情况
        prices = [10.0, 10.1, 10.2]
        expected = sum(prices) / len(prices)
        self.assertAlmostEqual(self.optimizer.calculate_twap(prices), expected)

        # 空数据
        self.assertEqual(self.optimizer.calculate_twap([]), 0.0)

    def test_imbalance_calculation(self):
        # 测试不平衡度计算
        self.optimizer.initialize()

        bids = [(10.0, 100), (9.9, 200), (9.8, 300)]
        asks = [(10.1, 150), (10.2, 250), (10.3, 350)]

        # 计算前5档不平衡度
        bid_vol = sum(vol for _, vol in bids[:5])
        ask_vol = sum(vol for _, vol in asks[:5])
        expected = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        self.assertAlmostEqual(self.optimizer.calculate_imbalance(bids, asks), expected)

        # 空数据
        self.assertEqual(self.optimizer.calculate_imbalance([], []), 0.0)

    def test_order_optimization(self):
        # 测试订单优化
        self.optimizer.initialize()

        # TWAP订单
        twap_params = {'slices': 5, 'interval': 60}
        result = self.optimizer.optimize_order('TWAP', twap_params)
        self.assertEqual(result['strategy'], 'TWAP')
        self.assertEqual(result['slices'], 5)

        # VWAP订单
        vwap_params = {'volume_pct': 0.2, 'aggressiveness': 0.7}
        result = self.optimizer.optimize_order('VWAP', vwap_params)
        self.assertEqual(result['strategy'], 'VWAP')
        self.assertEqual(result['volume_pct'], 0.2)

        # IOC订单
        ioc_params = {'price': 10.0, 'quantity': 100}
        result = self.optimizer.optimize_order('IOC', ioc_params)
        self.assertEqual(result['strategy'], 'IOC')
        self.assertEqual(result['price'], 10.0)

if __name__ == '__main__':
    unittest.main()
