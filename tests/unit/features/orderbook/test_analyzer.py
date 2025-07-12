#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from src.features.orderbook.analyzer import OrderbookAnalyzer
from src.features.manager import FeatureManager

class TestOrderbookAnalyzer(unittest.TestCase):
    def setUp(self):
        self.manager = FeatureManager()
        self.analyzer = OrderbookAnalyzer(self.manager)

        # 准备测试数据
        self.symbol = "600000.SH"
        self.bids = [(10.0, 100), (9.9, 200), (9.8, 300)]
        self.asks = [(10.1, 150), (10.2, 250), (10.3, 350)]

    def test_update_orderbook(self):
        # 测试订单簿更新
        self.analyzer.update_orderbook(self.symbol, self.bids, self.asks)
        self.assertIn(self.symbol, self.analyzer.orderbook_cache)

        # 检查买盘排序
        bids = self.analyzer.orderbook_cache[self.symbol]['bids']
        self.assertTrue(all(bids[i][0] >= bids[i+1][0] for i in range(len(bids)-1)))

        # 检查卖盘排序
        asks = self.analyzer.orderbook_cache[self.symbol]['asks']
        self.assertTrue(all(asks[i][0] <= asks[i+1][0] for i in range(len(asks)-1)))

    def test_calculate_metrics(self):
        # 测试指标计算
        self.analyzer.update_orderbook(self.symbol, self.bids, self.asks)
        metrics = self.analyzer.calculate_metrics(self.symbol)

        # 检查基础指标
        self.assertIn('spread', metrics)
        self.assertAlmostEqual(metrics['spread'], 0.1)

        self.assertIn('imbalance', metrics)
        self.assertTrue(-1 <= metrics['imbalance'] <= 1)

        self.assertIn('depth', metrics)
        self.assertIsInstance(metrics['depth'], dict)

    def test_register_features(self):
        # 测试特征注册
        self.analyzer.register_features()
        self.assertTrue('orderbook_imbalance' in self.manager.features)
        self.assertTrue('orderbook_spread' in self.manager.features)

if __name__ == '__main__':
    unittest.main()
