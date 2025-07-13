#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pandas as pd
from src.features.technical.processor import TechnicalIndicator
from src.features.feature_manager import FeatureManager

class TestTechnicalProcessor(unittest.TestCase):
    def setUp(self):
        self.manager = FeatureManager()
        self.processor = TechnicalIndicator(self.manager)

    def test_basic_indicators(self):
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        indicators = self.processor.calculate_basic_indicators("TEST", prices)

        # 测试MA5计算
        self.assertAlmostEqual(indicators['MA5'], np.mean(prices[-5:]))

        # 测试MA10计算
        self.assertAlmostEqual(indicators['MA10'], np.mean(prices[-10:]))

        # 测试MACD计算
        self.assertTrue('MACD' in indicators)

    def test_complex_indicators(self):
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        self.processor.calculate_basic_indicators("TEST", prices)
        indicators = self.processor.calculate_complex_indicators("TEST", prices)

        # 测试布林带
        self.assertTrue('BOLL_UP' in indicators)
        self.assertTrue('BOLL_MID' in indicators)
        self.assertTrue('BOLL_LOW' in indicators)

        # 测试RSI
        self.assertTrue('RSI14' in indicators)
        self.assertTrue(0 <= indicators['RSI14'] <= 100)

    def test_process_interface(self):
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        results = self.processor.process("TEST", prices)

        # 测试返回类型
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        self.assertTrue(all(hasattr(r, 'name') for r in results))
        self.assertTrue(all(hasattr(r, 'value') for r in results))
        self.assertTrue(all(hasattr(r, 'timestamp') for r in results))

if __name__ == '__main__':
    unittest.main()
