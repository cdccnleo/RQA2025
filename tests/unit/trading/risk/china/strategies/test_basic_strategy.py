"""A股基础策略测试"""

import unittest
import numpy as np
from src.trading.strategies.china.basic_strategy import BasicChinaStrategy
from unittest.mock import MagicMock

class TestBasicChinaStrategy(unittest.TestCase):
    """A股基础策略测试类"""

    def setUp(self):
        """测试初始化"""
        self.strategy = BasicChinaStrategy()
        self.strategy.metrics = MagicMock()

        # 模拟测试数据
        self.sample_data = {
            'close': np.array([10.0, 10.5, 11.0, 10.8, 11.2, 11.5]),
            'volume': np.array([10000, 12000, 15000, 13000, 18000, 20000])
        }

    def test_technical_signals(self):
        """测试技术指标信号"""
        signals = self.strategy.calculate_technical_signals(self.sample_data)

        # 检查返回的信号字典
        self.assertIn('macd', signals)
        self.assertIn('rsi', signals)
        self.assertIn('bollinger', signals)

        # 检查信号值范围
        self.assertTrue(-1 <= signals['macd'] <= 1)
        self.assertTrue(0 <= signals['rsi'] <= 100)
        self.assertTrue(len(signals['bollinger']) == 3)  # 上中下轨
        
    def test_volume_signals(self):
        """测试量价关系信号"""
        signals = self.strategy.calculate_volume_signals(self.sample_data)
        
        self.assertIn('divergence', signals)
        self.assertIn('breakout', signals)
        
        # 量价背离应为布尔值
        self.assertIsInstance(signals['divergence'], bool)
        # 放量突破应为数值
        self.assertIsInstance(signals['breakout'], (int, float))
        
    def test_ma_signals(self):
        """测试均线系统信号"""
        signals = self.strategy.calculate_ma_signals(self.sample_data)
        
        self.assertIn('cross', signals)
        self.assertIn('trend', signals)

        # 均线交叉信号范围
        self.assertTrue(-1 <= signals['cross'] <= 1)
        # 均线趋势应为数值
        self.assertIsInstance(signals['trend'], (int, float))

    def test_combined_signal(self):
        """测试综合信号生成"""
        signal = self.strategy.generate_signals(self.sample_data)

        # 检查返回结构
        self.assertIn('signal', signal)
        self.assertIn('details', signal)

        # 检查综合信号范围
        self.assertTrue(-1 <= signal['signal'] <= 1)

        # 检查详细信号
        details = signal['details']
        self.assertIn('macd', details)
        self.assertIn('rsi', details)
        self.assertIn('bollinger', details)
        self.assertIn('divergence', details)
        self.assertIn('breakout', details)
        self.assertIn('cross', details)
        self.assertIn('trend', details)

        # 检查性能记录是否被调用
        self.strategy.metrics.record_signal_generation.assert_called_once()

if __name__ == '__main__':
    unittest.main()
