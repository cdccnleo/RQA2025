"""科创板策略测试"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from src.trading.strategies.china.star_market_strategy import StarMarketStrategy

class TestStarMarketStrategy(unittest.TestCase):
    """科创板策略测试类"""

    def setUp(self):
        """测试初始化"""
        self.strategy = StarMarketStrategy()
        self.strategy.metrics = MagicMock()
        self.strategy.risk_engine = MagicMock()

        # 模拟测试数据
        self.sample_data = {
            'close': np.array([50.0, 51.0, 52.5, 53.0, 54.0, 55.0]),
            'volume': np.array([20000, 25000, 30000, 28000, 35000, 40000]),
            'bid_volume': np.array([10000, 12000, 15000, 13000, 18000, 20000]),
            'ask_volume': np.array([10000, 13000, 15000, 15000, 17000, 20000]),
            'timestamp': ['2024-04-20 14:30:00', '2024-04-20 14:45:00',
                         '2024-04-20 15:00:00', '2024-04-20 15:15:00',
                         '2024-04-20 15:30:00', '2024-04-20 15:45:00']
        }

    def test_risk_check(self):
        """测试科创板风控检查"""
        # 设置风检通过
        self.strategy.risk_engine.check.return_value = {'allowed': True}

        signal = self.strategy.generate_signals(self.sample_data)
        self.assertIn('signal', signal)

        # 设置风检不通过
        self.strategy.risk_engine.check.return_value = {
            'allowed': False,
            'reason': 'price_limit'
        }

        signal = self.strategy.generate_signals(self.sample_data)
        self.assertEqual(signal['signal'], 0)
        self.assertEqual(signal['reason'], 'price_limit')

    def test_liquidity_calculation(self):
        """测试流动性计算"""
        liquidity = self.strategy._calculate_liquidity(self.sample_data)

        # 检查流动性值范围
        self.assertGreater(liquidity, 0)
        self.assertLess(liquidity, max(self.sample_data['bid_volume']) + max(self.sample_data['ask_volume']))

    def test_institutional_flow(self):
        """测试机构资金流检测"""
        flow = self.strategy._detect_institutional_flow(self.sample_data)

        # 检查资金流值范围
        self.assertGreaterEqual(flow, min(self.sample_data['volume']))
        self.assertLessEqual(flow, max(self.sample_data['volume']))

    @patch.object(StarMarketStrategy, '_is_after_hours')
    @patch.object(StarMarketStrategy, '_generate_after_hours_signal')
    def test_after_hours_trading(self, mock_generate, mock_check):
        """测试盘后交易处理"""
        # 设置不在盘后时段
        mock_check.return_value = False
        result = self.strategy.handle_after_hours_trading(self.sample_data)
        self.assertIsNone(result)

        # 设置在盘后时段
        mock_check.return_value = True
        mock_generate.return_value = {'signal': 0.5}
        result = self.strategy.handle_after_hours_trading(self.sample_data)
        self.assertEqual(result, {'signal': 0.5})

    def test_signal_generation(self):
        """测试综合信号生成"""
        # 设置风检通过
        self.strategy.risk_engine.check.return_value = {'allowed': True}

        signal = self.strategy.generate_signals(self.sample_data)

        # 检查返回结构
        self.assertIn('signal', signal)
        self.assertIn('details', signal)

        # 检查信号范围
        self.assertTrue(-1 <= signal['signal'] <= 1)

        # 检查详细信号
        details = signal['details']
        self.assertIn('macd', details)
        self.assertIn('rsi', details)
        self.assertIn('volatility', details)
        self.assertIn('liquidity', details)
        self.assertIn('institutional', details)

        # 检查性能记录是否被调用
        self.strategy.metrics.record_signal_generation.assert_called_once()

if __name__ == '__main__':
    unittest.main()
