"""实时引擎压力测试单元测试"""

import unittest
from unittest.mock import MagicMock, patch
import time
from src.engine.stress_test import StressTester

class MockEngine:
    def __init__(self):
        self.process = MagicMock()

class TestStressTester(unittest.TestCase):
    def setUp(self):
        self.engine = MockEngine()
        self.tester = StressTester(self.engine)

    def test_normal_market_test(self):
        """测试正常市场场景"""
        with patch('time.time', return_value=123456789):
            data = self.tester._generate_normal_market_data()
            self.assertIn('symbol', data)
            self.assertGreaterEqual(data['price'], 10)
            self.assertLessEqual(data['price'], 100)
            self.assertEqual(data['type'], 'normal')

    def test_extreme_market_test(self):
        """测试极端市场场景"""
        with patch('time.time', return_value=123456789):
            data = self.tester._generate_extreme_market_data()
            self.assertIn('symbol', data)
            self.assertGreaterEqual(data['price'], 1)
            self.assertLessEqual(data['price'], 1000)
            self.assertEqual(data['type'], 'extreme')

    def test_burst_market_test(self):
        """测试突发大量数据场景"""
        with patch('time.time', return_value=123456789):
            data_list = self.tester._generate_burst_market_data()
            self.assertEqual(len(data_list), 100)
            for data in data_list:
                self.assertEqual(data['type'], 'normal')

    @patch('psutil.cpu_percent', return_value=50)
    @patch('psutil.virtual_memory')
    @patch.object(StressTester, '_data_generator')
    @patch.object(StressTester, '_data_consumer')
    def test_run_test(self, mock_consumer, mock_generator, mock_mem, mock_cpu):
        """测试运行压力测试"""
        mock_mem.return_value.percent = 60
        mock_consumer.return_value = None
        mock_generator.return_value = None

        stats = self.tester.run_test(duration=1, scenario='normal')

        self.assertIn('total_messages', stats)
        self.assertIn('avg_latency', stats)
        self.assertIn('max_latency', stats)
        self.assertIn('min_latency', stats)
        self.assertIn('errors', stats)

        # 确保调用了系统监控
        mock_cpu.assert_called()
        mock_mem.assert_called()

if __name__ == '__main__':
    unittest.main()
