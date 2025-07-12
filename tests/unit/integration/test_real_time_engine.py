"""实时引擎集成测试"""
import unittest
import asyncio
from unittest.mock import MagicMock
from src.engine.real_time_engine import RealTimeEngine
from src.engine.level2_adapter import Level2Adapter

class TestRealTimeEngineIntegration(unittest.TestCase):
    """实时引擎集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.engine = RealTimeEngine()
        self.adapter = Level2Adapter(self.engine)

        # 模拟数据
        self.test_symbol = "600519.SH"
        self.test_data = {
            "code": "600519",
            "timestamp": "20240329150000000",
            "bid": [("180000", "100"), ("179950", "200")],
            "ask": [("180050", "150"), ("180100", "300")],
            "limit_up": True
        }

    def test_data_processing_flow(self):
        """测试数据处理流程"""
        # 注册模拟处理器
        mock_handler = MagicMock()
        self.engine.register_handler('order_book', mock_handler)

        # 启动引擎
        self.engine.start()

        # 处理测试数据
        self.adapter._process_raw_data(self.test_data)

        # 验证数据处理结果
        mock_handler.assert_called_once()
        processed_data = mock_handler.call_args[0][0]

        # 验证关键字段
        self.assertEqual(processed_data['symbol'], self.test_symbol)
        self.assertEqual(len(processed_data['bids']), 2)
        self.assertEqual(len(processed_data['asks']), 2)
        self.assertEqual(processed_data['status'], 'up')

    def test_limit_status_update(self):
        """测试涨跌停状态更新"""
        # 处理测试数据
        self.adapter._process_raw_data(self.test_data)

        # 验证状态更新
        status = self.adapter.get_limit_status(self.test_symbol)
        self.assertEqual(status, 'up')

    def test_performance_metrics(self):
        """测试性能指标收集"""
        # 处理多组测试数据
        for _ in range(100):
            self.adapter._process_raw_data(self.test_data)

        # 获取性能指标
        metrics = self.engine.get_metrics()

        # 验证基本指标
        self.assertGreater(metrics['throughput'], 0)
        self.assertGreaterEqual(metrics['latency'], 0)

    def test_a_share_specific_features(self):
        """测试A股特有功能"""
        # 处理测试数据
        self.adapter._process_raw_data(self.test_data)

        # 验证价格转换(除以10000)
        bids = self.adapter._parse_level2_bids(self.test_data['bid'])
        self.assertEqual(bids[0][0], 18.0)  # 180000 / 10000

        # 验证代码标准化
        normalized = self.adapter._normalize_symbol("000001")
        self.assertEqual(normalized, "000001.SZ")

    async def test_async_integration(self):
        """测试异步集成"""
        # 创建并启动适配器
        adapter = await Level2Adapter.create_and_start_adapter(self.engine)

        # 验证基本功能
        self.assertIsInstance(adapter, Level2Adapter)
        self.assertEqual(len(adapter.symbol_map), 2)

if __name__ == '__main__':
    unittest.main()
