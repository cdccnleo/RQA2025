import unittest
from src.data.data_loader import DataLoader
from src.data.adapters.china import ChinaStockAdapter

class TestChinaIntegration(unittest.TestCase):
    """中国市场数据加载集成测试"""

    def setUp(self):
        self.loader = DataLoader()
        self.test_config = {
            'market': 'china',
            'data_type': 'stock',
            'symbol': '600000',
            'parallel': False
        }

    def test_adapter_registration(self):
        """测试中国市场适配器注册"""
        self.assertIsInstance(
            self.loader.adapters['china_stock'],
            ChinaStockAdapter
        )

    def test_data_loading_flow(self):
        """测试完整数据加载流程"""
        result = self.loader.load_data(self.test_config)

        # 验证返回数据结构
        self.assertIsNotNone(result)
        self.assertIn('raw_data', result)
        self.assertIn('metadata', result)

        # 验证中国市场特有字段
        metadata = result['metadata']
        self.assertEqual(metadata['market'], 'china')
        self.assertEqual(metadata['data_type'], 'stock')
        self.assertIn('regulation_status', metadata)

    def test_validation_flow(self):
        """测试验证流程"""
        # 模拟测试数据
        test_data = {
            'raw_data': {
                'type': 'C',
                'price_change_pct': 9.5,  # 在10%涨跌幅限制内
                'is_halted': False,
                'primary_price': 10.0,
                'secondary_price': 10.05
            },
            'metadata': self.test_config
        }

        # 获取适配器实例
        adapter = self.loader.adapters['china_stock']

        # 执行验证
        is_valid = adapter.validate(test_data)
        self.assertTrue(is_valid)

        # 测试涨跌停违规情况
        test_data['raw_data']['price_change_pct'] = 11.0
        is_valid = adapter.validate(test_data)
        self.assertFalse(is_valid)

if __name__ == '__main__':
    unittest.main()
