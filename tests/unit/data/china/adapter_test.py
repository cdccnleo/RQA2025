import unittest
from unittest.mock import patch
from src.data.adapters.china import ChinaDataAdapter
from src.data.models import DataModel

class TestChinaDataAdapter(unittest.TestCase):
    """统一后的中国数据适配器测试"""

    def setUp(self):
        self.adapter = ChinaDataAdapter()
        self.sample_data = {
            "symbol": "600000.SH",
            "price": 42.50,
            "volume": 100000
        }

    def test_data_loading(self):
        """测试数据加载基本功能"""
        result = self.adapter.load_data(self.sample_data)
        self.assertIsInstance(result, DataModel)
        self.assertEqual(result.symbol, "600000.SH")

    def test_local_validation(self):
        """测试中国市场的本地验证规则"""
        invalid_data = self.sample_data.copy()
        invalid_data["price"] = -1  # 无效价格

        with self.assertRaises(ValueError):
            self.adapter.validate(invalid_data)

    # 从test_adapter.py合并的特殊场景测试
    @patch('src.data.adapters.china.ExchangeCalendar')
    def test_trading_day_validation(self, mock_calendar):
        """测试交易日验证"""
        mock_calendar.is_trading_day.return_value = False
        data = self.sample_data.copy()
        data["date"] = "2024-01-01"  # 非交易日

        with self.assertRaises(ValueError):
            self.adapter.validate(data)

    def test_margin_trading_data(self):
        """测试融资融券数据适配"""
        margin_data = {
            "symbol": "600000.SH",
            "margin_balance": 1000000,
            "short_balance": 500000
        }

        result = self.adapter.load_data(margin_data)
        self.assertEqual(result.margin_ratio, 2.0)

    # 新增性能测试
    def test_batch_processing(self):
        """测试批量数据处理性能"""
        batch = [self.sample_data] * 1000
        results = self.adapter.batch_process(batch)
        self.assertEqual(len(results), 1000)
        self.assertTrue(all(isinstance(r, DataModel) for r in results))

if __name__ == '__main__':
    unittest.main()
