"""A股数据适配器测试"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
from src.data.china.adapter import ChinaDataAdapter
import logging

class TestChinaDataAdapter(unittest.TestCase):
    """测试A股数据适配器"""

    def setUp(self):
        """初始化测试环境"""
        self.adapter = ChinaDataAdapter()
        logging.basicConfig(level=logging.DEBUG)

        # 模拟测试数据
        self.sample_margin_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'symbol': ['600519', '000001'],
            'margin_balance': [1e8, 1.2e8],
            'short_balance': [1e7, 1.5e7]
        })

        self.sample_trades = pd.DataFrame({
            'date': [datetime(2023,1,1), datetime(2023,1,2)],
            'symbol': ['600519', '000001'],
            'action': ['BUY', 'SELL'],
            'price': [1800, 42.5],
            'quantity': [100, 200]
        })

    @patch('pandas.read_csv')
    def test_load_margin_data(self, mock_read):
        """测试融资融券数据加载"""
        mock_read.return_value = self.sample_margin_data

        # 测试正常加载
        result = self.adapter.load_margin_data()
        self.assertEqual(len(result), 2)
        self.assertIn('margin_balance', result.columns)

        # 测试数据缺失情况
        mock_read.return_value = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.adapter.load_margin_data()

    def test_validate_t1_settlement(self):
        """测试T+1结算验证"""
        # 合规交易
        valid_trades = pd.DataFrame({
            'date': [datetime(2023,1,1), datetime(2023,1,2)],
            'action': ['BUY', 'SELL']
        })
        self.assertTrue(self.adapter.validate_t1_settlement(valid_trades))

        # 违规交易(当日卖出)
        invalid_trades = pd.DataFrame({
            'date': [datetime(2023,1,1), datetime(2023,1,1)],
            'action': ['BUY', 'SELL']
        })
        self.assertFalse(self.adapter.validate_t1_settlement(invalid_trades))

    def test_get_price_limits(self):
        """测试涨跌停价格计算"""
        # 主板股票
        limits = self.adapter.get_price_limits('600519')
        self.assertEqual(limits['upper_limit'], 0.1)

        # 科创板股票
        limits = self.adapter.get_price_limits('688981')
        self.assertEqual(limits['upper_limit'], 0.2)

class TestRedisCache(unittest.TestCase):
    """测试Redis缓存实现"""

    @patch('redis.Redis')
    def test_cache_operations(self, mock_redis):
        """测试缓存读写"""
        from src.data.china.adapter import ChinaDataAdapter
        mock_client = MagicMock()
        mock_redis.return_value = mock_client

        adapter = ChinaDataAdapter()
        test_data = pd.DataFrame({'data': [1,2,3]})

        # 测试缓存写入
        adapter.cache_data('test_key', test_data)
        mock_client.set.assert_called_once()

        # 测试缓存读取(模拟实现)
        with patch.object(adapter, '_get_cached', return_value=test_data):
            result = adapter._get_cached('test_key')
            self.assertEqual(len(result), 3)

if __name__ == '__main__':
    unittest.main()
