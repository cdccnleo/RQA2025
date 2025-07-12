"""数据层集成测试"""
import unittest
import pandas as pd
from datetime import datetime, timedelta

from src.data.adapters.china_data_adapter import ChinaDataAdapter
from src.data.decoders.level2_decoder import Level2Decoder
from src.data.validation.data_validator import DataValidator, DataType

class TestDataLayerIntegration(unittest.TestCase):
    """数据层集成测试用例"""

    def setUp(self):
        """测试初始化"""
        self.adapter = ChinaDataAdapter()
        self.decoder = Level2Decoder()
        self.validator = DataValidator()

        # 测试数据
        self.test_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        self.end_date = self.test_date

        # Level2测试数据
        self.level2_data = bytes.fromhex(
            'AA5500000000000000000000000000000000000000000000000000000000000000'
            '000000000000000000000000000000000000000000000000000000000000000000'
        )

    def test_margin_data_flow(self):
        """测试融资融券数据流"""
        # 获取数据
        margin_data = self.adapter.load_margin_data(self.start_date, self.end_date)

        # 验证数据格式
        self.assertIsInstance(margin_data, pd.DataFrame)
        self.assertGreater(len(margin_data), 0)

        # 验证数据内容
        required_columns = ['date', 'symbol', 'margin_balance', 'short_balance']
        for col in required_columns:
            self.assertIn(col, margin_data.columns)

        # 验证数据有效性
        validation_result = self.validator.validate(margin_data, DataType.MARGIN_TRADING)
        self.assertTrue(validation_result['is_valid'],
                       f"Validation failed: {validation_result['errors']}")

    def test_dragon_board_flow(self):
        """测试龙虎榜数据流"""
        # 获取数据
        dragon_data = self.adapter.load_dragon_board(self.test_date)

        # 验证数据格式
        self.assertIsInstance(dragon_data, dict)
        self.assertIn('buy_seats', dragon_data)
        self.assertIn('sell_seats', dragon_data)
        self.assertIn('symbols', dragon_data)

        # 验证数据有效性
        validation_result = self.validator.validate(dragon_data, DataType.DRAGON_BOARD)
        self.assertTrue(validation_result['is_valid'],
                       f"Validation failed: {validation_result['errors']}")

    def test_level2_decoding_flow(self):
        """测试Level2解码数据流"""
        # 解码数据
        decoded_data = self.decoder.decode(self.level2_data)

        # 验证数据格式
        self.assertIsInstance(decoded_data, dict)
        required_keys = ['symbol', 'price', 'volume', 'bids', 'asks', 'market']
        for key in required_keys:
            self.assertIn(key, decoded_data)

        # 验证数据有效性
        validation_result = self.validator.validate(decoded_data, DataType.LEVEL2)
        self.assertTrue(validation_result['is_valid'],
                       f"Validation failed: {validation_result['errors']}")

    def test_batch_decoding(self):
        """测试批量解码"""
        # 准备批量数据
        batch_data = [self.level2_data] * 10

        # 批量解码
        results = self.decoder.decode_batch(batch_data)

        # 验证结果
        self.assertEqual(len(results), len(batch_data))
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('symbol', result)

if __name__ == '__main__':
    unittest.main()
