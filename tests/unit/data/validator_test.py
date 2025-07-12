import unittest
from unittest.mock import patch
import pandas as pd
from src.data.validators import (
    DataValidator,
    ChinaDataValidator,
    QualityMonitor
)

class TestDataValidators(unittest.TestCase):
    """统一后的数据验证测试"""

    def setUp(self):
        self.validator = DataValidator()
        self.china_validator = ChinaDataValidator()
        self.quality_monitor = QualityMonitor()

        # 测试数据
        self.valid_stock_data = pd.DataFrame({
            'symbol': ['600000.SH', '000001.SZ'],
            'price': [42.50, 15.80],
            'volume': [100000, 200000]
        })

        self.invalid_stock_data = pd.DataFrame({
            'symbol': ['600000.SH', '000001.SZ'],
            'price': [42.50, -15.80],  # 无效价格
            'volume': [100000, 200000]
        })

    def test_basic_validation(self):
        """测试基础数据验证"""
        result = self.validator.validate(self.valid_stock_data)
        self.assertTrue(result.all())

        result = self.validator.validate(self.invalid_stock_data)
        self.assertFalse(result.all())

    # 从test_data_processor.py合并的测试
    def test_missing_data_handling(self):
        """测试缺失数据处理"""
        data_with_nan = self.valid_stock_data.copy()
        data_with_nan.loc[0, 'price'] = None

        result = self.validator.validate(data_with_nan)
        self.assertFalse(result[0])  # 第一行应失败

    # 从test_quality_monitor.py合并的测试
    @patch('src.data.validators.QualityMonitor._calculate_metrics')
    def test_quality_metrics(self, mock_calc):
        """测试质量指标计算"""
        mock_calc.return_value = {'completeness': 0.99}
        metrics = self.quality_monitor.evaluate(self.valid_stock_data)
        self.assertGreater(metrics['completeness'], 0.95)

    # 中国市场特定验证测试
    def test_china_price_limit(self):
        """测试中国涨跌停限制"""
        data = self.valid_stock_data.copy()
        data.loc[0, 'price'] = 42.50 * 1.1  # 涨停价

        result = self.china_validator.validate_price(data)
        self.assertTrue(result[0])  # 涨停价应通过

        data.loc[1, 'price'] = 15.80 * 0.9  # 跌停价
        result = self.china_validator.validate_price(data)
        self.assertTrue(result.all())

    # 新增性能测试
    def test_large_data_validation(self):
        """测试大数据量验证性能"""
        large_data = pd.concat([self.valid_stock_data] * 1000)
        import time
        start = time.time()
        result = self.validator.validate(large_data)
        elapsed = time.time() - start
        self.assertTrue(result.all())
        self.assertLess(elapsed, 1.0)  # 应在1秒内完成

if __name__ == '__main__':
    unittest.main()
