import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.features.processors import (
    FeatureEngineer,
    TechnicalProcessor,
    FeatureSelector
)

class TestFeatureProcessors(unittest.TestCase):
    """统一后的特征处理器测试"""

    def setUp(self):
        self.engineer = FeatureEngineer()
        self.tech_processor = TechnicalProcessor()
        self.selector = FeatureSelector()

        # 测试数据
        self.sample_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [10000, 12000, 15000, 9000, 11000]
        })

    def test_basic_feature_engineering(self):
        """测试基础特征工程"""
        features = self.engineer.transform(self.sample_data)
        self.assertIn('returns', features.columns)
        self.assertIn('log_volume', features.columns)

    # 从test_feature_engineer.py合并的测试
    def test_custom_feature_generation(self):
        """测试自定义特征生成"""
        def custom_func(df):
            return df['close'] / df['volume']

        features = self.engineer.add_custom_feature(
            self.sample_data,
            'price_volume_ratio',
            custom_func
        )
        self.assertIn('price_volume_ratio', features.columns)

    # 从test_technical_processor.py合并的测试
    def test_rsi_calculation(self):
        """测试RSI指标计算"""
        rsi = self.tech_processor.calculate_rsi(self.sample_data['close'])
        self.assertEqual(len(rsi), len(self.sample_data))
        self.assertTrue(all(0 <= x <= 100 for x in rsi[1:]))

    # 从test_feature_selector.py合并的测试
    @patch('src.features.processors.FeatureSelector._train_model')
    def test_feature_importance(self, mock_train):
        """测试特征重要性评估"""
        mock_train.return_value = {'close': 0.8, 'volume': 0.2}
        importance = self.selector.evaluate_importance(
            self.sample_data,
            target=np.array([0, 1, 0, 1, 0])
        )
        self.assertGreater(importance['close'], importance['volume'])

    # 新增性能测试
    def test_large_data_processing(self):
        """测试大数据量处理性能"""
        large_data = pd.concat([self.sample_data] * 1000)
        import time
        start = time.time()
        result = self.engineer.transform(large_data)
        elapsed = time.time() - start
        self.assertEqual(len(result), len(large_data))
        self.assertLess(elapsed, 1.0)  # 应在1秒内完成

if __name__ == '__main__':
    unittest.main()
