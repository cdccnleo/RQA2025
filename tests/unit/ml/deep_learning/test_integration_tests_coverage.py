#!/usr/bin/env python3
"""
深度学习集成测试覆盖率专项测试

目标：提升 deep_learning/core/integration_tests.py 的实际业务流程测试覆盖率
"""

import sys
import os
import tempfile
import shutil
import unittest
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# 确保正确的模块路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入测试模块
try:
    from src.ml.deep_learning.core.data_pipeline import DataPipeline
    from src.ml.deep_learning.core.model_service import ModelService
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    IMPORT_SUCCESS = False


def create_test_financial_data(n_samples: int = 100) -> pd.DataFrame:
    """创建测试用的金融数据"""
    np.random.seed(42)
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    volumes = np.abs(np.random.randn(n_samples)) * 1000 + 100

    return pd.DataFrame({
        'timestamp': timestamps,
        'close': close_prices,
        'volume': volumes
    })


class TestDeepLearningIntegrationCoverage(unittest.TestCase):
    """深度学习集成测试覆盖率测试"""

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        if not IMPORT_SUCCESS:
            self.skipTest("依赖模块导入失败")

    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'test_dir') and self.test_dir:
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_data_pipeline_feature_engineering(self):
        """测试数据管道特征工程功能"""
        # 创建测试数据
        data = create_test_financial_data(100)
        pipeline = DataPipeline()

        # 测试数据拆分
        split = pipeline.split(data)

        # 验证拆分结果
        self.assertGreater(len(split.train), 0)
        self.assertGreater(len(split.validation), 0)
        self.assertGreater(len(split.test), 0)

        # 验证数据总量
        total_len = len(split.train) + len(split.validation) + len(split.test)
        self.assertEqual(total_len, len(data))

        # 验证特征列存在
        required_columns = ['timestamp', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, split.train.columns)
            self.assertIn(col, split.validation.columns)
            self.assertIn(col, split.test.columns)

        print("✅ 数据管道特征工程测试通过")

    def test_data_pipeline_validation(self):
        """测试数据管道验证功能"""
        # 创建测试数据
        data = create_test_financial_data(50)

        # 验证数据基本属性
        self.assertGreater(len(data), 0)
        self.assertIn('timestamp', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)

        # 验证数据类型
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data['timestamp']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['close']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['volume']))

        # 验证数值范围
        self.assertTrue(data['close'].min() > 0)
        self.assertTrue(data['volume'].min() >= 0)

        # 验证无空值
        self.assertFalse(data.isnull().any().any())

        # 使用DataPipeline进行验证
        pipeline = DataPipeline()
        split = pipeline.split(data)

        # 验证拆分后数据仍然有效
        for split_data in [split.train, split.validation, split.test]:
            if len(split_data) > 0:
                self.assertFalse(split_data.isnull().any().any())
                self.assertTrue(split_data['close'].min() > 0)

        print("✅ 数据管道验证测试通过")

    def test_model_service_inference(self):
        """测试模型服务推理功能"""
        service = ModelService()

        try:
            # 尝试使用sklearn
            from sklearn.linear_model import LinearRegression

            # 创建训练数据
            X = np.random.rand(10, 2)
            y = np.random.rand(10)

            # 训练简单模型
            model = LinearRegression()
            model.fit(X, y)

            # 保存模型
            service.save_model("test_model", "v1.0", model, {"accuracy": 0.85})

            # 加载模型
            loaded_model = service.load_model("test_model", "v1.0")

            # 进行推理
            test_data = np.random.rand(5, 2)
            predictions = loaded_model.predict(test_data)

            # 验证推理结果
            self.assertEqual(len(predictions), 5)
            self.assertTrue(np.all(np.isfinite(predictions)))

            # 验证模型列表
            models = service.list_models()
            self.assertIn(("test_model", "v1.0"), models)

            print("✅ sklearn模型推理测试通过")

        except ImportError:
            # 使用Mock测试
            print("sklearn不可用，使用Mock推理测试")

            # 创建Mock模型
            class MockModel:
                def predict(self, data):
                    return [0.5] * len(data) if hasattr(data, '__len__') else [0.5]

            mock_model = MockModel()
            service.save_model("mock_model", "v1.0", mock_model)

            # 进行Mock推理
            loaded_model = service.load_model("mock_model", "v1.0")
            predictions = loaded_model.predict([1, 2, 3])

            self.assertEqual(len(predictions), 3)
            self.assertEqual(predictions[0], 0.5)

            print("✅ Mock模型推理测试通过")

    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 创建数据
        data = create_test_financial_data(50)

        # 数据处理
        pipeline = DataPipeline()
        split = pipeline.split(data)

        # 模型训练和推理
        service = ModelService()

        try:
            from sklearn.linear_model import LinearRegression

            # 使用训练数据训练模型
            X_train = split.train[['close', 'volume']].values
            y_train = split.train['close'].shift(-1).fillna(split.train['close']).values

            model = LinearRegression()
            model.fit(X_train, y_train)

            # 保存模型
            service.save_model("e2e_model", "v1.0", model)

            # 使用测试数据进行推理
            X_test = split.test[['close', 'volume']].values
            loaded_model = service.load_model("e2e_model", "v1.0")
            predictions = loaded_model.predict(X_test)

            # 验证结果
            self.assertEqual(len(predictions), len(split.test))
            self.assertTrue(np.all(np.isfinite(predictions)))

            print("✅ 端到端工作流测试通过")

        except ImportError:
            print("sklearn不可用，跳过端到端测试")
            self.skipTest("需要sklearn进行端到端测试")


if __name__ == '__main__':
    unittest.main(verbosity=2)
