#!/usr/bin/env python3
import queue


import queue


"""
深度学习集成测试

测试模型服务、数据管道和训练系统的集成功能
    创建时间: 2025年2月
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import json
import unittest
import logging
from typing import Dict, Any
from datetime import datetime
import tempfile
import shutil

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from .deep_learning_manager import DeepLearningManager
    from .data_preprocessor import DataPreprocessor
    from .data_pipeline import (
        DataPipeline,
        DataBatch
    )
    from .model_service import ModelService, ModelServiceAPI
    from .distributed_trainer import DistributedTrainer
    print("✅ 深度学习模块导入成功")
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ 深度学习模块导入失败: {e}")
    IMPORT_SUCCESS = False
    # 创建Mock类以避免导入错误

    class MockModelService:
        pass

    class MockModelServiceAPI:
        pass

    class MockDistributedTrainer:
        pass
    ModelService = MockModelService
    ModelServiceAPI = MockModelServiceAPI
    DistributedTrainer = MockDistributedTrainer

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


class TestDataPipeline(unittest.TestCase):

    """数据管道测试"""

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.pipeline = None

    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop_pipeline()
        if hasattr(self, 'test_dir') and self.test_dir:
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_csv_data_source(self):
        """测试CSV数据源"""
        # 创建测试CSV文件
        test_data = {
            'timestamp': pd.date_range('2023 - 01 - 01', periods=100, freq='H'),
            'close': np.cumsum(np.random.randn(100) * 0.01 + 0.001),
            'volume': np.abs(np.random.randn(100)) * 1000 + 100
        }
        df = pd.DataFrame(test_data)
        csv_path = os.path.join(self.test_dir, 'test_data.csv')
        df.to_csv(csv_path, index=False)

        # 创建数据源
        source_config = {
            'type': 'csv',
            'file_path': csv_path,
            'chunk_size': 10
        }

        pipeline = DataPipeline({})
        data_source = pipeline.create_data_source(source_config)

        # 测试连接
        self.assertTrue(data_source.connect())

        # 测试读取数据
        batches = list(data_source.read_data(batch_size=10))
        self.assertGreater(len(batches), 0)

        # 验证数据
        first_batch = batches[0]
        self.assertEqual(len(first_batch.data), 10)
        self.assertIsNotNone(first_batch.metadata)

        data_source.disconnect()
        logger.info("✅ CSV数据源测试通过")

    def test_feature_engineering(self):
        """测试特征工程"""
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

        logger.info("✅ 特征工程测试通过")

    def test_data_validation(self):
        """测试数据验证"""
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

        # 允许有少量空值（测试数据清洗功能）
        null_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        self.assertLess(null_ratio, 0.1)  # 空值比例小于10%

        # 使用DataPipeline进行验证
        pipeline = DataPipeline()
        split = pipeline.split(data)

        # 验证拆分后数据仍然有效
        for split_data in [split.train, split.validation, split.test]:
            if len(split_data) > 0:
                # 允许有空值，但空值比例应该合理
                if len(split_data) > 0:
                    null_ratio = split_data.isnull().sum().sum() / (len(split_data) * len(split_data.columns))
                    self.assertLess(null_ratio, 0.1)  # 空值比例小于10%
                # 验证close价格为正数（非空值）
                valid_close = split_data['close'].dropna()
                if len(valid_close) > 0:
                    self.assertTrue(valid_close.min() > 0)

        logger.info("✅ 数据验证测试通过")


class TestModelService(unittest.TestCase):

    """模型服务测试"""

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.service = ModelService({"model_dir": self.test_dir})

    def tearDown(self):
        """测试后清理"""
        self.service.stop_service()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_model_registration(self):
        """测试模型注册"""
        # 创建一个简单的测试模型
        manager = DeepLearningManager(self.test_dir)

        # 创建LSTM模型
        model = manager.create_lstm_model(
            input_shape=(30, 5),
            output_units=1,
            model_name="test_model"
        )

        # 保存模型
        model_path = manager.save_model("test_model")

        # 注册模型
        version = self.service.register_model(
            model_name="test_model",
            model_path=model_path,
            metrics={'accuracy': 0.85}
        )

        self.assertIsNotNone(version)
        self.assertIn("test_model", self.service.model_versions)

        # 获取模型信息
        info = self.service.get_model_info("test_model")
        self.assertIsNotNone(info)
        self.assertEqual(info['model_name'], "test_model")

        logger.info("✅ 模型注册测试通过")

def test_model_inference_standalone():
    """独立的模型推理测试"""
    # 创建简单的测试模型（使用sklearn）
    try:
        from sklearn.linear_model import LinearRegression
        import numpy as np

        # 创建训练数据
        X = np.random.rand(10, 2)
        y = np.random.rand(10)

        # 训练简单模型
        model = LinearRegression()
        model.fit(X, y)

        # 保存模型 - 使用简单的ModelService
        from .model_service import ModelService as SimpleModelService
        service = SimpleModelService()
        service.save_model("test_model", "v1.0", model, {"accuracy": 0.85})

        # 加载模型
        loaded_model = service.load_model("test_model", "v1.0")

        # 进行推理
        test_data = np.random.rand(5, 2)
        predictions = loaded_model.predict(test_data)

        # 验证推理结果
        assert len(predictions) == 5
        assert np.all(np.isfinite(predictions))

        # 验证模型列表
        models = service.list_models()
        assert ("test_model", "v1.0") in models

        logger.info("✅ 模型推理测试通过")
        return True

    except ImportError:
        # 如果sklearn不可用，创建Mock测试
        logger.warning("sklearn不可用，使用Mock推理测试")
        # 创建Mock模型服务
        from .model_service import ModelService as SimpleModelService
        service = SimpleModelService()

        # 创建Mock模型
        class MockModel:
            def predict(self, data):
                return [0.5] * len(data) if hasattr(data, '__len__') else [0.5]

        mock_model = MockModel()
        service.save_model("mock_model", "v1.0", mock_model)

        # 进行Mock推理
        loaded_model = service.load_model("mock_model", "v1.0")
        predictions = loaded_model.predict([1, 2, 3])

        assert len(predictions) == 3
        assert predictions[0] == 0.5

        logger.info("✅ Mock模型推理测试通过")
        return True

    def test_service_statistics(self):
        """测试服务统计"""
        stats = self.service.get_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn('total_requests', stats)
        self.assertIn('successful_requests', stats)
        self.assertIn('failed_requests', stats)

        logger.info("✅ 服务统计测试通过")


class TestIntegration(unittest.TestCase):

    """集成测试"""

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.manager = DeepLearningManager({"model_dir": self.test_dir})
        self.preprocessor = DataPreprocessor()
        self.service = ModelService({"model_dir": self.test_dir})

    def tearDown(self):
        """测试后清理"""
        self.service.stop_service()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_end_to_end_pipeline(self):
        """测试端到端管道"""
        # 创建测试数据
        data = create_test_financial_data(200)
        data_path = os.path.join(self.test_dir, 'test_data.csv')
        data.to_csv(data_path, index=False)

        # 创建数据管道
        pipeline = create_financial_data_pipeline()
        source_config = {
            'type': 'csv',
            'file_path': data_path,
            'chunk_size': 50
        }
        data_source = pipeline.create_data_source(source_config)
        pipeline.set_data_source(data_source)

        # 启动管道
        self.assertTrue(pipeline.start_pipeline())

        try:
            # 处理数据
            batches = []
            for batch in pipeline.process_data_stream(queue.Queue(), max_batches=2):
                batches.append(batch)

                self.assertGreater(len(batches), 0)
            for batch in batches:
                self.assertIn('features_engineered', batch.metadata)

            logger.info(f"✅ 端到端管道测试通过，处理了 {len(batches)} 个批次")

        finally:
            pipeline.stop_pipeline()

    def test_model_training_and_serving(self):
        """测试模型训练和服务"""
        # 创建训练数据
        data = create_test_financial_data(300)

        # 预处理数据
        processed_data = self.preprocessor.preprocess_price_data(
            data,
            target_column='close',
            feature_columns=['open', 'high', 'low', 'close', 'volume']
        )

        model_name = "integration_test_model"
        training_history = self.manager.train_model(
            model_name=model_name,
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            X_val=processed_data['X_val'],
            y_val=processed_data['y_val'],
            epochs=2,
            batch_size=16
        )

        self.assertIsNotNone(training_history)
        self.assertIn('final_loss', training_history)

        # 保存模型
        model_path = self.manager.save_model(model_name)

        # 注册到服务
        version = self.service.register_model(
            model_name=model_name,
            model_path=model_path,
            metrics={'loss': training_history['final_loss']}
        )

        # 验证模型信息
        info = self.service.get_model_info(model_name)
        self.assertIsNotNone(info)
        self.assertEqual(info['model_name'], model_name)

        logger.info("✅ 模型训练和服务测试通过")


class PerformanceTest(unittest.TestCase):

    """性能测试"""

    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.service = ModelService(self.test_dir)

    def tearDown(self):
        """测试后清理"""
        self.service.stop_service()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_concurrent_requests(self):
        """测试并发请求"""
        import concurrent.futures

        # 创建测试数据
        test_data = np.random.randn(10, 30, 5)

        # 并发执行推理请求

        def make_request():

            try:
                # 这里会失败，因为没有注册模型，但可以测试错误处理
                result = self.service.predict("nonexistent_model", test_data)
                return result.get('status') == 'failed'  # 应该失败
            except:
                return True  # 异常也是预期的

            # 执行并发测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

            # 验证所有请求都得到了处理（即使失败）
            self.assertEqual(len(results), 10)
            self.assertTrue(all(results))

            logger.info("✅ 并发请求测试通过")

    def test_large_data_processing(self):
        """测试大数据处理"""
        # 创建大数据
        large_data = create_test_financial_data(1000)

        start_time = time.time()

        # 预处理大数据
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_price_data(
            large_data,
            target_column='close',
            feature_columns=['open', 'high', 'low', 'close', 'volume']
        )

        processing_time = time.time() - start_time

        # 验证处理结果
        self.assertIsNotNone(processed_data['X_train'])
        self.assertGreater(len(processed_data['feature_columns']), 0)

        # 性能要求：每1000样本处理时间不超过10秒
        self.assertLess(processing_time, 10.0)

        logger.info(f"✅ 大数据处理测试通过，处理时间: {processing_time:.2f}秒")


class TestSuite:

    """测试套件"""

    def __init__(self):

        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'test_details': []
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🧪 开始运行深度学习集成测试套件")

        # 创建测试加载器和运行器
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        # 添加测试类
        test_classes = [
            TestDataPipeline,
            TestModelService,
            TestIntegration,
            PerformanceTest
        ]

        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))

            # 运行测试
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)

            # 统计结果
            self.test_results['total_tests'] = result.testsRun
            self.test_results['passed_tests'] = result.testsRun - \
                len(result.failures) - len(result.errors)
            self.test_results['failed_tests'] = len(result.failures) + len(result.errors)
            self.test_results['skipped_tests'] = len(result.skipped)

            logger.info(
                f"测试完成: {self.test_results['passed_tests']}/{self.test_results['total_tests']} 通过")

            return self.test_results.copy()


def create_test_financial_data(n_samples: int = 500) -> pd.DataFrame:
    """创建测试金融数据"""
    np.random.seed(42)

    # 创建时间序列
    dates = pd.date_range(start='2023 - 01 - 01', periods=n_samples, freq='H')

    # 生成价格数据
    base_price = np.cumsum(np.random.randn(n_samples) * 0.01 + 0.001)
    base_price = (base_price - base_price.min()) / (base_price.max() - base_price.min())
    # 确保最小值为正数
    base_price = base_price + 1.0  # 价格从1.0开始

    # 创建数据框
    data = {
        'timestamp': dates,
        'close': base_price,
        'open': base_price + np.random.randn(n_samples) * 0.01,
        'high': base_price + np.random.rand(n_samples) * 0.02,
        'low': base_price - np.random.rand(n_samples) * 0.02,
        'volume': np.abs(np.random.randn(n_samples)) * 1000 + 100
    }

    # 添加一些NaN值来测试数据清洗
    mask = np.random.random(n_samples) < 0.05  # 5 % 的缺失值
    for col in ['close', 'volume']:
        data[col] = np.where(mask, np.nan, data[col])

        df = pd.DataFrame(data)
        return df


def generate_test_report(results: Dict[str, Any], output_path: str):
    """生成测试报告"""
    report = {
        'title': 'RQA2025深度学习集成测试报告',
        'generated_at': datetime.now().isoformat(),
        'test_results': results,
        'summary': {
            'total_tests': results['total_tests'],
            'passed_tests': results['passed_tests'],
            'failed_tests': results['failed_tests'],
            'skipped_tests': results['skipped_tests'],
            'pass_rate': results['passed_tests'] / max(results['total_tests'], 1)
        },
        'recommendations': []
    }

    # 生成建议
    if results['failed_tests'] > 0:
        report['recommendations'].append("修复失败的测试用例")
    if results['skipped_tests'] > 0:
        report['recommendations'].append("实现跳过的测试用例")
    if results['passed_tests'] < results['total_tests'] * 0.95:
        report['recommendations'].append("提高测试覆盖率和质量")

    # 保存报告
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf - 8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"测试报告已生成: {output_path}")


def main():
    """主函数 - 集成测试演示"""
    print("🧪 RQA2025深度学习集成测试")
    print("="*50)

    # 创建测试套件
    test_suite = TestSuite()

    # 运行所有测试
    results = test_suite.run_all_tests()

    # 显示结果
    print("\n📊 测试结果汇总:")
    print(f"   总测试数: {results['total_tests']}")
    print(f"   通过测试: {results['passed_tests']}")
    print(f"   失败测试: {results['failed_tests']}")
    print(f"   跳过测试: {results['skipped_tests']}")
    print(f"   成功率: {results.get('success_rate', 0):.1f}%")
    # 生成测试报告
    report_path = "models / deep_learning / reports / integration_test_report.json"
    generate_test_report(results, report_path)

    print(f"\n📋 测试报告已生成: {report_path}")

    if results['failed_tests'] == 0 and results['pass_rate'] >= 0.95:
        print("\n🎉 所有集成测试通过！系统集成质量达标")
        return True
    else:
        print("\n⚠️ 部分测试未通过，需要进一步优化")
        return False

    if __name__ == "__main__":
        success = main()
        sys.exit(0 if success else 1)
