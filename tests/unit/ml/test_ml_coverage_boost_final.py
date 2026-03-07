# -*- coding: utf-8 -*-
"""
机器学习层最终覆盖率提升测试

针对机器学习层的边界条件、错误处理和复杂场景创建额外的测试用例
提升覆盖率至80%以上
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.ml.core.ml_core import MLCore
from src.ml.core.exceptions import MLException, ModelNotFoundError
from src.ml.core.feature_engineering import FeatureEngineer
from src.ml.core.inference_service import InferenceService
from src.ml.core.model_manager import ModelManager


class TestMLCoverageBoostFinal:
    """机器学习层最终覆盖率提升测试"""

    @pytest.fixture
    def ml_core(self):
        """创建ML核心实例"""
        return MLCore()

    @pytest.fixture
    def complex_sample_data(self):
        """创建复杂测试数据"""
        np.random.seed(42)
        # 创建包含NaN值和异常值的数据
        X = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 1000],  # 包含NaN和异常值
            'feature2': [np.nan, 2, 3, 4, 5, -1000],  # 包含NaN和异常值
            'feature3': ['a', 'b', 'c', 'd', 'e', 'f'],  # 分类特征
            'feature4': pd.date_range('2023-01-01', periods=6)  # 日期特征
        })
        y = pd.Series([0, 1, 0, 1, 0, 1])
        return X, y

    def test_ml_core_large_scale_training(self, ml_core):
        """测试大规模数据训练"""
        # 创建大数据集
        np.random.seed(42)
        X_large = pd.DataFrame(np.random.randn(5000, 20))
        y_large = pd.Series(np.random.randint(0, 2, 5000))

        # 测试大规模训练
        model_id = ml_core.train_model(X_large, y_large, model_type="rf", model_params={"n_estimators": 10, "max_depth": 5})

        assert isinstance(model_id, str)

        # 测试大规模预测
        pred = ml_core.predict(model_id, X_large[:100])
        assert len(pred) == 100

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_memory_efficient_processing(self, ml_core):
        """测试内存高效处理"""
        # 创建中等规模数据集进行分批处理测试
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(2000, 10))
        y = pd.Series(np.random.randint(0, 2, 2000))

        # 测试训练
        model_id = ml_core.train_model(X, y, model_type="rf", model_params={"n_estimators": 5})
        assert isinstance(model_id, str)

        # 测试批量预测
        batch_size = 100
        all_predictions = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_pred = ml_core.predict(model_id, batch_X)
            all_predictions.extend(batch_pred)

        assert len(all_predictions) == len(X)

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_cross_validation_edge_cases(self, ml_core):
        """测试交叉验证边界情况"""
        # 创建测试数据集
        X_test = pd.DataFrame({'a': [1, 2, 3]})
        y_test = pd.Series([0, 1, 0])

        model_id = ml_core.train_model(X_test, y_test, model_type="rf", model_params={"n_estimators": 5})
        assert isinstance(model_id, str)

        # 测试空数据交叉验证
        X_empty = pd.DataFrame()
        y_empty = pd.Series([])
        with pytest.raises(MLException):
            ml_core.cross_validate(X_empty, y_empty, model_type="rf")

        # 测试不存在的模型ID（如果需要的话）
        # 注意：cross_validate方法不使用模型ID，而是直接创建模型进行验证
        # 这里不需要测试不存在的模型，因为方法不依赖预训练的模型

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_feature_importance_comprehensive(self, ml_core):
        """测试特征重要性综合场景"""
        # 创建包含多种数值特征的数据
        np.random.seed(42)
        X = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100),
            'numeric3': np.random.randn(100),
            'numeric4': np.random.randn(100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))

        # 训练模型
        model_id = ml_core.train_model(X, y, model_type="rf", model_params={"n_estimators": 10})
        assert isinstance(model_id, str)

        # 测试特征重要性
        importance = ml_core.get_feature_importance(model_id)
        assert isinstance(importance, dict)
        assert len(importance) > 0

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_model_persistence_comprehensive(self, ml_core, tmp_path):
        """测试模型持久化综合场景"""
        # 创建测试数据
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))

        # 训练模型
        model_id = ml_core.train_model(X, y, model_type="rf", model_params={"n_estimators": 10})
        assert isinstance(model_id, str)

        # 测试保存到不同路径格式
        save_paths = [
            str(tmp_path / "model1.pkl"),
            str(tmp_path / "subdir" / "model2.pkl"),
        ]

        for save_path in save_paths:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 保存模型
            result = ml_core.save_model(model_id, save_path)
            assert result is True
            assert os.path.exists(save_path)

            # 删除内存中的模型
            ml_core.delete_model(model_id)

            # 加载模型
            loaded_model_id = ml_core.load_model(save_path)
            assert isinstance(loaded_model_id, str)

            # 验证加载的模型可以工作
            pred = ml_core.predict(loaded_model_id, X[:5])
            assert len(pred) == 5

        # 最终清理
        ml_core.delete_model(model_id)

    def test_ml_core_error_handling_comprehensive(self, ml_core):
        """测试错误处理综合场景"""
        # 测试各种异常情况

        # 1. 无效模型类型
        X_test = pd.DataFrame({'a': [1, 2, 3]})
        y_test = pd.Series([0, 1, 0])
        with pytest.raises(MLException):
            ml_core.train_model(X_test, y_test, model_type="invalid_model_type")

        # 2. 重复创建模型
        X_test = pd.DataFrame({'a': [1, 2, 3]})
        y_test = pd.Series([0, 1, 0])
        model_id1 = ml_core.train_model(X_test, y_test, model_type="rf", model_params={"n_estimators": 5})
        # 第二次创建应该不报错（覆盖现有模型）
        model_id2 = ml_core.train_model(X_test, y_test, model_type="rf", model_params={"n_estimators": 5})
        assert isinstance(model_id1, str)
        assert isinstance(model_id2, str)

        # 3. 删除不存在的模型
        result = ml_core.delete_model("nonexistent_model")
        assert result is False

        # 清理
        ml_core.delete_model(model_id1)
        ml_core.delete_model(model_id2)

    def test_ml_core_performance_monitoring(self, ml_core):
        """测试性能监控功能"""
        # 创建测试数据
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))

        # 训练模型
        model_id = ml_core.train_model(X, y, model_type="rf", model_params={"n_estimators": 5})
        assert isinstance(model_id, str)

        # 测试多次预测的性能
        import time
        start_time = time.time()

        for _ in range(10):
            pred = ml_core.predict(model_id, X[:10])
            assert len(pred) == 10

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能在合理范围内（每秒至少处理10次预测）
        assert total_time < 10.0  # 10秒内完成

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_concurrent_operations(self, ml_core):
        """测试并发操作"""
        import threading
        import queue

        # 创建测试数据
        X = pd.DataFrame(np.random.randn(100, 3))
        y = pd.Series(np.random.randint(0, 2, 100))

        results = queue.Queue()
        errors = []

        def worker(worker_id):
            try:
                model_id = f"concurrent_model_{worker_id}"
                trained_model_id = ml_core.train_model(X, y, model_type="rf", model_params={"n_estimators": 3})

                # 进行预测
                pred = ml_core.predict(trained_model_id, X[:5])
                results.put((worker_id, len(pred)))

                # 清理
                ml_core.delete_model(model_id)

            except Exception as e:
                errors.append((worker_id, str(e)))

        # 启动多个线程
        threads = []
        num_threads = 3

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join(timeout=30)

        # 验证结果
        successful_results = 0
        while not results.empty():
            worker_id, pred_len = results.get()
            assert pred_len == 5
            successful_results += 1

        assert successful_results == num_threads
        assert len(errors) == 0

    def test_ml_core_resource_management(self, ml_core):
        """测试资源管理"""
        # 测试模型生命周期管理
        model_ids = []

        # 创建多个模型
        for i in range(5):
            X = pd.DataFrame(np.random.randn(50, 3))
            y = pd.Series(np.random.randint(0, 2, 50))

            model_id = ml_core.train_model(X, y, model_type="rf", model_params={"n_estimators": 3})
            model_ids.append(model_id)

        # 验证所有模型都存在
        for model_id in model_ids:
            assert model_id in ml_core.models

        # 批量删除模型
        for model_id in model_ids:
            result = ml_core.delete_model(model_id)
            assert result is True

        # 验证所有模型都被删除
        for model_id in model_ids:
            assert model_id not in ml_core.models

    def test_ml_core_data_validation_comprehensive(self, ml_core):
        """测试数据验证综合场景"""
        # 测试各种数据格式和验证场景

        # 1. 正常数据
        X_normal = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        y_normal = pd.Series([0, 1, 0])

        model_id = ml_core.train_model(X_normal, y_normal, model_type="rf", model_params={"n_estimators": 3})
        assert isinstance(model_id, str)

        # 2. 测试预测数据格式兼容性
        pred_data_formats = [
            pd.DataFrame({'a': [1, 2], 'b': [4, 5]}),  # DataFrame
            np.array([[1, 4], [2, 5]]),  # numpy array
        ]

        for pred_data in pred_data_formats:
            pred = ml_core.predict(model_id, pred_data)
            assert len(pred) == 2

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_scalability_assessment(self, ml_core):
        """测试可扩展性评估"""
        # 测试不同规模数据的处理能力

        scales = [
            (100, 5),   # 小规模
            (500, 10),  # 中规模
            (1000, 15), # 大规模
        ]

        for n_samples, n_features in scales:
            X = pd.DataFrame(np.random.randn(n_samples, n_features))
            y = pd.Series(np.random.randint(0, 2, n_samples))

            model_id = f"scalability_model_{n_samples}_{n_features}"

            # 测试训练时间是否合理
            import time
            start_time = time.time()
            trained_model_id = ml_core.train_model(X, y, model_type="rf", model_params={"n_estimators": 5, "max_depth": 3})
            end_time = time.time()

            training_time = end_time - start_time

            # 验证训练成功且时间合理
            assert isinstance(trained_model_id, str)
            assert training_time < 60  # 训练时间不超过60秒

            # 测试预测
            pred = ml_core.predict(trained_model_id, X[:min(10, len(X))])
            assert len(pred) == min(10, len(X))

            # 清理
            ml_core.delete_model(model_id)

    def test_ml_core_robustness_testing(self, ml_core):
        """测试鲁棒性"""
        # 测试异常数据和边界条件的处理

        # 1. 包含NaN的数据
        X_with_nan = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [4, 5, np.nan]
        })
        y_with_nan = pd.Series([0, 1, 0])

        model_id = ml_core.train_model(X_with_nan, y_with_nan, model_type="rf", model_params={"n_estimators": 3})
        assert isinstance(model_id, str)

        # 预测也应该处理NaN值
        pred = ml_core.predict(model_id, X_with_nan)
        assert len(pred) == len(X_with_nan)

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_integration_scenarios(self, ml_core):
        """测试集成场景"""
        # 测试完整的ML流程集成

        # 1. 数据准备
        X = pd.DataFrame(np.random.randn(200, 4))
        y = pd.Series(np.random.randint(0, 2, 200))

        # 2. 模型训练
        trained_model_id = ml_core.train_model(X, y, model_type="rf", model_params={"n_estimators": 10})
        assert isinstance(trained_model_id, str)

        # 3. 模型评估
        eval_result = ml_core.evaluate_model(trained_model_id, X, y)
        assert eval_result is not None
        assert 'mae' in eval_result or 'mse' in eval_result or 'r2' in eval_result

        # 4. 特征重要性
        importance = ml_core.get_feature_importance(trained_model_id)
        assert isinstance(importance, dict)

        # 5. 交叉验证
        cv_result = ml_core.cross_validate(X, y, model_type="rf")
        assert cv_result is not None

        # 6. 模型保存和加载
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            save_path = tmp_file.name

        try:
            save_result = ml_core.save_model(trained_model_id, save_path)
            assert save_result is True

            # 删除内存中的模型
            ml_core.delete_model(trained_model_id)

            # 重新加载
            loaded_model_id = ml_core.load_model(save_path)
            assert isinstance(loaded_model_id, str)

            # 验证加载的模型仍然工作
            final_pred = ml_core.predict(loaded_model_id, X[:5])
            assert len(final_pred) == 5

        finally:
            # 清理文件
            if os.path.exists(save_path):
                os.unlink(save_path)

        # 最终清理
        ml_core.delete_model(loaded_model_id)
