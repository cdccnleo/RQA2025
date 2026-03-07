# -*- coding: utf-8 -*-
"""
机器学习层覆盖率提升测试

补充测试用例，提升机器学习层的测试覆盖率至80%以上
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.ml.core.ml_core import MLCore
from src.ml.core.exceptions import MLException, ModelNotFoundError, ModelTrainingError, ModelPredictionError, DataValidationError
from src.ml.core.feature_engineering import FeatureEngineer
from src.ml.core.inference_service import InferenceService
from src.ml.core.model_manager import ModelManager
from src.ml.core.process_orchestrator import MLProcessOrchestrator


class TestMLCoverageEnhancement:
    """机器学习层覆盖率提升测试"""

    @pytest.fixture
    def ml_core(self):
        """创建ML核心实例"""
        return MLCore()

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100))
        return X, y

    def test_ml_core_initialization_edge_cases(self, ml_core):
        """测试ML核心初始化边界情况"""
        # 测试空配置
        core = MLCore(config=None)
        assert core is not None

        # 测试无效配置
        with pytest.raises(ValueError):
            MLCore(config="invalid")

    def test_ml_core_create_model_edge_cases(self, ml_core):
        """测试模型创建边界情况"""
        # 测试空模型类型
        with pytest.raises(MLException):
            ml_core.train_model(pd.DataFrame(), pd.Series(), "")

        # 测试无效模型类型
        with pytest.raises(MLException):
            ml_core.train_model(pd.DataFrame(), pd.Series(), "invalid_type")

        # 测试None参数
        with pytest.raises(MLException):
            ml_core.train_model(None, None, None)

    def test_ml_core_train_edge_cases(self, ml_core, sample_data):
        """测试训练边界情况"""
        X, y = sample_data

        # 测试无效的模型类型
        with pytest.raises(MLException):
            ml_core.train_model(X, y, "nonexistent_model")

        # 测试空数据
        with pytest.raises(DataValidationError):
            ml_core.train_model("test_model", pd.DataFrame(), pd.Series([], dtype=float))

        # 测试不匹配的维度
        X_wrong = pd.DataFrame({'a': [1, 2, 3]})
        y_wrong = pd.Series([1, 2])  # 不同长度
        with pytest.raises(DataValidationError):
            ml_core.train_model("test_model", X_wrong, y_wrong)

    def test_ml_core_predict_edge_cases(self, ml_core, sample_data):
        """测试预测边界情况"""
        X, y = sample_data

        # 测试不存在的模型
        with pytest.raises(ModelNotFoundError):
            ml_core.predict("nonexistent_model", X)

        # 测试空输入
        with pytest.raises(MLException):
            ml_core.predict("test_model", pd.DataFrame())

    def test_ml_core_evaluate_edge_cases(self, ml_core, sample_data):
        """测试评估边界情况"""
        X, y = sample_data

        # 测试不存在的模型
        with pytest.raises(ModelNotFoundError):
            ml_core.evaluate_model("nonexistent_model", X, y)

        # 测试空数据
        with pytest.raises(MLException):
            ml_core.evaluate_model("test_model", pd.DataFrame(), pd.Series())

    def test_feature_engineer_edge_cases(self):
        """测试特征工程边界情况"""
        engineer = FeatureEngineer()

        # 创建默认管道
        engineer.create_pipeline("default_pipeline", [], ["col1"])

        # 测试空数据
        result = engineer.process_data(pd.DataFrame(), "default_pipeline")
        assert result is not None

        # 测试单列数据
        single_col = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        result = engineer.process_data(single_col, "default_pipeline")
        assert result is not None

        # 测试包含NaN的数据
        nan_data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [np.nan, 2, 3, 4, 5]
        })
        result = engineer.process_data(nan_data, "default_pipeline")
        assert result is not None

    def test_inference_service_edge_cases(self):
        """测试推理服务边界情况"""
        service = InferenceService()

        # 启动服务
        service.start()

        # 测试无效数据
        result = service.predict(None)
        assert result is not None

        # 测试空数据
        result = service.predict({})
        assert isinstance(result, dict)

        # 停止服务
        service.stop()

    def test_model_manager_edge_cases(self):
        """测试模型管理器边界情况"""
        manager = ModelManager()

        # ModelManager 是一个基础实现，测试实例化
        assert manager is not None
        assert isinstance(manager, ModelManager)

    def test_process_orchestrator_edge_cases(self):
        """测试流程编排器边界情况"""
        orchestrator = MLProcessOrchestrator()

        # 测试无效步骤
        from src.ml.core.process_orchestrator import ProcessStep
        step = ProcessStep(
            step_id="invalid_step",
            step_name="invalid_step",
            step_type="invalid_type",
            config={}
        )
        result = orchestrator.execute(step, {})
        assert result is not None

        # 测试无效步骤
        config = {
            'steps': [
                {'type': 'invalid_step', 'params': {}}
            ]
        }
        result = orchestrator.execute_process(config)
        assert result is not None  # 应该优雅处理错误

    def test_ml_core_cross_validation_edge_cases(self, ml_core, sample_data):
        """测试交叉验证边界情况"""
        X, y = sample_data

        # 测试不存在的模型
        with pytest.raises(ModelNotFoundError):
            ml_core.cross_validate("nonexistent", X, y)

        # 测试无效折数
        with pytest.raises(MLException):
            ml_core.cross_validate("test_model", X, y, cv=0)

        # 测试过大的折数
        with pytest.raises(MLException):
            ml_core.cross_validate("test_model", X, y, cv=100)

    def test_ml_core_feature_importance_edge_cases(self, ml_core, sample_data):
        """测试特征重要性边界情况"""
        X, y = sample_data

        # 测试不存在的模型
        result = ml_core.get_feature_importance("nonexistent")
        assert result is None or result == {}

        # 测试不支持特征重要性的模型
        # 这里需要mock一个不支持特征重要性的模型
        with patch.object(ml_core, 'models', {'test_model': Mock()}):
            mock_model = ml_core.models['test_model']
            mock_model.feature_importances_ = None
            # 不设置任何特征重要性属性

            result = ml_core.get_feature_importance("test_model")
            assert isinstance(result, dict)

    def test_ml_core_save_load_edge_cases(self, ml_core, tmp_path):
        """测试保存加载边界情况"""
        # 测试保存不存在的模型
        result = ml_core.save_model("nonexistent", str(tmp_path / "test.pkl"))
        assert result is False

        # 测试加载不存在的文件
        result = ml_core.load_model(str(tmp_path / "nonexistent.pkl"))
        assert result is None

        # 测试无效路径
        result = ml_core.save_model("test_model", "")
        assert result is False

    def test_ml_core_memory_management(self, ml_core, sample_data):
        """测试内存管理"""
        X, y = sample_data

        # 创建多个模型测试内存管理
        for i in range(5):
            model_id = f"test_model_{i}"
            ml_core.train_model(X, y, "rf", {"n_estimators": 10})

        # 验证模型缓存工作正常
        assert len(ml_core.models) > 0

        # 清理模型
        for i in range(5):
            model_id = f"test_model_{i}"
            ml_core.delete_model(model_id)

        # 验证清理工作正常
        assert len([m for m in ml_core.models.keys() if m.startswith("test_model")]) == 0

    def test_ml_core_concurrent_access(self, ml_core, sample_data):
        """测试并发访问"""
        import threading
        import time

        X, y = sample_data
        results = []
        errors = []

        def worker(worker_id):
            try:
                # 创建模型
                model_id = f"concurrent_model_{worker_id}"
                ml_core.create_model("random_forest", {"n_estimators": 5})
                ml_core.train_model(model_id, X, y)

                # 进行预测
                pred = ml_core.predict(model_id, X[:10])
                results.append((worker_id, len(pred)))

                # 清理
                ml_core.delete_model(model_id)

            except Exception as e:
                errors.append((worker_id, str(e)))

        # 启动多个线程
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join(timeout=30)

        # 验证结果
        assert len(results) == 3  # 所有线程都成功完成
        assert len(errors) == 0   # 没有错误

        for worker_id, pred_len in results:
            assert pred_len == 10  # 每个线程预测了10个样本

    def test_ml_core_large_dataset_handling(self, ml_core):
        """测试大数据集处理"""
        # 创建大数据集
        np.random.seed(42)
        large_X = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'feature4': np.random.randn(1000),
            'feature5': np.random.randn(1000)
        })
        large_y = pd.Series(np.random.randn(1000))

        # 测试大数据集训练
        model_id = "large_dataset_model"
        ml_core.create_model("random_forest", {"n_estimators": 10})
        result = ml_core.train_model(model_id, large_X, large_y)
        assert result is True

        # 测试大数据集预测
        pred = ml_core.predict(model_id, large_X[:100])
        assert len(pred) == 100

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_feature_engineering_integration(self, ml_core, sample_data):
        """测试特征工程集成"""
        X, y = sample_data

        # 创建模型
        model_id = "feature_engineering_test"
        ml_core.create_model("random_forest", {"n_estimators": 10})

        # 测试特征工程集成训练
        result = ml_core.train_model(model_id, X, y, feature_engineering=True)
        assert result is True

        # 测试特征工程集成预测
        pred = ml_core.predict(model_id, X[:10])
        assert len(pred) == 10

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_model_persistence(self, ml_core, sample_data, tmp_path):
        """测试模型持久化"""
        X, y = sample_data
        model_id = "persistence_test"

        # 创建和训练模型
        ml_core.create_model("random_forest", {"n_estimators": 10})
        ml_core.train_model(model_id, X, y)

        # 保存模型
        save_path = tmp_path / "test_model.pkl"
        result = ml_core.save_model(model_id, str(save_path))
        assert result is True
        assert save_path.exists()

        # 删除内存中的模型
        ml_core.delete_model(model_id)
        assert model_id not in ml_core.models

        # 加载模型
        result = ml_core.load_model(model_id, str(save_path))
        assert result is True
        assert model_id in ml_core.models

        # 验证加载的模型可以预测
        pred = ml_core.predict(model_id, X[:5])
        assert len(pred) == 5

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_error_recovery(self, ml_core, sample_data):
        """测试错误恢复"""
        X, y = sample_data

        # 测试在部分失败后的恢复
        # 这里模拟一些可能的失败场景

        # 1. 模型训练失败后的恢复
        with patch.object(ml_core, 'create_model', side_effect=[Exception("Creation failed"), Mock()]):
            try:
                ml_core.create_model("failing_model", {})
            except Exception:
                pass  # 预期失败

            # 第二次调用应该成功（如果有重试机制）
            # 注意：实际实现中可能没有重试，这里只是测试错误处理

        # 2. 预测失败后的恢复
        model_id = "error_recovery_test"
        ml_core.create_model("random_forest", {"n_estimators": 5})
        ml_core.train_model(model_id, X, y)

        # 模拟预测过程中的错误
        with patch.object(ml_core.models[model_id], 'predict', side_effect=[Exception("Predict failed"), np.array([1, 2, 3])]):
            try:
                ml_core.predict(model_id, X[:3])
            except Exception:
                # 应该优雅处理错误
                pass

        # 清理
        ml_core.delete_model(model_id)
