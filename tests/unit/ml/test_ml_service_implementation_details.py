#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML服务实现细节测试

测试MLService的内部实现细节，包括默认组件和私有方法
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.ml.core.ml_service import (
    MLService, MLServiceStatus,
    _DefaultFeatureEngineering, _DefaultModelManager, _DefaultInferenceService
)


class TestMLServiceImplementationDetails:
    """ML服务实现细节测试"""

    def test_default_feature_engineering_extract_features(self):
        """测试默认特征工程的特征提取"""
        fe = _DefaultFeatureEngineering()

        # 测试DataFrame输入
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })

        features = fe.extract_features(df, {})
        assert features.features['feature1'] == 1.0
        assert features.features['feature2'] == 4.0
        assert features.symbol == "default"

    def test_default_feature_engineering_preprocessing(self):
        """测试默认特征工程的预处理"""
        fe = _DefaultFeatureEngineering()

        from src.ml.core.ml_service import MLFeatures
        features = MLFeatures(
            timestamp=pd.Timestamp.now(),
            symbol="test",
            features={'f1': 1.0, 'f2': 2.0}
        )

        # 测试预处理
        processed = fe.preprocess_features(features, {})
        assert processed.features == features.features

    def test_default_feature_engineering_selection(self):
        """测试默认特征工程的特征选择"""
        fe = _DefaultFeatureEngineering()

        from src.ml.core.ml_service import MLFeatures
        features = MLFeatures(
            timestamp=pd.Timestamp.now(),
            symbol="test",
            features={'f1': 1.0, 'f2': 2.0, 'f3': 3.0}
        )

        # 测试特征选择
        selected = fe.select_features(features, {})
        assert selected.features == features.features

    def test_default_feature_engineering_process(self):
        """测试默认特征工程的process方法"""
        fe = _DefaultFeatureEngineering()

        from src.ml.core.ml_service import MLFeatures
        features = MLFeatures(
            timestamp=pd.Timestamp.now(),
            symbol="test",
            features={'f1': 1.0, 'f2': 2.0}
        )

        # 测试process方法
        processed = fe.process(features)
        assert processed == features

    def test_default_model_manager_operations(self):
        """测试默认模型管理器的操作"""
        mm = _DefaultModelManager()

        # 测试列出模型
        models = mm.list_models()
        assert models == []

        # 测试获取模型信息
        info = mm.get_model_info("nonexistent")
        assert info is None

    def test_default_inference_service_operations(self):
        """测试默认推理服务的操作"""
        inf = _DefaultInferenceService()

        # 测试启动
        result = inf.start()
        assert result is True

        # 测试异步启动
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(inf.start_async())
            assert result is True
        finally:
            loop.close()

        # 测试停止
        result = inf.stop()
        assert result is True

        # 测试预测（应该抛出异常）
        with pytest.raises(RuntimeError, match="Inference service not configured"):
            inf.predict(None)

    def test_ml_service_private_model_creation_methods(self):
        """测试MLService的私有模型创建方法"""
        service = MLService()

        # 测试线性回归模型创建
        config = {"params": {}}
        model = service._create_linear_regression_model(config)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

        # 测试随机森林模型创建
        model = service._create_random_forest_model(config)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

        # 测试XGBoost模型创建
        model = service._create_xgboost_model(config)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_ml_service_private_training_method(self):
        """测试MLService的私有训练方法"""
        service = MLService()

        # 创建测试数据
        X = np.random.randn(20, 2)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(20) * 0.1
        training_data = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'target': y
        })

        # 创建模型
        model = service._create_linear_regression_model({})

        # 测试训练
        result = service._train_model_instance(model, training_data, {})
        assert result['status'] == 'completed'
        assert 'performance' in result
        assert 'mse' in result['performance']
        assert 'rmse' in result['performance']
        assert 'r2_score' in result['performance']

    def test_ml_service_private_prediction_method(self):
        """测试MLService的私有预测方法"""
        service = MLService()

        # 创建并训练模型
        model = service._create_linear_regression_model({})
        X = np.random.randn(20, 2)
        y = X[:, 0] + X[:, 1] + np.random.randn(20) * 0.1
        model.fit(X, y)

        from src.ml.core.ml_service import MLFeatures
        features = MLFeatures(
            timestamp=pd.Timestamp.now(),
            symbol="test",
            features={'feature1': 1.0, 'feature2': 2.0}
        )

        # 测试预测
        prediction = service._predict_with_model(model, features)
        assert isinstance(prediction, (int, float))

    def test_ml_service_private_hyperparameter_generation(self):
        """测试MLService的私有超参数生成方法"""
        service = MLService()

        # 测试参数组合生成
        param_space = {
            'alpha': [0.01, 0.1, 1.0],
            'max_iter': [100, 200]
        }

        combinations = service._generate_param_combinations(param_space)
        assert len(combinations) > 0
        assert isinstance(combinations[0], dict)

        # 测试空参数空间
        empty_combinations = service._generate_param_combinations({})
        assert len(empty_combinations) > 0  # 应该返回默认组合

    def test_ml_service_private_performance_evaluation(self):
        """测试MLService的私有性能评估方法"""
        service = MLService()

        # 创建模型和数据
        model = service._create_linear_regression_model({})
        X = np.random.randn(30, 2)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(30) * 0.1
        model.fit(X, y)

        validation_data = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'target': y
        })

        # 测试性能评估
        score = service._evaluate_model_performance("test_model", validation_data)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # R²分数应该在0-1范围内

    def test_ml_service_initialization_with_config(self):
        """测试MLService使用不同配置的初始化"""
        # 测试空配置
        service1 = MLService({})
        assert service1.status == MLServiceStatus.STOPPED
        assert service1.max_workers == 4

        # 测试自定义配置
        config = {
            "max_workers": 8,
            "custom_param": "value"
        }
        service2 = MLService(config)
        assert service2.max_workers == 8
        assert service2.config["custom_param"] == "value"

        # 测试None配置
        service3 = MLService(None)
        assert service3.config is not None
        assert service3.status == MLServiceStatus.STOPPED

    def test_ml_service_stats_tracking(self):
        """测试MLService的统计信息跟踪"""
        service = MLService()
        service.start()

        initial_stats = service.stats.copy()

        # 创建训练数据并训练模型
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = X[:, 0] + X[:, 1] + np.random.randn(20) * 0.1
        training_data = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'target': y
        })

        # 训练模型
        service.train_model("test_model", training_data, {"algorithm": "linear_regression"})

        # 执行预测
        service.predict(pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]}))

        # 验证统计信息已更新
        assert service.stats["training_sessions"] == initial_stats["training_sessions"] + 1
        assert service.stats["inference_requests"] == initial_stats["inference_requests"] + 1

        service.stop()

    def test_ml_service_error_handling_in_private_methods(self):
        """测试MLService私有方法中的错误处理"""
        service = MLService()

        # 测试使用无效数据训练模型
        invalid_data = "not_a_dataframe"
        model = service._create_linear_regression_model({})

        result = service._train_model_instance(model, invalid_data, {})
        assert result['status'] == 'failed'
        assert 'error' in result

        # 测试使用无效数据进行性能评估
        score = service._evaluate_model_performance("test_model", invalid_data)
        assert score == 0.0  # 应该返回默认分数

    def test_ml_service_model_retrieval_edge_cases(self):
        """测试MLService模型检索的边界情况"""
        service = MLService()

        # 测试获取不存在模型的信息
        info = service.get_model_info("nonexistent")
        assert info is None

        # 测试获取不存在模型的性能
        performance = service.get_model_performance("nonexistent")
        assert performance is None

        # 测试卸载不存在的模型
        result = service.unload_model("nonexistent")
        assert result is False

    def test_ml_service_batch_prediction_edge_cases(self):
        """测试MLService批量预测的边界情况"""
        service = MLService()
        service.start()

        # 测试空请求列表
        responses = service.predict_batch([])
        assert responses == []

        # 测试无效模型的批量预测
        from src.ml.core.ml_service import MLInferenceRequest, MLFeatures

        invalid_request = MLInferenceRequest(
            request_id="test",
            model_id="nonexistent_model",
            features=MLFeatures(
                timestamp=pd.Timestamp.now(),
                symbol="test",
                features={'f1': 1.0}
            )
        )

        responses = service.predict_batch([invalid_request])
        assert len(responses) == 1
        assert responses[0].success is False
        assert "未找到" in responses[0].error_message

        service.stop()

    def test_ml_service_hyperparameter_optimization_edge_cases(self):
        """测试MLService超参数优化的边界情况"""
        service = MLService()

        # 测试空参数空间和空数据（应该返回错误）
        empty_df = pd.DataFrame()
        result = service.optimize_hyperparameters("test", {}, empty_df)
        assert 'error' in result

        # 测试无效训练数据
        param_space = {'alpha': [0.1, 1.0]}
        result = service.optimize_hyperparameters("test", param_space, "invalid_data")
        assert 'error' in result
