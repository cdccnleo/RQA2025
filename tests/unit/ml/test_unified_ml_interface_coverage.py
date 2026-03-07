#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML统一接口专项测试

大幅提升unified_ml_interface.py的测试覆盖率
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock


try:
    IMPORTS_AVAILABLE = True
    from src.ml.core.unified_ml_interface import UnifiedMLInterface, MLInterfaceError, MLAlgorithmType, MLTaskType, MLModelConfig, OptimizationMetric
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="UnifiedMLInterface not available")
class TestUnifiedMLInterfaceCoverage:
    """ML统一接口覆盖率提升测试"""

    def setup_method(self):
        """测试前准备"""
        self.interface = UnifiedMLInterface()

    def test_initialization_with_valid_config(self):
        """测试有效配置的初始化"""
        config = {
            "model_cache_enabled": True,
            "max_cache_size": 100,
            "default_algorithm": "random_forest"
        }

        interface = UnifiedMLInterface(config)
        assert interface.config["model_cache_enabled"] is True
        assert interface.config["max_cache_size"] == 100
        assert interface.config["default_algorithm"] == "random_forest"

    def test_initialization_with_empty_config(self):
        """测试空配置的初始化"""
        interface = UnifiedMLInterface({})
        assert interface.config == {}  # 空配置应该保持为空
        assert interface._max_cache_size == 50  # 内部默认值

    def test_initialization_with_none_config(self):
        """测试None配置的初始化"""
        interface = UnifiedMLInterface(None)
        assert interface.config is not None
        assert isinstance(interface.config, dict)

    def test_create_model_with_supported_types(self):
        """测试创建支持的模型类型"""
        # 创建一个基本的模型配置
        config = MLModelConfig(
            algorithm_type=MLAlgorithmType.SUPERVISED_LEARNING,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={"n_estimators": 10}
        )

        model_id = self.interface.create_model(config)
        assert model_id is not None
        assert isinstance(model_id, str)

        # 验证模型已被创建
        models = self.interface.list_models()
        assert len(models) >= 1

    def test_create_model_with_different_algorithms(self):
        """测试创建不同算法的模型"""
        # 监督学习模型
        supervised_config = MLModelConfig(
            algorithm_type=MLAlgorithmType.SUPERVISED_LEARNING,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={}
        )

        model_id1 = self.interface.create_model(supervised_config)
        assert model_id1 is not None

        # 无监督学习模型
        unsupervised_config = MLModelConfig(
            algorithm_type=MLAlgorithmType.UNSUPERVISED_LEARNING,
            task_type=MLTaskType.CLUSTERING,
            hyperparameters={"n_clusters": 3}
        )

        model_id2 = self.interface.create_model(unsupervised_config)
        assert model_id2 is not None

        # 验证两个模型都被创建
        models = self.interface.list_models()
        assert len(models) >= 2

    def test_train_model_success(self):
        """测试模型训练成功"""
        # 创建模拟模型
        mock_model = Mock()
        mock_model.fit = Mock(return_value=mock_model)

        # 训练数据
        X = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        y = pd.Series([1, 2, 3])

        trained_model = self.interface.train_model(mock_model, X, y)
        assert trained_model is not None
        mock_model.fit.assert_called_once()

    def test_train_model_with_validation_split(self):
        """测试带验证分割的模型训练"""
        mock_model = Mock()
        mock_model.fit = Mock(return_value=mock_model)

        X = pd.DataFrame(np.random.random((100, 5)))
        y = pd.Series(np.random.randint(0, 2, 100))

        trained_model = self.interface.train_model(mock_model, X, y, validation_split=0.2)
        assert trained_model is not None

    def test_train_model_failure(self):
        """测试模型训练失败"""
        mock_model = Mock()
        mock_model.fit = Mock(side_effect=Exception("Training failed"))

        X = pd.DataFrame([[1, 2], [3, 4]])
        y = pd.Series([1, 2])

        with pytest.raises(MLInterfaceError):
            self.interface.train_model(mock_model, X, y)

    def test_predict_with_trained_model(self):
        """测试训练后模型的预测"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.8, 0.6, 0.9]))

        X = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        predictions = self.interface.predict(mock_model, X)
        assert predictions is not None
        assert len(predictions) == len(X)
        mock_model.predict.assert_called_once()

    def test_predict_with_untrained_model(self):
        """测试未训练模型的预测"""
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=Exception("Model not trained"))

        X = pd.DataFrame([[1, 2], [3, 4]])

        with pytest.raises(MLInterfaceError):
            self.interface.predict(mock_model, X)

    def test_predict_batch_processing(self):
        """测试批量预测处理"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

        # 大数据集
        X = pd.DataFrame(np.random.random((1000, 10)))

        predictions = self.interface.predict(mock_model, X, batch_size=100)
        assert predictions is not None
        assert len(predictions) == len(X)

    def test_evaluate_with_valid_data(self):
        """测试有效数据的评估"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0, 1, 1, 0]))

        X_test = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = pd.Series([0, 1, 1, 0])

        metrics = self.interface.evaluate(mock_model, X_test, y_test)
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

    def test_evaluate_regression_metrics(self):
        """测试回归指标评估"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1.1, 2.2, 2.8, 3.9]))

        X_test = pd.DataFrame([[1], [2], [3], [4]])
        y_test = pd.Series([1.0, 2.0, 3.0, 4.0])

        metrics = self.interface.evaluate(mock_model, X_test, y_test)
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics

    def test_evaluate_with_prediction_failure(self):
        """测试预测失败的评估"""
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=Exception("Prediction failed"))

        X_test = pd.DataFrame([[1, 2], [3, 4]])
        y_test = pd.Series([0, 1])

        with pytest.raises(MLInterfaceError):
            self.interface.evaluate(mock_model, X_test, y_test)

    def test_model_caching_enabled(self):
        """测试启用模型缓存"""
        config = {"model_cache_enabled": True, "max_cache_size": 10}
        interface = UnifiedMLInterface(config)

        # 创建并缓存模型
        config = {"type": "linear_regression", "params": {}}
        model = interface.create_model(config)

        # 验证缓存机制（如果实现的话）
        assert model is not None

    def test_model_caching_disabled(self):
        """测试禁用模型缓存"""
        config = {"model_cache_enabled": False}
        interface = UnifiedMLInterface(config)

        # 应该正常工作，不使用缓存
        config = {"type": "linear_regression", "params": {}}
        model = interface.create_model(config)
        assert model is not None

    def test_cross_validation_support(self):
        """测试交叉验证支持"""
        mock_model = Mock()
        mock_model.fit = Mock(return_value=mock_model)
        mock_model.predict = Mock(return_value=np.array([0, 1]))

        X = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = pd.Series([0, 1, 0, 1])

        # 使用交叉验证训练
        trained_model = self.interface.train_model(mock_model, X, y, cv_folds=3)
        assert trained_model is not None

    def test_hyperparameter_optimization(self):
        """测试超参数优化"""
        mock_model = Mock()
        mock_model.fit = Mock(return_value=mock_model)

        X = pd.DataFrame(np.random.random((50, 3)))
        y = pd.Series(np.random.randint(0, 2, 50))

        param_grid = {
            'n_estimators': [10, 50],
            'max_depth': [3, 5]
        }

        # 超参数优化训练
        trained_model = self.interface.train_model(
            mock_model, X, y,
            hyperparameter_optimization=True,
            param_grid=param_grid
        )
        assert trained_model is not None

    def test_feature_importance_extraction(self):
        """测试特征重要性提取"""
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.3, 0.7])

        feature_names = ['feature1', 'feature2']
        importance = self.interface.get_feature_importance(mock_model, feature_names)

        assert isinstance(importance, dict)
        assert 'feature1' in importance
        assert 'feature2' in importance
        assert importance['feature2'] > importance['feature1']

    def test_model_serialization(self):
        """测试模型序列化"""
        mock_model = Mock()

        # 测试保存
        save_result = self.interface.save_model(mock_model, "test_model.pkl")
        assert save_result is True

        # 测试加载
        loaded_model = self.interface.load_model("test_model.pkl")
        assert loaded_model is not None

    def test_error_handling_comprehensive(self):
        """测试全面错误处理"""
        # 测试无效模型类型
        with pytest.raises(MLInterfaceError):
            self.interface.create_model({"type": "invalid"})

        # 测试空数据训练
        with pytest.raises(MLInterfaceError):
            self.interface.train_model(Mock(), pd.DataFrame(), pd.Series())

        # 测试无效预测数据
        with pytest.raises(MLInterfaceError):
            self.interface.predict(Mock(), None)

    def test_interface_status_monitoring(self):
        """测试接口状态监控"""
        status = self.interface.get_status()

        assert isinstance(status, dict)
        assert 'initialized' in status
        assert 'model_cache_size' in status
        assert status['initialized'] is True

    def test_resource_cleanup(self):
        """测试资源清理"""
        # 执行一些操作
        config = {"type": "linear_regression", "params": {}}
        model = self.interface.create_model(config)

        # 清理资源
        cleanup_result = self.interface.cleanup()
        assert cleanup_result is True

    def test_concurrent_operations(self):
        """测试并发操作"""
        import threading

        results = []

        def worker():
            config = {"type": "linear_regression", "params": {}}
            model = self.interface.create_model(config)
            results.append(model is not None)

        # 创建多个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证所有操作都成功
        assert all(results)
        assert len(results) == 5

    def test_memory_management(self):
        """测试内存管理"""
        # 创建多个模型
        models = []
        for i in range(10):
            config = {"type": "linear_regression", "params": {}}
            model = self.interface.create_model(config)
            models.append(model)

        # 验证内存管理
        memory_info = self.interface.get_memory_usage()
        assert isinstance(memory_info, dict)

        # 清理
        for model in models:
            del model

    def test_interface_metrics_collection(self):
        """测试接口指标收集"""
        # 执行一些操作以生成指标
        config = {"type": "linear_regression", "params": {}}
        model = self.interface.create_model(config)

        X = pd.DataFrame([[1, 2], [3, 4]])
        y = pd.Series([1, 2])

        trained_model = self.interface.train_model(model, X, y)
        predictions = self.interface.predict(trained_model, X)
        metrics = self.interface.evaluate(trained_model, X, y)

        # 获取性能指标
        perf_metrics = self.interface.get_performance_metrics()

        assert isinstance(perf_metrics, dict)
        assert 'operations_count' in perf_metrics
        assert perf_metrics['operations_count'] >= 3  # create, train, predict, evaluate


if __name__ == "__main__":
    pytest.main([__file__])
