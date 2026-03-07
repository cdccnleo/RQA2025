#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML服务模型管理功能测试

测试MLService的完整模型生命周期管理功能
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.ml.core.ml_service import MLService, MLServiceStatus


class TestMLServiceModelManagement:
    """ML服务模型管理功能测试"""

    def setup_method(self):
        """测试前准备"""
        self.service = MLService()
        self.service.start()

    def teardown_method(self):
        """测试后清理"""
        self.service.stop()

    def test_model_loading_and_unloading(self):
        """测试模型加载和卸载"""
        model_id = "test_linear_model"
        model_config = {
            "algorithm": "linear_regression",
            "params": {"alpha": 0.1}
        }

        # 测试加载模型
        result = self.service.load_model(model_id, model_config)
        assert result is True

        # 验证模型已加载
        model_info = self.service.get_model_info(model_id)
        assert model_info is not None
        assert model_info["id"] == model_id
        assert model_info["algorithm"] == "linear_regression"
        assert model_info["status"] == "loaded"

        # 验证模型列表包含该模型
        models = self.service.list_models()
        assert len(models) > 0
        model_ids = [m["id"] for m in models]
        assert model_id in model_ids

        # 测试卸载模型
        result = self.service.unload_model(model_id)
        assert result is True

        # 验证模型已卸载
        model_info = self.service.get_model_info(model_id)
        assert model_info is None

    def test_multiple_model_loading(self):
        """测试多模型加载"""
        model_configs = [
            ("model1", {"algorithm": "linear_regression", "params": {"alpha": 0.1}}),
            ("model2", {"algorithm": "random_forest", "params": {"n_estimators": 10}}),
            ("model3", {"algorithm": "xgboost", "params": {"max_depth": 5}}),
        ]

        # 加载多个模型
        for model_id, config in model_configs:
            result = self.service.load_model(model_id, config)
            assert result is True

        # 验证所有模型都已加载
        models = self.service.list_models()
        assert len(models) >= 3

        loaded_ids = [m["id"] for m in models if m["status"] == "loaded"]
        for model_id, _ in model_configs:
            assert model_id in loaded_ids

        # 卸载所有模型
        for model_id, _ in model_configs:
            result = self.service.unload_model(model_id)
            assert result is True

    def test_model_training_workflow(self):
        """测试模型训练工作流"""
        model_id = "trained_model"

        # 创建训练数据
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

        training_data = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'feature3': X[:, 2],
            'target': y
        })

        model_config = {
            "algorithm": "linear_regression",
            "params": {}
        }

        # 训练模型
        result = self.service.train_model(model_id, training_data, model_config)
        assert result is True

        # 验证模型已训练并有性能指标
        model_info = self.service.get_model_info(model_id)
        assert model_info is not None
        assert model_info["status"] == "loaded"

        performance = self.service.get_model_performance(model_id)
        assert performance is not None
        assert "mse" in performance
        assert "rmse" in performance
        assert "r2_score" in performance
        assert "training_samples" in performance

        # 验证性能指标合理
        assert performance["training_samples"] == 100
        assert performance["mse"] >= 0
        assert performance["rmse"] >= 0
        assert -1 <= performance["r2_score"] <= 1

    def test_model_prediction_after_training(self):
        """测试训练后模型预测"""
        model_id = "prediction_test_model"

        # 创建训练数据
        np.random.seed(42)
        X_train = np.random.randn(50, 2)
        y_train = 2 * X_train[:, 0] + X_train[:, 1] + np.random.randn(50) * 0.1

        training_data = pd.DataFrame({
            'feature1': X_train[:, 0],
            'feature2': X_train[:, 1],
            'target': y_train
        })

        # 训练模型
        model_config = {"algorithm": "linear_regression", "params": {}}
        result = self.service.train_model(model_id, training_data, model_config)
        assert result is True

        # 创建测试数据
        X_test = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])

        # 测试预测
        predictions = []
        for i in range(len(X_test)):
            test_data = pd.DataFrame({
                'feature1': [X_test[i, 0]],
                'feature2': [X_test[i, 1]]
            })
            pred = self.service.predict(test_data)
            predictions.append(pred)

        assert len(predictions) == 3
        assert all(isinstance(p, (int, float, np.ndarray)) for p in predictions)

        # 验证预测结果是数组或标量
        for pred in predictions:
            if isinstance(pred, np.ndarray):
                assert pred.shape == (1,) or pred.ndim == 0

    def test_batch_prediction_functionality(self):
        """测试批量预测功能"""
        from src.ml.core.ml_service import MLInferenceRequest, MLFeatures

        model_id = "batch_test_model"

        # 创建训练数据并训练模型
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(30) * 0.1

        training_data = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'target': y
        })

        model_config = {"algorithm": "linear_regression", "params": {}}
        result = self.service.train_model(model_id, training_data, model_config)
        assert result is True

        # 创建批量请求
        requests = []
        for i in range(5):
            features = MLFeatures(
                timestamp=pd.Timestamp.now(),
                symbol="test",
                features={
                    'feature1': float(i),
                    'feature2': float(i * 2)
                }
            )
            request = MLInferenceRequest(
                request_id=f"batch_req_{i}",
                model_id=model_id,
                features=features,
                inference_type="batch"
            )
            requests.append(request)

        # 执行批量预测
        responses = self.service.predict_batch(requests)

        # 验证响应
        assert len(responses) == 5
        for response in responses:
            assert response.success is True
            assert response.prediction is not None
            assert response.confidence is not None
            assert response.processing_time_ms > 0
            assert response.processing_time_ms > 0

    def test_hyperparameter_optimization(self):
        """测试超参数优化功能"""
        model_id = "hpo_test_model"

        # 创建训练数据
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(30) * 0.1

        training_data = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'target': y
        })

        # 定义参数空间
        param_space = {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'max_iter': [100, 500, 1000]
        }

        # 执行超参数优化
        result = self.service.optimize_hyperparameters(model_id, param_space, training_data)

        # 验证结果
        assert isinstance(result, dict)
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'total_combinations' in result

        # 验证最佳参数
        best_params = result['best_params']
        assert isinstance(best_params, dict)

        # 验证最佳分数
        best_score = result['best_score']
        assert isinstance(best_score, (int, float))

        # 验证组合数量
        total_combinations = result['total_combinations']
        assert isinstance(total_combinations, int)
        assert total_combinations > 0

    def test_model_performance_tracking(self):
        """测试模型性能跟踪"""
        model_id = "performance_test_model"

        # 训练模型
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1] + np.random.randn(50) * 0.1

        training_data = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'target': y
        })

        model_config = {"algorithm": "linear_regression", "params": {}}
        result = self.service.train_model(model_id, training_data, model_config)
        assert result is True

        # 获取性能指标
        performance = self.service.get_model_performance(model_id)
        assert performance is not None

        # 验证性能指标结构
        required_metrics = ['mse', 'rmse', 'r2_score', 'training_samples']
        for metric in required_metrics:
            assert metric in performance

        # 验证指标值合理性
        assert performance['mse'] >= 0
        assert performance['rmse'] >= 0
        assert performance['training_samples'] == 50
        assert -1 <= performance['r2_score'] <= 1

    def test_service_status_comprehensive(self):
        """测试服务状态综合信息"""
        # 加载几个模型
        model_configs = [
            ("status_model1", {"algorithm": "linear_regression"}),
            ("status_model2", {"algorithm": "random_forest"}),
        ]

        for model_id, config in model_configs:
            self.service.load_model(model_id, config)

        # 获取服务状态
        status = self.service.get_service_status()

        # 验证状态结构
        assert isinstance(status, dict)
        assert 'status' in status
        assert 'stats' in status
        assert 'max_workers' in status
        assert 'loaded_models' in status
        assert 'available_models' in status

        # 验证状态值
        assert status['status'] == 'running'
        assert status['max_workers'] == 4  # 默认值
        assert status['loaded_models'] >= 2
        assert isinstance(status['available_models'], list)
        assert len(status['available_models']) >= 2

        # 验证统计信息
        stats = status['stats']
        assert isinstance(stats, dict)
        assert 'inference_requests' in stats
        assert 'model_loads' in stats
        assert 'training_sessions' in stats

    def test_error_handling_in_model_operations(self):
        """测试模型操作中的错误处理"""
        # 测试加载无效模型
        result = self.service.load_model("invalid_model", {"algorithm": "invalid_type"})
        assert result is False

        # 测试卸载不存在的模型
        result = self.service.unload_model("nonexistent_model")
        assert result is False

        # 测试获取不存在模型的信息
        info = self.service.get_model_info("nonexistent_model")
        assert info is None

        # 测试获取不存在模型的性能
        performance = self.service.get_model_performance("nonexistent_model")
        assert performance is None

    def test_different_algorithm_support(self):
        """测试不同算法支持"""
        algorithms_to_test = [
            ("linear_regression", {"params": {}}),
            ("random_forest", {"params": {"n_estimators": 5}}),
            ("xgboost", {"params": {"max_depth": 3}}),
        ]

        for alg_name, config in algorithms_to_test:
            model_id = f"test_{alg_name}"

            # 加载模型
            result = self.service.load_model(model_id, {"algorithm": alg_name, **config})
            assert result is True

            # 验证模型信息
            info = self.service.get_model_info(model_id)
            assert info is not None
            assert info["algorithm"] == alg_name

            # 卸载模型
            result = self.service.unload_model(model_id)
            assert result is True

    def test_model_replacement_and_update(self):
        """测试模型替换和更新"""
        model_id = "replace_test_model"

        # 加载第一个版本
        config_v1 = {"algorithm": "linear_regression", "version": "1.0"}
        result = self.service.load_model(model_id, config_v1)
        assert result is True

        # 验证第一个版本
        info_v1 = self.service.get_model_info(model_id)
        assert info_v1 is not None
        assert info_v1["config"]["version"] == "1.0"

        # 替换为第二个版本
        config_v2 = {"algorithm": "random_forest", "version": "2.0"}
        result = self.service.load_model(model_id, config_v2)  # 重新加载会替换
        assert result is True

        # 验证已替换为第二个版本
        info_v2 = self.service.get_model_info(model_id)
        assert info_v2 is not None
        assert info_v2["algorithm"] == "random_forest"
        assert info_v2["config"]["version"] == "2.0"

    def test_concurrent_model_operations(self):
        """测试并发模型操作"""
        import threading
        import time

        results = []
        errors = []

        def model_operation_worker(worker_id):
            """并发执行模型操作的worker"""
            try:
                model_id = f"concurrent_model_{worker_id}"

                # 创建训练数据
                np.random.seed(42 + worker_id)
                X = np.random.randn(20, 2)
                y = X[:, 0] + 2 * X[:, 1] + np.random.randn(20) * 0.1

                training_data = pd.DataFrame({
                    'feature1': X[:, 0],
                    'feature2': X[:, 1],
                    'target': y
                })

                # 训练模型
                config = {"algorithm": "linear_regression", "params": {}}
                result = self.service.train_model(model_id, training_data, config)
                results.append(("train", worker_id, result))

                # 获取模型信息
                info = self.service.get_model_info(model_id)
                results.append(("info", worker_id, info is not None))

                # 执行简单预测
                test_data = pd.DataFrame({'feature1': [1.0], 'feature2': [2.0]})
                pred = self.service.predict(test_data)
                results.append(("predict", worker_id, pred is not None))

                # 卸载模型
                unload_result = self.service.unload_model(model_id)
                results.append(("unload", worker_id, unload_result))

            except Exception as e:
                errors.append((worker_id, str(e)))

        # 创建多个线程
        threads = []
        num_threads = 5

        for i in range(num_threads):
            t = threading.Thread(target=model_operation_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(errors) == 0, f"Found errors in concurrent operations: {errors}"

        # 验证每个worker都完成了4个操作 (train, info, predict, unload)
        expected_operations = num_threads * 4
        assert len(results) == expected_operations, f"Expected {expected_operations} operations, got {len(results)}"

        # 验证操作结果
        for operation, worker_id, result in results:
            assert result is True or result is not None, f"Operation {operation} failed for worker {worker_id}"
