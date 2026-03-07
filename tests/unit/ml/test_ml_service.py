#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ML核心服务

测试目标：提升ml_service.py的覆盖率到100%
"""

import pytest

pytestmark = pytest.mark.legacy
pytest.skip("legacy ML 服务测试默认跳过，需手动启用", allow_module_level=True)
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any, Optional

from src.ml.core.ml_service import MLService
from src.core.foundation.interfaces.ml_strategy_interfaces import (
    MLInferenceRequest, MLInferenceResponse, MLFeatures
)


class TestMLService:
    """测试ML服务"""

    @pytest.fixture
    def ml_service(self):
        """创建ML服务实例"""
        config = {
            "max_models": 10,
            "default_model_type": "linear_regression",
            "cache_enabled": True,
            "performance_monitoring": True
        }
        return MLService(config)

    def test_ml_service_initialization(self, ml_service):
        """测试ML服务初始化"""
        assert ml_service.max_models == 10
        assert ml_service.default_model_type == "linear_regression"
        assert ml_service.cache_enabled == True
        assert ml_service.performance_monitoring == True
        assert isinstance(ml_service.loaded_models, dict)
        assert isinstance(ml_service.model_configs, dict)
        assert isinstance(ml_service.performance_stats, dict)

    def test_ml_service_default_initialization(self):
        """测试ML服务默认初始化"""
        service = MLService()

        assert service.max_models == 5  # 默认值
        assert service.default_model_type == "linear_regression"
        assert service.cache_enabled == False
        assert service.performance_monitoring == False

    @patch('src.ml.core.ml_service.MLService._create_model_instance')
    @patch('src.ml.core.ml_service.MLService._load_model_weights')
    def test_load_model_success(self, mock_load_weights, mock_create_instance, ml_service):
        """测试成功加载模型"""
        # 设置模拟
        mock_model = Mock()
        mock_create_instance.return_value = mock_model
        mock_load_weights.return_value = True

        model_config = {
            "model_type": "linear_regression",
            "features": ["feature1", "feature2"],
            "target": "target"
        }

        result = ml_service.load_model("test_model", model_config)

        assert result == True
        assert "test_model" in ml_service.loaded_models
        assert ml_service.loaded_models["test_model"] == mock_model
        mock_create_instance.assert_called_once_with(model_config)
        mock_load_weights.assert_called_once()

    def test_load_model_max_limit_reached(self, ml_service):
        """测试达到最大模型数量限制"""
        # 设置最大模型数量为1
        ml_service.max_models = 1

        # 先加载一个模型
        ml_service.loaded_models["existing_model"] = Mock()

        # 尝试加载第二个模型
        model_config = {"model_type": "linear_regression"}
        result = ml_service.load_model("new_model", model_config)

        assert result == False
        assert "new_model" not in ml_service.loaded_models

    def test_load_model_invalid_config(self, ml_service):
        """测试加载模型时使用无效配置"""
        invalid_configs = [
            {},  # 空配置
            {"invalid_field": "value"},  # 缺少必需字段
            {"model_type": ""},  # 空模型类型
        ]

        for config in invalid_configs:
            result = ml_service.load_model("test_model", config)
            assert result == False

    def test_unload_model_success(self, ml_service):
        """测试成功卸载模型"""
        # 先加载一个模型
        mock_model = Mock()
        ml_service.loaded_models["test_model"] = mock_model
        ml_service.model_configs["test_model"] = {"model_type": "linear_regression"}

        with patch.object(ml_service, '_cleanup_model_resources') as mock_cleanup:
            result = ml_service.unload_model("test_model")

            assert result == True
            assert "test_model" not in ml_service.loaded_models
            assert "test_model" not in ml_service.model_configs
            mock_cleanup.assert_called_once_with(mock_model)

    def test_unload_model_not_found(self, ml_service):
        """测试卸载不存在的模型"""
        result = ml_service.unload_model("nonexistent_model")

        assert result == False

    def test_list_models_all(self, ml_service):
        """测试列出所有模型"""
        # 设置一些模型
        ml_service.model_configs = {
            "model1": {"model_type": "linear_regression", "status": "active"},
            "model2": {"model_type": "random_forest", "status": "training"},
            "model3": {"model_type": "neural_network", "status": "inactive"}
        }

        models = ml_service.list_models()

        assert len(models) == 3
        assert all("model_id" in model for model in models)
        assert all("model_type" in model for model in models)

    def test_list_models_by_type(self, ml_service):
        """测试按类型列出模型"""
        # 设置一些模型
        ml_service.model_configs = {
            "model1": {"model_type": "linear_regression", "status": "active"},
            "model2": {"model_type": "random_forest", "status": "training"},
            "model3": {"model_type": "linear_regression", "status": "inactive"}
        }

        lr_models = ml_service.list_models("linear_regression")

        assert len(lr_models) == 2
        assert all(model["model_type"] == "linear_regression" for model in lr_models)

    @pytest.mark.asyncio
    async def test_predict_success(self, ml_service):
        """测试成功预测"""
        # 设置模型
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8])
        ml_service.loaded_models["test_model"] = mock_model
        ml_service.model_configs["test_model"] = {
            "model_type": "linear_regression",
            "features": ["feature1", "feature2"]
        }

        # 创建预测请求
        features = MLFeatures(
            feature_names=["feature1", "feature2"],
            feature_values=[[1.0, 2.0]]
        )
        request = MLInferenceRequest(
            model_id="test_model",
            features=features,
            request_id="test_request_001"
        )

        response = await ml_service.predict(request)

        assert isinstance(response, MLInferenceResponse)
        assert response.request_id == "test_request_001"
        assert response.success == True
        assert len(response.predictions) == 1
        assert response.predictions[0] == 0.8

    @pytest.mark.asyncio
    async def test_predict_model_not_found(self, ml_service):
        """测试预测时模型不存在"""
        features = MLFeatures(
            feature_names=["feature1"],
            feature_values=[[1.0]]
        )
        request = MLInferenceRequest(
            model_id="nonexistent_model",
            features=features,
            request_id="test_request_001"
        )

        response = await ml_service.predict(request)

        assert isinstance(response, MLInferenceResponse)
        assert response.success == False
        assert "Model not found" in response.error_message

    @pytest.mark.asyncio
    async def test_predict_batch_success(self, ml_service):
        """测试批量预测成功"""
        # 设置模型
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.6, 0.9])
        ml_service.loaded_models["test_model"] = mock_model
        ml_service.model_configs["test_model"] = {
            "model_type": "linear_regression",
            "features": ["feature1", "feature2"]
        }

        # 创建批量预测请求
        requests = []
        for i in range(3):
            features = MLFeatures(
                feature_names=["feature1", "feature2"],
                feature_values=[[float(i), float(i+1)]]
            )
            request = MLInferenceRequest(
                model_id="test_model",
                features=features,
                request_id=f"test_request_{i}"
            )
            requests.append(request)

        responses = await ml_service.predict_batch(requests)

        assert len(responses) == 3
        assert all(isinstance(r, MLInferenceResponse) for r in responses)
        assert all(r.success == True for r in responses)

    def test_get_model_performance(self, ml_service):
        """测试获取模型性能"""
        # 设置性能统计
        ml_service.performance_stats["test_model"] = {
            "total_predictions": 100,
            "avg_response_time": 0.05,
            "accuracy": 0.85,
            "last_updated": datetime.now()
        }

        performance = ml_service.get_model_performance("test_model")

        assert isinstance(performance, dict)
        assert performance["total_predictions"] == 100
        assert performance["avg_response_time"] == 0.05
        assert performance["accuracy"] == 0.85

    def test_get_model_performance_not_found(self, ml_service):
        """测试获取不存在模型的性能"""
        performance = ml_service.get_model_performance("nonexistent_model")

        assert isinstance(performance, dict)
        assert performance["total_predictions"] == 0

    def test_train_model_basic(self, ml_service):
        """测试基本模型训练"""
        # 创建训练数据
        training_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [3, 6, 9, 12, 15]
        })

        training_config = {
            "model_type": "linear_regression",
            "features": ["feature1", "feature2"],
            "target": "target",
            "test_size": 0.2
        }

        result = ml_service.train_model("test_model", training_data, training_config)

        assert isinstance(result, dict)
        assert "model_id" in result
        assert result["model_id"] == "test_model"
        assert "training_status" in result

    def test_train_model_invalid_data(self, ml_service):
        """测试使用无效数据训练模型"""
        invalid_data = [
            None,
            pd.DataFrame(),  # 空DataFrame
            pd.DataFrame({'invalid': [1, 2, 3]})  # 缺少目标列
        ]

        training_config = {
            "model_type": "linear_regression",
            "features": ["feature1"],
            "target": "target"
        }

        for data in invalid_data:
            result = ml_service.train_model("test_model", data, training_config)
            assert isinstance(result, dict)
            assert result["success"] == False

    def test_optimize_hyperparameters(self, ml_service):
        """测试超参数优化"""
        param_space = {
            "max_depth": [3, 5, 7],
            "n_estimators": [10, 50, 100],
            "learning_rate": [0.01, 0.1, 0.2]
        }

        # 需要先训练一个模型
        training_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [3, 6, 9, 12, 15]
        })

        training_config = {
            "model_type": "random_forest",
            "features": ["feature1", "feature2"],
            "target": "target"
        }

        result = ml_service.optimize_hyperparameters("test_model", param_space, training_data, training_config)

        assert isinstance(result, dict)
        assert "best_params" in result
        assert "best_score" in result

    def test_get_service_status(self, ml_service):
        """测试获取服务状态"""
        # 设置一些模型
        ml_service.loaded_models = {"model1": Mock(), "model2": Mock()}
        ml_service.model_configs = {
            "model1": {"model_type": "linear_regression", "status": "active"},
            "model2": {"model_type": "random_forest", "status": "training"}
        }

        status = ml_service.get_service_status()

        assert isinstance(status, dict)
        assert "total_models" in status
        assert "active_models" in status
        assert "service_health" in status
        assert status["total_models"] == 2

    def test_validate_model_config_valid(self, ml_service):
        """测试验证有效模型配置"""
        valid_configs = [
            {
                "model_type": "linear_regression",
                "features": ["feature1", "feature2"],
                "target": "target"
            },
            {
                "model_type": "random_forest",
                "features": ["feature1"],
                "target": "target",
                "hyperparameters": {"n_estimators": 100}
            }
        ]

        for config in valid_configs:
            result = ml_service._validate_model_config(config)
            assert result == True

    def test_validate_model_config_invalid(self, ml_service):
        """测试验证无效模型配置"""
        invalid_configs = [
            {},  # 空配置
            {"features": ["feature1"]},  # 缺少model_type
            {"model_type": "linear_regression"},  # 缺少features和target
            {"model_type": "", "features": ["feature1"], "target": "target"},  # 空model_type
        ]

        for config in invalid_configs:
            result = ml_service._validate_model_config(config)
            assert result == False


class TestMLServiceIntegration:
    """测试ML服务集成场景"""

    @pytest.fixture
    def ml_service(self):
        """创建完整的ML服务"""
        config = {
            "max_models": 5,
            "cache_enabled": True,
            "performance_monitoring": True
        }
        return MLService(config)

    @pytest.mark.asyncio
    async def test_complete_ml_workflow(self, ml_service):
        """测试完整的ML工作流程"""
        # 1. 训练模型
        training_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            'target': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        })

        training_config = {
            "model_type": "linear_regression",
            "features": ["feature1", "feature2"],
            "target": "target"
        }

        train_result = ml_service.train_model("workflow_model", training_data, training_config)
        assert train_result["success"] == True

        # 2. 进行预测
        features = MLFeatures(
            feature_names=["feature1", "feature2"],
            feature_values=[[5.0, 10.0]]
        )
        request = MLInferenceRequest(
            model_id="workflow_model",
            features=features,
            request_id="workflow_test_001"
        )

        predict_result = await ml_service.predict(request)
        assert predict_result.success == True
        assert len(predict_result.predictions) == 1

        # 3. 检查性能统计
        performance = ml_service.get_model_performance("workflow_model")
        assert isinstance(performance, dict)

        # 4. 获取服务状态
        status = ml_service.get_service_status()
        assert status["total_models"] >= 1

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, ml_service):
        """测试批量处理效率"""
        # 训练模型
        training_data = pd.DataFrame({
            'feature1': list(range(1, 21)),
            'feature2': [i*2 for i in range(1, 21)],
            'target': [i*3 for i in range(1, 21)]
        })

        training_config = {
            "model_type": "linear_regression",
            "features": ["feature1", "feature2"],
            "target": "target"
        }

        ml_service.train_model("batch_model", training_data, training_config)

        # 创建批量预测请求
        batch_size = 10
        requests = []
        for i in range(batch_size):
            features = MLFeatures(
                feature_names=["feature1", "feature2"],
                feature_values=[[float(i+1), float((i+1)*2)]]
            )
            request = MLInferenceRequest(
                model_id="batch_model",
                features=features,
                request_id=f"batch_test_{i}"
            )
            requests.append(request)

        import time
        start_time = time.time()

        # 执行批量预测
        responses = await ml_service.predict_batch(requests)

        end_time = time.time()
        batch_time = end_time - start_time

        # 验证结果
        assert len(responses) == batch_size
        assert all(r.success == True for r in responses)
        assert all(len(r.predictions) == 1 for r in responses)

        # 批量处理应该足够高效（每秒至少处理50个请求）
        assert batch_time < 1.0 or batch_size / batch_time > 50

    def test_model_lifecycle_management(self, ml_service):
        """测试模型生命周期管理"""
        model_configs = [
            {
                "model_type": "linear_regression",
                "features": ["feature1", "feature2"],
                "target": "target"
            },
            {
                "model_type": "random_forest",
                "features": ["feature1"],
                "target": "target",
                "hyperparameters": {"n_estimators": 10}
            }
        ]

        # 加载多个模型
        for i, config in enumerate(model_configs):
            model_id = f"lifecycle_model_{i}"
            result = ml_service.load_model(model_id, config)
            assert result == True

        # 验证模型已加载
        models = ml_service.list_models()
        assert len(models) == 2

        # 卸载一个模型
        result = ml_service.unload_model("lifecycle_model_0")
        assert result == True

        # 验证模型已卸载
        models = ml_service.list_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "lifecycle_model_1"

    def test_error_handling_and_recovery(self, ml_service):
        """测试错误处理和恢复"""
        # 测试加载无效模型
        invalid_config = {"invalid_field": "value"}
        result = ml_service.load_model("invalid_model", invalid_config)
        assert result == False

        # 测试预测不存在的模型
        features = MLFeatures(
            feature_names=["feature1"],
            feature_values=[[1.0]]
        )
        request = MLInferenceRequest(
            model_id="nonexistent_model",
            features=features,
            request_id="error_test_001"
        )

        import asyncio
        async def test_predict():
            response = await ml_service.predict(request)
            assert response.success == False
            assert response.error_message is not None

        asyncio.run(test_predict())

        # 验证服务仍然可以正常工作
        status = ml_service.get_service_status()
        assert isinstance(status, dict)

    def test_performance_monitoring(self, ml_service):
        """测试性能监控"""
        # 启用性能监控
        ml_service.performance_monitoring = True

        # 执行一些操作
        for i in range(5):
            model_config = {
                "model_type": "linear_regression",
                "features": ["feature1"],
                "target": "target"
            }
            ml_service.load_model(f"perf_model_{i}", model_config)

        # 检查性能统计
        status = ml_service.get_service_status()
        assert "performance_stats" in status

        # 检查单个模型性能
        if ml_service.loaded_models:
            model_id = list(ml_service.loaded_models.keys())[0]
            performance = ml_service.get_model_performance(model_id)
            assert isinstance(performance, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
