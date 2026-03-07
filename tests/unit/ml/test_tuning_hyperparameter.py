import pytest
import pandas as pd
from unittest.mock import Mock, patch

try:
    from src.ml.core.ml_service import MLService
except ImportError:
    pytest.skip("无法导入ML服务", allow_module_level=True)

class TestTuningHyperparameter:
    @pytest.fixture
    def ml_service(self):
        """创建ML服务实例"""
        service = MLService()
        service.start()  # 启动服务用于测试
        return service

    @pytest.fixture
    def sample_training_data(self):
        """创建示例训练数据"""
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })
        return data

    def test_hyperparameter_optimization_basic(self, ml_service, sample_training_data):
        """测试基本超参数优化功能"""
        # 首先训练一个模型
        model_config = {
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 10, "max_depth": 5}
        }

        model_id = ml_service.train_model(sample_training_data, "target", model_config)
        assert model_id is not None

        # 测试超参数优化
        param_space = {
            "n_estimators": [10, 50, 100],
            "max_depth": [3, 5, 7]
        }

        result = ml_service.optimize_hyperparameters(model_id, param_space, sample_training_data)
        assert isinstance(result, dict)
        # 结果应该包含最佳参数或错误信息
        assert "error" not in result or "best_params" in result

    def test_hyperparameter_optimization_invalid_data(self, ml_service):
        """测试超参数优化使用无效数据"""
        param_space = {"n_estimators": [10, 50]}

        # 测试空数据
        result = ml_service.optimize_hyperparameters("invalid_model", param_space, pd.DataFrame())
        assert "error" in result

        # 测试空参数空间
        result = ml_service.optimize_hyperparameters("invalid_model", {}, pd.DataFrame([{"a": 1}]))
        assert "error" in result
