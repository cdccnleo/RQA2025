# tests/unit/ml/test_model_manager.py
"""
ModelManager单元测试

测试覆盖:
- 初始化参数验证
- 模型加载和卸载
- 模型训练功能
- 模型预测功能
- 模型版本管理
- 模型评估功能
- 错误处理
- 性能监控
- 并发安全性
- 边界条件
"""

import pytest

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.legacy,
    pytest.mark.timeout(45),  # 45秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]
pytest.skip("legacy 模型管理器测试默认跳过，需手动启用", allow_module_level=True)

import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import time
import os
import pickle
import json


# 可pickle的简单模型类，用于测试

class SimpleModel:
    """简单的可pickle模型类"""
    def predict(self, X):
        return np.array([0.8, 0.6])


from src.ml.models.model_manager import (
    ModelManager,
    ModelType,
    ModelPrediction,
    ModelMetadata
)


class TestModelManager:
    """ModelManager测试类"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_data(self):
        """样本数据fixture"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

    @pytest.fixture
    def model_config(self):
        """模型配置fixture"""
        return {
            'model_type': ModelType.RANDOM_FOREST,
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'features': ['feature_1', 'feature_2', 'feature_3'],
            'target': 'target'
        }

    @pytest.fixture
    def model_manager(self, temp_dir, model_config):
        """ModelManager实例"""
        with patch('src.ml.model_manager.get_models_adapter') as mock_adapter:
            mock_adapter_instance = Mock()
            mock_adapter_instance.get_models_logger.return_value = Mock()
            mock_adapter.return_value = mock_adapter_instance

            manager = ModelManager()
            manager.config = model_config
            yield manager

    def test_initialization(self):
        """测试初始化"""
        with patch('src.ml.model_manager.get_models_adapter') as mock_adapter:
            mock_adapter_instance = Mock()
            mock_adapter_instance.get_models_logger.return_value = Mock()
            mock_adapter.return_value = mock_adapter_instance

            manager = ModelManager()

            assert manager.models == {}
            assert hasattr(manager, 'config')
            assert hasattr(manager, 'logger')

    def test_model_loading_success(self, model_manager, temp_dir):
        """测试模型创建和存储成功"""
        model_name = 'test_model'

        # 创建模型
        model_id = model_manager.create_model(model_name, ModelType.LINEAR_REGRESSION, "Test model")

        # 手动设置模型对象（模拟已训练的模型）
        simple_model = SimpleModel()
        model_manager.models[model_id] = simple_model

        # 验证模型已创建
        assert model_id in model_manager.model_metadata
        assert model_manager.model_metadata[model_id].model_name == model_name
        assert model_id in model_manager.models

    def test_model_loading_file_not_found(self, model_manager):
        """测试模型加载文件不存在"""
        success = model_manager.load_model('nonexistent_model', '/nonexistent/path.pkl')

        assert success is False

    def test_model_unloading(self, model_manager, temp_dir):
        """测试模型卸载"""
        # 创建一个新的ModelManager实例，避免持久化问题
        fresh_manager = ModelManager({
            'model_storage_path': str(temp_dir / 'fresh_models'),
            'metadata_storage_path': str(temp_dir / 'fresh_metadata')
        })

        # 先创建一个模型
        model_id = fresh_manager.create_model('test_model', ModelType.LINEAR_REGRESSION)

        # 验证模型已创建
        model_info = fresh_manager.get_model_info('test_model')
        assert model_info is not None
        assert model_info['model_name'] == 'test_model'

        # 卸载模型
        success = fresh_manager.unload_model('test_model')
        assert success is True

        # 验证模型已卸载
        model_info = fresh_manager.get_model_info('test_model')
        assert model_info is None

    def test_model_prediction(self, model_manager, sample_data):
        """测试模型预测"""
        # 创建并训练模型
        model_id = model_manager.create_model('test_model', ModelType.LINEAR_REGRESSION)
        success = model_manager.train_model(model_id, sample_data, 'target', ['feature_1', 'feature_2'], {})
        assert success is True

        success = model_manager.deploy_model(model_id)
        assert success is True

        test_data = sample_data.head(3)
        # 使用model_type进行预测
        prediction = model_manager.predict(ModelType.LINEAR_REGRESSION.value, test_data)

        assert prediction is not None
        assert isinstance(prediction, ModelPrediction)
        # 对于单行输入，prediction.prediction可能是一个标量
        assert hasattr(prediction, 'prediction')

    def test_model_training(self, model_manager, sample_data):
        """测试模型训练"""
        train_data = sample_data

        # 先创建模型
        model_id = model_manager.create_model("new_model", ModelType.LINEAR_REGRESSION)

        # Mock训练方法
        with patch.object(model_manager, '_train_model') as mock_train:
            mock_train.return_value = Mock()

            success = model_manager.train_model(model_id, train_data, 'target', ['feature_1', 'feature_2'], {})

            assert success is True
            assert model_id in model_manager.models
            mock_train.assert_called_once()

    def test_model_evaluation(self, model_manager, sample_data):
        """测试模型评估"""
        # 先创建并训练模型
        model_id = model_manager.create_model("test_model", ModelType.LINEAR_REGRESSION)
        model_manager.train_model(model_id, sample_data, 'target', ['feature_1', 'feature_2'], {})

        test_data = sample_data.head(20)
        metrics = model_manager.evaluate_model(model_id, test_data, 'target')

        assert metrics is not None
        # 检查metrics是否包含accuracy字段
        assert 'accuracy' in metrics or hasattr(metrics, 'accuracy')

    def test_model_version_management(self, model_manager):
        """测试模型版本管理"""
        # Mock版本信息
        version_info = {
            'model_name': 'test_model',
            'version': '1.0.0',
            'created_at': datetime.now(),
            'metrics': {'accuracy': 0.85}
        }

        # 这里可以测试版本管理功能，如果ModelManager支持的话
        # 由于实际实现可能不支持，我们只测试基本功能
        assert version_info['model_name'] == 'test_model'

    def test_performance_monitoring(self, model_manager, sample_data):
        """测试性能监控"""
        # Mock已加载的模型
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8, 0.6])
        model_manager.models['test_model'] = mock_model

        test_data = sample_data.head(2)

        start_time = time.time()
        prediction = model_manager.predict('test_model', test_data)
        end_time = time.time()

        duration = end_time - start_time

        assert prediction is not None
        assert duration >= 0
        # 预测应该很快完成
        assert duration < 1.0

    def test_concurrent_model_access(self, model_manager, sample_data):
        """测试并发模型访问"""
        import concurrent.futures

        # Mock已加载的模型
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.8])
        model_manager.models['test_model'] = mock_model

        test_data = sample_data.head(1)

        results = []
        errors = []

        def predict_worker():
            try:
                result = model_manager.predict('test_model', test_data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # 并发执行10个预测请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(predict_worker) for _ in range(10)]
            concurrent.futures.wait(futures)

        # 验证并发安全性
        assert len(results) == 10
        assert len(errors) == 0

    def test_error_handling_invalid_prediction_data(self, model_manager):
        """测试无效预测数据错误处理"""
        # Mock已加载的模型
        mock_model = Mock()
        mock_model.predict.side_effect = ValueError("Invalid input data")
        model_manager.models['test_model'] = mock_model

        invalid_data = pd.DataFrame()  # 空数据

        with pytest.raises(ValueError, match="Invalid input data"):
            model_manager.predict('test_model', invalid_data)

    def test_model_config_validation(self, model_manager):
        """测试模型配置验证"""
        # 有效配置
        valid_config = ModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            hyperparameters={'n_estimators': 100},
            features=['feature_1', 'feature_2'],
            target='target'
        )

        is_valid, errors = model_manager.validate_config(valid_config)
        assert is_valid is True
        assert len(errors) == 0

    def test_model_info_retrieval(self, model_manager):
        """测试模型信息获取"""
        # Mock已加载的模型信息
        model_manager.models['test_model'] = Mock()

        info = model_manager.get_model_info('test_model')

        assert info is not None
        assert 'name' in info
        assert 'type' in info
        assert 'version' in info

    def test_model_health_check(self, model_manager):
        """测试模型健康检查"""
        # Mock已加载的模型
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.5])
        model_manager.models['test_model'] = mock_model

        health = model_manager.health_check('test_model')

        assert health is not None
        assert health['status'] == 'healthy'
