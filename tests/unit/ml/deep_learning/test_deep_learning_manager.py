"""
深度学习管理器模块测试
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.ml.deep_learning.core.deep_learning_manager import (
    get_models_adapter,
    get_data_preprocessor,
    get_trainer,
    get_model_service,
    TrainingResult
)


class TestDeepLearningManagerFunctions:
    """测试深度学习管理器函数"""

    def test_get_models_adapter(self):
        """测试获取模型适配器"""
        adapter = get_models_adapter()
        assert adapter is not None
        assert hasattr(adapter, 'get_models_logger')

    def test_get_data_preprocessor(self):
        """测试获取数据预处理器"""
        preprocessor = get_data_preprocessor()
        assert preprocessor is not None

        # 测试带配置的预处理器
        config = {'batch_size': 32, 'normalize': True}
        preprocessor_with_config = get_data_preprocessor(config)
        assert preprocessor_with_config is not None

    def test_get_trainer(self):
        """测试获取训练器"""
        trainer = get_trainer()
        assert trainer is not None
        assert hasattr(trainer, 'train')

    def test_trainer_train_method(self):
        """测试训练器训练方法"""
        trainer = get_trainer()

        # 创建模拟模型和数据
        model = Mock()
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [0.1, 0.2, 0.3]})
        config = {'model_id': 'test_model', 'epochs': 10}

        # 执行训练
        result = trainer.train(model, data, config)

        # 验证结果
        assert isinstance(result, TrainingResult)
        assert result.model_id == 'test_model'
        assert result.version == '1.0.0'
        assert result.metrics == {'accuracy': 1.0}
        assert 'weights' in result.artifacts

    def test_get_model_service(self):
        """测试获取模型服务"""
        service = get_model_service()
        assert service is not None
        assert hasattr(service, 'save_model')
        assert hasattr(service, 'load_model')

    def test_model_service_save_and_load(self):
        """测试模型服务保存和加载"""
        service = get_model_service()

        # 保存模型
        model_id = 'test_model'
        version = '1.0.0'
        model = Mock()
        metadata = {'description': 'Test model'}

        service.save_model(model_id, version, model, metadata)

        # 验证保存成功
        assert (model_id, version) in service.saved
        saved_model, saved_metadata = service.saved[(model_id, version)]
        assert saved_model == model
        assert saved_metadata == metadata

        # 加载模型
        loaded_model, loaded_metadata = service.load_model(model_id, version)

        # 验证加载结果
        assert loaded_model == model
        assert loaded_metadata == metadata

    def test_model_service_load_nonexistent(self):
        """测试加载不存在的模型"""
        service = get_model_service()

        with pytest.raises(FileNotFoundError):
            service.load_model('nonexistent', '1.0.0')


class TestTrainingResult:
    """测试训练结果数据类"""

    def test_training_result_creation(self):
        """测试训练结果创建"""
        result = TrainingResult(
            model_id='test_model',
            version='1.0.0',
            metrics={'accuracy': 0.95, 'loss': 0.05},
            artifacts={'weights': b'weight_data', 'config': {'param': 1}}
        )

        assert result.model_id == 'test_model'
        assert result.version == '1.0.0'
        assert result.metrics['accuracy'] == 0.95
        assert result.artifacts['weights'] == b'weight_data'

    def test_training_result_default_values(self):
        """测试训练结果默认值"""
        result = TrainingResult(
            model_id='test_model',
            version='1.0.0',
            metrics={},
            artifacts={}
        )

        assert result.model_id == 'test_model'
        assert result.version == '1.0.0'
        assert result.metrics == {}
        assert result.artifacts == {}


class TestDeepLearningManagerIntegration:
    """测试深度学习管理器集成功能"""

    def test_full_training_pipeline(self):
        """测试完整训练流水线"""
        # 获取组件
        preprocessor = get_data_preprocessor({'normalize': True})
        trainer = get_trainer()
        service = get_model_service()

        # 创建训练数据
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })

        # 预处理数据（模拟）
        processed_data = data  # 简化测试

        # 训练模型
        model = Mock()
        config = {
            'model_id': 'pipeline_test_model',
            'epochs': 5,
            'batch_size': 32
        }

        result = trainer.train(model, processed_data, config)

        # 保存模型
        service.save_model(
            result.model_id,
            result.version,
            model,
            {'pipeline': 'test', 'metrics': result.metrics}
        )

        # 验证流水线完成
        assert result.model_id == 'pipeline_test_model'
        assert (result.model_id, result.version) in service.saved

    def test_configuration_persistence(self):
        """测试配置持久化"""
        config1 = {'batch_size': 32, 'learning_rate': 0.01}
        config2 = {'batch_size': 64, 'learning_rate': 0.001}

        preprocessor1 = get_data_preprocessor(config1)
        preprocessor2 = get_data_preprocessor(config2)

        # 验证不同的配置创建不同的实例
        assert preprocessor1 is not preprocessor2

        # 验证配置正确传递（通过实例属性检查）
        # 注意：实际实现可能需要具体检查配置是否正确应用

    def test_error_handling(self):
        """测试错误处理"""
        service = get_model_service()

        # 测试加载不存在的模型
        with pytest.raises(FileNotFoundError):
            service.load_model('missing_model', '1.0.0')

        # 测试保存无效参数（如果有验证的话）
        # 这里简化测试，实际实现可能有更多验证

    def test_component_isolation(self):
        """测试组件隔离"""
        # 创建多个服务实例
        service1 = get_model_service()
        service2 = get_model_service()

        # 它们应该是独立的
        service1.save_model('model1', '1.0', Mock(), {})
        service2.save_model('model2', '1.0', Mock(), {})

        assert ('model1', '1.0') in service1.saved
        assert ('model2', '1.0') in service2.saved
        assert ('model1', '1.0') not in service2.saved
        assert ('model2', '1.0') not in service1.saved

    def test_resource_management(self):
        """测试资源管理"""
        # 创建多个训练器和服务
        trainers = [get_trainer() for _ in range(3)]
        services = [get_model_service() for _ in range(3)]

        # 验证都能正常工作
        for i, trainer in enumerate(trainers):
            model = Mock()
            data = pd.DataFrame({'x': [1, 2, 3]})
            config = {'model_id': f'test_model_{i}'}

            result = trainer.train(model, data, config)
            assert result.model_id == f'test_model_{i}'

        # 验证服务相互独立
        for i, service in enumerate(services):
            service.save_model(f'service_model_{i}', '1.0', Mock(), {})
            assert (f'service_model_{i}', '1.0') in service.saved
