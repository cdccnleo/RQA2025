#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelManager深度测试
测试模型管理器的完整生命周期和复杂场景
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
pytest.skip("legacy 模型管理器深度测试默认跳过，需手动启用", allow_module_level=True)
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
import json

from src.ml.model_manager import (
    ModelManager,
    ModelType,
    ModelStatus,
    ModelMetadata,
    ModelPrediction,
    FeatureType
)




class TestModelManagerLifecycle:
    """测试模型管理器完整生命周期"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'model_storage_path': str(self.temp_dir / 'models'),
            'metadata_storage_path': str(self.temp_dir / 'metadata'),
            'max_models_per_type': 10,
            'auto_save_interval': 300,
            'cache_enabled': True
        }

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_model_creation_and_initialization(self):
        """测试模型创建和初始化"""
        manager = ModelManager(self.config)

        # 验证初始化
        assert manager.config == self.config
        assert hasattr(manager, 'models')
        assert hasattr(manager, 'model_metadata')
        assert hasattr(manager, '_model_cache')

        # 验证存储路径创建
        assert (Path(self.config['model_storage_path'])).exists()
        assert (Path(self.config['metadata_storage_path'])).exists()

    def test_model_creation_workflow(self):
        """测试模型创建工作流程"""
        manager = ModelManager(self.config)

        model_name = "test_linear_regression"
        model_type = ModelType.LINEAR_REGRESSION
        hyperparameters = {"fit_intercept": True, "normalize": True}

        # 创建模型
        model_id = manager.create_model(model_name, model_type)

        # 验证模型创建
        assert model_id is not None
        assert model_id in manager.models

        # 验证元数据
        assert model_id in manager.model_metadata
        metadata = manager.model_metadata[model_id]
        assert metadata.model_name == model_name
        assert metadata.model_type == model_type
        assert metadata.status == ModelStatus.TRAINING

    def test_model_training_workflow(self):
        """测试模型训练工作流程"""
        manager = ModelManager(self.config)

        # 创建测试数据
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        y = pd.Series(X['feature1'] * 2 + X['feature2'] * -1 + np.random.randn(n_samples) * 0.1)

        training_data = pd.concat([X, y.rename('target')], axis=1)

        # 创建模型
        model_id = manager.create_model("test_lr", ModelType.LINEAR_REGRESSION)

        # 训练模型
        result = manager.train_model(
            model_id,
            training_data,
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3']
        )

        # 验证训练结果
        assert result is True
        assert manager.model_metadata[model_id].status == ModelStatus.TRAINED
        assert 'training_score' in manager.model_metadata[model_id].performance_metrics

        # 验证模型文件存在
        assert (Path(self.config['model_storage_path']) / f"{model_id}.pkl").exists()

    def test_model_prediction_workflow(self):
        """测试模型预测工作流程"""
        manager = ModelManager(self.config)

        # 创建和训练模型
        np.random.seed(42)
        n_samples = 50
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = pd.Series(X['feature1'] * 2 + X['feature2'] * -1 + np.random.randn(n_samples) * 0.1)

        training_data = pd.concat([X, y.rename('target')], axis=1)
        model_id = manager.create_model("test_lr", ModelType.LINEAR_REGRESSION)
        manager.train_model(model_id, training_data, target_column='target',
                          feature_columns=['feature1', 'feature2'])

        # 测试单次预测
        input_data = {'feature1': 1.0, 'feature2': -0.5}
        prediction = manager.predict("test_lr", input_data)

        assert isinstance(prediction, ModelPrediction)
        assert hasattr(prediction, 'prediction_id')
        assert hasattr(prediction, 'prediction_value')
        assert hasattr(prediction, 'confidence_score')
        assert hasattr(prediction, 'timestamp')

    def test_batch_prediction_workflow(self):
        """测试批量预测工作流程"""
        manager = ModelManager(self.config)

        # 创建和训练模型
        np.random.seed(42)
        n_samples = 30
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = pd.Series(X['feature1'] * 2 + X['feature2'] * -1 + np.random.randn(n_samples) * 0.1)

        training_data = pd.concat([X, y.rename('target')], axis=1)
        model_id = manager.create_model("test_lr", ModelType.LINEAR_REGRESSION)
        manager.train_model(model_id, training_data, target_column='target',
                          feature_columns=['feature1', 'feature2'])

        # 测试批量预测
        input_data_list = [
            {'feature1': 1.0, 'feature2': -0.5},
            {'feature1': 0.5, 'feature2': 1.2},
            {'feature1': -0.3, 'feature2': 0.8}
        ]

        predictions = manager.batch_predict("test_lr", input_data_list)

        assert len(predictions) == 3
        for prediction in predictions:
            assert isinstance(prediction, ModelPrediction)
            assert hasattr(prediction, 'prediction_value')

    def test_model_deployment_workflow(self):
        """测试模型部署工作流程"""
        manager = ModelManager(self.config)

        # 创建和训练模型
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20)
        })
        y = pd.Series(X['feature1'] * 2 + X['feature2'] * -1 + np.random.randn(20) * 0.1)
        training_data = pd.concat([X, y.rename('target')], axis=1)

        model_id = manager.create_model("test_lr", ModelType.LINEAR_REGRESSION)
        manager.train_model(model_id, training_data, target_column='target',
                          feature_columns=['feature1', 'feature2'])

        # 部署模型
        result = manager.deploy_model(model_id)

        # 验证部署结果
        assert result is True
        assert manager.model_metadata[model_id].status == ModelStatus.DEPLOYED
        assert model_id in manager.deployed_models

    def test_model_performance_monitoring(self):
        """测试模型性能监控"""
        manager = ModelManager(self.config)

        # 创建和训练模型
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(25),
            'feature2': np.random.randn(25)
        })
        y = pd.Series(X['feature1'] * 2 + X['feature2'] * -1 + np.random.randn(25) * 0.1)
        training_data = pd.concat([X, y.rename('target')], axis=1)

        model_id = manager.create_model("test_lr", ModelType.LINEAR_REGRESSION)
        manager.train_model(model_id, training_data, target_column='target',
                          feature_columns=['feature1', 'feature2'])

        # 获取性能指标
        performance = manager.get_model_performance(model_id)

        assert isinstance(performance, dict)
        assert 'training_score' in performance
        assert 'model_size' in performance
        assert 'last_updated' in performance

    def test_model_listing_and_filtering(self):
        """测试模型列表和过滤"""
        manager = ModelManager(self.config)

        # 创建多个模型
        model_ids = []
        for i, model_type in enumerate([ModelType.LINEAR_REGRESSION,
                                       ModelType.RANDOM_FOREST,
                                       ModelType.XGBOOST]):
            model_id = manager.create_model(f"test_model_{i}", model_type)
            model_ids.append(model_id)

        # 测试列出所有模型
        all_models = manager.list_models()
        assert len(all_models) == 3

        # 测试按类型过滤
        lr_models = manager.list_models(ModelType.LINEAR_REGRESSION)
        assert len(lr_models) == 1
        assert lr_models[0]['model_type'] == ModelType.LINEAR_REGRESSION

        # 测试按状态过滤
        created_models = manager.list_models(status=ModelStatus.CREATED)
        assert len(created_models) == 3

    def test_model_deletion_workflow(self):
        """测试模型删除工作流程"""
        manager = ModelManager(self.config)

        # 创建模型
        model_id = manager.create_model("test_model", ModelType.LINEAR_REGRESSION)

        # 删除模型
        result = manager.delete_model(model_id)

        # 验证删除结果
        assert result is True
        assert model_id not in manager.models
        assert model_id not in manager.model_metadata

        # 验证文件已被删除
        model_file = Path(self.config['model_storage_path']) / f"{model_id}.pkl"
        metadata_file = Path(self.config['metadata_storage_path']) / f"{model_id}.json"
        assert not model_file.exists()
        assert not metadata_file.exists()


class TestModelManagerAdvancedFeatures:
    """测试模型管理器高级功能"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'model_storage_path': str(self.temp_dir / 'models'),
            'metadata_storage_path': str(self.temp_dir / 'metadata'),
            'max_models_per_type': 2,  # 限制每个类型的模型数量
            'auto_save_interval': 1,  # 1秒自动保存
            'cache_enabled': True
        }

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_model_limit_enforcement(self):
        """测试模型数量限制"""
        manager = ModelManager(self.config)

        # 创建超过限制的模型
        model_ids = []
        for i in range(3):  # 超过max_models_per_type=2的限制
            model_id = manager.create_model(f"test_model_{i}", ModelType.LINEAR_REGRESSION)
            model_ids.append(model_id)

        # 验证只有最新的2个模型被保留
        lr_models = manager.list_models(ModelType.LINEAR_REGRESSION)
        assert len(lr_models) == 2

        # 验证最早的模型已被删除
        assert model_ids[0] not in manager.models

    def test_auto_save_functionality(self):
        """测试自动保存功能"""
        manager = ModelManager(self.config)

        # 启动自动保存
        manager.start_auto_save()

        # 创建模型
        model_id = manager.create_model("test_model", ModelType.LINEAR_REGRESSION)

        # 等待自动保存
        import time
        time.sleep(2)  # 等待超过auto_save_interval

        # 验证元数据文件被创建
        metadata_file = Path(self.config['metadata_storage_path']) / f"{model_id}.json"
        assert metadata_file.exists()

        # 停止自动保存
        manager.stop_auto_save()

    def test_model_caching(self):
        """测试模型缓存功能"""
        manager = ModelManager(self.config)

        # 创建和训练模型
        np.random.seed(42)
        X = pd.DataFrame({'feature1': np.random.randn(20), 'feature2': np.random.randn(20)})
        y = pd.Series(X['feature1'] * 2 + np.random.randn(20) * 0.1)
        training_data = pd.concat([X, y.rename('target')], axis=1)

        model_id = manager.create_model("test_lr", ModelType.LINEAR_REGRESSION)
        manager.train_model(model_id, training_data, target_column='target',
                          feature_columns=['feature1', 'feature2'])

        # 第一次预测（应该加载到缓存）
        input_data = {'feature1': 1.0, 'feature2': -0.5}
        prediction1 = manager.predict("test_lr", input_data)

        # 验证缓存
        assert "test_lr" in manager._model_cache

        # 第二次预测（应该使用缓存）
        prediction2 = manager.predict("test_lr", input_data)

        # 验证结果一致
        assert prediction1.prediction_value == prediction2.prediction_value

    def test_error_handling(self):
        """测试错误处理"""
        manager = ModelManager(self.config)

        # 测试预测不存在的模型
        with pytest.raises(ValueError):
            manager.predict("non_existent_model", {'feature1': 1.0})

        # 测试删除不存在的模型
        result = manager.delete_model("non_existent_model")
        assert result is False

        # 测试无效的训练数据
        model_id = manager.create_model("test_model", ModelType.LINEAR_REGRESSION)

        # 空数据
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            manager.train_model(model_id, empty_data, target_column='target', feature_columns=['feature1'])

        # 缺少目标列
        invalid_data = pd.DataFrame({'feature1': [1, 2, 3]})
        with pytest.raises(ValueError):
            manager.train_model(model_id, invalid_data, target_column='non_existent_target', feature_columns=['feature1'])

    def test_concurrent_access(self):
        """测试并发访问"""
        manager = ModelManager(self.config)

        # 创建多个模型
        model_ids = []
        for i in range(5):
            model_id = manager.create_model(f"concurrent_model_{i}", ModelType.LINEAR_REGRESSION)
            model_ids.append(model_id)

        # 并发训练模型
        import threading
        results = []
        errors = []

        def train_model_worker(model_id, index):
            try:
                # 创建训练数据
                np.random.seed(index)
                X = pd.DataFrame({
                    'feature1': np.random.randn(15),
                    'feature2': np.random.randn(15)
                })
                y = pd.Series(X['feature1'] * 2 + np.random.randn(15) * 0.1)
                training_data = pd.concat([X, y.rename('target')], axis=1)

                result = manager.train_model(model_id, training_data,
                                           target_column='target',
                                           feature_columns=['feature1', 'feature2'])
                results.append((model_id, result))
            except Exception as e:
                errors.append((model_id, str(e)))

        # 启动多个线程
        threads = []
        for i, model_id in enumerate(model_ids):
            thread = threading.Thread(target=train_model_worker, args=(model_id, i))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 5  # 所有模型都应该训练成功
        assert len(errors) == 0   # 不应该有错误

        for model_id, success in results:
            assert success is True
            assert manager.model_metadata[model_id].status == ModelStatus.TRAINED

    def test_model_statistics(self):
        """测试模型统计信息"""
        manager = ModelManager(self.config)

        # 创建多个不同类型的模型
        model_types = [ModelType.LINEAR_REGRESSION, ModelType.RANDOM_FOREST, ModelType.XGBOOST]
        model_ids = []

        for model_type in model_types:
            model_id = manager.create_model(f"stats_model_{model_type.value}", model_type)
            model_ids.append(model_id)

            # 为部分模型添加训练数据
            if model_type == ModelType.LINEAR_REGRESSION:
                np.random.seed(42)
                X = pd.DataFrame({'feature1': np.random.randn(20), 'feature2': np.random.randn(20)})
                y = pd.Series(X['feature1'] * 2 + np.random.randn(20) * 0.1)
                training_data = pd.concat([X, y.rename('target')], axis=1)
                manager.train_model(model_id, training_data, target_column='target',
                                  feature_columns=['feature1', 'feature2'])

        # 获取统计信息
        stats = manager.get_model_statistics()

        assert isinstance(stats, dict)
        assert 'total_models' in stats
        assert 'models_by_type' in stats
        assert 'models_by_status' in stats
        assert 'total_storage_size' in stats

        # 验证统计数据
        assert stats['total_models'] == 3
        assert ModelType.LINEAR_REGRESSION.value in stats['models_by_type']
        assert ModelStatus.CREATED.value in stats['models_by_status']
        assert ModelStatus.TRAINED.value in stats['models_by_status']


if __name__ == "__main__":
    pytest.main([__file__])
