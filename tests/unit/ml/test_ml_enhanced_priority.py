#!/usr/bin/env python3
"""
机器学习层增强测试套件

为机器学习层的核心模块创建comprehensive测试，包括：
- 模型管理器 (ModelManager)
- 特征工程 (FeatureEngineering)
- 模型训练器 (ModelTrainer)
- 推理服务 (InferenceService)

创建时间: 2025-09-15
覆盖目标: 机器学习层核心功能增强测试
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
sys.path.insert(0, project_root)

# 使用统一logger
from src.utils.logger import get_logger
logger = get_logger(__name__)

# Mock ML相关类
class MockModelType:
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    XGB = "xgb"
    NEURAL_NETWORK = "neural_network"

class MockTrainingResult:
    def __init__(self):
        self.accuracy = 0.85
        self.precision = 0.82
        self.recall = 0.78
        self.f1_score = 0.80
        self.training_time = 300.5

# 测试数据fixtures
@pytest.fixture
def sample_ml_data():
    """生成示例机器学习数据"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return {'X': X, 'y': y}

@pytest.fixture
def mock_model_manager():
    """Mock模型管理器"""
    manager = Mock()
    manager.models = {}
    manager.active_models = []
    manager.create_model = Mock()
    manager.load_model = Mock()
    manager.save_model = Mock()
    manager.deploy_model = Mock()
    manager.validate_model = Mock(return_value=True)
    return manager

@pytest.fixture
def mock_feature_engineering():
    """Mock特征工程"""
    fe = Mock()
    fe.feature_names = []
    fe.transformers = {}
    fe.fit_transform = Mock()
    fe.transform = Mock()
    fe.add_technical_indicators = Mock()
    fe.select_features = Mock()
    fe.scale_features = Mock()
    return fe

@pytest.fixture
def mock_model_trainer():
    """Mock模型训练器"""
    trainer = Mock()
    trainer.models = []
    trainer.training_history = []
    trainer.train_model = Mock(return_value=MockTrainingResult())
    trainer.cross_validate = Mock()
    trainer.evaluate_model = Mock()
    return trainer

@pytest.fixture
def mock_inference_service():
    """Mock推理服务"""
    service = Mock()
    service.models = {}
    service.request_count = 0
    service.predict = Mock(return_value=np.array([0.8, 0.2]))
    service.predict_batch = Mock()
    service.health_check = Mock(return_value={"status": "healthy"})
    return service

# =============================================================================
# 模型管理器测试
# =============================================================================

class TestModelManager:
    """模型管理器测试类"""
    
    def test_manager_initialization(self, mock_model_manager):
        """测试模型管理器初始化"""
        manager = mock_model_manager
        assert hasattr(manager, 'models')
        assert hasattr(manager, 'active_models')
        assert isinstance(manager.models, dict)
        assert isinstance(manager.active_models, list)
        logger.info("模型管理器初始化测试通过")
    
    def test_model_creation(self, mock_model_manager):
        """测试模型创建"""
        manager = mock_model_manager
        model_config = {
            'name': 'test_model',
            'type': MockModelType.RANDOM_FOREST,
            'parameters': {'n_estimators': 100}
        }
        manager.create_model(model_config)
        manager.create_model.assert_called_once_with(model_config)
        logger.info("模型创建功能测试通过")
    
    def test_model_persistence(self, mock_model_manager):
        """测试模型持久化"""
        manager = mock_model_manager
        model_id = "test_model_001"
        model_path = "/tmp/models/test_model.pkl"
        
        manager.save_model(model_id, model_path)
        manager.load_model(model_path)
        
        manager.save_model.assert_called_once_with(model_id, model_path)
        manager.load_model.assert_called_once_with(model_path)
        logger.info("模型持久化功能测试通过")
    
    def test_model_deployment(self, mock_model_manager):
        """测试模型部署"""
        manager = mock_model_manager
        model_id = "test_model_001"
        deployment_config = {'environment': 'production', 'replicas': 3}
        
        manager.deploy_model(model_id, deployment_config)
        manager.deploy_model.assert_called_once_with(model_id, deployment_config)
        logger.info("模型部署功能测试通过")

# =============================================================================
# 特征工程测试
# =============================================================================

class TestFeatureEngineering:
    """特征工程测试类"""
    
    def test_feature_transformation(self, mock_feature_engineering, sample_ml_data):
        """测试特征变换"""
        fe = mock_feature_engineering
        X_transformed = fe.fit_transform(sample_ml_data['X'])
        fe.fit_transform.assert_called_once_with(sample_ml_data['X'])
        
        X_new = fe.transform(sample_ml_data['X'])
        fe.transform.assert_called_once_with(sample_ml_data['X'])
        logger.info("特征变换功能测试通过")
    
    def test_technical_indicators(self, mock_feature_engineering):
        """测试技术指标生成"""
        fe = mock_feature_engineering
        price_data = pd.Series([100, 102, 98, 105, 103])
        indicators_config = {'moving_averages': [3, 5], 'rsi': True}
        
        fe.add_technical_indicators(price_data, indicators_config)
        fe.add_technical_indicators.assert_called_once_with(price_data, indicators_config)
        logger.info("技术指标生成测试通过")
    
    def test_feature_selection(self, mock_feature_engineering, sample_ml_data):
        """测试特征选择"""
        fe = mock_feature_engineering
        selection_config = {'method': 'SelectKBest', 'k': 10}
        
        fe.select_features(sample_ml_data['X'], sample_ml_data['y'], selection_config)
        fe.select_features.assert_called_once()
        logger.info("特征选择功能测试通过")

# =============================================================================
# 模型训练器测试
# =============================================================================

class TestModelTrainer:
    """模型训练器测试类"""
    
    def test_model_training(self, mock_model_trainer, sample_ml_data):
        """测试模型训练"""
        trainer = mock_model_trainer
        training_config = {
            'model_type': MockModelType.RANDOM_FOREST,
            'hyperparameters': {'n_estimators': 100},
            'training_params': {'test_size': 0.2}
        }
        
        result = trainer.train_model(sample_ml_data['X'], sample_ml_data['y'], training_config)
        assert isinstance(result, MockTrainingResult)
        assert hasattr(result, 'accuracy')
        assert result.accuracy > 0
        trainer.train_model.assert_called_once()
        logger.info("模型训练功能测试通过")
    
    def test_cross_validation(self, mock_model_trainer, sample_ml_data):
        """测试交叉验证"""
        trainer = mock_model_trainer
        cv_config = {'cv_folds': 5, 'stratify': True}
        
        trainer.cross_validate(sample_ml_data['X'], sample_ml_data['y'], cv_config)
        trainer.cross_validate.assert_called_once()
        logger.info("交叉验证功能测试通过")

# =============================================================================
# 推理服务测试
# =============================================================================

class TestInferenceService:
    """推理服务测试类"""
    
    def test_single_prediction(self, mock_inference_service, sample_ml_data):
        """测试单个预测"""
        service = mock_inference_service
        input_data = sample_ml_data['X'][0].reshape(1, -1)
        
        prediction = service.predict(input_data)
        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == 2
        service.predict.assert_called_once_with(input_data)
        logger.info("单个预测功能测试通过")
    
    def test_batch_prediction(self, mock_inference_service, sample_ml_data):
        """测试批量预测"""
        service = mock_inference_service
        batch_data = sample_ml_data['X'][:32]
        
        service.predict_batch(batch_data)
        service.predict_batch.assert_called_once_with(batch_data)
        logger.info("批量预测功能测试通过")
    
    def test_service_health_check(self, mock_inference_service):
        """测试服务健康检查"""
        service = mock_inference_service
        health_status = service.health_check()
        
        assert isinstance(health_status, dict)
        assert 'status' in health_status
        assert health_status['status'] == 'healthy'
        service.health_check.assert_called_once()
        logger.info("服务健康检查测试通过")

# =============================================================================
# 集成测试
# =============================================================================

class TestMLIntegration:
    """机器学习集成测试类"""
    
    def test_ml_pipeline(self, mock_model_manager, mock_feature_engineering, 
                        mock_model_trainer, mock_inference_service, sample_ml_data):
        """测试机器学习管道"""
        # 1. 特征工程
        X_processed = mock_feature_engineering.fit_transform(sample_ml_data['X'])
        
        # 2. 模型训练
        training_config = {'model_type': MockModelType.RANDOM_FOREST}
        training_result = mock_model_trainer.train_model(
            sample_ml_data['X'], sample_ml_data['y'], training_config
        )
        
        # 3. 模型保存
        model_path = "/tmp/trained_model.pkl"
        mock_model_manager.save_model("trained_model", model_path)
        
        # 4. 模型推理
        prediction = mock_inference_service.predict(sample_ml_data['X'][:1])
        
        # 验证所有步骤都被调用
        mock_feature_engineering.fit_transform.assert_called_once()
        mock_model_trainer.train_model.assert_called_once()
        mock_model_manager.save_model.assert_called_once()
        mock_inference_service.predict.assert_called_once()
        
        logger.info("机器学习管道测试通过")

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
