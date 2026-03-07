#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML层 - 模型部署综合测试

测试模型序列化、加载、推理服务
"""

import pytest
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


class TestModelSerialization:
    """测试模型序列化"""
    
    @pytest.fixture
    def trained_model(self):
        """创建训练好的模型"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return model, X[:5]  # 返回模型和测试数据
    
    def test_pickle_save_model(self, trained_model, tmp_path):
        """测试使用pickle保存模型"""
        model, _ = trained_model
        model_path = tmp_path / "model.pkl"
        
        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        assert model_path.exists()
        assert model_path.stat().st_size > 0
    
    def test_pickle_load_model(self, trained_model, tmp_path):
        """测试使用pickle加载模型"""
        model, X_test = trained_model
        model_path = tmp_path / "model.pkl"
        
        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 加载模型
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # 验证加载的模型可以预测
        predictions = loaded_model.predict(X_test)
        
        assert len(predictions) == len(X_test)
    
    def test_save_model_metadata(self, trained_model, tmp_path):
        """测试保存模型元数据"""
        model, _ = trained_model
        metadata_path = tmp_path / "metadata.json"
        
        metadata = {
            'model_type': type(model).__name__,
            'n_features': model.n_features_in_,
            'created_at': '2025-11-02',
            'version': '1.0.0'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        assert metadata_path.exists()


class TestModelLoading:
    """测试模型加载"""
    
    def test_load_model_from_disk(self, tmp_path):
        """测试从磁盘加载模型"""
        # 创建并保存一个简单模型
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        model_path = tmp_path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 加载模型
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        assert hasattr(loaded_model, 'predict')
    
    def test_validate_loaded_model(self, tmp_path):
        """测试验证加载的模型"""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        original_model = LogisticRegression(random_state=42, max_iter=1000)
        original_model.fit(X, y)
        
        # 保存和加载
        model_path = tmp_path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(original_model, f)
        
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # 验证预测结果一致
        original_pred = original_model.predict(X[:10])
        loaded_pred = loaded_model.predict(X[:10])
        
        assert np.array_equal(original_pred, loaded_pred)


class TestInferenceService:
    """测试推理服务"""
    
    @pytest.fixture
    def inference_model(self):
        """创建推理模型"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return model
    
    def test_single_prediction(self, inference_model):
        """测试单条预测"""
        sample = np.array([[0.5, 0.3, 0.8, 0.2, 0.9]])
        
        prediction = inference_model.predict(sample)
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
    
    def test_batch_prediction(self, inference_model):
        """测试批量预测"""
        batch_size = 10
        samples = np.random.rand(batch_size, 5)
        
        predictions = inference_model.predict(samples)
        
        assert len(predictions) == batch_size
    
    def test_prediction_probability(self, inference_model):
        """测试预测概率"""
        sample = np.array([[0.5, 0.3, 0.8, 0.2, 0.9]])
        
        probabilities = inference_model.predict_proba(sample)
        
        assert probabilities.shape == (1, 2)  # 二分类
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # 概率和为1
    
    def test_inference_latency(self, inference_model):
        """测试推理延迟"""
        import time
        
        sample = np.random.rand(1, 5)
        
        start = time.time()
        _ = inference_model.predict(sample)
        latency = time.time() - start
        
        # 推理应该很快（<100ms）
        assert latency < 0.1


class TestModelVersioning:
    """测试模型版本管理"""
    
    def test_save_model_with_version(self, tmp_path):
        """测试保存带版本的模型"""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        version = "1.0.0"
        model_path = tmp_path / f"model_v{version}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        assert model_path.exists()
    
    def test_load_specific_model_version(self, tmp_path):
        """测试加载特定版本模型"""
        # 创建多个版本
        versions = ["1.0.0", "1.1.0", "2.0.0"]
        
        for version in versions:
            model_path = tmp_path / f"model_v{version}.pkl"
            model = LogisticRegression(random_state=42)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # 加载特定版本
        target_version = "1.1.0"
        target_path = tmp_path / f"model_v{target_version}.pkl"
        
        assert target_path.exists()
    
    def test_list_available_model_versions(self, tmp_path):
        """测试列出可用模型版本"""
        # 创建多个版本
        versions = ["1.0.0", "1.1.0", "2.0.0"]
        
        for version in versions:
            model_path = tmp_path / f"model_v{version}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({}, f)
        
        # 列出所有版本
        model_files = list(tmp_path.glob("model_v*.pkl"))
        
        assert len(model_files) == 3


class TestModelMonitoring:
    """测试模型监控"""
    
    def test_track_prediction_distribution(self):
        """测试跟踪预测分布"""
        predictions = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
        
        # 统计预测分布
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        assert distribution[0] == 4
        assert distribution[1] == 6
    
    def test_detect_prediction_drift(self):
        """测试检测预测漂移"""
        # 历史预测分布
        historical_dist = {0: 0.5, 1: 0.5}  # 50-50分布
        
        # 当前预测分布
        current_predictions = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1])  # 80-20分布
        current_dist = {
            0: (current_predictions == 0).sum() / len(current_predictions),
            1: (current_predictions == 1).sum() / len(current_predictions)
        }
        
        # 检测漂移
        drift = abs(current_dist[1] - historical_dist[1])
        drift_threshold = 0.2
        
        has_drift = drift > drift_threshold
        
        assert has_drift == True  # 30%的漂移


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

