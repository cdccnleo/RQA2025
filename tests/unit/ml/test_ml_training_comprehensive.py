#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML层 - 模型训练综合测试

测试训练流程、超参数调优、模型选择、训练监控
"""

import pytest

pytestmark = pytest.mark.legacy
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


class TestModelTraining:
    """测试模型训练流程"""
    
    @pytest.fixture
    def classification_dataset(self):
        """创建分类数据集"""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def test_train_logistic_regression(self, classification_dataset):
        """测试训练逻辑回归模型"""
        X_train, X_test, y_train, y_test = classification_dataset
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'coef_')
        assert model.score(X_train, y_train) > 0.5
    
    def test_train_decision_tree(self, classification_dataset):
        """测试训练决策树模型"""
        X_train, X_test, y_train, y_test = classification_dataset
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        assert hasattr(model, 'tree_')
        assert model.score(X_train, y_train) > 0.5
    
    def test_training_with_validation_split(self, classification_dataset):
        """测试带验证集的训练"""
        X_train, X_test, y_train, y_test = classification_dataset
        
        # 再分出验证集
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_tr, y_tr)
        
        # 在验证集上评估
        val_score = model.score(X_val, y_val)
        
        assert val_score > 0.3


class TestHyperparameterTuning:
    """测试超参数调优"""
    
    @pytest.fixture
    def tuning_dataset(self):
        """创建调优数据集"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        return X, y
    
    def test_grid_search(self, tuning_dataset):
        """测试网格搜索"""
        X, y = tuning_dataset
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
        
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X, y)
        
        assert hasattr(grid_search, 'best_params_')
        assert grid_search.best_score_ > 0
    
    def test_manual_parameter_search(self, tuning_dataset):
        """测试手动参数搜索"""
        X, y = tuning_dataset
        
        # 测试不同的C值
        best_score = 0
        best_c = None
        
        for c in [0.1, 1.0, 10.0]:
            model = LogisticRegression(C=c, random_state=42, max_iter=1000)
            model.fit(X, y)
            score = model.score(X, y)
            
            if score > best_score:
                best_score = score
                best_c = c
        
        assert best_c is not None
        assert best_score > 0
    
    def test_learning_rate_tuning(self):
        """测试学习率调优"""
        learning_rates = [0.001, 0.01, 0.1]
        
        # 模拟不同学习率的效果
        results = {}
        for lr in learning_rates:
            results[lr] = 0.8 + np.random.random() * 0.1  # 模拟性能
        
        best_lr = max(results, key=results.get)
        
        assert best_lr in learning_rates


class TestModelSelection:
    """测试模型选择"""
    
    @pytest.fixture
    def selection_dataset(self):
        """创建模型选择数据集"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test
    
    def test_compare_multiple_models(self, selection_dataset):
        """测试比较多个模型"""
        X_train, X_test, y_train, y_test = selection_dataset
        
        models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'tree': DecisionTreeClassifier(random_state=42)
        }
        
        scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            scores[name] = model.score(X_test, y_test)
        
        assert len(scores) == 2
        assert all(score > 0 for score in scores.values())
    
    def test_select_best_model(self, selection_dataset):
        """测试选择最佳模型"""
        X_train, X_test, y_train, y_test = selection_dataset
        
        models = {
            'model1': LogisticRegression(random_state=42, max_iter=1000),
            'model2': DecisionTreeClassifier(random_state=42, max_depth=3)
        }
        
        scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            scores[name] = model.score(X_test, y_test)
        
        best_model_name = max(scores, key=scores.get)
        
        assert best_model_name in models
    
    def test_ensemble_model_selection(self):
        """测试集成模型选择"""
        # 模拟多个基模型的性能
        base_models = {
            'model_a': 0.85,
            'model_b': 0.82,
            'model_c': 0.88
        }
        
        # 选择top-2模型进行集成
        top_2 = sorted(base_models.items(), key=lambda x: x[1], reverse=True)[:2]
        
        assert len(top_2) == 2
        assert top_2[0][0] == 'model_c'  # 最佳模型


class TestTrainingMonitoring:
    """测试训练监控"""
    
    def test_track_training_loss(self):
        """测试跟踪训练损失"""
        # 模拟训练过程中的损失
        training_losses = []
        
        for epoch in range(10):
            loss = 1.0 / (epoch + 1)  # 模拟损失下降
            training_losses.append(loss)
        
        # 验证损失下降
        assert training_losses[0] > training_losses[-1]
        assert len(training_losses) == 10
    
    def test_track_validation_metrics(self):
        """测试跟踪验证指标"""
        validation_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': []
        }
        
        # 模拟5个epoch的验证指标
        for _ in range(5):
            validation_metrics['accuracy'].append(0.7 + np.random.random() * 0.2)
            validation_metrics['precision'].append(0.6 + np.random.random() * 0.3)
            validation_metrics['recall'].append(0.65 + np.random.random() * 0.25)
        
        assert all(len(v) == 5 for v in validation_metrics.values())
    
    def test_early_stopping(self):
        """测试早停机制"""
        val_losses = [1.0, 0.8, 0.7, 0.72, 0.71, 0.73]  # 从epoch 3开始不再下降
        patience = 2
        
        best_loss = float('inf')
        epochs_no_improve = 0
        should_stop = False
        
        for epoch, loss in enumerate(val_losses):
            if loss < best_loss:
                best_loss = loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                should_stop = True
                break
        
        assert should_stop is True


class TestTrainingOptimization:
    """测试训练优化"""
    
    def test_batch_training(self):
        """测试批量训练"""
        X = np.random.rand(1000, 10)
        y = np.random.randint(0, 2, 1000)
        
        batch_size = 32
        n_batches = len(X) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            assert len(X_batch) == batch_size
    
    def test_data_augmentation(self):
        """测试数据增强"""
        original_data = np.array([[1, 2], [3, 4], [5, 6]])
        
        # 添加噪声进行增强
        noise = np.random.normal(0, 0.1, original_data.shape)
        augmented_data = original_data + noise
        
        assert augmented_data.shape == original_data.shape
    
    def test_learning_rate_scheduling(self):
        """测试学习率调度"""
        initial_lr = 0.1
        decay_rate = 0.95
        
        learning_rates = []
        lr = initial_lr
        
        for epoch in range(10):
            learning_rates.append(lr)
            lr *= decay_rate
        
        # 验证学习率递减
        assert learning_rates[0] > learning_rates[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

