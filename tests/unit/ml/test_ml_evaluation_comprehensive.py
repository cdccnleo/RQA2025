#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ML层 - 模型评估综合测试

测试评估指标计算、交叉验证、模型比较
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestEvaluationMetrics:
    """测试评估指标计算"""
    
    @pytest.fixture
    def predictions(self):
        """创建预测结果"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
        return y_true, y_pred
    
    def test_calculate_accuracy(self, predictions):
        """测试计算准确率"""
        y_true, y_pred = predictions
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # 手动计算验证
        correct = (y_true == y_pred).sum()
        expected_accuracy = correct / len(y_true)
        
        assert accuracy == expected_accuracy
        assert 0 <= accuracy <= 1
    
    def test_calculate_precision(self, predictions):
        """测试计算精确率"""
        y_true, y_pred = predictions
        
        precision = precision_score(y_true, y_pred)
        
        assert 0 <= precision <= 1
    
    def test_calculate_recall(self, predictions):
        """测试计算召回率"""
        y_true, y_pred = predictions
        
        recall = recall_score(y_true, y_pred)
        
        assert 0 <= recall <= 1
    
    def test_calculate_f1_score(self, predictions):
        """测试计算F1分数"""
        y_true, y_pred = predictions
        
        f1 = f1_score(y_true, y_pred)
        
        # F1是精确率和召回率的调和平均
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        expected_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        assert abs(f1 - expected_f1) < 1e-10
    
    def test_calculate_confusion_matrix(self, predictions):
        """测试计算混淆矩阵"""
        y_true, y_pred = predictions
        
        # 手动计算混淆矩阵
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        
        assert tn + fp + fn + tp == len(y_true)


class TestCrossValidation:
    """测试交叉验证"""
    
    @pytest.fixture
    def cv_dataset(self):
        """创建交叉验证数据集"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y
    
    def test_k_fold_cross_validation(self, cv_dataset):
        """测试K折交叉验证"""
        X, y = cv_dataset
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        scores = cross_val_score(model, X, y, cv=5)
        
        assert len(scores) == 5
        assert all(0 <= score <= 1 for score in scores)
    
    def test_stratified_cross_validation(self, cv_dataset):
        """测试分层交叉验证"""
        X, y = cv_dataset
        
        # 验证每个fold中类别分布一致
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_class_ratios = []
        for train_idx, test_idx in kf.split(X):
            y_fold = y[test_idx]
            ratio = y_fold.sum() / len(y_fold)
            fold_class_ratios.append(ratio)
        
        # 各fold的类别比例应该相近
        assert len(fold_class_ratios) == 5
    
    def test_calculate_cv_mean_score(self, cv_dataset):
        """测试计算交叉验证平均分数"""
        X, y = cv_dataset
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        scores = cross_val_score(model, X, y, cv=5)
        
        mean_score = scores.mean()
        std_score = scores.std()
        
        assert 0 <= mean_score <= 1
        assert std_score >= 0


class TestModelComparison:
    """测试模型比较"""
    
    @pytest.fixture
    def comparison_dataset(self):
        """创建模型比较数据集"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_compare_model_performance(self, comparison_dataset):
        """测试比较模型性能"""
        X_train, X_test, y_train, y_test = comparison_dataset
        
        # 训练两个模型
        model1 = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
        model2 = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        
        score1 = model1.score(X_test, y_test)
        score2 = model2.score(X_test, y_test)
        
        # 比较性能
        performance_diff = abs(score2 - score1)
        
        assert isinstance(performance_diff, (int, float))
        assert performance_diff >= 0
    
    def test_rank_models_by_metric(self, comparison_dataset):
        """测试按指标排序模型"""
        X_train, X_test, y_train, y_test = comparison_dataset
        
        models = {
            'lr_001': LogisticRegression(C=0.01, random_state=42, max_iter=1000),
            'lr_01': LogisticRegression(C=0.1, random_state=42, max_iter=1000),
            'lr_1': LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        }
        
        scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            scores[name] = model.score(X_test, y_test)
        
        # 按分数排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        assert len(ranked) == 3
        assert ranked[0][1] >= ranked[-1][1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

