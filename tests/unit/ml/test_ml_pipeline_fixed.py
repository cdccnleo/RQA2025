#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习管道简化测试
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TestMLPipelineSimple:
    """机器学习管道简化测试"""

    def test_model_training_and_prediction(self):
        """测试模型训练和预测"""
        # 创建简单的二分类数据集
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # 生成特征
        X = np.random.randn(n_samples, n_features)

        # 生成目标变量（基于前3个特征的线性组合）
        weights = np.array([0.8, 0.6, 0.4] + [0.0] * (n_features - 3))
        linear_combination = X.dot(weights)
        prob = 1 / (1 + np.exp(-linear_combination))  # sigmoid
        y = (prob > 0.5).astype(int)

        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 训练模型
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 进行预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # 验证预测结果
        assert len(y_pred) == len(y_test), "预测结果长度不匹配"
        assert y_prob.shape == (len(y_test), 2), "预测概率形状不正确"
        assert np.all((y_prob >= 0) & (y_prob <= 1)), "预测概率不在[0,1]范围内"
        assert np.allclose(y_prob.sum(axis=1), 1.0), "预测概率每行和不为1"

        print("✅ 模型训练和预测测试通过")

    def test_model_evaluation_metrics(self):
        """测试模型评估指标"""
        # 创建已知性能的模型预测结果
        np.random.seed(42)

        # 真实的二分类标签
        y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1])

        # 模型预测结果（包含一些错误）
        y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])

        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)

        # 手动计算准确率验证
        correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        manual_accuracy = correct_predictions / len(y_true)

        assert abs(accuracy - manual_accuracy) < 1e-10, "准确率计算不正确"

        # 验证准确率在合理范围内
        assert 0.5 <= accuracy <= 1.0, f"准确率异常: {accuracy}"

        print(f"✅ 准确率验证: sklearn={accuracy:.4f}, manual={manual_accuracy:.4f}")

    def test_model_cross_validation(self):
        """测试模型交叉验证"""
        # 创建数据集
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 简单的分类规则

        # 执行简单的交叉验证（3折）
        n_splits = 3
        fold_size = len(X) // n_splits
        cv_scores = []

        for i in range(n_splits):
            # 分割数据
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(X)

            X_test = X[start_idx:end_idx]
            y_test = y[start_idx:end_idx]

            X_train = np.concatenate([X[:start_idx], X[end_idx:]], axis=0)
            y_train = np.concatenate([y[:start_idx], y[end_idx:]], axis=0)

            # 训练和评估
            model = RandomForestClassifier(n_estimators=20, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores.append(accuracy)

        # 计算交叉验证统计
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        # 验证交叉验证结果
        assert len(cv_scores) == n_splits, "交叉验证折数不正确"
        assert 0.4 <= mean_score <= 1.0, f"平均交叉验证分数异常: {mean_score}"
        assert std_score < 0.2, f"交叉验证稳定性不足: {std_score}"

        print(f"✅ 交叉验证测试: 平均分数={mean_score:.4f}, 标准差={std_score:.4f}")

    def test_model_overfitting_detection(self):
        """测试模型过拟合检测"""
        # 创建简单的训练和测试数据集
        np.random.seed(42)
        n_samples = 1000
        n_features = 20

        # 生成特征
        X = np.random.randn(n_samples, n_features)

        # 生成目标变量（只与前3个特征相关）
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)

        # 分割数据集
        split_idx = int(n_samples * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 训练复杂模型（容易过拟合）
        complex_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
        complex_model.fit(X_train, y_train)

        # 评估训练和测试性能
        train_pred = complex_model.predict(X_train)
        test_pred = complex_model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        overfitting_gap = train_accuracy - test_accuracy

        # 验证过拟合检测
        assert train_accuracy >= test_accuracy, "训练准确率应不低于测试准确率"
        assert overfitting_gap < 0.2, f"过拟合程度过高: {overfitting_gap}"

        # 对于这个数据集，训练准确率应该明显高于随机水平
        assert train_accuracy > 0.7, f"训练准确率过低: {train_accuracy}"
        assert test_accuracy > 0.6, f"测试准确率过低: {test_accuracy}"

        print(f"✅ 过拟合检测: 训练={train_accuracy:.4f}, 测试={test_accuracy:.4f}, 差距={overfitting_gap:.4f}")

    def test_model_prediction_consistency(self):
        """测试模型预测一致性"""
        # 创建数据集
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        # 训练模型
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        # 多次预测同一数据
        n_predictions = 10
        predictions = []

        test_sample = X[:5]  # 使用前5个样本

        for _ in range(n_predictions):
            pred = model.predict(test_sample)
            prob = model.predict_proba(test_sample)
            predictions.append((pred, prob))

        # 检查预测一致性
        first_pred, first_prob = predictions[0]

        for i, (pred, prob) in enumerate(predictions[1:], 1):
            # 预测结果应该完全一致（随机森林是确定性的）
            np.testing.assert_array_equal(pred, first_pred, f"第{i}次预测结果不一致")
            np.testing.assert_array_almost_equal(prob, first_prob, decimal=10, err_msg=f"第{i}次预测概率不一致")

        # 检查预测概率的合理性
        for pred, prob in predictions:
            assert np.all(prob >= 0), "预测概率包含负数"
            assert np.all(prob <= 1), "预测概率超过1"
            assert np.allclose(prob.sum(axis=1), 1.0), "预测概率每行和不为1"

        print("✅ 模型预测一致性测试通过")
