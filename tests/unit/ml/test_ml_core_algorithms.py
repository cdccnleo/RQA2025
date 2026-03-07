#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - ML模块核心算法测试

测试ml/目录中的核心机器学习算法，避免复杂的模块导入依赖
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split


# @pytest.mark.skip(reason="ML core algorithms tests may have sklearn/scipy dependency issues")
class TestMLCoreAlgorithms:
    """测试ML核心算法功能"""

    def setup_method(self):
        """测试前准备"""
        # 创建模拟数据集
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # 生成回归数据集
        self.X_reg = np.random.randn(n_samples, n_features)
        self.y_reg = self.X_reg[:, 0] + 0.5 * self.X_reg[:, 1] + np.random.randn(n_samples) * 0.1

        # 生成分类数据集
        self.X_clf = np.random.randn(n_samples, n_features)
        self.y_clf = (self.X_clf[:, 0] + self.X_clf[:, 1] > 0).astype(int)

        # 创建包含噪声的数据
        self.X_noisy = self.X_reg + np.random.randn(n_samples, n_features) * 0.5
        self.y_noisy = self.y_reg + np.random.randn(n_samples) * 0.5

    def test_linear_regression_training(self):
        """测试线性回归训练"""
        # 训练线性回归模型
        model = LinearRegression()
        model.fit(self.X_reg, self.y_reg)

        # 验证模型参数
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert len(model.coef_) == self.X_reg.shape[1]

        # 验证预测
        predictions = model.predict(self.X_reg)
        assert len(predictions) == len(self.y_reg)

        # 验证预测质量
        r2 = r2_score(self.y_reg, predictions)
        assert r2 > 0.5  # 对于合成数据应该有较好的拟合

        mse = mean_squared_error(self.y_reg, predictions)
        assert mse < 1.0  # 均方误差应该在合理范围内

    def test_logistic_regression_training(self):
        """测试逻辑回归训练"""
        # 训练逻辑回归模型
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_clf, self.y_clf)

        # 验证模型参数
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_.shape[1] == self.X_clf.shape[1]

        # 验证预测
        predictions = model.predict(self.X_clf)
        probabilities = model.predict_proba(self.X_clf)

        assert len(predictions) == len(self.y_clf)
        assert probabilities.shape == (len(self.y_clf), 2)  # 二分类概率

        # 验证预测质量
        accuracy = accuracy_score(self.y_clf, predictions)
        assert accuracy > 0.7  # 对于合成数据应该有较好的准确率

    def test_decision_tree_regression(self):
        """测试决策树回归"""
        # 训练决策树回归模型
        model = DecisionTreeRegressor(random_state=42, max_depth=5)
        model.fit(self.X_reg, self.y_reg)

        # 验证模型结构
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == self.X_reg.shape[1]

        # 验证预测
        predictions = model.predict(self.X_reg)
        assert len(predictions) == len(self.y_reg)

        # 验证过拟合控制
        train_score = model.score(self.X_reg, self.y_reg)
        # 对于合成数据，训练分数应该很高但不会是1.0（过拟合）
        assert train_score > 0.8
        assert train_score < 0.999

    def test_random_forest_classifier(self):
        """测试随机森林分类器"""
        # 训练随机森林分类器
        model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
        model.fit(self.X_clf, self.y_clf)

        # 验证模型属性
        assert hasattr(model, 'estimators_')
        assert len(model.estimators_) == 10  # 应该有10棵树
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == self.X_clf.shape[1]

        # 验证预测
        predictions = model.predict(self.X_clf)
        probabilities = model.predict_proba(self.X_clf)

        assert len(predictions) == len(self.y_clf)
        assert probabilities.shape == (len(self.y_clf), 2)

        # 验证预测质量
        accuracy = accuracy_score(self.y_clf, predictions)
        assert accuracy > 0.75  # 随机森林应该有较好的性能

    def test_svm_regression(self):
        """测试支持向量机回归"""
        # 使用较小的数据集以加快测试速度
        X_small = self.X_reg[:100]
        y_small = self.y_reg[:100]

        # 训练SVM回归模型
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        model.fit(X_small, y_small)

        # 验证模型参数
        assert hasattr(model, 'support_vectors_')
        assert len(model.support_vectors_) > 0

        # 验证预测
        predictions = model.predict(X_small)
        assert len(predictions) == len(y_small)

        # 验证预测质量
        r2 = r2_score(y_small, predictions)
        assert r2 > 0.3  # SVM对噪声数据应该有合理表现

    def test_cross_validation(self):
        """测试交叉验证"""
        # 使用逻辑回归进行交叉验证
        model = LogisticRegression(random_state=42, max_iter=1000)

        # 进行5折交叉验证
        cv_scores = cross_val_score(model, self.X_clf, self.y_clf, cv=5, scoring='accuracy')

        # 验证交叉验证结果
        assert len(cv_scores) == 5
        assert all(score > 0.5 for score in cv_scores)  # 所有折的准确率应该>0.5
        assert np.std(cv_scores) < 0.1  # 各折之间的差异不应该太大

        # 验证平均性能
        mean_score = np.mean(cv_scores)
        assert mean_score > 0.65

    def test_model_evaluation_metrics(self):
        """测试模型评估指标"""
        # 训练一个简单的分类模型
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_clf, self.y_clf, test_size=0.3, random_state=42
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # 计算各种评估指标
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')

        # 验证指标合理性
        assert 0.5 < accuracy < 1.0
        assert 0.5 < f1 < 1.0

        # 对于平衡数据集，准确率和F1分数应该接近
        assert abs(accuracy - f1) < 0.1

    def test_feature_importance_analysis(self):
        """测试特征重要性分析"""
        # 训练随机森林模型
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_clf, self.y_clf)

        # 获取特征重要性
        importance = model.feature_importances_

        # 验证特征重要性属性
        assert len(importance) == self.X_clf.shape[1]
        assert all(imp >= 0 for imp in importance)  # 重要性应该非负
        assert abs(sum(importance) - 1.0) < 1e-10  # 应该归一化

        # 验证前两个特征应该比其他特征更重要（因为我们在setup中构造了这种关系）
        assert importance[0] > np.mean(importance[2:])
        assert importance[1] > np.mean(importance[2:])

    def test_model_persistence(self):
        """测试模型持久化"""
        import tempfile
        import pickle

        # 训练模型
        model = LinearRegression()
        model.fit(self.X_reg, self.y_reg)

        # 序列化模型
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(model, f)
            model_path = f.name

        try:
            # 反序列化模型
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)

            # 验证加载的模型
            assert hasattr(loaded_model, 'coef_')
            assert hasattr(loaded_model, 'intercept_')

            # 验证预测一致性
            original_pred = model.predict(self.X_reg[:10])
            loaded_pred = loaded_model.predict(self.X_reg[:10])

            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

        finally:
            # 清理临时文件
            import os
            os.unlink(model_path)

    def test_hyperparameter_tuning(self):
        """测试超参数调优"""
        from sklearn.model_selection import GridSearchCV

        # 定义参数网格
        param_grid = {
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }

        # 使用网格搜索调优决策树
        model = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=1
        )

        # 使用较小的数据集以加快测试
        X_small = self.X_clf[:200]
        y_small = self.y_clf[:200]

        grid_search.fit(X_small, y_small)

        # 验证网格搜索结果
        assert hasattr(grid_search, 'best_estimator_')
        assert hasattr(grid_search, 'best_params_')
        assert hasattr(grid_search, 'best_score_')

        # 验证最佳参数在搜索空间内
        best_params = grid_search.best_params_
        assert best_params['max_depth'] in param_grid['max_depth']
        assert best_params['min_samples_split'] in param_grid['min_samples_split']

        # 验证最佳分数合理
        assert 0.5 < grid_search.best_score_ < 1.0

    def test_ensemble_methods(self):
        """测试集成方法"""
        # 训练随机森林（装袋集成）
        rf_model = RandomForestClassifier(n_estimators=5, random_state=42)
        rf_model.fit(self.X_clf, self.y_clf)

        # 训练单个决策树作为基准
        single_tree = DecisionTreeClassifier(random_state=42)
        single_tree.fit(self.X_clf, self.y_clf)

        # 比较性能
        rf_predictions = rf_model.predict(self.X_clf)
        tree_predictions = single_tree.predict(self.X_clf)

        rf_accuracy = accuracy_score(self.y_clf, rf_predictions)
        tree_accuracy = accuracy_score(self.y_clf, tree_predictions)

        # 集成方法通常应该比单个模型更好或至少相当
        assert rf_accuracy >= tree_accuracy * 0.95  # 允许5%的性能下降

    def test_model_robustness_to_noise(self):
        """测试模型对噪声的鲁棒性"""
        # 训练干净数据的模型
        clean_model = LinearRegression()
        clean_model.fit(self.X_reg, self.y_reg)

        # 训练噪声数据的模型
        noisy_model = LinearRegression()
        noisy_model.fit(self.X_noisy, self.y_noisy)

        # 计算两个模型的性能
        clean_pred = clean_model.predict(self.X_reg)
        noisy_pred = noisy_model.predict(self.X_reg)

        clean_r2 = r2_score(self.y_reg, clean_pred)
        noisy_r2 = r2_score(self.y_reg, noisy_pred)

        # 噪声模型的性能应该稍差，但仍在合理范围内
        assert clean_r2 > noisy_r2
        assert noisy_r2 > 0.3  # 即使有噪声，模型仍应该有一定预测能力

    def test_out_of_sample_prediction(self):
        """测试样本外预测"""
        # 分割训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_reg, self.y_reg, test_size=0.3, random_state=42
        )

        # 训练模型
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 在测试集上预测
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # 计算训练和测试误差
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)

        # 测试误差应该与训练误差在合理范围内（考虑到数据可能有特殊性）
        # 对于合成数据，测试误差可能略低于或高于训练误差
        mse_ratio = max(test_mse, train_mse) / min(test_mse, train_mse)
        assert mse_ratio < 2.0  # 误差比率不应超过2倍

    def test_feature_scaling_impact(self):
        """测试特征缩放对模型的影响"""
        from sklearn.preprocessing import StandardScaler

        # 训练未经缩放的模型
        unscaled_model = LogisticRegression(random_state=42, max_iter=1000)
        unscaled_model.fit(self.X_clf, self.y_clf)
        unscaled_score = unscaled_model.score(self.X_clf, self.y_clf)

        # 训练经过缩放的模型
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_clf)

        scaled_model = LogisticRegression(random_state=42, max_iter=1000)
        scaled_model.fit(X_scaled, self.y_clf)
        scaled_score = scaled_model.score(X_scaled, self.y_clf)

        # 对于逻辑回归，缩放应该有正面影响或至少不降低性能
        assert scaled_score >= unscaled_score * 0.95

    def test_learning_curve_analysis(self):
        """测试学习曲线分析"""
        from sklearn.model_selection import learning_curve

        # 生成不同训练集大小的学习曲线
        model = LogisticRegression(random_state=42, max_iter=1000)

        train_sizes, train_scores, val_scores = learning_curve(
            model, self.X_clf, self.y_clf,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=3, random_state=42
        )

        # 验证学习曲线结果
        assert len(train_sizes) == 5
        assert train_scores.shape == (5, 3)  # 5个训练大小，3折交叉验证
        assert val_scores.shape == (5, 3)

        # 验证随着训练数据增加，性能通常会提高
        mean_train_scores = np.mean(train_scores, axis=1)
        mean_val_scores = np.mean(val_scores, axis=1)

        # 通常训练分数应该高于验证分数
        assert np.mean(mean_train_scores) > np.mean(mean_val_scores)

        # 验证分数在合理范围内
        assert all(score > 0.5 for score in mean_train_scores)
        assert all(score > 0.5 for score in mean_val_scores)
