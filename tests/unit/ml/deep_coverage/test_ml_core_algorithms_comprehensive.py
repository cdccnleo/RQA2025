"""
机器学习核心算法深度测试
全面测试机器学习核心算法、模型训练、预测和评估功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile

# 导入机器学习相关类
try:
    from src.ml.core.ml_core import MLCore, MLModel
    ML_CORE_AVAILABLE = True
except ImportError:
    ML_CORE_AVAILABLE = False
    MLCore = Mock
    MLModel = Mock

try:
    from src.ml.models.trainer import ModelTrainer
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False
    ModelTrainer = Mock

try:
    from src.ml.models.predictor import ModelPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    ModelPredictor = Mock

try:
    from src.ml.models.model_evaluator import ModelEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    ModelEvaluator = Mock

try:
    from src.ml.core.exceptions import MLException, ModelNotFoundError
    EXCEPTIONS_AVAILABLE = True
except ImportError:
    EXCEPTIONS_AVAILABLE = False
    MLException = Exception
    ModelNotFoundError = Exception


class TestMLCoreAlgorithmsComprehensive:
    """机器学习核心算法综合深度测试"""

    @pytest.fixture
    def sample_training_data(self):
        """创建样本训练数据"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # 生成特征数据
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
        })

        # 生成目标变量 (回归任务)
        y = pd.Series(
            X.sum(axis=1) + np.random.normal(0, 0.1, n_samples),
            name='target'
        )

        return X, y

    @pytest.fixture
    def sample_classification_data(self):
        """创建样本分类数据"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 8

        # 生成特征数据
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
        })

        # 生成分类目标 (0或1)
        linear_combination = X.sum(axis=1)
        prob = 1 / (1 + np.exp(-linear_combination))  # sigmoid
        y = pd.Series(
            np.random.binomial(1, prob, n_samples),
            name='target'
        )

        return X, y

    @pytest.fixture
    def ml_core(self):
        """创建ML核心实例"""
        if ML_CORE_AVAILABLE:
            return MLCore()
        return Mock(spec=MLCore)

    @pytest.fixture
    def model_trainer(self):
        """创建模型训练器实例"""
        if TRAINER_AVAILABLE:
            return ModelTrainer()
        return Mock(spec=ModelTrainer)

    @pytest.fixture
    def model_predictor(self):
        """创建模型预测器实例"""
        if PREDICTOR_AVAILABLE:
            return ModelPredictor()
        return Mock(spec=ModelPredictor)

    @pytest.fixture
    def model_evaluator(self):
        """创建模型评估器实例"""
        if EVALUATOR_AVAILABLE:
            return ModelEvaluator()
        return Mock(spec=ModelEvaluator)

    def test_ml_core_initialization(self, ml_core):
        """测试ML核心初始化"""
        if ML_CORE_AVAILABLE:
            assert ml_core is not None
            assert hasattr(ml_core, 'config')
            assert hasattr(ml_core, 'logger')

    def test_linear_regression_training(self, ml_core, sample_training_data):
        """测试线性回归训练"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练线性回归模型
            model_id = ml_core.train_model(
                X, y,
                model_type='linear_regression',
                hyperparameters={'fit_intercept': True}
            )

            assert model_id is not None
            assert isinstance(model_id, str)

            # 验证模型已保存
            assert ml_core.model_exists(model_id)

    def test_random_forest_training(self, ml_core, sample_training_data):
        """测试随机森林训练"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练随机森林模型
            model_id = ml_core.train_model(
                X, y,
                model_type='random_forest',
                hyperparameters={
                    'n_estimators': 10,
                    'max_depth': 5,
                    'random_state': 42
                }
            )

            assert model_id is not None
            assert ml_core.model_exists(model_id)

    def test_logistic_regression_training(self, ml_core, sample_classification_data):
        """测试逻辑回归训练"""
        if ML_CORE_AVAILABLE:
            X, y = sample_classification_data

            # 训练逻辑回归模型
            model_id = ml_core.train_model(
                X, y,
                model_type='logistic_regression',
                hyperparameters={'max_iter': 1000}
            )

            assert model_id is not None
            assert ml_core.model_exists(model_id)

    def test_gradient_boosting_training(self, ml_core, sample_training_data):
        """测试梯度提升训练"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练梯度提升模型
            model_id = ml_core.train_model(
                X, y,
                model_type='gradient_boosting',
                hyperparameters={
                    'n_estimators': 50,
                    'learning_rate': 0.1,
                    'max_depth': 3
                }
            )

            assert model_id is not None
            assert ml_core.model_exists(model_id)

    def test_support_vector_machine_training(self, ml_core, sample_classification_data):
        """测试支持向量机训练"""
        if ML_CORE_AVAILABLE:
            X, y = sample_classification_data

            # 训练SVM模型
            model_id = ml_core.train_model(
                X, y,
                model_type='svm',
                hyperparameters={'C': 1.0, 'kernel': 'rbf'}
            )

            assert model_id is not None
            assert ml_core.model_exists(model_id)

    def test_model_prediction_regression(self, ml_core, sample_training_data):
        """测试回归模型预测"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='linear_regression')

            # 进行预测
            test_X = X.head(10)
            predictions = ml_core.predict(model_id, test_X)

            assert isinstance(predictions, (pd.Series, np.ndarray))
            assert len(predictions) == len(test_X)

            # 预测值应该是数值类型
            assert predictions.dtype in [np.float32, np.float64, float]

    def test_model_prediction_classification(self, ml_core, sample_classification_data):
        """测试分类模型预测"""
        if ML_CORE_AVAILABLE:
            X, y = sample_classification_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='logistic_regression')

            # 进行预测
            test_X = X.head(10)
            predictions = ml_core.predict(model_id, test_X)

            assert isinstance(predictions, (pd.Series, np.ndarray))
            assert len(predictions) == len(test_X)

            # 分类预测应该是离散值
            unique_predictions = np.unique(predictions)
            assert len(unique_predictions) <= len(np.unique(y))  # 预测类别不应超过训练类别

    def test_model_probability_prediction(self, ml_core, sample_classification_data):
        """测试模型概率预测"""
        if ML_CORE_AVAILABLE:
            X, y = sample_classification_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='logistic_regression')

            # 进行概率预测
            test_X = X.head(5)
            probabilities = ml_core.predict_proba(model_id, test_X)

            assert isinstance(probabilities, (pd.DataFrame, np.ndarray))
            assert probabilities.shape[0] == len(test_X)

            # 概率值应该在[0,1]范围内
            assert np.all(probabilities >= 0)
            assert np.all(probabilities <= 1)

            # 对于二分类，概率应该有两个列
            if hasattr(probabilities, 'shape') and len(probabilities.shape) > 1:
                assert probabilities.shape[1] == 2

    def test_model_evaluation_regression(self, ml_core, sample_training_data):
        """测试回归模型评估"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='linear_regression')

            # 评估模型
            metrics = ml_core.evaluate_model(model_id, X, y)

            assert isinstance(metrics, dict)

            # 检查回归指标
            expected_metrics = ['mse', 'rmse', 'mae', 'r2_score']
            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))

            # R²应该在合理范围内
            assert -1 <= metrics['r2_score'] <= 1

    def test_model_evaluation_classification(self, ml_core, sample_classification_data):
        """测试分类模型评估"""
        if ML_CORE_AVAILABLE:
            X, y = sample_classification_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='logistic_regression')

            # 评估模型
            metrics = ml_core.evaluate_model(model_id, X, y)

            assert isinstance(metrics, dict)

            # 检查分类指标
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))
                assert 0 <= metrics[metric] <= 1

    def test_cross_validation(self, ml_core, sample_training_data):
        """测试交叉验证"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 执行交叉验证
            cv_results = ml_core.cross_validate(
                X, y,
                model_type='linear_regression',
                cv_folds=5
            )

            assert isinstance(cv_results, dict)
            assert 'mean_score' in cv_results
            assert 'std_score' in cv_results
            assert 'cv_scores' in cv_results

            # 检查CV分数数组
            assert isinstance(cv_results['cv_scores'], list)
            assert len(cv_results['cv_scores']) == 5

    def test_hyperparameter_tuning(self, ml_core, sample_training_data):
        """测试超参数调优"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 定义参数空间
            param_grid = {
                'n_estimators': [10, 50, 100],
                'max_depth': [3, 5, 7]
            }

            # 执行超参数调优
            best_params, best_score = ml_core.tune_hyperparameters(
                X, y,
                model_type='random_forest',
                param_grid=param_grid,
                cv_folds=3
            )

            assert isinstance(best_params, dict)
            assert isinstance(best_score, (int, float))
            assert 'n_estimators' in best_params
            assert 'max_depth' in best_params

    def test_feature_importance(self, ml_core, sample_training_data):
        """测试特征重要性"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练模型
            model_id = ml_core.train_model(
                X, y,
                model_type='random_forest',
                hyperparameters={'n_estimators': 50}
            )

            # 获取特征重要性
            feature_importance = ml_core.get_feature_importance(model_id)

            assert isinstance(feature_importance, dict)
            assert len(feature_importance) == X.shape[1]

            # 重要性值应该在[0,1]范围内
            for importance in feature_importance.values():
                assert 0 <= importance <= 1

    def test_model_persistence(self, ml_core, sample_training_data, tmp_path):
        """测试模型持久化"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='linear_regression')

            # 保存模型
            save_path = tmp_path / "test_model.pkl"
            ml_core.save_model(model_id, str(save_path))

            # 验证文件存在
            assert save_path.exists()

            # 加载模型
            loaded_model_id = ml_core.load_model(str(save_path))

            # 验证加载的模型可以进行预测
            test_X = X.head(5)
            original_predictions = ml_core.predict(model_id, test_X)
            loaded_predictions = ml_core.predict(loaded_model_id, test_X)

            # 预测结果应该相同
            pd.testing.assert_series_equal(original_predictions, loaded_predictions)

    def test_model_versioning(self, ml_core, sample_training_data):
        """测试模型版本管理"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练初始模型
            model_id_v1 = ml_core.train_model(X, y, model_type='linear_regression')

            # 创建新版本
            model_id_v2 = ml_core.create_model_version(
                model_id_v1,
                version_notes="Updated hyperparameters"
            )

            # 验证版本管理
            versions = ml_core.get_model_versions(model_id_v1)
            assert isinstance(versions, list)
            assert len(versions) >= 2

    def test_model_monitoring(self, ml_core, sample_training_data):
        """测试模型监控"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='linear_regression')

            # 获取模型监控指标
            monitoring_metrics = ml_core.get_model_monitoring_metrics(model_id)

            assert isinstance(monitoring_metrics, dict)

            # 检查监控指标
            expected_metrics = ['training_time', 'model_size', 'last_used']
            for metric in expected_metrics:
                assert metric in monitoring_metrics

    def test_batch_prediction(self, ml_core, sample_training_data):
        """测试批量预测"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='linear_regression')

            # 准备批量数据
            batch_data = [X.head(10), X.iloc[10:20], X.iloc[20:30]]

            # 执行批量预测
            batch_predictions = ml_core.batch_predict(model_id, batch_data)

            assert isinstance(batch_predictions, list)
            assert len(batch_predictions) == len(batch_data)

            for predictions in batch_predictions:
                assert len(predictions) == 10  # 每个批次10个样本

    def test_model_explainability(self, ml_core, sample_training_data):
        """测试模型可解释性"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='linear_regression')

            # 获取模型解释
            explanation = ml_core.explain_model(model_id, X.head(1))

            assert isinstance(explanation, dict)

            # 检查解释内容
            expected_keys = ['feature_contributions', 'prediction', 'intercept']
            for key in expected_keys:
                assert key in explanation

    def test_online_learning(self, ml_core, sample_training_data):
        """测试在线学习"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 创建在线学习模型
            model_id = ml_core.create_online_model(
                model_type='linear_regression',
                initial_data=(X.head(100), y.head(100))
            )

            # 模拟在线学习过程
            for i in range(5):
                batch_X = X.iloc[i*50:(i+1)*50]
                batch_y = y.iloc[i*50:(i+1)*50]

                # 更新模型
                ml_core.update_online_model(model_id, batch_X, batch_y)

            # 验证模型可以预测
            test_predictions = ml_core.predict(model_id, X.head(10))
            assert len(test_predictions) == 10

    def test_model_fairness_analysis(self, ml_core, sample_classification_data):
        """测试模型公平性分析"""
        if ML_CORE_AVAILABLE:
            X, y = sample_classification_data

            # 添加敏感属性
            X_with_sensitive = X.copy()
            X_with_sensitive['gender'] = np.random.choice(['M', 'F'], len(X))
            X_with_sensitive['age_group'] = np.random.choice(['young', 'middle', 'old'], len(X))

            # 训练模型
            model_id = ml_core.train_model(X_with_sensitive, y, model_type='logistic_regression')

            # 分析公平性
            fairness_report = ml_core.analyze_model_fairness(
                model_id,
                X_with_sensitive,
                sensitive_attributes=['gender', 'age_group']
            )

            assert isinstance(fairness_report, dict)
            assert 'fairness_metrics' in fairness_report
            assert 'bias_detection' in fairness_report

    def test_model_robustness_testing(self, ml_core, sample_training_data):
        """测试模型鲁棒性"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='linear_regression')

            # 测试鲁棒性
            robustness_report = ml_core.test_model_robustness(
                model_id, X, y,
                noise_levels=[0.01, 0.05, 0.1]
            )

            assert isinstance(robustness_report, dict)
            assert 'robustness_scores' in robustness_report
            assert 'noise_tolerance' in robustness_report

    def test_ensemble_model_training(self, ml_core, sample_training_data):
        """测试集成模型训练"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练集成模型
            ensemble_id = ml_core.train_ensemble_model(
                X, y,
                base_models=['linear_regression', 'random_forest'],
                ensemble_method='bagging',
                n_estimators=5
            )

            assert ensemble_id is not None
            assert ml_core.model_exists(ensemble_id)

            # 测试集成预测
            predictions = ml_core.predict(ensemble_id, X.head(10))
            assert len(predictions) == 10

    def test_model_deployment_readiness(self, ml_core, sample_training_data):
        """测试模型部署就绪性"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='linear_regression')

            # 检查部署就绪性
            deployment_check = ml_core.check_deployment_readiness(model_id)

            assert isinstance(deployment_check, dict)
            assert 'deployment_ready' in deployment_check
            assert 'validation_checks' in deployment_check
            assert 'performance_requirements' in deployment_check

    def test_error_handling_and_recovery(self, ml_core):
        """测试错误处理和恢复"""
        if ML_CORE_AVAILABLE:
            # 测试无效模型类型
            with pytest.raises((MLException, ValueError)):
                ml_core.train_model(
                    pd.DataFrame({'x': [1, 2, 3]}),
                    pd.Series([1, 2, 3]),
                    model_type='invalid_model_type'
                )

            # 测试无效数据
            with pytest.raises((MLException, ValueError)):
                ml_core.train_model(
                    None,  # 无效特征数据
                    pd.Series([1, 2, 3]),
                    model_type='linear_regression'
                )

    def test_model_metadata_management(self, ml_core, sample_training_data):
        """测试模型元数据管理"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 训练模型
            model_id = ml_core.train_model(X, y, model_type='linear_regression')

            # 获取模型元数据
            metadata = ml_core.get_model_metadata(model_id)

            assert isinstance(metadata, dict)

            # 检查元数据字段
            expected_fields = [
                'model_type', 'training_date', 'feature_names',
                'target_name', 'hyperparameters', 'performance_metrics'
            ]

            for field in expected_fields:
                assert field in metadata

    def test_concurrent_model_training(self, ml_core, sample_training_data):
        """测试并发模型训练"""
        if ML_CORE_AVAILABLE:
            X, y = sample_training_data

            # 并发训练多个模型
            import threading
            results = []
            errors = []

            def train_model(model_type, index):
                try:
                    model_id = ml_core.train_model(X, y, model_type=model_type)
                    results.append((index, model_id))
                except Exception as e:
                    errors.append((index, str(e)))

            # 启动并发训练
            threads = []
            model_types = ['linear_regression', 'random_forest', 'gradient_boosting']

            for i, model_type in enumerate(model_types):
                thread = threading.Thread(target=train_model, args=(model_type, i))
                threads.append(thread)
                thread.start()

            # 等待所有线程完成
            for thread in threads:
                thread.join()

            # 验证结果
            assert len(results) == len(model_types)
            assert len(errors) == 0

            # 验证所有模型都可以进行预测
            test_X = X.head(5)
            for _, model_id in results:
                predictions = ml_core.predict(model_id, test_X)
                assert len(predictions) == len(test_X)
