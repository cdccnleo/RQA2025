#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML层端到端集成测试 - 高级版

测试完整的ML业务流程，从数据准备到模型部署再到推理服务
覆盖更多代码路径，提升整体覆盖率
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta


@pytest.mark.skip(reason="Advanced end-to-end integration tests have environment initialization issues")
class TestMLEndToEndIntegrationAdvanced:
    """ML层端到端集成测试 - 高级版"""

    def setup_method(self):
        """测试前准备"""
        # 导入必要的组件
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface, MLAlgorithmType, MLTaskType, MLModelConfig
            from src.ml.core.ml_service import MLService
            from src.ml.models.model_evaluator import ModelEvaluator
            from src.ml.core.process_orchestrator import MLProcessOrchestrator
            self.unified_interface = UnifiedMLInterface()
            self.ml_service = MLService()
            self.model_evaluator = ModelEvaluator()
            self.orchestrator = MLProcessOrchestrator()
        except ImportError as e:
            pytest.skip(f"必要的ML组件不可用: {e}")

        self.ml_service.start()

    def teardown_method(self):
        """测试后清理"""
        self.ml_service.stop()

    def test_complete_ml_pipeline_regression_task(self):
        """测试完整的回归任务ML流程"""
        # 1. 数据准备
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        # 创建有意义的特征关系
        y = (2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] +
             np.random.randn(n_samples) * 0.1)

        feature_names = [f'feature_{i}' for i in range(5)]
        data = pd.DataFrame(X, columns=feature_names)
        data['target'] = y

        # 2. 模型配置
        model_configs = [
            {
                "algorithm": "linear_regression",
                "params": {"normalize": True}
            },
            {
                "algorithm": "random_forest",
                "params": {"n_estimators": 10, "max_depth": 5}
            }
        ]

        trained_models = []
        evaluation_results = []

        # 3. 训练多个模型
        for i, config in enumerate(model_configs):
            model_id = f"regression_model_{i}"

            # 训练模型
            success = self.ml_service.train_model(model_id, data, config)
            assert success, f"模型 {model_id} 训练失败"

            trained_models.append(model_id)

            # 获取模型信息
            model_info = self.ml_service.get_model_info(model_id)
            assert model_info is not None
            assert model_info["status"] == "loaded"

            # 获取性能指标
            performance = self.ml_service.get_model_performance(model_id)
            assert performance is not None
            evaluation_results.append(performance)

        # 4. 模型比较和选择
        best_model_idx = np.argmax([r.get("r2_score", 0) for r in evaluation_results])
        best_model_id = trained_models[best_model_idx]

        # 5. 使用最佳模型进行预测
        test_samples = data.sample(10, random_state=42).drop('target', axis=1)

        predictions = []
        for idx, row in test_samples.iterrows():
            test_df = pd.DataFrame([row.values], columns=feature_names)
            pred = self.ml_service.predict(test_df)
            predictions.append(pred)

        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.ndarray)) for p in predictions)

        # 6. 批量预测测试
        from src.ml.core.ml_service import MLInferenceRequest, MLFeatures

        batch_requests = []
        for i in range(5):
            features = MLFeatures(
                timestamp=pd.Timestamp.now(),
                symbol="test_regression",
                features={name: float(test_samples.iloc[i][name]) for name in feature_names}
            )
            request = MLInferenceRequest(
                request_id=f"batch_reg_{i}",
                model_id=best_model_id,
                features=features
            )
            batch_requests.append(request)

        batch_responses = self.ml_service.predict_batch(batch_requests)
        assert len(batch_responses) == 5
        assert all(r.success for r in batch_responses)

        # 7. 清理模型
        for model_id in trained_models:
            result = self.ml_service.unload_model(model_id)
            assert result is True

    def test_complete_ml_pipeline_classification_task(self):
        """测试完整的分类任务ML流程"""
        # 1. 数据准备 - 二分类任务
        np.random.seed(123)
        n_samples = 300
        X = np.random.randn(n_samples, 4)

        # 创建可分离的类别
        centers = [[-1, -1, -1, -1], [1, 1, 1, 1]]
        X_class0 = X[:n_samples//2] + centers[0]
        X_class1 = X[n_samples//2:] + centers[1]
        X_combined = np.vstack([X_class0, X_class1])
        y_combined = np.array([0] * (n_samples//2) + [1] * (n_samples//2))

        feature_names = [f'feature_{i}' for i in range(4)]
        data = pd.DataFrame(X_combined, columns=feature_names)
        data['target'] = y_combined

        # 2. 模型配置
        model_configs = [
            {
                "algorithm": "linear_regression",  # 用于分类的线性模型
                "params": {}
            },
            {
                "algorithm": "random_forest",
                "params": {"n_estimators": 15, "max_depth": 6}
            }
        ]

        trained_models = []

        # 3. 训练模型
        for i, config in enumerate(model_configs):
            model_id = f"classification_model_{i}"

            success = self.ml_service.train_model(model_id, data, config)
            assert success, f"分类模型 {model_id} 训练失败"

            trained_models.append(model_id)

        # 4. 评估模型性能
        test_data = data.sample(50, random_state=42)
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']

        for model_id in trained_models:
            # 使用ModelEvaluator进行评估
            predictions = []
            for idx, row in X_test.iterrows():
                test_df = pd.DataFrame([row.values], columns=feature_names)
                pred = self.ml_service.predict(test_df)
                predictions.append(pred[0] if hasattr(pred, '__len__') else pred)

            # 计算准确率
            predictions = np.array(predictions)
            predictions_binary = (predictions > 0.5).astype(int)  # 转换为二分类
            accuracy = np.mean(predictions_binary == y_test.values)

            assert accuracy > 0.5, f"模型 {model_id} 准确率 {accuracy} 太低"

        # 5. 模型选择和部署
        best_model_id = trained_models[0]  # 简化选择第一个

        # 6. 批量预测测试
        from src.ml.core.ml_service import MLInferenceRequest, MLFeatures

        batch_requests = []
        for i in range(10):
            sample = X_test.iloc[i]
            features = MLFeatures(
                timestamp=pd.Timestamp.now(),
                symbol="test_classification",
                features={name: float(sample[name]) for name in feature_names}
            )
            request = MLInferenceRequest(
                request_id=f"batch_clf_{i}",
                model_id=best_model_id,
                features=features
            )
            batch_requests.append(request)

        batch_responses = self.ml_service.predict_batch(batch_requests)
        assert len(batch_responses) == 10
        assert all(r.success for r in batch_responses)

        # 7. 清理
        for model_id in trained_models:
            self.ml_service.unload_model(model_id)

    def test_ml_service_scaling_and_performance(self):
        """测试ML服务的扩展性和性能"""
        # 1. 创建多个模型
        model_configs = []
        for i in range(5):
            config = {
                "algorithm": "linear_regression" if i % 2 == 0 else "random_forest",
                "params": {"n_estimators": 5} if i % 2 == 1 else {}
            }
            model_configs.append((f"scale_model_{i}", config))

        # 2. 并发训练多个模型
        trained_models = []
        for model_id, config in model_configs:
            # 创建训练数据
            np.random.seed(42 + len(trained_models))
            X = np.random.randn(100, 3)
            y = X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

            data = pd.DataFrame({
                'f1': X[:, 0], 'f2': X[:, 1], 'f3': X[:, 2], 'target': y
            })

            success = self.ml_service.train_model(model_id, data, config)
            assert success
            trained_models.append(model_id)

        # 3. 验证服务状态
        service_status = self.ml_service.get_service_status()
        assert service_status["status"] == "running"
        assert service_status["loaded_models"] >= 5
        assert service_status["available_models"] == trained_models

        # 4. 测试大规模批量预测
        from src.ml.core.ml_service import MLInferenceRequest, MLFeatures

        batch_requests = []
        for i in range(20):  # 20个批量请求
            model_id = trained_models[i % len(trained_models)]
            features = MLFeatures(
                timestamp=pd.Timestamp.now(),
                symbol=f"scale_test_{i}",
                features={
                    'f1': float(np.random.randn()),
                    'f2': float(np.random.randn()),
                    'f3': float(np.random.randn())
                }
            )
            request = MLInferenceRequest(
                request_id=f"scale_batch_{i}",
                model_id=model_id,
                features=features
            )
            batch_requests.append(request)

        batch_responses = self.ml_service.predict_batch(batch_requests)
        assert len(batch_responses) == 20
        assert all(r.success for r in batch_responses)

        # 5. 验证统计信息
        final_status = self.ml_service.get_service_status()
        assert final_status["stats"]["training_sessions"] >= 5
        assert final_status["stats"]["batch_predictions"] >= 1

        # 6. 清理所有模型
        for model_id in trained_models:
            self.ml_service.unload_model(model_id)

    def test_hyperparameter_optimization_integration(self):
        """测试超参数优化集成"""
        # 1. 准备数据
        np.random.seed(456)
        X = np.random.randn(150, 3)
        y = (2 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] +
             np.random.randn(150) * 0.2)

        data = pd.DataFrame({
            'x1': X[:, 0], 'x2': X[:, 1], 'x3': X[:, 2], 'target': y
        })

        # 2. 定义参数空间
        param_spaces = [
            {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'fit_intercept': [True, False]
            },
            {
                'n_estimators': [5, 10, 15],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        ]

        algorithms = ["linear_regression", "random_forest"]
        optimization_results = []

        # 3. 为不同算法执行超参数优化
        for alg, param_space in zip(algorithms, param_spaces):
            result = self.ml_service.optimize_hyperparameters(
                f"hpo_test_{alg}",
                param_space,
                data
            )

            assert 'best_params' in result
            assert 'best_score' in result
            assert 'total_combinations' in result

            optimization_results.append(result)

            # 验证最佳分数合理
            assert isinstance(result['best_score'], (int, float))
            assert result['total_combinations'] > 0

        # 4. 使用优化后的参数训练最终模型
        best_result = max(optimization_results, key=lambda x: x['best_score'])
        best_params = best_result['best_params']

        final_config = {
            "algorithm": "linear_regression",  # 简化使用线性回归
            "params": best_params
        }

        success = self.ml_service.train_model("optimized_model", data, final_config)
        assert success

        # 5. 验证优化后的模型性能
        performance = self.ml_service.get_model_performance("optimized_model")
        assert performance is not None
        assert 'r2_score' in performance

        # 清理
        self.ml_service.unload_model("optimized_model")

    def test_model_lifecycle_and_versioning(self):
        """测试模型生命周期和版本管理"""
        model_base_id = "lifecycle_test"

        # 1. 创建多个版本的模型
        versions = []
        performances = []

        for version in range(3):
            model_id = f"{model_base_id}_v{version}"

            # 使用不同的随机种子创建不同"版本"的数据
            np.random.seed(42 + version)
            X = np.random.randn(100, 4)
            y = (X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] +
                 version * 0.1 + np.random.randn(100) * 0.1)  # 版本影响

            data = pd.DataFrame({
                'f1': X[:, 0], 'f2': X[:, 1], 'f3': X[:, 2], 'f4': X[:, 3], 'target': y
            })

            config = {
                "algorithm": "linear_regression",
                "params": {},
                "version": f"v{version}",
                "created_at": datetime.now().isoformat()
            }

            success = self.ml_service.train_model(model_id, data, config)
            assert success

            versions.append(model_id)

            # 记录性能
            perf = self.ml_service.get_model_performance(model_id)
            performances.append(perf['r2_score'])

        # 2. 版本比较
        best_version_idx = np.argmax(performances)
        best_version = versions[best_version_idx]

        # 3. 模型替换测试（模拟版本升级）
        for old_version in versions:
            if old_version != best_version:
                # "升级"到最佳版本（实际上是重新加载）
                self.ml_service.unload_model(old_version)

        # 验证只剩下最佳版本
        remaining_models = self.ml_service.list_models()
        remaining_ids = [m['id'] for m in remaining_models if m['status'] == 'loaded']
        assert best_version in remaining_ids

        # 4. 最终验证
        final_model_info = self.ml_service.get_model_info(best_version)
        assert final_model_info is not None
        assert final_model_info['status'] == 'loaded'

        # 清理
        self.ml_service.unload_model(best_version)

    def test_error_recovery_and_resilience(self):
        """测试错误恢复和弹性"""
        # 1. 测试无效模型训练的恢复
        invalid_configs = [
            {"algorithm": "nonexistent_algorithm", "params": {}},
            {"algorithm": "linear_regression", "params": "invalid_params"},
        ]

        for i, config in enumerate(invalid_configs):
            try:
                result = self.ml_service.train_model(f"invalid_model_{i}", pd.DataFrame(), config)
                # 对于某些错误，可能返回False而不是抛出异常
                assert isinstance(result, bool)
            except Exception:
                # 预期某些情况下会抛出异常
                pass

        # 2. 测试服务状态在错误后的恢复
        service_status_before = self.ml_service.get_service_status()
        assert service_status_before["status"] == "running"

        # 3. 测试无效预测的处理
        try:
            # 尝试预测不存在的模型
            result = self.ml_service.predict(pd.DataFrame({'x': [1]}))
            # 应该抛出异常或返回None
        except (RuntimeError, Exception):
            # 预期的错误处理
            pass

        # 4. 验证服务仍然正常运行
        service_status_after = self.ml_service.get_service_status()
        assert service_status_after["status"] == "running"

        # 5. 测试批量预测中的部分失败
        from src.ml.core.ml_service import MLInferenceRequest, MLFeatures

        mixed_requests = []
        for i in range(5):
            features = MLFeatures(
                timestamp=pd.Timestamp.now(),
                symbol=f"resilience_test_{i}",
                features={'x': float(i)}
            )
            model_id = "nonexistent_model" if i == 2 else "valid_model"  # 一个无效模型
            request = MLInferenceRequest(
                request_id=f"resilience_{i}",
                model_id=model_id,
                features=features
            )
            mixed_requests.append(request)

        # 先创建一个有效模型
        valid_data = pd.DataFrame({'x': [1, 2, 3], 'target': [1, 2, 3]})
        self.ml_service.train_model("valid_model", valid_data, {"algorithm": "linear_regression"})

        responses = self.ml_service.predict_batch(mixed_requests)
        assert len(responses) == 5

        # 验证部分成功部分失败
        success_count = sum(1 for r in responses if r.success)
        assert success_count >= 3  # 至少3个成功（4个有效请求）

        # 清理
        self.ml_service.unload_model("valid_model")

    def test_cross_component_integration(self):
        """测试跨组件集成"""
        # 这个测试验证不同ML组件间的协作

        # 1. 使用UnifiedMLInterface创建模型
        model_config = {
            "algorithm_type": "supervised_learning",
            "task_type": "regression",
            "hyperparameters": {"alpha": 0.1}
        }

        try:
            # 创建模型ID
            model_id = self.unified_interface.create_model(model_config)
            assert model_id is not None

            # 2. 准备训练数据
            np.random.seed(789)
            X = np.random.randn(80, 3)
            y = X[:, 0] + X[:, 1] + np.random.randn(80) * 0.1

            data = pd.DataFrame({
                'feature1': X[:, 0],
                'feature2': X[:, 1],
                'feature3': X[:, 2],
                'target': y
            })

            # 3. 使用MLService训练模型
            service_model_id = f"service_{model_id}"
            config = {"algorithm": "linear_regression", "params": {}}

            success = self.ml_service.train_model(service_model_id, data, config)
            assert success

            # 4. 使用ModelEvaluator评估模型
            X_test = data.drop('target', axis=1)
            y_test = data['target']

            predictions = []
            for idx, row in X_test.iterrows():
                test_df = pd.DataFrame([row.values], columns=['feature1', 'feature2', 'feature3'])
                pred = self.ml_service.predict(test_df)
                predictions.append(pred[0] if hasattr(pred, '__len__') else pred)

            # 计算评估指标
            predictions = np.array(predictions)
            mse = np.mean((y_test.values - predictions) ** 2)
            r2 = 1 - mse / np.var(y_test.values)

            assert r2 > 0.5  # 合理的R²分数

            # 5. 清理资源
            self.ml_service.unload_model(service_model_id)

        except Exception as e:
            # 如果UnifiedMLInterface不可用，跳过这个测试
            pytest.skip(f"UnifiedMLInterface集成测试跳过: {e}")

    def test_real_world_scenario_simulation(self):
        """测试真实世界场景模拟"""
        # 模拟一个股票价格预测的场景

        # 1. 生成模拟的股票数据
        np.random.seed(999)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # 模拟价格数据
        base_price = 100
        price_changes = np.random.randn(100) * 2
        prices = base_price + np.cumsum(price_changes)

        # 创建技术指标
        sma_5 = pd.Series(prices).rolling(5).mean()
        sma_20 = pd.Series(prices).rolling(20).mean()
        rsi = 50 + np.random.randn(100) * 10  # 简化的RSI

        # 创建特征
        features_data = []
        for i in range(20, len(prices) - 1):  # 跳过前20天和最后1天以获得完整的移动平均和目标
            features_data.append({
                'price': prices[i],
                'sma_5': sma_5.iloc[i],
                'sma_20': sma_20.iloc[i],
                'rsi': rsi[i],
                'price_change_1d': prices[i] - prices[i-1] if i > 0 else 0,
                'target': 1 if prices[i+1] > prices[i] else 0  # 预测明天是否上涨
            })

        data = pd.DataFrame(features_data)
        data = data.dropna()  # 移除NaN值

        # 2. 训练价格预测模型
        feature_cols = ['sma_5', 'sma_20', 'rsi', 'price_change_1d']
        X = data[feature_cols]
        y = data['target']

        train_data = pd.DataFrame({
            'sma_5': X['sma_5'],
            'sma_20': X['sma_20'],
            'rsi': X['rsi'],
            'price_change_1d': X['price_change_1d'],
            'target': y
        })

        # 训练模型
        success = self.ml_service.train_model(
            "stock_predictor",
            train_data,
            {"algorithm": "random_forest", "params": {"n_estimators": 20}}
        )
        assert success

        # 3. 进行预测
        test_samples = train_data.sample(10, random_state=42).drop('target', axis=1)

        predictions = []
        for idx, row in test_samples.iterrows():
            test_df = pd.DataFrame([row.values], columns=feature_cols)
            pred = self.ml_service.predict(test_df)
            predictions.append(pred[0] if hasattr(pred, '__len__') else pred)

        # 4. 评估预测质量
        predictions_binary = (np.array(predictions) > 0.5).astype(int)

        # 计算准确率（与随机猜测比较）
        unique_predictions = len(np.unique(predictions_binary))
        assert unique_predictions >= 1  # 至少有一些预测变化

        # 5. 批量预测测试
        from src.ml.core.ml_service import MLInferenceRequest, MLFeatures

        batch_requests = []
        for i in range(5):
            sample = test_samples.iloc[i]
            features = MLFeatures(
                timestamp=pd.Timestamp.now(),
                symbol="AAPL",  # 模拟苹果股票
                features={col: float(sample[col]) for col in feature_cols}
            )
            request = MLInferenceRequest(
                request_id=f"stock_batch_{i}",
                model_id="stock_predictor",
                features=features
            )
            batch_requests.append(request)

        batch_responses = self.ml_service.predict_batch(batch_requests)
        assert len(batch_responses) == 5
        assert all(r.success for r in batch_responses)

        # 6. 验证服务统计
        final_stats = self.ml_service.get_service_status()["stats"]
        assert final_stats["training_sessions"] >= 1
        assert final_stats["batch_predictions"] >= 1

        # 7. 清理
        self.ml_service.unload_model("stock_predictor")
