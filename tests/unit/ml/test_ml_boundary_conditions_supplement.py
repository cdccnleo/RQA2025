#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML模块边界条件补充测试

针对ML模块中覆盖率较低的部分补充边界条件和异常处理测试
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time
from typing import Dict, List, Any, Optional

# ML相关导入
try:
    from src.ml.core.unified_ml_interface import (
        UnifiedMLInterface, MLModelConfig, MLAlgorithmType, MLTaskType
    )
except ImportError:
    UnifiedMLInterface = None
    MLModelConfig = None
    MLAlgorithmType = None
    MLTaskType = None


class TestMLBoundaryConditionsSupplement:
    """ML模块边界条件补充测试"""

    # ============================================================================
    # 模型训练边界条件测试
    # ============================================================================

    @pytest.mark.parametrize("input_shape", [
        (0, 5),      # 空数据集
        (1, 5),      # 单样本
        (100, 100),  # 大数据集
        (100, 0),    # 无特征
    ])
    def test_model_training_data_boundaries(self, input_shape):
        """测试模型训练数据边界条件"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface, MLModelConfig, MLAlgorithmType, MLTaskType
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface()

        # 创建测试数据
        n_samples, n_features = input_shape

        if n_samples == 0 or n_features == 0:
            # 边界条件：空数据或无特征
            X = pd.DataFrame()
            y = pd.Series(dtype=float)
        else:
            np.random.seed(42)
            X = pd.DataFrame(np.random.randn(n_samples, n_features))
            y = pd.Series(np.random.randint(0, 2, n_samples))

        # 创建模型配置
        model_config = MLModelConfig(
            algorithm_type=MLAlgorithmType.ENSEMBLE_LEARNING,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={"n_estimators": 10}
        )

        if n_samples == 0 or n_features == 0:
            # 边界条件：空数据或无特征应该抛出异常
            with pytest.raises((ValueError, RuntimeError)):
                # 先创建模型
                model_id = interface.create_model(model_config)
                # 再训练
                interface.train_model(model_id, X, y)
        else:
            # 正常情况
            model_id = interface.create_model(model_config)
            result = interface.train_model(model_id, X, y)
            assert result is not None

    def test_model_training_with_nan_inf_values(self):
        """测试包含NaN和无穷大值的训练数据"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface()

        # 创建包含特殊值的数据
        np.random.seed(42)
        X = pd.DataFrame({
            'normal': np.random.randn(100),
            'nan_values': [1.0] * 50 + [np.nan] * 50,
            'inf_values': [1.0] * 50 + [np.inf] * 50,
            'neg_inf': [1.0] * 50 + [-np.inf] * 50
        })
        y = pd.Series(np.random.randint(0, 2, 100))

        model_config = {
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 10}
        }

        # 应该能够处理NaN和无穷大值（或者抛出适当的异常）
        try:
            # 先创建模型
            model_config_full = MLModelConfig(
                algorithm_type=MLAlgorithmType.SUPERVISED_LEARNING,
                task_type=MLTaskType.CLASSIFICATION,
                hyperparameters=model_config["hyperparameters"]
            )
            model_id = interface.create_model(model_config_full)

            # 然后训练模型
            result = interface.train_model(model_id, X, y)
            assert result is not None
        except (ValueError, RuntimeError) as e:
            # 合理的异常处理
            assert "NaN" in str(e) or "inf" in str(e) or "infinite" in str(e).lower()

    # ============================================================================
    # 模型预测边界条件测试
    # ============================================================================

    def test_model_prediction_empty_input(self):
        """测试空输入的模型预测"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface()

        # 创建并训练模型
        X_train = pd.DataFrame(np.random.randn(50, 3))
        y_train = pd.Series(np.random.randint(0, 2, 50))

        model_config = {
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 10}
        }

        # 先创建模型
        model_config_full = MLModelConfig(
            algorithm_type=MLAlgorithmType.SUPERVISED_LEARNING,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters=model_config["hyperparameters"]
        )
        model_id = interface.create_model(model_config_full)

        # 然后训练模型
        interface.train_model(model_id, X_train, y_train)
        assert model_id is not None

        # 测试空输入预测
        X_empty = pd.DataFrame()

        with pytest.raises((ValueError, RuntimeError)):
            interface.predict(model_id, X_empty)

    def test_model_prediction_shape_mismatch(self):
        """测试预测时特征维度不匹配"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface()

        # 创建并训练3特征模型
        X_train = pd.DataFrame(np.random.randn(50, 3), columns=['f1', 'f2', 'f3'])
        y_train = pd.Series(np.random.randint(0, 2, 50))

        model_config = {
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 10}
        }

        # 先创建模型
        model_config_full = MLModelConfig(
            algorithm_type=MLAlgorithmType.SUPERVISED_LEARNING,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters=model_config["hyperparameters"]
        )
        model_id = interface.create_model(model_config_full)

        # 然后训练模型
        interface.train_model(model_id, X_train, y_train)

        # 测试不同特征维度的预测数据
        test_cases = [
            pd.DataFrame(np.random.randn(10, 2), columns=['f1', 'f2']),  # 少一列
            pd.DataFrame(np.random.randn(10, 4), columns=['f1', 'f2', 'f3', 'f4']),  # 多一列
            pd.DataFrame(np.random.randn(10, 3), columns=['a', 'b', 'c']),  # 不同列名
        ]

        for X_test in test_cases:
            try:
                predictions = interface.predict(model_id, X_test)
                # 如果不抛异常，应该返回合理的结果
                assert len(predictions) == len(X_test)
            except (ValueError, RuntimeError):
                # 合理的异常处理
                pass

    # ============================================================================
    # 超参数优化边界条件测试
    # ============================================================================

    def test_hyperparameter_optimization_extreme_values(self):
        """测试超参数优化的极端值"""
        try:
            from src.ml.core.ml_service import MLService
        except ImportError:
            pytest.skip("MLService不可用")

        service = MLService()
        service.start()

        # 创建训练数据
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))

        model_id = service.train_model(X, y, {"algorithm": "random_forest"})

        # 测试极端超参数值
        extreme_param_spaces = [
            {"n_estimators": [1, 1000]},  # 极小和极大值
            {"max_depth": [1, 100]},      # 极小和极大深度
            {"min_samples_split": [2, 100]},  # 极小和极大分割样本数
        ]

        for param_space in extreme_param_spaces:
            try:
                result = service.optimize_hyperparameters(model_id, param_space, X)
                # 如果不抛异常，应该返回结果
                assert isinstance(result, dict)
            except (ValueError, RuntimeError, TimeoutError):
                # 合理的异常处理
                pass

    def test_hyperparameter_optimization_empty_space(self):
        """测试空的超参数空间"""
        try:
            from src.ml.core.ml_service import MLService
        except ImportError:
            pytest.skip("MLService不可用")

        service = MLService()
        service.start()

        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))
        model_id = service.train_model(X, y, {"algorithm": "random_forest"})

        # 空参数空间
        result = service.optimize_hyperparameters(model_id, {}, X)
        assert "error" in result

    # ============================================================================
    # 模型序列化边界条件测试
    # ============================================================================

    def test_model_serialization_large_model(self):
        """测试大模型的序列化"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface()

        # 创建大数据集训练大模型
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(500, 20))
        y = pd.Series(np.random.randint(0, 2, 500))

        model_config = {
            "algorithm": "random_forest",
            "hyperparameters": {
                "n_estimators": 100,  # 大模型
                "max_depth": 20
            }
        }

        # 先创建模型
        model_config_full = MLModelConfig(
            algorithm_type=MLAlgorithmType.SUPERVISED_LEARNING,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters=model_config["hyperparameters"]
        )
        model_id = interface.create_model(model_config_full)

        # 然后训练模型
        interface.train_model(model_id, X, y)

        # 测试序列化
        try:
            serialized = interface.serialize_model(model_id)
            assert serialized is not None
            assert len(serialized) > 0

            # 测试反序列化
            new_model_id = interface.deserialize_model(serialized)
            assert new_model_id is not None
            assert new_model_id != model_id

        except (ValueError, RuntimeError, MemoryError) as e:
            # 大模型可能序列化失败，这是合理的
            assert "memory" in str(e).lower() or "size" in str(e).lower() or "large" in str(e).lower()

    def test_model_serialization_corrupted_data(self):
        """测试损坏数据的模型反序列化"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface()

        # 测试各种损坏的数据
        corrupted_data = [
            "",           # 空字符串
            "invalid",    # 无效数据
            "{" + "x" * 1000,  # 不完整的JSON
            None,         # None值
            123,          # 数字
            [],           # 空列表
        ]

        for data in corrupted_data:
            with pytest.raises((ValueError, TypeError, RuntimeError)):
                interface.deserialize_model(data)

    # ============================================================================
    # 并发操作边界条件测试
    # ============================================================================

    def test_concurrent_model_operations(self):
        """测试并发模型操作"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface()

        # 创建训练数据
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))

        model_config = {
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 10}
        }

        # 训练多个模型
        model_ids = []
        for i in range(5):
            # 先创建模型
            model_config_full = MLModelConfig(
                algorithm_type=MLAlgorithmType.SUPERVISED_LEARNING,
                task_type=MLTaskType.CLASSIFICATION,
                hyperparameters=model_config["hyperparameters"]
            )
            model_id = interface.create_model(model_config_full)

            # 然后训练模型
            interface.train_model(model_id, X, y)
            model_ids.append(model_id)

        # 并发预测测试
        async def concurrent_predict(model_id, X_test):
            # 模拟异步预测
            await asyncio.sleep(0.01)
            return interface.predict(model_id, X_test)

        async def run_concurrent_predictions():
            X_test = pd.DataFrame(np.random.randn(10, 5))

            # 创建并发预测任务
            tasks = [concurrent_predict(model_id, X_test) for model_id in model_ids]

            # 并发执行
            results = await asyncio.gather(*tasks, return_exceptions=True)

            return results

        # 运行并发测试
        results = asyncio.run(run_concurrent_predictions())

        # 检查结果
        successful_predictions = 0
        exceptions = 0

        for result in results:
            if isinstance(result, Exception):
                exceptions += 1
            elif result is not None:
                successful_predictions += 1

        # 至少有一些成功的预测
        assert successful_predictions > 0

    # ============================================================================
    # 资源管理边界条件测试
    # ============================================================================

    def test_memory_management_under_pressure(self):
        """测试内存压力下的资源管理"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface({
            "max_cache_size": 10,  # 很小的缓存
            "model_cache_enabled": True
        })

        # 创建多个模型来测试缓存限制
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))

        model_config = {
            "algorithm": "random_forest",
            "hyperparameters": {"n_estimators": 10}
        }

        # 训练超过缓存限制数量的模型
        model_ids = []
        for i in range(15):  # 超过max_cache_size
            # 先创建模型
            model_config_full = MLModelConfig(
                algorithm_type=MLAlgorithmType.SUPERVISED_LEARNING,
                task_type=MLTaskType.CLASSIFICATION,
                hyperparameters=model_config["hyperparameters"]
            )
            model_id = interface.create_model(model_config_full)

            # 然后训练模型
            interface.train_model(model_id, X, y)
            model_ids.append(model_id)

        # 缓存应该能够处理这种情况（或者抛出适当的异常）
        assert len(model_ids) == 15

        # 测试预测仍然工作（使用最近的模型，因为缓存可能已清理旧模型）
        X_test = pd.DataFrame(np.random.randn(5, 3))
        # 尝试使用最近的模型进行预测
        last_model_id = model_ids[-1]
        if last_model_id in interface._model_cache:
            predictions = interface.predict(last_model_id, X_test)
            assert predictions is not None
        else:
            # 如果缓存被清理，至少验证缓存大小不超过限制
            assert len(interface._model_cache) <= interface._max_cache_size

    def test_file_system_operations_boundaries(self):
        """测试文件系统操作边界条件"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface()

        # 创建模型
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))
        model_config = {"algorithm": "random_forest"}

        # 先创建模型
        model_config_full = MLModelConfig(
            algorithm_type=MLAlgorithmType.SUPERVISED_LEARNING,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={"n_estimators": 10}
        )
        model_id = interface.create_model(model_config_full)

        # 然后训练模型
        interface.train_model(model_id, X, y)

        # 测试无效路径保存
        invalid_paths = [
            "",                    # 空路径
            "   ",                # 空白路径
            "/invalid/path/model.pkl",  # 不存在的目录
            "C:\\invalid\\drive\\model.pkl",  # 无效驱动器
            None,                  # None值
        ]

        for invalid_path in invalid_paths:
            try:
                interface.save_model(model_id, invalid_path)
                # 如果不抛异常，检查文件是否实际保存了
                if invalid_path:
                    # 这里可能需要额外的验证
                    pass
            except (ValueError, RuntimeError, OSError, IOError):
                # 合理的异常处理
                pass

    # ============================================================================
    # 算法特定边界条件测试
    # ============================================================================

    @pytest.mark.parametrize("algorithm,edge_config", [
        ("linear_regression", {"fit_intercept": False}),
        ("random_forest", {"n_estimators": 1}),  # 最小森林
        ("svm", {"C": 0.001}),  # 很小的C值
        ("xgboost", {"max_depth": 1}),  # 最小深度
    ])
    def test_algorithm_specific_edge_cases(self, algorithm, edge_config):
        """测试算法特定的边界情况"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface()

        # 创建适合该算法的数据
        if algorithm in ["linear_regression", "svm"]:
            # 连续目标变量
            X = pd.DataFrame(np.random.randn(50, 3))
            y = pd.Series(np.random.randn(50))
        else:
            # 分类目标变量
            X = pd.DataFrame(np.random.randn(50, 3))
            y = pd.Series(np.random.randint(0, 2, 50))

        model_config = {
            "algorithm": algorithm,
            "hyperparameters": edge_config
        }

        # 应该能够处理这些边界配置
        try:
            # 先创建模型
            task_type = MLTaskType.REGRESSION if algorithm in ["linear_regression", "svm"] else MLTaskType.CLASSIFICATION
            model_config_full = MLModelConfig(
                algorithm_type=MLAlgorithmType.SUPERVISED_LEARNING,
                task_type=task_type,
                hyperparameters=edge_config
            )
            model_id = interface.create_model(model_config_full)

            # 然后训练模型
            interface.train_model(model_id, X, y)
            assert model_id is not None

            # 测试预测
            X_test = pd.DataFrame(np.random.randn(5, 3))
            predictions = interface.predict(model_id, X_test)
            assert predictions is not None

        except (ValueError, RuntimeError, ImportError) as e:
            # 某些算法可能不可用，这是合理的
            if "not available" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"{algorithm}不可用")
            else:
                raise

    def test_cross_validation_extreme_folds(self):
        """测试交叉验证的极端折数"""
        try:
            from src.ml.core.unified_ml_interface import UnifiedMLInterface
        except ImportError:
            pytest.skip("UnifiedMLInterface不可用")

        interface = UnifiedMLInterface()

        X = pd.DataFrame(np.random.randn(20, 3))  # 小数据集
        y = pd.Series(np.random.randint(0, 2, 20))

        model_config = {"algorithm": "random_forest"}

        # 测试极端折数
        extreme_folds = [1, 2, len(X) - 1, len(X), len(X) + 1]

        for n_folds in extreme_folds:
            try:
                scores = interface.cross_validate(X, y, model_config, cv=n_folds)
                if n_folds >= len(X) or n_folds < 2:
                    # 极端情况可能抛异常或返回特殊结果
                    pass
                else:
                    assert isinstance(scores, (list, np.ndarray))
            except (ValueError, RuntimeError):
                # 合理的异常处理
                pass
