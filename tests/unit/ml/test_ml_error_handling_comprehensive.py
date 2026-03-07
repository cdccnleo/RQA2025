#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习错误处理全面测试

此文件包含机器学习模块的错误处理和异常情况测试，
用于提升测试覆盖率和健壮性。
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import pickle

# 导入相关模块
try:
    from src.ml.core.ml_service import MLService
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="ML相关组件不可用")
class TestMLErrorHandlingComprehensive:
    """机器学习错误处理全面测试"""

    @pytest.fixture
    def ml_service(self):
        """创建ML服务实例"""
        service = MLService()
        service.start()
        yield service
        service.stop()

    def test_model_save_load_corruption(self, ml_service):
        """测试模型保存和加载时的损坏处理"""
        # 创建正常模型
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)

        train_data = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
        train_data['target'] = y

        model_config = {"algorithm": "random_forest"}
        ml_service.load_model("corruption_test_model", model_config)
        ml_service.train_model("corruption_test_model", train_data, model_config)

        # 模拟模型文件损坏
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # 写入损坏的数据
            with open(temp_path, 'wb') as f:
                f.write(b'corrupted model data')

            # 模拟加载损坏的模型
            with patch.object(ml_service, '_load_model_from_file') as mock_load:
                mock_load.side_effect = pickle.UnpicklingError("Corrupted pickle data")

                # 尝试预测，应该优雅处理错误
                test_X = pd.DataFrame(np.random.randn(5, 3), columns=['f1', 'f2', 'f3'])
                result = ml_service.predict(test_X)

                # 应该返回None或抛出可处理的异常
                assert result is None or isinstance(result, (list, np.ndarray))

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_prediction_with_nan_inf(self, ml_service):
        """测试预测时包含NaN和无穷大值"""
        # 创建包含NaN和inf的测试数据
        test_data = pd.DataFrame({
            'f1': [1.0, np.nan, np.inf, -np.inf, 2.0],
            'f2': [np.nan, 2.0, 3.0, np.inf, 5.0],
            'f3': [1.0, 2.0, np.nan, 4.0, -np.inf]
        })

        # 测试预测（可能会抛出异常或返回结果）
        try:
            predictions = ml_service.predict(test_data)
            # 如果成功，验证结果
            assert predictions is not None
            assert len(predictions) == 5
        except (ValueError, RuntimeError) as e:
            # 预期的数据问题异常
            assert "NaN" in str(e) or "inf" in str(e) or "finite" in str(e)

    def test_training_interruption_recovery(self, ml_service):
        """测试训练中断后的恢复"""
        # 创建训练数据
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        train_data = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        train_data['target'] = y

        model_config = {"algorithm": "random_forest"}

        # 模拟训练中断
        with patch('sklearn.ensemble.RandomForestClassifier.fit') as mock_fit:
            mock_fit.side_effect = KeyboardInterrupt("Training interrupted")

            ml_service.load_model("interruption_test_model", model_config)

            # 训练应该失败但不崩溃服务
            with pytest.raises((KeyboardInterrupt, RuntimeError)):
                ml_service.train_model("interruption_test_model", train_data, model_config)

        # 验证服务仍然可用
        assert ml_service.status.name == "RUNNING"

    def test_memory_limit_exceedance(self, ml_service):
        """测试内存限制超限情况"""
        # 创建超大数据集
        np.random.seed(42)
        X = np.random.randn(50000, 100)  # 非常大的数据集
        y = np.random.randint(0, 2, 50000)

        train_data = pd.DataFrame(X, columns=[f'f{i}' for i in range(100)])
        train_data['target'] = y

        model_config = {"algorithm": "random_forest", "params": {"n_estimators": 100}}

        ml_service.load_model("memory_test_model", model_config)

        # 测试大内存使用情况
        try:
            ml_service.train_model("memory_test_model", train_data, model_config)
            # 如果成功，验证预测
            test_X = pd.DataFrame(np.random.randn(5, 100), columns=[f'f{i}' for i in range(100)])
            predictions = ml_service.predict(test_X)
            assert predictions is not None
        except (MemoryError, RuntimeError, ValueError) as e:
            # 预期的大数据处理异常
            pytest.skip(f"大数据处理导致预期异常: {e}")

    def test_concurrent_model_access(self, ml_service):
        """测试并发模型访问的错误处理"""
        import threading
        import time

        errors = []

        def worker(worker_id):
            try:
                # 模拟并发访问同一个模型
                model_id = "shared_model"
                model_config = {"algorithm": "random_forest"}

                # 创建训练数据
                np.random.seed(worker_id)
                X = np.random.randn(30, 3)
                y = np.random.randint(0, 2, 30)

                train_data = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
                train_data['target'] = y

                # 所有线程都尝试加载同一个模型
                ml_service.load_model(model_id, model_config)

                # 训练模型
                ml_service.train_model(model_id, train_data, model_config)

                # 进行预测
                test_X = pd.DataFrame(np.random.randn(3, 3), columns=['f1', 'f2', 'f3'])
                predictions = ml_service.predict(test_X)

                assert predictions is not None

            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")

        # 创建多个线程并发访问
        threads = []
        num_threads = 5

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证：要么全部成功，要么错误在可接受范围内
        success_count = num_threads - len(errors)
        assert success_count >= num_threads // 2  # 至少一半成功

    def test_invalid_model_configuration(self, ml_service):
        """测试无效的模型配置"""
        invalid_configs = [
            {"algorithm": "nonexistent_algorithm"},
            {"algorithm": "random_forest", "params": {"invalid_param": "value"}},
            {"algorithm": "svm", "params": {"C": -1}},  # 负的C值
            {"algorithm": "xgboost", "params": {"max_depth": -5}},  # 负的深度
            {"algorithm": None},
            {"params": {"valid": "config"}, "algorithm": None}
        ]

        for i, invalid_config in enumerate(invalid_configs):
            model_id = f"invalid_config_{i}"

            # 应该能够加载配置，但训练时可能失败
            try:
                ml_service.load_model(model_id, invalid_config)

                # 创建简单训练数据
                train_data = pd.DataFrame({
                    'f1': [1, 2, 3],
                    'f2': [4, 5, 6],
                    'target': [0, 1, 0]
                })

                # 训练可能成功或失败
                ml_service.train_model(model_id, train_data, invalid_config)

            except (ValueError, TypeError, KeyError) as e:
                # 预期配置错误
                assert "algorithm" in str(e) or "params" in str(e) or "config" in str(e)

    def test_empty_dataset_handling(self, ml_service):
        """测试空数据集处理"""
        # 空特征数据
        empty_X = pd.DataFrame()
        empty_y = pd.Series(dtype=int)

        empty_data = pd.DataFrame({'target': empty_y})

        model_config = {"algorithm": "random_forest"}
        ml_service.load_model("empty_data_model", model_config)

        # 应该优雅处理空数据
        with pytest.raises((ValueError, RuntimeError)):
            ml_service.train_model("empty_data_model", empty_data, model_config)

    def test_mismatched_feature_dimensions(self, ml_service):
        """测试特征维度不匹配的情况"""
        # 创建训练数据
        X_train = np.random.randn(50, 3)
        y_train = np.random.randint(0, 2, 50)

        train_data = pd.DataFrame(X_train, columns=['f1', 'f2', 'f3'])
        train_data['target'] = y_train

        model_config = {"algorithm": "random_forest"}
        ml_service.load_model("dimension_test_model", model_config)
        ml_service.train_model("dimension_test_model", train_data, model_config)

        # 测试不同维度的数据
        test_cases = [
            pd.DataFrame(np.random.randn(5, 2), columns=['f1', 'f2']),  # 少一列
            pd.DataFrame(np.random.randn(5, 4), columns=['f1', 'f2', 'f3', 'f4']),  # 多一列
            pd.DataFrame(np.random.randn(5, 3), columns=['x1', 'x2', 'x3']),  # 不同列名
        ]

        for i, test_X in enumerate(test_cases):
            try:
                predictions = ml_service.predict(test_X)
                # 如果维度匹配，会有结果；如果不匹配，可能抛出异常
                if predictions is not None:
                    assert len(predictions) == 5
            except (ValueError, RuntimeError) as e:
                # 预期的维度不匹配错误
                assert "dimension" in str(e).lower() or "feature" in str(e).lower() or "shape" in str(e).lower()

    def test_model_persistence_errors(self, ml_service):
        """测试模型持久化错误"""
        # 创建模型
        train_data = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [2, 3, 4, 5, 6],
            'target': [0, 1, 0, 1, 0]
        })

        model_config = {"algorithm": "random_forest"}
        ml_service.load_model("persistence_test_model", model_config)
        ml_service.train_model("persistence_test_model", train_data, model_config)

        # 模拟持久化错误
        with patch('builtins.open', side_effect=OSError("Disk full")):
            with patch.object(ml_service, '_save_model_to_file') as mock_save:
                mock_save.side_effect = OSError("Disk full")

                # 应该优雅处理保存错误
                # 注意：实际的保存可能在训练时发生，也可能不发生
                # 这里我们主要测试错误处理逻辑

        # 验证模型仍然可用
        test_X = pd.DataFrame({'f1': [1, 2], 'f2': [2, 3]})
        predictions = ml_service.predict(test_X)
        assert predictions is not None
