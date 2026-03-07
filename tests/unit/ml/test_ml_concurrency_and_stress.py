#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML层并发和压力测试

测试MLService在高并发和压力情况下的表现
"""

import pytest
import numpy as np
import pandas as pd
import threading
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.ml.core.ml_service import MLService


@pytest.mark.skip(reason="Concurrency and stress tests have environment initialization issues")
class TestMLConcurrencyAndStress:
    """ML层并发和压力测试"""

    def setup_method(self):
        """测试前准备"""
        self.service = MLService()
        self.service.start()

        # 预先训练一些模型用于并发测试
        self._prepare_test_models()

    def teardown_method(self):
        """测试后清理"""
        # 清理所有测试模型
        models = self.service.list_models()
        for model_info in models:
            if model_info['id'].startswith('concurrency_') or model_info['id'].startswith('stress_'):
                self.service.unload_model(model_info['id'])

        self.service.stop()

    def _prepare_test_models(self):
        """准备测试模型"""
        np.random.seed(42)

        # 创建多个预训练模型
        for i in range(5):
            model_id = f"concurrency_base_{i}"

            # 生成训练数据
            X = np.random.randn(50, 3)
            y = X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(50) * 0.1

            data = pd.DataFrame({
                'f1': X[:, 0], 'f2': X[:, 1], 'f3': X[:, 2], 'target': y
            })

            config = {"algorithm": "linear_regression", "params": {}}
            success = self.service.train_model(model_id, data, config)
            assert success, f"准备模型 {model_id} 失败"

    def test_concurrent_model_training(self):
        """测试并发模型训练"""
        def train_worker(worker_id):
            """训练工作线程"""
            model_id = f"concurrency_train_{worker_id}"

            # 为每个worker生成不同的数据
            np.random.seed(100 + worker_id)
            X = np.random.randn(30, 4)
            y = (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] +
                 np.random.randn(30) * 0.1)

            data = pd.DataFrame({
                'x1': X[:, 0], 'x2': X[:, 1], 'x3': X[:, 2], 'x4': X[:, 3], 'target': y
            })

            config = {"algorithm": "linear_regression", "params": {}}
            success = self.service.train_model(model_id, data, config)

            return worker_id, success, model_id

        # 并发执行训练
        num_workers = 10
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(train_worker, i) for i in range(num_workers)]
            results = []

            for future in as_completed(futures):
                worker_id, success, model_id = future.result()
                results.append((worker_id, success, model_id))
                assert success, f"Worker {worker_id} 训练失败"

        # 验证所有模型都已创建
        models = self.service.list_models()
        trained_model_ids = [m['id'] for m in models if m['id'].startswith('concurrency_train_')]
        assert len(trained_model_ids) == num_workers

        # 清理训练的模型
        for model_id in trained_model_ids:
            self.service.unload_model(model_id)

    def test_concurrent_predictions(self):
        """测试并发预测"""
        prediction_results = []
        errors = []

        def prediction_worker(worker_id):
            """预测工作线程"""
            try:
                # 使用预训练的模型进行预测
                base_model_idx = worker_id % 5
                model_id = f"concurrency_base_{base_model_idx}"

                # 生成预测数据
                np.random.seed(200 + worker_id)
                test_data = pd.DataFrame({
                    'f1': [np.random.randn()],
                    'f2': [np.random.randn()],
                    'f3': [np.random.randn()]
                })

                # 执行预测
                prediction = self.service.predict(test_data)
                prediction_results.append((worker_id, prediction))

                return worker_id, prediction

            except Exception as e:
                errors.append((worker_id, str(e)))
                return worker_id, None

        # 并发执行预测
        num_workers = 20
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(prediction_worker, i) for i in range(num_workers)]

            for future in as_completed(futures):
                worker_id, result = future.result()
                if result is not None:
                    assert isinstance(result, (int, float, np.ndarray))

        # 验证结果
        assert len(prediction_results) == num_workers
        assert len(errors) == 0, f"预测错误: {errors}"

    def test_mixed_concurrent_operations(self):
        """测试混合并发操作"""
        operation_results = []
        errors = []

        def mixed_operation_worker(worker_id):
            """混合操作工作线程"""
            try:
                results = []

                # 操作1: 获取服务状态
                status = self.service.get_service_status()
                results.append(("status_check", worker_id, status['status'] == 'running'))

                # 操作2: 列出模型
                models = self.service.list_models()
                results.append(("list_models", worker_id, isinstance(models, list)))

                # 操作3: 预测（使用现有模型）
                if models:  # 如果有模型
                    test_data = pd.DataFrame({
                        'f1': [1.0], 'f2': [2.0], 'f3': [3.0]
                    })
                    prediction = self.service.predict(test_data)
                    results.append(("prediction", worker_id, prediction is not None))

                # 操作4: 获取服务统计
                status_after = self.service.get_service_status()
                results.append(("stats_check", worker_id, 'stats' in status_after))

                operation_results.extend(results)
                return worker_id, results

            except Exception as e:
                errors.append((worker_id, str(e)))
                return worker_id, None

        # 并发执行混合操作
        num_workers = 15
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(mixed_operation_worker, i) for i in range(num_workers)]

            for future in as_completed(futures):
                worker_id, results = future.result()
                if results:
                    for operation, w_id, success in results:
                        assert success, f"Worker {w_id} 操作 {operation} 失败"

        assert len(errors) == 0, f"混合操作错误: {errors}"

        # 验证总的操作数量
        expected_operations = num_workers * 4  # 每个worker执行4个操作
        assert len(operation_results) == expected_operations

    def test_resource_stress_test(self):
        """测试资源压力测试"""
        # 创建大量模型来测试资源管理
        stress_models = []

        try:
            # 批量创建模型
            for i in range(20):  # 创建20个模型
                model_id = f"stress_model_{i}"

                # 使用较小的数据集来快速训练
                np.random.seed(300 + i)
                X = np.random.randn(20, 2)
                y = X[:, 0] + X[:, 1] + np.random.randn(20) * 0.1

                data = pd.DataFrame({
                    'x1': X[:, 0], 'x2': X[:, 1], 'target': y
                })

                config = {"algorithm": "linear_regression", "params": {}}
                success = self.service.train_model(model_id, data, config)
                assert success, f"压力测试模型 {model_id} 创建失败"

                stress_models.append(model_id)

            # 验证所有模型都已创建
            models = self.service.list_models()
            stress_model_count = len([m for m in models if m['id'].startswith('stress_model_')])
            assert stress_model_count == 20

            # 测试批量预测性能
            batch_predictions = []

            # 创建批量预测请求
            from src.ml.core.ml_service import MLInferenceRequest, MLFeatures

            batch_requests = []
            for i in range(50):  # 50个批量请求
                model_id = stress_models[i % len(stress_models)]
                features = MLFeatures(
                    timestamp=pd.Timestamp.now(),
                    symbol="stress_test",
                    features={'x1': float(np.random.randn()), 'x2': float(np.random.randn())}
                )
                request = MLInferenceRequest(
                    request_id=f"stress_batch_{i}",
                    model_id=model_id,
                    features=features
                )
                batch_requests.append(request)

            # 执行批量预测
            responses = self.service.predict_batch(batch_requests)
            assert len(responses) == 50

            # 验证响应质量
            success_count = sum(1 for r in responses if r.success)
            assert success_count >= 45, f"批量预测成功率太低: {success_count}/50"

        finally:
            # 清理压力测试模型
            for model_id in stress_models:
                try:
                    self.service.unload_model(model_id)
                except:
                    pass  # 忽略清理错误

    def test_service_state_consistency_under_load(self):
        """测试负载下服务状态一致性"""
        initial_stats = self.service.get_service_status()

        def load_worker(worker_id):
            """负载工作线程"""
            operations_performed = 0

            try:
                # 执行多种操作
                for j in range(5):  # 每个worker执行5轮操作
                    # 操作1: 预测
                    test_data = pd.DataFrame({
                        'f1': [float(j)], 'f2': [float(worker_id)], 'f3': [1.0]
                    })
                    prediction = self.service.predict(test_data)
                    operations_performed += 1

                    # 操作2: 获取状态
                    status = self.service.get_service_status()
                    assert status['status'] == 'running'
                    operations_performed += 1

                    # 操作3: 列出模型
                    models = self.service.list_models()
                    assert isinstance(models, list)
                    operations_performed += 1

                return worker_id, operations_performed, True

            except Exception as e:
                return worker_id, operations_performed, False

        # 高并发负载测试
        num_workers = 12
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(load_worker, i) for i in range(num_workers)]
            results = []

            for future in as_completed(futures):
                worker_id, operations, success = future.result()
                results.append((worker_id, operations, success))
                assert success, f"Worker {worker_id} 负载测试失败"

        # 验证最终状态
        final_stats = self.service.get_service_status()
        assert final_stats['status'] == 'running'

        # 验证统计信息合理增长
        inference_growth = final_stats['stats']['inference_requests'] - initial_stats['stats']['inference_requests']
        expected_min_inferences = num_workers * 5  # 每个worker 5次预测
        assert inference_growth >= expected_min_inferences

    def test_memory_and_cleanup_under_stress(self):
        """测试压力下的内存管理和清理"""
        # 创建和销毁大量模型来测试内存管理
        temp_models = []

        try:
            # 第一轮：创建模型
            for i in range(15):
                model_id = f"cleanup_test_{i}"

                np.random.seed(400 + i)
                X = np.random.randn(25, 3)
                y = X[:, 0] + X[:, 1] + X[:, 2] + np.random.randn(25) * 0.1

                data = pd.DataFrame({
                    'a': X[:, 0], 'b': X[:, 1], 'c': X[:, 2], 'target': y
                })

                config = {"algorithm": "linear_regression", "params": {}}
                success = self.service.train_model(model_id, data, config)
                assert success

                temp_models.append(model_id)

            # 中间状态检查
            mid_stats = self.service.get_service_status()
            mid_model_count = mid_stats['loaded_models']
            assert mid_model_count >= 15

            # 第二轮：并发操作
            def concurrent_cleanup_worker(worker_id):
                """并发清理工作线程"""
                try:
                    # 随机选择模型进行操作
                    for _ in range(3):
                        if temp_models:
                            model_id = temp_models[worker_id % len(temp_models)]

                            # 执行预测
                            test_data = pd.DataFrame({
                                'a': [1.0], 'b': [2.0], 'c': [3.0]
                            })
                            prediction = self.service.predict(test_data)

                            # 获取模型信息
                            info = self.service.get_model_info(model_id)

                            assert prediction is not None
                            assert info is not None

                    return worker_id, True
                except Exception as e:
                    return worker_id, False

            # 并发执行操作
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(concurrent_cleanup_worker, i) for i in range(10)]
                for future in as_completed(futures):
                    worker_id, success = future.result()
                    assert success, f"清理测试 worker {worker_id} 失败"

            # 第三轮：逐步清理
            cleanup_order = temp_models.copy()
            np.random.shuffle(cleanup_order)  # 随机清理顺序

            for model_id in cleanup_order:
                success = self.service.unload_model(model_id)
                assert success, f"清理模型 {model_id} 失败"

                # 验证模型确实被清理了
                info = self.service.get_model_info(model_id)
                assert info is None, f"模型 {model_id} 应该已被清理"

            # 最终状态检查
            final_stats = self.service.get_service_status()
            final_model_count = final_stats['loaded_models']

            # 确保清理后的模型数量合理（可能还有其他测试的模型）
            assert final_model_count < mid_model_count

        except Exception as e:
            # 如果出现异常，确保清理所有临时模型
            for model_id in temp_models:
                try:
                    self.service.unload_model(model_id)
                except:
                    pass
            raise e

    def test_service_degradation_under_extreme_load(self):
        """测试极端负载下的服务降级"""
        # 测试服务在极端情况下的表现

        # 记录初始状态
        initial_status = self.service.get_service_status()

        extreme_operations = []

        def extreme_load_worker(worker_id):
            """极端负载工作线程"""
            operations = 0
            errors = 0

            try:
                # 执行大量快速操作
                for j in range(10):  # 每个线程执行10次操作
                    try:
                        # 快速预测操作
                        test_data = pd.DataFrame({
                            'f1': [float(j % 3)], 'f2': [float(worker_id % 5)], 'f3': [1.0]
                        })
                        prediction = self.service.predict(test_data)
                        operations += 1

                        # 短暂延迟模拟真实场景
                        time.sleep(0.001)

                    except Exception as e:
                        errors += 1
                        # 在极端负载下，允许一些错误但不应该太多

                return worker_id, operations, errors

            except Exception as e:
                return worker_id, operations, 999  # 表示严重错误

        # 极端并发测试
        num_extreme_workers = 20
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(extreme_load_worker, i) for i in range(num_extreme_workers)]

            total_operations = 0
            total_errors = 0

            for future in as_completed(futures):
                worker_id, operations, errors = future.result()
                total_operations += operations
                total_errors += errors

                extreme_operations.append((worker_id, operations, errors))

                # 每个worker至少应该完成一些操作
                assert operations > 0, f"Worker {worker_id} 没有完成任何操作"

        # 验证服务仍然可用
        final_status = self.service.get_service_status()
        assert final_status['status'] == 'running'

        # 验证统计信息
        inference_requests = final_status['stats']['inference_requests']
        assert inference_requests >= total_operations

        # 验证错误率在可接受范围内（允许一定比例的错误）
        error_rate = total_errors / (total_operations + total_errors) if (total_operations + total_errors) > 0 else 0
        assert error_rate < 0.1, f"错误率太高: {error_rate:.2%}"  # 允许最多10%的错误率
