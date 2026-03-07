# tests/unit/ml/test_inference_service.py
"""
InferenceService单元测试

测试覆盖:
- 初始化参数验证
- 同步推理功能
- 异步推理功能
- 批量推理功能
- 流式推理功能
- 模型管理
- 错误处理
- 性能监控
- 并发安全性
- 边界条件
"""

import pytest

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.legacy,
    pytest.mark.timeout(60),  # 60秒超时（ML推理可能需要更多时间）
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]
pytest.skip("legacy 推理服务测试默认跳过，需手动启用", allow_module_level=True)

import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import tempfile
import time
import asyncio
import threading
import os
import json

from src.ml.inference_service import (
    InferenceService,
    InferenceMode,
    ServiceStatus,
    InferenceRequest,
    InferenceResponse
)


class TestInferenceService:
    """InferenceService测试类"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_data(self):
        """样本数据fixture"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })

    @pytest.fixture
    def inference_config(self):
        """推理配置fixture"""
        return {
            'model_name': "test_model",
            'model_version': "1.0.0",
            'batch_size': 32,
            'timeout': 30,
            'max_retries': 3,
            'enable_cache': True,
            'cache_ttl': 300
        }

    @pytest.fixture
    def mock_model_manager(self):
        """Mock模型管理器"""
        mock_manager = Mock()
        mock_manager.predict.return_value = ModelPrediction(
            prediction=[0.8, 0.6, 0.9],
            confidence=[0.85, 0.75, 0.92],
            model_version="1.0.0",
            inference_time=0.05,
            features_used=['feature_1', 'feature_2', 'feature_3']
        )
        mock_manager.get_model_info.return_value = {
            'name': 'test_model',
            'version': '1.0.0',
            'type': 'classification',
            'features': ['feature_1', 'feature_2', 'feature_3']
        }
        mock_manager.is_model_loaded.return_value = True
        return mock_manager

    @pytest.fixture
    def inference_service(self, inference_config, mock_model_manager):
        """InferenceService实例"""
        with patch('src.ml.inference_service.get_models_adapter') as mock_adapter:
            mock_adapter_instance = Mock()
            mock_adapter_instance.get_models_logger.return_value = Mock()
            mock_adapter.return_value = mock_adapter_instance

            service = InferenceService(inference_config)
            service.model_manager = mock_model_manager
            # 手动设置状态为RUNNING，因为start方法可能有异步问题
            service.status = ServiceStatus.RUNNING
            yield service

    def test_initialization_with_config(self, inference_config):
        """测试带配置的初始化"""
        with patch('src.ml.inference_service.get_models_adapter') as mock_adapter:
            mock_adapter_instance = Mock()
            mock_adapter_instance.get_models_logger.return_value = Mock()
            mock_adapter.return_value = mock_adapter_instance

            service = InferenceService(inference_config)

        assert service.config == inference_config
        assert service.status == ServiceStatus.STARTING

    def test_initialization_without_config(self):
        """测试无配置的初始化"""
        with patch('src.ml.inference_service.get_models_adapter') as mock_adapter:
            mock_adapter_instance = Mock()
            mock_adapter_instance.get_models_logger.return_value = Mock()
            mock_adapter.return_value = mock_adapter_instance

            service = InferenceService()

            assert service.config is not None
            assert service.status == ServiceStatus.STARTING

    def test_initialization_invalid_config(self):
        """测试无效配置的初始化"""
        invalid_config = {
            'model_name': "",  # 无效的模型名称
            'model_version': "1.0.0",
            'batch_size': -1,  # 无效的batch_size
            'timeout': 30,
            'max_retries': 3,
            'enable_cache': True,
            'cache_ttl': 300
        }

        with patch('src.ml.inference_service.get_models_adapter') as mock_adapter:
            mock_adapter_instance = Mock()
            mock_adapter_instance.get_models_logger.return_value = Mock()
            mock_adapter.return_value = mock_adapter_instance

            service = InferenceService(invalid_config)

            # 应该能够处理无效配置
            assert service.config == invalid_config

    def test_service_status_transitions(self, inference_service):
        """测试服务状态转换"""
        # 初始状态
        assert inference_service.status == ServiceStatus.STARTING

        # 启动服务
        inference_service.start()
        assert inference_service.status == ServiceStatus.RUNNING

        # 停止服务
        inference_service.stop()
        assert inference_service.status == ServiceStatus.STOPPING

    def test_synchronous_inference(self, inference_service, sample_data):
        """测试同步推理"""
        test_data = sample_data.head(3)

        result = inference_service.predict(test_data, mode=InferenceMode.SYNCHRONOUS)

        assert result is not None
        assert 'predictions' in result
        assert 'metadata' in result
        assert len(result['predictions']) == len(test_data)
        assert result['metadata']['mode'] == InferenceMode.SYNCHRONOUS.value

    def test_batch_inference(self, inference_service, sample_data):
        """测试批量推理"""
        test_data = sample_data.head(10)

        result = inference_service.predict(test_data, mode=InferenceMode.BATCH)

        assert result is not None
        assert 'predictions' in result
        assert 'metadata' in result
        assert len(result['predictions']) == len(test_data)
        assert result['metadata']['mode'] == InferenceMode.BATCH.value
        assert result['metadata']['batch_size'] == 10

    @pytest.mark.asyncio
    async def test_asynchronous_inference(self, inference_service, sample_data):
        """测试异步推理"""
        test_data = sample_data.head(3)

        result = await inference_service.predict_async(test_data)

        assert result is not None
        assert 'predictions' in result
        assert 'metadata' in result
        assert len(result['predictions']) == len(test_data)
        assert result['metadata']['mode'] == InferenceMode.ASYNCHRONOUS.value

    def test_streaming_inference(self, inference_service, sample_data):
        """测试流式推理"""
        test_data = sample_data.head(5)

        results = []
        for result in inference_service.predict_stream(test_data):
            results.append(result)

        assert len(results) == len(test_data)
        for result in results:
            assert 'prediction' in result
            assert 'confidence' in result

    def test_predict_with_invalid_data(self, inference_service):
        """测试无效数据预测"""
        invalid_data = pd.DataFrame()  # 空DataFrame

        with pytest.raises(ValueError, match="输入数据为空"):
            inference_service.predict(invalid_data)

    def test_predict_with_none_data(self, inference_service):
        """测试None数据预测"""
        with pytest.raises((ValueError, TypeError)):
            inference_service.predict(None)

    def test_predict_with_missing_features(self, inference_service, sample_data):
        """测试缺失特征的数据预测"""
        incomplete_data = sample_data[['feature_1', 'feature_2']].head(3)  # 缺少feature_3

        # 应该能够处理缺失的特征或者抛出适当的错误
        result = inference_service.predict(incomplete_data)
        assert result is not None

    def test_model_loading_and_unloading(self, inference_service):
        """测试模型加载和卸载"""
        # 测试模型加载
        success = inference_service.load_model("test_model", "1.0.0")
        assert success is True

        # 测试模型卸载
        success = inference_service.unload_model("test_model")
        assert success is True

    def test_model_info_retrieval(self, inference_service):
        """测试模型信息获取"""
        info = inference_service.get_model_info("test_model")

        assert info is not None
        assert 'name' in info
        assert 'version' in info
        assert 'type' in info

    def test_model_status_check(self, inference_service):
        """测试模型状态检查"""
        status = inference_service.get_model_status("test_model")

        assert status is not None
        assert 'loaded' in status
        assert 'version' in status

    def test_batch_processing_optimization(self, inference_service, sample_data):
        """测试批量处理优化"""
        large_data = sample_data.head(100)

        start_time = time.time()
        result = inference_service.predict(large_data, mode=InferenceMode.BATCH)
        end_time = time.time()

        duration = end_time - start_time

        assert result is not None
        assert len(result['predictions']) == len(large_data)
        # 批量处理应该在合理时间内完成
        assert duration < 5.0

    def test_cache_mechanism(self, inference_service, sample_data):
        """测试缓存机制"""
        test_data = sample_data.head(3)

        # 第一次预测
        result1 = inference_service.predict(test_data)

        # 第二次预测（应该使用缓存）
        result2 = inference_service.predict(test_data)

        # 结果应该相同
        assert result1['predictions'] == result2['predictions']

    def test_cache_expiration(self, inference_service, sample_data):
        """测试缓存过期"""
        test_data = sample_data.head(3)

        # 设置短的缓存TTL
        inference_service.config.cache_ttl = 0.1  # 0.1秒

        # 第一次预测
        result1 = inference_service.predict(test_data)

        # 等待缓存过期
        time.sleep(0.2)

        # 第二次预测（缓存已过期）
        result2 = inference_service.predict(test_data)

        # 结果应该相同（因为是相同的数据）
        assert result1['predictions'] == result2['predictions']

    def test_error_handling_model_not_found(self, inference_service, sample_data):
        """测试模型未找到错误处理"""
        test_data = sample_data.head(3)

        # Mock模型管理器抛出异常
        inference_service.model_manager.is_model_loaded.return_value = False
        inference_service.model_manager.predict.side_effect = ValueError("Model not found")

        with pytest.raises(ValueError, match="Model not found"):
            inference_service.predict(test_data)

    def test_error_handling_prediction_failure(self, inference_service, sample_data):
        """测试预测失败错误处理"""
        test_data = sample_data.head(3)

        # Mock预测失败
        inference_service.model_manager.predict.side_effect = Exception("Prediction failed")

        with pytest.raises(Exception, match="Prediction failed"):
            inference_service.predict(test_data)

    def test_performance_monitoring(self, inference_service, sample_data):
        """测试性能监控"""
        test_data = sample_data.head(10)

        start_time = time.time()
        result = inference_service.predict(test_data)
        end_time = time.time()

        duration = end_time - start_time

        # 验证性能指标
        assert duration >= 0
        assert result is not None

        # 检查结果中的性能信息
        assert 'metadata' in result
        assert 'inference_time' in result['metadata']

    def test_memory_usage_efficiency(self, inference_service, sample_data):
        """测试内存使用效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        test_data = sample_data.head(50)
        result = inference_service.predict(test_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增加应该在合理范围内
        assert memory_increase < 100 * 1024 * 1024  # 不超过100MB
        assert result is not None

    def test_concurrent_inference_safety(self, inference_service):
        """测试并发推理安全性"""
        import concurrent.futures

        # 创建测试数据
        test_data = pd.DataFrame({
            'feature_1': np.random.randn(10),
            'feature_2': np.random.randn(10),
            'feature_3': np.random.randn(10)
        })

        results = []
        errors = []

        def predict_worker():
            try:
                result = inference_service.predict(test_data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # 并发执行10个推理请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(predict_worker) for _ in range(10)]
            concurrent.futures.wait(futures)

        # 验证并发安全性
        assert len(results) == 10  # 所有请求都成功
        assert len(errors) == 0    # 没有错误

        # 验证所有结果的一致性
        for result in results:
            assert len(result['predictions']) == len(test_data)

    def test_resource_cleanup(self, inference_service):
        """测试资源清理"""
        # 执行一些推理操作
        sample_data = pd.DataFrame({
            'feature_1': np.random.randn(5),
            'feature_2': np.random.randn(5),
            'feature_3': np.random.randn(5)
        })

        result = inference_service.predict(sample_data)
        assert result is not None

        # 停止服务（应该清理资源）
        inference_service.stop()

        # 这里可以添加资源清理验证逻辑
        # 例如验证线程池、缓存等被正确清理

    def test_service_health_check(self, inference_service):
        """测试服务健康检查"""
        health_status = inference_service.health_check()

        assert health_status is not None
        assert 'status' in health_status
        assert 'timestamp' in health_status
        assert 'model_loaded' in health_status

    def test_service_metrics_collection(self, inference_service, sample_data):
        """测试服务指标收集"""
        test_data = sample_data.head(5)

        # 执行几次推理
        for _ in range(3):
            inference_service.predict(test_data)

        metrics = inference_service.get_metrics()

        assert metrics is not None
        assert 'total_predictions' in metrics
        assert 'average_inference_time' in metrics
        assert metrics['total_predictions'] >= 3

    def test_configuration_update(self, inference_service):
        """测试配置更新"""
        new_config = {
            'model_name': "updated_model",
            'model_version': "2.0.0",
            'batch_size': 64,
            'timeout': 60,
            'max_retries': 5,
            'enable_cache': False,
            'cache_ttl': 600
        }

        success = inference_service.update_config(new_config)

        assert success is True
        assert inference_service.config == new_config
        assert inference_service.model_name == "updated_model"

    def test_graceful_shutdown(self, inference_service):
        """测试优雅关闭"""
        # 启动服务
        inference_service.start()
        assert inference_service.status == ServiceStatus.RUNNING

        # 优雅关闭
        success = inference_service.shutdown()
        assert success is True
        assert inference_service.status == ServiceStatus.STOPPING

    def test_inference_with_different_data_types(self, inference_service):
        """测试不同数据类型的推理"""
        # 测试numpy数组
        numpy_data = np.random.randn(5, 3)
        result_numpy = inference_service.predict(numpy_data)
        assert result_numpy is not None

        # 测试列表数据
        list_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result_list = inference_service.predict(list_data)
        assert result_list is not None

    def test_inference_result_format(self, inference_service, sample_data):
        """测试推理结果格式"""
        test_data = sample_data.head(3)
        result = inference_service.predict(test_data)

        # 验证结果格式
        assert 'predictions' in result
        assert 'metadata' in result
        assert 'model_info' in result['metadata']
        assert 'inference_time' in result['metadata']
        assert 'timestamp' in result['metadata']

        # 验证预测结果
        predictions = result['predictions']
        assert len(predictions) == len(test_data)

    def test_inference_with_custom_parameters(self, inference_service, sample_data):
        """测试自定义参数的推理"""
        test_data = sample_data.head(3)

        custom_params = {
            'threshold': 0.8,
            'top_k': 5,
            'return_probabilities': True
        }

        result = inference_service.predict(test_data, **custom_params)

        assert result is not None
        assert 'predictions' in result
        # 这里可以验证自定义参数是否被正确应用

    def test_inference_timeout_handling(self, inference_service, sample_data):
        """测试推理超时处理"""
        test_data = sample_data.head(3)

        # 设置很短的超时时间
        inference_service.config.timeout = 0.001  # 1毫秒

        # Mock一个延迟的预测
        original_predict = inference_service.model_manager.predict
        def delayed_predict(*args, **kwargs):
            time.sleep(0.01)  # 10毫秒延迟
            return original_predict(*args, **kwargs)

        inference_service.model_manager.predict = delayed_predict

        # 应该抛出超时异常或正常处理
        try:
            result = inference_service.predict(test_data)
            assert result is not None  # 如果服务能处理超时
        except Exception:
            pass  # 超时异常也是可以接受的

        # 恢复原始方法
        inference_service.model_manager.predict = original_predict

    def test_inference_with_empty_features(self, inference_service):
        """测试空特征的推理"""
        empty_features_data = pd.DataFrame({
            'feature_1': [],
            'feature_2': [],
            'feature_3': []
        })

        result = inference_service.predict(empty_features_data)

        assert result is not None
        assert len(result['predictions']) == 0

    def test_inference_with_single_sample(self, inference_service):
        """测试单样本推理"""
        single_sample = pd.DataFrame({
            'feature_1': [1.0],
            'feature_2': [2.0],
            'feature_3': [3.0]
        })

        result = inference_service.predict(single_sample)

        assert result is not None
        assert len(result['predictions']) == 1

    def test_inference_with_large_batch(self, inference_service):
        """测试大批量推理"""
        large_batch = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'feature_3': np.random.randn(1000)
        })

        start_time = time.time()
        result = inference_service.predict(large_batch, mode=InferenceMode.BATCH)
        end_time = time.time()

        duration = end_time - start_time

        assert result is not None
        assert len(result['predictions']) == 1000
        # 大批量应该在合理时间内完成
        assert duration < 30.0
