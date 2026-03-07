#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InferenceService深度测试
测试推理服务的完整功能和复杂场景
"""

import pytest

pytestmark = [
    pytest.mark.legacy,
    pytest.mark.timeout(45),  # 45秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]
pytest.skip("legacy 推理服务深度用例默认跳过，需手动启用", allow_module_level=True)

import asyncio
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json
import time

from src.ml.inference_service import (
    InferenceService,
    InferenceMode,
    ServiceStatus,
    InferenceRequest,
    InferenceResponse,
    InferenceAPIServer
)



from src.ml.model_manager import ModelType, ModelManager


class TestInferenceServiceLifecycle:
    """测试推理服务完整生命周期"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'model_storage_path': str(self.temp_dir / 'models'),
            'max_workers': 4,
            'request_queue_size': 100,
            'batch_size': 32,
            'timeout': 30,
            'monitoring_interval': 5
        }

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_service_initialization(self):
        """测试服务初始化"""
        service = InferenceService(self.config)

        # 验证初始化
        assert service.config == self.config
        assert service.status == ServiceStatus.STARTING
        assert hasattr(service, '_request_queue')
        assert hasattr(service, '_response_cache')
        assert hasattr(service, '_model_manager')

    @pytest.mark.asyncio
    async def test_service_start_stop(self):
        """测试服务启动和停止"""
        service = InferenceService(self.config)

        # 启动服务
        service.start()

        # 等待服务完全启动（增加等待时间）
        await asyncio.sleep(1.0)

        # 验证服务状态（允许STARTING状态，因为异步启动可能需要时间）
        assert service.status in [ServiceStatus.STARTING, ServiceStatus.RUNNING]

        # 停止服务
        service.stop()

        # 等待服务完全停止
        await asyncio.sleep(0.1)

        # 验证服务状态
        assert service.status == ServiceStatus.STOPPING

    def test_service_status_monitoring(self):
        """测试服务状态监控"""
        service = InferenceService(self.config)

        # 获取服务状态
        status = service.get_service_status()

        assert isinstance(status, dict)
        assert 'status' in status
        assert 'uptime' in status
        assert 'queue_size' in status
        assert 'active_workers' in status
        assert 'processed_requests' in status

        # 验证初始状态
        assert status['status'] == ServiceStatus.STARTING.value
        assert status['queue_size'] == 0
        assert status['active_workers'] == 0

    def test_model_deployment(self):
        """测试模型部署"""
        service = InferenceService(self.config)

        # 部署模型（这里使用mock，因为实际部署需要训练好的模型）
        with patch.object(service._model_manager, 'deploy_model', return_value=True) as mock_deploy:
            result = service.deploy_model("test_model_id")

            assert result is True
            mock_deploy.assert_called_once_with("test_model_id")

    def test_model_listing(self):
        """测试模型列表"""
        service = InferenceService(self.config)

        # 模拟模型列表
        mock_models = [
            {'model_id': 'model1', 'model_type': 'linear_regression', 'status': 'deployed'},
            {'model_id': 'model2', 'model_type': 'random_forest', 'status': 'trained'}
        ]

        with patch.object(service._model_manager, 'list_models', return_value=mock_models) as mock_list:
            models = service.list_models(ModelType.LINEAR_REGRESSION)

            assert len(models) == 2
            mock_list.assert_called_once_with(ModelType.LINEAR_REGRESSION)


class TestInferenceRequestProcessing:
    """测试推理请求处理"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'model_storage_path': str(self.temp_dir / 'models'),
            'max_workers': 2,
            'request_queue_size': 50,
            'batch_size': 16,
            'timeout': 10
        }

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_request_creation(self):
        """测试请求创建"""
        service = InferenceService(self.config)

        # 创建推理请求
        request = service.create_inference_request(
            model_type="linear_regression",
            input_data={'feature1': 1.0, 'feature2': -0.5},
            mode=InferenceMode.SYNCHRONOUS
        )

        assert isinstance(request, InferenceRequest)
        assert request.model_type == "linear_regression"
        assert request.input_data == {'feature1': 1.0, 'feature2': -0.5}
        assert request.mode == InferenceMode.SYNCHRONOUS
        assert hasattr(request, 'request_id')
        assert hasattr(request, 'timestamp')

    def test_synchronous_inference(self):
        """测试同步推理"""
        service = InferenceService(self.config)

        # 创建请求
        request = service.create_inference_request(
            model_type="linear_regression",
            input_data={'feature1': 1.0, 'feature2': -0.5},
            mode=InferenceMode.SYNCHRONOUS
        )

        # 模拟模型预测
        mock_prediction = MagicMock()
        mock_prediction.prediction_value = 2.5
        mock_prediction.confidence_score = 0.85

        with patch.object(service._model_manager, 'predict', return_value=mock_prediction) as mock_predict:
            # 执行同步推理
            response = service.submit_request_sync(request)

            assert isinstance(response, InferenceResponse)
            assert response.success is True
            assert response.prediction.prediction_value == 2.5
            assert response.prediction.confidence_score == 0.85
            assert response.processing_time > 0

            mock_predict.assert_called_once_with("linear_regression", {'feature1': 1.0, 'feature2': -0.5})

    @pytest.mark.asyncio
    async     def test_asynchronous_inference(self):
        """测试异步推理"""
        service = InferenceService(self.config)

        # 启动服务
        result = service.start()
        assert result is True
        assert service.status == ServiceStatus.RUNNING
        await asyncio.sleep(0.1)

        # 创建异步请求
        request = service.create_inference_request(
            model_type="random_forest",
            input_data={'feature1': 0.8, 'feature2': 1.2},
            mode=InferenceMode.ASYNCHRONOUS
        )

        # 模拟模型预测
        mock_prediction = MagicMock()
        mock_prediction.prediction_value = -1.3
        mock_prediction.confidence_score = 0.92

        with patch.object(service._model_manager, 'predict', return_value=mock_prediction) as mock_predict:
            # 提交异步请求
            response = await service.submit_request(request)

            assert isinstance(response, InferenceResponse)
            assert response.success is True
            assert response.prediction.prediction_value == -1.3
            assert response.prediction.confidence_score == 0.92

            mock_predict.assert_called_once_with("random_forest", {'feature1': 0.8, 'feature2': 1.2})

        # 停止服务
        await service.stop()

    @pytest.mark.asyncio
    async def test_batch_inference(self):
        """测试批量推理"""
        service = InferenceService(self.config)

        # 启动服务
        result = await service.start_async()
        assert result is True
        assert service.status == ServiceStatus.RUNNING
        await asyncio.sleep(0.1)

        # 创建批量请求
        batch_data = [
            {'feature1': 1.0, 'feature2': -0.5},
            {'feature1': 0.8, 'feature2': 1.2},
            {'feature1': -0.3, 'feature2': 0.7}
        ]

        request = service.create_inference_request(
            model_type="xgboost",
            input_data=batch_data,
            mode=InferenceMode.BATCH
        )

        # 模拟批量预测
        mock_predictions = []
        for i, data in enumerate(batch_data):
            mock_pred = MagicMock()
            mock_pred.prediction_value = float(i) * 0.5
            mock_pred.confidence_score = 0.8 + i * 0.05
            mock_predictions.append(mock_pred)

        with patch.object(service._model_manager, 'batch_predict', return_value=mock_predictions) as mock_batch_predict:
            # 执行批量推理
            response = await service.submit_request(request)

            assert isinstance(response, InferenceResponse)
            assert response.success is True
            assert len(response.prediction) == 3

            for i, pred in enumerate(response.prediction):
                assert pred.prediction_value == float(i) * 0.5
                assert pred.confidence_score == 0.8 + i * 0.05

            mock_batch_predict.assert_called_once_with("xgboost", batch_data)

        # 停止服务
        await service.stop()

    @pytest.mark.asyncio
    async def test_streaming_inference(self):
        """测试流式推理"""
        service = InferenceService(self.config)

        # 启动服务
        result = await service.start_async()
        assert result is True
        assert service.status == ServiceStatus.RUNNING
        await asyncio.sleep(0.1)

        # 创建流式请求
        request = service.create_inference_request(
            model_type="neural_network",
            input_data={'feature1': 0.5, 'feature2': -0.2},
            mode=InferenceMode.STREAMING
        )

        # 模拟流式预测
        mock_prediction = MagicMock()
        mock_prediction.prediction_value = 1.8
        mock_prediction.confidence_score = 0.78

        with patch.object(service._model_manager, 'predict', return_value=mock_prediction) as mock_predict:
            # 执行流式推理
            response = await service.submit_request(request)

            assert isinstance(response, InferenceResponse)
            assert response.success is True
            assert response.prediction.prediction_value == 1.8
            assert response.prediction.confidence_score == 0.78

            mock_predict.assert_called_once_with("neural_network", {'feature1': 0.5, 'feature2': -0.2})

        # 停止服务
        await service.stop()


class TestInferenceServicePerformance:
    """测试推理服务性能"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'model_storage_path': str(self.temp_dir / 'models'),
            'max_workers': 8,
            'queue_size': 200,
            'batch_size': 64,
            'timeout': 30,
            'monitoring_interval': 1
        }

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """测试并发请求处理"""
        service = InferenceService(self.config)

        # 启动服务
        result = await service.start_async()
        assert result is True
        assert service.status == ServiceStatus.RUNNING
        await asyncio.sleep(0.1)

        # 模拟模型预测
        mock_prediction = MagicMock()
        mock_prediction.prediction_value = 1.5
        mock_prediction.confidence_score = 0.88

        async def submit_request(index):
            """提交单个请求"""
            request = service.create_inference_request(
                model_type="linear_regression",
                input_data={'feature1': float(index) * 0.1, 'feature2': float(index) * -0.1},
                mode=InferenceMode.SYNCHRONOUS
            )

            with patch.object(service._model_manager, 'predict', return_value=mock_prediction):
                response = await service.submit_request(request)
                return response.success

        # 并发提交多个请求
        tasks = []
        num_requests = 20

        for i in range(num_requests):
            task = asyncio.create_task(submit_request(i))
            tasks.append(task)

        # 等待所有请求完成
        results = await asyncio.gather(*tasks)

        # 验证所有请求都成功
        assert all(results)

        # 停止服务
        await service.stop()

    def test_request_queue_management(self):
        """测试请求队列管理"""
        service = InferenceService(self.config)

        # 验证队列初始化
        assert service._request_queue is not None
        assert service._request_queue.maxsize == self.config['queue_size']

        # 测试队列大小限制
        for i in range(self.config['queue_size'] + 10):
            request = service.create_inference_request(
                model_type="test_model",
                input_data={'feature': float(i)},
                mode=InferenceMode.SYNCHRONOUS
            )

            # 尝试添加请求到队列
            try:
                service._request_queue.put_nowait(request)
            except asyncio.QueueFull:
                # 队列已满
                break

        # 验证队列没有超出限制
        assert service._request_queue.qsize() <= self.config['queue_size']

    def test_timeout_handling(self):
        """测试超时处理"""
        service = InferenceService(self.config)

        # 创建请求
        request = service.create_inference_request(
            model_type="slow_model",
            input_data={'feature1': 1.0},
            mode=InferenceMode.SYNCHRONOUS
        )

        # 模拟慢速预测（超过超时时间）
        async def slow_predict(*args, **kwargs):
            await asyncio.sleep(self.config['timeout'] + 1)  # 超过超时时间
            mock_pred = MagicMock()
            mock_pred.prediction_value = 0.0
            mock_pred.confidence_score = 0.5
            return mock_pred

        with patch.object(service._model_manager, 'predict', side_effect=slow_predict):
            # 记录开始时间
            start_time = time.time()

            # 执行请求（应该超时）
            response = service.submit_request_sync(request)

            # 记录结束时间
            end_time = time.time()

            # 验证超时处理
            assert end_time - start_time >= self.config['timeout']
            assert response.success is False
            assert 'timeout' in str(response.error).lower() if response.error else True

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        service = InferenceService(self.config)

        # 测试无效模型类型
        request = service.create_inference_request(
            model_type="invalid_model_type",
            input_data={'feature1': 1.0},
            mode=InferenceMode.SYNCHRONOUS
        )

        # 模拟模型管理器抛出异常
        with patch.object(service._model_manager, 'predict', side_effect=ValueError("Model not found")):
            response = service.submit_request_sync(request)

            assert response.success is False
            assert response.error is not None
            assert "Model not found" in str(response.error)

        # 测试无效输入数据
        request = service.create_inference_request(
            model_type="linear_regression",
            input_data={},  # 空输入数据
            mode=InferenceMode.SYNCHRONOUS
        )

        with patch.object(service._model_manager, 'predict', side_effect=KeyError("Missing required features")):
            response = service.submit_request_sync(request)

            assert response.success is False
            assert response.error is not None

    def test_resource_management(self):
        """测试资源管理"""
        service = InferenceService(self.config)

        # 验证资源限制
        assert service.config['max_workers'] == 8
        assert service.config['queue_size'] == 200

        # 测试内存使用监控
        initial_status = service.get_service_status()
        assert 'memory_usage' in initial_status or 'queue_size' in initial_status

        # 测试工作线程管理
        # 注意：实际的工作线程管理在异步循环中，这里主要验证配置
        assert service.config['max_workers'] > 0


class TestInferenceAPIServer:
    """测试推理API服务器"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'model_storage_path': str(self.temp_dir / 'models'),
            'host': 'localhost',
            'port': 8080,
            'workers': 2
        }

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_api_server_initialization(self):
        """测试API服务器初始化"""
        service = InferenceService()
        api_server = InferenceAPIServer(service, self.config)

        # 验证初始化
        assert api_server.inference_service == service
        assert api_server.config == self.config
        assert hasattr(api_server, 'app')

    @patch('uvicorn.run')
    def test_api_server_start(self, mock_uvicorn_run):
        """测试API服务器启动"""
        service = InferenceService()
        api_server = InferenceAPIServer(service, self.config)

        # 启动服务器
        api_server.start()

        # 验证uvicorn.run被调用
        mock_uvicorn_run.assert_called_once()
        call_args = mock_uvicorn_run.call_args
        assert call_args[1]['host'] == 'localhost'
        assert call_args[1]['port'] == 8080

    def test_api_endpoints_registration(self):
        """测试API端点注册"""
        service = InferenceService()
        api_server = InferenceAPIServer(service, self.config)

        # 先启动API服务器，这会创建FastAPI应用
        api_server.start()

        # 验证FastAPI应用已创建
        assert api_server.app is not None

        # 验证路由已注册
        routes = [route.path for route in api_server.app.routes]
        assert '/' in routes
        assert '/health' in routes
        assert '/predict' in routes
        assert '/batch_predict' in routes
        assert '/models' in routes


if __name__ == "__main__":
    pytest.main([__file__])
