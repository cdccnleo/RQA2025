import asyncio
from unittest.mock import Mock, patch
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.ml.core.inference_service import (
    InferenceMode,
    InferenceRequest,
    InferenceResponse,
    InferenceService,
    ServiceStatus,
)
from src.ml.core.model_manager import ModelPrediction


class DummyModelManager:
    def __init__(self):
        self.predict_calls = []
        self.batch_predict_calls = []

    def predict(self, model_type, input_data):
        self.predict_calls.append((model_type, input_data))
        return [
            {"label": "A", "score": 0.9},
            {"label": "B", "score": 0.8},
        ]

    def batch_predict(self, model_type, input_data):
        self.batch_predict_calls.append((model_type, input_data))
        return [
            ModelPrediction(
                request_id="req",
                model_type=model_type,
                prediction={"label": "batch", "score": 0.95},
                confidence=0.9,
                processing_time_ms=10.0,
                success=True,
            )
        ]


@pytest.fixture(autouse=True)
def patch_models_adapter():
    with patch("ml.core.inference_service.get_models_adapter") as mock_adapter:
        mock_adapter.return_value = Mock(get_models_logger=lambda: Mock())
        yield mock_adapter


@pytest.fixture
def inference_service():
    service = InferenceService({"max_workers": 2})
    service.model_manager = DummyModelManager()
    service.status = ServiceStatus.RUNNING
    service.loop = asyncio.new_event_loop()
    yield service
    if service.loop and service.loop.is_running():
        service.loop.stop()
    if service.loop and not service.loop.is_closed():
        service.loop.close()
    service.executor.shutdown(wait=False)


def test_predict_synchronous(inference_service):
    data = pd.DataFrame({"feature": [1.0, 2.0]})
    result = inference_service.predict(data, mode=InferenceMode.SYNCHRONOUS)

    assert result["metadata"]["mode"] == InferenceMode.SYNCHRONOUS.value
    assert len(result["predictions"]) == 2
    assert inference_service.model_manager.predict_calls


def test_predict_batch_dataframe(inference_service):
    data = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    result = inference_service.predict(data, mode=InferenceMode.BATCH)

    assert result["metadata"]["mode"] == InferenceMode.BATCH.value
    assert result["metadata"]["batch_size"] == len(data)
    assert inference_service.model_manager.predict_calls


@pytest.mark.asyncio
async def test_submit_request(inference_service):
    inference_service.request_queue = asyncio.Queue()
    inference_service.response_queue = asyncio.Queue()

    request = InferenceRequest(
        request_id="req-1",
        model_type="basic",
        input_data={"feature": 1.0},
        mode=InferenceMode.SYNCHRONOUS,
    )

    async def worker():
        response = await inference_service._process_request(request)
        await inference_service.response_queue.put(response)

    task = asyncio.create_task(worker())
    response = await inference_service.submit_request(request)

    assert response.success is True
    assert isinstance(response, InferenceResponse)
    await task


def test_get_service_status(inference_service):
    status = inference_service.get_service_status()
    assert status["status"] == ServiceStatus.RUNNING.value
    assert status["configuration"]["max_workers"] == 2


def test_start_when_already_running_returns_true():
    service = InferenceService({"model_manager": DummyModelManager()})
    service.status = ServiceStatus.RUNNING
    try:
        assert service.start() is True
        assert service.status == ServiceStatus.RUNNING
    finally:
        service.executor.shutdown(wait=False)
        if service.loop and not service.loop.is_closed():
            service.loop.close()


def test_stop_is_noop_when_service_not_running():
    service = InferenceService({"model_manager": DummyModelManager()})
    try:
        service.stop()
        assert service.status == ServiceStatus.STOPPED
    finally:
        service.executor.shutdown(wait=False)
        if service.loop and not service.loop.is_closed():
            service.loop.close()


@pytest.mark.asyncio
async def test_submit_request_requires_running_service():
    service = InferenceService({"model_manager": DummyModelManager()})
    request = InferenceRequest(
        request_id="req-not-running",
        model_type="default",
        input_data={"value": 0},
    )
    with pytest.raises(RuntimeError, match="not running"):
        await service.submit_request(request)
    service.executor.shutdown(wait=False)
    if service.loop and not service.loop.is_closed():
        service.loop.close()


def test_start_initializes_event_loop():
    service = InferenceService({"model_manager": DummyModelManager()})
    try:
        assert service.status == ServiceStatus.STOPPED
        assert service.start() is True
        assert service.status == ServiceStatus.RUNNING
        assert service.loop is not None
    finally:
        service.stop()
        service.executor.shutdown(wait=False)
        if service.loop and not service.loop.is_closed():
            service.loop.close()


@pytest.mark.asyncio
async def test_start_async_executes_in_background():
    service = InferenceService({"model_manager": DummyModelManager()})
    try:
        result = await service.start_async()
        assert result is True
        assert service.status == ServiceStatus.RUNNING
    finally:
        service.stop()
        service.executor.shutdown(wait=False)
        if service.loop and not service.loop.is_closed():
            service.loop.close()


def test_predict_requires_running_service():
    service = InferenceService({"model_manager": DummyModelManager()})
    with pytest.raises(RuntimeError, match="not running"):
        service.predict({"feature": 1})
    service.executor.shutdown(wait=False)
    if service.loop and not service.loop.is_closed():
        service.loop.close()


@pytest.mark.asyncio
async def test_submit_request_timeout_falls_back_to_inline_processing(monkeypatch):
    service = InferenceService({"model_manager": DummyModelManager()})
    try:
        service.start()

        async def fake_wait_for(coro, timeout):
            raise asyncio.TimeoutError

        monkeypatch.setattr("ml.core.inference_service.asyncio.wait_for", fake_wait_for)

        request = InferenceRequest(
            request_id="req-timeout",
            model_type="default",
            input_data={"value": 1},
            timeout_seconds=0,
        )

        response = await service.submit_request(request)
        assert response.success is True
        assert service.stats["requests_processed"] == 1
        queued_request = await service.request_queue.get()
        assert queued_request.request_id == "req-timeout"
    finally:
        service.stop()
        service.executor.shutdown(wait=False)
        if service.loop and not service.loop.is_closed():
            service.loop.close()


@pytest.mark.asyncio
async def test_process_request_records_failures():
    class FailingModelManager:
        def predict(self, model_type, input_data):
            raise ValueError("prediction failed")

    service = InferenceService({"model_manager": FailingModelManager()})
    try:
        service.start()

        request = InferenceRequest(
            request_id="req-fail",
            model_type="default",
            input_data={"value": 2},
        )

        response = await service.submit_request_sync(request)
        assert response.success is False
        assert "prediction failed" in response.error_message
        assert service.stats["requests_failed"] == 1
    finally:
        service.stop()
        service.executor.shutdown(wait=False)
        if service.loop and not service.loop.is_closed():
            service.loop.close()

