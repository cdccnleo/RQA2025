import pytest

from src.ml.core.ml_service import MLService, MLServiceStatus
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class _FakeInferenceService:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.raise_error = False
        self.predictions = []

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def predict(self, data, mode=None):
        if self.raise_error:
            raise RuntimeError("inference failed")
        self.predictions.append((data, mode))
        return {"data": data, "mode": mode}


class _FakeModelManager:
    def __init__(self, models=None):
        self.models = models or [{"model_id": "m1"}, {"model_id": "m2"}]
        self.info_calls = []

    def list_models(self):
        return self.models

    def get_model_info(self, model_id):
        self.info_calls.append(model_id)
        return {"model_id": model_id, "status": "ready"}


@pytest.fixture
def ml_service():
    inference = _FakeInferenceService()
    manager = _FakeModelManager()
    service = MLService(
        {
            "inference_service": inference,
            "model_manager": manager,
            "max_workers": 1,
        }
    )
    yield service, inference, manager
    service.executor.shutdown(wait=True)


def test_start_and_stop_transitions(ml_service):
    service, inference, _ = ml_service

    assert service.status == MLServiceStatus.STOPPED
    assert service.start() is True
    assert service.status == MLServiceStatus.RUNNING
    assert inference.started == 1

    # 再次启动应直接返回
    assert service.start() is True
    assert inference.started == 1

    service.stop()
    assert service.status == MLServiceStatus.STOPPED
    assert inference.stopped == 1

    # 未运行时 stop 不会触发额外调用
    service.stop()
    assert inference.stopped == 1


@pytest.mark.asyncio
async def test_start_async_uses_sync_start(ml_service):
    service, inference, _ = ml_service
    assert service.status == MLServiceStatus.STOPPED

    await service.start_async()
    assert service.status == MLServiceStatus.RUNNING
    assert inference.started == 1

    service.stop()


def test_predict_success_updates_stats(ml_service):
    service, inference, _ = ml_service
    service.start()

    result = service.predict({"x": 1}, mode="sync")
    assert result == {"data": {"x": 1}, "mode": "sync"}
    assert inference.predictions == [({"x": 1}, "sync")]
    assert service.stats["inference_requests"] == 1
    assert service.stats["inference_success"] == 1
    assert service.stats["inference_failed"] == 0


def test_predict_failure_tracks_errors(ml_service):
    service, inference, _ = ml_service
    service.start()
    inference.raise_error = True

    with pytest.raises(RuntimeError):
        service.predict({"boom": True})

    assert service.stats["inference_requests"] == 1
    assert service.stats["inference_success"] == 0
    assert service.stats["inference_failed"] == 1
    assert service.status == MLServiceStatus.RUNNING


def test_get_service_info_includes_models_and_stats(ml_service):
    service, _, manager = ml_service
    info = service.get_service_info()

    assert info["status"] == service.status.value
    assert info["models"]["total_models"] == len(manager.models)
    assert info["models"]["items"] == manager.models
    assert info["stats"] == service.stats


def test_model_manager_delegation(ml_service):
    service, _, manager = ml_service

    models = service.list_models()
    assert models == manager.models

    info = service.get_model_info("m2")
    assert info == {"model_id": "m2", "status": "ready"}
    assert manager.info_calls == ["m2"]


class FailingLogger:
    def __init__(self):
        self.messages = []

    def error(self, msg, *args, **kwargs):
        self.messages.append(("error", msg, args, kwargs))

    def info(self, msg, *args, **kwargs):
        self.messages.append(("info", msg, args, kwargs))


class NoAdapter:
    def get_models_logger(self):
        raise RuntimeError("logger boom")


def test_start_failure_moves_to_error_state(monkeypatch):
    class FailingInference:
        def start(self):
            raise RuntimeError("boom")

    service = MLService({"inference_service": FailingInference(), "max_workers": 1})
    assert service.start() is False
    assert service.status == MLServiceStatus.ERROR
    # 基础错误路径被触发即可


def test_predict_requires_running_service():
    service = MLService({"max_workers": 1})
    with pytest.raises(RuntimeError):
        service.predict({"x": 1})

    assert service.stats["inference_requests"] == 0
    assert service.stats["inference_success"] == 0
    assert service.stats["inference_failed"] == 0


def test_stop_without_inference_stop_method():
    class NoStopInference:
        def start(self):
            return True

    service = MLService({"inference_service": NoStopInference(), "max_workers": 1})
    service.start()
    service.stop()

    assert service.status == MLServiceStatus.STOPPED
import asyncio
from unittest.mock import Mock, patch

import pytest

from src.ml.core.ml_service import MLService, MLServiceStatus
from src.ml.core.model_manager import ModelPrediction


@pytest.fixture(autouse=True)
def patch_models_adapter():
    with patch("src.ml.core.ml_service.get_models_adapter") as mock_adapter:
        mock_adapter.return_value = Mock(get_models_logger=lambda: Mock())
        yield mock_adapter


class DummyInferenceService:
    def __init__(self):
        self.started = False
        self.predictions = []

    def start(self):
        self.started = True
        return True

    async def start_async(self):
        self.start()
        return True

    def stop(self):
        self.started = False

    def predict(self, data, mode=None):
        self.predictions.append((data, mode))
        return {"predictions": [{"label": "A", "score": 0.9}], "metadata": {"mode": "sync"}}


class DummyModelManager:
    def __init__(self):
        self.loaded = True
        self.models = {"model-A": {"version": "1.0.0"}}

    def list_models(self):
        return list(self.models.keys())

    def get_model_info(self, model_id):
        return self.models.get(model_id)

    def load_model(self, model_id):
        return ModelPrediction(
            request_id="",
            model_type=model_id,
            prediction=None,
            confidence=1.0,
            processing_time_ms=0.0,
            success=True,
        )


@pytest.fixture
def service():
    ml_service = MLService(config={"max_workers": 2})
    ml_service.inference_service = DummyInferenceService()
    ml_service.model_manager = DummyModelManager()
    return ml_service


def test_start_and_stop_service(service):
    assert service.status == MLServiceStatus.STOPPED

    service.start()
    assert service.status == MLServiceStatus.RUNNING
    assert service.inference_service.started is True

    service.stop()
    assert service.status == MLServiceStatus.STOPPED
    assert service.inference_service.started is False


@pytest.mark.asyncio
async def test_async_start(service):
    await service.start_async()
    assert service.status == MLServiceStatus.RUNNING


def test_get_service_info(service):
    service.start()
    info = service.get_service_info()

    assert info["status"] == MLServiceStatus.RUNNING.value
    assert "models" in info
    assert info["models"]["total_models"] == 1


def test_predict_delegates_to_inference_service(service):
    service.start()
    result = service.predict(data={"feature": 1.0})

    assert result["predictions"][0]["label"] == "A"
    assert service.inference_service.predictions


def test_list_models_and_get_model(service):
    models = service.list_models()
    model_ids = [m["model_id"] if isinstance(m, dict) else m for m in models]
    assert "model-A" in model_ids

    info = service.get_model_info("model-A")
    assert info["version"] == "1.0.0"


def test_default_model_manager_and_feature_engineering():
    service = MLService({"max_workers": 1})
    # 默认特征工程应原样返回
    payload = {"feature": 1}
    assert service.feature_engineering.process(payload) is payload
    # 默认模型管理返回空列表/None
    assert service.list_models() == []
    assert service.get_model_info("missing") is None
    service.executor.shutdown(wait=True)


def test_list_models_handles_missing_manager_methods():
    service = MLService({"max_workers": 1})
    service.model_manager = object()
    assert service.list_models() == []
    assert service.get_model_info("anything") is None
    service.executor.shutdown(wait=True)


def test_default_inference_service_requires_predict_configuration():
    service = MLService({"max_workers": 1})
    assert service.status == MLServiceStatus.STOPPED
    service.start()
    assert service.status == MLServiceStatus.RUNNING

    with pytest.raises(RuntimeError):
        service.predict({"x": 1})

    service.stop()
    assert service.status == MLServiceStatus.STOPPED
    service.executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_default_inference_service_start_async():
    service = MLService({"max_workers": 1})
    result = await service.inference_service.start_async()
    assert result is True
    service.executor.shutdown(wait=True)

