#!/usr/bin/env python3
"""
精简版 ML 核心服务，提供基础的启动、推理与模型管理能力。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

try:  # pragma: no cover
    from src.core.foundation.interfaces.ml_strategy_interfaces import (  # type: ignore
        IMLService,
        IMLFeatureEngineering,
        MLFeatures,
        MLInferenceRequest,
        MLInferenceResponse,
    )
except ImportError:  # pragma: no cover
    @dataclass
    class MLFeatures:
        data: Dict[str, Any]

    @dataclass
    class MLInferenceRequest:
        model_id: str
        features: MLFeatures
        inference_type: str = "sync"

    @dataclass
    class MLInferenceResponse:
        request_id: str
        success: bool
        prediction: Optional[Any] = None
        confidence: Optional[float] = None
        processing_time_ms: float = 0.0
        error_message: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None

    class IMLFeatureEngineering:
        def process(self, features: MLFeatures) -> MLFeatures:
            return features

    class IMLService:
        pass


try:  # pragma: no cover
    from src.core.integration import get_models_adapter as _get_models_adapter
except ImportError:  # pragma: no cover
    class _FallbackModelsAdapter:
        def get_models_logger(self):
            return logging.getLogger(__name__)

        def get_models_cache_manager(self):
            return None

        def get_models_config_manager(self):
            return None

    def _get_models_adapter():
        return _FallbackModelsAdapter()


get_models_adapter = _get_models_adapter


class MLServiceStatus(Enum):
    """服务状态枚举"""

    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


class _DefaultFeatureEngineering(IMLFeatureEngineering):
    def process(self, features: MLFeatures) -> MLFeatures:
        return features


class _DefaultModelManager:
    def list_models(self):
        return []

    def get_model_info(self, model_id: str):
        return None


class _DefaultInferenceService:
    def start(self):
        return True

    async def start_async(self):
        return True

    def stop(self):
        return True

    def predict(self, data, mode=None):
        raise RuntimeError("Inference service not configured")


class MLService(IMLService):
    """轻量级 ML 核心服务实现"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.status = MLServiceStatus.STOPPED

        try:  # pragma: no cover
            adapter = get_models_adapter()
            self.logger = adapter.get_models_logger()
        except Exception:  # pragma: no cover
            self.logger = logging.getLogger(__name__)

        self.max_workers = self.config.get("max_workers", 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.inference_service = self.config.get("inference_service") or _DefaultInferenceService()
        self.model_manager = self.config.get("model_manager") or _DefaultModelManager()
        self.feature_engineering = self.config.get("feature_engineering") or _DefaultFeatureEngineering()

        self.stats = {
            "inference_requests": 0,
            "inference_success": 0,
            "inference_failed": 0,
        }

    # ------------------------------------------------------------------ #
    # 生命周期管理
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        if self.status == MLServiceStatus.RUNNING:
            return True

        try:
            if hasattr(self.inference_service, "start"):
                self.inference_service.start()
            self.status = MLServiceStatus.RUNNING
            return True
        except Exception as exc:  # pragma: no cover
            self.logger.error("启动 MLService 失败: %s", exc)
            self.status = MLServiceStatus.ERROR
            return False

    async def start_async(self) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.start)

    def stop(self) -> None:
        if self.status != MLServiceStatus.RUNNING:
            return
        if hasattr(self.inference_service, "stop"):
            self.inference_service.stop()
        self.status = MLServiceStatus.STOPPED

    # ------------------------------------------------------------------ #
    # 服务信息
    # ------------------------------------------------------------------ #
    def get_service_info(self) -> Dict[str, Any]:
        models = self.list_models()
        return {
            "status": self.status.value,
            "config": self.config,
            "models": {
                "total_models": len(models),
                "items": models,
            },
            "stats": self.stats.copy(),
        }

    # ------------------------------------------------------------------ #
    # 推理相关
    # ------------------------------------------------------------------ #
    def predict(self, data: Any, mode: Optional[str] = None) -> Any:
        if self.status != MLServiceStatus.RUNNING:
            raise RuntimeError("MLService 未启动")

        self.stats["inference_requests"] += 1
        try:
            prediction = self.inference_service.predict(data, mode=mode)
            self.stats["inference_success"] += 1
            return prediction
        except Exception as exc:
            self.stats["inference_failed"] += 1
            self.logger.error("推理失败: %s", exc)
            raise

    # ------------------------------------------------------------------ #
    # 模型管理
    # ------------------------------------------------------------------ #
    def list_models(self) -> List[Any]:
        if hasattr(self.model_manager, "list_models"):
            return self.model_manager.list_models()
        return []

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        if hasattr(self.model_manager, "get_model_info"):
            return self.model_manager.get_model_info(model_id)
        return None

    # 预留扩展接口（当前测试未覆盖，保持占位）
    def load_model(self, model_id: str, model_config: Dict[str, Any]) -> bool:  # pragma: no cover
        return False

    def unload_model(self, model_id: str) -> bool:  # pragma: no cover
        return False


__all__ = [
    "MLService",
    "MLServiceStatus",
    "get_models_adapter",
]

