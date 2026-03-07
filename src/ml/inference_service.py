"""
推理服务模块（别名模块）
提供向后兼容的导入路径

实际实现在 core/inference_service.py 中
"""

import os

_FORCE_FALLBACK = os.environ.get("ML_FORCE_INFERENCE_FALLBACK") == "1"

try:
    if _FORCE_FALLBACK:
        raise ImportError("Forced fallback for testing")
    from .core.inference_service import InferenceService, InferenceMode
except ImportError:
    # 提供基础实现
    class InferenceService:  # type: ignore
        pass

    from enum import Enum

    class InferenceMode(Enum):  # type: ignore
        SYNCHRONOUS = "synchronous"
        ASYNCHRONOUS = "asynchronous"

__all__ = ["InferenceService", "InferenceMode"]

