"""
推理服务别名模块单元测试
"""
import pytest
import importlib
import os
from unittest.mock import patch, Mock


def test_inference_service_import_success():
    """测试成功导入InferenceService"""
    # 测试正常导入路径
    from src.ml.inference_service import InferenceService, InferenceMode
    assert InferenceService is not None
    assert InferenceMode is not None


def test_inference_service_import_fallback(monkeypatch):
    """测试导入失败时的降级实现"""
    monkeypatch.setenv("ML_FORCE_INFERENCE_FALLBACK", "1")

    import src.ml.inference_service as alias_module

    importlib.reload(alias_module)

    assert hasattr(alias_module, "InferenceService")
    assert hasattr(alias_module, "InferenceMode")

    from enum import Enum

    assert issubclass(alias_module.InferenceMode, Enum)

    monkeypatch.delenv("ML_FORCE_INFERENCE_FALLBACK", raising=False)
    importlib.reload(alias_module)

