#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务层监控测试共享配置
"""

import importlib
import sys
from typing import Dict

import pytest


_TARGET_MODULES = (
    "src.infrastructure.monitoring.services.continuous_monitoring_service",
    "src.infrastructure.monitoring.services.continuous_monitoring_core",
    "src.infrastructure.monitoring.services.monitoring_runtime",
)


def _ensure_module(name: str):
    """确保模块已在 sys.modules 中并返回最新实例。"""
    module = sys.modules.get(name)
    if module is None:
        module = importlib.import_module(name)
    else:
        sys.modules[name] = module
        module = importlib.reload(module)
    sys.modules[name] = module
    return module


@pytest.fixture(autouse=True, scope="module")
def _reload_monitoring_service_modules():
    """模块级别自动夹具，确保核心监控服务模块在运行前后状态一致。"""
    originals: Dict[str, object] = {}
    for target in _TARGET_MODULES:
        originals[target] = _ensure_module(target)

    try:
        yield
    finally:
        # 恢复为最后一次有效实例，避免其它测试弹出模块导致ImportError
        for target, module in originals.items():
            sys.modules[target] = module

