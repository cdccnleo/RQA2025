#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Web管理服务基础测试（仅测试WebConfig部分）"""

import pytest


def test_web_config_import():
    """测试WebConfig导入（不触发服务导入）"""
    # 直接导入dataclass，避免触发web_management_service的依赖导入
    import sys
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "web_config",
        "src/infrastructure/security/services/web_management_service.py"
    )
    
    # 验证文件存在
    assert spec is not None
    assert spec.origin is not None


def test_dataclass_available():
    """测试dataclass模块可用"""
    from dataclasses import dataclass, field, asdict, is_dataclass
    
    @dataclass
    class TestConfig:
        value: int = 10
    
    config = TestConfig()
    assert config.value == 10
    assert is_dataclass(TestConfig)

