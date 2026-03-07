#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层测试配置和导入辅助
提供统一的导入逻辑，解决pytest-xdist并发环境下的导入问题
"""

import sys
import os
import importlib
from typing import Optional, Tuple, Any
import pytest

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

# 在模块加载时确保路径正确（强制添加到最前面）
# 移除旧路径后重新插入，确保src路径在前面
for path in [PROJECT_ROOT, SRC_PATH]:
    if path in sys.path:
        sys.path.remove(path)

# 确保src路径在最前面，这样可以正确解析src.infrastructure模块
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, PROJECT_ROOT)

# 添加pytest hook确保在测试运行前路径正确
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """pytest配置钩子，确保路径正确"""
    # 移除旧路径后重新插入到前面
    for path in [PROJECT_ROOT, SRC_PATH]:
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, SRC_PATH)  # src路径应该在前面
    sys.path.insert(0, PROJECT_ROOT)

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """pytest收集测试项目钩子，设置路径"""
    # 再次确保路径正确
    for path in [PROJECT_ROOT, SRC_PATH]:
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, SRC_PATH)
    sys.path.insert(0, PROJECT_ROOT)

@pytest.fixture(scope="session", autouse=True)
def setup_infrastructure_test_environment():
    """设置基础设施层测试环境"""
    # 注意：我们使用Mock对象进行测试，不强制要求模块导入
    # 这允许我们在模块不可用时仍能进行测试覆盖率提升
    yield

    # 清理工作（如果需要）

@pytest.fixture
def mock_infrastructure_logger():
    """提供mock的基础设施层logger"""
    from unittest.mock import Mock

    mock_logger = Mock()
    mock_logger.info = Mock()
    mock_logger.error = Mock()
    mock_logger.warning = Mock()
    mock_logger.debug = Mock()

    return mock_logger
