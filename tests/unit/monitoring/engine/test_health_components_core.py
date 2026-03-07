#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health组件核心测试
补充health_components.py的测试覆盖
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    engine_health_components_module = importlib.import_module('src.monitoring.engine.health_components')
    HealthComponentFactory = getattr(engine_health_components_module, 'HealthComponentFactory', None)
    if HealthComponentFactory is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestHealthComponentFactory:
    """测试HealthComponentFactory类"""

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        if HealthComponentFactory is not None:
            info = HealthComponentFactory.get_factory_info()
            assert isinstance(info, dict)
            assert 'factory_name' in info
            assert 'version' in info
            assert 'total_healths' in info

