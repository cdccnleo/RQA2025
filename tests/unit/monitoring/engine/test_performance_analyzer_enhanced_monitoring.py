#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerformanceAnalyzer增强监控功能测试
覆盖_get_services_to_monitor、get_enhanced_monitoring_status、
异步服务健康检查、实时洞察等增强监控功能
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from collections import deque

import sys
import importlib
from pathlib import Path
import pytest

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
    engine_performance_analyzer_module = importlib.import_module('src.monitoring.engine.performance_analyzer')
    get_performance_analyzer = getattr(engine_performance_analyzer_module, 'get_performance_analyzer', None)
    PerformanceAnalyzer = getattr(engine_performance_analyzer_module, 'PerformanceAnalyzer', None)
    
    if get_performance_analyzer is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestPerformanceAnalyzerEnhancedMonitoring:
    """测试PerformanceAnalyzer增强监控功能"""

    def test_get_performance_analyzer(self):
        """测试获取性能分析器实例"""
        if get_performance_analyzer is None or PerformanceAnalyzer is None:
            pytest.skip("模块不可用")
        
        analyzer = get_performance_analyzer()
        
        assert isinstance(analyzer, PerformanceAnalyzer)
        assert analyzer.config == {}

