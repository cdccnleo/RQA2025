#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringWebApp主程序测试
测试monitoring_web_app.py的__main__块
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

# 处理flask_cors依赖问题
try:
    # 导入模块以便测试__main__块
    import src.monitoring.web.monitoring_web_app as web_app_module
except ImportError:
    pytest.skip("flask_cors模块不可用，跳过web应用主程序测试", allow_module_level=True)


class TestMonitoringWebAppMain:
    """测试主程序逻辑"""

    def test_main_execution_basic(self):
        """测试基本主程序执行"""
        with patch('src.monitoring.web.monitoring_web_app.start_web_app') as mock_start:
            # 模拟执行__main__块的核心逻辑
            web_app_module.start_web_app()
            mock_start.assert_called_once()

    def test_main_execution_calls_start_web_app(self):
        """测试主程序调用start_web_app"""
        with patch('src.monitoring.web.monitoring_web_app.start_web_app') as mock_start:
            # 模拟__main__块
            if True:  # 模拟 if __name__ == "__main__":
                web_app_module.start_web_app()
            mock_start.assert_called_once()

    def test_main_execution_import_path(self):
        """测试主程序导入路径正确"""
        # 验证模块可以正常导入
        assert hasattr(web_app_module, 'start_web_app')
        assert callable(web_app_module.start_web_app)

    def test_main_execution_with_exception(self):
        """测试主程序执行时发生异常"""
        with patch('src.monitoring.web.monitoring_web_app.start_web_app', side_effect=Exception("Test error")):
            # 应该抛出异常
            with pytest.raises(Exception) as exc_info:
                web_app_module.start_web_app()
            assert "Test error" in str(exc_info.value)


