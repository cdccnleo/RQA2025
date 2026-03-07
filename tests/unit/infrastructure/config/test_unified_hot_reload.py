#!/usr/bin/env python3
"""
测试unified_hot_reload模块

测试覆盖：
- UnifiedHotReload类的初始化和方法
- 全局函数start_hot_reload和stop_hot_reload
- 文件监视功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

try:
    from src.infrastructure.config.services.unified_hot_reload import (
        UnifiedHotReload, 
        start_hot_reload, 
        stop_hot_reload
    )
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestUnifiedHotReload:
    """测试UnifiedHotReload类"""

    def setup_method(self):
        """测试前准备"""
        # 重置全局实例
        import src.infrastructure.config.services.unified_hot_reload as module
        module._hot_reload_instance = None

    def test_initialization_with_hot_reload_disabled(self):
        """测试禁用热重载的初始化"""
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        
        assert hot_reload.enable_hot_reload is False
        assert hot_reload._hot_reload_service is None
        assert isinstance(hot_reload._watched_files, set)
        assert len(hot_reload._watched_files) == 0

    @patch('src.infrastructure.config.services.unified_hot_reload.HotReloadService')
    def test_initialization_with_hot_reload_enabled(self, mock_hot_reload_service_class):
        """测试启用热重载的初始化"""
        mock_service = Mock()
        mock_hot_reload_service_class.return_value = mock_service
        
        hot_reload = UnifiedHotReload(enable_hot_reload=True)
        
        assert hot_reload.enable_hot_reload is True
        assert hot_reload._hot_reload_service is mock_service
        assert isinstance(hot_reload._watched_files, set)

    @patch('src.infrastructure.config.services.unified_hot_reload.logger')
    def test_start_hot_reload_disabled(self, mock_logger):
        """测试禁用热重载时的启动"""
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        
        result = hot_reload.start_hot_reload()
        
        assert result is False
        mock_logger.warning.assert_called_with("热重载功能未启用")

    @patch('src.infrastructure.config.services.unified_hot_reload.logger')
    def test_start_hot_reload_enabled_success(self, mock_logger):
        """测试启用热重载时的成功启动"""
        mock_service = Mock()
        mock_service.start.return_value = True
        
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        hot_reload.enable_hot_reload = True
        hot_reload._hot_reload_service = mock_service
        
        result = hot_reload.start_hot_reload()
        
        assert result is True
        mock_service.start.assert_called_once()
        mock_logger.info.assert_called_with("热重载服务启动成功")

    @patch('src.infrastructure.config.services.unified_hot_reload.logger')
    def test_start_hot_reload_enabled_exception(self, mock_logger):
        """测试启用热重载时的异常情况"""
        mock_service = Mock()
        mock_service.start.side_effect = Exception("Test exception")
        
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        hot_reload.enable_hot_reload = True
        hot_reload._hot_reload_service = mock_service
        
        result = hot_reload.start_hot_reload()
        
        assert result is False
        mock_logger.error.assert_called()

    def test_stop_hot_reload_disabled(self):
        """测试禁用热重载时的停止"""
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        
        result = hot_reload.stop_hot_reload()
        
        assert result is True

    @patch('src.infrastructure.config.services.unified_hot_reload.logger')
    def test_stop_hot_reload_enabled_success(self, mock_logger):
        """测试启用热重载时的成功停止"""
        mock_service = Mock()
        mock_service.stop.return_value = True
        
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        hot_reload.enable_hot_reload = True
        hot_reload._hot_reload_service = mock_service
        
        result = hot_reload.stop_hot_reload()
        
        assert result is True
        mock_service.stop.assert_called_once()
        mock_logger.info.assert_called_with("热重载服务停止成功")

    def test_watch_file_disabled(self):
        """测试禁用热重载时的文件监视"""
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        
        result = hot_reload.watch_file("/test/path.json")
        
        assert result is False

    @patch('src.infrastructure.config.services.unified_hot_reload.logger')
    def test_watch_file_enabled_with_callback(self, mock_logger):
        """测试启用热重载时的文件监视（带回调）"""
        mock_service = Mock()
        mock_service.watch_file.return_value = True
        callback = Mock()
        
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        hot_reload.enable_hot_reload = True
        hot_reload._hot_reload_service = mock_service
        
        result = hot_reload.watch_file("/test/path.json", callback)
        
        assert result is True
        mock_service.watch_file.assert_called_once_with("/test/path.json", callback)
        assert "/test/path.json" in hot_reload._watched_files
        mock_logger.info.assert_called()

    @patch('src.infrastructure.config.services.unified_hot_reload.logger')
    def test_watch_file_enabled_without_callback(self, mock_logger):
        """测试启用热重载时的文件监视（无回调）"""
        mock_service = Mock()
        mock_service.watch_file.return_value = True
        
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        hot_reload.enable_hot_reload = True
        hot_reload._hot_reload_service = mock_service
        
        result = hot_reload.watch_file("/test/path.json")
        
        assert result is True
        mock_service.watch_file.assert_called_once()
        # 验证默认回调被设置
        call_args = mock_service.watch_file.call_args[0]
        assert call_args[0] == "/test/path.json"
        assert callable(call_args[1])  # 应该是默认回调函数

    def test_unwatch_file_disabled(self):
        """测试禁用热重载时的取消文件监视"""
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        
        result = hot_reload.unwatch_file("/test/path.json")
        
        assert result is True

    @patch('src.infrastructure.config.services.unified_hot_reload.logger')
    def test_unwatch_file_enabled(self, mock_logger):
        """测试启用热重载时的取消文件监视"""
        mock_service = Mock()
        mock_service.unwatch_file.return_value = True
        
        hot_reload = UnifiedHotReload(enable_hot_reload=False)
        hot_reload.enable_hot_reload = True
        hot_reload._hot_reload_service = mock_service
        hot_reload._watched_files.add("/test/path.json")
        
        result = hot_reload.unwatch_file("/test/path.json")
        
        assert result is True
        mock_service.unwatch_file.assert_called_once_with("/test/path.json")
        assert "/test/path.json" not in hot_reload._watched_files
        mock_logger.info.assert_called()


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestGlobalFunctions:
    """测试全局函数"""

    def setup_method(self):
        """测试前准备"""
        # 重置全局实例
        import src.infrastructure.config.services.unified_hot_reload as module
        module._hot_reload_instance = None

    @patch('src.infrastructure.config.services.unified_hot_reload.UnifiedHotReload')
    def test_start_hot_reload_first_call(self, mock_hot_reload_class):
        """测试第一次调用start_hot_reload"""
        mock_instance = Mock()
        mock_instance.start_hot_reload.return_value = True
        mock_hot_reload_class.return_value = mock_instance
        
        result = start_hot_reload()
        
        assert result is True
        mock_hot_reload_class.assert_called_once_with(enable_hot_reload=True)
        mock_instance.start_hot_reload.assert_called_once()

    @patch('src.infrastructure.config.services.unified_hot_reload._hot_reload_instance')
    def test_start_hot_reload_subsequent_calls(self, mock_instance_attr):
        """测试后续调用start_hot_reload"""
        mock_instance = Mock()
        mock_instance.start_hot_reload.return_value = True
        mock_instance_attr = mock_instance
        
        import src.infrastructure.config.services.unified_hot_reload as module
        module._hot_reload_instance = mock_instance
        
        result = start_hot_reload()
        
        assert result is True
        mock_instance.start_hot_reload.assert_called_once()

    def test_stop_hot_reload_no_instance(self):
        """测试没有实例时调用stop_hot_reload"""
        import src.infrastructure.config.services.unified_hot_reload as module
        module._hot_reload_instance = None
        
        result = stop_hot_reload()
        
        assert result is True

    def test_stop_hot_reload_with_instance(self):
        """测试有实例时调用stop_hot_reload"""
        mock_instance = Mock()
        mock_instance.stop_hot_reload.return_value = True
        
        import src.infrastructure.config.services.unified_hot_reload as module
        module._hot_reload_instance = mock_instance
        
        result = stop_hot_reload()
        
        assert result is True
        mock_instance.stop_hot_reload.assert_called_once()
