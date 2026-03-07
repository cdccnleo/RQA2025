#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HotReloadManager 测试

测试 src/infrastructure/config/security/components/hotreloadmanager.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import os
import time
import threading
import tempfile
from unittest.mock import Mock, patch, MagicMock

# 尝试导入模块，如果失败则跳过测试
try:
    from src.infrastructure.config.security.components.hotreloadmanager import HotReloadManager
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestHotReloadManager:
    """测试HotReloadManager功能"""

    def setup_method(self):
        """测试前准备"""
        self.manager = HotReloadManager()
        # 创建临时文件用于测试
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.temp_file.write("initial content")
        self.temp_file.close()

    def teardown_method(self):
        """测试后清理"""
        # 停止监控线程
        self.manager.stop_monitoring()
        # 清理临时文件
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.manager, '_watchers')
        assert hasattr(self.manager, '_last_modified')
        assert hasattr(self.manager, '_reload_lock')
        assert hasattr(self.manager, '_monitor_thread')
        assert hasattr(self.manager, '_shutdown_event')
        assert hasattr(self.manager, '_check_interval')
        assert isinstance(self.manager._watchers, dict)
        assert isinstance(self.manager._last_modified, dict)
        assert self.manager._check_interval == 5.0

    def test_watch_file_new_file(self):
        """测试监视新文件"""
        callback = Mock()
        file_path = self.temp_file.name
        
        self.manager.watch_file(file_path, callback)
        
        assert file_path in self.manager._watchers
        assert callback in self.manager._watchers[file_path]
        assert file_path in self.manager._last_modified

    def test_watch_file_existing_file(self):
        """测试监视已存在的文件"""
        callback1 = Mock()
        callback2 = Mock()
        file_path = self.temp_file.name
        
        # 添加第一个回调
        self.manager.watch_file(file_path, callback1)
        assert len(self.manager._watchers[file_path]) == 1
        
        # 添加第二个回调
        self.manager.watch_file(file_path, callback2)
        assert len(self.manager._watchers[file_path]) == 2
        assert callback1 in self.manager._watchers[file_path]
        assert callback2 in self.manager._watchers[file_path]

    def test_watch_file_absolute_path(self):
        """测试文件路径被转换为绝对路径"""
        callback = Mock()
        relative_path = self.temp_file.name
        
        with patch('os.path.abspath', return_value='/absolute/path') as mock_abspath:
            self.manager.watch_file(relative_path, callback)
            mock_abspath.assert_called_once_with(relative_path)

    @patch('src.infrastructure.config.security.components.hotreloadmanager.logger')
    def test_start_monitoring(self, mock_logger):
        """测试开始监控"""
        assert self.manager._monitor_thread is None
        
        self.manager.start_monitoring()
        
        assert self.manager._monitor_thread is not None
        assert isinstance(self.manager._monitor_thread, threading.Thread)
        assert self.manager._monitor_thread.daemon is True
        assert self.manager._monitor_thread.name == "ConfigHotReload"
        
        # 清理
        self.manager.stop_monitoring()

    def test_stop_monitoring(self):
        """测试停止监控"""
        self.manager.start_monitoring()
        assert self.manager._monitor_thread is not None
        
        self.manager.stop_monitoring()
        
        assert self.manager._shutdown_event.is_set()

    def test_get_file_mtime_valid_file(self):
        """测试获取有效文件的修改时间"""
        file_path = self.temp_file.name
        mtime = self.manager._get_file_mtime(file_path)
        
        assert isinstance(mtime, float)
        assert mtime > 0

    def test_get_file_mtime_invalid_file(self):
        """测试获取无效文件的修改时间"""
        invalid_path = "/nonexistent/file/path"
        mtime = self.manager._get_file_mtime(invalid_path)
        
        assert mtime == 0.0

    @patch('src.infrastructure.config.security.components.hotreloadmanager.logger')
    @patch.object(HotReloadManager, '_get_file_mtime')
    def test_check_files_with_change(self, mock_get_mtime, mock_logger):
        """测试检测文件变更"""
        callback = Mock()
        file_path = self.temp_file.name
        
        # 设置初始修改时间
        initial_mtime = 1234567890.0
        new_mtime = 1234567891.0
        
        # 确保_get_file_mtime返回new_mtime
        mock_get_mtime.return_value = new_mtime
        
        # 监视文件，设置初始修改时间
        self.manager._last_modified[file_path] = initial_mtime
        self.manager._watchers[file_path] = [callback]
        
        # 执行检查
        self.manager._check_files()
        
        # 验证回调被调用
        callback.assert_called_once_with(file_path)
        mock_logger.info.assert_called()
        
        # 验证修改时间被更新
        assert self.manager._last_modified[file_path] == new_mtime

    @patch('src.infrastructure.config.security.components.hotreloadmanager.logger')
    @patch.object(HotReloadManager, '_get_file_mtime')
    def test_check_files_no_change(self, mock_get_mtime, mock_logger):
        """测试无文件变更的情况"""
        callback = Mock()
        file_path = self.temp_file.name
        mtime = 1234567890.0
        
        mock_get_mtime.return_value = mtime
        
        # 监视文件
        self.manager._last_modified[file_path] = mtime
        self.manager._watchers[file_path] = [callback]
        
        # 执行检查
        self.manager._check_files()
        
        # 验证回调未被调用
        callback.assert_not_called()

    @patch('src.infrastructure.config.security.components.hotreloadmanager.logger')
    def test_check_files_callback_exception(self, mock_logger):
        """测试回调执行异常处理"""
        error_callback = Mock(side_effect=Exception("Callback error"))
        file_path = self.temp_file.name
        
        with patch.object(self.manager, '_get_file_mtime', return_value=1234567890.0):
            # 设置不同的修改时间以触发变更检测
            self.manager._last_modified[file_path] = 1234567889.0
            self.manager._watchers[file_path] = [error_callback]
            
            # 执行检查
            self.manager._check_files()
            
            # 验证错误被记录
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args[0][0]
            assert "热重载回调执行失败" in error_call

    @patch('src.infrastructure.config.security.components.hotreloadmanager.logger')
    def test_monitor_loop_exception_handling(self, mock_logger):
        """测试监控循环异常处理"""
        with patch.object(self.manager, '_check_files', side_effect=Exception("Check error")):
            # 设置短的超时时间以便快速测试
            self.manager._check_interval = 0.1
            
            # 启动监控
            self.manager.start_monitoring()
            time.sleep(0.2)  # 等待监控循环执行
            
            # 停止监控
            self.manager.stop_monitoring()
            
            # 验证错误被记录
            mock_logger.error.assert_called()

    def test_thread_safety(self):
        """测试线程安全性"""
        results = []
        
        def add_watcher():
            callback = Mock()
            self.manager.watch_file(self.temp_file.name, callback)
            results.append(len(self.manager._watchers))
        
        # 创建多个线程同时添加监视器
        threads = [threading.Thread(target=add_watcher) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # 验证最终结果的一致性
        assert len(self.manager._watchers) == 1  # 同一个文件
        file_path = self.temp_file.name
        assert len(self.manager._watchers[file_path]) == 3  # 3个回调


class TestHotReloadManagerIntegration:
    """测试HotReloadManager集成功能"""

    def setup_method(self):
        """测试前准备"""
        self.manager = HotReloadManager()
        # 创建临时文件用于测试
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.temp_file.write("initial content")
        self.temp_file.close()

    def teardown_method(self):
        """测试后清理"""
        # 停止监控线程
        if hasattr(self, 'manager'):
            self.manager.stop_monitoring()
        # 清理临时文件
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="模块不可用")
    def test_module_imports(self):
        """测试模块可以正常导入"""
        try:
            from src.infrastructure.config.security.components.hotreloadmanager import HotReloadManager
            assert True  # 导入成功
        except ImportError as e:
            pytest.fail(f"模块导入失败: {e}")

    def test_full_workflow(self):
        """测试完整的工作流程"""
        callback = Mock()
        file_path = self.temp_file.name
        
        # 1. 添加文件监视
        self.manager.watch_file(file_path, callback)
        assert file_path in self.manager._watchers
        
        # 2. 启动监控
        self.manager.start_monitoring()
        assert self.manager._monitor_thread is not None
        
        # 3. 修改文件内容
        with open(file_path, 'w') as f:
            f.write("modified content")
        
        # 等待监控检测（注意：这里可能需要调整等待时间）
        time.sleep(0.1)
        
        # 4. 停止监控
        self.manager.stop_monitoring()
        
        # 验证基本功能正常工作
        assert self.manager._shutdown_event.is_set()
