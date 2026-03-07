#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConfigAuditManager 测试

测试 src/infrastructure/config/security/components/configauditmanager.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

# 尝试导入模块，如果失败则跳过测试
try:
    from src.infrastructure.config.security.components.configauditmanager import ConfigAuditManager
    from src.infrastructure.config.security.components.configauditlog import ConfigAuditLog
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigAuditManager:
    """测试ConfigAuditManager功能"""

    def setup_method(self):
        """测试前准备"""
        self.manager = ConfigAuditManager()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.manager, 'audit_logs')
        assert hasattr(self.manager, '_lock')
        assert hasattr(self.manager, '_callbacks')
        assert isinstance(self.manager.audit_logs, list)
        assert isinstance(self.manager._callbacks, list)

    @patch('src.infrastructure.config.security.components.configauditmanager.time')
    @patch('src.infrastructure.config.security.components.configauditmanager.logger')
    def test_log_change_basic(self, mock_logger, mock_time):
        """测试基本的配置变更日志记录"""
        mock_time.time.return_value = 1234567890.0
        
        self.manager.log_change(
            action="SET",
            key="test.key",
            old_value="old_value",
            new_value="new_value",
            user="test_user",
            reason="test reason"
        )
        
        assert len(self.manager.audit_logs) == 1
        audit_log = self.manager.audit_logs[0]
        assert audit_log.action == "SET"
        assert audit_log.key == "test.key"
        assert audit_log.old_value == "old_value"
        assert audit_log.new_value == "new_value"
        assert audit_log.user == "test_user"
        assert audit_log.reason == "test reason"
        
        # 验证日志记录
        mock_logger.info.assert_called()

    @patch('src.infrastructure.config.security.components.configauditmanager.time')
    @patch('src.infrastructure.config.security.components.configauditmanager.logger')
    def test_log_change_with_defaults(self, mock_logger, mock_time):
        """测试使用默认参数的变更记录"""
        mock_time.time.return_value = 1234567890.0
        
        self.manager.log_change(action="GET", key="test.key")
        
        assert len(self.manager.audit_logs) == 1
        audit_log = self.manager.audit_logs[0]
        assert audit_log.action == "GET"
        assert audit_log.key == "test.key"
        assert audit_log.user == "system"  # 默认值
        assert audit_log.reason == ""  # 默认值

    @patch('src.infrastructure.config.security.components.configauditmanager.time')
    @patch('src.infrastructure.config.security.components.configauditmanager.logger')
    def test_log_change_with_callback(self, mock_logger, mock_time):
        """测试带回调的变更记录"""
        mock_time.time.return_value = 1234567890.0
        callback_mock = StandardMockBuilder.create_config_mock()
        
        self.manager.add_callback(callback_mock)
        self.manager.log_change(action="SET", key="test.key", new_value="new_value")
        
        # 验证回调被调用
        callback_mock.assert_called_once()
        callback_arg = callback_mock.call_args[0][0]
        assert isinstance(callback_arg, ConfigAuditLog)
        assert callback_arg.action == "SET"

    @patch('src.infrastructure.config.security.components.configauditmanager.time')
    @patch('src.infrastructure.config.security.components.configauditmanager.logger')
    def test_callback_exception_handling(self, mock_logger, mock_time):
        """测试回调异常处理"""
        mock_time.time.return_value = 1234567890.0
        error_callback = Mock(side_effect=Exception("Callback error"))
        
        self.manager.add_callback(error_callback)
        self.manager.log_change(action="SET", key="test.key")
        
        # 验证错误被记录
        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args[0][0]
        assert "审计回调执行失败" in error_call

    @patch('src.infrastructure.config.security.components.configauditmanager.time')
    def test_log_size_limit(self, mock_time):
        """测试日志大小限制"""
        mock_time.time.return_value = 1234567890.0
        
        # 添加超过限制的日志条数（模拟5001条）
        for i in range(5001):
            self.manager.log_change(action="SET", key=f"key.{i}")
        
        # 验证日志被截断到2500条
        assert len(self.manager.audit_logs) == 2500

    def test_get_audit_logs_no_filters(self):
        """测试获取审计日志（无过滤条件）"""
        # 添加一些测试日志
        with patch('src.infrastructure.config.security.components.configauditmanager.time') as mock_time:
            mock_time.time.return_value = 1234567890.0
            
            self.manager.log_change(action="SET", key="key1", user="user1")
            self.manager.log_change(action="GET", key="key2", user="user2")
            
            logs = self.manager.get_audit_logs()
            assert len(logs) == 2

    def test_get_audit_logs_with_key_filter(self):
        """测试按key过滤审计日志"""
        with patch('src.infrastructure.config.security.components.configauditmanager.time') as mock_time:
            mock_time.time.return_value = 1234567890.0
            
            self.manager.log_change(action="SET", key="test.key", user="user1")
            self.manager.log_change(action="GET", key="other.key", user="user2")
            
            logs = self.manager.get_audit_logs(key="test.key")
            assert len(logs) == 1
            assert logs[0].key == "test.key"

    def test_get_audit_logs_with_user_filter(self):
        """测试按用户过滤审计日志"""
        with patch('src.infrastructure.config.security.components.configauditmanager.time') as mock_time:
            mock_time.time.return_value = 1234567890.0
            
            self.manager.log_change(action="SET", key="key1", user="user1")
            self.manager.log_change(action="GET", key="key2", user="user2")
            
            logs = self.manager.get_audit_logs(user="user1")
            assert len(logs) == 1
            assert logs[0].user == "user1"

    def test_get_audit_logs_with_limit(self):
        """测试审计日志数量限制"""
        with patch('src.infrastructure.config.security.components.configauditmanager.time') as mock_time:
            mock_time.time.return_value = 1234567890.0
            
            # 添加5条日志
            for i in range(5):
                self.manager.log_change(action="SET", key=f"key.{i}")
            
            # 限制返回3条
            logs = self.manager.get_audit_logs(limit=3)
            assert len(logs) == 3

    def test_get_change_history(self):
        """测试获取配置项变更历史"""
        with patch('src.infrastructure.config.security.components.configauditmanager.time') as mock_time:
            mock_time.time.return_value = 1234567890.0
            
            # 为同一个key添加多条记录
            self.manager.log_change(action="SET", key="test.key", new_value="value1")
            self.manager.log_change(action="UPDATE", key="test.key", old_value="value1", new_value="value2")
            self.manager.log_change(action="DELETE", key="other.key")
            
            history = self.manager.get_change_history("test.key")
            assert len(history) == 2
            assert history[0].action == "SET"
            assert history[1].action == "UPDATE"

    def test_add_callback(self):
        """测试添加回调函数"""
        callback1 = Mock()
        callback2 = Mock()
        
        self.manager.add_callback(callback1)
        self.manager.add_callback(callback2)
        
        assert len(self.manager._callbacks) == 2
        assert callback1 in self.manager._callbacks
        assert callback2 in self.manager._callbacks

    def test_thread_safety(self):
        """测试线程安全性"""
        results = []
        
        def add_logs():
            with patch('src.infrastructure.config.security.components.configauditmanager.time') as mock_time:
                mock_time.time.return_value = 1234567890.0
                for i in range(10):
                    self.manager.log_change(action="SET", key=f"thread.key.{i}")
            results.append(len(self.manager.audit_logs))
        
        # 创建多个线程同时添加日志
        threads = [threading.Thread(target=add_logs) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # 验证最终结果的一致性
        assert len(self.manager.audit_logs) == 30  # 3个线程，每个10条


class TestConfigAuditManagerIntegration:
    """测试ConfigAuditManager集成功能"""

    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="模块不可用")
    def test_module_imports(self):
        """测试模块可以正常导入"""
        try:
            from src.infrastructure.config.security.components.configauditmanager import ConfigAuditManager
            from src.infrastructure.config.security.components.configauditlog import ConfigAuditLog
            assert True  # 导入成功
        except ImportError as e:
            pytest.fail(f"模块导入失败: {e}")

    @pytest.mark.skipif(not MODULE_AVAILABLE, reason="模块不可用")
    def test_audit_log_creation_integration(self):
        """测试审计日志创建的集成功能"""
        manager = ConfigAuditManager()
        
        with patch('src.infrastructure.config.security.components.configauditmanager.time') as mock_time:
            mock_time.time.return_value = 1234567890.0
            
            # 测试各种操作类型
            operations = ["SET", "GET", "UPDATE", "DELETE"]
            for op in operations:
                manager.log_change(action=op, key=f"test.{op.lower()}", new_value=f"value_{op}")
            
            assert len(manager.audit_logs) == len(operations)
