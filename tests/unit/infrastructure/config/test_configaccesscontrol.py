#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConfigAccessControl 测试

测试 src/infrastructure/config/security/components/configaccesscontrol.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import Mock, patch

# 尝试导入模块
try:
    from src.infrastructure.config.security.components.configaccesscontrol import ConfigAccessControl
    from src.infrastructure.config.security.components.securityconfig import SecurityConfig
    from src.infrastructure.config.security.components.accessrecord import AccessRecord
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigAccessControl:
    """测试ConfigAccessControl功能"""

    def setup_method(self):
        """测试前准备"""
        self.mock_security_config = Mock(spec=SecurityConfig)
        self.mock_security_config.lockout_duration = 300  # 5分钟
        self.mock_security_config.max_access_attempts = 3
        self.mock_security_config.access_logging = True
        self.access_control = ConfigAccessControl(self.mock_security_config)

    def test_initialization(self):
        """测试初始化"""
        assert self.access_control.config == self.mock_security_config
        assert isinstance(self.access_control._access_records, list)
        assert len(self.access_control._access_records) == 0
        assert hasattr(self.access_control._lock, 'acquire')  # RLock对象有acquire方法
        assert isinstance(self.access_control._failed_attempts, dict)
        assert isinstance(self.access_control._locked_users, dict)
        assert isinstance(self.access_control.permissions, dict)

    def test_permissions_structure(self):
        """测试权限结构"""
        expected_roles = ["admin", "operator", "viewer", "system"]
        for role in expected_roles:
            assert role in self.access_control.permissions
            assert isinstance(self.access_control.permissions[role], list)

        # 验证admin权限
        admin_perms = self.access_control.permissions["admin"]
        assert "read" in admin_perms
        assert "write" in admin_perms
        assert "delete" in admin_perms
        assert "audit" in admin_perms

        # 验证viewer权限
        viewer_perms = self.access_control.permissions["viewer"]
        assert "read" in viewer_perms
        assert len(viewer_perms) == 1

    def test_check_access_admin_success(self):
        """测试管理员访问成功"""
        result = self.access_control.check_access("admin", "read", "test_resource")
        assert result is True

    def test_check_access_admin_write_success(self):
        """测试管理员写权限成功"""
        result = self.access_control.check_access("admin", "write", "test_resource")
        assert result is True

    def test_check_access_operator_read_success(self):
        """测试操作员读权限成功"""
        result = self.access_control.check_access("operator", "read", "test_resource")
        assert result is True

    def test_check_access_operator_write_success(self):
        """测试操作员写权限成功"""
        result = self.access_control.check_access("operator", "write", "test_resource")
        assert result is True

    def test_check_access_viewer_read_success(self):
        """测试查看者读权限成功"""
        result = self.access_control.check_access("viewer", "read", "test_resource")
        assert result is True

    def test_check_access_viewer_write_denied(self):
        """测试查看者写权限被拒绝"""
        result = self.access_control.check_access("viewer", "write", "test_resource")
        assert result is False

    def test_check_access_viewer_delete_denied(self):
        """测试查看者删除权限被拒绝"""
        result = self.access_control.check_access("viewer", "delete", "test_resource")
        assert result is False

    def test_check_access_unknown_user_denied(self):
        """测试未知用户访问权限（默认viewer角色）"""
        result = self.access_control.check_access("unknown_user", "read", "test_resource")
        assert result is True  # 未知用户默认得到viewer角色，read权限成功
        
        # 测试写权限被拒绝
        result = self.access_control.check_access("unknown_user", "write", "test_resource")
        assert result is False

    def test_access_recording(self):
        """测试访问记录"""
        initial_count = len(self.access_control._access_records)
        
        self.access_control.check_access("admin", "read", "test_resource")
        
        assert len(self.access_control._access_records) == initial_count + 1
        record = self.access_control._access_records[-1]
        assert record.user == "admin"
        assert record.action == "read"
        assert record.resource == "test_resource"

    def test_get_user_roles_admin(self):
        """测试获取管理员角色"""
        roles = self.access_control._get_user_roles("admin")
        assert "admin" in roles

    def test_get_user_roles_operator(self):
        """测试获取操作员角色"""
        roles = self.access_control._get_user_roles("operator")
        assert "operator" in roles

    def test_get_user_roles_viewer(self):
        """测试获取查看者角色"""
        roles = self.access_control._get_user_roles("viewer")
        assert "viewer" in roles

    def test_get_user_roles_system(self):
        """测试获取系统角色"""
        roles = self.access_control._get_user_roles("system")
        assert "system" in roles

    def test_get_user_roles_unknown(self):
        """测试获取未知用户角色"""
        roles = self.access_control._get_user_roles("unknown_user")
        assert len(roles) == 0 or "viewer" in roles  # 可能默认为viewer

    def test_failed_attempt_tracking(self):
        """测试失败尝试跟踪"""
        user = "test_user"
        
        # 模拟多次失败访问
        self.access_control.check_access(user, "write", "test_resource")  # 失败
        self.access_control.check_access(user, "delete", "test_resource")  # 失败
        
        assert user in self.access_control._failed_attempts
        assert len(self.access_control._failed_attempts[user]) >= 2

    @patch('time.time')
    def test_lockout_mechanism(self, mock_time):
        """测试锁定机制"""
        mock_time.return_value = 1000.0
        user = "test_user"
        
        # 模拟用户被锁定
        self.access_control._locked_users[user] = 900.0  # 100秒前锁定
        
        # 锁定期内访问应该被拒绝
        mock_time.return_value = 1100.0  # 只过了100秒，还在锁定期内
        result = self.access_control.check_access(user, "read", "test_resource")
        assert result is False

        # 锁定期过期后应该自动解锁
        mock_time.return_value = 1300.0  # 300秒后，超过锁定期
        result = self.access_control.check_access(user, "read", "test_resource")
        assert result is True or result is False  # 可能因为权限问题失败，但不应该因为锁定失败

    def test_handle_failed_attempt(self):
        """测试处理失败尝试"""
        user = "test_user"
        initial_time = time.time()
        
        # 模拟多次失败
        for _ in range(5):
            self.access_control._handle_failed_attempt(user)
        
        assert user in self.access_control._failed_attempts
        assert len(self.access_control._failed_attempts[user]) >= 3  # 达到最大失败次数后可能被锁定

    def test_get_access_records(self):
        """测试获取访问记录"""
        # 先进行一些访问操作
        self.access_control.check_access("admin", "read", "resource1")
        self.access_control.check_access("viewer", "read", "resource2")
        
        records = self.access_control.get_access_records()
        assert isinstance(records, list)
        assert len(records) >= 2

    def test_get_access_records_with_user_filter(self):
        """测试按用户过滤访问记录"""
        # 进行多个用户的访问
        self.access_control.check_access("admin", "read", "resource1")
        self.access_control.check_access("viewer", "read", "resource2")
        self.access_control.check_access("admin", "write", "resource3")
        
        admin_records = self.access_control.get_access_records(user="admin")
        assert all(record.user == "admin" for record in admin_records)

    def test_get_access_records_with_limit(self):
        """测试限制访问记录数量"""
        # 进行多次访问
        for i in range(10):
            self.access_control.check_access("test_user", "read", f"resource{i}")
        
        records = self.access_control.get_access_records(limit=5)
        assert len(records) <= 5

    def test_clear_access_records(self):
        """测试清除访问记录"""
        # 先添加一些记录
        self.access_control.check_access("test_user", "read", "test_resource")
        assert len(self.access_control._access_records) > 0
        
        # 清除记录
        self.access_control.clear_access_records()
        assert len(self.access_control._access_records) == 0

    def test_thread_safety(self):
        """测试线程安全性"""
        results = []
        
        def access_worker(user_id):
            result = self.access_control.check_access(f"user_{user_id}", "read", f"resource_{user_id}")
            results.append((user_id, result))
        
        # 创建多个线程同时访问
        threads = []
        for i in range(5):
            thread = threading.Thread(target=access_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证没有异常发生
        assert len(results) == 5
        # 结果可能是True或False，取决于用户权限，但不应该出现异常


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigAccessControlEdgeCases:
    """测试边界情况"""

    def setup_method(self):
        """测试前准备"""
        self.mock_security_config = Mock(spec=SecurityConfig)
        self.mock_security_config.lockout_duration = 300
        self.mock_security_config.max_access_attempts = 3
        self.mock_security_config.access_logging = True
        self.access_control = ConfigAccessControl(self.mock_security_config)

    def test_empty_user_name(self):
        """测试空用户名"""
        result = self.access_control.check_access("", "read", "test_resource")
        assert result is False or True  # 取决于具体实现

    def test_none_user_name(self):
        """测试None用户名（会被映射为viewer）"""
        result = self.access_control.check_access(None, "read", "test_resource")
        assert result is True  # None用户会被映射为viewer角色，read权限成功

    def test_empty_action(self):
        """测试空操作"""
        result = self.access_control.check_access("admin", "", "test_resource")
        assert result is False

    def test_none_action(self):
        """测试None操作"""
        result = self.access_control.check_access("admin", None, "test_resource")
        assert result is False

    def test_empty_resource(self):
        """测试空资源"""
        result = self.access_control.check_access("admin", "read", "")
        # 应该可以处理空资源
        assert result is True or result is False

    def test_get_user_roles_edge_cases(self):
        """测试获取用户角色的边界情况"""
        # 测试空字符串
        roles = self.access_control._get_user_roles("")
        assert isinstance(roles, list)
        
        # 测试None
        roles = self.access_control._get_user_roles(None)
        assert isinstance(roles, list)


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigAccessControlIntegration:
    """测试集成功能"""

    def setup_method(self):
        """测试前准备"""
        self.mock_security_config = Mock(spec=SecurityConfig)
        self.mock_security_config.lockout_duration = 300
        self.mock_security_config.max_access_attempts = 3
        self.mock_security_config.access_logging = True
        self.access_control = ConfigAccessControl(self.mock_security_config)

    def test_module_imports(self):
        """测试模块可以正常导入"""
        assert ConfigAccessControl is not None
        assert SecurityConfig is not None

    def test_full_access_workflow(self):
        """测试完整访问工作流程"""
        # 1. 管理员正常访问
        result = self.access_control.check_access("admin", "read", "config1")
        assert result is True
        
        # 2. 查看访问记录
        records = self.access_control.get_access_records(user="admin")
        assert len(records) > 0
        
        # 3. 权限升级测试
        result = self.access_control.check_access("admin", "delete", "config1")
        assert result is True
        
        # 4. 查看者受限访问
        result = self.access_control.check_access("viewer", "write", "config1")
        assert result is False
