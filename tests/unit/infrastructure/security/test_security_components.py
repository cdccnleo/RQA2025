#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 安全组件测试

测试重构后的安全模块组件功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import asyncio
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# 导入被测试的组件
from src.infrastructure.security.auth.role_manager import RoleManager
from src.infrastructure.security.auth.session_manager import SessionManager
from src.infrastructure.security.audit.audit_manager import AuditManager
from src.infrastructure.security.monitoring.performance_monitor import PerformanceMonitor
from src.infrastructure.security.access.components.access_checker import AccessChecker, AccessDecision, AccessRequest
from src.infrastructure.security.access.components.audit_logger import AuditLogger
from src.infrastructure.security.access.components.config_manager import ConfigManager
from src.infrastructure.security.core.types import UserRole, Permission, AuditEventParams
from src.infrastructure.security.audit.audit_events import AuditEventType, AuditSeverity


class TestRoleManager(unittest.TestCase):
    """角色管理器测试"""

    def setUp(self):
        """测试前准备"""
        self.role_manager = RoleManager()

    def test_create_role(self):
        """测试创建角色"""
        role = self.role_manager.create_role(
            role_id="test_role",
            name="测试角色",
            permissions={"read", "write"}
        )

        self.assertEqual(role.role_id, "test_role")
        self.assertEqual(role.name, "测试角色")
        self.assertEqual(role.permissions, {"read", "write"})

    def test_get_role(self):
        """测试获取角色"""
        role = self.role_manager.create_role("test_role", "测试角色")
        retrieved = self.role_manager.get_role("test_role")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.role_id, "test_role")

    def test_create_role_from_template(self):
        """测试从模板创建角色"""
        role = self.role_manager.create_role_from_template(UserRole.ADMIN)

        self.assertIsNotNone(role)
        self.assertEqual(role.name, "管理员")
        self.assertTrue(len(role.permissions) > 0)  # 管理员应该有权限

    def test_role_permissions_inheritance(self):
        """测试角色权限继承"""
        # 创建父角色
        parent_role = self.role_manager.create_role(
            "parent", "父角色", permissions={"read"}
        )

        # 创建子角色
        child_role = self.role_manager.create_role(
            "child", "子角色", permissions={"write"}, parent_roles={"parent"}
        )

        # 获取子角色的所有权限
        all_permissions = child_role.get_all_permissions(self.role_manager.roles)

        self.assertIn("read", all_permissions)  # 继承的权限
        self.assertIn("write", all_permissions)  # 自己的权限


class TestAccessChecker(unittest.TestCase):
    """访问检查器测试"""

    def setUp(self):
        """测试前准备"""
        self.checker = AccessChecker()  # 重构后的访问检查器

    def test_allow_direct_permission(self):
        """测试直接权限允许"""
        request = AccessRequest(
            user_id="user1",
            resource="/api/data",
            action="read"
        )

        result = self.checker.check_access(request, {"read", "write"})

        self.assertEqual(result.decision, AccessDecision.ALLOW)
        self.assertIn("直接权限", result.reason)

    def test_deny_no_permission(self):
        """测试无权限拒绝"""
        request = AccessRequest(
            user_id="user1",
            resource="/api/admin",
            action="delete"
        )

        result = self.checker.check_access(request, {"read"})

        self.assertEqual(result.decision, AccessDecision.DENY)
        self.assertIn("无匹配权限", result.reason)

    def test_risk_score_calculation(self):
        """测试风险分数计算"""
        request = AccessRequest(
            user_id="user1",
            resource="/api/admin/config",
            action="delete"
        )

        result = self.checker.check_access(request, {"read"})

        # 删除操作和admin资源应该有较高风险分数
        self.assertGreater(result.risk_score, 0.5)

    async def test_async_check_access(self):
        """测试异步权限检查"""
        request = AccessRequest(
            user_id="user1",
            resource="/api/data",
            action="read"
        )

        result = await self.checker.check_access_async(request, {"read"})

        self.assertEqual(result.decision, AccessDecision.ALLOW)

    async def test_batch_async_check(self):
        """测试异步批量检查"""
        requests = [
            AccessRequest(user_id="user1", resource="/api/data", action="read"),
            AccessRequest(user_id="user1", resource="/api/data", action="write"),
        ]

        user_permissions = {"user1": {"read", "write"}}

        results = await self.checker.batch_check_access_async(
            requests, user_permissions, max_concurrency=2
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.decision == AccessDecision.ALLOW for r in results))


class TestSessionManager(unittest.TestCase):
    """会话管理器测试"""

    def setUp(self):
        """测试前准备"""
        self.session_mgr = SessionManager(session_timeout=60, max_sessions_per_user=2)

    def test_create_session(self):
        """测试创建会话"""
        session_id = self.session_mgr.create_session("user1", ip_address="127.0.0.1")

        self.assertIsNotNone(session_id)
        self.assertTrue(len(session_id) > 0)

    def test_get_session(self):
        """测试获取会话"""
        session_id = self.session_mgr.create_session("user1")
        session = self.session_mgr.get_session(session_id)

        self.assertIsNotNone(session)
        self.assertEqual(session.user_id, "user1")
        self.assertFalse(session.is_expired())

    def test_session_expiry(self):
        """测试会话过期"""
        # 创建一个会话并手动设置过期时间
        session_id = self.session_mgr.create_session("user1")
        session = self.session_mgr._sessions[session_id]
        session.expires_at = datetime.now() - timedelta(seconds=1)  # 已过期

        retrieved = self.session_mgr.get_session(session_id)
        self.assertIsNone(retrieved)  # 应该返回None

    def test_max_sessions_per_user(self):
        """测试每个用户的最大会话数"""
        # 创建两个会话
        session1 = self.session_mgr.create_session("user1")
        session2 = self.session_mgr.create_session("user1")

        # 第三个会话应该替换第一个
        session3 = self.session_mgr.create_session("user1")

        user_sessions = self.session_mgr._user_sessions["user1"]
        self.assertEqual(len(user_sessions), 2)  # 最多2个会话
        self.assertNotIn(session1, user_sessions)  # 第一个被替换了

    def test_invalidate_session(self):
        """测试使会话失效"""
        session_id = self.session_mgr.create_session("user1")
        success = self.session_mgr.invalidate_session(session_id)

        self.assertTrue(success)
        self.assertIsNone(self.session_mgr.get_session(session_id))


class TestAuditManager(unittest.TestCase):
    """审计管理器测试"""

    def setUp(self):
        """测试前准备"""
        self.audit_manager = AuditManager()

    def test_log_event(self):
        """测试记录审计事件"""
        from src.infrastructure.security.core.types import AuditEventParams

        params = AuditEventParams(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user1",
            session_id="session_123",
            resource="/api/admin",
            action="login",
            result="success",
            details={"method": "POST"},
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
            location="Local",
            risk_score=0.1,
            tags={"test", "login"}
        )

        event_id = self.audit_manager.log_event(params)

        self.assertIsNotNone(event_id)
        self.assertTrue(len(event_id) > 0)

    def test_query_events(self):
        """测试查询事件"""
        from src.infrastructure.security.core.types import QueryFilterParams

        # 创建一些测试事件
        params1 = AuditEventParams(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(),
            user_id="user1",
            action="login",
            result="success"
        )
        params2 = AuditEventParams(
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.MEDIUM,
            timestamp=datetime.now(),
            user_id="user2",
            action="read",
            result="success"
        )

        self.audit_manager.log_event(params1)
        self.audit_manager.log_event(params2)

        # 查询所有事件
        filter_params = QueryFilterParams(limit=10)
        events = self.audit_manager.query_events(filter_params)

        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0]['user_id'], 'user1')
        self.assertEqual(events[1]['user_id'], 'user2')

    def test_audit_statistics(self):
        """测试审计统计"""
        from src.infrastructure.security.audit.audit_manager import AuditManager

        # 创建一个临时审计管理器用于测试
        audit_mgr = AuditManager()

        # 这里可以添加更多的统计测试
        # 由于统计功能需要时间序列数据，我们这里只测试基本功能
        self.assertIsNotNone(audit_mgr)


class TestPerformanceMonitor(unittest.TestCase):
    """性能监控器测试"""

    def setUp(self):
        """测试前准备"""
        self.monitor = PerformanceMonitor(enabled=True)

    def test_record_operation(self):
        """测试记录操作"""
        self.monitor.record_operation("test_operation", 0.1, is_error=False)

        metrics = self.monitor.get_metrics("test_operation")
        self.assertEqual(metrics['total_calls'], 1)
        self.assertAlmostEqual(metrics['avg_time'], 0.1, places=3)

    def test_multiple_recordings(self):
        """测试多次记录"""
        operations = ["op1", "op2", "op1"]  # op1被调用两次
        times = [0.1, 0.2, 0.15]

        for op, time_taken in zip(operations, times):
            self.monitor.record_operation(op, time_taken)

        # 检查op1的统计
        op1_metrics = self.monitor.get_metrics("op1")
        self.assertEqual(op1_metrics['total_calls'], 2)
        self.assertAlmostEqual(op1_metrics['avg_time'], 0.125, places=3)

    def test_error_recording(self):
        """测试错误记录"""
        self.monitor.record_operation("error_op", 0.5, is_error=True)

        metrics = self.monitor.get_metrics("error_op")
        self.assertEqual(metrics['error_count'], 1)
        self.assertEqual(metrics['error_rate'], 100.0)

    def test_performance_report(self):
        """测试性能报告"""
        # 记录一些操作
        self.monitor.record_operation("fast_op", 0.01)
        self.monitor.record_operation("slow_op", 2.0)
        self.monitor.record_operation("error_op", 1.0, is_error=True)

        report = self.monitor.get_performance_report()

        self.assertIn('metrics', report)
        self.assertIn('system_stats', report)
        self.assertIn('bottlenecks', report)
        self.assertIn('recommendations', report)
        self.assertIn('summary', report)

        # 检查瓶颈检测
        bottlenecks = report['bottlenecks']
        slow_ops = [b for b in bottlenecks if 'slow_op' in b['operation']]
        self.assertTrue(len(slow_ops) > 0)


class TestAuditLogger(unittest.TestCase):
    """审计日志器测试"""

    def setUp(self):
        """测试前准备"""
        self.logger = AuditLogger()

    def test_log_audit_event(self):
        """测试记录审计事件"""
        from src.infrastructure.security.core.types import AuditEvent

        event = AuditEvent(
            event_id="test_123",
            timestamp=datetime.now(),
            user_id="user1",
            action="login",
            resource="/api/auth",
            permission="login",
            decision=AccessDecision.ALLOW,
            details={"method": "POST"}
        )

        self.logger.log_audit_event(event)
        # 验证事件被添加到队列中
        self.assertIn(event, self.logger.event_queue)

    def test_get_audit_statistics(self):
        """测试获取审计统计"""
        stats = self.logger.get_audit_statistics(days=1)

        # 验证统计结果结构
        self.assertIn('total_events', stats)
        self.assertIn('time_range', stats)
        self.assertIn('events_by_action', stats)
        self.assertIn('events_by_user', stats)
        self.assertIn('security_events', stats)

    def test_query_audit_logs(self):
        """测试查询审计日志"""
        events = self.logger.query_audit_logs(limit=10)
        self.assertIsInstance(events, list)


class TestConfigManager(unittest.TestCase):
    """配置管理器测试"""

    def setUp(self):
        """测试前准备"""
        self.config_mgr = ConfigManager()

    def test_get_config(self):
        """测试获取配置"""
        cache_config = self.config_mgr.get_config('cache')
        self.assertIsNotNone(cache_config)
        self.assertIn('enabled', cache_config)
        self.assertIn('max_size', cache_config)

    def test_set_config(self):
        """测试设置配置"""
        success = self.config_mgr.set_config('test.key', 'test_value')
        self.assertTrue(success)

        value = self.config_mgr.get_config('test.key')
        self.assertEqual(value, 'test_value')

    def test_config_validation(self):
        """测试配置验证"""
        # 有效的配置
        valid_config = {'cache': {'enabled': True, 'max_size': 100}}
        is_valid = self.config_mgr.validate_config(valid_config)
        self.assertTrue(is_valid)

        # 无效的配置
        invalid_config = {'cache': {'enabled': 'not_boolean'}}
        is_valid = self.config_mgr.validate_config(invalid_config)
        self.assertFalse(is_valid)

    def test_save_load_config(self):
        """测试配置保存和加载"""
        # 设置一些配置
        self.config_mgr.set_config('test.save', 'saved_value')

        # 保存配置（如果有持久化功能）
        # 这里主要测试配置管理的核心功能
        value = self.config_mgr.get_config('test.save')
        self.assertEqual(value, 'saved_value')


class TestNewSecurityComponents(unittest.TestCase):
    """新安全组件集成测试"""

    def test_component_integration(self):
        """测试组件集成"""
        # 创建各个组件实例
        role_mgr = RoleManager()
        session_mgr = SessionManager()
        audit_mgr = AuditManager()
        access_checker = AccessChecker()
        audit_logger = AuditLogger()
        config_mgr = ConfigManager()
        perf_monitor = PerformanceMonitor()

        # 验证所有组件都能正常创建
        self.assertIsNotNone(role_mgr)
        self.assertIsNotNone(session_mgr)
        self.assertIsNotNone(audit_mgr)
        self.assertIsNotNone(access_checker)
        self.assertIsNotNone(audit_logger)
        self.assertIsNotNone(config_mgr)
        self.assertIsNotNone(perf_monitor)

    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 1. 创建角色
        role_mgr = RoleManager()
        role = role_mgr.create_role(
            role_id="test_trader",
            name="测试交易员",
            permissions={"trade:execute", "data:read"}
        )
        self.assertIsNotNone(role)

        # 2. 创建会话
        session_mgr = SessionManager()
        session_id = session_mgr.create_session("test_user")
        self.assertIsNotNone(session_id)

        # 3. 记录审计事件
        audit_mgr = AuditManager()
        params = AuditEventParams(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.MEDIUM,
            timestamp=datetime.now(),
            user_id="test_user",
            action="login",
            result="success"
        )
        event_id = audit_mgr.log_event(params)
        self.assertIsNotNone(event_id)

        # 4. 权限检查
        access_checker = AccessChecker()
        request = AccessRequest(
            user_id="test_user",
            resource="/api/trade",
            permission="trade:execute"
        )
        result = access_checker.check_access(request, {"trade:execute"})
        self.assertEqual(result.decision, AccessDecision.ALLOW)


# 异步测试运行器
def run_async_tests():
    """运行异步测试"""
    async def run_tests():
        # 创建测试实例
        checker = AccessChecker()

        # 测试异步权限检查
        request = AccessRequest(user_id="test", resource="/test", action="read")
        result = await checker.check_access_async(request, {"read"})
        assert result.decision == AccessDecision.ALLOW

        print("✅ 异步测试通过")

    # 运行异步测试
    asyncio.run(run_tests())


if __name__ == '__main__':
    # 运行同步测试
    unittest.main()

    # 运行异步测试
    run_async_tests()
