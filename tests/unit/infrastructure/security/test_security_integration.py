#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 安全模块集成测试

测试各个安全组件间的协作和完整业务流程
包括用户认证、访问控制、审计日志等完整场景
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from typing import Dict, List, Any

from src.infrastructure.security.auth.user_manager import UserManager
from src.infrastructure.security.auth.role_manager import RoleManager
from src.infrastructure.security.access.components.access_checker import AccessChecker, AccessDecision
from src.infrastructure.security.audit.audit_manager import AuditManager
from src.infrastructure.security.access.components.config_manager import ConfigManager
from src.infrastructure.security.core.types import (
    User, UserRole, Permission, AuditEventParams, EventType, EventSeverity,
    UserCreationParams, QueryFilterParams
)


class MockPolicyManager:
    """模拟策略管理器"""

    def __init__(self):
        self.decisions = {}

    def evaluate_policies(self, request, user_permissions):
        key = f"{request.user_id}:{request.resource}:{request.permission}"
        return self.decisions.get(key, AccessDecision.ABSTAIN)


@pytest.fixture
def security_components(tmp_path):
    """创建完整的安保组件集合"""
    # 创建配置管理器

    # 创建审计管理器
    audit_manager = AuditManager(log_path=str(tmp_path / "audit"))

    # 创建策略管理器
    policy_manager = MockPolicyManager()

    config_manager = ConfigManager(config_path=tmp_path, enable_hot_reload=False)

    # 创建角色管理器
    role_manager = RoleManager()

    # 创建默认角色，提供测试所需权限
    trader_role_id = role_manager.create_role(
        role_id='role_trader',
        name='Trader',
        permissions={'trade:execute', 'data:read'}
    )
    admin_role_id = role_manager.create_role(
        role_id='role_admin',
        name='Admin',
        permissions={'system:admin', 'admin'}
    )

    # 创建用户管理器并注册默认用户
    user_manager = UserManager()
    trader = user_manager.create_user(UserCreationParams(
        username='admin1',
        email='admin1@example.com',
        roles={'admin'}
    ))
    workflow_user = user_manager.create_user(UserCreationParams(
        username='workflow_user',
        email='workflow@example.com',
        roles={'trader'}
    ))
    cache_user = user_manager.create_user(UserCreationParams(
        username='cache_test_user',
        email='cache@example.com',
        roles={'admin'}
    ))

    # 创建审计管理器
    audit_manager = AuditManager(log_path=str(tmp_path / "audit"))

    # 创建策略管理器
    policy_manager = MockPolicyManager()

    # 创建访问检查器
    access_checker = AccessChecker(
        user_manager=user_manager,
        role_manager=role_manager,
        policy_manager=policy_manager,
        audit_manager=audit_manager,
        cache_enabled=True
    )

    return {
        'config_manager': config_manager,
        'role_manager': role_manager,
        'user_manager': user_manager,
        'audit_manager': audit_manager,
        'access_checker': access_checker,
        'policy_manager': policy_manager
    }


class TestSecurityIntegration:
    """安全模块集成测试"""

    def test_complete_user_registration_and_access_flow(self, security_components):
        """测试完整的用户注册和访问流程"""
        components = security_components
        user_manager = components['user_manager']
        role_manager = components['role_manager']
        access_checker = components['access_checker']
        audit_manager = components['audit_manager']

        # 1. 创建用户
        user_params = UserCreationParams(
            username='john_doe',
            email='john.doe@example.com',
            password='secure_password123',
            roles={'trader'}  # 使用字符串角色名
        )

        user = user_manager.create_user(user_params)
        assert user is not None
        assert user.username == 'john_doe'  # user_id是自动生成的
        assert 'trader' in user.roles  # 角色存储为字符串

        # 3. 检查用户访问权限
        # 交易员应该有交易权限
        decision = access_checker.check_access(user.user_id, '/api/trade', 'trade:execute')
        assert decision == AccessDecision.ALLOW

        # 交易员不应该有管理员权限
        decision = access_checker.check_access(user.user_id, '/api/admin', 'system:admin')
        assert decision == AccessDecision.DENY

        # 4. 验证审计日志记录了访问
        events = audit_manager.query_events(QueryFilterParams())
        access_events = [e for e in events if e.get('event_type') == 'security']
        assert len(access_events) >= 2  # 至少有ALLOW和DENY事件

    def test_role_based_access_control_workflow(self, security_components):
        """测试基于角色的访问控制工作流程"""
        components = security_components
        role_manager = components['role_manager']
        user_manager = components['user_manager']
        access_checker = components['access_checker']

        # 1. 创建自定义角色
        analyst_role = role_manager.create_role(
            role_id='senior_analyst',
            name='高级分析师',
            permissions={'data:read', 'data:export', 'report:generate'},
            parent_roles={'analyst'}  # 继承分析师角色
        )

        # 2. 创建用户并分配角色
        user_params = UserCreationParams(
            username='analyst_user',
            email='analyst@example.com',
            password='password123',
            roles={'analyst'}  # 使用字符串角色名
        )
        user = user_manager.create_user(user_params)

        # 4. 验证权限继承
        # 应该有分析师的基本权限
        decision = access_checker.check_access(user.user_id, '/api/data', 'data:read')
        assert decision == AccessDecision.ALLOW

        # 应该有分析师的导出权限
        decision = access_checker.check_access(user.user_id, '/api/data', 'data:export')
        assert decision == AccessDecision.ALLOW

        # 不应该有交易权限
        decision = access_checker.check_access(user.user_id, '/api/trade', 'trade:execute')
        assert decision == AccessDecision.DENY

    def test_audit_and_monitoring_integration(self, security_components):
        """测试审计和监控的集成"""
        components = security_components
        access_checker = components['access_checker']
        audit_manager = components['audit_manager']

        # 1. 创建和激活用户
        user_params = UserCreationParams(
            username='monitor_user',
            email='monitor@example.com',
            password='password123',
            roles={'trader'}
        )
        user = components['user_manager'].create_user(user_params)

        # 2. 执行一系列访问操作
        operations = [
            ('/api/trade', 'trade:execute', AccessDecision.ALLOW),
            ('/api/admin', 'system:admin', AccessDecision.DENY),
            ('/api/data', 'data:read', AccessDecision.ALLOW),
            ('/api/restricted', 'secret:access', AccessDecision.DENY),
        ]

        for resource, permission, expected_decision in operations:
            decision = access_checker.check_access(user.user_id, resource, permission)
            assert decision == expected_decision

        # 3. 验证审计记录
        all_events = audit_manager.query_events(QueryFilterParams())

        # 应该记录了所有访问尝试
        security_events = [e for e in all_events if e.get('event_type') == 'security']
        assert len(security_events) >= len(operations)

        # 验证事件详情
        for event in security_events:
            assert 'user_id' in event
            assert 'resource' in event
            assert 'timestamp' in event
            assert 'severity' in event
            assert 'details' in event
            assert 'permission' in event['details']  # 权限在details中

    def test_configuration_hot_reload_integration(self, security_components):
        """测试配置热重载的集成"""
        components = security_components
        config_manager = components['config_manager']
        access_checker = components['access_checker']

        # 1. 修改配置
        config_manager.set_config('cache.max_size', 200)
        config_manager.set_config('security.max_login_attempts', 5)

        # 2. 保存配置
        success = config_manager._save_config()
        assert success

        # 3. 触发手动重载
        reload_success = config_manager.trigger_manual_reload()
        assert reload_success

        # 4. 验证配置已重载
        max_size = config_manager.get_config('cache.max_size')
        assert max_size == 200

    @pytest.mark.asyncio
    async def test_async_operations_integration(self, security_components):
        """测试异步操作的集成"""
        components = security_components
        access_checker = components['access_checker']

        # 1. 创建多个访问请求
        requests_data = [
            ('user1', '/api/trade', 'trade:execute'),
            ('user2', '/api/data', 'data:read'),
            ('user3', '/api/admin', 'system:admin'),
        ]

        # 2. 并发执行异步访问检查
        tasks = []
        for user_id, resource, permission in requests_data:
            task = access_checker.check_access_async(user_id, resource, permission)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # 3. 验证结果
        assert len(results) == 3
        # 结果将根据用户权限和资源权限而定

    def test_security_policy_enforcement(self, security_components):
        """测试安全策略执行"""
        components = security_components
        policy_manager = components['policy_manager']
        access_checker = components['access_checker']

        # 1. 设置策略决策
        policy_manager.decisions = {
            'policy_user:/api/special:special:access': AccessDecision.ALLOW,
            'policy_user:/api/blocked:blocked:access': AccessDecision.DENY,
        }

        # 2. 创建用户（没有直接权限）
        user = components['user_manager'].create_user(
            user_id='policy_user',
            username='policy_user',
            email='policy@example.com',
            password='password123',
            roles=[UserRole.GUEST]  # 只有基本权限
        )

        # 3. 测试策略允许
        decision = access_checker.check_access('policy_user', '/api/special', 'special:access')
        assert decision == AccessDecision.ALLOW

        # 4. 测试策略拒绝
        decision = access_checker.check_access('policy_user', '/api/blocked', 'blocked:access')
        assert decision == AccessDecision.DENY

    def test_cache_performance_integration(self, security_components):
        """测试缓存性能的集成"""
        components = security_components
        access_checker = components['access_checker']

        # 1. 创建用户
        user_params = UserCreationParams(
            username='cache_test_user',
            email='cache@example.com',
            password='password123',
            roles={'trader'}
        )
        user = components['user_manager'].create_user(user_params)

        # 2. 执行多次相同访问检查（测试缓存命中）
        resource, permission = '/api/trade', 'trade:execute'

        # 第一次访问（缓存未命中）
        decision1 = access_checker.check_access('cache_test_user', resource, permission)
        assert decision1 == AccessDecision.ALLOW

        # 重复访问（应该缓存命中）
        for _ in range(10):
            decision = access_checker.check_access('cache_test_user', resource, permission)
            assert decision == AccessDecision.ALLOW

        # 3. 检查缓存统计
        cache_stats = access_checker.get_cache_stats()
        assert cache_stats['total_entries'] > 0
        assert cache_stats['cache_hits'] >= 9  # 至少9次缓存命中
        assert cache_stats['cache_misses'] >= 1  # 至少1次缓存未命中

    def test_role_hierarchy_and_inheritance_integration(self, security_components):
        """测试角色层次结构和权限继承的集成"""
        components = security_components
        role_manager = components['role_manager']
        access_checker = components['access_checker']

        # 1. 创建角色层次结构
        # 员工 -> 经理 -> 总监
        employee_role = role_manager.create_role(
            'employee',
            '员工',
            permissions={'basic:read'}
        )

        manager_role = role_manager.create_role(
            'manager',
            '经理',
            permissions={'manager:approve', 'team:manage'},
            parent_roles={'employee'}
        )

        director_role = role_manager.create_role(
            'director',
            '总监',
            permissions={'director:decide', 'budget:approve'},
            parent_roles={'manager'}
        )

        # 2. 创建用户并分配角色
        director_params = UserCreationParams(
            username='director_user',
            email='director@example.com',
            password='password123',
            roles={'director'}  # 简化的角色分配
        )
        director_user = components['user_manager'].create_user(director_params)

        # 3. 验证权限继承
        # 总监应该继承所有上级角色的权限
        permissions_to_check = [
            ('basic:read', AccessDecision.ALLOW),  # 员工权限
            ('manager:approve', AccessDecision.ALLOW),  # 经理权限
            ('director:decide', AccessDecision.ALLOW),  # 总监权限
            ('admin:access', AccessDecision.DENY),  # 无权限
        ]

        for permission, expected_decision in permissions_to_check:
            decision = access_checker.check_access('director_user', '/api/test', permission)
            assert decision == expected_decision

    def test_error_handling_and_recovery_integration(self, security_components):
        """测试错误处理和恢复的集成"""
        components = security_components
        access_checker = components['access_checker']

        # 1. 测试不存在的用户
        decision = access_checker.check_access('nonexistent_user', '/api/test', 'test:access')
        assert decision == AccessDecision.DENY

        # 2. 测试无效的权限格式
        decision = access_checker.check_access('admin1', '/api/test', '')
        assert decision == AccessDecision.DENY

        # 3. 测试系统异常处理（模拟）
        with patch.object(access_checker, 'user_manager') as mock_um:
            mock_um.get_user.side_effect = Exception("Database connection failed")

            decision = access_checker.check_access('admin1', '/api/test', 'test:access')
            assert decision == AccessDecision.DENY  # 异常时默认拒绝

    def test_batch_operations_performance(self, security_components):
        """测试批量操作性能"""
        components = security_components
        access_checker = components['access_checker']

        # 1. 准备批量请求
        batch_requests = []
        for i in range(50):
            from src.infrastructure.security.access.components.access_checker import AccessRequest
            request = AccessRequest(
                user_id='admin1',  # 假设admin1存在
                resource=f'/api/resource{i}',
                permission='admin'  # 假设admin权限
            )
            batch_requests.append(request)

        # 2. 执行批量检查
        import time
        start_time = time.time()
        decisions = access_checker.batch_check_access(batch_requests)
        batch_time = time.time() - start_time

        # 3. 验证结果
        assert len(decisions) == 50
        assert all(d == AccessDecision.ALLOW for d in decisions)

        # 4. 验证性能（批量处理应该很快）
        assert batch_time < 2.0  # 50个请求应该在2秒内完成

    def test_security_compliance_reporting_integration(self, security_components):
        """测试安全合规报告的集成"""
        components = security_components
        audit_manager = components['audit_manager']

        # 1. 生成一些审计事件
        for i in range(10):
            event_type = EventType.SECURITY if i % 2 == 0 else EventType.DATA_OPERATION
            severity = EventSeverity.HIGH if i % 3 == 0 else EventSeverity.MEDIUM

            params = AuditEventParams(
                event_type=event_type,
                severity=severity,
                user_id=f'user{i}',
                resource=f'/api/resource{i}',
                action=f'action{i}',
                details={'compliance_check': True}
            )
            audit_manager.log_event(params)

        # 2. 生成合规报告
        compliance_report = audit_manager.get_compliance_report("general")

        # 3. 验证报告结构
        assert isinstance(compliance_report, dict)
        assert 'compliance_type' in compliance_report
        assert 'report_date' in compliance_report
        assert 'metrics' in compliance_report

        # 4. 验证合规指标
        metrics = compliance_report['metrics']
        assert 'total_auditable_events' in metrics
        assert metrics['total_auditable_events'] >= 10

    def test_end_to_end_security_workflow(self, security_components):
        """测试端到端的安保工作流程"""
        components = security_components

        # 1. 用户注册和角色分配
        user_params = UserCreationParams(
            username='workflow_user',
            email='workflow@example.com',
            password='password123',
            roles={'trader'}
        )
        user = components['user_manager'].create_user(user_params)

        # 2. 配置更新
        components['config_manager'].set_config('security.session_timeout', 3600)

        # 3. 访问控制检查
        decision = components['access_checker'].check_access(
            'workflow_user', '/api/trade', 'trade:execute'
        )
        assert decision == AccessDecision.ALLOW

        # 4. 审计日志记录
        events = components['audit_manager'].query_events(
            AuditManager.QueryFilterParams(user_ids={'workflow_user'})
        )
        assert len(events) > 0

        # 5. 生成安全报告
        report = components['audit_manager'].generate_security_report(
            AuditManager.ReportGenerationParams(report_type="security_summary")
        )
        assert report['report_type'] == 'security_summary'

        # 6. 验证缓存生效
        cache_stats = components['access_checker'].get_cache_stats()
        assert cache_stats['total_entries'] >= 1

        print("✅ 端到端安保工作流程测试通过")
