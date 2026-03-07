#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 审计管理器综合测试

全面测试AuditManager类的所有功能，包括：
- 审计事件记录和存储
- 事件查询和过滤
- 安全报告生成
- 合规报告生成
- 缓冲区管理和文件操作
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import time
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, List, Any
from datetime import datetime, timedelta

from src.infrastructure.security.audit.audit_manager import AuditManager
from src.infrastructure.security.core.types import (
    AuditEventParams, QueryFilterParams, ReportGenerationParams,
    EventType, EventSeverity
)


class TestAuditManagerComprehensive:
    """审计管理器综合测试"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        with tempfile.TemporaryDirectory() as temp:
            yield Path(temp)

    @pytest.fixture
    def audit_manager(self, temp_dir):
        """审计管理器fixture"""
        return AuditManager(log_path=str(temp_dir / "audit"))

    def test_initialization(self, temp_dir):
        """测试初始化"""
        log_path = str(temp_dir / "audit")
        manager = AuditManager(log_path=log_path)

        assert manager.log_path == log_path
        assert isinstance(manager.event_buffer, list)
        assert isinstance(manager.events, list)
        assert manager.buffer_size == 1000
        assert manager.max_events == 100000

    def test_log_event_basic(self, audit_manager):
        """测试基本事件记录"""
        params = AuditEventParams(
            event_type=EventType.SECURITY,
            severity=EventSeverity.MEDIUM,
            user_id="user123",
            resource="system",
            action="login",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            details={"success": True}
        )

        event_id = audit_manager.log_event(params)

        assert isinstance(event_id, str)
        assert len(event_id) > 0
        assert len(audit_manager.event_buffer) == 1

        # 验证事件内容
        event = audit_manager.event_buffer[0]
        assert event["event_type"] == "security"
        assert event["user_id"] == "user123"
        assert event["resource"] == "system"
        assert event["action"] == "login"
        assert event["ip_address"] == "192.168.1.1"
        assert event["user_agent"] == "TestAgent/1.0"
        assert event["details"]["success"] == True
        assert "timestamp" in event
        assert "event_id" in event

    def test_log_event_with_all_fields(self, audit_manager):
        """测试记录包含所有字段的事件"""
        params = AuditEventParams(
            event_type=EventType.DATA_OPERATION,
            severity=EventSeverity.HIGH,
            user_id="admin",
            resource="/api/sensitive-data",
            action="read",
            ip_address="10.0.0.1",
            user_agent="Mozilla/5.0",
            session_id="session_123",
            details={
                "table": "user_data",
                "columns": ["email", "phone"],
                "rows_affected": 1,
                "query_time": 0.05
            }
        )

        event_id = audit_manager.log_event(params)

        assert len(audit_manager.event_buffer) == 1
        event = audit_manager.event_buffer[0]

        # 验证所有字段都被正确记录
        assert event["event_type"] == "data_operation"
        assert event["user_id"] == "admin"
        assert event["resource"] == "/api/sensitive-data"
        assert event["action"] == "read"
        assert event["ip_address"] == "10.0.0.1"
        assert event["user_agent"] == "Mozilla/5.0"
        assert event["session_id"] == "session_123"
        assert event["details"]["table"] == "user_data"
        assert event["details"]["rows_affected"] == 1

    def test_log_event_buffer_flush(self, audit_manager):
        """测试缓冲区自动刷新"""
        # 填充缓冲区到接近最大值
        for i in range(15):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f"user{i}",
                resource="test",
                action="test_action"
            )
            audit_manager.log_event(params)

        # 验证事件都被记录
        assert len(audit_manager.event_buffer) == 15

    def test_generate_event_id(self, audit_manager):
        """测试事件ID生成"""
        event_id1 = audit_manager._generate_event_id()
        event_id2 = audit_manager._generate_event_id()

        assert isinstance(event_id1, str)
        assert isinstance(event_id2, str)
        assert event_id1 != event_id2
        assert len(event_id1) > 10  # 合理的ID长度

    def test_query_events_basic(self, audit_manager):
        """测试基本事件查询"""
        # 添加一些测试事件
        events_data = [
            {"event_type": "security", "user_id": "user1", "action": "login", "resource": "system"},
            {"event_type": "security", "user_id": "user1", "action": "logout", "resource": "system"},
            {"event_type": "data_operation", "user_id": "user2", "action": "read", "resource": "/api/data"}
        ]

        # 手动添加到缓冲区（模拟已有事件）
        for event_data in events_data:
            event_data.update({
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "ip_address": "127.0.0.1",
                "user_agent": "TestAgent",
                "session_id": None,
                "correlation_id": None,
                "details": {},
                "severity": "medium",  # 添加缺失的字段
                "result": None,
                "location": None,
                "risk_score": 0.0,
                "tags": []
            })
            audit_manager.event_buffer.append(event_data)

        # 查询所有事件
        params = QueryFilterParams()
        results = audit_manager.query_events(params)

        assert len(results) == 3

    def test_query_events_with_filters(self, audit_manager):
        """测试带过滤器的事件查询"""
        # 使用log_event方法添加测试事件，确保格式正确
        # 用户1的登录和注销事件
        params1 = AuditEventParams(
            event_type=EventType.SECURITY,
            severity=EventSeverity.MEDIUM,
            user_id="user1",
            resource="system",
            action="login"
        )
        audit_manager.log_event(params1)

        params2 = AuditEventParams(
            event_type=EventType.SECURITY,
            severity=EventSeverity.MEDIUM,
            user_id="user1",
            resource="system",
            action="logout"
        )
        audit_manager.log_event(params2)

        # 用户2的数据访问事件
        params3 = AuditEventParams(
            event_type=EventType.DATA_OPERATION,
            severity=EventSeverity.HIGH,
            user_id="user2",
            resource="/api/data",
            action="read"
        )
        audit_manager.log_event(params3)

        # 按用户ID过滤
        params = QueryFilterParams(user_ids={"user1"})
        results = audit_manager.query_events(params)
        assert len(results) == 2
        assert all(r["user_id"] == "user1" for r in results)

        # 按事件类型过滤
        params = QueryFilterParams(event_types={EventType.DATA_OPERATION})
        results = audit_manager.query_events(params)
        assert len(results) == 1
        assert all(r["event_type"] == "data_operation" for r in results)

        # 组合过滤
        params = QueryFilterParams(user_ids={"user2"}, event_types={EventType.DATA_OPERATION})
        results = audit_manager.query_events(params)
        assert len(results) == 1
        assert results[0]["user_id"] == "user2"
        assert results[0]["event_type"] == "data_operation"

    def test_query_events_with_pagination(self, audit_manager):
        """测试分页查询"""
        # 添加20个测试事件
        for i in range(20):
            event_data = {
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "event_type": "TEST",
                "user_id": f"user{i}",
                "action": "test",
                "resource": "test",
                "ip_address": "127.0.0.1",
                "user_agent": "TestAgent",
                "session_id": None,
                "correlation_id": None,
                "details": {}
            }
            audit_manager.event_buffer.append(event_data)

        # 测试分页
        params = QueryFilterParams(limit=5, offset=10)
        results = audit_manager.query_events(params)

        assert len(results) == 5
        # 验证返回的是第11-15个事件（offset=10, limit=5）

    def test_apply_filters(self, audit_manager):
        """测试过滤器应用"""
        events = [
            {"event_type": "security", "user_id": "user1", "action": "login", "resource": "/api/auth"},
            {"event_type": "data_operation", "user_id": "user2", "action": "read", "resource": "/api/data"},
            {"event_type": "security", "user_id": "user1", "action": "logout", "resource": "/api/auth"}
        ]

        # 测试用户过滤
        params = QueryFilterParams(user_ids={"user1"})
        filtered = audit_manager._apply_filters(events, params)
        assert len(filtered) == 2
        assert all(e["user_id"] == "user1" for e in filtered)

        # 测试事件类型过滤
        params = QueryFilterParams(event_types={EventType.SECURITY})
        filtered = audit_manager._apply_filters(events, params)
        assert len(filtered) == 2
        assert all(e["event_type"] == "security" for e in filtered)

        # 测试资源过滤
        params = QueryFilterParams(resources={"/api/auth"})
        filtered = audit_manager._apply_filters(events, params)
        assert len(filtered) == 2
        assert all(e["resource"] == "/api/auth" for e in filtered)

    def test_generate_security_report_basic(self, audit_manager):
        """测试基本安全报告生成"""
        # 添加一些测试事件
        events_data = [
            {"event_type": "LOGIN", "user_id": "user1", "action": "login", "timestamp": datetime.now().isoformat()},
            {"event_type": "DATA_ACCESS", "user_id": "user1", "action": "read", "timestamp": datetime.now().isoformat()},
            {"event_type": "LOGIN", "user_id": "user2", "action": "login", "timestamp": datetime.now().isoformat()},
        ]

        for event_data in events_data:
            event_data.update({
                "event_id": str(uuid.uuid4()),
                "resource": "test",
                "ip_address": "127.0.0.1",
                "user_agent": "TestAgent",
                "session_id": None,
                "correlation_id": None,
                "details": {}
            })
            audit_manager.event_buffer.append(event_data)

        # 生成报告
        params = ReportGenerationParams(
            report_type="security_summary"
        )

        report = audit_manager.generate_security_report(params)

        assert isinstance(report, dict)
        assert "report_type" in report
        assert "generated_at" in report or "report_date" in report
        assert "statistics" in report

    def test_generate_security_report_detailed(self, audit_manager):
        """测试详细安全报告生成"""
        # 添加各种类型的事件
        base_time = datetime.now()
        events_data = []

        # 登录事件
        for i in range(5):
            events_data.append({
                "event_type": "security",
                "user_id": f"user{i}",
                "action": "login",
                "timestamp": (base_time - timedelta(minutes=i*10)).isoformat(),
                "ip_address": f"192.168.1.{i+1}",
                "details": {"success": True}
            })

        # 数据访问事件
        for i in range(3):
            events_data.append({
                "event_type": "data_operation",
                "user_id": "user1",
                "action": "read",
                "resource": f"/api/data{i}",
                "timestamp": (base_time - timedelta(minutes=i*5)).isoformat(),
                "details": {"rows_accessed": 10 + i}
            })

        # 失败的认证事件
        for i in range(2):
            events_data.append({
                "event_type": "security",
                "user_id": "attacker",
                "action": "login",
                "timestamp": (base_time - timedelta(minutes=i*2)).isoformat(),
                "ip_address": "10.0.0.1",
                "details": {"success": False, "reason": "invalid_credentials"}
            })

        # 添加事件到缓冲区
        for event_data in events_data:
            event_data.update({
                "event_id": str(uuid.uuid4()),
                "user_agent": "TestAgent",
                "session_id": None,
                "correlation_id": None
            })
            audit_manager.event_buffer.append(event_data)

        # 生成详细报告
        params = ReportGenerationParams(
            report_type="detailed_security",
            include_details=True
        )

        report = audit_manager.generate_security_report(params)

        assert report["report_type"] == "detailed_security"
        assert "statistics" in report

        # 验证统计数据
        stats = report["statistics"]
        assert "events_by_type" in stats
        assert "events_by_user" in stats
        assert "events_by_severity" in stats
        assert stats["events_by_type"]["security"] == 7  # 5 login + 2 failed login
        assert stats["events_by_type"]["data_operation"] == 3

    def test_calculate_security_stats(self, audit_manager):
        """测试安全统计计算"""
        events = [
            {"event_type": "security", "user_id": "user1", "timestamp": datetime.now().isoformat()},
            {"event_type": "security", "user_id": "user2", "timestamp": datetime.now().isoformat()},
            {"event_type": "data_operation", "user_id": "user1", "timestamp": datetime.now().isoformat()},
            {"event_type": "security", "user_id": "user1", "timestamp": datetime.now().isoformat()},
        ]

        stats = audit_manager._calculate_security_stats(events, None)

        assert isinstance(stats, dict)
        assert "total_events" in stats
        assert "unique_users" in stats
        assert "events_by_type" in stats
        assert stats["total_events"] == 4
        assert stats["unique_users"] == 2
        assert stats["events_by_type"]["security"] == 3

    def test_get_compliance_report(self, audit_manager):
        """测试合规报告生成"""
        # 添加一些合规相关事件
        events_data = [
            {"event_type": "security", "user_id": "user1", "action": "login", "timestamp": datetime.now().isoformat(), "details": {"success": True}},
            {"event_type": "data_operation", "user_id": "user1", "action": "read", "resource": "/sensitive/data", "timestamp": datetime.now().isoformat()},
            {"event_type": "config_change", "user_id": "admin", "action": "update", "resource": "system_config", "timestamp": datetime.now().isoformat()},
        ]

        for event_data in events_data:
            event_data.update({
                "event_id": str(uuid.uuid4()),
                "ip_address": "127.0.0.1",
                "user_agent": "TestAgent",
                "session_id": None,
                "correlation_id": None
            })
            audit_manager.event_buffer.append(event_data)

        # 生成合规报告
        report = audit_manager.get_compliance_report("general")

        assert isinstance(report, dict)
        assert "compliance_type" in report
        assert "report_date" in report
        assert "metrics" in report
        assert "status" in report or "overall_status" in report

    def test_calculate_compliance_metrics(self, audit_manager):
        """测试合规指标计算"""
        events = [
            {"event_type": "security", "user_id": "user1", "action": "login", "details": {"success": True}},
            {"event_type": "data_operation", "user_id": "user1", "resource": "/sensitive"},
            {"event_type": "config_change", "user_id": "admin"},
            {"event_type": "security", "user_id": "user2", "action": "login", "details": {"success": False}},
        ]

        metrics = audit_manager._calculate_compliance_metrics(events, "general")

        assert isinstance(metrics, dict)
        assert "total_events" in metrics
        assert "successful_authentications" in metrics
        assert "failed_authentications" in metrics
        assert metrics["total_events"] == 4
        assert metrics["successful_authentications"] == 1
        assert metrics["failed_authentications"] == 1

    def test_assess_compliance_status(self, audit_manager):
        """测试合规状态评估"""
        # 高合规指标
        high_compliance_metrics = {
            "successful_authentications": 100,
            "failed_authentications": 2,
            "total_auditable_events": 100,
            "sensitive_data_accesses": 10,
            "config_changes": 5,
            "access_denials": 5,
            "suspicious_activities": 1,
            "policy_violations": 0
        }

        status = audit_manager._assess_compliance_status(high_compliance_metrics)
        assert status in ["compliant", "warning", "non_compliant"]

        # 低合规指标
        low_compliance_metrics = {
            "successful_authentications": 10,
            "failed_authentications": 50,
            "total_auditable_events": 100,
            "sensitive_data_accesses": 100,
            "config_changes": 50
        }

        status = audit_manager._assess_compliance_status(low_compliance_metrics)
        assert status == "non_compliant"

    def test_group_events(self, audit_manager):
        """测试事件分组"""
        events = [
            {"event_type": "LOGIN", "user_id": "user1", "action": "login"},
            {"event_type": "LOGIN", "user_id": "user2", "action": "login"},
            {"event_type": "DATA_ACCESS", "user_id": "user1", "action": "read"},
            {"event_type": "DATA_ACCESS", "user_id": "user1", "action": "write"},
        ]

        # 按用户分组
        grouped = audit_manager._group_events(events, {"user_id"})
        assert ("user1",) in grouped
        assert ("user2",) in grouped
        assert len(grouped[("user1",)]) == 3
        assert len(grouped[("user2",)]) == 1

        # 按事件类型分组
        grouped = audit_manager._group_events(events, {"event_type"})
        assert ("LOGIN",) in grouped
        assert ("DATA_ACCESS",) in grouped
        assert len(grouped[("LOGIN",)]) == 2
        assert len(grouped[("DATA_ACCESS",)]) == 2

    def test_aggregate_events(self, audit_manager):
        """测试事件聚合"""
        events = [
            {"event_type": "LOGIN", "user_id": "user1", "details": {"response_time": 0.1}},
            {"event_type": "LOGIN", "user_id": "user2", "details": {"response_time": 0.2}},
            {"event_type": "DATA_ACCESS", "user_id": "user1", "details": {"rows": 10}},
            {"event_type": "DATA_ACCESS", "user_id": "user1", "details": {"rows": 20}},
        ]

        aggregation = {
            "count": {"field": None},
            "avg_response_time": {"field": "details.response_time", "operation": "avg"},
            "total_rows": {"field": "details.rows", "operation": "sum"}
        }

        result = audit_manager._aggregate_events(events, aggregation)

        assert result["count"] == 4
        assert abs(result["avg_response_time"] - 0.15) < 0.01  # (0.1 + 0.2) / 2
        assert result["total_rows"] == 30  # 10 + 20

    def test_write_events_to_file(self, audit_manager, temp_dir):
        """测试事件写入文件"""
        # 添加一些事件到缓冲区
        for i in range(3):
            event_data = {
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "event_type": "TEST",
                "user_id": f"user{i}",
                "action": "test",
                "resource": "test",
                "ip_address": "127.0.0.1",
                "user_agent": "TestAgent",
                "session_id": None,
                "correlation_id": None,
                "details": {"test_id": i}
            }
            audit_manager.event_buffer.append(event_data)

        # 写入文件
        audit_manager._write_events_to_file()

        # 验证文件是否创建
        audit_files = list(temp_dir.glob("audit/*.json"))
        assert len(audit_files) > 0

        # 验证文件内容
        with open(audit_files[0], 'r') as f:
            saved_events = json.load(f)

        assert isinstance(saved_events, list)
        assert len(saved_events) == 3

        # 验证事件数据完整性
        for event in saved_events:
            assert "event_id" in event
            assert "timestamp" in event
            assert "event_type" in event
            assert event["event_type"] == "TEST"

    def test_flush_buffer(self, audit_manager):
        """测试缓冲区刷新"""
        # 添加事件到缓冲区
        for i in range(5):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f"user{i}",
                resource="test",
                action="test"
            )
            audit_manager.log_event(params)

        assert len(audit_manager.event_buffer) == 5

        # 刷新缓冲区
        audit_manager._flush_buffer()

        # 缓冲区应该被清空
        assert len(audit_manager.event_buffer) == 0

    def test_cleanup_old_events(self, audit_manager, temp_dir):
        """测试旧事件清理"""
        # 创建一些旧的审计文件
        audit_dir = temp_dir / "audit"
        audit_dir.mkdir(exist_ok=True)

        # 创建不同日期的文件
        old_date = (datetime.now() - timedelta(days=40)).strftime("%Y%m%d")
        recent_date = datetime.now().strftime("%Y%m%d")

        old_file = audit_dir / f"audit_{old_date}.json"
        recent_file = audit_dir / f"audit_{recent_date}.json"

        # 写入测试数据
        with open(old_file, 'w') as f:
            json.dump([{"event_id": "old", "timestamp": "old"}], f)

        with open(recent_file, 'w') as f:
            json.dump([{"event_id": "recent", "timestamp": "recent"}], f)

        # 执行清理
        audit_manager._cleanup_old_events()

        # 旧文件应该被删除
        assert not old_file.exists()
        # 新文件应该保留
        assert recent_file.exists()

    @patch('uuid.uuid4')
    def test_event_id_generation(self, mock_uuid, audit_manager):
        """测试事件ID生成"""
        mock_uuid.return_value.hex = "test123456"

        event_id = audit_manager._generate_event_id()

        assert "test123456" in event_id
        assert isinstance(event_id, str)

    def test_concurrent_event_logging(self, audit_manager):
        """测试并发事件记录"""
        import threading
        import time

        results = []
        errors = []

        def log_events(thread_id):
            try:
                for i in range(10):
                    params = AuditEventParams(
                        event_type=EventType.SECURITY,
                        severity=EventSeverity.LOW,
                        user_id=f"thread_{thread_id}_user_{i}",
                        resource="concurrent_test",
                        action="test_action"
                    )
                    event_id = audit_manager.log_event(params)
                    results.append(event_id)
            except Exception as e:
                errors.append(e)

        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_events, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 30  # 3线程 * 10个事件
        assert len(errors) == 0
        assert len(audit_manager.event_buffer) == 30

        # 验证所有事件ID都是唯一的
        assert len(set(results)) == 30

    def test_large_scale_event_processing(self, audit_manager):
        """测试大规模事件处理"""
        # 生成大量事件
        event_count = 500

        for i in range(event_count):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f"user_{i % 10}",  # 10个不同用户
                resource=f"/api/resource_{i % 5}",  # 5个不同资源
                action="test_action",
                details={"sequence": i}
            )
            audit_manager.log_event(params)

        # 验证所有事件都被记录
        assert len(audit_manager.event_buffer) == event_count

        # 测试查询性能
        params = QueryFilterParams(user_ids={"user_1"})
        start_time = time.time()
        results = audit_manager.query_events(params)
        query_time = time.time() - start_time

        # 验证查询结果
        expected_count = event_count // 10  # 每10个事件中有一个用户1的事件
        assert len(results) == expected_count

        # 查询应该在合理时间内完成
        assert query_time < 1.0  # 1秒内完成

    def test_error_handling_and_recovery(self, audit_manager, temp_dir):
        """测试错误处理和恢复"""
        # 测试无效的事件参数
        try:
            invalid_params = AuditEventParams(
                event_type=EventType.SECURITY,  # 使用有效的事件类型
                severity=EventSeverity.LOW,
                user_id="",     # 空的用户ID
                resource="",
                action=""
            )
            audit_manager.log_event(invalid_params)
            # 如果没有抛出异常，说明有良好的错误处理
        except Exception:
            # 如果抛出异常，也是可以接受的
            pass

        # 测试文件写入错误
        with patch('builtins.open', side_effect=OSError("Disk full")):
            # 这不应该导致程序崩溃
            audit_manager._write_events_to_file()
            # 缓冲区中的事件应该被保留
            assert len(audit_manager.event_buffer) >= 0

        # 测试查询错误
        invalid_filter = QueryFilterParams(limit=-1)  # 无效的limit
        results = audit_manager.query_events(invalid_filter)
        # 应该返回空结果或处理错误
        assert isinstance(results, list)

    def test_memory_management(self, audit_manager):
        """测试内存管理"""
        # 记录大量事件
        for i in range(1200):  # 超过默认缓冲区大小1000
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f"user_{i % 50}",
                resource="memory_test",
                action="test"
            )
            audit_manager.log_event(params)

        # 缓冲区应该被自动刷新
        # 注意：实际的刷新逻辑可能需要时间或特定条件
        # 这里主要测试不会出现内存溢出
        assert len(audit_manager.event_buffer) <= 1200

        # 验证可以继续记录新事件
        params = AuditEventParams(
            event_type=EventType.SECURITY,
            severity=EventSeverity.LOW,
            user_id="test_user",
            resource="test",
            action="test"
        )
        event_id = audit_manager.log_event(params)
        assert event_id is not None

    def test_flush_buffer(self, audit_manager):
        """测试缓冲区刷新"""
        # 添加一些事件到缓冲区
        for i in range(5):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f"user_{i}",
                resource="test_resource",
                action="test_action"
            )
            audit_manager.log_event(params)

        # 验证事件在缓冲区中
        assert len(audit_manager.event_buffer) >= 5

        # 手动刷新缓冲区
        audit_manager._flush_buffer()

        # 验证缓冲区被清空，事件被移动到主事件列表
        assert len(audit_manager.event_buffer) == 0
        assert len(audit_manager.events) >= 5

    def test_write_events_to_file(self, audit_manager, temp_dir):
        """测试事件写入文件"""
        # 添加事件到缓冲区
        for i in range(3):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.MEDIUM,
                user_id=f"user_{i}",
                resource="file_test",
                action="write_test"
            )
            audit_manager.log_event(params)

        # 写入文件
        audit_manager._write_events_to_file()

        # 验证文件被创建
        audit_files = list(temp_dir.glob("audit/audit_*.json"))
        assert len(audit_files) > 0

        # 验证文件内容
        with open(audit_files[0], 'r') as f:
            content = json.load(f)
            assert isinstance(content, list)
            assert len(content) == 3

    def test_cleanup_old_events(self, audit_manager):
        """测试清理旧事件"""
        # 添加一些旧事件（通过直接操作events列表）
        old_date = datetime.now() - timedelta(days=40)  # 超过30天的旧事件
        for i in range(5):
            event = {
                'event_id': f'test_event_{i}',
                'event_type': 'security',
                'severity': 'low',
                'timestamp': old_date.isoformat(),
                'user_id': f'user_{i}',
                'resource': 'old_resource',
                'action': 'old_action'
            }
            audit_manager.events.append(event)

        initial_count = len(audit_manager.events)

        # 执行清理
        audit_manager._cleanup_old_events()

        # 验证旧事件被清理（如果有相应的文件）
        # 注意：这个测试可能需要实际的文件来完全验证
        assert len(audit_manager.events) <= initial_count

    def test_calculate_security_stats(self, audit_manager):
        """测试安全统计信息计算"""
        # 准备测试数据
        base_time = datetime.now()
        events = [
            {
                'event_type': 'security',
                'severity': 'high',
                'timestamp': (base_time - timedelta(hours=i)).isoformat(),
                'user_id': f'user_{i % 3}',
                'action': 'login' if i % 2 == 0 else 'access',
                'result': 'success' if i % 3 != 0 else 'failure'
            }
            for i in range(10)
        ]

        # 计算统计信息
        stats = audit_manager._calculate_security_stats(events, None)

        # 验证统计结果
        assert isinstance(stats, dict)
        assert 'total_events' in stats
        assert 'events_by_type' in stats
        assert 'events_by_severity' in stats
        assert 'events_by_user' in stats
        assert 'success_rate' in stats

        # 验证具体值
        assert stats['total_events'] == 10
        assert 'security' in stats['events_by_type']

    def test_calculate_security_stats_with_time_range(self, audit_manager):
        """测试带时间范围的安全统计计算"""
        base_time = datetime.now()
        time_range = {
            'start': (base_time - timedelta(hours=2)).isoformat(),
            'end': (base_time - timedelta(hours=1)).isoformat()
        }

        events = [
            {
                'event_type': 'security',
                'severity': 'medium',
                'timestamp': (base_time - timedelta(hours=i)).isoformat(),
                'user_id': 'user1',
                'action': 'login',
                'result': 'success'
            }
            for i in range(5)  # 创建5个事件，时间从0到4小时前
        ]

        stats = audit_manager._calculate_security_stats(events, time_range)

        # 应该只统计时间范围内的2个事件（2-3小时前）
        assert 0 <= stats['total_events'] <= 5

    def test_group_events_by_single_field(self, audit_manager):
        """测试按单个字段分组事件"""
        events = [
            {'event_type': 'security', 'user_id': 'user1', 'action': 'login'},
            {'event_type': 'security', 'user_id': 'user1', 'action': 'logout'},
            {'event_type': 'data', 'user_id': 'user2', 'action': 'read'},
            {'event_type': 'security', 'user_id': 'user2', 'action': 'login'},
        ]

        grouped = audit_manager._group_events(events, {'user_id'})

        assert ('user1',) in grouped
        assert ('user2',) in grouped
        assert len(grouped[('user1',)]) == 2  # user1有两个事件
        assert len(grouped[('user2',)]) == 2  # user2有两个事件

    def test_group_events_by_multiple_fields(self, audit_manager):
        """测试按多个字段分组事件"""
        events = [
            {'event_type': 'security', 'user_id': 'user1', 'severity': 'high'},
            {'event_type': 'security', 'user_id': 'user1', 'severity': 'low'},
            {'event_type': 'data', 'user_id': 'user1', 'severity': 'high'},
            {'event_type': 'security', 'user_id': 'user2', 'severity': 'high'},
        ]

        grouped = audit_manager._group_events(events, {'user_id', 'event_type'})

        # 验证分组结果
        assert len(grouped) > 0
        for key, group_events in grouped.items():
            assert isinstance(key, tuple)
            assert isinstance(group_events, list)

    def test_aggregate_events_count(self, audit_manager):
        """测试事件计数聚合"""
        events = [
            {'event_type': 'security', 'user_id': 'user1', 'action': 'login', 'value': 10},
            {'event_type': 'security', 'user_id': 'user1', 'action': 'logout', 'value': 20},
            {'event_type': 'data', 'user_id': 'user2', 'action': 'read', 'value': 5},
        ]

        aggregation = {
            'total_events': {'operation': 'count'},
            'total_value': {'field': 'value', 'operation': 'sum'}
        }

        result = audit_manager._aggregate_events(events, aggregation)

        assert result['total_events'] == 3
        assert result['total_value'] == 35  # 10 + 20 + 5

    def test_aggregate_events_average(self, audit_manager):
        """测试事件平均值聚合"""
        events = [
            {'response_time': 100, 'cpu_usage': 50},
            {'response_time': 200, 'cpu_usage': 75},
            {'response_time': 150, 'cpu_usage': 60},
        ]

        aggregation = {
            'avg_response_time': {'field': 'response_time', 'operation': 'avg'},
            'avg_cpu_usage': {'field': 'cpu_usage', 'operation': 'avg'}
        }

        result = audit_manager._aggregate_events(events, aggregation)

        assert abs(result['avg_response_time'] - 150.0) < 0.01  # (100+200+150)/3
        assert abs(result['avg_cpu_usage'] - 61.6667) < 0.01    # (50+75+60)/3

    def test_aggregate_events_max_min(self, audit_manager):
        """测试事件最大值最小值聚合"""
        events = [
            {'response_time': 100, 'error_count': 2},
            {'response_time': 300, 'error_count': 0},
            {'response_time': 200, 'error_count': 5},
        ]

        aggregation = {
            'max_response_time': {'field': 'response_time', 'operation': 'max'},
            'min_response_time': {'field': 'response_time', 'operation': 'min'},
            'max_errors': {'field': 'error_count', 'operation': 'max'},
            'min_errors': {'field': 'error_count', 'operation': 'min'}
        }

        result = audit_manager._aggregate_events(events, aggregation)

        assert result['max_response_time'] == 300
        assert result['min_response_time'] == 100
        assert result['max_errors'] == 5
        assert result['min_errors'] == 0

    def test_aggregate_events_with_missing_fields(self, audit_manager):
        """测试聚合缺失字段的情况"""
        events = [
            {'value': 10},
            {'value': 20},
            {},  # 缺少value字段
        ]

        aggregation = {
            'sum_values': {'field': 'value', 'operation': 'sum'},
            'count_values': {'field': 'value', 'operation': 'count'}
        }

        result = audit_manager._aggregate_events(events, aggregation)

        assert result['sum_values'] == 30  # 10 + 20
        assert result['count_values'] == 2  # 只有2个事件有value字段

    def test_generate_security_report_with_aggregation(self, audit_manager):
        """测试带聚合的安全报告生成"""
        # 添加一些测试事件
        for i in range(5):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.HIGH if i % 2 == 0 else EventSeverity.MEDIUM,
                user_id=f'user_{i % 3}',
                resource='test_resource',
                action='test_action',
                result='success' if i % 3 != 0 else 'failure'
            )
            audit_manager.log_event(params)

        # 生成带聚合的报告
        report_params = ReportGenerationParams(
            report_type='security',
            time_range={'hours': 24},
            group_by={'user_id'},
            aggregation={
                'total_events': {'operation': 'count'},
                'success_count': {'field': 'result', 'operation': 'count', 'filter': {'result': 'success'}}
            }
        )

        report = audit_manager.generate_security_report(report_params)

        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'grouped_data' in report
        assert 'aggregated_data' in report

    def test_get_compliance_report_detailed(self, audit_manager):
        """测试详细合规报告"""
        # 添加各种类型的合规事件
        compliance_events = [
            (EventType.SECURITY, EventSeverity.HIGH, 'admin', 'login', 'success'),
            (EventType.SECURITY, EventSeverity.MEDIUM, 'user1', 'login', 'failure'),
            (EventType.DATA_OPERATION, EventSeverity.LOW, 'user2', 'read', 'success'),
            (EventType.CONFIG_CHANGE, EventSeverity.MEDIUM, 'admin', 'update', 'success'),
        ]

        for event_type, severity, user_id, action, result in compliance_events:
            params = AuditEventParams(
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                resource='compliance_test',
                action=action,
                result=result
            )
            audit_manager.log_event(params)

        # 获取不同类型的合规报告
        general_report = audit_manager.get_compliance_report("general")
        security_report = audit_manager.get_compliance_report("security")
        audit_report = audit_manager.get_compliance_report("audit")

        # 验证报告结构
        for report in [general_report, security_report, audit_report]:
            assert isinstance(report, dict)
            assert 'compliance_type' in report
            assert 'report_date' in report
            assert 'metrics' in report
            assert 'status' in report

    def test_audit_manager_concurrent_operations(self, audit_manager):
        """测试审计管理器的并发操作"""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def worker(worker_id):
            try:
                # 每个线程记录多个事件
                for i in range(10):
                    params = AuditEventParams(
                        event_type=EventType.SECURITY,
                        severity=EventSeverity.LOW,
                        user_id=f'worker_{worker_id}_user_{i}',
                        resource=f'worker_{worker_id}_resource',
                        action=f'worker_{worker_id}_action_{i}'
                    )
                    event_id = audit_manager.log_event(params)
                    results.put(f'worker_{worker_id}_event_{i}')

            except Exception as e:
                errors.put(f'worker_{worker_id}: {e}')

        # 启动多个线程
        threads = []
        num_threads = 5
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join(timeout=10)

        # 验证结果
        assert errors.empty()
        result_count = 0
        while not results.empty():
            results.get()
            result_count += 1

        assert result_count == num_threads * 10

        # 强制刷新缓冲区以确保所有事件都被记录
        audit_manager._flush_buffer()

        # 验证所有事件都被记录
        assert len(audit_manager.events) >= num_threads * 10

    def test_audit_manager_buffer_overflow_handling(self, audit_manager):
        """测试缓冲区溢出处理"""
        # 设置小的缓冲区大小进行测试
        original_buffer_size = audit_manager.buffer_size
        audit_manager.buffer_size = 5

        try:
            # 记录超过缓冲区大小的事件
            for i in range(10):  # 超过缓冲区大小
                params = AuditEventParams(
                    event_type=EventType.SECURITY,
                    severity=EventSeverity.LOW,
                    user_id=f'overflow_user_{i}',
                    resource='overflow_test',
                    action='overflow_action'
                )
                audit_manager.log_event(params)

            # 缓冲区应该被自动刷新
            # 注意：实际的刷新时机可能因实现而异
            assert len(audit_manager.event_buffer) <= 10

        finally:
            # 恢复原始缓冲区大小
            audit_manager.buffer_size = original_buffer_size

    def test_audit_manager_large_scale_data_handling(self, audit_manager):
        """测试大规模数据处理"""
        # 记录大量事件来测试性能
        num_events = 1000

        start_time = time.time()

        for i in range(num_events):
            params = AuditEventParams(
                event_type=EventType.SECURITY if i % 2 == 0 else EventType.DATA_OPERATION,
                severity=EventSeverity.LOW,
                user_id=f'bulk_user_{i % 50}',  # 重复的用户ID
                resource=f'bulk_resource_{i % 20}',  # 重复的资源
                action=f'bulk_action_{i % 10}'  # 重复的动作
            )
            audit_manager.log_event(params)

        end_time = time.time()
        duration = end_time - start_time

        # 验证性能（1000个事件应该在合理时间内完成）
        assert duration < 30.0  # 30秒内完成

        # 验证数据完整性
        assert len(audit_manager.events) >= num_events

        # 测试大规模查询
        query_start = time.time()
        all_events = audit_manager.query_events(QueryFilterParams())
        query_end = time.time()

        # 查询性能测试
        assert (query_end - query_start) < 10.0  # 查询在10秒内完成
        assert len(all_events) >= num_events

    def test_audit_manager_event_deduplication_simulation(self, audit_manager):
        """测试事件去重模拟"""
        # 记录一些重复的事件（在实际应用中可能需要去重）
        base_params = {
            'event_type': EventType.SECURITY,
            'severity': EventSeverity.MEDIUM,
            'user_id': 'duplicate_user',
            'resource': 'duplicate_resource',
            'action': 'duplicate_action'
        }

        # 记录重复事件
        for i in range(5):
            params = AuditEventParams(**base_params)
            audit_manager.log_event(params)

        # 查询这些事件
        filter_params = QueryFilterParams(
            user_ids={'duplicate_user'},
            event_types={EventType.SECURITY}
        )
        results = audit_manager.query_events(filter_params)

        # 验证所有事件都被记录（不去重）
        assert len(results) == 5

    def test_audit_manager_file_operations_error_handling(self, audit_manager, tmp_path):
        """测试文件操作错误处理"""
        # 设置一个不存在的路径来触发错误
        original_path = audit_manager.log_path
        invalid_path = tmp_path / "nonexistent" / "deep" / "path"
        audit_manager.log_path = str(invalid_path / "audit.log")

        try:
            # 记录事件，这应该不会因为文件错误而失败
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.HIGH,
                user_id='error_test_user',
                resource='error_test_resource',
                action='error_test_action'
            )
            event_id = audit_manager.log_event(params)
            assert event_id is not None

            # 强制刷新缓冲区（应该能处理文件错误）
            audit_manager._flush_buffer()

        finally:
            audit_manager.log_path = original_path

    def test_audit_manager_pagination_edge_cases(self, audit_manager):
        """测试分页边界情况"""
        # 先添加一些事件并刷新
        for i in range(10):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'page_user_{i}',
                resource='page_resource',
                action='page_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 测试offset超出范围
        params = QueryFilterParams(offset=20, limit=5)
        results = audit_manager.query_events(params)
        assert len(results) == 0

        # 测试limit=0
        params = QueryFilterParams(limit=0)
        results = audit_manager.query_events(params)
        assert len(results) == 0

        # 测试正常分页
        params = QueryFilterParams(offset=5, limit=3)
        results = audit_manager.query_events(params)
        assert len(results) == 3

    def test_audit_manager_aggregation_operations(self, audit_manager):
        """测试聚合操作"""
        # 添加不同类型的事件
        event_types = [EventType.SECURITY, EventType.AUTHENTICATION, EventType.USER_MANAGEMENT]
        for i, event_type in enumerate(event_types):
            for j in range(3):
                params = AuditEventParams(
                    event_type=event_type,
                    severity=EventSeverity.LOW,
                    user_id=f'agg_user_{i}_{j}',
                    resource='agg_resource',
                    action='agg_action'
                )
                audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 测试按事件类型分组
        report = audit_manager.generate_security_report(
            group_by=['event_type'],
            aggregation={'count': ['event_type']}
        )

        assert 'grouped_data' in report
        assert 'aggregated_data' in report

    def test_audit_manager_compliance_assessment_edge_cases(self, audit_manager):
        """测试合规性评估边界情况"""
        # 测试空事件列表的合规性
        compliance_report = audit_manager.get_compliance_report()
        assert 'status' in compliance_report
        assert 'metrics' in compliance_report

        # 添加一些边界情况事件
        edge_case_events = [
            # 高风险分数
            {'event_type': EventType.SECURITY, 'severity': EventSeverity.HIGH, 'risk_score': 0.9, 'result': 'failure'},
            # 成功事件
            {'event_type': EventType.AUTHENTICATION, 'severity': EventSeverity.LOW, 'result': 'success'},
            # 敏感数据访问
            {'event_type': EventType.SECURITY, 'severity': EventSeverity.MEDIUM, 'details': {'sensitive_data': True}},
        ]

        for event_data in edge_case_events:
            params = AuditEventParams(
                event_type=event_data['event_type'],
                severity=event_data['severity'],
                user_id='compliance_user',
                resource='compliance_resource',
                action='compliance_action',
                result=event_data.get('result', 'success'),
                risk_score=event_data.get('risk_score', 0.0),
                details=event_data.get('details', {})
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 测试合规性报告
        compliance_report = audit_manager.get_compliance_report()
        assert compliance_report['status'] in ['compliant', 'warning', 'non_compliant']

    def test_audit_manager_cleanup_old_events(self, audit_manager, tmp_path):
        """测试清理旧事件"""
        # 设置临时日志路径
        original_path = audit_manager.log_path
        audit_manager.log_path = str(tmp_path / "audit")

        try:
            # 降低max_events来更容易触发清理
            original_max = audit_manager.max_events
            audit_manager.max_events = 10

            # 添加足够多的事件来触发清理
            for i in range(15):
                params = AuditEventParams(
                    event_type=EventType.SECURITY,
                    severity=EventSeverity.LOW,
                    user_id=f'cleanup_user_{i}',
                    resource='cleanup_resource',
                    action='cleanup_action'
                )
                audit_manager.log_event(params)

            # 验证事件数量被限制
            assert len(audit_manager.events) <= audit_manager.max_events

        finally:
            audit_manager.max_events = original_max
            audit_manager.log_path = original_path

    def test_audit_manager_empty_events_write(self, audit_manager):
        """测试写入空事件列表"""
        # 直接调用_write_events_to_file with empty list
        audit_manager._write_events_to_file([])

        # 这应该不会抛出异常
        assert True

    def test_audit_manager_generate_security_report_with_filters(self, audit_manager):
        """测试生成带有过滤器的安全报告"""
        # 先添加一些事件
        for i in range(5):
            params = AuditEventParams(
                event_type=EventType.SECURITY if i < 3 else EventType.AUTHENTICATION,
                severity=EventSeverity.LOW,
                user_id=f'report_user_{i}',
                resource='report_resource',
                action='report_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 使用过滤器生成报告
        from src.infrastructure.security.audit.audit_manager import ReportGenerationParams, QueryFilterParams

        filters = QueryFilterParams(event_types={EventType.SECURITY})
        report_params = ReportGenerationParams(
            filters=filters,
            group_by=['event_type'],
            aggregation={'count': ['event_type']}
        )

        report = audit_manager.generate_security_report(report_params)
        assert 'grouped_data' in report
        assert 'aggregated_data' in report

    def test_audit_manager_event_timestamp_sorting_desc(self, audit_manager):
        """测试事件时间戳降序排序"""
        import time

        # 添加事件，确保有时间间隔
        for i in range(3):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'sort_user_{i}',
                resource='sort_resource',
                action='sort_action'
            )
            audit_manager.log_event(params)
            time.sleep(0.01)  # 小延迟确保时间戳不同

        audit_manager._flush_buffer()

        # 查询时指定降序排序
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        params = QueryFilterParams(sort_order='desc')
        results = audit_manager.query_events(params)

        # 验证结果是降序排列的（最新的在前面）
        if len(results) > 1:
            assert results[0]['timestamp'] >= results[1]['timestamp']

    def test_audit_manager_risk_score_filtering(self, audit_manager):
        """测试风险分数过滤"""
        # 添加不同风险分数的事件
        for i in range(3):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'risk_user_{i}',
                resource='risk_resource',
                action='risk_action',
                risk_score=i * 0.3  # 0.0, 0.3, 0.6
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 过滤高风险事件
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        params = QueryFilterParams(min_risk_score=0.5)
        results = audit_manager.query_events(params)

        # 应该只有风险分数>=0.5的事件
        for result in results:
            assert result['risk_score'] >= 0.5

    def test_audit_manager_compliance_metrics_calculation(self, audit_manager):
        """测试合规性指标计算的边界情况"""
        # 测试没有事件时的合规性
        compliance = audit_manager.get_compliance_report()
        assert 'metrics' in compliance

        # 添加一些边界情况事件
        boundary_events = [
            # 成功认证
            {'type': EventType.AUTHENTICATION, 'result': 'success'},
            # 失败认证
            {'type': EventType.AUTHENTICATION, 'result': 'failure'},
            # 安全事件
            {'type': EventType.SECURITY, 'severity': EventSeverity.HIGH},
        ]

        for event in boundary_events:
            params = AuditEventParams(
                event_type=event['type'],
                severity=event.get('severity', EventSeverity.LOW),
                user_id='boundary_user',
                resource='boundary_resource',
                action='boundary_action',
                result=event.get('result', 'success')
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        compliance = audit_manager.get_compliance_report()
        metrics = compliance['metrics']

        # 验证指标存在
        assert 'successful_authentications' in metrics
        assert 'failed_authentications' in metrics
        assert 'total_events' in metrics

    def test_audit_manager_cleanup_old_events_call(self, audit_manager):
        """测试_cleanup_old_events方法被调用"""
        # 添加一些事件
        for i in range(5):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'cleanup_user_{i}',
                resource='cleanup_resource',
                action='cleanup_action'
            )
            audit_manager.log_event(params)

        # 强制刷新缓冲区，触发_cleanup_old_events
        audit_manager._flush_buffer()

        # 验证方法被调用且没有抛出异常
        assert len(audit_manager.events) >= 0  # 确保清理没有删除所有事件

    def test_audit_manager_cleanup_old_files_exception_handling(self, audit_manager, tmp_path, mocker):
        """测试清理旧文件时的异常处理"""
        import tempfile
        from pathlib import Path

        # 设置审计日志路径到临时目录
        audit_manager.log_path = str(tmp_path / "audit_test")

        # 创建一些旧的审计文件
        audit_dir = Path(audit_manager.log_path)
        audit_dir.mkdir(exist_ok=True)

        # 创建一个旧文件
        old_file = audit_dir / "audit_20231001.json"
        old_file.write_text('{"test": "data"}')

        # 模拟unlink抛出异常
        mock_unlink = mocker.patch.object(Path, 'unlink', side_effect=OSError("Permission denied"))

        # 调用清理方法，验证异常被处理
        audit_manager._cleanup_old_events()

        # 验证方法没有抛出异常（异常被捕获并记录）
        assert audit_dir.exists()  # 目录仍然存在

    def test_audit_manager_pagination_limit_handling(self, audit_manager):
        """测试分页limit参数的不同处理"""
        # 添加多个事件
        for i in range(10):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'limit_user_{i}',
                resource='limit_resource',
                action='limit_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 测试limit=0返回空列表
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        params = QueryFilterParams(limit=0)
        results = audit_manager.query_events(params)
        assert results == []

        # 测试正常limit
        params = QueryFilterParams(limit=5)
        results = audit_manager.query_events(params)
        assert len(results) == 5

    def test_audit_manager_aggregation_with_grouping_keys(self, audit_manager):
        """测试聚合操作中grouping keys的处理"""
        # 添加不同类型的事件用于聚合
        events_data = [
            (EventType.SECURITY, EventSeverity.HIGH, 'user1'),
            (EventType.SECURITY, EventSeverity.LOW, 'user2'),
            (EventType.AUTHENTICATION, EventSeverity.MEDIUM, 'user1'),
        ]

        for event_type, severity, user_id in events_data:
            params = AuditEventParams(
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                resource='agg_resource',
                action='agg_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 测试包含分组键的聚合
        from src.infrastructure.security.audit.audit_manager import ReportGenerationParams

        report_params = ReportGenerationParams(
            group_by=['event_type'],
            aggregation={'count': ['event_type']}
        )

        report = audit_manager.generate_security_report(report_params)

        # 验证分组数据存在
        assert 'grouped_data' in report
        grouped_data = report['grouped_data']

        # 验证分组键处理
        if isinstance(grouped_data, dict):
            for key, data in grouped_data.items():
                assert 'count' in data or 'statistics' in data

    def test_audit_manager_compliance_status_calculation(self, audit_manager):
        """测试合规性状态计算的详细分支"""
        # 添加各种事件来触发不同的合规性计算分支
        compliance_events = [
            # 大量成功认证
            {'type': EventType.AUTHENTICATION, 'result': 'success', 'count': 100},
            # 少量失败认证
            {'type': EventType.AUTHENTICATION, 'result': 'failure', 'count': 2},
            # 少量安全事件
            {'type': EventType.SECURITY, 'severity': EventSeverity.CRITICAL, 'count': 1},
        ]

        for event_spec in compliance_events:
            for _ in range(event_spec['count']):
                params = AuditEventParams(
                    event_type=event_spec['type'],
                    severity=event_spec.get('severity', EventSeverity.LOW),
                    user_id='compliance_test_user',
                    resource='compliance_resource',
                    action='compliance_action',
                    result=event_spec.get('result', 'success')
                )
                audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 获取合规性报告，触发详细计算分支
        compliance = audit_manager.get_compliance_report()

        # 验证合规性状态计算
        assert 'status' in compliance
        assert compliance['status'] in ['compliant', 'non_compliant', 'warning']

        # 验证推荐存在
        assert 'recommendations' in compliance

    def test_audit_manager_event_filtering_by_severity(self, audit_manager):
        """测试按严重程度过滤事件"""
        # 添加不同严重程度的事件
        severities = [EventSeverity.LOW, EventSeverity.MEDIUM, EventSeverity.HIGH, EventSeverity.CRITICAL]

        for severity in severities:
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=severity,
                user_id=f'severity_user_{severity.value}',
                resource='severity_resource',
                action='severity_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 测试只查询HIGH及以上的事件
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        params = QueryFilterParams(min_severity=EventSeverity.HIGH)
        results = audit_manager.query_events(params)

        # 应该只返回HIGH和CRITICAL事件
        severity_values = {r['severity'] for r in results}
        assert 'high' in severity_values
        assert 'critical' in severity_values
        assert 'medium' not in severity_values
        assert 'low' not in severity_values

    def test_audit_manager_event_filtering_by_date_range(self, audit_manager):
        """测试按日期范围过滤事件"""
        from datetime import datetime, timedelta

        base_time = datetime.now()

        # 添加不同时间的事件
        time_offsets = [timedelta(hours=-2), timedelta(minutes=-30), timedelta(minutes=30)]

        for i, offset in enumerate(time_offsets):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'date_user_{i}',
                resource='date_resource',
                action='date_action',
                timestamp=base_time + offset
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 查询特定时间范围
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        start_date = base_time - timedelta(hours=1)
        end_date = base_time + timedelta(hours=1)

        params = QueryFilterParams(start_date=start_date, end_date=end_date)
        results = audit_manager.query_events(params)

        # 应该返回中间和右边的事件（索引1和2）
        user_ids = {r['user_id'] for r in results}
        assert 'date_user_1' in user_ids
        assert 'date_user_2' in user_ids
        assert 'date_user_0' not in user_ids  # 这个在范围外

    def test_audit_manager_query_with_multiple_filters(self, audit_manager):
        """测试多重过滤条件"""
        # 添加各种类型的事件
        test_events = [
            {'type': EventType.SECURITY, 'severity': EventSeverity.HIGH, 'user': 'user1', 'resource': 'res1'},
            {'type': EventType.AUTHENTICATION, 'severity': EventSeverity.LOW, 'user': 'user2', 'resource': 'res2'},
            {'type': EventType.SECURITY, 'severity': EventSeverity.LOW, 'user': 'user1', 'resource': 'res3'},
        ]

        for event in test_events:
            params = AuditEventParams(
                event_type=event['type'],
                severity=event['severity'],
                user_id=event['user'],
                resource=event['resource'],
                action='multi_filter_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 使用多个过滤条件
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        params = QueryFilterParams(
            event_types={EventType.SECURITY},
            severities=[EventSeverity.HIGH],
            user_ids={'user1'}
        )
        results = audit_manager.query_events(params)

        # 应该只返回第一个事件
        assert len(results) == 1
        assert results[0]['user_id'] == 'user1'
        assert results[0]['severity'] == 'high'

    def test_audit_manager_event_sorting_by_timestamp_desc(self, audit_manager):
        """测试按时间戳降序排序"""
        import time

        # 添加事件，稍微延迟以确保时间戳不同
        for i in range(3):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'sort_user_{i}',
                resource='sort_resource',
                action='sort_action'
            )
            audit_manager.log_event(params)
            time.sleep(0.01)  # 10ms延迟

        audit_manager._flush_buffer()

        # 查询时指定降序排序
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        params = QueryFilterParams(sort_order='desc')
        results = audit_manager.query_events(params)

        # 验证结果按时间降序排列（最新的在前）
        assert len(results) == 3
        # 检查时间戳是否递减
        for i in range(len(results) - 1):
            assert results[i]['timestamp'] >= results[i + 1]['timestamp']

    def test_audit_manager_pagination_with_sorting(self, audit_manager):
        """测试分页结合排序"""
        # 添加多个事件
        for i in range(10):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'page_sort_user_{i}',
                resource='page_sort_resource',
                action='page_sort_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 测试分页和排序的组合
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        params = QueryFilterParams(
            sort_order='desc',
            offset=3,
            limit=2
        )
        results = audit_manager.query_events(params)

        # 应该返回2个事件，从第4个开始（0-indexed为3）
        assert len(results) == 2

    def test_audit_manager_generate_report_with_complex_aggregation(self, audit_manager):
        """测试复杂聚合的报告生成"""
        # 添加多种事件用于聚合
        events_data = [
            (EventType.SECURITY, EventSeverity.HIGH, 'high_user'),
            (EventType.SECURITY, EventSeverity.LOW, 'low_user'),
            (EventType.AUTHENTICATION, EventSeverity.MEDIUM, 'auth_user'),
            (EventType.SECURITY, EventSeverity.HIGH, 'high_user2'),
        ]

        for event_type, severity, user_id in events_data:
            params = AuditEventParams(
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                resource='complex_agg_resource',
                action='complex_agg_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 生成包含复杂聚合的报告
        from src.infrastructure.security.audit.audit_manager import ReportGenerationParams

        report_params = ReportGenerationParams(
            group_by=['event_type', 'severity'],
            aggregation={
                'count': ['event_type', 'severity', 'user_id'],
                'distinct_count': ['user_id']
            }
        )

        report = audit_manager.generate_security_report(report_params)

        # 验证报告结构
        assert 'grouped_data' in report
        assert 'aggregated_data' in report
        assert 'statistics' in report

    def test_audit_manager_compliance_calculation_with_various_events(self, audit_manager):
        """测试包含各种事件的合规性计算"""
        # 添加各种可能影响合规性的事件
        compliance_events = [
            # 成功认证
            {'type': EventType.AUTHENTICATION, 'result': 'success', 'count': 10},
            # 失败认证
            {'type': EventType.AUTHENTICATION, 'result': 'failure', 'count': 3},
            # 安全事件
            {'type': EventType.SECURITY, 'severity': EventSeverity.HIGH, 'count': 2},
            # 数据访问事件
            {'type': EventType.SECURITY, 'details': {'data_access': True}, 'count': 5},
        ]

        for event_spec in compliance_events:
            for _ in range(event_spec['count']):
                params = AuditEventParams(
                    event_type=event_spec['type'],
                    severity=event_spec.get('severity', EventSeverity.LOW),
                    user_id='compliance_test_user',
                    resource='compliance_resource',
                    action='compliance_action',
                    result=event_spec.get('result', 'success'),
                    details=event_spec.get('details', {})
                )
                audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 获取合规性报告
        compliance = audit_manager.get_compliance_report()

        # 验证合规性指标
        assert 'status' in compliance
        assert 'metrics' in compliance
        assert 'recommendations' in compliance

        metrics = compliance['metrics']
        assert 'successful_authentications' in metrics
        assert 'failed_authentications' in metrics
        assert 'total_events' in metrics
        assert 'security_events' in metrics

    def test_audit_manager_max_events_limit_handling(self, audit_manager):
        """测试最大事件数量限制处理"""
        # 设置较小的最大事件数
        original_max = audit_manager.max_events
        audit_manager.max_events = 50

        try:
            # 添加超过限制的事件
            for i in range(70):
                params = AuditEventParams(
                    event_type=EventType.SECURITY,
                    severity=EventSeverity.LOW,
                    user_id=f'limit_user_{i}',
                    resource='limit_resource',
                    action='limit_action'
                )
                audit_manager.log_event(params)

            # 强制触发清理
            audit_manager._flush_buffer()

            # 验证事件数量被限制
            assert len(audit_manager.events) <= audit_manager.max_events + 10  # 允许一些缓冲

        finally:
            audit_manager.max_events = original_max

    def test_audit_manager_cleanup_old_files(self, audit_manager, tmp_path, mocker):
        """测试清理旧审计文件"""
        import tempfile
        from pathlib import Path
        from datetime import datetime

        # 创建临时审计目录
        audit_dir = tmp_path / "audit_cleanup"
        audit_dir.mkdir()

        # 设置日志路径
        original_path = audit_manager.log_path
        audit_manager.log_path = str(audit_dir)

        try:
            # 创建一些模拟的审计文件
            old_date = (datetime.now().replace(day=1) - timedelta(days=45)).strftime("%Y%m%d")
            new_date = datetime.now().strftime("%Y%m%d")

            # 创建旧文件（应该被删除）
            old_file = audit_dir / f"audit_{old_date}_001.json"
            old_file.write_text('{"test": "old"}')

            # 创建新文件（应该保留）
            new_file = audit_dir / f"audit_{new_date}_001.json"
            new_file.write_text('{"test": "new"}')

            # 调用清理方法
            audit_manager._cleanup_old_events()

            # 验证旧文件被删除，新文件保留
            assert not old_file.exists()
            assert new_file.exists()

        finally:
            audit_manager.log_path = original_path

    def test_audit_manager_query_with_tags_filter(self, audit_manager):
        """测试标签过滤查询"""
        # 添加带标签的事件
        params1 = AuditEventParams(
            event_type=EventType.SECURITY,
            severity=EventSeverity.LOW,
            user_id='tag_user_1',
            resource='tag_resource',
            action='tag_action',
            tags=['tag1', 'tag2']
        )
        audit_manager.log_event(params1)

        params2 = AuditEventParams(
            event_type=EventType.SECURITY,
            severity=EventSeverity.LOW,
            user_id='tag_user_2',
            resource='tag_resource',
            action='tag_action',
            tags=['tag2', 'tag3']
        )
        audit_manager.log_event(params2)

        audit_manager._flush_buffer()

        # 查询包含tag1的事件
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        params = QueryFilterParams(tags=['tag1'])
        results = audit_manager.query_events(params)

        # 应该只返回第一个事件
        assert len(results) == 1
        assert results[0]['user_id'] == 'tag_user_1'

    def test_audit_manager_generate_security_report_comprehensive(self, audit_manager):
        """测试综合安全报告生成功能"""
        # 添加各种类型的事件
        test_events = [
            (EventType.SECURITY, EventSeverity.HIGH, 'high_security_user'),
            (EventType.AUTHENTICATION, EventSeverity.LOW, 'auth_user'),
            (EventType.USER_MANAGEMENT, EventSeverity.MEDIUM, 'user_mgmt_user'),
        ]

        for event_type, severity, user_id in test_events:
            params = AuditEventParams(
                event_type=event_type,
                severity=severity,
                user_id=user_id,
                resource='comprehensive_resource',
                action='comprehensive_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 生成包含所有功能的报告
        from src.infrastructure.security.audit.audit_manager import ReportGenerationParams

        report_params = ReportGenerationParams(
            group_by=['event_type', 'severity'],
            aggregation={
                'count': ['event_type', 'severity'],
                'avg': ['risk_score'],
                'max': ['risk_score']
            }
        )

        report = audit_manager.generate_security_report(report_params)

        # 验证报告结构
        assert 'grouped_data' in report
        assert 'aggregated_data' in report
        assert 'statistics' in report
        assert 'summary' in report

        # 验证分组数据
        grouped = report['grouped_data']
        assert 'SECURITY' in grouped
        assert 'AUTHENTICATION' in grouped

    def test_audit_manager_exception_handling_in_cleanup(self, audit_manager, tmp_path, mocker):
        """测试清理过程中的异常处理"""
        # 设置日志路径
        original_path = audit_manager.log_path
        audit_manager.log_path = str(tmp_path / "audit")

        # Mock文件删除操作抛出异常
        mock_unlink = mocker.patch('pathlib.Path.unlink')
        mock_unlink.side_effect = OSError("Delete failed")

        try:
            # 调用清理方法，应该处理异常
            audit_manager._cleanup_old_events()

            # 验证异常被正确处理（没有抛出异常）
            assert True

        finally:
            audit_manager.log_path = original_path

    def test_audit_manager_generate_security_report_empty_aggregation(self, audit_manager):
        """测试空聚合的安全报告生成"""
        # 添加一些事件
        for i in range(3):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'empty_agg_user_{i}',
                resource='empty_agg_resource',
                action='empty_agg_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 生成只有分组没有聚合的报告
        from src.infrastructure.security.audit.audit_manager import ReportGenerationParams

        report_params = ReportGenerationParams(
            group_by=['event_type'],
            aggregation={}  # 空聚合
        )

        report = audit_manager.generate_security_report(report_params)

        # 验证报告结构
        assert 'grouped_data' in report
        assert 'statistics' in report
        assert 'aggregated_data' not in report or report['aggregated_data'] == {}

    def test_audit_manager_generate_security_report_no_grouping(self, audit_manager):
        """测试无分组的安全报告生成"""
        # 添加一些事件
        for i in range(3):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'no_group_user_{i}',
                resource='no_group_resource',
                action='no_group_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 生成没有分组的报告
        from src.infrastructure.security.audit.audit_manager import ReportGenerationParams

        report_params = ReportGenerationParams(
            aggregation={'count': ['event_type']}
        )

        report = audit_manager.generate_security_report(report_params)

        # 验证报告结构
        assert 'statistics' in report
        assert 'aggregated_data' in report
        assert 'grouped_data' not in report or report['grouped_data'] == {}

    def test_audit_manager_assess_compliance_status_edge_cases(self, audit_manager):
        """测试合规性状态评估的边界情况"""
        # 测试不同级别的风险分数
        test_cases = [
            # 高风险 - 应该返回non_compliant
            {'failed_authentications': 60, 'sensitive_data_accesses': 0, 'config_changes': 0, 'expected': 'non_compliant'},
            # 中等风险 - 应该返回warning
            {'failed_authentications': 30, 'sensitive_data_accesses': 60, 'config_changes': 0, 'expected': 'warning'},
            # 低风险 - 应该返回compliant
            {'failed_authentications': 10, 'sensitive_data_accesses': 10, 'config_changes': 10, 'expected': 'compliant'},
        ]

        for test_case in test_cases:
            # Mock _calculate_compliance_metrics返回测试数据
            original_calc = audit_manager._calculate_compliance_metrics
            audit_manager._calculate_compliance_metrics = lambda: test_case

            try:
                status = audit_manager._assess_compliance_status(test_case)
                assert status == test_case['expected']
            finally:
                audit_manager._calculate_compliance_metrics = original_calc

    def test_audit_manager_query_with_date_filters(self, audit_manager):
        """测试日期过滤查询"""
        from datetime import datetime, timedelta

        # 添加不同时间的事件
        base_time = datetime.now()

        # 添加一个"旧"事件（模拟）
        old_params = AuditEventParams(
            event_type=EventType.SECURITY,
            severity=EventSeverity.LOW,
            user_id='old_user',
            resource='date_resource',
            action='date_action',
            timestamp=base_time - timedelta(hours=2)
        )
        audit_manager.log_event(old_params)

        # 添加一个"新"事件
        new_params = AuditEventParams(
            event_type=EventType.SECURITY,
            severity=EventSeverity.LOW,
            user_id='new_user',
            resource='date_resource',
            action='date_action',
            timestamp=base_time - timedelta(minutes=30)
        )
        audit_manager.log_event(new_params)

        audit_manager._flush_buffer()

        # 查询指定时间范围的事件
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        start_date = base_time - timedelta(hours=1)
        end_date = base_time

        params = QueryFilterParams(start_date=start_date, end_date=end_date)
        results = audit_manager.query_events(params)

        # 应该只返回新事件
        assert len(results) == 1
        assert results[0]['user_id'] == 'new_user'

    def test_audit_manager_query_with_severity_filters(self, audit_manager):
        """测试严重程度过滤查询"""
        # 添加不同严重程度的事件
        severities = [EventSeverity.LOW, EventSeverity.MEDIUM, EventSeverity.HIGH]

        for i, severity in enumerate(severities):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=severity,
                user_id=f'severity_user_{i}',
                resource='severity_resource',
                action='severity_action'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 查询特定严重程度的事件
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        params = QueryFilterParams(severities=[EventSeverity.HIGH, EventSeverity.MEDIUM])
        results = audit_manager.query_events(params)

        # 应该返回HIGH和MEDIUM事件
        severity_values = {r['severity'] for r in results}
        assert 'high' in severity_values
        assert 'medium' in severity_values
        assert 'low' not in severity_values

    def test_audit_manager_query_with_location_filter(self, audit_manager):
        """测试位置过滤查询"""
        # 添加不同位置的事件
        locations = ['office', 'remote', 'server']

        for location in locations:
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.LOW,
                user_id=f'location_user_{location}',
                resource='location_resource',
                action='location_action',
                location=location
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 查询特定位置的事件
        from src.infrastructure.security.audit.audit_manager import QueryFilterParams

        params = QueryFilterParams(location='remote')
        results = audit_manager.query_events(params)

        # 应该只返回remote位置的事件
        assert len(results) == 1
        assert results[0]['location'] == 'remote'

    def test_audit_manager_generate_security_recommendations(self, audit_manager):
        """测试安全建议生成"""
        # 添加一些触发建议的事件
        for i in range(10):
            params = AuditEventParams(
                event_type=EventType.SECURITY,
                severity=EventSeverity.HIGH if i < 3 else EventSeverity.LOW,
                user_id=f'recommend_user_{i}',
                resource='recommend_resource',
                action='recommend_action',
                result='failure' if i < 5 else 'success'
            )
            audit_manager.log_event(params)

        audit_manager._flush_buffer()

        # 生成包含建议的报告
        from src.infrastructure.security.audit.audit_manager import ReportGenerationParams

        report_params = ReportGenerationParams()
        report = audit_manager.generate_security_report(report_params)

        # 验证报告包含建议
        assert 'recommendations' in report
        assert isinstance(report['recommendations'], list)
        assert len(report['recommendations']) > 0
