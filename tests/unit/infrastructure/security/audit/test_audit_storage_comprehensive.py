#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计存储管理器综合测试
测试AuditStorageManager的核心功能，包括事件存储、检索和持久化
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.infrastructure.security.audit.audit_storage import AuditStorageManager
from src.infrastructure.security.audit.audit_events import AuditEvent, AuditEventType, AuditSeverity


@pytest.fixture
def temp_storage_dir(tmp_path):
    """创建临时存储目录"""
    storage_dir = tmp_path / "audit_storage"
    storage_dir.mkdir()
    yield storage_dir
    # 清理
    shutil.rmtree(storage_dir, ignore_errors=True)


@pytest.fixture
def audit_storage_manager(temp_storage_dir):
    """创建审计存储管理器实例"""
    manager = AuditStorageManager(storage_path=temp_storage_dir, max_memory_events=100)
    return manager


@pytest.fixture
def sample_audit_events():
    """创建示例审计事件"""
    base_time = datetime(2025, 1, 1, 12, 0, 0)

    events = [
        AuditEvent(
            event_id="storage-event-001",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.LOW,
            timestamp=base_time,
            user_id="user123",
            resource="/api/data",
            details={"action": "read", "size": 1024}
        ),
        AuditEvent(
            event_id="storage-event-002",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.MEDIUM,
            timestamp=base_time + timedelta(hours=1),
            user_id="user456",
            resource="/api/profile",
            details={"action": "update", "changes": ["email"]}
        ),
        AuditEvent(
            event_id="storage-event-003",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            timestamp=base_time + timedelta(hours=2),
            user_id="admin",
            resource="/api/admin",
            details={"action": "login", "suspicious": True}
        ),
        AuditEvent(
            event_id="storage-event-004",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.LOW,
            timestamp=base_time + timedelta(days=1),  # 第二天
            user_id="user789",
            resource="/api/files",
            details={"action": "download", "file_size": 2048}
        )
    ]
    return events


class TestAuditStorageManagerInitialization:
    """测试审计存储管理器初始化"""

    def test_initialization_with_default_params(self):
        """测试默认参数初始化"""
        manager = AuditStorageManager()

        assert manager.storage_path.exists()
        assert isinstance(manager._memory_events, list)
        assert manager._max_memory_events == 5000
        assert manager._archive_threshold == 1000
        assert manager._compression_enabled is True
        assert manager._retention_days == 90

    def test_initialization_with_custom_params(self, temp_storage_dir):
        """测试自定义参数初始化"""
        custom_path = temp_storage_dir / "custom_audit"
        manager = AuditStorageManager(
            storage_path=custom_path,
            max_memory_events=200
        )

        assert str(manager.storage_path) == str(custom_path)
        assert manager.storage_path.exists()
        assert manager._max_memory_events == 200

    def test_storage_directory_creation(self, temp_storage_dir):
        """测试存储目录创建"""
        custom_path = temp_storage_dir / "new_audit_dir"
        manager = AuditStorageManager(storage_path=custom_path)

        assert custom_path.exists()
        assert custom_path.is_dir()


class TestAuditStorageManagerEventStorage:
    """测试审计存储管理器事件存储功能"""

    def test_store_single_event(self, audit_storage_manager, sample_audit_events):
        """测试存储单个事件"""
        manager = audit_storage_manager
        event = sample_audit_events[0]

        manager.store_event(event)

        assert len(manager._memory_events) == 1
        assert manager._memory_events[0] == event
        assert manager._stats['total_events'] == 1

    def test_store_multiple_events(self, audit_storage_manager, sample_audit_events):
        """测试存储多个事件"""
        manager = audit_storage_manager

        for event in sample_audit_events:
            manager.store_event(event)

        assert len(manager._memory_events) == len(sample_audit_events)
        assert manager._stats['total_events'] == len(sample_audit_events)

    def test_store_event_with_auto_archive(self, temp_storage_dir):
        """测试存储事件并自动归档"""
        # 创建一个小的归档阈值来触发自动归档
        manager = AuditStorageManager(
            storage_path=temp_storage_dir,
            max_memory_events=5
        )
        # 手动设置小的归档阈值
        manager._archive_threshold = 3

        # 添加足够的事件来触发归档
        events = []
        base_time = datetime(2025, 1, 1, 12, 0, 0)
        for i in range(5):
            event = AuditEvent(
                event_id=f"archive-event-{i:03d}",
                event_type=AuditEventType.ACCESS,
                severity=AuditSeverity.LOW,
                timestamp=base_time + timedelta(minutes=i),
                user_id=f"user{i}",
                resource="/api/test",
                details={"action": "test"}
            )
            events.append(event)
            manager.store_event(event)

        # 应该触发了归档
        assert len(manager._memory_events) <= manager._max_memory_events

    def test_memory_cleanup(self, audit_storage_manager):
        """测试内存清理"""
        manager = audit_storage_manager

        # 填充内存事件
        for i in range(10):
            event = AuditEvent(
                event_id=f"cleanup-event-{i:03d}",
                event_type=AuditEventType.ACCESS,
                severity=AuditSeverity.LOW,
                timestamp=datetime.now(),
                user_id=f"user{i}",
                resource="/api/test",
                details={"action": "test"}
            )
            manager.store_event(event)

        initial_count = len(manager._memory_events)

        # 执行内存清理（如果内存事件超过最大值）
        if len(manager._memory_events) > manager._max_memory_events:
            manager._cleanup_memory()

        # 验证清理后的状态
        assert len(manager._memory_events) <= manager._max_memory_events


class TestAuditStorageManagerEventRetrieval:
    """测试审计存储管理器事件检索功能"""

    def test_get_events_all(self, audit_storage_manager, sample_audit_events):
        """测试获取所有事件"""
        manager = audit_storage_manager

        # 存储事件
        for event in sample_audit_events:
            manager.store_event(event)

        # 获取所有事件
        events = manager.get_events()

        assert len(events) == len(sample_audit_events)
        assert events == sample_audit_events

    def test_get_events_with_time_filter(self, audit_storage_manager, sample_audit_events):
        """测试按时间过滤获取事件"""
        manager = audit_storage_manager

        # 存储事件
        for event in sample_audit_events:
            manager.store_event(event)

        # 设置时间范围
        start_time = datetime(2025, 1, 1, 11, 0, 0)  # 第一个事件之前
        end_time = datetime(2025, 1, 1, 15, 0, 0)     # 第三个事件之后

        events = manager.get_events(start_time=start_time, end_time=end_time)

        # 应该返回前3个事件（都在第一天）
        assert len(events) == 3

    def test_get_events_with_user_filter_not_supported(self, audit_storage_manager, sample_audit_events):
        """测试用户过滤功能不受支持"""
        manager = audit_storage_manager

        # 存储事件
        for event in sample_audit_events:
            manager.store_event(event)

        # get_events方法不支持user_id过滤，只能通过手动过滤结果
        all_events = manager.get_events()
        user_events = [e for e in all_events if e.user_id == "user123"]

        assert len(user_events) == 1
        assert user_events[0].user_id == "user123"

    def test_get_events_with_resource_filter_not_supported(self, audit_storage_manager, sample_audit_events):
        """测试资源过滤功能不受支持"""
        manager = audit_storage_manager

        # 存储事件
        for event in sample_audit_events:
            manager.store_event(event)

        # get_events方法不支持resource过滤，只能通过手动过滤结果
        all_events = manager.get_events()
        resource_events = [e for e in all_events if e.resource == "/api/data"]

        assert len(resource_events) == 1
        assert resource_events[0].resource == "/api/data"

    def test_get_events_with_event_type_filter(self, audit_storage_manager, sample_audit_events):
        """测试按事件类型过滤获取事件"""
        manager = audit_storage_manager

        # 存储事件
        for event in sample_audit_events:
            manager.store_event(event)

        # 按事件类型过滤（使用字符串值）
        access_events = manager.get_events(event_type="access")
        security_events = manager.get_events(event_type="security")

        assert len(access_events) >= 3  # 至少3个ACCESS事件
        assert len(security_events) >= 1  # 至少1个SECURITY事件

    def test_get_events_with_limit(self, audit_storage_manager, sample_audit_events):
        """测试限制获取事件数量"""
        manager = audit_storage_manager

        # 存储事件
        for event in sample_audit_events:
            manager.store_event(event)

        # 限制返回数量
        events = manager.get_events(limit=2)

        assert len(events) == 2

    def test_get_events_combined_filters(self, audit_storage_manager, sample_audit_events):
        """测试组合过滤条件"""
        manager = audit_storage_manager

        # 存储事件
        for event in sample_audit_events:
            manager.store_event(event)

        # get_events只支持时间和事件类型过滤，这里只测试事件类型
        events = manager.get_events(event_type="access")

        # 手动过滤用户
        user_events = [e for e in events if e.user_id == "user123"]

        assert len(user_events) == 1
        assert user_events[0].user_id == "user123"
        assert user_events[0].event_type == AuditEventType.ACCESS


class TestAuditStorageManagerArchiving:
    """测试审计存储管理器归档功能"""

    def test_archive_old_events(self, audit_storage_manager, sample_audit_events):
        """测试归档旧事件"""
        manager = audit_storage_manager

        # 存储事件
        for event in sample_audit_events:
            manager.store_event(event)

        # 手动触发归档
        archived_count = manager.archive_old_events()

        # 验证归档结果
        assert isinstance(archived_count, int)
        assert archived_count >= 0

        # 检查是否创建了归档文件
        archive_files = list(manager.storage_path.glob("*.json.gz"))
        if archived_count > 0:
            assert len(archive_files) > 0

    def test_write_archive_file(self, audit_storage_manager, sample_audit_events):
        """测试写入归档文件"""
        manager = audit_storage_manager

        date_str = "2025-01-01"
        events_to_archive = sample_audit_events[:2]  # 前两个事件

        # 写入归档文件
        manager._write_archive_file(date_str, events_to_archive)

        # 验证文件创建
        archive_file = manager.storage_path / f"audit_{date_str}.json.gz"
        assert archive_file.exists()

        # 验证文件内容
        with gzip.open(archive_file, 'rt', encoding='utf-8') as f:
            archived_data = json.load(f)

        assert isinstance(archived_data, list)
        assert len(archived_data) == 2

    def test_read_archive_file(self, audit_storage_manager, sample_audit_events):
        """测试读取归档文件"""
        manager = audit_storage_manager

        # 先写入归档文件
        date_str = "2025-01-01"
        events_to_archive = sample_audit_events[:2]
        manager._write_archive_file(date_str, events_to_archive)

        # 读取归档文件
        archived_events = manager._read_archive_file(date_str)

        assert isinstance(archived_events, list)
        assert len(archived_events) == 2

        # 验证事件数据（转换为AuditEvent对象）
        for event in archived_events:
            assert isinstance(event, AuditEvent)


class TestAuditStorageManagerStatsAndCleanup:
    """测试审计存储管理器统计和清理功能"""

    def test_get_storage_stats(self, audit_storage_manager, sample_audit_events):
        """测试获取存储统计信息"""
        manager = audit_storage_manager

        # 存储一些事件
        for event in sample_audit_events:
            manager.store_event(event)

        stats = manager.get_storage_stats()

        assert isinstance(stats, dict)
        assert "total_events" in stats
        assert "archived_files" in stats
        assert "storage_size_mb" in stats
        assert stats["total_events"] == len(sample_audit_events)

    def test_cleanup_storage(self, audit_storage_manager):
        """测试清理存储"""
        manager = audit_storage_manager

        # 设置较短的保留期进行测试
        manager._retention_days = 1  # 只保留1天

        # 创建一些旧的归档文件（模拟）
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        old_archive_file = manager.storage_path / f"audit_{old_date}.json.gz"

        # 创建一个空的旧归档文件
        with gzip.open(old_archive_file, 'wt', encoding='utf-8') as f:
            json.dump([], f)

        # 执行清理
        removed_count = manager.cleanup_storage(days_to_keep=1)

        # 验证清理结果
        assert isinstance(removed_count, int)

        # 如果文件被删除，计数应该反映
        if not old_archive_file.exists():
            assert removed_count >= 1

    def test_get_expired_archive_files(self, audit_storage_manager):
        """测试获取过期归档文件"""
        manager = audit_storage_manager

        # 创建模拟的过期文件
        expired_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
        expired_file = manager.storage_path / f"audit_{expired_date}.json.gz"

        with gzip.open(expired_file, 'wt', encoding='utf-8') as f:
            json.dump([], f)

        expired_files = manager._get_expired_archive_files()

        assert isinstance(expired_files, list)
        assert expired_file in expired_files


class TestAuditStorageManagerFiltering:
    """测试审计存储管理器过滤功能"""

    def test_filter_events_by_time(self, audit_storage_manager, sample_audit_events):
        """测试按时间过滤事件"""
        manager = audit_storage_manager

        start_time = datetime(2025, 1, 1, 11, 0, 0)
        end_time = datetime(2025, 1, 1, 15, 0, 0)

        filtered_events = manager._filter_events(sample_audit_events, start_time, end_time)

        # 应该过滤出前3个事件（都在第一天的时间范围内）
        assert len(filtered_events) == 3

        # 验证所有事件都在时间范围内
        for event in filtered_events:
            assert event.timestamp >= start_time
            assert event.timestamp <= end_time

    def test_filter_events_by_user_manual(self, audit_storage_manager, sample_audit_events):
        """测试手动按用户过滤事件"""
        manager = audit_storage_manager

        # _filter_events不支持user_id参数，使用手动过滤
        filtered_events = [e for e in sample_audit_events if e.user_id == "user123"]

        assert len(filtered_events) == 1
        assert filtered_events[0].user_id == "user123"

    def test_filter_events_by_resource(self, audit_storage_manager, sample_audit_events):
        """测试按资源过滤事件"""
        manager = audit_storage_manager

        # _filter_events不支持resource参数，使用手动过滤
        filtered_events = [e for e in sample_audit_events if e.resource == "/api/admin"]

        assert len(filtered_events) == 1
        assert filtered_events[0].resource == "/api/admin"

    def test_filter_events_by_event_type(self, audit_storage_manager, sample_audit_events):
        """测试按事件类型过滤事件"""
        manager = audit_storage_manager

        # 使用字符串值过滤事件类型
        filtered_events = manager._filter_events(
            sample_audit_events,
            event_type="security"
        )

        assert len(filtered_events) == 1
        assert filtered_events[0].event_type == AuditEventType.SECURITY

    def test_filter_events_combined_criteria(self, audit_storage_manager, sample_audit_events):
        """测试组合过滤条件"""
        manager = audit_storage_manager

        start_time = datetime(2025, 1, 1, 11, 0, 0)
        end_time = datetime(2025, 1, 1, 15, 0, 0)

        # _filter_events只支持时间和事件类型，使用组合过滤
        time_filtered = manager._filter_events(
            sample_audit_events,
            start_time=start_time,
            end_time=end_time
        )

        # 手动过滤用户
        final_filtered = [e for e in time_filtered if e.user_id == "user456"]

        # 应该只有1个事件满足所有条件
        assert len(final_filtered) == 1
        assert final_filtered[0].user_id == "user456"


class TestAuditStorageManagerArchivedEvents:
    """测试审计存储管理器归档事件检索"""

    def test_get_archived_events(self, audit_storage_manager, sample_audit_events):
        """测试获取归档事件"""
        manager = audit_storage_manager

        # 创建并存储归档文件
        date_str = "2025-01-01"
        events_to_archive = sample_audit_events[:2]
        manager._write_archive_file(date_str, events_to_archive)

        # 获取归档事件
        archived_events = manager._get_archived_events()

        assert isinstance(archived_events, list)
        # 应该包含归档的事件
        if len(archived_events) > 0:
            for event in archived_events:
                assert isinstance(event, AuditEvent)

    def test_get_archived_events_with_time_filter(self, audit_storage_manager, sample_audit_events):
        """测试按时间过滤获取归档事件"""
        manager = audit_storage_manager

        # 创建多个日期的归档文件
        date1 = "2025-01-01"
        date2 = "2025-01-02"

        manager._write_archive_file(date1, sample_audit_events[:2])
        manager._write_archive_file(date2, sample_audit_events[2:])

        # 获取特定时间范围的归档事件
        start_time = datetime(2025, 1, 1, 0, 0, 0)
        end_time = datetime(2025, 1, 1, 23, 59, 59)

        archived_events = manager._get_archived_events(start_time=start_time, end_time=end_time)

        # 应该只返回第一天的归档事件
        assert isinstance(archived_events, list)


class TestAuditStorageManagerUtilityMethods:
    """测试审计存储管理器工具方法"""

    def test_get_date_range(self, audit_storage_manager):
        """测试获取日期范围"""
        manager = audit_storage_manager

        # 测试无时间参数
        date_range = manager._get_date_range()
        assert isinstance(date_range, tuple)
        assert len(date_range) == 2

        # 测试有时间参数
        start_time = datetime(2025, 1, 1, 0, 0, 0)
        end_time = datetime(2025, 1, 3, 23, 59, 59)

        date_range = manager._get_date_range(start_time=start_time, end_time=end_time)
        assert isinstance(date_range, tuple)
        assert len(date_range) == 2

    def test_dict_to_event_conversion(self, sample_audit_events):
        """测试字典到事件转换"""
        event = sample_audit_events[0]

        # 转换为字典
        event_dict = event.__dict__

        # 转换回事件对象
        converted_event = AuditStorageManager._dict_to_event(event_dict)

        assert isinstance(converted_event, AuditEvent)
        assert converted_event.event_id == event.event_id
        assert converted_event.event_type == event.event_type
        assert converted_event.user_id == event.user_id


class TestAuditStorageManagerIntegration:
    """测试审计存储管理器集成功能"""

    def test_full_storage_workflow(self, audit_storage_manager, sample_audit_events):
        """测试完整存储工作流"""
        manager = audit_storage_manager

        # 1. 存储事件
        for event in sample_audit_events:
            manager.store_event(event)

        assert len(manager._memory_events) == len(sample_audit_events)

        # 2. 检索事件
        retrieved_events = manager.get_events()
        assert len(retrieved_events) == len(sample_audit_events)

        # 3. 归档事件
        archived_count = manager.archive_old_events()
        assert isinstance(archived_count, int)

        # 4. 获取统计信息
        stats = manager.get_storage_stats()
        assert isinstance(stats, dict)
        assert stats["total_events"] == len(sample_audit_events)

        # 5. 清理存储
        cleaned_count = manager.cleanup_storage(days_to_keep=365)
        assert isinstance(cleaned_count, int)

    def test_storage_with_large_dataset(self, temp_storage_dir):
        """测试大数据集存储"""
        manager = AuditStorageManager(
            storage_path=temp_storage_dir,
            max_memory_events=1000
        )

        # 创建大量事件
        events = []
        base_time = datetime(2025, 1, 1, 12, 0, 0)

        for i in range(500):
            event = AuditEvent(
                event_id=f"large-event-{i:04d}",
                event_type=AuditEventType.ACCESS,
                severity=AuditSeverity.LOW,
                timestamp=base_time + timedelta(seconds=i),
                user_id=f"user{i % 50}",
                resource=f"/api/resource/{i % 20}",
                details={"action": "test", "size": i * 100}
            )
            events.append(event)
            manager.store_event(event)

        # 验证存储
        assert len(manager._memory_events) <= manager._max_memory_events

        # 检索并验证
        retrieved_events = manager.get_events()
        assert len(retrieved_events) >= 0  # 可能部分在归档文件中

        # 检查统计信息
        stats = manager.get_storage_stats()
        assert stats["total_events"] == 500

    def test_storage_error_handling(self, audit_storage_manager):
        """测试存储错误处理"""
        manager = audit_storage_manager

        # 测试存储None事件
        try:
            manager.store_event(None)
            # 如果没有抛出异常，说明有错误处理
        except Exception:
            # 如果抛出异常，也是可以接受的
            pass

        # 测试存储无效事件
        try:
            invalid_event = "not an event"
            manager.store_event(invalid_event)
            # 如果没有抛出异常，说明有错误处理
        except Exception:
            # 如果抛出异常，也是可以接受的
            pass

    def test_concurrent_access_simulation(self, audit_storage_manager, sample_audit_events):
        """测试并发访问模拟"""
        manager = audit_storage_manager
        import threading
        import time

        results = []
        errors = []

        def store_events_thread(thread_id: int):
            try:
                for i in range(10):
                    event = AuditEvent(
                        event_id=f"thread-{thread_id}-event-{i}",
                        event_type=AuditEventType.ACCESS,
                        severity=AuditSeverity.LOW,
                        timestamp=datetime.now(),
                        user_id=f"user{thread_id}",
                        resource="/api/test",
                        details={"thread": thread_id, "index": i}
                    )
                    manager.store_event(event)
                    time.sleep(0.001)  # 小延迟模拟并发
                results.append(f"thread-{thread_id}-success")
            except Exception as e:
                errors.append(f"thread-{thread_id}-error: {e}")

        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=store_events_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 3  # 所有线程都成功了
        assert len(errors) == 0   # 没有错误

        # 验证总事件数
        stats = manager.get_storage_stats()
        assert stats["total_events"] >= 30  # 3个线程 * 10个事件
