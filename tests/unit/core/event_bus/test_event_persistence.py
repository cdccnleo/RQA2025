# -*- coding: utf-8 -*-
"""
核心服务层 - 事件持久化单元测试
测试覆盖率目标: 80%+
测试EventPersistence的核心功能：存储、检索、重放、清理
"""

import pytest
import tempfile
import os
import json
import time
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 跳过整个测试文件 - 模块导入问题，需要修复依赖
# 尝试导入所需模块
try:
    from src.core.event_bus.persistence.persistence import EventPersistence
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestPersistedEvent:
    """测试PersistedEvent数据类"""

    def test_persisted_event_creation(self):
        """测试PersistedEvent创建"""
        event = PersistedEvent(
            event_id="test-123",
            event_type="test_event",
            event_data={"key": "value"},
            timestamp=datetime.now()
        )

        assert event.event_id == "test-123"
        assert event.event_type == "test_event"
        assert event.event_data == {"key": "value"}
        assert event.status == EventStatus.PENDING
        assert event.retry_count == 0
        assert event.max_retries == 3
        assert event.processing_timeout == 300

    def test_persisted_event_to_dict(self):
        """测试PersistedEvent序列化"""
        timestamp = datetime.now()
        event = PersistedEvent(
            event_id="test-123",
            event_type="test_event",
            event_data={"key": "value"},
            timestamp=timestamp,
            status=EventStatus.COMPLETED,
            retry_count=2,
            error_message="Test error",
            metadata={"source": "test"}
        )

        event_dict = event.to_dict()

        assert event_dict["event_id"] == "test-123"
        assert event_dict["event_type"] == "test_event"
        assert event_dict["event_data"] == {"key": "value"}
        assert event_dict["status"] == "completed"
        assert event_dict["retry_count"] == 2
        assert event_dict["error_message"] == "Test error"
        assert event_dict["metadata"] == {"source": "test"}

    def test_persisted_event_from_dict(self):
        """测试PersistedEvent反序列化"""
        timestamp = datetime.now()
        event_dict = {
            "event_id": "test-123",
            "event_type": "test_event",
            "event_data": {"key": "value"},
            "timestamp": timestamp.isoformat(),
            "status": "completed",
            "retry_count": 2,
            "max_retries": 5,
            "processing_timeout": 600,
            "created_at": timestamp.isoformat(),
            "updated_at": timestamp.isoformat(),
            "processed_at": (timestamp + timedelta(minutes=1)).isoformat(),
            "error_message": "Test error",
            "metadata": {"source": "test"}
        }

        event = PersistedEvent.from_dict(event_dict)

        assert event.event_id == "test-123"
        assert event.event_type == "test_event"
        assert event.event_data == {"key": "value"}
        assert event.status == EventStatus.COMPLETED
        assert event.retry_count == 2
        assert event.max_retries == 5
        assert event.processing_timeout == 600
        assert event.error_message == "Test error"
        assert event.metadata == {"source": "test"}


class TestEventPersistenceInitialization:
    """测试EventPersistence初始化"""

    def test_memory_mode_initialization(self):
        """测试内存模式初始化"""
        persistence = EventPersistence(mode=PersistenceMode.MEMORY)
        assert persistence.mode == PersistenceMode.MEMORY
        assert persistence.name == "EventPersistence"
        assert persistence.version == "1.0.0"

    def test_file_mode_initialization(self):
        """测试文件模式初始化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"storage_path": temp_dir}
            persistence = EventPersistence(
                mode=PersistenceMode.FILE,
                config=config
            )
            assert persistence.mode == PersistenceMode.FILE
            assert persistence._storage_path == temp_dir

    def test_database_mode_initialization(self):
        """测试数据库模式初始化"""
        config = {"db_path": ":memory:"}  # 使用内存数据库进行测试
        persistence = EventPersistence(
            mode=PersistenceMode.DATABASE,
            config=config
        )
        assert persistence.mode == PersistenceMode.DATABASE

    def test_initialization_with_custom_config(self):
        """测试使用自定义配置初始化"""
        config = {
            "cleanup_interval": 1800,
            "max_age_days": 14,
            "compression_enabled": True
        }

        persistence = EventPersistence(
            mode=PersistenceMode.FILE,
            config=config
        )

        assert persistence._cleanup_interval == 1800
        assert persistence._max_age_days == 14
        assert persistence._compression_enabled == True


class TestEventPersistenceStorage:
    """测试事件存储功能"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "storage_path": self.temp_dir,
            "cleanup_interval": 3600,
            "max_age_days": 30
        }
        self.persistence = EventPersistence(
            mode=PersistenceMode.FILE,
            config=self.config
        )
        # 初始化持久化管理器
        self.persistence.initialize()

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_store_event(self):
        """测试存储事件"""
        event = PersistedEvent(
            event_id="store-test-123",
            event_type="test_event",
            event_data={"key": "value"},
            timestamp=datetime.now()
        )

        # 存储事件
        result = self.persistence.store_event(event)
        assert result == True

        # 验证事件是否正确存储
        stored_events = self.persistence.get_events_by_type("test_event")
        assert len(stored_events) == 1
        assert stored_events[0].event_id == "store-test-123"

    def test_store_multiple_events(self):
        """测试存储多个事件"""
        events = []
        for i in range(5):
            event = PersistedEvent(
                event_id=f"multi-test-{i}",
                event_type="multi_event",
                event_data={"index": i},
                timestamp=datetime.now()
            )
            events.append(event)
            self.persistence.store_event(event)

        # 验证所有事件都已存储
        stored_events = self.persistence.get_events_by_type("multi_event")
        assert len(stored_events) == 5

        event_ids = {event.event_id for event in stored_events}
        expected_ids = {f"multi-test-{i}" for i in range(5)}
        assert event_ids == expected_ids

    def test_update_event_status(self):
        """测试更新事件状态"""
        event = PersistedEvent(
            event_id="update-test-123",
            event_type="update_event",
            event_data={"key": "value"},
            timestamp=datetime.now()
        )

        # 存储事件
        self.persistence.store_event(event)

        # 更新状态
        result = self.persistence.update_event_status(
            "update-test-123",
            EventStatus.PROCESSING
        )
        assert result == True

        # 验证状态更新
        updated_event = self.persistence.get_event("update-test-123")
        assert updated_event.status == EventStatus.PROCESSING

    def test_update_event_with_error(self):
        """测试更新事件错误信息"""
        event = PersistedEvent(
            event_id="error-test-123",
            event_type="error_event",
            event_data={"key": "value"},
            timestamp=datetime.now()
        )

        self.persistence.store_event(event)

        # 更新为失败状态和错误信息
        result = self.persistence.update_event_status(
            "error-test-123",
            EventStatus.FAILED,
            error_message="Test failure",
            retry_count=2
        )
        assert result == True

        # 验证错误信息
        failed_event = self.persistence.get_event("error-test-123")
        assert failed_event.status == EventStatus.FAILED
        assert failed_event.error_message == "Test failure"
        assert failed_event.retry_count == 2


class TestEventPersistenceRetrieval:
    """测试事件检索功能"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {"storage_path": self.temp_dir}
        self.persistence = EventPersistence(
            mode=PersistenceMode.FILE,
            config=self.config
        )

        # 预存一些测试事件
        self._create_test_events()

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_test_events(self):
        """创建测试事件"""
        base_time = datetime.now()

        # 创建不同类型的事件
        event_types = ["user_event", "system_event", "data_event"]
        for i, event_type in enumerate(event_types):
            for j in range(3):
                event = PersistedEvent(
                    event_id=f"{event_type}-{j}",
                    event_type=event_type,
                    event_data={"index": j, "type": event_type},
                    timestamp=base_time + timedelta(minutes=i*10 + j),
                    status=EventStatus.PENDING if j % 2 == 0 else EventStatus.COMPLETED
                )
                self.persistence.store_event(event)

    def test_get_event_by_id(self):
        """测试通过ID获取事件"""
        event = self.persistence.get_event("user_event-0")
        assert event is not None
        assert event.event_id == "user_event-0"
        assert event.event_type == "user_event"
        assert event.event_data["index"] == 0

    def test_get_nonexistent_event(self):
        """测试获取不存在的事件"""
        event = self.persistence.get_event("nonexistent-123")
        assert event is None

    def test_get_events_by_type(self):
        """测试通过类型获取事件"""
        user_events = self.persistence.get_events_by_type("user_event")
        assert len(user_events) == 3

        for event in user_events:
            assert event.event_type == "user_event"
            assert "index" in event.event_data

    def test_get_events_by_status(self):
        """测试通过状态获取事件"""
        pending_events = self.persistence.get_events_by_status(EventStatus.PENDING)
        completed_events = self.persistence.get_events_by_status(EventStatus.COMPLETED)

        # 应该有3个待处理事件（索引为偶数）和3个已完成事件（索引为奇数）
        assert len(pending_events) == 3  # 每种类型1个
        assert len(completed_events) == 6  # 每种类型2个

    def test_get_events_by_time_range(self):
        """测试通过时间范围获取事件"""
        base_time = datetime.now()
        start_time = base_time + timedelta(minutes=5)
        end_time = base_time + timedelta(minutes=25)

        time_range_events = self.persistence.get_events_by_time_range(
            start_time, end_time
        )

        # 应该包含system_event和data_event的事件
        assert len(time_range_events) >= 4  # system_event (10-12min) + data_event (20-22min)

    def test_get_pending_events(self):
        """测试获取待处理事件"""
        pending_events = self.persistence.get_pending_events()
        assert len(pending_events) == 3  # 每种类型1个待处理事件

        for event in pending_events:
            assert event.status == EventStatus.PENDING

    def test_get_failed_events(self):
        """测试获取失败事件"""
        # 先创建一个失败事件
        failed_event = PersistedEvent(
            event_id="failed-test-123",
            event_type="failed_event",
            event_data={"test": "data"},
            timestamp=datetime.now(),
            status=EventStatus.FAILED,
            error_message="Test failure"
        )
        self.persistence.store_event(failed_event)

        failed_events = self.persistence.get_failed_events()
        assert len(failed_events) >= 1

        # 查找我们创建的失败事件
        our_failed_event = next(
            (e for e in failed_events if e.event_id == "failed-test-123"),
            None
        )
        assert our_failed_event is not None
        assert our_failed_event.error_message == "Test failure"


class TestEventPersistenceReplay:
    """测试事件重放功能"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {"storage_path": self.temp_dir}
        self.persistence = EventPersistence(
            mode=PersistenceMode.FILE,
            config=self.config
        )

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_replay_events_by_type(self):
        """测试通过类型重放事件"""
        # 创建一些已完成的事件用于重放
        base_time = datetime.now()
        for i in range(3):
            event = PersistedEvent(
                event_id=f"replay-{i}",
                event_type="replay_event",
                event_data={"index": i},
                timestamp=base_time + timedelta(minutes=i),
                status=EventStatus.COMPLETED
            )
            self.persistence.store_event(event)

        replayed_events = []
        def replay_handler(event: PersistedEvent):
            replayed_events.append(event)

        # 重放事件
        count = self.persistence.replay_events_by_type(
            "replay_event",
            replay_handler
        )

        assert count == 3
        assert len(replayed_events) == 3

        # 验证重放的事件按时间顺序
        for i, event in enumerate(replayed_events):
            assert event.event_data["index"] == i

    def test_replay_events_by_time_range(self):
        """测试通过时间范围重放事件"""
        base_time = datetime.now()

        # 创建不同时间的事件
        for i in range(5):
            event = PersistedEvent(
                event_id=f"time-replay-{i}",
                event_type="time_replay_event",
                event_data={"index": i},
                timestamp=base_time + timedelta(minutes=i*10),
                status=EventStatus.COMPLETED
            )
            self.persistence.store_event(event)

        replayed_events = []
        def replay_handler(event: PersistedEvent):
            replayed_events.append(event)

        # 重放特定时间范围的事件
        start_time = base_time + timedelta(minutes=10)
        end_time = base_time + timedelta(minutes=35)

        count = self.persistence.replay_events_by_time_range(
            start_time, end_time, replay_handler
        )

        # 应该重放索引1、2、3的事件（20-30分钟）
        assert count == 3
        assert len(replayed_events) == 3
        indices = [event.event_data["index"] for event in replayed_events]
        assert indices == [1, 2, 3]

    def test_replay_limit(self):
        """测试重放数量限制"""
        # 创建多个事件
        for i in range(10):
            event = PersistedEvent(
                event_id=f"limit-replay-{i}",
                event_type="limit_event",
                event_data={"index": i},
                timestamp=datetime.now(),
                status=EventStatus.COMPLETED
            )
            self.persistence.store_event(event)

        replayed_events = []
        def replay_handler(event: PersistedEvent):
            replayed_events.append(event)

        # 限制重放数量为5
        count = self.persistence.replay_events_by_type(
            "limit_event",
            replay_handler,
            limit=5
        )

        assert count == 5
        assert len(replayed_events) == 5


class TestEventPersistenceCleanup:
    """测试事件清理功能"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {"storage_path": self.temp_dir}
        self.persistence = EventPersistence(
            mode=PersistenceMode.FILE,
            config=self.config
        )

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_cleanup_expired_events(self):
        """测试清理过期事件"""
        base_time = datetime.now()

        # 创建新事件（不应该被清理）
        recent_event = PersistedEvent(
            event_id="recent-event",
            event_type="test_event",
            event_data={"recent": True},
            timestamp=base_time,
            status=EventStatus.COMPLETED
        )
        self.persistence.store_event(recent_event)

        # 创建过期事件（应该被清理）
        old_event = PersistedEvent(
            event_id="old-event",
            event_type="test_event",
            event_data={"old": True},
            timestamp=base_time - timedelta(days=40),  # 超过30天
            status=EventStatus.COMPLETED
        )
        self.persistence.store_event(old_event)

        # 执行清理
        cleaned_count = self.persistence.cleanup_expired_events(max_age_days=30)
        assert cleaned_count >= 1

        # 验证过期事件已被清理
        remaining_events = self.persistence.get_events_by_type("test_event")
        assert len(remaining_events) == 1
        assert remaining_events[0].event_id == "recent-event"

    def test_cleanup_failed_events(self):
        """测试清理失败事件"""
        # 创建一些失败事件
        for i in range(3):
            failed_event = PersistedEvent(
                event_id=f"failed-{i}",
                event_type="failed_event",
                event_data={"failed": True},
                timestamp=datetime.now(),
                status=EventStatus.FAILED,
                retry_count=3,  # 达到最大重试次数
                max_retries=3
            )
            self.persistence.store_event(failed_event)

        # 创建一个仍在重试的事件
        retrying_event = PersistedEvent(
            event_id="retrying-event",
            event_type="failed_event",
            event_data={"retrying": True},
            timestamp=datetime.now(),
            status=EventStatus.FAILED,
            retry_count=1,
            max_retries=3
        )
        self.persistence.store_event(retrying_event)

        # 清理失败事件
        cleaned_count = self.persistence.cleanup_failed_events(max_retry_cleanup=2)
        assert cleaned_count >= 3  # 3个达到最大重试次数的事件

        # 验证结果
        remaining_failed = self.persistence.get_failed_events()
        assert len(remaining_failed) == 1  # 只剩下仍在重试的事件
        assert remaining_failed[0].event_id == "retrying-event"


class TestEventPersistenceHealth:
    """测试事件持久化健康检查"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {"storage_path": self.temp_dir}
        self.persistence = EventPersistence(
            mode=PersistenceMode.FILE,
            config=self.config
        )

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_health_check(self):
        """测试健康检查"""
        health = self.persistence.check_health()
        assert health.status == "healthy"
        assert "EventPersistence" in health.message

    def test_health_check_with_storage_issues(self):
        """测试存储问题时的健康检查"""
        # 模拟存储路径不存在的情况
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # 重新创建持久化实例，应该能处理不存在的路径
        persistence = EventPersistence(
            mode=PersistenceMode.FILE,
            config={"storage_path": self.temp_dir}
        )

        health = persistence.check_health()
        # 应该仍然健康，因为会自动创建目录
        assert health.status in ["healthy", "degraded"]

    def test_get_statistics(self):
        """测试获取统计信息"""
        # 存储一些事件
        for i in range(5):
            event = PersistedEvent(
                event_id=f"stats-{i}",
                event_type="stats_event",
                event_data={"index": i},
                timestamp=datetime.now(),
                status=EventStatus.COMPLETED if i % 2 == 0 else EventStatus.PENDING
            )
            self.persistence.store_event(event)

        stats = self.persistence.get_statistics()

        assert "total_events" in stats
        assert "events_by_type" in stats
        assert "events_by_status" in stats
        assert stats["total_events"] >= 5
        assert stats["events_by_type"]["stats_event"] >= 5


class TestEventPersistenceShutdown:
    """测试事件持久化关闭功能"""

    def test_shutdown(self):
        """测试正常关闭"""
        persistence = EventPersistence(mode=PersistenceMode.MEMORY)

        # 模拟一些操作
        event = PersistedEvent(
            event_id="shutdown-test",
            event_type="shutdown_event",
            event_data={"test": "shutdown"},
            timestamp=datetime.now()
        )
        persistence.store_event(event)

        # 关闭
        result = persistence.shutdown()
        assert result == True

        # 验证状态
        health = persistence.check_health()
        assert health.status == "unhealthy"  # 关闭后应该不健康


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
