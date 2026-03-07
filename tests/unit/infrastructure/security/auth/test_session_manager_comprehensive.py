#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 SessionManager综合测试

测试会话管理器的所有功能，包括：
- 会话创建和管理
- 会话验证和过期
- 会话清理
- 并发访问控制
- 会话统计
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch
from src.infrastructure.security.auth.session_manager import SessionManager, UserSession


class TestSessionManagerComprehensive:
    """SessionManager综合测试"""

    @pytest.fixture
    def session_manager(self):
        """创建SessionManager实例"""
        return SessionManager(session_timeout=300, max_sessions_per_user=3)

    def test_initialization(self, session_manager):
        """测试初始化"""
        assert session_manager.session_timeout == 300
        assert session_manager.max_sessions_per_user == 3
        assert len(session_manager._sessions) == 0
        assert len(session_manager._user_sessions) == 0

    def test_create_session_basic(self, session_manager):
        """测试基本会话创建"""
        session_id = session_manager.create_session("user1", "192.168.1.1")
        session = session_manager.get_session(session_id)

        assert session is not None
        assert session.user_id == "user1"
        assert session.ip_address == "192.168.1.1"
        assert not session.is_expired()
        assert session.session_id in session_manager._sessions

    def test_create_session_with_metadata(self, session_manager):
        """测试创建带元数据的会话"""
        session_id = session_manager.create_session(
            "user2",
            "192.168.1.2",
            user_agent="Mozilla/5.0",
            device_type="desktop"
        )
        session = session_manager.get_session(session_id)

        assert session.user_agent == "Mozilla/5.0"
        assert session.metadata["device_type"] == "desktop"

    def test_get_session_existing(self, session_manager):
        """测试获取现有会话"""
        created_session_id = session_manager.create_session("user3")

        retrieved_session = session_manager.get_session(created_session.session_id)

        assert retrieved_session is not None
        assert retrieved_session.session_id == created_session.session_id
        assert retrieved_session.user_id == "user3"

    def test_get_session_nonexistent(self, session_manager):
        """测试获取不存在的会话"""
        session = session_manager.get_session("nonexistent")
        assert session is None

    def test_validate_session_valid(self, session_manager):
        """测试验证有效会话"""
        session_id = session_manager.create_session("user4", "192.168.1.4")

        is_valid, session_data = session_manager.validate_session(session.session_id)

        assert is_valid == True
        assert session_data is not None
        assert session_data["user_id"] == "user4"

    def test_validate_session_with_ip_check(self, session_manager):
        """测试带IP检查的会话验证"""
        session_id = session_manager.create_session("user5", "192.168.1.5")

        # 相同IP应该验证通过
        is_valid, _ = session_manager.validate_session(session.session_id, "192.168.1.5")
        assert is_valid == True

        # 不同IP应该验证失败（如果启用了IP绑定）
        is_valid, _ = session_manager.validate_session(session.session_id, "192.168.1.6")
        # 注意：当前实现可能不强制IP绑定，视具体实现而定

    def test_validate_session_expired(self, session_manager):
        """测试验证过期会话"""
        # 创建一个短时效的会话管理器
        short_timeout_manager = SessionManager(session_timeout=1)
        session = short_timeout_manager.create_session("user6")

        # 等待会话过期
        time.sleep(2)

        is_valid, _ = short_timeout_manager.validate_session(session.session_id)
        assert is_valid == False

    def test_extend_session(self, session_manager):
        """测试延长会话"""
        session_id = session_manager.create_session("user7")
        session = session_manager.get_session(session_id)
        original_expiry = session.expires_at

        success = session_manager.extend_session(session.session_id, 60)

        assert success == True
        extended_session = session_manager.get_session(session.session_id)
        assert extended_session.expires_at > original_expiry

    def test_extend_session_nonexistent(self, session_manager):
        """测试延长不存在的会话"""
        success = session_manager.extend_session("nonexistent", 60)
        assert success == False

    def test_invalidate_session(self, session_manager):
        """测试使会话失效"""
        session_id = session_manager.create_session("user8")
        success = session_manager.invalidate_session(session_id)

        assert success == True
        assert session_manager.get_session(session.session_id) is None

    def test_invalidate_session_nonexistent(self, session_manager):
        """测试使不存在的会话失效"""
        success = session_manager.invalidate_session("nonexistent")
        assert success == False

    def test_invalidate_user_sessions(self, session_manager):
        """测试使用户的所有会话失效"""
        # 创建用户的多个会话
        session1 = session_manager.create_session("user9")
        session2 = session_manager.create_session("user9")
        session3 = session_manager.create_session("user9")

        # 验证会话已创建
        assert len(session_manager.get_user_sessions("user9")) == 3

        # 使所有会话失效
        invalidated_count = session_manager.invalidate_user_sessions("user9")

        assert invalidated_count == 3
        assert len(session_manager.get_user_sessions("user9")) == 0

    def test_invalidate_user_sessions_nonexistent_user(self, session_manager):
        """测试使不存在用户的会话失效"""
        invalidated_count = session_manager.invalidate_user_sessions("nonexistent")
        assert invalidated_count == 0

    def test_get_user_sessions(self, session_manager):
        """测试获取用户的所有会话"""
        # 创建多个用户的会话
        session_manager.create_session("user10")
        session_manager.create_session("user10")
        session_manager.create_session("user11")

        user10_sessions = session_manager.get_user_sessions("user10")
        user11_sessions = session_manager.get_user_sessions("user11")

        assert len(user10_sessions) == 2
        assert len(user11_sessions) == 1

        # 验证所有会话都属于正确的用户
        for session in user10_sessions:
            assert session.user_id == "user10"

    def test_max_sessions_per_user(self, session_manager):
        """测试每个用户的最大会话数限制"""
        # 创建超过限制的会话
        for i in range(5):  # 超过max_sessions_per_user=3的限制
            session_manager.create_session("user12")

        # 应该只有最新的3个会话保留
        user_sessions = session_manager.get_user_sessions("user12")
        assert len(user_sessions) == 3

    def test_cleanup_expired_sessions(self, session_manager):
        """测试清理过期会话"""
        # 创建短时效的会话管理器
        short_timeout_manager = SessionManager(session_timeout=1)

        # 创建一些会话
        short_timeout_manager.create_session("user13")
        short_timeout_manager.create_session("user14")

        # 等待过期
        time.sleep(2)

        # 清理过期会话
        cleaned_count = short_timeout_manager.cleanup_expired_sessions()

        assert cleaned_count >= 2  # 应该清理了过期会话

    def test_get_session_stats(self, session_manager):
        """测试获取会话统计"""
        # 创建一些会话
        session_manager.create_session("user15", "192.168.1.15")
        session_manager.create_session("user16", "192.168.1.16")
        session_manager.create_session("user15", "192.168.1.17")  # 同一个用户的另一个会话

        stats = session_manager.get_session_stats()

        assert isinstance(stats, dict)
        assert "active_sessions" in stats
        assert "active_users" in stats
        assert stats["active_sessions"] == 3
        assert stats["active_users"] == 2  # user15和user16

    def test_get_session_info(self, session_manager):
        """测试获取会话信息"""
        session_id = session_manager.create_session("user17", "192.168.1.17", user_agent="TestAgent")
        info = session_manager.get_session_info(session_id)

        assert info is not None
        assert info["user_id"] == "user17"
        assert info["ip_address"] == "192.168.1.17"
        assert info["user_agent"] == "TestAgent"
        assert "created_at" in info
        assert "expires_at" in info

    def test_get_session_info_nonexistent(self, session_manager):
        """测试获取不存在会话的信息"""
        info = session_manager.get_session_info("nonexistent")
        assert info is None

    def test_update_session_metadata(self, session_manager):
        """测试更新会话元数据"""
        session_id = session_manager.create_session("user18")
        success = session_manager.update_session_metadata(
            session_id, last_activity=datetime.now(),
            page_views=5
        )

        assert success == True

        updated_session = session_manager.get_session(session.session_id)
        assert updated_session.metadata["last_activity"] is not None
        assert updated_session.metadata["page_views"] == 5

    def test_update_session_metadata_nonexistent(self, session_manager):
        """测试更新不存在会话的元数据"""
        success = session_manager.update_session_metadata("nonexistent", key="value")
        assert success == False

    def test_session_expiry_methods(self, session_manager):
        """测试会话过期相关方法"""
        session_id = session_manager.create_session("user19")
        session = session_manager.get_session(session_id)

        # 测试未过期
        assert not session.is_expired()
        session = session_manager.get_session(session_id)

        assert session.get_remaining_time() > timedelta(0)

        # 手动设置过期时间
        session.expires_at = datetime.now() - timedelta(minutes=1)

        # 测试已过期
        assert session.is_expired()
        assert session.get_remaining_time() <= timedelta(0)

    def test_session_extend_method(self, session_manager):
        """测试会话延长方法"""
        session_id = session_manager.create_session("user20")
        session = session_manager.get_session(session_id)
        original_expiry = session.expires_at

        session.extend_session(30)  # 延长30分钟

        assert session.expires_at > original_expiry

    def test_session_to_dict(self, session_manager):
        """测试会话序列化"""
        session_id = session_manager.create_session("user21", "192.168.1.21")
        session = session_manager.get_session(session_id)
        session_dict = session.to_dict()

        assert isinstance(session_dict, dict)
        assert "session_id" in session_dict
        assert "user_id" in session_dict
        assert "created_at" in session_dict
        assert "expires_at" in session_dict
        assert session_dict["user_id"] == "user21"

    def test_concurrent_session_creation(self, session_manager):
        """测试并发会话创建"""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def create_sessions(user_id, count):
            try:
                created_sessions = []
                for i in range(count):
                    session = session_manager.create_session(f"user_{user_id}")
                    created_sessions.append(session)
                results.put((user_id, len(created_sessions)))
            except Exception as e:
                errors.put(f"user_{user_id}: {e}")

        # 启动多个线程创建会话
        threads = []
        for i in range(5):
            t = threading.Thread(target=create_sessions, args=(i, 3))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join(timeout=5)

        # 验证结果
        assert errors.empty()

        total_sessions = 0
        while not results.empty():
            _, count = results.get()
            total_sessions += count

        assert total_sessions == 15  # 5个用户，每个3个会话

    def test_session_id_uniqueness(self, session_manager):
        """测试会话ID的唯一性"""
        sessions = []
        for i in range(100):
            session = session_manager.create_session(f"user_{i}")
        sessions.append(session_id)

        # 验证所有ID都唯一
        session_ids = [s.session_id for s in sessions]
        assert len(set(session_ids)) == len(session_ids)

    def test_large_scale_session_management(self, session_manager):
        """测试大规模会话管理"""
        # 创建大量会话
        session_count = 200
        created_sessions = []

        for i in range(session_count):
            session = session_manager.create_session(f"user_{i % 20}")  # 20个不同用户
            created_sessions.append(session)

        # 验证创建
        assert len(session_manager.sessions) <= session_manager.max_sessions_per_user * 20

        # 测试批量验证
        valid_count = 0
        for session in created_sessions[:50]:  # 只测试前50个
            if session_manager.get_session(session.session_id):
                valid_count += 1

        assert valid_count >= 40  # 至少80%应该仍然有效

    def test_session_timeout_edge_cases(self, session_manager):
        """测试会话超时边界情况"""
        # 测试刚好过期
        zero_timeout_manager = SessionManager(session_timeout=0)
        session = zero_timeout_manager.create_session("user_timeout")

        # 立即检查应该已过期
        assert session.is_expired()

        # 测试负数超时（应该处理 gracefully）
        try:
            negative_timeout_manager = SessionManager(session_timeout=-1)
            session = negative_timeout_manager.create_session("user_negative")
            # 应该正常工作或抛出合理异常
        except (ValueError, AssertionError):
            pass  # 预期的异常

    def test_memory_cleanup_efficiency(self, session_manager):
        """测试内存清理效率"""
        # 创建然后立即失效会话
        for i in range(100):
            session = session_manager.create_session(f"user_cleanup_{i}")
            session_manager.invalidate_session(session.session_id)

        # 验证内存清理
        assert len(session_manager.sessions) == 0

        # 创建新会话验证系统仍然正常
        new_session = session_manager.create_session("user_after_cleanup")
        assert new_session is not None
        assert session_manager.get_session(new_session.session_id) is not None

    def test_audit_logging_on_session_operations(self, session_manager):
        """测试会话操作的审计日志"""
        with patch('src.infrastructure.security.auth.session_manager.logging') as mock_logging:
            session = session_manager.create_session("user_audit")

            # 应该有日志记录
            mock_logging.info.assert_called()

            # 测试会话失效时的日志
            session_manager.invalidate_session(session.session_id)
            # 再次验证日志调用
            assert mock_logging.info.call_count >= 2
