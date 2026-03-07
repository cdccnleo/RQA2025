#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 分布式锁管理器深度测试
测试DistributedLockManager的核心分布式锁功能、并发控制和容错机制
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import logging
import threading
import time
from datetime import datetime
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import pytest

from infrastructure.distributed.distributed_lock import (
    DistributedLockManager, LockInfo
)


class TestLockInfo:
    """LockInfo测试"""

    def test_lock_info_initialization(self):
        """测试LockInfo初始化"""
        lock_info = LockInfo(
            lock_id="test_lock",
            owner="test_owner",
            acquired_time=1234567890.0,
            ttl=30
        )

        assert lock_info.lock_id == "test_lock"
        assert lock_info.owner == "test_owner"
        assert lock_info.acquired_time == 1234567890.0
        assert lock_info.ttl == 30
        assert lock_info.renew_count == 0

    def test_lock_info_with_renew_count(self):
        """测试带续期次数的LockInfo"""
        lock_info = LockInfo(
            lock_id="renewed_lock",
            owner="owner",
            acquired_time=time.time(),
            ttl=60,
            renew_count=5
        )

        assert lock_info.renew_count == 5


class TestDistributedLockManagerInitialization:
    """DistributedLockManager初始化测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        manager = DistributedLockManager()

        assert manager.config == {}
        assert isinstance(manager.logger, logging.Logger)
        assert hasattr(manager, '_locks')
        assert hasattr(manager, '_lock')

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = {
            'max_lock_time': 300,
            'cleanup_interval': 60,
            'retry_attempts': 3
        }
        manager = DistributedLockManager(config)

        assert manager.config == config

    def test_initialization_empty_config(self):
        """测试空配置初始化"""
        manager = DistributedLockManager({})

        assert manager.config == {}


class TestDistributedLockManagerBasicOperations:
    """DistributedLockManager基本操作测试"""

    @pytest.fixture
    def manager(self):
        """DistributedLockManager fixture"""
        return DistributedLockManager()

    def test_acquire_lock_success(self, manager):
        """测试成功获取锁"""
        lock_key = "test_lock_1"
        owner = "test_owner"

        result = manager.acquire_lock(lock_key, owner, ttl=30)

        assert result is True

        # 验证锁信息
        lock_info = manager.get_lock_info(lock_key)
        assert lock_info is not None
        assert lock_info.lock_id == lock_key
        assert lock_info.owner == owner
        assert lock_info.ttl == 30
        assert lock_info.renew_count == 0

    def test_acquire_lock_already_held(self, manager):
        """测试获取已持有的锁"""
        lock_key = "test_lock_2"
        owner1 = "owner1"
        owner2 = "owner2"

        # 第一个owner获取锁
        result1 = manager.acquire_lock(lock_key, owner1, ttl=30)
        assert result1 is True

        # 第二个owner尝试获取同一锁，应该失败
        result2 = manager.acquire_lock(lock_key, owner2, ttl=30)
        assert result2 is False

        # 锁仍然属于第一个owner
        lock_info = manager.get_lock_info(lock_key)
        assert lock_info.owner == owner1

    def test_try_acquire_lock_success(self, manager):
        """测试非阻塞获取锁成功"""
        lock_key = "test_lock_3"
        owner = "test_owner"

        result = manager.try_acquire_lock(lock_key, owner, ttl=30)

        assert result is True

        lock_info = manager.get_lock_info(lock_key)
        assert lock_info is not None
        assert lock_info.owner == owner

    def test_try_acquire_lock_failure(self, manager):
        """测试非阻塞获取锁失败"""
        lock_key = "test_lock_4"
        owner1 = "owner1"
        owner2 = "owner2"

        # 第一个owner获取锁
        manager.try_acquire_lock(lock_key, owner1, ttl=30)

        # 第二个owner尝试获取，应该立即失败
        result = manager.try_acquire_lock(lock_key, owner2, ttl=30)
        assert result is False

    def test_release_lock_success(self, manager):
        """测试成功释放锁"""
        lock_key = "test_lock_5"
        owner = "test_owner"

        # 获取锁
        manager.acquire_lock(lock_key, owner, ttl=30)

        # 释放锁
        result = manager.release_lock(lock_key, owner)
        assert result is True

        # 验证锁已被释放
        lock_info = manager.get_lock_info(lock_key)
        assert lock_info is None

    def test_release_lock_wrong_owner(self, manager):
        """测试错误的owner释放锁"""
        lock_key = "test_lock_6"
        owner1 = "owner1"
        owner2 = "owner2"

        # owner1获取锁
        manager.acquire_lock(lock_key, owner1, ttl=30)

        # owner2尝试释放锁，应该失败
        result = manager.release_lock(lock_key, owner2)
        assert result is False

        # 锁仍然存在且属于owner1
        lock_info = manager.get_lock_info(lock_key)
        assert lock_info is not None
        assert lock_info.owner == owner1

    def test_release_lock_not_exists(self, manager):
        """测试释放不存在的锁"""
        result = manager.release_lock("nonexistent_lock", "owner")
        assert result is False


class TestDistributedLockManagerRenewal:
    """DistributedLockManager锁续期测试"""

    @pytest.fixture
    def manager(self):
        """DistributedLockManager fixture"""
        return DistributedLockManager()

    def test_renew_lock_success(self, manager):
        """测试成功续期锁"""
        lock_key = "renew_lock_1"
        owner = "test_owner"

        # 获取锁
        manager.acquire_lock(lock_key, owner, ttl=30)
        original_lock_info = manager.get_lock_info(lock_key)

        # 续期锁
        result = manager.renew_lock(lock_key, owner, ttl=60)
        assert result is True

        # 验证锁信息已更新
        renewed_lock_info = manager.get_lock_info(lock_key)
        assert renewed_lock_info is not None
        assert renewed_lock_info.owner == owner
        assert renewed_lock_info.ttl == 60
        assert renewed_lock_info.renew_count == original_lock_info.renew_count + 1

    def test_renew_lock_wrong_owner(self, manager):
        """测试错误owner续期锁"""
        lock_key = "renew_lock_2"
        owner1 = "owner1"
        owner2 = "owner2"

        # owner1获取锁
        manager.acquire_lock(lock_key, owner1, ttl=30)

        # owner2尝试续期，应该失败
        result = manager.renew_lock(lock_key, owner2, ttl=60)
        assert result is False

        # 锁信息保持不变
        lock_info = manager.get_lock_info(lock_key)
        assert lock_info.owner == owner1
        assert lock_info.ttl == 30
        assert lock_info.renew_count == 0

    def test_renew_lock_not_exists(self, manager):
        """测试续期不存在的锁"""
        result = manager.renew_lock("nonexistent", "owner", ttl=60)
        assert result is False


class TestDistributedLockManagerExpiration:
    """DistributedLockManager锁过期测试"""

    @pytest.fixture
    def manager(self):
        """DistributedLockManager fixture"""
        return DistributedLockManager()

    def test_lock_expiration(self, manager):
        """测试锁过期"""
        lock_key = "expire_lock_1"
        owner = "test_owner"

        # 获取一个短TTL的锁
        manager.acquire_lock(lock_key, owner, ttl=1)  # 1秒TTL

        # 验证锁存在
        lock_info = manager.get_lock_info(lock_key)
        assert lock_info is not None

        # 等待锁过期
        time.sleep(1.1)

        # 手动清理过期锁
        manager._cleanup_expired_lock(lock_key)

        # 验证锁已被清理
        lock_info = manager.get_lock_info(lock_key)
        assert lock_info is None

    def test_cleanup_expired_locks(self, manager):
        """测试清理所有过期锁"""
        # 创建多个锁
        locks = []
        for i in range(5):
            lock_key = f"expire_lock_{i}"
            owner = f"owner_{i}"
            ttl = 1 if i < 3 else 60  # 前3个1秒，后2个60秒

            manager.acquire_lock(lock_key, owner, ttl)
            locks.append((lock_key, owner, ttl))

        # 等待前3个锁过期
        time.sleep(1.1)

        # 清理过期锁
        manager.cleanup_expired_locks()

        # 验证前3个锁已被清理，后2个仍然存在
        for i, (lock_key, owner, ttl) in enumerate(locks):
            lock_info = manager.get_lock_info(lock_key)
            if i < 3:  # 应该已被清理
                assert lock_info is None, f"Lock {lock_key} should be expired"
            else:  # 应该仍然存在
                assert lock_info is not None, f"Lock {lock_key} should still exist"
                assert lock_info.owner == owner


class TestDistributedLockManagerConcurrency:
    """DistributedLockManager并发测试"""

    @pytest.fixture
    def manager(self):
        """DistributedLockManager fixture"""
        return DistributedLockManager()

    def test_concurrent_lock_acquisition(self, manager):
        """测试并发锁获取"""
        lock_key = "concurrent_lock"
        results = []
        exceptions = []

        def acquire_worker(owner_id: str):
            """获取锁的工作线程"""
            try:
                result = manager.acquire_lock(lock_key, owner_id, ttl=30)
                results.append((owner_id, result))
            except Exception as e:
                exceptions.append(f"{owner_id}: {e}")

        # 启动多个线程同时尝试获取同一锁
        threads = []
        for i in range(10):
            t = threading.Thread(target=acquire_worker, args=(f"owner_{i}",))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 不应该有异常
        assert len(exceptions) == 0, f"Concurrent exceptions: {exceptions}"

        # 应该只有一个线程成功获取锁
        successful_acquires = [r for r in results if r[1] is True]
        assert len(successful_acquires) == 1, f"Expected 1 successful acquire, got {len(successful_acquires)}"

        # 其他线程应该失败
        failed_acquires = [r for r in results if r[1] is False]
        assert len(failed_acquires) == 9, f"Expected 9 failed acquires, got {len(failed_acquires)}"

    def test_lock_contention_and_release(self, manager):
        """测试锁竞争和释放"""
        lock_key = "contention_lock"
        operations = []
        exceptions = []

        def contention_worker(worker_id: str):
            """竞争锁的工作线程"""
            try:
                # 尝试获取锁
                acquired = manager.acquire_lock(lock_key, worker_id, ttl=30)
                if acquired:
                    operations.append(f"{worker_id}: acquired")
                    # 持有锁一段时间
                    time.sleep(0.1)
                    # 释放锁
                    released = manager.release_lock(lock_key, worker_id)
                    operations.append(f"{worker_id}: released ({released})")
                else:
                    operations.append(f"{worker_id}: failed to acquire")
            except Exception as e:
                exceptions.append(f"{worker_id}: {e}")

        # 启动多个线程竞争锁
        threads = []
        for i in range(5):
            t = threading.Thread(target=contention_worker, args=(f"worker_{i}",))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 不应该有异常
        assert len(exceptions) == 0, f"Contention exceptions: {exceptions}"

        # 应该有获取和释放操作
        acquire_ops = [op for op in operations if "acquired" in op]
        release_ops = [op for op in operations if "released" in op]

        assert len(acquire_ops) == 5, f"All workers should eventually acquire the lock"
        assert len(release_ops) == 5, f"All acquired locks should be released"

    def test_renewal_under_contention(self, manager):
        """测试争用情况下的锁续期"""
        lock_key = "renewal_lock"
        renewal_results = []
        exceptions = []

        def renewal_worker(worker_id: str):
            """续期锁的工作线程"""
            try:
                # 只有第一个worker能获取锁
                if worker_id == "worker_0":
                    acquired = manager.acquire_lock(lock_key, worker_id, ttl=5)
                    if acquired:
                        # 续期多次
                        for i in range(3):
                            time.sleep(0.1)
                            renewed = manager.renew_lock(lock_key, worker_id, ttl=5)
                            renewal_results.append(f"{worker_id}: renewal {i} = {renewed}")
                        # 最终释放
                        manager.release_lock(lock_key, worker_id)
                    else:
                        renewal_results.append(f"{worker_id}: failed to acquire")
                else:
                    # 其他worker尝试续期（应该失败）
                    renewed = manager.renew_lock(lock_key, worker_id, ttl=5)
                    renewal_results.append(f"{worker_id}: renewal attempt = {renewed}")
            except Exception as e:
                exceptions.append(f"{worker_id}: {e}")

        # 启动线程
        threads = []
        for i in range(4):
            t = threading.Thread(target=renewal_worker, args=(f"worker_{i}",))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        # 不应该有异常
        assert len(exceptions) == 0, f"Renewal exceptions: {exceptions}"

        # 验证续期结果
        successful_renewals = [r for r in renewal_results if "renewal 0 = True" in r or "renewal 1 = True" in r or "renewal 2 = True" in r]
        failed_renewals = [r for r in renewal_results if "= False" in r]

        assert len(successful_renewals) == 3, f"Expected 3 successful renewals, got {len(successful_renewals)}"
        assert len(failed_renewals) == 3, f"Expected 3 failed renewals, got {len(failed_renewals)}"


class TestDistributedLockManagerMonitoring:
    """DistributedLockManager监控测试"""

    @pytest.fixture
    def manager(self):
        """DistributedLockManager fixture"""
        return DistributedLockManager()

    def test_list_active_locks_empty(self, manager):
        """测试列出空的活跃锁"""
        active_locks = manager.list_active_locks()

        assert isinstance(active_locks, list)
        assert len(active_locks) == 0

    def test_list_active_locks_with_locks(self, manager):
        """测试列出活跃锁"""
        # 创建多个锁
        locks = []
        for i in range(3):
            lock_key = f"active_lock_{i}"
            owner = f"owner_{i}"
            manager.acquire_lock(lock_key, owner, ttl=60)
            locks.append((lock_key, owner))

        active_locks = manager.list_active_locks()

        assert len(active_locks) == 3

        # 验证所有锁都在列表中
        lock_ids = {lock.lock_id for lock in active_locks}
        owners = {lock.owner for lock in active_locks}

        expected_lock_ids = {lock_key for lock_key, _ in locks}
        expected_owners = {owner for _, owner in locks}

        assert lock_ids == expected_lock_ids
        assert owners == expected_owners

    def test_get_lock_info_details(self, manager):
        """测试获取锁详细信息"""
        lock_key = "detail_lock"
        owner = "detail_owner"

        # 获取锁
        manager.acquire_lock(lock_key, owner, ttl=45)

        # 获取锁信息
        lock_info = manager.get_lock_info(lock_key)

        assert lock_info is not None
        assert lock_info.lock_id == lock_key
        assert lock_info.owner == owner
        assert lock_info.ttl == 45
        assert lock_info.renew_count == 0
        assert isinstance(lock_info.acquired_time, float)
        assert lock_info.acquired_time > 0

    def test_force_release_lock(self, manager):
        """测试强制释放锁"""
        lock_key = "force_release_lock"
        owner = "original_owner"

        # 获取锁
        manager.acquire_lock(lock_key, owner, ttl=60)

        # 验证锁存在
        lock_info = manager.get_lock_info(lock_key)
        assert lock_info is not None
        assert lock_info.owner == owner

        # 强制释放锁
        result = manager.force_release_lock(lock_key)
        assert result is True

        # 验证锁已被释放
        lock_info = manager.get_lock_info(lock_key)
        assert lock_info is None

    def test_force_release_nonexistent_lock(self, manager):
        """测试强制释放不存在的锁"""
        result = manager.force_release_lock("nonexistent_lock")
        assert result is False


class TestDistributedLockManagerErrorHandling:
    """DistributedLockManager错误处理测试"""

    @pytest.fixture
    def manager(self):
        """DistributedLockManager fixture"""
        return DistributedLockManager()

    def test_lock_operations_with_invalid_parameters(self, manager):
        """测试无效参数的锁操作"""
        # 无效的锁键
        result = manager.acquire_lock("", "owner", ttl=30)
        assert result is False  # 应该拒绝空键

        result = manager.acquire_lock(None, "owner", ttl=30)
        assert result is False  # 应该拒绝None键

        # 无效的owner
        result = manager.acquire_lock("lock", "", ttl=30)
        assert result is False  # 应该拒绝空owner

        # 无效的TTL
        result = manager.acquire_lock("lock", "owner", ttl=0)
        assert result is False  # 应该拒绝零TTL

        result = manager.acquire_lock("lock", "owner", ttl=-1)
        assert result is False  # 应该拒绝负TTL

    def test_lock_operations_under_resource_constraints(self, manager):
        """测试资源约束下的锁操作"""
        # 创建大量锁来测试内存使用
        lock_count = 1000

        for i in range(lock_count):
            lock_key = f"resource_test_lock_{i}"
            owner = f"owner_{i % 10}"  # 10个不同的owner
            manager.acquire_lock(lock_key, owner, ttl=300)

        # 验证所有锁都创建成功
        active_locks = manager.list_active_locks()
        assert len(active_locks) == lock_count

        # 清理所有锁
        for i in range(lock_count):
            lock_key = f"resource_test_lock_{i}"
            owner = f"owner_{i % 10}"
            manager.release_lock(lock_key, owner)

        # 验证所有锁都释放了
        active_locks = manager.list_active_locks()
        assert len(active_locks) == 0


class TestDistributedLockManagerIntegration:
    """DistributedLockManager集成测试"""

    def test_distributed_lock_workflow_simulation(self):
        """测试分布式锁工作流模拟"""
        manager = DistributedLockManager()

        # 模拟一个需要锁保护的资源
        shared_resource = {"counter": 0, "access_log": []}
        resource_lock = "shared_resource_lock"

        def access_resource(worker_id: str, iterations: int):
            """访问共享资源的工作函数"""
            for i in range(iterations):
                # 获取锁
                acquired = manager.acquire_lock(resource_lock, worker_id, ttl=10)
                if acquired:
                    try:
                        # 模拟对资源的独占访问
                        current_value = shared_resource["counter"]
                        time.sleep(0.001)  # 模拟处理时间
                        shared_resource["counter"] = current_value + 1
                        shared_resource["access_log"].append(f"{worker_id}:{i}")
                    finally:
                        # 释放锁
                        manager.release_lock(resource_lock, worker_id)
                else:
                    # 如果获取锁失败，记录失败
                    shared_resource["access_log"].append(f"{worker_id}:{i}:failed")

        # 启动多个线程并发访问资源
        threads = []
        for i in range(5):
            t = threading.Thread(target=access_resource, args=(f"worker_{i}", 20))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        final_counter = shared_resource["counter"]
        access_log = shared_resource["access_log"]

        # 计数器应该等于总迭代次数（100）
        assert final_counter == 100, f"Expected counter=100, got {final_counter}"

        # 不应该有失败的访问
        failed_accesses = [log for log in access_log if "failed" in log]
        assert len(failed_accesses) == 0, f"Found failed accesses: {failed_accesses}"

        # 验证访问日志的完整性
        assert len(access_log) == 100, f"Expected 100 log entries, got {len(access_log)}"

    def test_lock_timeout_and_recovery(self):
        """测试锁超时和恢复"""
        manager = DistributedLockManager()

        lock_key = "timeout_test_lock"
        owner1 = "owner1"
        owner2 = "owner2"

        # owner1获取锁
        acquired1 = manager.acquire_lock(lock_key, owner1, ttl=2)  # 2秒TTL
        assert acquired1 is True

        # 验证owner2无法获取锁
        acquired2 = manager.try_acquire_lock(lock_key, owner2, ttl=30)
        assert acquired2 is False

        # 等待锁过期
        time.sleep(2.5)

        # 手动清理过期锁（在实际实现中这可能是定时任务）
        manager._cleanup_expired_lock(lock_key)

        # 现在owner2应该能够获取锁
        acquired3 = manager.try_acquire_lock(lock_key, owner2, ttl=30)
        assert acquired3 is True

        # 验证锁现在属于owner2
        lock_info = manager.get_lock_info(lock_key)
        assert lock_info.owner == owner2

    def test_performance_under_load(self):
        """测试负载下的性能"""
        manager = DistributedLockManager()

        lock_keys = [f"perf_lock_{i}" for i in range(10)]
        operations = []
        exceptions = []

        def performance_worker(worker_id: str):
            """性能测试线程"""
            try:
                start_time = time.time()
                op_count = 0

                for i in range(100):
                    lock_key = lock_keys[i % len(lock_keys)]

                    # 获取锁
                    if manager.acquire_lock(lock_key, worker_id, ttl=30):
                        op_count += 1
                        # 短暂持有
                        time.sleep(0.001)
                        # 释放锁
                        manager.release_lock(lock_key, worker_id)

                end_time = time.time()
                duration = end_time - start_time
                ops_per_second = op_count / duration if duration > 0 else 0

                operations.append((worker_id, op_count, ops_per_second))

            except Exception as e:
                exceptions.append(f"{worker_id}: {e}")

        # 启动多个性能测试线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=performance_worker, args=(f"perf_worker_{i}",))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        # 不应该有异常
        assert len(exceptions) == 0, f"Performance test exceptions: {exceptions}"

        # 验证性能
        total_operations = sum(count for _, count, _ in operations)
        avg_ops_per_second = sum(ops for _, _, ops in operations) / len(operations)

        assert total_operations == 500, f"Expected 500 total operations, got {total_operations}"
        assert avg_ops_per_second > 30, f"Average performance too low: {avg_ops_per_second} ops/sec"

        print(f"Performance test: {total_operations} operations, {avg_ops_per_second:.1f} ops/sec average")
