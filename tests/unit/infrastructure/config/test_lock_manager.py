import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from src.infrastructure.config.services.lock_manager import LockManager

class TestLockManager:
    """锁管理测试套件"""

    @pytest.fixture
    def lock_manager(self):
        return LockManager()

    @pytest.mark.unit
    def test_basic_lock_operations(self, lock_manager):
        """测试基本锁操作"""
        # 获取锁
        assert lock_manager.acquire("test_lock")
        # 验证锁状态
        assert lock_manager.is_locked("test_lock")
        # 释放锁
        assert lock_manager.release("test_lock")
        assert not lock_manager.is_locked("test_lock")

    @pytest.mark.unit
    def test_lock_timeout(self, lock_manager):
        """测试锁超时机制"""
        # 第一个线程获取锁并保持
        def holder():
            assert lock_manager.acquire("timeout_lock")
            time.sleep(2)  # 保持锁超过超时时间
            lock_manager.release("timeout_lock")

        holder_thread = threading.Thread(target=holder)
        holder_thread.start()

        time.sleep(0.5)  # 确保holder先获取锁

        # 测试超时获取失败
        result = []
        event = threading.Event()

        def worker():
            acquired = lock_manager.acquire("timeout_lock", timeout=0.5)
            result.append(acquired)
            event.set()

        t = threading.Thread(target=worker)
        t.start()
        event.wait(timeout=1.0)

        assert len(result) == 1
        assert not result[0]  # 应超时失败
        holder_thread.join()

    @pytest.mark.unit
    def test_deadlock_detection(self, lock_manager):
        """测试死锁检测"""
        lock_manager.acquire("lock_A")
        deadlock_detected = []

        def worker():
            try:
                if lock_manager.acquire("lock_B"):
                    try:
                        if not lock_manager.acquire("lock_A", timeout=0.1):
                            deadlock_detected.append(True)
                    finally:
                        lock_manager.release("lock_B")
            except Exception:
                pass

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert deadlock_detected  # 死锁被检测到
        lock_manager.release("lock_A")

    @pytest.mark.unit
    def test_lock_statistics(self, lock_manager):
        """测试锁统计信息"""
        for _ in range(3):
            lock_manager.acquire("stat_lock")
            time.sleep(0.01)
            lock_manager.release("stat_lock")

        stats = lock_manager.get_stats("stat_lock")
        assert stats["acquire_count"] == 3
        assert stats["hold_time"] > 0

    @pytest.mark.unit
    def test_concurrent_operations(self, lock_manager):
        """测试并发锁操作"""
        counter = 0
        lock_name = "concurrent_lock"

        def worker():
            nonlocal counter
            if lock_manager.acquire(lock_name):
                try:
                    current = counter
                    time.sleep(0.01)
                    counter = current + 1
                finally:
                    lock_manager.release(lock_name)

        # 使用线程池测试并发
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(100)]
            for f in futures:
                f.result()

        assert counter == 100  # 确保没有竞态条件
        stats = lock_manager.get_stats(lock_name)
        assert stats["contention_count"] > 0  # 应有争用记录

    @pytest.mark.stress
    def test_high_contention(self, lock_manager):
        """测试高竞争场景下的锁性能"""
        start_time = time.time()
        self.test_concurrent_operations(lock_manager)
        duration = time.time() - start_time
        print(f"\n高竞争测试耗时: {duration:.2f}秒")
