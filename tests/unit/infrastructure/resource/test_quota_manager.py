import pytest
import time
import threading
from datetime import datetime
from unittest.mock import patch
from src.infrastructure.resource.quota_manager import QuotaManager

# Fixtures
@pytest.fixture
def quota_manager():
    """QuotaManager测试实例"""
    return QuotaManager()

# 测试用例
class TestQuotaManager:
    def test_set_and_get_quota(self, quota_manager):
        """测试配额设置和获取"""
        # 设置配额
        quota_manager.set_quota(
            strategy="strategy1",
            cpu_percent=30.0,
            gpu_memory_mb=2048,
            max_workers=5
        )

        # 验证配额设置
        quota = quota_manager.get_quota("strategy1")
        assert quota['cpu_percent'] == 30.0
        assert quota['gpu_memory_mb'] == 2048
        assert quota['max_workers'] == 5

        # 更新部分配额
        quota_manager.set_quota(
            strategy="strategy1",
            cpu_percent=40.0
        )
        quota = quota_manager.get_quota("strategy1")
        assert quota['cpu_percent'] == 40.0  # 更新
        assert quota['gpu_memory_mb'] == 2048  # 保持不变

    def test_check_quota(self, quota_manager):
        """测试配额检查"""
        # 设置配额
        quota_manager.set_quota(
            strategy="strategy1",
            cpu_percent=30.0,
            gpu_memory_mb=2048,
            max_workers=2
        )

        # 测试CPU配额
        assert quota_manager.check_quota("strategy1", current_cpu=25.0) is True
        assert quota_manager.check_quota("strategy1", current_cpu=35.0) is False

        # 测试GPU配额
        assert quota_manager.check_quota("strategy1", current_gpu_mem=1024) is True
        assert quota_manager.check_quota("strategy1", current_gpu_mem=3072) is False

        # 测试工作线程配额
        quota_manager.register_worker("strategy1", "worker1")
        quota_manager.register_worker("strategy1", "worker2")
        assert quota_manager.check_quota("strategy1") is False  # 达到最大工作线程数

    def test_worker_registration(self, quota_manager):
        """测试工作线程注册/注销"""
        # 设置配额
        quota_manager.set_quota(
            strategy="strategy1",
            max_workers=2
        )

        # 注册工作线程
        assert quota_manager.register_worker("strategy1", "worker1") is True
        assert quota_manager.register_worker("strategy1", "worker2") is True
        assert quota_manager.register_worker("strategy1", "worker3") is False  # 超过配额

        # 获取资源使用情况
        usage = quota_manager.get_resource_usage("strategy1")
        assert usage['workers'] == 2

        # 注销工作线程
        quota_manager.unregister_worker("strategy1", "worker1")
        usage = quota_manager.get_resource_usage("strategy1")
        assert usage['workers'] == 1

        # 可以注册新工作线程
        assert quota_manager.register_worker("strategy1", "worker3") is True

    def test_no_quota_strategy(self, quota_manager):
        """测试无配额策略"""
        assert quota_manager.check_quota("no_quota_strategy") is True
        assert quota_manager.register_worker("no_quota_strategy", "worker1") is True
        assert quota_manager.get_quota("no_quota_strategy") == {}

    def test_thread_safety(self, quota_manager):
        """测试线程安全性"""
        quota_manager.set_quota(
            strategy="concurrent_strategy",
            max_workers=100
        )

        results = []

        def worker(worker_id):
            time.sleep(0.01)  # 增加竞争概率
            result = quota_manager.register_worker("concurrent_strategy", f"worker_{worker_id}")
            results.append(result)

        # 创建多个线程并发注册
        threads = []
        for i in range(150):  # 超过配额限制
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 验证只有100个工作线程注册成功
        assert sum(results) == 100
        usage = quota_manager.get_resource_usage("concurrent_strategy")
        assert usage['workers'] == 100

    def test_partial_quota(self, quota_manager):
        """测试部分配额设置"""
        # 只设置CPU配额
        quota_manager.set_quota(
            strategy="partial_strategy",
            cpu_percent=50.0
        )

        # 检查其他资源不受限
        assert quota_manager.check_quota(
            "partial_strategy",
            current_cpu=60.0  # 超过CPU配额
        ) is False

        assert quota_manager.check_quota(
            "partial_strategy",
            current_cpu=40.0,
            current_gpu_mem=9999  # GPU无配额限制
        ) is True

        # 工作线程无限制
        for i in range(10):
            quota_manager.register_worker("partial_strategy", f"worker_{i}")
        usage = quota_manager.get_resource_usage("partial_strategy")
        assert usage['workers'] == 10

    def test_resource_usage_timestamp(self, quota_manager):
        """测试资源使用时间戳"""
        test_time = "2023-01-01T00:00:00"
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = test_time
            usage = quota_manager.get_resource_usage("any_strategy")
            assert usage['timestamp'] == test_time
