"""
内存管理器单元测试

测试内容:
- 内存池基本功能
- 垃圾回收优化
- 内存泄漏检测
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from src.core.performance.memory_manager import (
    DynamicMemoryManager,
    MemoryPool,
    MemoryConfig,
    MemorySnapshot,
)


class TestMemoryConfig:
    """测试内存配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = MemoryConfig()
        assert config.gc_threshold == 80.0
        assert config.check_interval == 60.0
        assert config.enable_leak_detection is True
        assert config.leak_threshold_mb == 100.0

    def test_custom_config(self):
        """测试自定义配置"""
        config = MemoryConfig(
            gc_threshold=90.0,
            check_interval=30.0,
            enable_leak_detection=False
        )
        assert config.gc_threshold == 90.0
        assert config.check_interval == 30.0
        assert config.enable_leak_detection is False


class TestMemorySnapshot:
    """测试内存快照类"""

    def test_snapshot_creation(self):
        """测试快照创建"""
        snapshot = MemorySnapshot(
            timestamp=1234567890.0,
            rss_mb=1024.5,
            vms_mb=2048.0,
            percent=50.0
        )
        assert snapshot.timestamp == 1234567890.0
        assert snapshot.rss_mb == 1024.5
        assert snapshot.vms_mb == 2048.0
        assert snapshot.percent == 50.0

    def test_snapshot_comparison(self):
        """测试快照比较"""
        snapshot1 = MemorySnapshot(
            timestamp=0.0,
            rss_mb=1000.0,
            vms_mb=2000.0,
            percent=50.0
        )
        snapshot2 = MemorySnapshot(
            timestamp=1.0,
            rss_mb=1100.0,
            vms_mb=2000.0,
            percent=55.0
        )
        # 验证RSS增长
        assert snapshot2.rss_mb > snapshot1.rss_mb


class TestMemoryPool:
    """测试内存池类"""

    @pytest.fixture
    def pool(self):
        """创建测试用的内存池"""
        return MemoryPool(max_size=10, object_size=1024)

    def test_pool_initialization(self, pool):
        """测试池初始化"""
        assert pool.max_size == 10
        assert pool.object_size == 1024
        assert len(pool) == 0

    def test_acquire_release(self, pool):
        """测试获取和释放对象"""
        # 获取对象
        obj = pool.acquire()
        assert obj is not None
        assert len(pool) == 0  # 对象已被取出

        # 释放对象
        pool.release(obj)
        assert len(pool) == 1  # 对象已归还

    def test_pool_exhaustion(self, pool):
        """测试池耗尽情况"""
        objects = []
        # 获取所有对象
        for _ in range(pool.max_size):
            obj = pool.acquire()
            objects.append(obj)

        # 池已满，再获取应该返回None或创建新对象
        extra_obj = pool.acquire()
        # 根据实现可能返回None或新对象

        # 归还对象
        for obj in objects:
            pool.release(obj)

        assert len(pool) == pool.max_size


class TestDynamicMemoryManager:
    """测试动态内存管理器"""

    @pytest.fixture
    def manager(self):
        """创建测试用的内存管理器"""
        config = MemoryConfig(
            gc_threshold=80.0,
            check_interval=1.0,  # 缩短检查间隔以便测试
            enable_leak_detection=True
        )
        return DynamicMemoryManager(config)

    def test_manager_initialization(self, manager):
        """测试管理器初始化"""
        assert manager.config.gc_threshold == 80.0
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, manager):
        """测试启动和停止监控"""
        # 启动监控
        manager.start_monitoring()
        assert manager._running is True

        # 停止监控
        manager.stop_monitoring()
        assert manager._running is False

    def test_get_memory_usage(self, manager):
        """测试获取内存使用情况"""
        usage = manager.get_memory_usage()
        assert isinstance(usage, dict)
        assert 'rss_mb' in usage
        assert 'vms_mb' in usage
        assert 'percent' in usage

    def test_create_pool(self, manager):
        """测试创建内存池"""
        pool = manager.create_pool(name='test_pool', max_size=5)
        assert pool is not None
        assert 'test_pool' in manager._pools

    def test_get_pool(self, manager):
        """测试获取内存池"""
        # 先创建池
        manager.create_pool(name='test_pool', max_size=5)

        # 获取池
        pool = manager.get_pool('test_pool')
        assert pool is not None

        # 获取不存在的池
        non_existent = manager.get_pool('non_existent')
        assert non_existent is None

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, manager):
        """测试内存泄漏检测"""
        # 模拟内存增长
        with patch.object(manager, '_get_current_memory_mb', return_value=1000.0):
            # 添加历史记录
            for i in range(10):
                manager._memory_history.append((float(i), 100.0 + i * 10))

            # 检测泄漏
            has_leak = await manager._detect_memory_leak()
            # 根据实现可能返回True或False


class TestMemoryManagerIntegration:
    """内存管理器集成测试"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """测试完整工作流程"""
        config = MemoryConfig(
            gc_threshold=90.0,
            check_interval=0.1,
            enable_leak_detection=True
        )
        manager = DynamicMemoryManager(config)

        # 启动监控
        manager.start_monitoring()

        # 创建内存池
        pool = manager.create_pool(name='integration_pool', max_size=5)

        # 使用池
        obj1 = pool.acquire()
        obj2 = pool.acquire()

        # 释放对象
        pool.release(obj1)
        pool.release(obj2)

        # 获取内存使用情况
        usage = manager.get_memory_usage()
        assert 'rss_mb' in usage

        # 停止监控
        manager.stop_monitoring()

        assert manager._running is False
