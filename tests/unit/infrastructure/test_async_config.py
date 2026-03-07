"""
测试异步配置管理器

覆盖 async_config.py 中的 AsyncConfigManager 类
"""

import pytest
from unittest.mock import AsyncMock
from src.infrastructure.async_config import AsyncConfigManager, AsyncConfig


class TestAsyncConfigManager:
    """AsyncConfigManager 类测试"""

    def test_initialization(self):
        """测试初始化"""
        manager = AsyncConfigManager()

        assert manager.configs == {}
        assert isinstance(manager.configs, dict)

    def test_async_config_alias(self):
        """测试AsyncConfig别名"""
        # AsyncConfig 应该是 AsyncConfigManager 的别名
        assert AsyncConfig is AsyncConfigManager

        config = AsyncConfig()
        assert isinstance(config, AsyncConfigManager)

    @pytest.mark.asyncio
    async def test_get_config_existing_key(self):
        """测试获取存在的配置"""
        manager = AsyncConfigManager()
        manager.configs = {"database_url": "postgresql://localhost/test"}

        result = await manager.get_config("database_url")

        assert result == "postgresql://localhost/test"

    @pytest.mark.asyncio
    async def test_get_config_nonexistent_key(self):
        """测试获取不存在的配置"""
        manager = AsyncConfigManager()

        result = await manager.get_config("nonexistent_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_config(self):
        """测试设置配置"""
        manager = AsyncConfigManager()

        await manager.set_config("api_key", "secret123")

        assert manager.configs["api_key"] == "secret123"

    @pytest.mark.asyncio
    async def test_set_config_overwrite(self):
        """测试覆盖配置"""
        manager = AsyncConfigManager()

        # 设置初始值
        await manager.set_config("timeout", 30)
        assert manager.configs["timeout"] == 30

        # 覆盖值
        await manager.set_config("timeout", 60)
        assert manager.configs["timeout"] == 60

    @pytest.mark.asyncio
    async def test_multiple_configs(self):
        """测试多个配置项"""
        manager = AsyncConfigManager()

        # 设置多个配置
        await manager.set_config("host", "localhost")
        await manager.set_config("port", 8080)
        await manager.set_config("debug", True)

        # 验证所有配置
        assert await manager.get_config("host") == "localhost"
        assert await manager.get_config("port") == 8080
        assert await manager.get_config("debug") == True

    @pytest.mark.asyncio
    async def test_config_types(self):
        """测试不同类型的配置值"""
        manager = AsyncConfigManager()

        test_cases = [
            ("string", "hello world"),
            ("number", 42),
            ("boolean", True),
            ("list", [1, 2, 3]),
            ("dict", {"key": "value"}),
            ("none", None),
        ]

        for key, value in test_cases:
            await manager.set_config(key, value)
            result = await manager.get_config(key)
            assert result == value

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """测试并发访问"""
        import asyncio
        manager = AsyncConfigManager()

        async def set_and_get(key, value):
            await manager.set_config(key, value)
            return await manager.get_config(key)

        # 创建多个并发任务
        tasks = []
        for i in range(10):
            task = asyncio.create_task(set_and_get(f"key_{i}", f"value_{i}"))
            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks)

        # 验证结果
        for i, result in enumerate(results):
            assert result == f"value_{i}"

    @pytest.mark.asyncio
    async def test_config_isolation(self):
        """测试配置隔离"""
        manager1 = AsyncConfigManager()
        manager2 = AsyncConfigManager()

        # 在manager1中设置配置
        await manager1.set_config("shared_key", "value1")

        # 在manager2中设置不同的值
        await manager2.set_config("shared_key", "value2")

        # 验证隔离性
        assert await manager1.get_config("shared_key") == "value1"
        assert await manager2.get_config("shared_key") == "value2"

    @pytest.mark.asyncio
    async def test_empty_key_handling(self):
        """测试空键处理"""
        manager = AsyncConfigManager()

        # 设置空字符串键
        await manager.set_config("", "empty_key_value")
        result = await manager.get_config("")
        assert result == "empty_key_value"

        # 获取不存在的空键
        result = await manager.get_config("")
        assert result == "empty_key_value"

    @pytest.mark.asyncio
    async def test_special_characters_in_keys(self):
        """测试键中的特殊字符"""
        manager = AsyncConfigManager()

        special_keys = [
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
            "key with spaces",
            "key@domain.com"
        ]

        for key in special_keys:
            await manager.set_config(key, f"value_for_{key}")
            result = await manager.get_config(key)
            assert result == f"value_for_{key}"

    @pytest.mark.asyncio
    async def test_large_config_values(self):
        """测试大配置值"""
        manager = AsyncConfigManager()

        # 大字符串
        large_string = "x" * 10000
        await manager.set_config("large_string", large_string)
        result = await manager.get_config("large_string")
        assert result == large_string

        # 大列表
        large_list = list(range(1000))
        await manager.set_config("large_list", large_list)
        result = await manager.get_config("large_list")
        assert result == large_list

    @pytest.mark.asyncio
    async def test_config_persistence(self):
        """测试配置持久性"""
        manager = AsyncConfigManager()

        # 设置一些配置
        configs = {
            "app_name": "MyApp",
            "version": "1.0.0",
            "features": ["auth", "cache", "logging"],
            "settings": {"debug": True, "timeout": 30}
        }

        for key, value in configs.items():
            await manager.set_config(key, value)

        # 验证所有配置都存在
        for key, expected_value in configs.items():
            actual_value = await manager.get_config(key)
            assert actual_value == expected_value

        # 验证内部状态
        assert len(manager.configs) == len(configs)

    @pytest.mark.asyncio
    async def test_async_operations_are_actually_async(self):
        """测试异步操作确实是异步的"""
        import asyncio
        manager = AsyncConfigManager()

        # 记录开始时间
        start_time = asyncio.get_event_loop().time()

        # 执行一些异步操作
        tasks = []
        for i in range(100):
            task = asyncio.create_task(manager.set_config(f"key_{i}", f"value_{i}"))
            tasks.append(task)

        await asyncio.gather(*tasks)

        # 验证所有配置都被设置
        for i in range(100):
            result = await manager.get_config(f"key_{i}")
            assert result == f"value_{i}"

        # 验证执行时间合理（异步操作应该很快完成）
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        assert execution_time < 1.0  # 应该在1秒内完成

    @pytest.mark.asyncio
    async def test_error_handling_in_async_operations(self):
        """测试异步操作中的错误处理"""
        manager = AsyncConfigManager()

        # 测试正常操作不会抛出异常
        try:
            await manager.set_config("test_key", "test_value")
            result = await manager.get_config("test_key")
            assert result == "test_value"
        except Exception as e:
            pytest.fail(f"Async operation failed unexpectedly: {e}")

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """测试内存效率"""
        import sys
        manager = AsyncConfigManager()

        # 记录初始内存使用
        initial_memory = sys.getsizeof(manager.configs)

        # 添加许多配置
        for i in range(1000):
            await manager.set_config(f"key_{i}", f"value_{i}")

        # 记录添加后的内存使用
        final_memory = sys.getsizeof(manager.configs)

        # 内存使用应该合理增长
        memory_growth = final_memory - initial_memory
        assert memory_growth > 0  # 应该有内存增长
        assert memory_growth < 100000  # 但不应该过大（合理的字典大小）

    @pytest.mark.asyncio
    async def test_thread_safety_simulation(self):
        """测试线程安全性模拟"""
        import asyncio
        manager = AsyncConfigManager()

        async def worker(worker_id, num_operations):
            """模拟工作线程"""
            for i in range(num_operations):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"

                await manager.set_config(key, value)

                # 偶尔读取以验证一致性
                if i % 10 == 0:
                    retrieved = await manager.get_config(key)
                    assert retrieved == value

        # 创建多个"线程"模拟并发
        num_workers = 10
        operations_per_worker = 50

        tasks = []
        for worker_id in range(num_workers):
            task = asyncio.create_task(worker(worker_id, operations_per_worker))
            tasks.append(task)

        # 等待所有任务完成
        await asyncio.gather(*tasks)

        # 验证最终状态
        total_expected_keys = num_workers * operations_per_worker
        assert len(manager.configs) == total_expected_keys

        # 随机检查一些键值对
        for worker_id in range(min(3, num_workers)):  # 检查前3个worker
            for i in range(0, operations_per_worker, 10):  # 每10个检查一个
                key = f"worker_{worker_id}_key_{i}"
                expected_value = f"worker_{worker_id}_value_{i}"
                actual_value = await manager.get_config(key)
                assert actual_value == expected_value