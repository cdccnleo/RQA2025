"""
测试异步配置管理器
"""

import pytest
import asyncio

from src.infrastructure.async_config import AsyncConfigManager, AsyncConfig


class TestAsyncConfigManager:
    """测试异步配置管理器"""

    def setup_method(self):
        """测试前准备"""
        self.config_manager = AsyncConfigManager()

    def test_async_config_manager_init(self):
        """测试异步配置管理器初始化"""
        assert self.config_manager is not None
        assert hasattr(self.config_manager, 'configs')
        assert isinstance(self.config_manager.configs, dict)
        assert len(self.config_manager.configs) == 0

    @pytest.mark.asyncio
    async def test_get_config_existing(self):
        """测试获取存在的配置"""
        # 先设置配置
        await self.config_manager.set_config("test_key", "test_value")

        # 获取配置
        value = await self.config_manager.get_config("test_key")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_get_config_nonexistent(self):
        """测试获取不存在的配置"""
        value = await self.config_manager.get_config("nonexistent_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_config(self):
        """测试设置配置"""
        key = "database_url"
        value = "postgresql://localhost:5432/rqa2025"

        await self.config_manager.set_config(key, value)

        # 验证配置已设置
        assert key in self.config_manager.configs
        assert self.config_manager.configs[key] == value

    @pytest.mark.asyncio
    async def test_set_config_override(self):
        """测试覆盖设置配置"""
        key = "api_timeout"

        # 第一次设置
        await self.config_manager.set_config(key, 30)
        assert self.config_manager.configs[key] == 30

        # 第二次设置，覆盖原有值
        await self.config_manager.set_config(key, 60)
        assert self.config_manager.configs[key] == 60

    @pytest.mark.asyncio
    async def test_multiple_configs(self):
        """测试多个配置项"""
        configs = {
            "database_host": "localhost",
            "database_port": 5432,
            "api_timeout": 30,
            "debug_mode": True,
            "max_connections": 100
        }

        # 设置多个配置
        for key, value in configs.items():
            await self.config_manager.set_config(key, value)

        # 验证所有配置都已设置
        assert len(self.config_manager.configs) == len(configs)

        # 逐个验证配置值
        for key, expected_value in configs.items():
            actual_value = await self.config_manager.get_config(key)
            assert actual_value == expected_value

    @pytest.mark.asyncio
    async def test_config_types(self):
        """测试不同类型的配置值"""
        test_configs = {
            "string_config": "hello world",
            "int_config": 42,
            "float_config": 3.14,
            "bool_config": True,
            "list_config": [1, 2, 3, 4, 5],
            "dict_config": {"nested": "value", "count": 10},
            "none_config": None
        }

        # 设置不同类型的配置
        for key, value in test_configs.items():
            await self.config_manager.set_config(key, value)

        # 验证所有类型的配置都能正确存储和检索
        for key, expected_value in test_configs.items():
            actual_value = await self.config_manager.get_config(key)
            assert actual_value == expected_value
            assert type(actual_value) == type(expected_value)

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """测试并发访问"""
        async def set_and_get_config(index):
            key = f"config_{index}"
            value = f"value_{index}"

            # 设置配置
            await self.config_manager.set_config(key, value)

            # 获取配置
            retrieved_value = await self.config_manager.get_config(key)

            return retrieved_value == value

        # 创建多个并发任务
        tasks = [set_and_get_config(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # 验证所有并发操作都成功
        assert all(results)
        assert len(self.config_manager.configs) == 10

    @pytest.mark.asyncio
    async def test_config_persistence(self):
        """测试配置持久性"""
        # 设置一些配置
        configs = {"temp_key": "temp_value", "persistent_key": "persistent_value"}
        for key, value in configs.items():
            await self.config_manager.set_config(key, value)

        # 模拟重新创建管理器（在实际应用中，这可能从持久化存储加载）
        new_manager = AsyncConfigManager()

        # 手动复制配置来模拟持久化
        new_manager.configs = self.config_manager.configs.copy()

        # 验证配置在新的管理器实例中仍然存在
        for key, expected_value in configs.items():
            actual_value = await new_manager.get_config(key)
            assert actual_value == expected_value

    @pytest.mark.asyncio
    async def test_empty_key_handling(self):
        """测试空键处理"""
        # 设置空字符串键
        await self.config_manager.set_config("", "empty_key_value")

        value = await self.config_manager.get_config("")
        assert value == "empty_key_value"

    @pytest.mark.asyncio
    async def test_special_characters_in_keys(self):
        """测试键中的特殊字符"""
        special_keys = {
            "key-with-dashes": "value1",
            "key_with_underscores": "value2",
            "key.with.dots": "value3",
            "key with spaces": "value4",
            "key@with@symbols": "value5"
        }

        # 设置包含特殊字符的键
        for key, value in special_keys.items():
            await self.config_manager.set_config(key, value)

        # 验证都能正确存储和检索
        for key, expected_value in special_keys.items():
            actual_value = await self.config_manager.get_config(key)
            assert actual_value == expected_value

    @pytest.mark.asyncio
    async def test_large_config_values(self):
        """测试大型配置值"""
        # 创建一个大型配置值
        large_value = {
            "data": list(range(1000)),  # 大型列表
            "nested": {
                "level1": {
                    "level2": {
                        "level3": "deep_value",
                        "large_array": [i**2 for i in range(100)]
                    }
                }
            },
            "description": "A very large configuration value for testing"
        }

        await self.config_manager.set_config("large_config", large_value)

        retrieved_value = await self.config_manager.get_config("large_config")

        assert retrieved_value == large_value
        assert len(retrieved_value["data"]) == 1000
        assert retrieved_value["nested"]["level1"]["level2"]["level3"] == "deep_value"


class TestAsyncConfig:
    """测试AsyncConfig别名"""

    def test_async_config_alias(self):
        """测试AsyncConfig是AsyncConfigManager的别名"""
        # AsyncConfig应该是AsyncConfigManager的别名
        assert AsyncConfig is AsyncConfigManager

        # 可以通过AsyncConfig创建实例
        config = AsyncConfig()
        assert isinstance(config, AsyncConfigManager)

    @pytest.mark.asyncio
    async def test_async_config_functionality(self):
        """测试AsyncConfig的功能性"""
        config = AsyncConfig()

        # 测试基本功能
        await config.set_config("test_key", "test_value")
        value = await config.get_config("test_key")

        assert value == "test_value"


class TestAsyncConfigManagerIntegration:
    """测试异步配置管理器集成场景"""

    def setup_method(self):
        """测试前准备"""
        self.config_manager = AsyncConfigManager()

    @pytest.mark.asyncio
    async def test_configuration_workflow(self):
        """测试配置工作流"""
        # 1. 初始化配置
        await self.config_manager.set_config("app.name", "RQA2025")
        await self.config_manager.set_config("app.version", "2.0.0")
        await self.config_manager.set_config("app.debug", True)

        # 2. 数据库配置
        await self.config_manager.set_config("database.host", "localhost")
        await self.config_manager.set_config("database.port", 5432)
        await self.config_manager.set_config("database.name", "rqa2025")

        # 3. API配置
        await self.config_manager.set_config("api.host", "0.0.0.0")
        await self.config_manager.set_config("api.port", 8000)
        await self.config_manager.set_config("api.timeout", 30)

        # 验证配置完整性
        app_name = await self.config_manager.get_config("app.name")
        db_host = await self.config_manager.get_config("database.host")
        api_port = await self.config_manager.get_config("api.port")

        assert app_name == "RQA2025"
        assert db_host == "localhost"
        assert api_port == 8000

        # 验证配置数量
        assert len(self.config_manager.configs) == 9

    @pytest.mark.asyncio
    async def test_dynamic_configuration_update(self):
        """测试动态配置更新"""
        # 初始配置
        await self.config_manager.set_config("feature.enabled", True)
        await self.config_manager.set_config("threshold.value", 100)

        # 运行时更新配置
        await self.config_manager.set_config("feature.enabled", False)
        await self.config_manager.set_config("threshold.value", 150)

        # 验证更新后的配置
        feature_enabled = await self.config_manager.get_config("feature.enabled")
        threshold_value = await self.config_manager.get_config("threshold.value")

        assert feature_enabled == False
        assert threshold_value == 150

    @pytest.mark.asyncio
    async def test_empty_key_handling(self):
        """测试空键处理"""
        # 测试空字符串键
        await self.config_manager.set_config("", "empty_key_value")
        value = await self.config_manager.get_config("")
        assert value == "empty_key_value"

        # 测试None键（如果允许的话）
        try:
            await self.config_manager.set_config(None, "none_key_value")
            none_value = await self.config_manager.get_config(None)
            assert none_value == "none_key_value"
        except TypeError:
            # 如果不支持None键，这是预期的行为
            pass

    @pytest.mark.asyncio
    async def test_special_characters_in_keys(self):
        """测试键中的特殊字符"""
        special_keys = {
            "key.with.dots": "dot_value",
            "key-with-dashes": "dash_value",
            "key_with_underscores": "underscore_value",
            "key with spaces": "space_value",
            "key@symbol": "symbol_value"
        }

        # 设置包含特殊字符的键
        for key, value in special_keys.items():
            await self.config_manager.set_config(key, value)

        # 验证所有特殊字符键都能正确存储和检索
        for key, expected_value in special_keys.items():
            actual_value = await self.config_manager.get_config(key)
            assert actual_value == expected_value

    @pytest.mark.asyncio
    async def test_large_config_values(self):
        """测试大配置值"""
        # 测试大数据
        large_list = list(range(10000))
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        large_string = "x" * 100000  # 100KB字符串

        # 设置大配置值
        await self.config_manager.set_config("large_list", large_list)
        await self.config_manager.set_config("large_dict", large_dict)
        await self.config_manager.set_config("large_string", large_string)

        # 验证大配置值能正确存储和检索
        retrieved_list = await self.config_manager.get_config("large_list")
        retrieved_dict = await self.config_manager.get_config("large_dict")
        retrieved_string = await self.config_manager.get_config("large_string")

        assert retrieved_list == large_list
        assert retrieved_dict == large_dict
        assert retrieved_string == large_string
