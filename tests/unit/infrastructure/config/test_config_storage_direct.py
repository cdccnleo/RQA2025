"""
直接测试配置存储功能的测试文件
测试src/infrastructure/config/storage目录下可独立导入的实际代码
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.config.storage.types.consistencylevel import ConsistencyLevel
from src.infrastructure.config.storage.types.storagetype import StorageType
from src.infrastructure.config.storage.types.configitem import ConfigItem


class TestConfigStorageDirect:
    """直接测试配置存储功能"""

    def test_storage_type_enum(self):
        """测试StorageType枚举"""
        assert StorageType.MEMORY.value == "memory"
        assert StorageType.FILE.value == "file"
        assert StorageType.DATABASE.value == "database"
        assert StorageType.REDIS.value == "redis"

    def test_consistency_level_enum(self):
        """测试ConsistencyLevel枚举"""
        assert ConsistencyLevel.STRONG.value == "strong"
        assert ConsistencyLevel.EVENTUAL.value == "eventual"
        assert ConsistencyLevel.CAUSAL.value == "causal"

    def test_config_item_creation(self):
        """测试ConfigItem创建"""
        import time
        item = ConfigItem("test_key", "test_value", scope="global", timestamp=time.time())
        assert item.key == "test_key"
        assert item.value == "test_value"
        assert item.scope == "global"

    def test_config_item_with_metadata(self):
        """测试ConfigItem带元数据"""
        import time
        metadata = {"description": "test config", "version": "1.0"}
        item = ConfigItem("test_key", "test_value", scope="global", timestamp=time.time(), metadata=metadata)
        assert item.metadata == metadata
        assert item.metadata.get("description") == "test config"
        assert item.metadata.get("version") == "1.0"
