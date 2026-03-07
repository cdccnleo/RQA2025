#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层配置存储测试
专门用于提高配置存储模块的测试覆盖率
"""

import pytest
import os
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入基础设施层配置存储模块
try:
    from src.infrastructure.config.storage.config_storage import (
        FileConfigStorage, MemoryConfigStorage, DistributedConfigStorage,
        ConfigScope, StorageType, DistributedStorageType, ConsistencyLevel,
        StorageConfig, ConfigItem, ConfigStorage
    )
    
    # 创建工厂函数
    def create_file_storage(path, **kwargs):
        """创建文件存储实例"""
        config = StorageConfig()
        config.type = StorageType.FILE
        config.path = path
        # 设置其他配置参数
        config.backup_enabled = kwargs.get('backup_enabled', True)
        config.max_backups = kwargs.get('max_backups', 10)
        return FileConfigStorage(config)
        
    def create_memory_storage():
        """创建内存存储实例"""
        config = StorageConfig()
        config.type = StorageType.MEMORY
        return MemoryConfigStorage(config)
        
    def create_distributed_storage(storage_type="redis", **kwargs):
        """创建分布式存储实例"""
        config = StorageConfig()
        config.type = StorageType.DISTRIBUTED
        if storage_type.lower() == "redis":
            config.distributed_type = DistributedStorageType.REDIS
        elif storage_type.lower() == "etcd":
            config.distributed_type = DistributedStorageType.ETCD
        elif storage_type.lower() == "consul":
            config.distributed_type = DistributedStorageType.CONSUL
        elif storage_type.lower() == "zookeeper":
            config.distributed_type = DistributedStorageType.ZOOKEEPER
        else:
            config.distributed_type = DistributedStorageType.REDIS
        config.consistency_level = kwargs.get('consistency_level', ConsistencyLevel.EVENTUAL)
        return DistributedConfigStorage(config)
        
    def create_storage(storage_type, **kwargs):
        """创建存储实例的通用函数"""
        if storage_type == "file":
            path = kwargs.get("path", "test_config.json")
            return create_file_storage(path, **kwargs)
        elif storage_type == "memory":
            return create_memory_storage()
        elif storage_type == "distributed":
            storage_type_param = kwargs.get("storage_type", "redis")
            return create_distributed_storage(storage_type_param, **kwargs)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    STORAGE_AVAILABLE = True
except ImportError as e:
    print(f"无法导入配置存储模块: {e}")
    STORAGE_AVAILABLE = False
    FileConfigStorage = None
    MemoryConfigStorage = None
    DistributedConfigStorage = None
    ConfigScope = None
    StorageType = None
    DistributedStorageType = None
    ConsistencyLevel = None
    StorageConfig = None
    ConfigItem = None
    ConfigStorage = None
    create_file_storage = None
    create_memory_storage = None
    create_distributed_storage = None
    create_storage = None


class TestFileConfigStorage:
    """文件配置存储测试"""

    def setup_method(self):
        """测试前设置"""
        if STORAGE_AVAILABLE and FileConfigStorage is not None and StorageConfig is not None and StorageType is not None:
            self.test_dir = tempfile.mkdtemp()
            self.test_file = os.path.join(self.test_dir, 'test_config.json')
            config = StorageConfig(
                type=StorageType.FILE,
                path=self.test_file
            )
            self.storage = FileConfigStorage(config)

    def teardown_method(self):
        """测试后清理"""
        if STORAGE_AVAILABLE and hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)

    def test_file_storage_initialization(self):
        """测试文件存储初始化"""
        if not STORAGE_AVAILABLE or FileConfigStorage is None or StorageType is None:
            pytest.skip("配置存储模块不可用")
            
        assert self.storage is not None
        assert self.storage.config.type == StorageType.FILE
        assert self.storage.config.path == self.test_file

    def test_file_storage_set_get(self):
        """测试文件存储设置和获取配置"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        # 测试设置配置
        result = self.storage.set('test_key', 'test_value', ConfigScope.APPLICATION)
        assert result == True
        
        # 测试获取配置
        value = self.storage.get('test_key', ConfigScope.APPLICATION)
        assert value == 'test_value'
        
        # 测试不存在的配置
        value = self.storage.get('nonexistent_key', ConfigScope.APPLICATION)
        assert value is None

    def test_file_storage_delete(self):
        """测试文件存储删除配置"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        # 先设置配置
        self.storage.set('delete_key', 'delete_value', ConfigScope.APPLICATION)
        assert self.storage.get('delete_key', ConfigScope.APPLICATION) == 'delete_value'
        
        # 删除配置
        result = self.storage.delete('delete_key', ConfigScope.APPLICATION)
        assert result == True
        
        # 验证配置已删除
        value = self.storage.get('delete_key', ConfigScope.APPLICATION)
        assert value is None
        
        # 删除不存在的配置
        result = self.storage.delete('nonexistent_key', ConfigScope.APPLICATION)
        assert result == False

    def test_file_storage_exists(self):
        """测试文件存储检查配置存在性"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        # 测试不存在的配置
        assert self.storage.exists('nonexistent_key', ConfigScope.APPLICATION) == False
        
        # 设置配置后测试
        self.storage.set('exists_key', 'exists_value', ConfigScope.APPLICATION)
        assert self.storage.exists('exists_key', ConfigScope.APPLICATION) == True

    def test_file_storage_list_keys(self):
        """测试文件存储列出配置键"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        # 初始状态应该没有键
        keys = self.storage.list_keys()
        assert isinstance(keys, list)
        assert len(keys) == 0
        
        # 设置一些配置
        self.storage.set('key1', 'value1', ConfigScope.APPLICATION)
        self.storage.set('key2', 'value2', ConfigScope.GLOBAL)
        self.storage.set('key3', 'value3', ConfigScope.USER)
        
        # 列出所有键
        keys = self.storage.list_keys()
        assert len(keys) == 3
        assert 'key1' in keys
        assert 'key2' in keys
        assert 'key3' in keys
        
        # 按作用域列出键
        app_keys = self.storage.list_keys(ConfigScope.APPLICATION)
        assert len(app_keys) == 1
        assert 'key1' in app_keys

    def test_file_storage_save_load(self):
        """测试文件存储保存和加载配置"""
        if not STORAGE_AVAILABLE or ConfigScope is None or StorageType is None:
            pytest.skip("配置存储模块不可用")
            
        # 设置一些配置
        test_data = {
            'string_key': 'test_string',
            'int_key': 42,
            'float_key': 3.14,
            'bool_key': True,
            'list_key': [1, 2, 3],
            'dict_key': {'nested': 'value'}
        }
        
        for key, value in test_data.items():
            self.storage.set(key, value, ConfigScope.APPLICATION)
        
        # 保存配置
        save_result = self.storage.save()
        assert save_result == True
        assert os.path.exists(self.test_file) == True
        
        # 创建新的存储实例并加载配置
        if StorageConfig is not None and StorageType is not None and FileConfigStorage is not None:
            config = StorageConfig(type=StorageType.FILE, path=self.test_file)
            new_storage = FileConfigStorage(config)
        
            # 验证加载的数据
            for key, value in test_data.items():
                loaded_value = new_storage.get(key, ConfigScope.APPLICATION)
                assert loaded_value == value

    def test_file_storage_backup(self):
        """测试文件存储备份功能"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        # 设置配置并保存
        self.storage.set('backup_key', 'backup_value', ConfigScope.APPLICATION)
        self.storage.save()
        
        # 启用备份并再次保存
        self.storage.config.backup_enabled = True
        self.storage.config.max_backups = 3
        self.storage.set('backup_key2', 'backup_value2', ConfigScope.APPLICATION)
        self.storage.save()
        
        # 检查备份目录是否存在
        backup_dir = Path(self.test_file).parent / "backups"
        assert backup_dir.exists() == True

    def test_file_storage_error_handling(self):
        """测试文件存储错误处理"""
        if not STORAGE_AVAILABLE:
            pytest.skip("配置存储模块不可用")
            
        # 测试无效路径
        with patch('builtins.open', side_effect=Exception("Test error")):
            result = self.storage.save()
            assert result == False


class TestMemoryConfigStorage:
    """内存配置存储测试"""

    def setup_method(self):
        """测试前设置"""
        if STORAGE_AVAILABLE and MemoryConfigStorage is not None and StorageConfig is not None and StorageType is not None:
            config = StorageConfig(type=StorageType.MEMORY)
            self.storage = MemoryConfigStorage(config)

    def test_memory_storage_initialization(self):
        """测试内存存储初始化"""
        if not STORAGE_AVAILABLE or MemoryConfigStorage is None or StorageType is None:
            pytest.skip("配置存储模块不可用")
            
        assert self.storage is not None
        assert self.storage.config.type == StorageType.MEMORY

    def test_memory_storage_set_get(self):
        """测试内存存储设置和获取配置"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        # 测试设置配置
        result = self.storage.set('mem_key', 'mem_value', ConfigScope.APPLICATION)
        assert result == True
        
        # 测试获取配置
        value = self.storage.get('mem_key', ConfigScope.APPLICATION)
        assert value == 'mem_value'

    def test_memory_storage_delete(self):
        """测试内存存储删除配置"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        # 先设置配置
        self.storage.set('mem_delete_key', 'mem_delete_value', ConfigScope.APPLICATION)
        assert self.storage.get('mem_delete_key', ConfigScope.APPLICATION) == 'mem_delete_value'
        
        # 删除配置
        result = self.storage.delete('mem_delete_key', ConfigScope.APPLICATION)
        assert result == True
        
        # 验证配置已删除
        value = self.storage.get('mem_delete_key', ConfigScope.APPLICATION)
        assert value is None

    def test_memory_storage_exists(self):
        """测试内存存储检查配置存在性"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        # 测试不存在的配置
        assert self.storage.exists('mem_nonexistent_key', ConfigScope.APPLICATION) == False
        
        # 设置配置后测试
        self.storage.set('mem_exists_key', 'mem_exists_value', ConfigScope.APPLICATION)
        assert self.storage.exists('mem_exists_key', ConfigScope.APPLICATION) == True

    def test_memory_storage_list_keys(self):
        """测试内存存储列出配置键"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        # 设置一些配置
        self.storage.set('mem_key1', 'mem_value1', ConfigScope.APPLICATION)
        self.storage.set('mem_key2', 'mem_value2', ConfigScope.GLOBAL)
        
        # 列出所有键
        keys = self.storage.list_keys()
        assert len(keys) >= 2
        assert 'mem_key1' in keys
        assert 'mem_key2' in keys


class TestDistributedConfigStorage:
    """分布式配置存储测试"""

    def setup_method(self):
        """测试前设置"""
        if STORAGE_AVAILABLE and DistributedConfigStorage is not None and StorageConfig is not None and StorageType is not None and DistributedStorageType is not None and ConsistencyLevel is not None:
            config = StorageConfig(
                type=StorageType.DISTRIBUTED,
                distributed_type=DistributedStorageType.REDIS,
                consistency_level=ConsistencyLevel.EVENTUAL
            )
            # 使用mock客户端进行测试
            with patch('src.infrastructure.config.core.config_storage.DistributedConfigStorage._initialize_client'):
                self.storage = DistributedConfigStorage(config)
                self.storage._client = Mock()

    def test_distributed_storage_initialization(self):
        """测试分布式存储初始化"""
        if not STORAGE_AVAILABLE or DistributedConfigStorage is None or StorageType is None or DistributedStorageType is None:
            pytest.skip("配置存储模块不可用")
            
        assert self.storage is not None
        assert self.storage.config.type == StorageType.DISTRIBUTED
        assert self.storage.config.distributed_type == DistributedStorageType.REDIS

    def test_distributed_storage_set_get(self):
        """测试分布式存储设置和获取配置"""
        if not STORAGE_AVAILABLE or ConfigScope is None or DistributedConfigStorage is None:
            pytest.skip("配置存储模块不可用")
            
        # 设置mock返回值 (仅在storage可用时)
        if hasattr(self, 'storage') and self.storage is not None and hasattr(self.storage, '_client'):
            self.storage._client.set.return_value = True
            self.storage._client.get.return_value = json.dumps({
                'value': 'dist_value',
                'key': 'dist_key',
                'scope': 'application',
                'timestamp': 1234567890,
                'version': 'v1',
                'metadata': {}
            })
        
        # 测试设置配置
        result = self.storage.set('dist_key', 'dist_value', ConfigScope.APPLICATION)
        assert result == True
        
        # 测试获取配置
        value = self.storage.get('dist_key', ConfigScope.APPLICATION)
        assert value == 'dist_value'

    def test_distributed_storage_delete(self):
        """测试分布式存储删除配置"""
        if not STORAGE_AVAILABLE or ConfigScope is None or DistributedConfigStorage is None:
            pytest.skip("配置存储模块不可用")
            
        # 设置mock返回值 (仅在storage可用时)
        if hasattr(self, 'storage') and self.storage is not None and hasattr(self.storage, '_client'):
            self.storage._client.delete.return_value = True
        
        # 删除配置
        result = self.storage.delete('dist_delete_key', ConfigScope.APPLICATION)
        assert result == True

    def test_distributed_storage_exists(self):
        """测试分布式存储检查配置存在性"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        # 设置mock返回值 (仅在storage可用时)
        if hasattr(self, 'storage') and self.storage is not None and hasattr(self.storage, '_client'):
            self.storage._client.exists.return_value = True
        
        # 测试配置存在
        result = self.storage.exists('dist_exists_key', ConfigScope.APPLICATION)
        assert result == True

    def test_distributed_storage_list_keys(self):
        """测试分布式存储列出配置键"""
        if not STORAGE_AVAILABLE or DistributedConfigStorage is None:
            pytest.skip("配置存储模块不可用")
            
        # 设置mock返回值 (仅在storage可用时)
        if hasattr(self, 'storage') and self.storage is not None and hasattr(self.storage, '_client'):
            self.storage._client.keys.return_value = ['config:application:key1', 'config:application:key2']
        
        # 列出键
        keys = self.storage.list_keys()
        assert isinstance(keys, list)


class TestStorageFactoryFunctions:
    """存储工厂函数测试"""

    def test_create_file_storage(self):
        """测试创建文件存储"""
        if not STORAGE_AVAILABLE or create_file_storage is None or StorageType is None:
            pytest.skip("配置存储模块不可用")
            
        storage = create_file_storage("test_factory_config.json")
        assert isinstance(storage, FileConfigStorage)
        if StorageType is not None:
            assert storage.config.type == StorageType.FILE

    def test_create_memory_storage(self):
        """测试创建内存存储"""
        if not STORAGE_AVAILABLE or create_memory_storage is None or StorageType is None:
            pytest.skip("配置存储模块不可用")
            
        storage = create_memory_storage()
        assert isinstance(storage, MemoryConfigStorage)
        if StorageType is not None:
            assert storage.config.type == StorageType.MEMORY

    def test_create_distributed_storage(self):
        """测试创建分布式存储"""
        if not STORAGE_AVAILABLE or create_distributed_storage is None or StorageType is None or DistributedStorageType is None:
            pytest.skip("配置存储模块不可用")
            
        storage = create_distributed_storage("redis")
        assert isinstance(storage, DistributedConfigStorage)
        if StorageType is not None and DistributedStorageType is not None:
            assert storage.config.type == StorageType.DISTRIBUTED
            assert storage.config.distributed_type == DistributedStorageType.REDIS

    def test_create_storage(self):
        """测试通用存储创建函数"""
        if not STORAGE_AVAILABLE or create_storage is None:
            pytest.skip("配置存储模块不可用")
            
        # 测试文件存储
        file_storage = create_storage("file")
        assert isinstance(file_storage, FileConfigStorage)
        
        # 测试内存存储
        memory_storage = create_storage("memory")
        assert isinstance(memory_storage, MemoryConfigStorage)
        
        # 测试分布式存储
        distributed_storage = create_storage("distributed")
        assert isinstance(distributed_storage, DistributedConfigStorage)
        
        # 测试无效类型
        if create_storage is not None:
            with pytest.raises(ValueError):
                create_storage("invalid_type")


class TestConfigStorageEnums:
    """配置存储枚举测试"""

    def test_config_scope_enum(self):
        """测试配置作用域枚举"""
        if not STORAGE_AVAILABLE or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        assert ConfigScope.GLOBAL.value == "global"
        assert ConfigScope.USER.value == "user"
        assert ConfigScope.SESSION.value == "session"
        assert ConfigScope.APPLICATION.value == "application"

    def test_storage_type_enum(self):
        """测试存储类型枚举"""
        if not STORAGE_AVAILABLE or StorageType is None:
            pytest.skip("配置存储模块不可用")
            
        assert StorageType.FILE.value == "file"
        assert StorageType.MEMORY.value == "memory"
        assert StorageType.DISTRIBUTED.value == "distributed"

    def test_distributed_storage_type_enum(self):
        """测试分布式存储类型枚举"""
        if not STORAGE_AVAILABLE or DistributedStorageType is None:
            pytest.skip("配置存储模块不可用")
            
        assert DistributedStorageType.REDIS.value == "redis"
        assert DistributedStorageType.ETCD.value == "etcd"
        assert DistributedStorageType.CONSUL.value == "consul"
        assert DistributedStorageType.ZOOKEEPER.value == "zookeeper"

    def test_consistency_level_enum(self):
        """测试一致性级别枚举"""
        if not STORAGE_AVAILABLE or ConsistencyLevel is None:
            pytest.skip("配置存储模块不可用")
            
        assert ConsistencyLevel.STRONG.value == "strong"
        assert ConsistencyLevel.EVENTUAL.value == "eventual"
        assert ConsistencyLevel.CAUSAL.value == "causal"


class TestConfigStorageDataClasses:
    """配置存储数据类测试"""

    def test_config_item_dataclass(self):
        """测试配置项数据类"""
        if not STORAGE_AVAILABLE or ConfigItem is None or ConfigScope is None:
            pytest.skip("配置存储模块不可用")
            
        item = ConfigItem(
            key="test_key",
            value="test_value",
            scope=ConfigScope.APPLICATION,
            timestamp=1234567890.0,
            version="1.0",
            metadata={"source": "test"}
        )
        
        assert item.key == "test_key"
        assert item.value == "test_value"
        assert item.scope == ConfigScope.APPLICATION
        assert item.timestamp == 1234567890.0
        assert item.version == "1.0"
        assert item.metadata == {"source": "test"}

    def test_storage_config_dataclass(self):
        """测试存储配置数据类"""
        if not STORAGE_AVAILABLE or StorageConfig is None or StorageType is None or DistributedStorageType is None or ConsistencyLevel is None:
            pytest.skip("配置存储模块不可用")
            
        config = StorageConfig(
            type=StorageType.FILE,
            path="/test/path",
            distributed_type=DistributedStorageType.REDIS,
            consistency_level=ConsistencyLevel.STRONG,
            backup_enabled=True,
            max_backups=5
        )
        
        assert config.type == StorageType.FILE
        assert config.path == "/test/path"
        assert config.distributed_type == DistributedStorageType.REDIS
        assert config.consistency_level == ConsistencyLevel.STRONG
        assert config.backup_enabled == True
        assert config.max_backups == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src.infrastructure.config.core.config_storage', '--cov-report=term-missing'])
