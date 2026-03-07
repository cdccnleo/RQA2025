#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Config Storage Factory
配置存储工厂函数测试，验证工厂函数的正确性和功能完整性
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import sys
import os
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# 导入基础设施层配置存储模块
from src.infrastructure.config.storage import (
    create_file_storage, create_memory_storage, create_distributed_storage, create_storage,
    FileConfigStorage, MemoryConfigStorage, DistributedConfigStorage,
    ConfigScope, StorageType, DistributedStorageType, ConsistencyLevel,
    StorageConfig, ConfigItem
)

class TestConfigStorageFactory(unittest.TestCase):
    """测试配置存储工厂函数"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")

    def tearDown(self):
        """测试清理"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create_file_storage_basic(self):
        """测试创建文件存储 - 基本功能"""
        storage = create_file_storage(self.config_file)
        
        # 验证返回类型
        self.assertIsInstance(storage, FileConfigStorage)
        self.assertIsInstance(storage, FileConfigStorage)
        
        # 验证配置
        self.assertEqual(storage.config.type, StorageType.FILE)
        self.assertEqual(storage.config.path, self.config_file)
        self.assertTrue(storage.config.backup_enabled)
        self.assertEqual(storage.config.max_backups, 10)

    def test_create_file_storage_with_parameters(self):
        """测试创建文件存储 - 带参数"""
        storage = create_file_storage(
            self.config_file,
            backup_enabled=False,
            max_backups=5
        )
        
        # 验证配置参数
        self.assertEqual(storage.config.type, StorageType.FILE)
        self.assertEqual(storage.config.path, self.config_file)
        self.assertFalse(storage.config.backup_enabled)
        self.assertEqual(storage.config.max_backups, 5)

    def test_create_memory_storage_basic(self):
        """测试创建内存存储 - 基本功能"""
        storage = create_memory_storage()
        
        # 验证返回类型
        self.assertIsInstance(storage, MemoryConfigStorage)
        
        # 验证配置
        self.assertEqual(storage.config.type, StorageType.MEMORY)

    def test_create_memory_storage_with_parameters(self):
        """测试创建内存存储 - 带参数"""
        storage = create_memory_storage(
            backup_enabled=True,
            max_backups=3
        )
        
        # 验证配置参数
        self.assertEqual(storage.config.type, StorageType.MEMORY)
        self.assertTrue(storage.config.backup_enabled)
        self.assertEqual(storage.config.max_backups, 3)

    def test_create_distributed_storage_redis(self):
        """测试创建分布式存储 - Redis"""
        storage = create_distributed_storage("redis")
        
        # 验证返回类型
        self.assertIsInstance(storage, DistributedConfigStorage)
        
        # 验证配置
        self.assertEqual(storage.config.type, StorageType.DISTRIBUTED)
        self.assertEqual(storage.config.distributed_type, DistributedStorageType.REDIS)
        self.assertEqual(storage.config.consistency_level, ConsistencyLevel.EVENTUAL)

    def test_create_distributed_storage_etcd(self):
        """测试创建分布式存储 - etcd"""
        storage = create_distributed_storage("etcd")
        
        # 验证配置
        self.assertEqual(storage.config.type, StorageType.DISTRIBUTED)
        self.assertEqual(storage.config.distributed_type, DistributedStorageType.ETCD)

    def test_create_distributed_storage_consul(self):
        """测试创建分布式存储 - Consul"""
        storage = create_distributed_storage("consul")
        
        # 验证配置
        self.assertEqual(storage.config.type, StorageType.DISTRIBUTED)
        self.assertEqual(storage.config.distributed_type, DistributedStorageType.CONSUL)

    def test_create_distributed_storage_zookeeper(self):
        """测试创建分布式存储 - ZooKeeper"""
        storage = create_distributed_storage("zookeeper")
        
        # 验证配置
        self.assertEqual(storage.config.type, StorageType.DISTRIBUTED)
        self.assertEqual(storage.config.distributed_type, DistributedStorageType.ZOOKEEPER)

    def test_create_distributed_storage_with_parameters(self):
        """测试创建分布式存储 - 带参数"""
        storage = create_distributed_storage(
            "redis",
            consistency_level=ConsistencyLevel.STRONG,
            backup_enabled=True,
            max_backups=7
        )
        
        # 验证配置参数
        self.assertEqual(storage.config.type, StorageType.DISTRIBUTED)
        self.assertEqual(storage.config.distributed_type, DistributedStorageType.REDIS)
        self.assertEqual(storage.config.consistency_level, ConsistencyLevel.STRONG)
        self.assertTrue(storage.config.backup_enabled)
        self.assertEqual(storage.config.max_backups, 7)

    def test_create_storage_file(self):
        """测试通用创建函数 - 文件存储"""
        storage = create_storage("file", path=self.config_file)
        
        # 验证返回类型
        self.assertIsInstance(storage, FileConfigStorage)
        
        # 验证配置
        self.assertEqual(storage.config.type, StorageType.FILE)
        self.assertEqual(storage.config.path, self.config_file)

    def test_create_storage_memory(self):
        """测试通用创建函数 - 内存存储"""
        storage = create_storage("memory")
        
        # 验证返回类型
        self.assertIsInstance(storage, MemoryConfigStorage)
        
        # 验证配置
        self.assertEqual(storage.config.type, StorageType.MEMORY)

    def test_create_storage_distributed(self):
        """测试通用创建函数 - 分布式存储"""
        storage = create_storage("distributed", dist_type="redis")
        
        # 验证返回类型
        self.assertIsInstance(storage, DistributedConfigStorage)
        
        # 验证配置
        self.assertEqual(storage.config.type, StorageType.DISTRIBUTED)
        self.assertEqual(storage.config.distributed_type, DistributedStorageType.REDIS)

    def test_create_storage_invalid_type(self):
        """测试通用创建函数 - 无效类型"""
        with self.assertRaises(ValueError) as context:
            create_storage("invalid_type")
        
        self.assertIn("Unsupported storage type", str(context.exception))

    def test_factory_functions_integration(self):
        """测试工厂函数集成"""
        # 创建不同类型的存储
        file_storage = create_file_storage(self.config_file)
        memory_storage = create_memory_storage()
        redis_storage = create_distributed_storage("redis")
        
        # 验证所有存储都实现了IConfigStorage接口
        from src.infrastructure.config.storage.types.iconfigstorage import IConfigStorage
        self.assertIsInstance(file_storage, IConfigStorage)
        self.assertIsInstance(memory_storage, IConfigStorage)
        self.assertIsInstance(redis_storage, IConfigStorage)
        
        # 测试基本操作
        test_key = "test.key"
        test_value = "test_value"
        scope = ConfigScope.APPLICATION
        
        # 文件存储操作
        self.assertTrue(file_storage.set(test_key, test_value, scope))
        self.assertEqual(file_storage.get(test_key, scope), test_value)
        
        # 内存存储操作
        self.assertTrue(memory_storage.set(test_key, test_value, scope))
        self.assertEqual(memory_storage.get(test_key, scope), test_value)
        
        # 验证exists方法
        self.assertTrue(file_storage.exists(test_key, scope))
        self.assertTrue(memory_storage.exists(test_key, scope))
        
        # 验证list_keys方法
        file_keys = file_storage.list_keys(scope)
        memory_keys = memory_storage.list_keys(scope)
        self.assertIn(test_key, file_keys)
        self.assertIn(test_key, memory_keys)

    def test_factory_functions_config_compatibility(self):
        """测试工厂函数配置兼容性"""
        # 测试文件存储配置兼容性
        file_storage = create_file_storage(
            self.config_file,
            backup_enabled=False,
            max_backups=15
        )
        
        # 验证配置正确性
        self.assertEqual(file_storage.config.backup_enabled, False)
        self.assertEqual(file_storage.config.max_backups, 15)
        
        # 测试分布式存储配置兼容性
        distributed_storage = create_distributed_storage(
            "etcd",
            consistency_level=ConsistencyLevel.STRONG
        )
        
        # 验证配置正确性
        self.assertEqual(distributed_storage.config.distributed_type, DistributedStorageType.ETCD)
        self.assertEqual(distributed_storage.config.consistency_level, ConsistencyLevel.STRONG)

if __name__ == '__main__':
    unittest.main()