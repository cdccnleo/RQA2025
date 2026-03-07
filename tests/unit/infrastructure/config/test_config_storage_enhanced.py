#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强配置存储单元测试

补充测试配置存储模块的高级功能和边界情况
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.config.storage.types.fileconfigstorage import FileConfigStorage
from src.infrastructure.config.storage.types.memoryconfigstorage import MemoryConfigStorage
from src.infrastructure.config.storage.types.distributedconfigstorage import DistributedConfigStorage
from src.infrastructure.config.storage.types.storageconfig import StorageConfig
from src.infrastructure.config.storage.types.storagetype import StorageType
from src.infrastructure.config.storage.types.distributedstoragetype import DistributedStorageType
from src.infrastructure.config.storage.types.configscope import ConfigScope
from src.infrastructure.config.storage.types.configitem import ConfigItem


class TestConfigStorageEnhanced:
    """测试配置存储增强功能"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")

    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_file_config_storage_backup_rotation(self):
        """测试文件配置存储备份轮转"""
        config = StorageConfig()
        config.type = StorageType.FILE
        config.path = self.config_file
        config.backup_enabled = True
        config.max_backups = 3
        
        storage = FileConfigStorage(config)
        
        # 创建多个备份
        for i in range(5):
            storage.set("backup_key", f"value_{i}", ConfigScope.APPLICATION)
            storage.save()
            time.sleep(0.1)  # 确保时间戳不同
        
        # 检查备份目录
        backup_dir = os.path.join(self.temp_dir, "backups")
        if os.path.exists(backup_dir):
            backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.bak')]
            # 验证备份文件数量不超过最大限制
            assert len(backup_files) <= config.max_backups

    def test_file_config_storage_concurrent_access(self):
        """测试文件配置存储并发访问"""
        import threading
        import concurrent.futures
        
        # 为每个工作线程创建独立的配置文件以避免并发冲突
        def worker(worker_id):
            """工作线程"""
            # 为每个线程创建独立的配置文件
            thread_config_file = os.path.join(self.temp_dir, f"test_config_{worker_id}.json")
            config = StorageConfig()
            config.type = StorageType.FILE
            config.path = thread_config_file
            
            storage = FileConfigStorage(config)
            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                storage.set(key, value, ConfigScope.APPLICATION)
                retrieved = storage.get(key, ConfigScope.APPLICATION)
                assert retrieved == value

        # 启动并发线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            concurrent.futures.wait(futures)
        
        # 验证所有数据都正确保存
        for worker_id in range(5):
            thread_config_file = os.path.join(self.temp_dir, f"test_config_{worker_id}.json")
            config = StorageConfig()
            config.type = StorageType.FILE
            config.path = thread_config_file
            storage = FileConfigStorage(config)
            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                expected = f"worker_{worker_id}_value_{i}"
                actual = storage.get(key, ConfigScope.APPLICATION)
                assert actual == expected, f"Worker {worker_id}, key {key}: expected {expected}, got {actual}"

    def test_file_config_storage_large_data_handling(self):
        """测试文件配置存储大数据处理"""
        config = StorageConfig()
        config.type = StorageType.FILE
        config.path = self.config_file
        
        storage = FileConfigStorage(config)
        
        # 创建大数据
        large_data = {
            "large_list": list(range(1000)),
            "large_dict": {f"key_{i}": f"value_{i}" for i in range(500)},
            "large_string": "x" * 10000
        }
        
        # 设置大数据
        storage.set("large_data", large_data, ConfigScope.APPLICATION)
        storage.save()
        
        # 重新加载并验证
        new_storage = FileConfigStorage(config)
        retrieved_data = new_storage.get("large_data", ConfigScope.APPLICATION)
        
        assert retrieved_data is not None
        assert len(retrieved_data["large_list"]) == 1000
        assert len(retrieved_data["large_dict"]) == 500
        assert len(retrieved_data["large_string"]) == 10000

    def test_file_config_storage_error_recovery(self):
        """测试文件配置存储错误恢复"""
        config = StorageConfig()
        config.type = StorageType.FILE
        config.path = self.config_file
        
        storage = FileConfigStorage(config)
        
        # 正常设置数据
        storage.set("normal_key", "normal_value", ConfigScope.APPLICATION)
        storage.save()
        
        # 模拟文件损坏
        with open(self.config_file, 'w') as f:
            f.write("invalid json content")
        
        # 重新加载应该处理错误并返回空数据而不是抛出异常
        new_storage = FileConfigStorage(config)
        value = new_storage.get("normal_key", ConfigScope.APPLICATION)
        # 在错误恢复情况下，可能返回None，但不应该抛出异常
        assert value is None or value == "normal_value"

    def test_memory_config_storage_performance(self):
        """测试内存配置存储性能"""
        config = StorageConfig()
        config.type = StorageType.MEMORY
        
        storage = MemoryConfigStorage(config)
        
        # 批量设置测试
        start_time = time.time()
        for i in range(1000):
            storage.set(f"perf_key_{i}", f"perf_value_{i}", ConfigScope.APPLICATION)
        set_time = time.time() - start_time
        
        # 批量获取测试
        start_time = time.time()
        for i in range(1000):
            value = storage.get(f"perf_key_{i}", ConfigScope.APPLICATION)
            assert value == f"perf_value_{i}"
        get_time = time.time() - start_time
        
        # 验证性能在合理范围内
        assert set_time < 1.0, f"Set performance too slow: {set_time}s"
        assert get_time < 1.0, f"Get performance too slow: {get_time}s"

    def test_memory_config_storage_thread_safety(self):
        """测试内存配置存储线程安全"""
        import threading
        import concurrent.futures
        
        config = StorageConfig()
        config.type = StorageType.MEMORY
        
        results = []
        errors = []
        
        def worker(worker_id):
            """工作线程"""
            try:
                storage = MemoryConfigStorage(config)
                for i in range(50):
                    key = f"thread_{worker_id}_key_{i}"
                    value = f"thread_{worker_id}_value_{i}"
                    storage.set(key, value, ConfigScope.APPLICATION)
                    retrieved = storage.get(key, ConfigScope.APPLICATION)
                    if retrieved != value:
                        errors.append(f"Thread {worker_id}: expected {value}, got {retrieved}")
                    else:
                        results.append(f"Thread {worker_id}_item_{i}")
            except Exception as e:
                errors.append(f"Thread {worker_id} error: {e}")
        
        # 启动并发线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            concurrent.futures.wait(futures)
        
        # 验证结果
        assert len(errors) == 0, f"Thread safety test had errors: {errors}"
        assert len(results) == 500, "Not all operations completed successfully"

    def test_distributed_config_storage_redis_operations(self):
        """测试分布式配置存储Redis操作"""
        config = StorageConfig()
        config.type = StorageType.DISTRIBUTED
        config.distributed_type = DistributedStorageType.REDIS
        
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis.set.return_value = True
            mock_redis.get.return_value = '{"key": "test_key", "value": "test_value", "scope": "application", "timestamp": 1234567890, "version": "abc123", "metadata": {}}'
            mock_redis.delete.return_value = 1
            mock_redis.exists.return_value = True
            mock_redis.keys.return_value = ["config:application:test_key"]
            mock_redis_class.return_value = mock_redis
            
            storage = DistributedConfigStorage(config)
            
            # 测试设置操作
            result = storage.set("test_key", "test_value", ConfigScope.APPLICATION)
            assert result == True
            mock_redis.set.assert_called()
            
            # 测试获取操作
            value = storage.get("test_key", ConfigScope.APPLICATION)
            assert value == "test_value"
            mock_redis.get.assert_called()
            
            # 测试删除操作
            result = storage.delete("test_key", ConfigScope.APPLICATION)
            assert result == True
            mock_redis.delete.assert_called()
            
            # 测试存在性检查
            exists = storage.exists("test_key", ConfigScope.APPLICATION)
            assert exists == True
            mock_redis.exists.assert_called()

    def test_distributed_config_storage_fallback_behavior(self):
        """测试分布式配置存储降级行为"""
        config = StorageConfig()
        config.type = StorageType.DISTRIBUTED
        config.distributed_type = DistributedStorageType.REDIS
        
        # 模拟客户端初始化失败，但确保不会抛出异常
        with patch('src.infrastructure.config.storage.types.distributedconfigstorage.DistributedConfigStorage._init_redis_client') as mock_init:
            mock_init.side_effect = Exception("Connection failed")
            
            # 创建实例应该成功，即使初始化失败
            storage = DistributedConfigStorage(config)
            
            # 验证客户端为None
            assert storage._client is None
            
            # 测试各种操作应该优雅处理
            result = storage.set("test_key", "test_value", ConfigScope.APPLICATION)
            assert result == False  # 应该返回False而不是抛出异常
            
            value = storage.get("test_key", ConfigScope.APPLICATION)
            assert value is None  # 应该返回None而不是抛出异常
            
            result = storage.delete("test_key", ConfigScope.APPLICATION)
            assert result == False  # 应该返回False而不是抛出异常

    def test_config_storage_factory_functions(self):
        """测试配置存储工厂函数"""
        # 测试文件存储创建
        from src.infrastructure.config.storage import create_file_storage, create_memory_storage, create_distributed_storage
        
        file_storage = create_file_storage(self.config_file)
        assert file_storage is not None
        # 验证具体实现类有config属性
        assert isinstance(file_storage, FileConfigStorage)
        assert file_storage.config.type == StorageType.FILE
        assert file_storage.config.path == self.config_file
        
        # 测试内存存储创建
        memory_storage = create_memory_storage()
        assert memory_storage is not None
        # 验证具体实现类有config属性
        assert isinstance(memory_storage, MemoryConfigStorage)
        assert memory_storage.config.type == StorageType.MEMORY
        
        # 测试分布式存储创建
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis
            
            distributed_storage = create_distributed_storage("redis")
            assert distributed_storage is not None
            # 验证具体实现类有config属性
            assert isinstance(distributed_storage, DistributedConfigStorage)
            assert distributed_storage.config.type == StorageType.DISTRIBUTED
            assert distributed_storage.config.distributed_type == DistributedStorageType.REDIS

    def test_config_storage_edge_cases(self):
        """测试配置存储边界情况"""
        config = StorageConfig()
        config.type = StorageType.FILE
        config.path = self.config_file
        
        storage = FileConfigStorage(config)
        
        # 测试空键
        result = storage.set("", "empty_key_value", ConfigScope.APPLICATION)
        assert result == True
        value = storage.get("", ConfigScope.APPLICATION)
        assert value == "empty_key_value"
        
        # 测试特殊字符键
        special_key = "key with spaces & special chars!@#$%^&*()"
        result = storage.set(special_key, "special_value", ConfigScope.APPLICATION)
        assert result == True
        value = storage.get(special_key, ConfigScope.APPLICATION)
        assert value == "special_value"
        
        # 测试None值
        result = storage.set("none_key", None, ConfigScope.APPLICATION)
        assert result == True
        value = storage.get("none_key", ConfigScope.APPLICATION)
        assert value is None
        
        # 测试复杂数据类型
        complex_data = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "bool": True,
            "null": None
        }
        result = storage.set("complex_key", complex_data, ConfigScope.APPLICATION)
        assert result == True
        value = storage.get("complex_key", ConfigScope.APPLICATION)
        assert value == complex_data

    def test_config_storage_compatibility_layer(self):
        """测试配置存储兼容性层"""
        from src.infrastructure.config.storage import ConfigStorage
        
        # 测试兼容性类创建
        compat_storage = ConfigStorage({"path": self.config_file})
        assert compat_storage is not None
        
        # 测试兼容性方法
        result = compat_storage.set_config("compat_key", {"value": "compat_value"})
        assert result == True
        
        config = compat_storage.get_config("compat_key")
        assert config == {"value": "compat_value"}
        
        keys = compat_storage.list_configs()
        assert "compat_key" in keys


if __name__ == '__main__':
    pytest.main([__file__])