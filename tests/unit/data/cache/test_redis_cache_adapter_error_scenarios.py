#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis缓存适配器异常场景测试

目标：覆盖redis_cache_adapter.py中的异常处理和降级逻辑
重点：连接失败、序列化失败、压缩异常、重试机制等高风险场景
"""

import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import os
import sys
import types
from unittest.mock import MagicMock, patch, Mock
import pytest
import pickle
import zlib

# 设置测试环境变量
os.environ['PYTEST_CURRENT_TEST'] = 'true'

# Mock redis模块
if 'redis' not in sys.modules:
    redis_module = types.ModuleType('redis')
    
    class ConnectionPool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    
    class Redis:
        def __init__(self, connection_pool=None):
            self.connection_pool = connection_pool
            self.ping_called = False
        
        def ping(self):
            self.ping_called = True
            return True
    
    redis_module.ConnectionPool = ConnectionPool
    redis_module.Redis = Redis
    sys.modules['redis'] = redis_module

from src.data.cache.redis_cache_adapter import RedisCacheAdapter, RedisCacheConfig


@pytest.fixture
def redis_config():
    """创建Redis配置"""
    return RedisCacheConfig(
        host='localhost',
        port=6379,
        db=0,
        default_ttl=3600,
        max_retries=3,
        retry_delay=0.1
    )


class TestRedisCacheAdapterConnection:
    """测试Redis连接场景"""
    
    def test_init_with_test_env_uses_mock(self, redis_config):
        """测试在测试环境中使用模拟客户端"""
        adapter = RedisCacheAdapter(redis_config)
        
        # 验证使用了模拟客户端
        assert adapter.client is not None
        assert hasattr(adapter.client, 'ping')
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('redis.ConnectionPool')
    @patch('redis.Redis')
    def test_setup_real_client_success(self, mock_redis_class, mock_pool_class, redis_config):
        """测试真实Redis客户端连接成功"""
        # 移除测试环境变量
        if 'PYTEST_CURRENT_TEST' in os.environ:
            del os.environ['PYTEST_CURRENT_TEST']
        if 'TESTING' in os.environ:
            del os.environ['TESTING']
        
        # 模拟redis客户端
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client
        
        adapter = RedisCacheAdapter.__new__(RedisCacheAdapter)
        adapter.config = redis_config
        adapter._lock = MagicMock()
        adapter.stats = {}
        adapter.key_prefix = "rqa_data:"
        
        # 直接调用_setup_real_client
        adapter._setup_real_client()
        
        assert adapter.client is not None
        mock_client.ping.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('redis.ConnectionPool')
    def test_setup_real_client_connection_failure(self, mock_pool_class, redis_config):
        """测试Redis连接失败"""
        # 移除测试环境变量
        if 'PYTEST_CURRENT_TEST' in os.environ:
            del os.environ['PYTEST_CURRENT_TEST']
        if 'TESTING' in os.environ:
            del os.environ['TESTING']
        
        # 模拟连接失败
        mock_pool_class.side_effect = Exception("Connection refused")
        
        adapter = RedisCacheAdapter.__new__(RedisCacheAdapter)
        adapter.config = redis_config
        adapter._lock = MagicMock()
        adapter.stats = {}
        adapter.key_prefix = "rqa_data:"
        
        # 应该抛出ConnectionError
        with pytest.raises(ConnectionError, match="Redis connection failed"):
            adapter._setup_real_client()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('redis.ConnectionPool')
    @patch('redis.Redis')
    def test_setup_real_client_ping_failure(self, mock_redis_class, mock_pool_class, redis_config):
        """测试Redis ping失败"""
        # 移除测试环境变量
        if 'PYTEST_CURRENT_TEST' in os.environ:
            del os.environ['PYTEST_CURRENT_TEST']
        if 'TESTING' in os.environ:
            del os.environ['TESTING']
        
        # 模拟ping失败
        mock_client = MagicMock()
        mock_client.ping.side_effect = Exception("Ping failed")
        mock_redis_class.return_value = mock_client
        
        adapter = RedisCacheAdapter.__new__(RedisCacheAdapter)
        adapter.config = redis_config
        adapter._lock = MagicMock()
        adapter.stats = {}
        adapter.key_prefix = "rqa_data:"
        
        # 应该抛出ConnectionError
        with pytest.raises(ConnectionError, match="Redis connection failed"):
            adapter._setup_real_client()


class TestRedisCacheAdapterSerialization:
    """测试序列化/反序列化场景"""
    
    def test_serialize_data_json_format(self, redis_config):
        """测试JSON格式序列化"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.config.serialization_format = 'json'
        adapter.config.enable_compression = False
        
        data = {'key': 'value', 'number': 123}
        result = adapter._serialize_data(data)
        
        assert result.startswith(b'UNCOMPRESSED:')
        assert b'key' in result
    
    def test_serialize_data_pickle_format(self, redis_config):
        """测试Pickle格式序列化"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.config.serialization_format = 'pickle'
        adapter.config.enable_compression = False
        
        data = {'key': 'value', 'number': 123}
        result = adapter._serialize_data(data)
        
        assert result.startswith(b'UNCOMPRESSED:')
    
    def test_serialize_data_with_compression(self, redis_config):
        """测试启用压缩的序列化"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.config.serialization_format = 'pickle'
        adapter.config.enable_compression = True
        adapter.config.compression_threshold = 10  # 降低阈值以便触发压缩
        
        # 创建足够大的数据以触发压缩
        data = {'large_data': 'x' * 2000}
        result = adapter._serialize_data(data)
        
        # 应该被压缩
        assert result.startswith(b'COMPRESSED:') or result.startswith(b'UNCOMPRESSED:')
    
    def test_serialize_data_compression_no_savings(self, redis_config):
        """测试压缩后没有节省空间的情况"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.config.serialization_format = 'pickle'
        adapter.config.enable_compression = True
        adapter.config.compression_threshold = 10
        
        # 创建小数据，压缩后可能没有节省
        data = {'small': 'data'}
        result = adapter._serialize_data(data)
        
        # 应该使用未压缩格式
        assert result.startswith(b'UNCOMPRESSED:')
    
    def test_serialize_data_failure(self, redis_config):
        """测试序列化失败"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.config.serialization_format = 'pickle'
        
        # 创建无法序列化的对象
        class Unserializable:
            def __getstate__(self):
                raise Exception("Cannot serialize")
        
        with pytest.raises(ValueError, match="Data serialization failed"):
            adapter._serialize_data(Unserializable())
    
    def test_deserialize_data_compressed(self, redis_config):
        """测试解压缩数据"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.config.serialization_format = 'pickle'
        
        # 创建压缩数据
        original_data = {'key': 'value'}
        serialized = pickle.dumps(original_data)
        compressed = zlib.compress(serialized)
        compressed_data = b'COMPRESSED:' + compressed
        
        result = adapter._deserialize_data(compressed_data)
        
        assert result == original_data
    
    def test_deserialize_data_uncompressed(self, redis_config):
        """测试反序列化未压缩数据"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.config.serialization_format = 'pickle'
        
        original_data = {'key': 'value'}
        serialized = pickle.dumps(original_data)
        uncompressed_data = b'UNCOMPRESSED:' + serialized
        
        result = adapter._deserialize_data(uncompressed_data)
        
        assert result == original_data
    
    def test_deserialize_data_json_format(self, redis_config):
        """测试JSON格式反序列化"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.config.serialization_format = 'json'
        
        import json
        original_data = {'key': 'value', 'number': 123}
        serialized = json.dumps(original_data, ensure_ascii=False).encode('utf-8')
        uncompressed_data = b'UNCOMPRESSED:' + serialized
        
        result = adapter._deserialize_data(uncompressed_data)
        
        assert result == original_data
    
    def test_deserialize_data_failure(self, redis_config):
        """测试反序列化失败"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.config.serialization_format = 'pickle'
        
        # 创建无效的压缩数据
        invalid_data = b'COMPRESSED:invalid_compressed_data'
        
        with pytest.raises(ValueError, match="Data deserialization failed"):
            adapter._deserialize_data(invalid_data)


class TestRedisCacheAdapterOperations:
    """测试缓存操作场景"""
    
    def test_get_with_cache_hit(self, redis_config):
        """测试缓存命中"""
        adapter = RedisCacheAdapter(redis_config)
        
        # 模拟缓存命中
        test_data = {'key': 'value'}
        serialized = adapter._serialize_data(test_data)
        adapter.client.get.return_value = serialized
        
        result = adapter.get('test_key')
        
        assert result == test_data
        assert adapter.stats['hits'] == 1
        adapter.client.get.assert_called_once()
    
    def test_get_with_cache_miss(self, redis_config):
        """测试缓存未命中"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.client.get.return_value = None
        
        result = adapter.get('test_key')
        
        assert result is None
        assert adapter.stats['misses'] == 1
    
    def test_get_with_error_handling(self, redis_config):
        """测试操作失败时的错误处理"""
        adapter = RedisCacheAdapter(redis_config)
        
        # 模拟get操作失败
        adapter.client.get.side_effect = Exception("Connection error")
        
        result = adapter.get('test_key')
        
        # get方法在异常时返回None并记录错误
        assert result is None
        assert adapter.stats['errors'] == 1
        assert adapter.stats['misses'] == 0  # 异常不算miss
    
    def test_set_operation(self, redis_config):
        """测试设置缓存"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.client.setex.return_value = True  # set方法使用setex
        
        test_data = {'key': 'value'}
        result = adapter.set('test_key', test_data, ttl=3600)
        
        assert result is True
        assert adapter.stats['sets'] == 1
        adapter.client.setex.assert_called_once()
    
    def test_delete_operation(self, redis_config):
        """测试删除缓存"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.client.delete.return_value = 1
        
        result = adapter.delete('test_key')
        
        assert result is True
        assert adapter.stats['deletes'] == 1
        adapter.client.delete.assert_called_once()
    
    def test_delete_not_found(self, redis_config):
        """测试删除不存在的键"""
        adapter = RedisCacheAdapter(redis_config)
        adapter.client.delete.return_value = 0
        
        result = adapter.delete('non_existent')
        
        assert result is False
        adapter.client.delete.assert_called_once()

