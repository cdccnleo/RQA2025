#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存工具函数全面测试

目标：提升cache_utils.py的测试覆盖率到80%以上
"""

import pytest
import sys
import hashlib
import json
import gzip
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.cache.utils.cache_utils import (
    handle_cache_exceptions,
    serialize_cache_key,
    deserialize_cache_key,
    generate_cache_key,
    calculate_hash,
    estimate_size,
    compress_data,
    decompress_data
)


class TestHandleCacheExceptions:
    """测试缓存异常处理装饰器"""
    
    def test_decorator_success(self):
        """测试装饰器正常执行"""
        @handle_cache_exceptions(default_return=None)
        def success_func():
            return "success"
        
        result = success_func()
        assert result == "success"
    
    def test_decorator_with_exception(self):
        """测试装饰器捕获异常"""
        @handle_cache_exceptions(default_return="default")
        def error_func():
            raise ValueError("Test error")
        
        result = error_func()
        assert result == "default"
    
    def test_decorator_without_params(self):
        """测试不带参数的装饰器"""
        @handle_cache_exceptions
        def func_no_params():
            return "test"
        
        result = func_no_params()
        assert result == "test"
    
    def test_decorator_without_params_with_exception(self):
        """测试不带参数的装饰器处理异常"""
        @handle_cache_exceptions
        def error_func():
            raise RuntimeError("Error")
        
        result = error_func()
        assert result is None
    
    def test_decorator_with_log_level(self):
        """测试装饰器不同的日志级别"""
        @handle_cache_exceptions(default_return="default", log_level="warning")
        def warning_func():
            raise ValueError("Warning error")
        
        result = warning_func()
        assert result == "default"
    
    def test_decorator_with_info_log_level(self):
        """测试装饰器使用info日志级别"""
        @handle_cache_exceptions(default_return="info_default", log_level="info")
        def info_func():
            raise Exception("Info error")
        
        result = info_func()
        assert result == "info_default"
    
    def test_decorator_with_debug_log_level(self):
        """测试装饰器使用debug日志级别"""
        @handle_cache_exceptions(default_return="debug_default", log_level="debug")
        def debug_func():
            raise Exception("Debug error")
        
        result = debug_func()
        assert result == "debug_default"
    
    def test_decorator_with_args_and_kwargs(self):
        """测试装饰器处理参数"""
        @handle_cache_exceptions(default_return="default")
        def func_with_params(a, b, c=None):
            if c:
                return f"{a}-{b}-{c}"
            return f"{a}-{b}"
        
        result1 = func_with_params("x", "y")
        assert result1 == "x-y"
        
        result2 = func_with_params("x", "y", c="z")
        assert result2 == "x-y-z"


class TestCacheKeyOperations:
    """测试缓存键操作"""
    
    def test_serialize_cache_key_simple(self):
        """测试简单的缓存键序列化"""
        key_parts = ("user", 123, "profile")
        result = serialize_cache_key(key_parts)
        assert result == "user:123:profile"
    
    def test_serialize_cache_key_with_special_chars(self):
        """测试包含特殊字符的缓存键序列化"""
        key_parts = ("data", "key:value", 456)
        result = serialize_cache_key(key_parts)
        assert "data" in result
        assert "456" in result
    
    def test_serialize_cache_key_empty(self):
        """测试空元组序列化"""
        result = serialize_cache_key(())
        assert result == ""
    
    def test_deserialize_cache_key_simple(self):
        """测试简单的缓存键反序列化"""
        key = "user:123:profile"
        result = deserialize_cache_key(key)
        assert result == ("user", "123", "profile")
    
    def test_deserialize_cache_key_single(self):
        """测试单个元素的反序列化"""
        key = "single"
        result = deserialize_cache_key(key)
        assert result == ("single",)
    
    def test_deserialize_cache_key_empty(self):
        """测试空字符串反序列化"""
        key = ""
        result = deserialize_cache_key(key)
        assert result == ("",)
    
    def test_generate_cache_key_args_only(self):
        """测试仅使用位置参数生成缓存键"""
        result = generate_cache_key("user", 123, "data")
        assert result == "user:123:data"
    
    def test_generate_cache_key_kwargs_only(self):
        """测试仅使用关键字参数生成缓存键"""
        result = generate_cache_key(user_id=123, action="read")
        assert "user_id=123" in result
        assert "action=read" in result
    
    def test_generate_cache_key_mixed(self):
        """测试混合参数生成缓存键"""
        result = generate_cache_key("prefix", 123, type="user", status="active")
        assert "prefix" in result
        assert "123" in result
        assert "type=user" in result
        assert "status=active" in result
    
    def test_generate_cache_key_empty(self):
        """测试空参数生成缓存键"""
        result = generate_cache_key()
        assert result == ""
    
    def test_generate_cache_key_sorted_kwargs(self):
        """测试关键字参数的排序一致性"""
        result1 = generate_cache_key(z=3, a=1, m=2)
        result2 = generate_cache_key(m=2, z=3, a=1)
        assert result1 == result2


class TestHashOperations:
    """测试哈希操作"""
    
    def test_calculate_hash_string(self):
        """测试字符串哈希计算"""
        data = "test_data"
        result = calculate_hash(data)
        expected = hashlib.md5(str(data).encode()).hexdigest()
        assert result == expected
    
    def test_calculate_hash_integer(self):
        """测试整数哈希计算"""
        data = 12345
        result = calculate_hash(data)
        expected = hashlib.md5(str(data).encode()).hexdigest()
        assert result == expected
    
    def test_calculate_hash_dict(self):
        """测试字典哈希计算"""
        data = {"key": "value", "num": 123}
        result = calculate_hash(data)
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hash length
    
    def test_calculate_hash_list(self):
        """测试列表哈希计算"""
        data = [1, 2, 3, "test"]
        result = calculate_hash(data)
        assert isinstance(result, str)
        assert len(result) == 32
    
    def test_calculate_hash_consistency(self):
        """测试相同数据的哈希一致性"""
        data = "test_consistency"
        result1 = calculate_hash(data)
        result2 = calculate_hash(data)
        assert result1 == result2


class TestSizeEstimation:
    """测试大小估算"""
    
    def test_estimate_size_none(self):
        """测试None的大小估算"""
        result = estimate_size(None)
        assert result == 16
    
    def test_estimate_size_string(self):
        """测试字符串的大小估算"""
        data = "test_string"
        result = estimate_size(data)
        assert result > 0
        assert isinstance(result, int)
    
    def test_estimate_size_integer(self):
        """测试整数的大小估算"""
        data = 12345
        result = estimate_size(data)
        assert result > 0
    
    def test_estimate_size_list(self):
        """测试列表的大小估算"""
        data = [1, 2, 3, 4, 5]
        result = estimate_size(data)
        assert result > 0
    
    def test_estimate_size_dict(self):
        """测试字典的大小估算"""
        data = {"key1": "value1", "key2": "value2"}
        result = estimate_size(data)
        assert result > 0
    
    def test_estimate_size_empty_structures(self):
        """测试空数据结构的大小估算"""
        assert estimate_size([]) > 0
        assert estimate_size({}) > 0
        assert estimate_size("") > 0


class TestDataCompression:
    """测试数据压缩和解压缩"""
    
    def test_compress_bytes(self):
        """测试字节数据压缩"""
        data = b"test data for compression"
        result = compress_data(data)
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    def test_compress_string(self):
        """测试字符串数据压缩"""
        data = "test string for compression"
        result = compress_data(data)
        assert isinstance(result, bytes)
    
    def test_compress_dict(self):
        """测试字典数据压缩"""
        data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        result = compress_data(data)
        assert isinstance(result, bytes)
    
    def test_compress_list(self):
        """测试列表数据压缩"""
        data = [1, 2, 3, "test", {"nested": "dict"}]
        result = compress_data(data)
        assert isinstance(result, bytes)
    
    def test_decompress_none(self):
        """测试解压缩None"""
        result = decompress_data(None)
        assert result is None
    
    def test_decompress_valid_data(self):
        """测试解压缩有效数据"""
        original = {"key": "value", "number": 123}
        compressed = compress_data(original)
        decompressed = decompress_data(compressed)
        assert decompressed == original
    
    def test_compress_decompress_string(self):
        """测试字符串的压缩和解压缩"""
        original = "test string data"
        compressed = compress_data(original)
        decompressed = decompress_data(compressed)
        assert decompressed == original
    
    def test_compress_decompress_complex(self):
        """测试复杂数据结构的压缩和解压缩"""
        original = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ],
            "metadata": {
                "version": "1.0",
                "count": 2
            }
        }
        compressed = compress_data(original)
        decompressed = decompress_data(compressed)
        assert decompressed == original
    
    def test_decompress_non_json_data(self):
        """测试解压缩非JSON数据"""
        # 压缩纯字节数据（不是JSON格式）
        original_bytes = b"non-json binary data"
        compressed = gzip.compress(original_bytes)
        
        # 解压缩应该处理异常情况
        result = decompress_data(compressed)
        # 由于不是JSON，应该返回字符串或原始数据
        assert result is not None


class TestCacheUtilsEdgeCases:
    """测试缓存工具的边界情况"""
    
    def test_generate_cache_key_with_none_values(self):
        """测试包含None值的缓存键生成"""
        result = generate_cache_key("key", None, status=None)
        assert "key" in result
        assert "None" in result
    
    def test_serialize_with_unicode(self):
        """测试Unicode字符的序列化"""
        key_parts = ("用户", "数据", "缓存")
        result = serialize_cache_key(key_parts)
        assert "用户" in result
        assert "数据" in result
        assert "缓存" in result
    
    def test_calculate_hash_empty_string(self):
        """测试空字符串的哈希计算"""
        result = calculate_hash("")
        expected = hashlib.md5("".encode()).hexdigest()
        assert result == expected
    
    def test_estimate_size_large_object(self):
        """测试大对象的大小估算"""
        large_list = list(range(10000))
        result = estimate_size(large_list)
        assert result > 1000  # 应该有相当大的大小


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.infrastructure.cache.utils.cache_utils", 
                 "--cov-report=term-missing"])

