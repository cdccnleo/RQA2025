#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模式管理器全面测试

目标：提升config_schema.py的测试覆盖率到80%以上
"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.cache.utils.config_schema import (
    SimpleConfigSchemaManager,
    CacheConfigSchema,
    SchemaError
)


class TestSimpleConfigSchemaManager:
    """测试简单配置模式管理器"""
    
    def test_init_without_path(self):
        """测试无路径初始化"""
        manager = SimpleConfigSchemaManager()
        assert manager._schema_path is None
        assert len(manager._builtin_schemas) == 3
    
    def test_init_with_path(self):
        """测试带路径初始化"""
        manager = SimpleConfigSchemaManager(schema_path="/path/to/schema")
        assert manager._schema_path == "/path/to/schema"
    
    def test_get_default_schema(self):
        """测试获取默认模式"""
        manager = SimpleConfigSchemaManager()
        schema = manager.get_schema('default')
        
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'env' in schema['properties']
        assert 'version' in schema['properties']
        assert schema['properties']['env']['type'] == 'string'
        assert schema['properties']['env']['required'] is True
    
    def test_get_database_schema(self):
        """测试获取数据库模式"""
        manager = SimpleConfigSchemaManager()
        schema = manager.get_schema('database')
        
        assert 'host' in schema
        assert 'port' in schema
        assert 'username' in schema
        assert 'password' in schema
        assert schema['host']['type'] == 'string'
        assert schema['host']['required'] is True
        assert schema['port']['type'] == 'number'
        assert schema['port']['min'] == 1024
        assert schema['port']['max'] == 65535
    
    def test_get_cache_schema(self):
        """测试获取缓存模式"""
        manager = SimpleConfigSchemaManager()
        schema = manager.get_schema('cache')
        
        assert 'l1_cache' in schema
        assert 'l2_cache' in schema
        assert 'max_size' in schema['l1_cache']
        assert 'expire_after' in schema['l1_cache']
        assert schema['l1_cache']['max_size']['type'] == 'number'
        assert schema['l1_cache']['max_size']['min'] == 1
    
    def test_get_schema_invalid_type(self):
        """测试获取无效模式类型"""
        manager = SimpleConfigSchemaManager()
        
        with pytest.raises(SchemaError) as excinfo:
            manager.get_schema('invalid_type')
        
        assert "未知的模式类型" in str(excinfo.value)
    
    def test_get_schema_default_when_not_specified(self):
        """测试未指定类型时获取默认模式"""
        manager = SimpleConfigSchemaManager()
        schema = manager.get_schema()
        
        assert schema['type'] == 'object'
        assert 'properties' in schema
    
    def test_builtin_schemas_structure(self):
        """测试内置模式结构"""
        manager = SimpleConfigSchemaManager()
        
        assert 'default' in manager._builtin_schemas
        assert 'database' in manager._builtin_schemas
        assert 'cache' in manager._builtin_schemas


class TestCacheConfigSchema:
    """测试缓存配置模式验证器"""
    
    def test_init(self):
        """测试初始化"""
        schema = CacheConfigSchema()
        
        assert schema.schema is not None
        assert schema.schema['type'] == 'object'
        assert 'properties' in schema.schema
        assert 'required' in schema.schema
        assert 'cache_type' in schema.schema['required']
    
    def test_validate_valid_memory_config(self):
        """测试验证有效的内存缓存配置"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'max_size': 1000,
            'ttl': 3600
        }
        
        result = schema.validate(config)
        assert result is True
    
    def test_validate_valid_redis_config(self):
        """测试验证有效的Redis缓存配置"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'redis',
            'max_size': 10000,
            'ttl': 7200
        }
        
        result = schema.validate(config)
        assert result is True
    
    def test_validate_valid_file_config(self):
        """测试验证有效的文件缓存配置"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'file',
            'max_size': 5000,
            'ttl': 3600
        }
        
        result = schema.validate(config)
        assert result is True
    
    def test_validate_valid_hybrid_config(self):
        """测试验证有效的混合缓存配置"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'hybrid',
            'max_size': 8000,
            'ttl': 1800
        }
        
        result = schema.validate(config)
        assert result is True
    
    def test_validate_missing_cache_type(self):
        """测试验证缺少cache_type的配置"""
        schema = CacheConfigSchema()
        config = {
            'max_size': 1000,
            'ttl': 3600
        }
        
        result = schema.validate(config)
        assert result is False
    
    def test_validate_invalid_cache_type(self):
        """测试验证无效的cache_type"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'invalid_type',
            'max_size': 1000
        }
        
        result = schema.validate(config)
        assert result is False
    
    def test_validate_invalid_max_size_negative(self):
        """测试验证负数max_size"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'max_size': -1
        }
        
        result = schema.validate(config)
        assert result is False
    
    def test_validate_invalid_max_size_zero(self):
        """测试验证零max_size"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'max_size': 0
        }
        
        result = schema.validate(config)
        assert result is False
    
    def test_validate_invalid_max_size_type(self):
        """测试验证非整数max_size"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'max_size': "invalid"
        }
        
        result = schema.validate(config)
        assert result is False
    
    def test_validate_invalid_ttl_negative(self):
        """测试验证负数ttl"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'ttl': -1
        }
        
        result = schema.validate(config)
        assert result is False
    
    def test_validate_invalid_ttl_type(self):
        """测试验证非整数ttl"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'ttl': "invalid"
        }
        
        result = schema.validate(config)
        assert result is False
    
    def test_validate_ttl_zero(self):
        """测试验证ttl为0（应该有效）"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'ttl': 0
        }
        
        result = schema.validate(config)
        assert result is True
    
    def test_validate_non_dict_config(self):
        """测试验证非字典配置"""
        schema = CacheConfigSchema()
        
        assert schema.validate(None) is False
        assert schema.validate([]) is False
        assert schema.validate("string") is False
        assert schema.validate(123) is False
    
    def test_validate_config_valid(self):
        """测试validate_config方法返回有效结果"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'max_size': 1000,
            'ttl': 3600
        }
        
        is_valid, errors = schema.validate_config(config)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_config_missing_cache_type(self):
        """测试validate_config缺少cache_type"""
        schema = CacheConfigSchema()
        config = {
            'max_size': 1000
        }
        
        is_valid, errors = schema.validate_config(config)
        assert is_valid is False
        assert len(errors) > 0
        assert any("cache_type" in err for err in errors)
    
    def test_validate_config_invalid_cache_type(self):
        """测试validate_config无效cache_type"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'invalid'
        }
        
        is_valid, errors = schema.validate_config(config)
        assert is_valid is False
        assert len(errors) > 0
        assert any("cache_type" in err for err in errors)
    
    def test_validate_config_multi_level_type(self):
        """测试validate_config支持multi_level类型"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'multi_level'
        }
        
        is_valid, errors = schema.validate_config(config)
        assert is_valid is True
    
    def test_validate_config_invalid_max_size(self):
        """测试validate_config无效max_size"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'max_size': -5
        }
        
        is_valid, errors = schema.validate_config(config)
        assert is_valid is False
        assert any("max_size" in err for err in errors)
    
    def test_validate_config_invalid_ttl(self):
        """测试validate_config无效ttl"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'ttl': -10
        }
        
        is_valid, errors = schema.validate_config(config)
        assert is_valid is False
        assert any("ttl" in err for err in errors)
    
    def test_validate_config_non_dict(self):
        """测试validate_config非字典输入"""
        schema = CacheConfigSchema()
        
        is_valid, errors = schema.validate_config("not a dict")
        assert is_valid is False
        assert len(errors) > 0
        assert any("字典类型" in err for err in errors)
    
    def test_validate_config_multiple_errors(self):
        """测试validate_config多个错误"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'max_size': -1,
            'ttl': -5
        }
        
        is_valid, errors = schema.validate_config(config)
        assert is_valid is False
        assert len(errors) >= 2  # 应该至少有max_size和ttl两个错误
    
    def test_get_default_config_memory(self):
        """测试获取内存缓存默认配置"""
        schema = CacheConfigSchema()
        config = schema.get_default_config('memory')
        
        assert config['cache_type'] == 'memory'
        assert config['max_size'] == 1000
        assert config['ttl'] == 3600
        assert config['eviction_policy'] == 'lru'
        assert 'monitoring' in config
        assert config['monitoring']['enabled'] is True
    
    def test_get_default_config_redis(self):
        """测试获取Redis缓存默认配置"""
        schema = CacheConfigSchema()
        config = schema.get_default_config('redis')
        
        assert config['cache_type'] == 'redis'
        assert config['max_size'] == 10000
        assert 'redis_config' in config
        assert config['redis_config']['host'] == 'localhost'
        assert config['redis_config']['port'] == 6379
        assert config['redis_config']['db'] == 0
    
    def test_get_default_config_file(self):
        """测试获取文件缓存默认配置"""
        schema = CacheConfigSchema()
        config = schema.get_default_config('file')
        
        assert config['cache_type'] == 'file'
        assert config['max_size'] == 5000
        assert config['ttl'] == 7200
        assert 'file_config' in config
        assert config['file_config']['cache_dir'] == './cache'
        assert config['file_config']['max_file_size'] == '100MB'
        assert config['file_config']['compression'] is True
    
    def test_get_default_config_no_type(self):
        """测试获取默认配置（未指定类型）"""
        schema = CacheConfigSchema()
        config = schema.get_default_config()
        
        # 应该返回memory类型的默认配置
        assert config['cache_type'] == 'memory'
    
    def test_get_default_config_invalid_type(self):
        """测试获取无效类型的默认配置"""
        schema = CacheConfigSchema()
        config = schema.get_default_config('invalid_type')
        
        # 应该返回memory类型的默认配置
        assert config['cache_type'] == 'memory'
    
    def test_schema_structure(self):
        """测试模式结构完整性"""
        schema = CacheConfigSchema()
        
        assert 'type' in schema.schema
        assert 'properties' in schema.schema
        assert 'required' in schema.schema
        
        properties = schema.schema['properties']
        assert 'cache_type' in properties
        assert 'max_size' in properties
        assert 'ttl' in properties
        assert 'redis_config' in properties
        assert 'file_config' in properties
        assert 'eviction_policy' in properties
        assert 'monitoring' in properties


class TestSchemaError:
    """测试模式错误异常"""
    
    def test_schema_error_creation(self):
        """测试创建SchemaError异常"""
        error = SchemaError("Test error message")
        assert str(error) == "Test error message"
    
    def test_schema_error_raise(self):
        """测试抛出SchemaError异常"""
        with pytest.raises(SchemaError):
            raise SchemaError("Test error")
    
    def test_schema_error_catch(self):
        """测试捕获SchemaError异常"""
        try:
            raise SchemaError("Test error")
        except SchemaError as e:
            assert "Test error" in str(e)
    
    def test_schema_error_inheritance(self):
        """测试SchemaError继承自Exception"""
        assert issubclass(SchemaError, Exception)


class TestEdgeCases:
    """测试边界情况"""
    
    def test_validate_with_extra_fields(self):
        """测试验证包含额外字段的配置"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'max_size': 1000,
            'ttl': 3600,
            'extra_field': 'extra_value'
        }
        
        # 应该忽略额外字段并验证通过
        result = schema.validate(config)
        assert result is True
    
    def test_validate_config_with_extra_fields(self):
        """测试validate_config处理额外字段"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'unknown_field': 'value'
        }
        
        is_valid, errors = schema.validate_config(config)
        # 应该忽略未知字段
        assert is_valid is True
    
    def test_large_max_size(self):
        """测试大max_size值"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'max_size': 1000000
        }
        
        result = schema.validate(config)
        assert result is True
    
    def test_large_ttl(self):
        """测试大ttl值"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'ttl': 86400  # 24小时
        }
        
        result = schema.validate(config)
        assert result is True
    
    def test_validate_float_values(self):
        """测试浮点数值（应该失败，因为要求整数）"""
        schema = CacheConfigSchema()
        config = {
            'cache_type': 'memory',
            'max_size': 1000.5
        }
        
        result = schema.validate(config)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v",
                 "--cov=src.infrastructure.cache.utils.config_schema",
                 "--cov-report=term-missing"])

