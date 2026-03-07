#!/usr/bin/env python3
"""
基础设施层 - 配置模式验证测试

测试config_schema.py中的配置模式管理功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.cache.utils.config_schema import (
    SimpleConfigSchemaManager,
    CacheConfigSchema,
    SchemaError
)


class TestSimpleConfigSchemaManager:
    """测试简单配置模式管理器"""

    def test_initialization(self):
        """测试初始化"""
        manager = SimpleConfigSchemaManager()
        assert manager._schema_path is None
        assert 'default' in manager._builtin_schemas
        assert 'database' in manager._builtin_schemas
        assert 'cache' in manager._builtin_schemas

    def test_initialization_with_path(self):
        """测试带路径初始化"""
        manager = SimpleConfigSchemaManager("/path/to/schema")
        assert manager._schema_path == "/path/to/schema"

    def test_get_schema_default(self):
        """测试获取默认模式"""
        manager = SimpleConfigSchemaManager()
        schema = manager.get_schema('default')

        assert isinstance(schema, dict)
        assert 'type' in schema
        assert 'properties' in schema
        assert 'env' in schema['properties']
        assert 'version' in schema['properties']

    def test_get_schema_database(self):
        """测试获取数据库模式"""
        manager = SimpleConfigSchemaManager()
        schema = manager.get_schema('database')

        assert isinstance(schema, dict)
        assert 'host' in schema
        assert 'port' in schema
        assert 'username' in schema
        assert 'password' in schema

        # 检查host是必需的
        assert schema['host']['required'] is True

    def test_get_schema_cache(self):
        """测试获取缓存模式"""
        manager = SimpleConfigSchemaManager()
        schema = manager.get_schema('cache')

        assert isinstance(schema, dict)
        assert 'l1_cache' in schema
        assert 'l2_cache' in schema

    def test_get_schema_unknown_type(self):
        """测试获取未知模式类型"""
        manager = SimpleConfigSchemaManager()

        with pytest.raises(SchemaError) as exc_info:
            manager.get_schema('unknown')

        assert "未知的模式类型: unknown" in str(exc_info.value)

    def test_builtin_schemas_structure(self):
        """测试内置模式的结构"""
        manager = SimpleConfigSchemaManager()

        # 测试所有内置模式都是字典
        for schema_name in ['default', 'database', 'cache']:
            schema = manager._builtin_schemas[schema_name]
            assert isinstance(schema, dict)
            assert len(schema) > 0


class TestCacheConfigSchema:
    """测试缓存配置模式验证器"""

    def test_initialization(self):
        """测试初始化"""
        validator = CacheConfigSchema()
        assert hasattr(validator, 'schema')
        assert isinstance(validator.schema, dict)
        assert 'type' in validator.schema
        assert 'properties' in validator.schema
        assert 'required' in validator.schema

    def test_schema_structure(self):
        """测试模式结构"""
        validator = CacheConfigSchema()
        schema = validator.schema

        # 检查必需字段
        assert 'cache_type' in schema['required']

        # 检查属性
        properties = schema['properties']
        required_props = [
            'cache_type', 'max_size', 'ttl', 'redis_config',
            'file_config', 'eviction_policy', 'monitoring'
        ]

        for prop in required_props:
            assert prop in properties

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        validator = CacheConfigSchema()

        valid_config = {
            'cache_type': 'memory',
            'max_size': 1000,
            'ttl': 3600
        }

        assert validator.validate(valid_config) is True

    def test_validate_invalid_config_type(self):
        """测试验证无效配置类型"""
        validator = CacheConfigSchema()

        invalid_configs = [
            None,  # 非字典
            [],    # 非字典
            "string",  # 非字典
        ]

        for config in invalid_configs:
            assert validator.validate(config) is False

    def test_validate_missing_required_field(self):
        """测试验证缺少必需字段"""
        validator = CacheConfigSchema()

        config_without_type = {
            'max_size': 1000,
            'ttl': 3600
        }

        assert validator.validate(config_without_type) is False

    def test_validate_invalid_cache_type(self):
        """测试验证无效缓存类型"""
        validator = CacheConfigSchema()

        invalid_config = {
            'cache_type': 'invalid_type',
            'max_size': 1000
        }

        assert validator.validate(invalid_config) is False

    def test_validate_invalid_max_size(self):
        """测试验证无效最大大小"""
        validator = CacheConfigSchema()

        invalid_configs = [
            {'cache_type': 'memory', 'max_size': 0},      # 小于1
            {'cache_type': 'memory', 'max_size': -1},     # 负数
            {'cache_type': 'memory', 'max_size': '1000'}, # 非整数
        ]

        for config in invalid_configs:
            assert validator.validate(config) is False

    def test_validate_invalid_ttl(self):
        """测试验证无效TTL"""
        validator = CacheConfigSchema()

        invalid_configs = [
            {'cache_type': 'memory', 'ttl': -1},     # 负数
            {'cache_type': 'memory', 'ttl': '3600'}, # 非整数
        ]

        for config in invalid_configs:
            assert validator.validate(config) is False

    def test_validate_config_with_details(self):
        """测试详细配置验证"""
        validator = CacheConfigSchema()

        # 有效配置
        valid_config = {
            'cache_type': 'redis',
            'max_size': 5000,
            'ttl': 7200
        }

        is_valid, errors = validator.validate_config(valid_config)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_config_invalid_type(self):
        """测试详细验证无效类型"""
        validator = CacheConfigSchema()

        is_valid, errors = validator.validate_config("not_a_dict")
        assert is_valid is False
        assert len(errors) > 0
        assert "配置必须是字典类型" in errors[0]

    def test_validate_config_missing_required(self):
        """测试详细验证缺少必需字段"""
        validator = CacheConfigSchema()

        config = {'max_size': 1000}
        is_valid, errors = validator.validate_config(config)
        assert is_valid is False
        assert len(errors) > 0
        assert "缺少必需字段: cache_type" in errors[0]

    def test_validate_config_invalid_cache_type(self):
        """测试详细验证无效缓存类型"""
        validator = CacheConfigSchema()

        config = {'cache_type': 'invalid'}
        is_valid, errors = validator.validate_config(config)
        assert is_valid is False
        assert len(errors) > 0
        assert "无效的cache_type值" in errors[0]

    def test_get_default_config_memory(self):
        """测试获取内存缓存默认配置"""
        validator = CacheConfigSchema()

        config = validator.get_default_config('memory')
        assert config['cache_type'] == 'memory'
        assert config['max_size'] == 1000
        assert config['ttl'] == 3600
        assert config['eviction_policy'] == 'lru'

    def test_get_default_config_redis(self):
        """测试获取Redis缓存默认配置"""
        validator = CacheConfigSchema()

        config = validator.get_default_config('redis')
        assert config['cache_type'] == 'redis'
        assert config['max_size'] == 10000
        assert 'redis_config' in config
        assert config['redis_config']['host'] == 'localhost'
        assert config['redis_config']['port'] == 6379

    def test_get_default_config_file(self):
        """测试获取文件缓存默认配置"""
        validator = CacheConfigSchema()

        config = validator.get_default_config('file')
        assert config['cache_type'] == 'file'
        assert 'file_config' in config
        assert config['file_config']['cache_dir'] == './cache'

    def test_get_default_config_unknown_type(self):
        """测试获取未知类型默认配置"""
        validator = CacheConfigSchema()

        # 未知类型应该返回memory默认配置
        config = validator.get_default_config('unknown')
        assert config['cache_type'] == 'memory'


class TestSchemaError:
    """测试模式错误异常"""

    def test_exception_inheritance(self):
        """测试异常继承"""
        assert issubclass(SchemaError, Exception)

    def test_exception_creation(self):
        """测试异常创建"""
        error = SchemaError("Test error message")
        assert str(error) == "Test error message"

    def test_exception_with_custom_message(self):
        """测试自定义错误消息"""
        error = SchemaError("Custom schema error")
        assert "Custom schema error" in str(error)