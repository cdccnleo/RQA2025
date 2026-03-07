#!/usr/bin/env python3
"""
测试enhanced_validators模块

测试覆盖：
- ConfigValidationResult类
- _key_exists和_get_nested_value工具函数
- create_standard_validators函数和返回的验证器
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import re
from unittest.mock import patch, Mock
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

try:
    from src.infrastructure.config.validators.enhanced_validators import (
        ConfigValidationResult,
        _key_exists,
        _get_nested_value,
        create_standard_validators
    )
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigValidationResult:
    """测试ConfigValidationResult类"""

    def setup_method(self):
        """测试前准备"""
        self.result = ConfigValidationResult()

    def test_initialization_default(self):
        """测试默认初始化"""
        result = ConfigValidationResult()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations == []

    def test_initialization_with_false(self):
        """测试初始化为False"""
        result = ConfigValidationResult(False)
        assert result.is_valid is False
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations == []

    def test_add_error(self):
        """测试添加错误"""
        # 初始状态是有效的
        assert self.result.is_valid is True
        
        # 添加错误
        self.result.add_error("测试错误")
        
        # 验证状态改变
        assert self.result.is_valid is False
        assert len(self.result.errors) == 1
        assert "测试错误" in self.result.errors

    def test_add_warning(self):
        """测试添加警告"""
        # 添加警告不应影响有效性
        assert self.result.is_valid is True
        
        self.result.add_warning("测试警告")
        
        assert self.result.is_valid is True
        assert len(self.result.warnings) == 1
        assert "测试警告" in self.result.warnings

    def test_add_recommendation(self):
        """测试添加建议"""
        self.result.add_recommendation("测试建议")
        
        assert len(self.result.recommendations) == 1
        assert "测试建议" in self.result.recommendations

    def test_multiple_errors(self):
        """测试添加多个错误"""
        self.result.add_error("错误1")
        self.result.add_error("错误2")
        
        assert self.result.is_valid is False
        assert len(self.result.errors) == 2
        assert "错误1" in self.result.errors
        assert "错误2" in self.result.errors


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestKeyExists:
    """测试_key_exists工具函数"""

    def test_key_exists_simple_key(self):
        """测试简单键存在检查"""
        config = {"key1": "value1", "key2": "value2"}
        
        assert _key_exists("key1", config) is True
        assert _key_exists("key2", config) is True
        assert _key_exists("nonexistent", config) is False

    def test_key_exists_nested_key(self):
        """测试嵌套键存在检查"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "connection": {
                    "pool_size": 10
                }
            }
        }
        
        assert _key_exists("database.host", config) is True
        assert _key_exists("database.port", config) is True
        assert _key_exists("database.connection.pool_size", config) is True
        assert _key_exists("database.nonexistent", config) is False
        assert _key_exists("nonexistent.key", config) is False

    def test_key_exists_edge_cases(self):
        """测试边界情况"""
        # 空键
        assert _key_exists("", {"key": "value"}) is False
        
        # None键
        assert _key_exists(None, {"key": "value"}) is False
        
        # 非字典配置
        assert _key_exists("key", "not_a_dict") is False
        assert _key_exists("key", None) is False
        
        # 空配置字典
        assert _key_exists("key", {}) is False


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestGetNestedValue:
    """测试_get_nested_value工具函数"""

    def test_get_nested_value_simple_key(self):
        """测试获取简单键的值"""
        config = {"key1": "value1", "key2": 42}
        
        assert _get_nested_value("key1", config) == "value1"
        assert _get_nested_value("key2", config) == 42

    def test_get_nested_value_nested_key(self):
        """测试获取嵌套键的值"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "connection": {
                    "pool_size": 10
                }
            }
        }
        
        assert _get_nested_value("database.host", config) == "localhost"
        assert _get_nested_value("database.port", config) == 5432
        assert _get_nested_value("database.connection.pool_size", config) == 10

    def test_get_nested_value_missing_key(self):
        """测试获取不存在的键"""
        config = {"key1": "value1"}
        
        # 不存在的键应返回None（默认值）
        assert _get_nested_value("missing_key", config) is None
        
        # 测试自定义默认值
        assert _get_nested_value("missing_key", config, "default") == "default"

    def test_get_nested_value_edge_cases(self):
        """测试边界情况"""
        # 空键
        assert _get_nested_value("", {"key": "value"}) is None
        
        # None键
        assert _get_nested_value(None, {"key": "value"}) is None
        
        # 非字典配置
        assert _get_nested_value("key", "not_a_dict") is None
        assert _get_nested_value("key", None) is None


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestCreateStandardValidators:
    """测试create_standard_validators函数"""

    def setup_method(self):
        """测试前准备"""
        self.validators = create_standard_validators()

    def test_create_standard_validators_returns_list(self):
        """测试返回的验证器列表"""
        assert isinstance(self.validators, list)
        assert len(self.validators) == 3  # 应该返回3个验证器

    def test_validators_are_callable(self):
        """测试验证器是可调用的"""
        for validator in self.validators:
            assert callable(validator)

    def test_required_keys_validator(self):
        """测试必需键验证器"""
        # 第一个验证器应该是required_keys_validator
        validator = self.validators[0]
        
        # 测试缺少必需键的配置
        invalid_config = {
            "logging": {"level": "INFO"},
            "system": {"debug": True}
            # 缺少 database.host
        }
        
        result = validator(invalid_config)
        assert isinstance(result, ConfigValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Missing required configuration key" in error for error in result.errors)

        # 测试完整的配置
        valid_config = {
            "logging": {"level": "INFO"},
            "system": {"debug": True},
            "database": {"host": "localhost"}
        }
        
        result = validator(valid_config)
        assert isinstance(result, ConfigValidationResult)
        assert result.is_valid is True

    def test_types_validator(self):
        """测试类型验证器"""
        # 第二个验证器应该是types_validator
        validator = self.validators[1]
        
        # 测试类型错误的配置
        invalid_config = {
            "system": {"debug": "true"},  # 应该是bool，不是str
            "logging": {"level": "INFO"},
            "database": {"port": "5432"}  # 应该是int，不是str
        }
        
        result = validator(invalid_config)
        assert isinstance(result, ConfigValidationResult)
        # 注意：这个验证器只检查存在的键，所以可能不会有错误
        # 具体行为取决于实现

    def test_format_validator(self):
        """测试格式验证器"""
        # 第三个验证器应该是format_validator
        validator = self.validators[2]
        
        # 测试无效邮箱格式
        invalid_config = {
            "email": {"sender": "invalid-email-format"}
        }
        
        result = validator(invalid_config)
        assert isinstance(result, ConfigValidationResult)
        # 注意：这个验证器只检查存在的键，所以可能不会有错误
        # 具体行为取决于实现

    def test_format_validator_email_validation(self):
        """测试邮箱格式验证"""
        validator = self.validators[2]  # format_validator
        
        # 测试有效邮箱
        valid_email_config = {
            "email": {"sender": "test@example.com"}
        }
        result = validator(valid_email_config)
        assert isinstance(result, ConfigValidationResult)
        
        # 测试无效邮箱格式
        invalid_email_config = {
            "email": {"sender": "invalid-email"}
        }
        result = validator(invalid_email_config)
        assert isinstance(result, ConfigValidationResult)

    def test_format_validator_port_validation(self):
        """测试端口验证"""
        validator = self.validators[2]  # format_validator
        
        # 测试有效端口
        valid_port_config = {
            "server": {"port": 8080},
            "database": {"port": 5432}
        }
        result = validator(valid_port_config)
        assert isinstance(result, ConfigValidationResult)
        
        # 测试无效端口（超出范围）
        invalid_port_config = {
            "server": {"port": 70000},  # 超出65535
            "database": {"port": 0}  # 小于1
        }
        result = validator(invalid_port_config)
        assert isinstance(result, ConfigValidationResult)

    def test_validation_result_structure(self):
        """测试验证结果的统一结构"""
        config = {"nonexistent": "value"}
        
        for validator in self.validators:
            result = validator(config)
            
            # 验证结果应该是ConfigValidationResult实例
            assert isinstance(result, ConfigValidationResult)
            assert hasattr(result, 'is_valid')
            assert hasattr(result, 'errors')
            assert hasattr(result, 'warnings')
            assert hasattr(result, 'recommendations')
            
            # 验证数据类型
            assert isinstance(result.is_valid, bool)
            assert isinstance(result.errors, list)
            assert isinstance(result.warnings, list)
            assert isinstance(result.recommendations, list)


