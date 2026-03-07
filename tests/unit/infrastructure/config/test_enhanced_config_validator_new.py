#!/usr/bin/env python3
"""
测试enhanced_config_validator模块

测试覆盖：
- ValidationLevel枚举
- ValidationError类
- EnhancedConfigValidator类的所有方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import os
from unittest.mock import patch, Mock
import sys
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

try:
    from src.infrastructure.config.utils.enhanced_config_validator import (
        ValidationLevel,
        ValidationError,
        EnhancedConfigValidator
    )
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestValidationLevel:
    """测试ValidationLevel枚举"""

    def test_validation_levels(self):
        """测试验证级别枚举值"""
        assert ValidationLevel.STRICT.value == "strict"
        assert ValidationLevel.NORMAL.value == "normal"
        assert ValidationLevel.LAX.value == "lax"


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestValidationError:
    """测试ValidationError类"""

    def test_validation_error_initialization(self):
        """测试ValidationError初始化"""
        error = ValidationError(
            field="test.field",
            message="测试错误消息",
            severity="error",
            suggestion="测试建议"
        )
        
        assert error.field == "test.field"
        assert error.message == "测试错误消息"
        assert error.severity == "error"
        assert error.suggestion == "测试建议"

    def test_validation_error_default_values(self):
        """测试ValidationError默认值"""
        error = ValidationError(
            field="test.field",
            message="测试错误消息"
        )
        
        assert error.field == "test.field"
        assert error.message == "测试错误消息"
        assert error.severity == "error"  # 默认值
        assert error.suggestion == ""  # 默认值


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestEnhancedConfigValidator:
    """测试EnhancedConfigValidator类"""

    def setup_method(self):
        """测试前准备"""
        self.validator = EnhancedConfigValidator()

    def test_initialization_default(self):
        """测试默认初始化"""
        assert self.validator.validation_level == ValidationLevel.NORMAL
        assert isinstance(self.validator.custom_validators, dict)
        assert len(self.validator.custom_validators) == 0
        assert isinstance(self.validator.validation_stats, dict)
        assert self.validator.validation_stats['total_validations'] == 0

    def test_initialization_with_level(self):
        """测试指定验证级别初始化"""
        validator_strict = EnhancedConfigValidator(ValidationLevel.STRICT)
        assert validator_strict.validation_level == ValidationLevel.STRICT

    def test_validate_database_config_valid(self):
        """测试有效的数据库配置验证"""
        valid_config = {
            'host': 'localhost',
            'port': 5432,
            'username': 'user',
            'database': 'testdb'
        }
        
        errors = self.validator.validate_database_config(valid_config)
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_database_config_missing_fields(self):
        """测试缺少必需字段的数据库配置验证"""
        invalid_config = {
            'host': 'localhost',
            'port': 5432
            # 缺少 username 和 database
        }
        
        errors = self.validator.validate_database_config(invalid_config)
        assert isinstance(errors, list)
        assert len(errors) >= 2  # 至少缺少2个字段
        assert all(isinstance(error, ValidationError) for error in errors)

    def test_validate_database_config_invalid_port(self):
        """测试无效端口的数据库配置验证"""
        invalid_config = {
            'host': 'localhost',
            'port': 70000,  # 超出范围
            'username': 'user',
            'database': 'testdb'
        }
        
        errors = self.validator.validate_database_config(invalid_config)
        assert len(errors) >= 1
        assert any(error.field == "database.port" for error in errors)

    def test_validate_database_config_invalid_connection_pool(self):
        """测试无效连接池配置"""
        invalid_config = {
            'host': 'localhost',
            'port': 5432,
            'username': 'user',
            'database': 'testdb',
            'connection_pool': {
                'min_connections': 10,
                'max_connections': 5  # 最小连接数大于最大连接数
            }
        }
        
        errors = self.validator.validate_database_config(invalid_config)
        assert len(errors) >= 1
        assert any("最小连接数不能大于最大连接数" in error.message for error in errors)

    def test_validate_api_config_valid(self):
        """测试有效的API配置验证"""
        valid_config = {
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'retry': {
                'max_attempts': 3
            }
        }
        
        errors = self.validator.validate_api_config(valid_config)
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_api_config_invalid_url(self):
        """测试无效URL的API配置验证"""
        invalid_config = {
            'base_url': 'invalid-url',  # 无效URL
            'timeout': 30
        }
        
        errors = self.validator.validate_api_config(invalid_config)
        assert len(errors) >= 1
        assert any("API基础URL必须以http://或https://开头" in error.message for error in errors)

    def test_validate_api_config_invalid_timeout(self):
        """测试无效超时的API配置验证"""
        invalid_config = {
            'base_url': 'https://api.example.com',
            'timeout': -5  # 负数超时
        }
        
        errors = self.validator.validate_api_config(invalid_config)
        assert len(errors) >= 1
        assert any("超时时间" in error.message for error in errors)

    def test_validate_logging_config_valid(self):
        """测试有效的日志配置验证"""
        valid_config = {
            'level': 'INFO',
            'file': '/tmp/test.log'
        }
        
        with patch('os.path.exists', return_value=True):
            errors = self.validator.validate_logging_config(valid_config)
            assert isinstance(errors, list)

    def test_validate_logging_config_invalid_level(self):
        """测试无效日志级别的配置验证"""
        invalid_config = {
            'level': 'INVALID_LEVEL'
        }
        
        errors = self.validator.validate_logging_config(invalid_config)
        assert len(errors) >= 1
        assert any("日志级别" in error.message for error in errors)

    def test_validate_logging_config_file_dir_not_exists(self):
        """测试日志文件目录不存在的配置验证"""
        invalid_config = {
            'level': 'INFO',
            'file': '/nonexistent/dir/test.log'
        }
        
        with patch('os.path.exists', return_value=False):
            errors = self.validator.validate_logging_config(invalid_config)
            assert len(errors) >= 1

    def test_validate_security_config_encryption_enabled_missing_key(self):
        """测试启用加密但缺少密钥文件的配置验证"""
        invalid_config = {
            'encryption': {
                'enabled': True
                # 缺少 key_file
            }
        }
        
        errors = self.validator.validate_security_config(invalid_config)
        assert len(errors) >= 1
        assert any("启用加密时必须指定密钥文件" in error.message for error in errors)

    def test_validate_security_config_access_control_missing_user_file(self):
        """测试启用访问控制但缺少用户文件的配置验证"""
        invalid_config = {
            'access_control': {
                'enabled': True
                # 缺少 user_file
            }
        }
        
        errors = self.validator.validate_security_config(invalid_config)
        assert len(errors) >= 1
        assert any("启用访问控制时必须指定用户文件" in error.message for error in errors)

    def test_validate_monitoring_config_prometheus_invalid_port(self):
        """测试Prometheus端口无效的配置验证"""
        invalid_config = {
            'prometheus': {
                'enabled': True,
                'port': 70000  # 超出范围
            }
        }
        
        errors = self.validator.validate_monitoring_config(invalid_config)
        assert len(errors) >= 1
        assert any("Prometheus端口" in error.message for error in errors)

    def test_validate_monitoring_config_alerting_missing_smtp(self):
        """测试启用告警但缺少SMTP服务器的配置验证"""
        invalid_config = {
            'alerting': {
                'enabled': True,
                'email': {
                    'enabled': True
                    # 缺少 smtp_server
                }
            }
        }
        
        errors = self.validator.validate_monitoring_config(invalid_config)
        assert len(errors) >= 1
        assert any("启用邮件告警时必须配置SMTP服务器" in error.message for error in errors)

    def test_validate_full_config_complete(self):
        """测试完整配置验证"""
        complete_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'username': 'user',
                'database': 'testdb'
            },
            'api': {
                'base_url': 'https://api.example.com',
                'timeout': 30
            },
            'logging': {
                'level': 'INFO'
            },
            'security': {},
            'monitoring': {}
        }
        
        with patch('os.path.exists', return_value=True):
            results = self.validator.validate_full_config(complete_config)
            
            assert isinstance(results, dict)
            assert 'database' in results
            assert 'api' in results
            assert 'logging' in results
            assert 'security' in results
            assert 'monitoring' in results
            
            # 验证统计信息更新
            assert self.validator.validation_stats['total_validations'] == 1

    def test_validate_full_config_with_errors(self):
        """测试包含错误的完整配置验证"""
        config_with_errors = {
            'database': {
                'host': 'localhost'
                # 缺少必需字段
            },
            'api': {
                'base_url': 'invalid-url'
            }
        }
        
        results = self.validator.validate_full_config(config_with_errors)
        
        # 应该有验证错误
        total_errors = sum(len(errors) for errors in results.values())
        assert total_errors > 0
        assert self.validator.validation_stats['failed_validations'] == 1

    def test_add_custom_validator(self):
        """测试添加自定义验证器"""
        def custom_validator(config):
            return []
        
        self.validator.add_custom_validator('test_validator', custom_validator)
        
        assert 'test_validator' in self.validator.custom_validators
        assert self.validator.custom_validators['test_validator'] == custom_validator

    def test_get_validation_stats(self):
        """测试获取验证统计信息"""
        stats = self.validator.get_validation_stats()
        
        assert isinstance(stats, dict)
        assert 'total_validations' in stats
        assert 'passed_validations' in stats
        assert 'failed_validations' in stats
        assert 'errors_by_type' in stats
        
        # 验证返回的是副本
        stats['total_validations'] = 999
        assert self.validator.validation_stats['total_validations'] == 0

    def test_multiple_validations_update_stats(self):
        """测试多次验证更新统计信息"""
        valid_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'username': 'user',
                'database': 'testdb'
            },
            'api': {},
            'logging': {},
            'security': {},
            'monitoring': {}
        }
        
        # 执行多次验证
        self.validator.validate_full_config(valid_config)
        self.validator.validate_full_config(valid_config)
        
        assert self.validator.validation_stats['total_validations'] == 2
        assert self.validator.validation_stats['passed_validations'] == 2

    def test_empty_configs_handling(self):
        """测试空配置的处理"""
        empty_config = {}
        
        results = self.validator.validate_full_config(empty_config)
        
        assert isinstance(results, dict)
        assert len(results) == 5  # 有5个配置部分
        
        # 空配置应该有验证错误（缺少必需字段）
        total_errors = sum(len(errors) for errors in results.values())
        assert total_errors > 0


