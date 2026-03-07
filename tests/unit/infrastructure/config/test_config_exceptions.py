#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置异常类测试
测试配置系统的各种异常类
"""

import pytest
from unittest.mock import Mock

from src.infrastructure.config.config_exceptions import (
    ConfigError,
    ConfigValidationError,
    ConfigLoadError,
    ConfigNotFoundError,
    ConfigTypeError,
    ConfigValueError,
    ConfigAccessError,
    ConfigSecurityError,
    ConfigTimeoutError,
    ConfigFormatError,
    ConfigMergeError,
    ConfigVersionError,
    ConfigBackupError,
    ConfigRestoreError,
    ConfigEncryptionError,
    ConfigDecryptionError,
    ConfigNetworkError,
    ConfigQuotaExceededError
)


class TestConfigExceptions:
    """测试配置异常类"""

    def test_config_error_basic(self):
        """测试基础配置错误"""
        error = ConfigError("Test error")
        assert str(error) == "Test error"
        assert error.config_key is None
        assert error.details == {}
        assert error.error_type == "config_error"

    def test_config_error_with_params(self):
        """测试带参数的配置错误"""
        details = {"test": "value"}
        error = ConfigError("Test error", config_key="test.key", details=details, error_type="custom_error")
        assert error.config_key == "test.key"
        assert error.details == details
        assert error.error_type == "custom_error"

    def test_config_validation_error_basic(self):
        """测试配置验证错误"""
        error = ConfigValidationError("Validation failed", config_key="test.key")
        assert error.config_key == "test.key"
        assert error.errors == []

    def test_config_validation_error_with_errors(self):
        """测试带验证错误的配置验证错误"""
        validation_errors = ["error1", "error2"]
        error = ConfigValidationError("Validation failed", validation_errors, config_key="test.key")
        assert error.errors == validation_errors

    def test_config_validation_error_kwargs(self):
        """测试通过kwargs传递参数的配置验证错误"""
        error = ConfigValidationError("Validation failed", config_key="test.key", validation_errors=["error1"])
        assert error.config_key == "test.key"
        assert error.errors == ["error1"]

    def test_config_load_error_basic(self):
        """测试配置加载错误"""
        error = ConfigLoadError("Load failed")
        assert error.context.get('source') is None
        assert error.source is None

    def test_config_load_error_with_context(self):
        """测试带上下文的配置加载错误"""
        context = {"source": "file.json", "path": "/tmp/config.json"}
        error = ConfigLoadError("Load failed", context)
        assert error.context == context
        assert error.source == "file.json"

    def test_config_load_error_with_source(self):
        """测试带源信息的配置加载错误"""
        error = ConfigLoadError("Load failed", "database", config_key="test.key")
        assert error.source == "database"
        assert error.config_key == "test.key"

    def test_config_not_found_error(self):
        """测试配置未找到错误"""
        error = ConfigNotFoundError("test.key")
        assert error.config_key == "test.key"
        assert "test.key" in str(error)

    def test_config_not_found_error_with_locations(self):
        """测试带搜索位置的配置未找到错误"""
        locations = ["/etc/config", "~/.config"]
        error = ConfigNotFoundError("test.key", locations)
        assert error.config_key == "test.key"
        assert error.details['searched_locations'] == locations

    def test_config_type_error_basic(self):
        """测试配置类型错误"""
        error = ConfigTypeError("Type mismatch")
        assert error.config_key is None

    def test_config_type_error_with_types(self):
        """测试带类型信息的配置类型错误"""
        error = ConfigTypeError("Type mismatch", "int", "str", "test.key", "invalid_value")
        assert error.config_key == "test.key"
        assert error.details['expected_type'] == "int"
        assert error.details['actual_type'] == "str"
        assert error.details['value'] == "invalid_value"

    def test_config_value_error_basic(self):
        """测试配置值错误"""
        error = ConfigValueError("Invalid value")
        assert error.config_key is None

    def test_config_value_error_with_details(self):
        """测试带详细信息的配置值错误"""
        error = ConfigValueError("Invalid value", "test.key", "int", "str", "invalid")
        assert error.config_key == "test.key"
        assert error.details['expected_type'] == "int"
        assert error.details['actual_type'] == "str"
        assert error.details['value'] == "invalid"
        assert "test.key" in str(error)
        assert "int" in str(error)
        assert "str" in str(error)

    def test_config_access_error(self):
        """测试配置访问错误"""
        error = ConfigAccessError("Access denied", "test.key")
        assert error.config_key == "test.key"
        assert error.error_type == "config_error"

    def test_config_security_error(self):
        """测试配置安全错误"""
        error = ConfigSecurityError("Security violation", "test.key")
        assert error.config_key == "test.key"
        assert error.error_type == "config_error"

    def test_config_timeout_error(self):
        """测试配置超时错误"""
        error = ConfigTimeoutError("Operation timed out", timeout_seconds=30)
        assert error.details['timeout_seconds'] == 30
        assert error.error_type == "config_error"

    def test_config_network_error(self):
        """测试配置网络错误"""
        error = ConfigNetworkError("Network failed", endpoint="localhost:8080", timeout=30.0)
        assert error.details['endpoint'] == "localhost:8080"
        assert error.details['timeout'] == 30.0
        assert error.error_type == "config_error"

    def test_config_format_error(self):
        """测试配置格式错误"""
        error = ConfigFormatError("Invalid format", format_type="json", line_number=10)
        assert error.details['format_type'] == "json"
        assert error.details['line_number'] == 10
        assert error.error_type == "config_error"

    def test_config_merge_error(self):
        """测试配置合并错误"""
        conflict_keys = ["database.host", "cache.port"]
        error = ConfigMergeError("Merge conflict", conflict_keys)
        assert error.details['conflicting_keys'] == conflict_keys
        assert error.error_type == "config_error"

    def test_config_version_error(self):
        """测试配置版本错误"""
        error = ConfigVersionError("Version mismatch", version="1.0", expected_version="2.0")
        assert error.details['version'] == "1.0"
        assert error.details['expected_version'] == "2.0"
        assert error.error_type == "config_error"

    def test_config_backup_error(self):
        """测试配置备份错误"""
        error = ConfigBackupError("Backup failed", backup_path="/tmp/backup")
        assert error.details['backup_path'] == "/tmp/backup"
        assert error.error_type == "config_error"

    def test_config_restore_error(self):
        """测试配置恢复错误"""
        error = ConfigRestoreError("Restore failed", backup_path="/tmp/backup", restore_point="corrupted")
        assert error.details['backup_path'] == "/tmp/backup"
        assert error.details['restore_point'] == "corrupted"
        assert error.error_type == "config_error"

    def test_config_encryption_error(self):
        """测试配置加密错误"""
        error = ConfigEncryptionError("Encryption failed", operation="encrypt")
        assert error.details['operation'] == "encrypt"
        assert error.error_type == "config_error"

    def test_config_decryption_error(self):
        """测试配置解密错误"""
        error = ConfigDecryptionError("Decryption failed", operation="decrypt")
        assert error.details['operation'] == "decrypt"
        assert error.error_type == "config_error"

    def test_config_quota_exceeded_error(self):
        """测试配置配额超限错误"""
        error = ConfigQuotaExceededError("Quota exceeded", quota_type="config_keys", current_usage=100, limit=50)
        assert error.details['quota_type'] == "config_keys"
        assert error.details['current_usage'] == 100
        assert error.details['limit'] == 50
        assert error.error_type == "config_error"