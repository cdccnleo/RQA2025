#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置安全测试
测试配置加密、访问控制、审计等安全功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import base64

from src.infrastructure.config.security.secure_config import (
    SecureEmailConfig,
    SecureConfig,
    get_email_config
)
from src.infrastructure.config.security.enhanced_secure_config import (
    EnhancedSecureConfigManager,
    ConfigEncryptionManager,
    ConfigAccessControl,
    ConfigAuditManager,
    SecurityConfig
)


class TestSecureEmailConfig:
    """测试安全邮件配置"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "email_config.json"
        self.encrypted_path = self.temp_dir / "email_config.encrypted.json"

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """测试初始化"""
        config = SecureEmailConfig(str(self.config_path))
        assert config.config_path == self.config_path
        assert config._key is None
        assert config._cipher is None

    def test_encryption_decryption(self):
        """测试加密解密功能"""
        config = SecureEmailConfig(str(self.config_path))

        test_data = "test_password_123"
        encrypted = config._encrypt_value(test_data)
        decrypted = config._decrypt_value(encrypted)

        assert decrypted == test_data
        assert encrypted != test_data

    def test_encryption_key_generation(self):
        """测试加密密钥生成"""
        config = SecureEmailConfig(str(self.config_path))

        key1 = config._get_encryption_key()
        key2 = config._get_encryption_key()

        assert key1 == key2  # 应该返回相同的密钥
        assert isinstance(key1, bytes)

    @patch.dict(os.environ, {'EMAIL_ENCRYPTION_KEY': 'dGVzdF9rZXlfMTIzNDU2Nzg5MDEyMzQ1Njc4OTA='})
    def test_encryption_key_from_env(self):
        """测试从环境变量获取加密密钥"""
        config = SecureEmailConfig(str(self.config_path))

        key = config._get_encryption_key()
        expected_key = base64.urlsafe_b64decode('dGVzdF9rZXlfMTIzNDU2Nzg5MDEyMzQ1Njc4OTA=')

        assert key == expected_key

    def test_save_load_encrypted_config(self):
        """测试保存和加载加密配置"""
        config = SecureEmailConfig(str(self.config_path))

        test_config = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "test@example.com",
            "password": "secret_password",
            "from_email": "test@example.com",
            "to_emails": ["admin@example.com"]
        }

        # 保存加密配置
        config.save_encrypted_config(test_config, str(self.encrypted_path))

        # 加载加密配置
        loaded_config = config.load_encrypted_config(str(self.encrypted_path))

        assert loaded_config["smtp_server"] == test_config["smtp_server"]
        assert loaded_config["username"] == test_config["username"]
        assert loaded_config["password"] == test_config["password"]
        assert loaded_config["to_emails"] == test_config["to_emails"]

    def test_config_validation(self):
        """测试配置验证"""
        config = SecureEmailConfig(str(self.config_path))

        # 有效配置
        valid_config = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "test@example.com",
            "password": "secret_password",
            "from_email": "test@example.com",
            "to_emails": ["admin@example.com"]
        }
        assert config.validate_config(valid_config) == True

        # 无效配置 - 缺少必需字段
        invalid_config = {
            "smtp_server": "smtp.example.com",
            "username": "test@example.com"
        }
        assert config.validate_config(invalid_config) == False

    def test_load_config_with_env_variables(self):
        """测试加载包含环境变量的配置"""
        # 创建测试配置文件
        config_data = {
            "smtp_server": "${SMTP_SERVER}",
            "smtp_port": "${SMTP_PORT}",
            "username": "${SMTP_USERNAME}",
            "password": "${SMTP_PASSWORD}",
            "from_email": "${SMTP_FROM}",
            "to_emails": ["${SMTP_TO}"]
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f)

        # 设置环境变量
        env_vars = {
            'SMTP_SERVER': 'smtp.test.com',
            'SMTP_PORT': '587',
            'SMTP_USERNAME': 'test@test.com',
            'SMTP_PASSWORD': 'test_password',
            'SMTP_FROM': 'from@test.com',
            'SMTP_TO': 'to@test.com'
        }

        with patch.dict(os.environ, env_vars):
            config = SecureEmailConfig(str(self.config_path))
            loaded_config = config.load_config()

            assert loaded_config["smtp_server"] == "smtp.test.com"
            assert loaded_config["smtp_port"] == 587
            assert loaded_config["username"] == "test@test.com"
            assert loaded_config["password"] == "test_password"
            assert loaded_config["from_email"] == "from@test.com"
            assert loaded_config["to_emails"] == ["to@test.com"]


class TestSecureConfig:
    """测试通用安全配置"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "secure_config.json"

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """测试初始化"""
        config = SecureConfig(str(self.config_path))
        assert config.config_path == self.config_path
        assert config._key is None
        assert config._cipher is None
        assert config._config_cache == {}

    def test_initialization_success(self):
        """测试初始化成功"""
        config = SecureConfig(str(self.config_path))
        assert config.initialize() == True
        assert config._cipher is not None

    def test_encrypt_decrypt_value(self):
        """测试值加密解密"""
        config = SecureConfig(str(self.config_path))
        config.initialize()

        test_value = "sensitive_data_123"
        encrypted = config.encrypt_value(test_value)
        decrypted = config.decrypt_value(encrypted)

        assert decrypted == test_value
        assert encrypted != test_value

    def test_get_set_secure_value_from_env(self):
        """测试从环境变量获取和设置安全值"""
        config = SecureConfig(str(self.config_path))

        test_key = "TEST_SECRET"
        test_value = "my_secret_value"

        # 设置安全值
        assert config.set_secure_value(test_key, test_value) == True

        # 验证环境变量已设置
        env_key = f"SECURE_{test_key.upper()}"
        assert env_key in os.environ

        # 获取安全值
        retrieved_value = config.get_secure_value(test_key)
        assert retrieved_value == test_value

    def test_get_secure_value_with_default(self):
        """测试获取不存在的安全值返回默认值"""
        config = SecureConfig(str(self.config_path))

        default_value = "default_secret"
        retrieved_value = config.get_secure_value("NON_EXISTENT_KEY", default_value)

        assert retrieved_value == default_value

    def test_get_status(self):
        """测试获取状态"""
        config = SecureConfig(str(self.config_path))

        status = config.get_status()
        assert status["initialized"] == False
        assert status["cached_values"] == 0
        assert status["config_path"] == str(self.config_path)
        assert status["encryption_enabled"] == True

        # 初始化后状态
        config.initialize()
        status = config.get_status()
        assert status["initialized"] == True


class TestEnhancedSecureConfigManager:
    """测试增强版安全配置管理器"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "config"
        self.config_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """测试初始化"""
        manager = EnhancedSecureConfigManager(str(self.config_dir))
        assert manager.config_dir == self.config_dir
        assert manager.encryption is not None
        assert manager.access_control is not None
        assert manager.audit is not None
        assert manager.hot_reload is not None

    def test_save_load_config(self):
        """测试保存和加载配置"""
        manager = EnhancedSecureConfigManager(str(self.config_dir))

        config_file = "test_config.json"
        test_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "admin",
                "password": "secret123"
            },
            "api": {
                "key": "api_secret_key",
                "timeout": 30
            }
        }

        # 保存配置
        manager.save_config(config_file, test_config, user="admin")

        # 加载配置
        loaded_config = manager.load_config(config_file, user="admin")

        assert loaded_config["database"]["host"] == "localhost"
        assert loaded_config["database"]["username"] == "admin"
        assert loaded_config["database"]["password"] == "secret123"

    def test_access_control(self):
        """测试访问控制"""
        security_config = SecurityConfig()
        access_control = ConfigAccessControl(security_config)

        # 测试管理员权限
        assert access_control.check_access("admin", "read", "test_config") == True
        assert access_control.check_access("admin", "write", "test_config") == True
        assert access_control.check_access("admin", "delete", "test_config") == True

        # 测试查看者权限
        assert access_control.check_access("viewer", "read", "test_config") == True
        assert access_control.check_access("viewer", "write", "test_config") == False

    def test_audit_logging(self):
        """测试审计日志"""
        audit_manager = ConfigAuditManager()

        # 记录配置变更
        audit_manager.log_change(
            action="update",
            key="database.password",
            old_value="old_password",
            new_value="new_password",
            user="admin",
            reason="密码更新"
        )

        # 获取审计日志
        logs = audit_manager.get_audit_logs()
        assert len(logs) == 1
        assert logs[0].action == "update"
        assert logs[0].key == "database.password"
        assert logs[0].user == "admin"

    def test_get_set_nested_value(self):
        """测试获取和设置嵌套配置值"""
        manager = EnhancedSecureConfigManager(str(self.config_dir))

        config_file = "nested_config.json"
        test_config = {
            "database": {
                "connection": {
                    "host": "localhost",
                    "port": 5432
                }
            }
        }

        # 保存配置
        manager.save_config(config_file, test_config, user="admin")

        # 获取嵌套值
        host = manager.get_value(config_file, "database.connection.host", user="admin")
        assert host == "localhost"

        # 设置嵌套值
        manager.set_value(config_file, "database.connection.port", 3306, user="admin")

        # 验证修改
        port = manager.get_value(config_file, "database.connection.port", user="admin")
        assert port == 3306


class TestConfigEncryptionManager:
    """测试配置加密管理器"""

    def test_encryption_decryption(self):
        """测试加密解密"""
        manager = ConfigEncryptionManager()

        test_data = "sensitive_configuration_data"
        encrypted = manager.encrypt(test_data)
        decrypted = manager.decrypt(encrypted)

        assert decrypted == test_data
        assert encrypted != test_data

    def test_different_contexts(self):
        """测试不同上下文的加密"""
        manager = ConfigEncryptionManager()

        data = "test_data"
        encrypted1 = manager.encrypt(data, "context1")
        encrypted2 = manager.encrypt(data, "context2")

        # 不同上下文应该产生不同的加密结果
        assert encrypted1 != encrypted2

        # 但解密应该正确
        decrypted1 = manager.decrypt(encrypted1, "context1")
        decrypted2 = manager.decrypt(encrypted2, "context2")

        assert decrypted1 == data
        assert decrypted2 == data


if __name__ == "__main__":
    pytest.main([__file__])
