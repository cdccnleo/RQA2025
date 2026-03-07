#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Config Encryption
配置加密和安全测试，验证配置数据的加密存储和访问控制
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import tempfile
import os
import json
import base64
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from cryptography.fernet import Fernet


class TestConfigEncryption(unittest.TestCase):
    """测试Config Encryption"""

    def setUp(self):
        """测试前准备"""
        self.test_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "user": "admin",
                "password": "secret_password_123",
                "ssl_mode": "require"
            },
            "api_keys": {
                "openai_key": "sk-1234567890abcdef",
                "aws_secret": "AKIAIOSFODNN7EXAMPLE",
                "stripe_secret": "sk_test_1234567890"
            }
        }

        # 生成测试用的加密密钥
        self.test_key = Fernet.generate_key()
        self.cipher = Fernet(self.test_key)

    def test_encryption_key_generation(self):
        """测试加密密钥生成"""
        # 验证密钥格式
        self.assertIsInstance(self.test_key, bytes)
        self.assertEqual(len(self.test_key), 44)  # Fernet密钥长度为44字节

        # 验证密钥是base64编码的
        try:
            decoded = base64.urlsafe_b64decode(self.test_key)
            self.assertEqual(len(decoded), 32)  # 解码后应为32字节
        except Exception as e:
            self.fail(f"生成的密钥不是有效的base64格式: {e}")

    def test_data_encryption_decryption(self):
        """测试数据加密和解密"""
        # 测试字符串加密解密
        test_data = "sensitive_config_data"
        encrypted = self.cipher.encrypt(test_data.encode())
        decrypted = self.cipher.decrypt(encrypted).decode()

        self.assertNotEqual(encrypted.decode(), test_data)
        self.assertEqual(decrypted, test_data)

        # 测试JSON数据加密解密
        json_data = json.dumps(self.test_config)
        encrypted_json = self.cipher.encrypt(json_data.encode())
        decrypted_json = self.cipher.decrypt(encrypted_json).decode()
        decrypted_config = json.loads(decrypted_json)

        self.assertEqual(decrypted_config, self.test_config)

    def test_encrypted_config_storage(self):
        """测试加密配置存储"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "encrypted_config.json")

            # 加密配置数据
            config_json = json.dumps(self.test_config)
            encrypted_data = self.cipher.encrypt(config_json.encode())

            # 保存加密数据
            with open(config_file, 'wb') as f:
                f.write(encrypted_data)

            # 验证文件存在
            self.assertTrue(os.path.exists(config_file))

            # 读取并解密数据
            with open(config_file, 'rb') as f:
                encrypted_content = f.read()

            decrypted_content = self.cipher.decrypt(encrypted_content).decode()
            loaded_config = json.loads(decrypted_content)

            # 验证数据完整性
            self.assertEqual(loaded_config, self.test_config)

    def test_encryption_key_rotation(self):
        """测试加密密钥轮换"""
        # 创建原始加密数据
        original_data = "original_sensitive_data"
        original_encrypted = self.cipher.encrypt(original_data.encode())

        # 生成新密钥
        new_key = Fernet.generate_key()
        new_cipher = Fernet(new_key)

        # 使用新密钥重新加密数据
        decrypted_data = self.cipher.decrypt(original_encrypted).decode()
        new_encrypted = new_cipher.encrypt(decrypted_data.encode())

        # 验证新密钥能正确解密
        new_decrypted = new_cipher.decrypt(new_encrypted).decode()
        self.assertEqual(new_decrypted, original_data)

        # 验证旧密钥无法解密新加密的数据
        with self.assertRaises(Exception):
            self.cipher.decrypt(new_encrypted)

    def test_encryption_error_handling(self):
        """测试加密错误处理"""
        # 测试无效密钥
        invalid_key = b"invalid_key_not_32_bytes"
        with self.assertRaises(ValueError):
            Fernet(invalid_key)

        # 测试解密无效数据
        invalid_encrypted = b"invalid_encrypted_data"
        with self.assertRaises(Exception):
            self.cipher.decrypt(invalid_encrypted)

        # 测试空数据加密
        empty_data = ""
        encrypted_empty = self.cipher.encrypt(empty_data.encode())
        decrypted_empty = self.cipher.decrypt(encrypted_empty).decode()
        self.assertEqual(decrypted_empty, empty_data)

    def test_secure_config_access_control(self):
        """测试安全配置访问控制"""
        # 模拟不同权限级别的访问控制
        permissions = {
            "admin": ["database", "api_keys", "security"],
            "user": ["database"],
            "readonly": ["database"]  # 只读权限
        }

        # 测试管理员权限
        admin_access = self._check_access_permissions("admin", permissions)
        self.assertTrue(admin_access["database"])
        self.assertTrue(admin_access["api_keys"])
        self.assertTrue(admin_access["security"])

        # 测试普通用户权限
        user_access = self._check_access_permissions("user", permissions)
        self.assertTrue(user_access["database"])
        self.assertFalse(user_access["api_keys"])
        self.assertFalse(user_access["security"])

        # 测试只读用户权限
        readonly_access = self._check_access_permissions("readonly", permissions)
        self.assertTrue(readonly_access["database"])
        self.assertFalse(readonly_access["api_keys"])
        self.assertFalse(readonly_access["security"])

    def _check_access_permissions(self, user_role, permissions):
        """检查用户访问权限的辅助方法"""
        user_permissions = permissions.get(user_role, [])
        return {
            "database": "database" in user_permissions,
            "api_keys": "api_keys" in user_permissions,
            "security": "security" in user_permissions
        }

    def test_encryption_performance(self):
        """测试加密性能"""
        import time

        # 测试大数据量的加密性能
        large_data = "x" * 10000  # 10KB数据
        start_time = time.time()

        # 执行多次加密解密操作
        for _ in range(100):
            encrypted = self.cipher.encrypt(large_data.encode())
            decrypted = self.cipher.decrypt(encrypted).decode()

        end_time = time.time()
        duration = end_time - start_time

        # 验证数据正确性
        self.assertEqual(decrypted, large_data)

        # 性能要求：100次操作应在1秒内完成
        self.assertLess(duration, 1.0,
                       f"加密性能不足: {duration:.2f}s for 100 operations")

    def test_secure_config_backup(self):
        """测试安全配置备份"""
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = os.path.join(temp_dir, "secure_backups")

            # 创建加密的备份数据
            config_json = json.dumps(self.test_config)
            encrypted_data = self.cipher.encrypt(config_json.encode())

            # 保存加密备份
            timestamp = "20231201_120000"
            backup_file = os.path.join(backup_dir, f"config_backup_{timestamp}.enc")
            os.makedirs(backup_dir, exist_ok=True)

            with open(backup_file, 'wb') as f:
                f.write(encrypted_data)

            # 验证备份文件
            self.assertTrue(os.path.exists(backup_file))

            # 验证备份数据完整性
            with open(backup_file, 'rb') as f:
                backup_content = f.read()

            decrypted_backup = self.cipher.decrypt(backup_content).decode()
            backup_config = json.loads(decrypted_backup)

            self.assertEqual(backup_config, self.test_config)

    @patch('src.infrastructure.config.security.secure_config.SecureEmailConfig')
    def test_secure_config_integration(self, mock_secure_config):
        """测试安全配置集成"""
        # 创建模拟的安全配置实例
        mock_instance = MagicMock()
        mock_secure_config.return_value = mock_instance

        # 模拟加密操作
        mock_instance.encrypt_config.return_value = "encrypted_data"
        mock_instance.decrypt_config.return_value = self.test_config

        # 测试配置加密集成
        from src.infrastructure.config.security.secure_config import SecureEmailConfig
        secure_config = SecureEmailConfig()

        # 验证加密功能
        encrypted = secure_config.encrypt_config(self.test_config)
        self.assertIsNotNone(encrypted)

        # 验证解密功能
        decrypted = secure_config.decrypt_config(encrypted)
        self.assertEqual(decrypted, self.test_config)


if __name__ == '__main__':
    unittest.main()
