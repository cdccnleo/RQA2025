#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025适配器安全配置测试
测试SecureConfigManager的功能

该文件位于 tests/unit/adapters/ 目录下，遵循标准的测试目录结构：
- tests/unit/ : 单元测试
- tests/integration/ : 集成测试
- tests/e2e/ : 端到端测试
"""

import sys
import os
import tempfile
import shutil
import pytest
from pathlib import Path

# 添加项目根目录到路径，以便导入模块
project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入适配器模块
import importlib
try:
    adapters_module = importlib.import_module('src.adapters')
    SecureConfigManager = getattr(adapters_module, 'SecureConfigManager', None)
    if SecureConfigManager is None:
        pytest.skip("SecureConfigManager不可用", allow_module_level=True)
except ImportError:
    pytest.skip("适配器模块导入失败", allow_module_level=True)

@pytest.fixture
def temp_key_file():
    """临时密钥文件fixture"""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
        # 生成有效的Fernet密钥并写入文件
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        temp_file.write(key)
    yield temp_file_path
    # 清理
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)

@pytest.fixture
def secure_manager(temp_key_file):
    """安全配置管理器fixture"""
    return SecureConfigManager(temp_key_file)

def test_secure_config_manager_initialization(secure_manager):
    """测试安全配置管理器初始化"""
    assert secure_manager is not None
    assert hasattr(secure_manager, 'encrypt_sensitive_data')
    assert hasattr(secure_manager, 'decrypt_sensitive_data')
    assert hasattr(secure_manager, 'secure_config')
    assert hasattr(secure_manager, 'unsecure_config')

def test_secure_config_encryption_decryption(secure_manager):
    """测试配置的加密和解密功能"""
    # 测试数据
    test_config = {
        'host': 'localhost',
        'port': 8888,
        'username': 'test_user',
        'password': 'secret_password_123',
        'api_key': 'sk-1234567890abcde',
        'token': 'bearer_token_xyz',
        'database': {
            'host': 'db.example.com',
            'password': 'db_secret_456'
        },
        'normal_setting': 'normal_value'
    }

    # 测试配置加密
    secure_config = secure_manager.secure_config(test_config)
    assert secure_config is not None

    # 验证敏感字段已被加密 (password, api_key, token等)
    assert secure_config['password'] != test_config['password']
    assert secure_config['api_key'] != test_config['api_key']
    assert secure_config['token'] != test_config['token']
    assert secure_config['database']['password'] != test_config['database']['password']

    # 验证非敏感字段未被加密
    assert secure_config['host'] == test_config['host']
    assert secure_config['port'] == test_config['port']
    assert secure_config['normal_setting'] == test_config['normal_setting']

    # 测试配置解密
    unsecure_config = secure_manager.unsecure_config(secure_config)

    # 验证解密后与原始配置一致
    assert unsecure_config == test_config

def test_individual_encrypt_decrypt(secure_manager):
    """测试单独的加密/解密功能"""
    test_password = "my_secret_password"

    # 测试加密
    encrypted = secure_manager.encrypt_sensitive_data(test_password)
    assert encrypted is not None
    assert encrypted != test_password

    # 测试解密
    decrypted = secure_manager.decrypt_sensitive_data(encrypted)
    assert decrypted == test_password

def test_key_rotation(secure_manager):
    """测试密钥轮换功能"""
    # 测试密钥轮换
    rotate_success = secure_manager.rotate_key()
    assert rotate_success is True

    # 验证轮换后加密/解密仍然正常
    test_password = "password_after_rotate"
    encrypted = secure_manager.encrypt_sensitive_data(test_password)
    decrypted = secure_manager.decrypt_sensitive_data(encrypted)

    assert decrypted == test_password

def test_secure_config_edge_cases(secure_manager):
    """测试边界情况"""
    # 测试空配置
    empty_config = {}
    secure_empty = secure_manager.secure_config(empty_config)
    unsecure_empty = secure_manager.unsecure_config(secure_empty)
    assert unsecure_empty == empty_config

    # 测试只有非敏感数据的配置
    non_sensitive_config = {
        'host': 'localhost',
        'port': 8080,
        'timeout': 30
    }
    secure_non_sensitive = secure_manager.secure_config(non_sensitive_config)
    # 非敏感数据应该保持不变
    assert secure_non_sensitive == non_sensitive_config

def test_encrypt_decrypt_various_data_types(secure_manager):
    """测试不同数据类型的加密解密"""
    # 测试字符串
    test_str = "test_string"
    encrypted_str = secure_manager.encrypt_sensitive_data(test_str)
    decrypted_str = secure_manager.decrypt_sensitive_data(encrypted_str)
    assert decrypted_str == test_str

    # 测试包含特殊字符的字符串
    test_special = "test@#$%^&*()_+{}|:<>?[]\\;',./"
    encrypted_special = secure_manager.encrypt_sensitive_data(test_special)
    decrypted_special = secure_manager.decrypt_sensitive_data(encrypted_special)
    assert decrypted_special == test_special

    # 测试空字符串
    test_empty = ""
    encrypted_empty = secure_manager.encrypt_sensitive_data(test_empty)
    decrypted_empty = secure_manager.decrypt_sensitive_data(encrypted_empty)
    assert decrypted_empty == test_empty


def test_encrypt_decrypt_large_data(secure_manager):
    """测试大数据量的加密解密"""
    # 测试大数据量
    large_data = "A" * 10000  # 10KB数据
    encrypted_large = secure_manager.encrypt_sensitive_data(large_data)
    decrypted_large = secure_manager.decrypt_sensitive_data(encrypted_large)
    assert decrypted_large == large_data

    # 验证加密后的数据与原始数据不同
    assert encrypted_large != large_data


def test_encrypt_decrypt_unicode_data(secure_manager):
    """测试Unicode数据的加密解密"""
    # 测试中文
    chinese_data = "你好，世界！这是一个测试字符串。"
    encrypted_chinese = secure_manager.encrypt_sensitive_data(chinese_data)
    decrypted_chinese = secure_manager.decrypt_sensitive_data(encrypted_chinese)
    assert decrypted_chinese == chinese_data

    # 测试表情符号
    emoji_data = "🚀✨🎉🔥💯"
    encrypted_emoji = secure_manager.encrypt_sensitive_data(emoji_data)
    decrypted_emoji = secure_manager.decrypt_sensitive_data(encrypted_emoji)
    assert decrypted_emoji == emoji_data


def test_secure_config_with_special_characters(secure_manager):
    """测试包含特殊字符的配置"""
    special_config = {
        'password': 'p@ssw0rd!#$%^&*()',
        'token': 'token_with_特殊字符_🚀',
        'url': 'https://example.com/path?param=value&other=测试',
        'normal': 'normal_value'
    }

    secure_config = secure_manager.secure_config(special_config)
    unsecure_config = secure_manager.unsecure_config(secure_config)

    assert unsecure_config == special_config


def test_multiple_secure_managers():
    """测试多个SecureConfigManager实例"""
    import tempfile
    import os

    managers = []
    keys = []

    # 创建多个管理器实例
    for i in range(3):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            temp_file.write(key)

        manager = SecureConfigManager(temp_file_path)
        managers.append(manager)
        keys.append(temp_file_path)

    try:
        test_data = "test_data_123"

        # 测试每个管理器都能独立工作
        for i, manager in enumerate(managers):
            encrypted = manager.encrypt_sensitive_data(f"{test_data}_{i}")
            decrypted = manager.decrypt_sensitive_data(encrypted)
            assert decrypted == f"{test_data}_{i}"

        # 验证不同管理器使用不同的密钥
        encrypted_0 = managers[0].encrypt_sensitive_data(test_data)
        encrypted_1 = managers[1].encrypt_sensitive_data(test_data)

        # 不同密钥加密的结果应该不同
        assert encrypted_0 != encrypted_1

        # 验证交叉解密失败
        with pytest.raises(Exception):
            managers[0].decrypt_sensitive_data(encrypted_1)

    finally:
        # 清理临时文件
        for key_file in keys:
            if os.path.exists(key_file):
                os.unlink(key_file)
