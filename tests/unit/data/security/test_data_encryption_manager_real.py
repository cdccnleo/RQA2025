# -*- coding: utf-8 -*-
"""
数据加密管理器真实实现测试
测试 DataEncryptionManager 的核心功能
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


import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.data.security.data_encryption_manager import (
    DataEncryptionManager,
    EncryptionResult,
)


@pytest.fixture
def encryption_manager(tmp_path):
    """创建加密管理器实例"""
    key_store = tmp_path / "keys"
    return DataEncryptionManager(
        key_store_path=str(key_store),
        enable_audit=False
    )


def test_encrypt_and_decrypt_basic(encryption_manager):
    """测试基本加密解密功能"""
    test_data = "Hello, RQA2025!"
    
    # 加密
    encrypted_result = encryption_manager.encrypt_data(
        test_data,
        algorithm="AES-256-GCM"
    )
    
    assert encrypted_result is not None
    assert encrypted_result.encrypted_data is not None
    assert encrypted_result.key_id is not None
    assert encrypted_result.algorithm == "AES-256-GCM"
    
    # 解密
    decrypted_result = encryption_manager.decrypt_data(encrypted_result)
    
    assert decrypted_result is not None
    assert decrypted_result.decrypted_data.decode('utf-8') == test_data


def test_encrypt_bytes_data(encryption_manager):
    """测试字节数据加密"""
    test_bytes = b"Binary data for encryption"
    
    encrypted_result = encryption_manager.encrypt_data(
        test_bytes,
        algorithm="AES-256-GCM"
    )
    
    decrypted_result = encryption_manager.decrypt_data(encrypted_result)
    
    assert decrypted_result.decrypted_data == test_bytes


def test_generate_key_and_use(encryption_manager):
    """测试密钥生成和使用"""
    # 生成新密钥
    key_id = encryption_manager.generate_key(
        algorithm="AES-256",
        expires_in_days=30
    )
    
    assert key_id is not None
    assert key_id in encryption_manager.keys
    
    key = encryption_manager.keys[key_id]
    assert key.is_active
    assert key.expires_at is not None
    assert (key.expires_at - datetime.now()).days <= 30
    
    # 使用新生成的密钥加密
    encrypted_result = encryption_manager.encrypt_data(
        "test data",
        algorithm="AES-256-GCM",
        key_id=key_id
    )
    
    assert encrypted_result.key_id == key_id


def test_key_rotation(encryption_manager):
    """测试密钥轮换"""
    # 生成初始密钥
    initial_key_id = encryption_manager.generate_key("AES-256")
    initial_key = encryption_manager.keys[initial_key_id]
    
    # 模拟密钥使用次数达到阈值
    initial_key.usage_count = 10000
    
    # 触发密钥轮换检查
    encryption_manager._check_key_rotation(initial_key)
    
    # 验证旧密钥被标记为非活跃
    assert not initial_key.is_active
    
    # 验证有新密钥生成
    assert encryption_manager.current_key_id != initial_key_id


def test_encrypt_batch(encryption_manager):
    """测试批量加密"""
    data_list = [
        {'data': 'item1', 'metadata': {'id': 1}},
        {'data': 'item2', 'metadata': {'id': 2}},
        {'data': 'item3', 'metadata': {'id': 3}},
    ]
    
    results = encryption_manager.encrypt_batch(
        data_list,
        algorithm="AES-256-GCM"
    )
    
    assert len(results) == 3
    assert all(isinstance(r, EncryptionResult) for r in results)
    assert all(r.encrypted_data for r in results)


def test_decrypt_batch(encryption_manager):
    """测试批量解密"""
    # 先加密一批数据
    data_list = [
        {'data': 'test1', 'metadata': {}},
        {'data': 'test2', 'metadata': {}},
    ]
    
    encrypted_results = encryption_manager.encrypt_batch(
        data_list,
        algorithm="AES-256-GCM"
    )
    
    # 批量解密
    decrypted_results = encryption_manager.decrypt_batch(encrypted_results)
    
    assert len(decrypted_results) == 2
    assert decrypted_results[0].decrypted_data.decode('utf-8') == 'test1'
    assert decrypted_results[1].decrypted_data.decode('utf-8') == 'test2'


def test_cleanup_expired_keys(encryption_manager):
    """测试清理过期密钥"""
    # 生成一个已过期的密钥
    expired_key_id = encryption_manager.generate_key("AES-256")
    expired_key = encryption_manager.keys[expired_key_id]
    expired_key.expires_at = datetime.now() - timedelta(days=1)
    
    # 清理过期密钥
    cleaned_count = encryption_manager.cleanup_expired_keys()
    
    assert cleaned_count >= 1
    assert expired_key_id not in encryption_manager.keys


def test_get_encryption_stats(encryption_manager):
    """测试获取加密统计信息"""
    # 生成几个密钥
    encryption_manager.generate_key("AES-256")
    encryption_manager.generate_key("AES-256")
    
    stats = encryption_manager.get_encryption_stats()
    
    assert stats['total_keys'] >= 2
    assert stats['active_keys'] >= 2
    assert 'algorithm_usage' in stats
    assert stats['current_key_id'] is not None


def test_invalid_key_id_raises_error(encryption_manager):
    """测试无效密钥ID抛出错误"""
    with pytest.raises(ValueError, match="无效的密钥ID"):
        encryption_manager.encrypt_data(
            "test",
            algorithm="AES-256-GCM",
            key_id="nonexistent_key"
        )


def test_invalid_algorithm_raises_error(encryption_manager):
    """测试无效算法抛出错误"""
    with pytest.raises(ValueError, match="不支持的加密算法"):
        encryption_manager.encrypt_data(
            "test",
            algorithm="INVALID-ALGORITHM"
        )

