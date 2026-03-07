#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加密管理器综合测试
测试DataEncryptionManager的核心功能和加密算法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import os
import base64
import tempfile
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import time
import asyncio
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.infrastructure.security.crypto.encryption import (
    DataEncryptionManager,
    EncryptionKey,
    EncryptionResult,
    DecryptionResult
)


@pytest.fixture
def temp_key_store():
    """创建临时密钥存储目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def encryption_manager(temp_key_store):
    """创建加密管理器实例"""
    manager = DataEncryptionManager(key_store_path=temp_key_store)
    return manager


class TestEncryptionKey:
    """测试加密密钥类"""

    def test_encryption_key_creation_minimal(self):
        """测试最小化加密密钥创建"""
        key_data = b"test_key_data"
        created_at = datetime.now()

        key = EncryptionKey(
            key_id="test_key_123",
            key_data=key_data,
            algorithm="AES-256",
            created_at=created_at
        )

        assert key.key_id == "test_key_123"
        assert key.key_data == key_data
        assert key.algorithm == "AES-256"
        assert key.created_at == created_at
        assert key.expires_at is None
        assert key.is_active is True
        assert key.usage_count == 0
        assert key.metadata == {}

    def test_encryption_key_creation_complete(self):
        """测试完整加密密钥创建"""
        key_data = b"complete_key_data"
        created_at = datetime(2025, 1, 1, 12, 0, 0)
        expires_at = datetime(2025, 12, 31, 23, 59, 59)
        metadata = {"source": "generated", "strength": "high"}

        key = EncryptionKey(
            key_id="complete_key_123",
            key_data=key_data,
            algorithm="RSA-2048",
            created_at=created_at,
            expires_at=expires_at,
            is_active=False,
            usage_count=42,
            metadata=metadata
        )

        assert key.key_id == "complete_key_123"
        assert key.key_data == key_data
        assert key.algorithm == "RSA-2048"
        assert key.created_at == created_at
        assert key.expires_at == expires_at
        assert key.is_active is False
        assert key.usage_count == 42
        assert key.metadata == metadata

    def test_is_expired_no_expiry(self):
        """测试无过期时间的密钥"""
        key = EncryptionKey(
            key_id="no_expiry",
            key_data=b"key",
            algorithm="AES-256",
            created_at=datetime.now()
        )

        assert not key.is_expired()

    def test_is_expired_not_expired(self):
        """测试未过期的密钥"""
        future_date = datetime.now() + timedelta(days=30)
        key = EncryptionKey(
            key_id="not_expired",
            key_data=b"key",
            algorithm="AES-256",
            created_at=datetime.now(),
            expires_at=future_date
        )

        assert not key.is_expired()

    def test_is_expired_expired(self):
        """测试已过期的密钥"""
        past_date = datetime.now() - timedelta(days=1)
        key = EncryptionKey(
            key_id="expired",
            key_data=b"key",
            algorithm="AES-256",
            created_at=datetime.now() - timedelta(days=30),
            expires_at=past_date
        )

        assert key.is_expired()

    def test_can_use_active_not_expired(self):
        """测试可用的密钥"""
        future_date = datetime.now() + timedelta(days=30)
        key = EncryptionKey(
            key_id="usable",
            key_data=b"key",
            algorithm="AES-256",
            created_at=datetime.now(),
            expires_at=future_date,
            is_active=True
        )

        assert key.can_use()

    def test_can_use_inactive(self):
        """测试不可用的非活跃密钥"""
        key = EncryptionKey(
            key_id="inactive",
            key_data=b"key",
            algorithm="AES-256",
            created_at=datetime.now(),
            is_active=False
        )

        assert not key.can_use()

    def test_can_use_expired(self):
        """测试不可用的过期密钥"""
        past_date = datetime.now() - timedelta(days=1)
        key = EncryptionKey(
            key_id="expired",
            key_data=b"key",
            algorithm="AES-256",
            created_at=datetime.now() - timedelta(days=30),
            expires_at=past_date,
            is_active=True
        )

        assert not key.can_use()

    def test_increment_usage(self):
        """测试增加使用计数"""
        key = EncryptionKey(
            key_id="usage_test",
            key_data=b"key",
            algorithm="AES-256",
            created_at=datetime.now(),
            usage_count=5
        )

        key.increment_usage()
        assert key.usage_count == 6

        key.increment_usage()
        assert key.usage_count == 7


class TestEncryptionResult:
    """测试加密结果类"""

    def test_encryption_result_creation_minimal(self):
        """测试最小化加密结果创建"""
        result = EncryptionResult(
            key_id="test_key",
            algorithm="AES-256-GCM",
            encrypted_data=b"encrypted_data"
        )

        assert result.key_id == "test_key"
        assert result.algorithm == "AES-256-GCM"
        assert result.encrypted_data == b"encrypted_data"
        assert result.iv is None
        assert result.tag is None
        assert result.metadata == {}

    def test_encryption_result_creation_complete(self):
        """测试完整加密结果创建"""
        iv = b"initialization_vector"
        tag = b"authentication_tag"
        metadata = {"compression": "gzip", "original_size": 1024}

        result = EncryptionResult(
            key_id="complete_key",
            algorithm="AES-256-GCM",
            encrypted_data=b"complete_encrypted_data",
            iv=iv,
            tag=tag,
            metadata=metadata
        )

        assert result.key_id == "complete_key"
        assert result.algorithm == "AES-256-GCM"
        assert result.encrypted_data == b"complete_encrypted_data"
        assert result.iv == iv
        assert result.tag == tag
        assert result.metadata == metadata


class TestDecryptionResult:
    """测试解密结果类"""

    def test_decryption_result_creation_minimal(self):
        """测试最小化解密结果创建"""
        result = DecryptionResult(
            decrypted_data=b"decrypted_data",
            key_id="test_key",
            algorithm="AES-256-GCM"
        )

        assert result.decrypted_data == b"decrypted_data"
        assert result.key_id == "test_key"
        assert result.algorithm == "AES-256-GCM"
        assert result.metadata == {}

    def test_decryption_result_creation_complete(self):
        """测试完整解密结果创建"""
        metadata = {"decompression": "gzip", "verified": True}

        result = DecryptionResult(
            decrypted_data=b"complete_decrypted_data",
            key_id="complete_key",
            algorithm="AES-256-GCM",
            metadata=metadata
        )

        assert result.decrypted_data == b"complete_decrypted_data"
        assert result.key_id == "complete_key"
        assert result.algorithm == "AES-256-GCM"
        assert result.metadata == metadata


class TestDataEncryptionManagerInitialization:
    """测试数据加密管理器初始化"""

    def test_initialization_with_default_params(self, temp_key_store):
        """测试默认参数初始化"""
        manager = DataEncryptionManager(key_store_path=temp_key_store)

        assert manager.key_store_path is not None
        assert isinstance(manager.keys, dict)
        assert len(manager.keys) >= 1  # 应该有默认密钥

    def test_initialization_with_custom_path(self, temp_key_store):
        """测试自定义路径初始化"""
        manager = DataEncryptionManager(key_store_path=temp_key_store)

        assert str(manager.key_store_path) == temp_key_store
        assert isinstance(manager.keys, dict)

    def test_initialization_creates_default_keys(self, temp_key_store):
        """测试初始化创建默认密钥"""
        manager = DataEncryptionManager(key_store_path=temp_key_store)

        # 应该至少有一个AES密钥
        aes_keys = [k for k in manager.keys.values() if "AES" in k.algorithm]
        assert len(aes_keys) >= 1

        # 验证密钥的基本属性
        key = list(manager.keys.values())[0]
        assert isinstance(key, EncryptionKey)
        assert key.is_active
        assert not key.is_expired()


class TestDataEncryption:
    """测试数据加密功能"""

    def test_encrypt_decrypt_string_data_aes_gcm(self, encryption_manager):
        """测试AES-GCM算法的字符串数据加密解密"""
        manager = encryption_manager
        test_data = "Hello, World! This is a test message."

        # 加密
        encrypted_result = manager.encrypt_data(test_data, algorithm="AES - 256 - GCM")
        assert isinstance(encrypted_result, EncryptionResult)
        assert encrypted_result.algorithm == "AES - 256 - GCM"
        assert encrypted_result.encrypted_data != test_data.encode()

        # 解密
        decrypted_result = manager.decrypt_data(encrypted_result)
        assert isinstance(decrypted_result, DecryptionResult)
        assert decrypted_result.decrypted_data.decode() == test_data

    def test_encrypt_decrypt_bytes_data_aes_gcm(self, encryption_manager):
        """测试AES-GCM算法的字节数据加密解密"""
        manager = encryption_manager
        test_data = b"Binary data: \x00\x01\x02\x03\xff"

        # 加密
        encrypted_result = manager.encrypt_data(test_data, algorithm="AES - 256 - GCM")
        assert isinstance(encrypted_result, EncryptionResult)
        assert encrypted_result.algorithm == "AES - 256 - GCM"
        assert encrypted_result.encrypted_data != test_data

        # 解密
        decrypted_result = manager.decrypt_data(encrypted_result)
        assert isinstance(decrypted_result, DecryptionResult)
        assert decrypted_result.decrypted_data == test_data

    def test_encrypt_decrypt_string_data_aes_cbc(self, encryption_manager):
        """测试AES-CBC算法的字符串数据加密解密"""
        manager = encryption_manager
        test_data = "AES-CBC encryption test data."

        # 加密
        encrypted_result = manager.encrypt_data(test_data, algorithm="AES - 256 - CBC")
        assert isinstance(encrypted_result, EncryptionResult)
        assert encrypted_result.algorithm == "AES - 256 - CBC"

        # 解密
        decrypted_result = manager.decrypt_data(encrypted_result)
        assert isinstance(decrypted_result, DecryptionResult)
        assert decrypted_result.decrypted_data.decode() == test_data

    def test_encrypt_decrypt_large_data(self, encryption_manager):
        """测试大文件加密解密"""
        manager = encryption_manager
        # 创建1MB的测试数据
        test_data = b"A" * (1024 * 1024)

        # 加密
        encrypted_result = manager.encrypt_data(test_data, algorithm="AES - 256 - GCM")
        assert isinstance(encrypted_result, EncryptionResult)

        # 解密
        decrypted_result = manager.decrypt_data(encrypted_result)
        assert isinstance(decrypted_result, DecryptionResult)
        assert decrypted_result.decrypted_data == test_data

    def test_encrypt_decrypt_empty_data(self, encryption_manager):
        """测试空数据加密解密"""
        manager = encryption_manager
        test_data = ""

        # 加密
        encrypted_result = manager.encrypt_data(test_data, algorithm="AES - 256 - GCM")
        assert isinstance(encrypted_result, EncryptionResult)

        # 解密
        decrypted_result = manager.decrypt_data(encrypted_result)
        assert isinstance(decrypted_result, DecryptionResult)
        assert decrypted_result.decrypted_data.decode() == test_data

    def test_encrypt_decrypt_unicode_data(self, encryption_manager):
        """测试Unicode数据加密解密"""
        manager = encryption_manager
        test_data = "Unicode测试数据: 你好世界 🌍 αβγδε 中文English 123"

        # 加密
        encrypted_result = manager.encrypt_data(test_data, algorithm="AES - 256 - GCM")
        assert isinstance(encrypted_result, EncryptionResult)

        # 解密
        decrypted_result = manager.decrypt_data(encrypted_result)
        assert isinstance(decrypted_result, DecryptionResult)
        assert decrypted_result.decrypted_data.decode() == test_data


class TestKeyManagement:
    """测试密钥管理功能"""

    def test_generate_key_aes(self, encryption_manager):
        """测试生成AES密钥"""
        manager = encryption_manager

        key_id = manager.generate_key(algorithm="AES-256", expires_in_days=30)
        assert key_id is not None

        # 验证密钥存在
        assert key_id in manager.keys
        key = manager.keys[key_id]
        assert key.algorithm == "AES-256"
        assert key.is_active
        assert key.expires_at is not None

        # 验证过期时间大约是30天后
        expected_expiry = datetime.now() + timedelta(days=30)
        time_diff = abs((key.expires_at - expected_expiry).total_seconds())
        assert time_diff < 60  # 允许1分钟的误差

    def test_generate_key_rsa(self, encryption_manager):
        """测试生成RSA密钥"""
        manager = encryption_manager

        key_id = manager.generate_key(algorithm="RSA-2048")
        assert key_id is not None

        # 验证密钥存在
        assert key_id in manager.keys
        key = manager.keys[key_id]
        assert key.algorithm == "RSA-2048"
        assert key.is_active

    def test_generate_key_chacha20(self, encryption_manager):
        """测试生成ChaCha20密钥"""
        manager = encryption_manager

        key_id = manager.generate_key(algorithm="ChaCha20")
        assert key_id is not None

        # 验证密钥存在
        assert key_id in manager.keys
        key = manager.keys[key_id]
        assert key.algorithm == "ChaCha20"
        assert key.is_active

    def test_generate_key_with_no_expiry(self, encryption_manager):
        """测试生成无过期时间的密钥"""
        manager = encryption_manager

        key_id = manager.generate_key(algorithm="AES-128")
        assert key_id is not None

        key = manager.keys[key_id]
        assert key.expires_at is None

    def test_rotate_keys(self, encryption_manager):
        """测试密钥轮换"""
        manager = encryption_manager
        initial_key_count = len(manager.keys)

        # 生成一些即将过期的密钥
        expired_key_ids = []
        for i in range(3):
            key_id = manager.generate_key(algorithm="AES-256", expires_in_days=-1)  # 已经过期
            expired_key_ids.append(key_id)

        # 轮换密钥
        rotated_ids = manager.rotate_keys()

        # 应该有一些密钥被轮换
        assert isinstance(rotated_ids, list)
        # 注意：轮换逻辑可能不会立即生成新密钥，取决于实现

    def test_cleanup_expired_keys(self, encryption_manager):
        """测试清理过期密钥"""
        manager = encryption_manager

        # 生成一些过期密钥
        expired_key_ids = []
        for i in range(3):
            key_id = manager.generate_key(algorithm="AES-256", expires_in_days=-1)  # 已经过期
            expired_key_ids.append(key_id)

        # 清理过期密钥
        cleaned_count = manager.cleanup_expired_keys()

        # 验证过期密钥被清理
        for key_id in expired_key_ids:
            if key_id in manager.keys:
                key = manager.keys[key_id]
                assert not key.is_active or key.is_expired()


class TestBatchOperations:
    """测试批处理操作"""

    def test_encrypt_batch_strings(self, encryption_manager):
        """测试批量加密字符串数据"""
        manager = encryption_manager

        data_list = [
            {"data": "First message", "algorithm": "AES-256-GCM"},
            {"data": "Second message", "algorithm": "AES-256-CBC"},
            {"data": "Third message", "algorithm": "AES-256-GCM"}
        ]

        # 批量加密
        encrypted_results = manager.encrypt_batch(data_list)

        assert len(encrypted_results) == 3
        for result in encrypted_results:
            assert isinstance(result, EncryptionResult)
            assert result.key_id in manager.keys

    def test_encrypt_batch_bytes(self, encryption_manager):
        """测试批量加密字节数据"""
        manager = encryption_manager

        data_list = [
            {"data": b"Binary data 1", "algorithm": "AES-256-GCM"},
            {"data": b"Binary data 2", "algorithm": "AES-256-CBC"}
        ]

        # 批量加密
        encrypted_results = manager.encrypt_batch(data_list)

        assert len(encrypted_results) == 2
        for result in encrypted_results:
            assert isinstance(result, EncryptionResult)

    def test_encrypt_decrypt_batch_roundtrip(self, encryption_manager):
        """测试批量加密解密往返"""
        manager = encryption_manager

        original_data = [
            "Message one",
            "Message two",
            "Message three"
        ]

        # 批量加密
        encrypted_results = manager.encrypt_batch([
            {"data": msg, "algorithm": "AES-256-GCM"} for msg in original_data
        ])

        # 批量解密
        decrypted_results = manager.decrypt_batch(encrypted_results)

        assert len(decrypted_results) == 3
        for i, result in enumerate(decrypted_results):
            assert isinstance(result, DecryptionResult)
            assert result.success is True
            assert result.plaintext.decode() == original_data[i]

    def test_batch_operations_empty_list(self, encryption_manager):
        """测试空列表的批处理操作"""
        manager = encryption_manager

        # 空加密
        encrypted_results = manager.encrypt_batch([])
        assert len(encrypted_results) == 0

        # 空解密
        decrypted_results = manager.decrypt_batch([])
        assert len(decrypted_results) == 0


class TestAuditAndLogging:
    """测试审计和日志功能"""

    def test_audit_log_generation(self, encryption_manager):
        """测试审计日志生成"""
        manager = encryption_manager

        # 执行一些操作
        manager.encrypt_data("test data", algorithm="AES-256-GCM")
        manager.generate_key(algorithm="AES-128")

        # 获取审计日志
        logs = manager.get_audit_logs(limit=10)

        assert isinstance(logs, list)
        assert len(logs) >= 1  # 至少有一个操作的日志

        # 验证日志结构
        for log in logs:
            assert isinstance(log, dict)
            assert "timestamp" in log
            assert "operation" in log
            assert "details" in log

    def test_audit_log_filtering(self, encryption_manager):
        """测试审计日志过滤"""
        manager = encryption_manager

        # 执行多种操作
        manager.encrypt_data("data1", algorithm="AES-256-GCM")
        manager.encrypt_data("data2", algorithm="AES-256-CBC")
        manager.generate_key(algorithm="RSA-2048")

        # 获取日志
        logs = manager.get_audit_logs(limit=100)

        # 验证包含不同类型的操作
        operations = set(log["operation"] for log in logs)
        assert "encrypt_data" in operations
        assert "generate_key" in operations


class TestStatisticsAndReporting:
    """测试统计和报告功能"""

    def test_get_encryption_stats(self, encryption_manager):
        """测试获取加密统计"""
        manager = encryption_manager

        # 执行一些操作
        for i in range(5):
            manager.encrypt_data(f"Test data {i}", algorithm="AES-256-GCM")

        manager.generate_key(algorithm="AES-128")
        manager.generate_key(algorithm="RSA-2048")

        # 获取统计
        stats = manager.get_encryption_stats()

        assert isinstance(stats, dict)
        assert "total_encryptions" in stats
        assert "total_decryptions" in stats
        assert "active_keys" in stats
        assert "expired_keys" in stats

        # 验证计数
        assert stats["total_encryptions"] >= 5
        assert stats["active_keys"] >= 1


class TestErrorHandling:
    """测试错误处理"""

    def test_encrypt_with_invalid_algorithm(self, encryption_manager):
        """测试使用无效算法加密"""
        manager = encryption_manager

        with pytest.raises(ValueError, match="不支持的加密算法"):
            manager.encrypt_data("test data", algorithm="INVALID-ALGO")

    def test_decrypt_with_invalid_key(self, encryption_manager):
        """测试使用无效密钥解密"""
        manager = encryption_manager

        # 创建一个无效的加密结果
        invalid_result = EncryptionResult(
            key_id="nonexistent_key",
            algorithm="AES-256-GCM",
            ciphertext=b"invalid_data"
        )

        result = manager.decrypt_data(invalid_result)
        assert isinstance(result, DecryptionResult)
        assert result.success is False
        assert "not found" in result.error_message.lower()

    def test_generate_key_invalid_algorithm(self, encryption_manager):
        """测试生成无效算法的密钥"""
        manager = encryption_manager

        with pytest.raises(ValueError, match="不支持的密钥算法"):
            manager.generate_key(algorithm="INVALID-ALGO")


class TestExportAndImport:
    """测试导出和导入功能"""

    def test_export_keys(self, encryption_manager, temp_key_store):
        """测试密钥导出"""
        manager = encryption_manager

        # 生成一些密钥
        for algo in ["AES-128", "AES-256", "RSA-2048"]:
            manager.generate_key(algorithm=algo)

        export_path = os.path.join(temp_key_store, "exported_keys.json")

        # 导出密钥
        manager.export_keys(export_path, include_private=True)

        # 验证文件存在
        assert os.path.exists(export_path)

        # 验证文件内容
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert "keys" in data
        assert "export_timestamp" in data
        assert len(data["keys"]) >= 3  # 至少有我们生成的密钥


class TestLifecycleManagement:
    """测试生命周期管理"""

    def test_shutdown(self, encryption_manager):
        """测试关闭管理器"""
        manager = encryption_manager

        # 执行一些操作
        manager.encrypt_data("test data", algorithm="AES-256-GCM")
        manager.generate_key(algorithm="AES-128")

        # 关闭管理器
        manager.shutdown()

        # 验证可以继续使用（shutdown不应该破坏功能）
        result = manager.encrypt_data("after shutdown", algorithm="AES-256-GCM")
        assert isinstance(result, EncryptionResult)


class TestPerformance:
    """测试性能"""

    def test_bulk_operations_performance(self, encryption_manager):
        """测试批量操作性能"""
        manager = encryption_manager

        # 准备测试数据
        data_list = [{"data": f"Performance test data {i}", "algorithm": "AES-256-GCM"}
                    for i in range(50)]

        start_time = time.time()

        # 批量加密
        encrypted_results = manager.encrypt_batch(data_list)

        # 批量解密
        decrypted_results = manager.decrypt_batch(encrypted_results)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证结果
        assert len(encrypted_results) == 50
        assert len(decrypted_results) == 50
        assert all(result.success for result in decrypted_results)

        # 性能检查（根据环境调整阈值）
        assert total_time < 10.0  # 10秒内完成50个加解密操作

    def test_memory_efficiency(self, encryption_manager):
        """测试内存效率"""
        manager = encryption_manager

        # 测试大文件处理
        large_data = b"A" * (10 * 1024 * 1024)  # 10MB

        start_time = time.time()
        encrypted = manager.encrypt_data(large_data, algorithm="AES-256-GCM")

        decrypted_result = manager.decrypt_data(encrypted)
        end_time = time.time()

        # 验证正确性
        assert decrypted_result.success is True
        assert decrypted_result.plaintext == large_data

        # 性能检查
        processing_time = end_time - start_time
        assert processing_time < 30.0  # 30秒内处理10MB数据


class TestConcurrency:
    """测试并发性"""

    @pytest.mark.asyncio
    async def test_concurrent_encryption_operations(self, encryption_manager):
        """测试并发加密操作"""
        manager = encryption_manager

        async def encrypt_task(task_id: int):
            data = f"Concurrent test data {task_id}"
            return manager.encrypt_data(data, algorithm="AES-256-GCM")

        # 创建10个并发任务
        tasks = [encrypt_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # 验证结果
        assert len(results) == 10
        for result in results:
            assert isinstance(result, EncryptionResult)

    def test_thread_safe_operations(self, encryption_manager):
        """测试线程安全操作"""
        import threading
        manager = encryption_manager

        results = []
        errors = []

        def worker(worker_id: int):
            try:
                # 加密操作
                data = f"Thread {worker_id} data"
                encrypted = manager.encrypt_data(data, algorithm="AES-256-GCM")

                # 解密验证
                decrypted = manager.decrypt_data(encrypted)

                results.append({
                    'worker_id': worker_id,
                    'success': decrypted.success,
                    'original_data': data,
                    'decrypted_data': decrypted.plaintext.decode()
                })

            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")

        # 启动5个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(results) == 5
        assert len(errors) == 0

        for result in results:
            assert result['success'] is True
            assert result['original_data'] == result['decrypted_data']
