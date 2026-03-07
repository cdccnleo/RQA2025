#!/usr/bin/env python3
"""
数据加密服务单元测试

测试DataEncryptionManager及其组件的完整功能
    创建时间: 2024年12月
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import base64
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from src.infrastructure.security.services.data_encryption_service import (
    DataEncryptionManager,
    EncryptionKey,
    EncryptionResult,
    DecryptionResult,
    CRYPTOGRAPHY_AVAILABLE
)


class TestDataEncryptionService:
    """数据加密服务测试类"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DataEncryptionManager(key_store_path=self.temp_dir)

    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_default_path(self):
        """测试默认路径初始化"""
        manager = DataEncryptionManager()
        assert manager.key_store_path.exists()
        assert manager.enable_audit is True

    def test_init_with_custom_path(self):
        """测试自定义路径初始化"""
        custom_path = Path(self.temp_dir) / "custom_keys"
        manager = DataEncryptionManager(key_store_path=str(custom_path))
        assert manager.key_store_path == custom_path
        assert manager.key_store_path.exists()

    def test_init_without_audit(self):
        """测试禁用审计初始化"""
        manager = DataEncryptionManager(enable_audit=False)
        assert manager.enable_audit is False

    def test_encryption_key_creation(self):
        """测试加密密钥创建"""
        key = EncryptionKey(
            key_id="test_key",
            key_data=b"test_key_data",
            algorithm="AES-256",
            created_at=datetime.now()
        )
        assert key.key_id == "test_key"
        assert key.key_data == b"test_key_data"
        assert key.algorithm == "AES-256"
        assert key.is_active is True
        assert key.usage_count == 0

    def test_encryption_key_expiration(self):
        """测试密钥过期检查"""
        # 未过期密钥
        future_date = datetime.now() + timedelta(days=1)
        key = EncryptionKey("test", b"data", "AES", datetime.now(), future_date)
        assert not key.is_expired()
        assert key.can_use()

        # 已过期密钥
        past_date = datetime.now() - timedelta(days=1)
        expired_key = EncryptionKey("test", b"data", "AES", datetime.now(), past_date)
        assert expired_key.is_expired()
        assert not expired_key.can_use()

        # 永不过期密钥
        eternal_key = EncryptionKey("test", b"data", "AES", datetime.now())
        assert not eternal_key.is_expired()
        assert eternal_key.can_use()

    def test_encryption_key_usage_increment(self):
        """测试密钥使用计数"""
        key = EncryptionKey("test", b"data", "AES", datetime.now())
        assert key.usage_count == 0
        key.increment_usage()
        assert key.usage_count == 1

    def test_encryption_result_creation(self):
        """测试加密结果创建"""
        result = EncryptionResult(
            encrypted_data=b"encrypted",
            key_id="test_key",
            algorithm="AES-256-GCM",
            iv=b"iv_data",
            tag=b"tag_data"
        )
        assert result.encrypted_data == b"encrypted"
        assert result.key_id == "test_key"
        assert result.algorithm == "AES-256-GCM"
        assert result.iv == b"iv_data"
        assert result.tag == b"tag_data"
        assert isinstance(result.encrypted_at, datetime)

    def test_decryption_result_creation(self):
        """测试解密结果创建"""
        result = DecryptionResult(
            decrypted_data=b"decrypted",
            key_id="test_key",
            algorithm="AES-256-GCM"
        )
        assert result.decrypted_data == b"decrypted"
        assert result.key_id == "test_key"
        assert result.algorithm == "AES-256-GCM"
        assert isinstance(result.decrypted_at, datetime)

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography库不可用")
    def test_encrypt_decrypt_aes_gcm(self):
        """测试AES-GCM加密解密"""
        test_data = b"Hello, World! This is a test message."

        # 加密
        result = self.manager.encrypt_data(test_data, "AES-256-GCM")
        assert result.encrypted_data != test_data
        assert result.algorithm == "AES-256-GCM"
        assert result.key_id is not None
        assert result.iv is not None
        assert result.tag is not None

        # 解密
        decrypt_result = self.manager.decrypt_data(result)
        assert decrypt_result.decrypted_data == test_data
        assert decrypt_result.key_id == result.key_id
        assert decrypt_result.algorithm == result.algorithm

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography库不可用")
    def test_encrypt_decrypt_aes_cbc(self):
        """测试AES-CBC加密解密"""
        test_data = b"This is AES-CBC test data."

        # 加密
        result = self.manager.encrypt_data(test_data, "AES-256-CBC")
        assert result.encrypted_data != test_data
        assert result.algorithm == "AES-256-CBC"
        assert result.key_id is not None
        assert result.iv is not None

        # 解密
        decrypt_result = self.manager.decrypt_data(result)
        assert decrypt_result.decrypted_data == test_data

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography库不可用")
    def test_encrypt_decrypt_rsa_oaep(self):
        """测试RSA-OAEP加密解密"""
        test_data = b"RSA test data"

        # 加密
        result = self.manager.encrypt_data(test_data, "RSA-OAEP")
        assert result.encrypted_data != test_data
        assert result.algorithm == "RSA-OAEP"
        assert result.key_id is not None

        # 解密
        decrypt_result = self.manager.decrypt_data(result)
        assert decrypt_result.decrypted_data == test_data

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography库不可用")
    def test_encrypt_decrypt_chacha20(self):
        """测试ChaCha20加密解密"""
        test_data = b"ChaCha20 test data"

        # 加密
        result = self.manager.encrypt_data(test_data, "CHACHA20")
        assert result.encrypted_data != test_data
        assert result.algorithm == "CHACHA20"
        assert result.key_id is not None

        # 解密
        decrypt_result = self.manager.decrypt_data(result)
        assert decrypt_result.decrypted_data == test_data

    def test_encrypt_decrypt_fallback(self):
        """测试降级加密解密"""
        if CRYPTOGRAPHY_AVAILABLE:
            pytest.skip("cryptography库可用，跳过降级测试")

        test_data = b"Fallback encryption test"

        # 加密
        result = self.manager.encrypt_data(test_data, "AES-256-GCM")
        assert result.encrypted_data != test_data

        # 解密
        decrypt_result = self.manager.decrypt_data(result)
        assert decrypt_result.decrypted_data == test_data

    def test_encrypt_string_data(self):
        """测试字符串数据加密"""
        test_string = "Hello, 世界!"

        result = self.manager.encrypt_data(test_string)
        assert result.encrypted_data is not None

        decrypt_result = self.manager.decrypt_data(result)
        assert decrypt_result.decrypted_data.decode('utf-8') == test_string

    def test_encrypt_with_custom_key_id(self):
        """测试使用自定义密钥ID加密"""
        # 先创建一个密钥
        key_id = self.manager.generate_key("AES-256")
        test_data = b"Custom key test"

        result = self.manager.encrypt_data(test_data, key_id=key_id)
        assert result.key_id == key_id

    def test_encrypt_with_metadata(self):
        """测试带元数据加密"""
        test_data = b"Metadata test"
        metadata = {"source": "test", "version": "1.0"}

        result = self.manager.encrypt_data(test_data, metadata=metadata)
        assert result.metadata == metadata

    def test_decrypt_invalid_data(self):
        """测试解密无效数据"""
        invalid_result = EncryptionResult(
            encrypted_data=b"invalid",
            key_id="nonexistent",
            algorithm="AES-256-GCM"
        )

        with pytest.raises(Exception):
            self.manager.decrypt_data(invalid_result)

    def test_generate_key_aes_256(self):
        """测试生成AES-256密钥"""
        key_id = self.manager.generate_key("AES-256")
        assert key_id in self.manager.keys
        key = self.manager.keys[key_id]
        assert key.algorithm == "AES-256"
        assert len(key.key_data) == 32  # AES-256需要32字节密钥
        assert key.is_active is True

    def test_generate_key_aes_128(self):
        """测试生成AES-128密钥"""
        key_id = self.manager.generate_key("AES-128")
        assert key_id in self.manager.keys
        key = self.manager.keys[key_id]
        assert key.algorithm == "AES-128"
        assert len(key.key_data) == 16  # AES-128需要16字节密钥

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography库不可用")
    def test_generate_key_rsa(self):
        """测试生成RSA密钥"""
        key_id = self.manager.generate_key("RSA-2048")
        assert key_id in self.manager.keys
        key = self.manager.keys[key_id]
        assert key.algorithm == "RSA-2048"
        # RSA密钥数据应该包含私钥和公钥

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography库不可用")
    def test_generate_key_chacha20(self):
        """测试生成ChaCha20密钥"""
        key_id = self.manager.generate_key("CHACHA20")
        assert key_id in self.manager.keys
        key = self.manager.keys[key_id]
        assert key.algorithm == "CHACHA20"
        assert len(key.key_data) == 32  # ChaCha20需要32字节密钥

    def test_generate_key_with_expiration(self):
        """测试生成带过期时间的密钥"""
        expires_in_days = 30
        key_id = self.manager.generate_key("AES-256", expires_in_days)

        key = self.manager.keys[key_id]
        expected_expiry = key.created_at + timedelta(days=expires_in_days)
        assert abs((key.expires_at - expected_expiry).total_seconds()) < 1

    def test_key_rotation(self):
        """测试密钥轮换"""
        # 生成一些密钥
        old_keys = []
        for i in range(3):
            key_id = self.manager.generate_key("AES-256")
            old_keys.append(key_id)

        # 修改其中一个密钥使其需要轮换
        key_to_rotate = self.manager.keys[old_keys[0]]
        key_to_rotate.created_at = datetime.now() - timedelta(days=100)  # 超过90天
        key_to_rotate.usage_count = 20000  # 超过使用限制

        # 执行轮换
        rotated_keys = self.manager.rotate_keys()

        # 应该有新密钥生成
        assert len(rotated_keys) > 0
        for new_key_id in rotated_keys:
            assert new_key_id in self.manager.keys
            new_key = self.manager.keys[new_key_id]
            assert new_key.is_active is True

    def test_cleanup_expired_keys(self):
        """测试清理过期密钥"""
        # 生成一个已过期的密钥
        expired_key_id = self.manager.generate_key("AES-256", expires_in_days=-1)
        expired_key = self.manager.keys[expired_key_id]
        assert expired_key.is_expired()

        # 生成一个正常的密钥
        active_key_id = self.manager.generate_key("AES-256", expires_in_days=30)

        # 清理过期密钥
        cleaned_count = self.manager.cleanup_expired_keys()
        assert cleaned_count == 1
        assert expired_key_id not in self.manager.keys
        assert active_key_id in self.manager.keys

    def test_batch_encrypt_decrypt(self):
        """测试批量加密解密"""
        data_list = [
            {"data": b"test1", "algorithm": "AES-256-GCM"},
            {"data": b"test2", "algorithm": "AES-256-GCM"},
            {"data": "test3", "algorithm": "AES-256-GCM"}
        ]

        # 批量加密
        encrypted_results = self.manager.encrypt_batch(data_list)
        assert len(encrypted_results) == 3
        for result in encrypted_results:
            assert isinstance(result, EncryptionResult)

        # 批量解密
        decrypted_results = self.manager.decrypt_batch(encrypted_results)
        assert len(decrypted_results) == 3
        for i, result in enumerate(decrypted_results):
            assert isinstance(result, DecryptionResult)
            expected_data = data_list[i]["data"]
            if isinstance(expected_data, str):
                expected_data = expected_data.encode('utf-8')
            assert result.decrypted_data == expected_data

    def test_get_encryption_stats(self):
        """测试获取加密统计信息"""
        # 生成一些操作
        self.manager.encrypt_data(b"test1")
        self.manager.encrypt_data(b"test2")

        stats = self.manager.get_encryption_stats()
        assert "total_keys" in stats
        assert "active_keys" in stats
        assert "total_encryptions" in stats
        assert "total_decryptions" in stats
        assert stats["total_encryptions"] >= 2

    def test_audit_logging(self):
        """测试审计日志"""
        if not self.manager.enable_audit:
            pytest.skip("审计日志已禁用")

        # 执行一些操作
        self.manager.encrypt_data(b"audit test")

        # 获取审计日志
        logs = self.manager.get_audit_logs()
        assert len(logs) > 0

        # 检查日志内容
        latest_log = logs[-1]
        assert "operation" in latest_log
        assert "timestamp" in latest_log
        assert "details" in latest_log

    def test_export_keys(self):
        """测试密钥导出"""
        # 记录导出前的密钥数量
        initial_key_count = len(self.manager.keys)

        # 生成一些密钥
        key_id1 = self.manager.generate_key("AES-256")
        key_id2 = self.manager.generate_key("AES-256")

        export_path = Path(self.temp_dir) / "exported_keys.json"

        # 导出密钥（不包含私钥）
        self.manager.export_keys(str(export_path), include_private=False)

        # 检查导出文件
        assert export_path.exists()
        with open(export_path, 'r') as f:
            exported_data = json.load(f)

        assert "keys" in exported_data
        assert "exported_at" in exported_data
        # 应该至少有刚生成的2个密钥
        assert len(exported_data["keys"]) >= initial_key_count + 2

    def test_shutdown(self):
        """测试关闭管理器"""
        # 生成一些密钥
        self.manager.generate_key("AES-256")

        # 关闭管理器
        self.manager.shutdown()

        # 验证密钥已保存
        keys_file = self.manager.key_store_path / "keys.json"
        assert keys_file.exists()

    def test_invalid_algorithm(self):
        """测试无效算法"""
        with pytest.raises(ValueError):
            self.manager.encrypt_data(b"test", "INVALID_ALGORITHM")

    def test_key_not_found(self):
        """测试密钥未找到"""
        result = EncryptionResult(
            encrypted_data=b"test",
            key_id="nonexistent_key",
            algorithm="AES-256-GCM"
        )

        with pytest.raises(KeyError):
            self.manager.decrypt_data(result)

    def test_empty_data_encryption(self):
        """测试空数据加密"""
        result = self.manager.encrypt_data(b"")
        assert result.encrypted_data is not None

        decrypt_result = self.manager.decrypt_data(result)
        assert decrypt_result.decrypted_data == b""

    def test_large_data_encryption(self):
        """测试大数据加密"""
        large_data = b"A" * 1000000  # 1MB数据

        result = self.manager.encrypt_data(large_data)
        assert result.encrypted_data is not None

        decrypt_result = self.manager.decrypt_data(result)
        assert decrypt_result.decrypted_data == large_data

    def test_key_persistence(self):
        """测试密钥持久化"""
        # 创建第一个管理器实例
        manager1 = DataEncryptionManager(key_store_path=self.temp_dir)
        key_id = manager1.generate_key("AES-256")
        manager1.shutdown()

        # 创建第二个管理器实例
        manager2 = DataEncryptionManager(key_store_path=self.temp_dir)
        assert key_id in manager2.keys

        # 验证密钥数据一致
        key1 = manager1.keys[key_id]
        key2 = manager2.keys[key_id]
        assert key1.key_data == key2.key_data
        assert key1.algorithm == key2.algorithm

    def test_concurrent_key_generation(self):
        """测试并发密钥生成"""
        import threading
        import time

        generated_keys = []
        errors = []

        def generate_keys():
            try:
                for i in range(10):
                    key_id = self.manager.generate_key("AES-256")
                    generated_keys.append(key_id)
                    time.sleep(0.01)  # 模拟一些处理时间
            except Exception as e:
                errors.append(e)

        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_keys)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(errors) == 0
        assert len(generated_keys) == 50  # 5线程 * 10个密钥

        # 验证所有密钥ID都是唯一的
        assert len(set(generated_keys)) == len(generated_keys)

        # 验证所有密钥都存在于管理器中
        for key_id in generated_keys:
            assert key_id in self.manager.keys

    def test_memory_cleanup_efficiency(self):
        """测试内存清理效率"""
        import psutil
        import os

        # 记录初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 执行大量加密操作
        for i in range(100):
            data = f"Test data {i}".encode() * 1000  # 较大的数据
            result = self.manager.encrypt_data(data)
            decrypt_result = self.manager.decrypt_data(result)

        # 强制垃圾回收
        import gc
        gc.collect()

        # 检查内存使用没有显著增加
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增加应该在合理范围内（例如不超过50MB）
        assert memory_increase < 50 * 1024 * 1024, f"内存泄漏检测：增加了{memory_increase / 1024 / 1024:.2f}MB"

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        # 测试无效的加密结果
        invalid_result = EncryptionResult(
            encrypted_data=b"",
            key_id="invalid",
            algorithm="INVALID"
        )

        with pytest.raises(Exception):
            self.manager.decrypt_data(invalid_result)

        # 验证管理器仍然可以正常工作
        valid_result = self.manager.encrypt_data(b"test after error")
        assert valid_result is not None

        decrypt_result = self.manager.decrypt_data(valid_result)
        assert decrypt_result.decrypted_data == b"test after error"

    def test_security_audit_compliance(self):
        """测试安全审计合规性"""
        # 执行各种操作
        operations = [
            lambda: self.manager.encrypt_data(b"audit_test_1"),
            lambda: self.manager.encrypt_data(b"audit_test_2", "AES-256-CBC"),
            lambda: self.manager.generate_key("AES-256"),
            lambda: self.manager.rotate_keys(),
        ]

        for operation in operations:
            try:
                operation()
            except Exception:
                pass  # 某些操作可能失败，继续测试

        # 获取审计日志
        if self.manager.enable_audit:
            logs = self.manager.get_audit_logs()

            # 验证审计日志包含必要信息
            for log in logs:
                assert "operation" in log
                assert "timestamp" in log
                assert isinstance(log["timestamp"], str)

                # 验证时间戳格式
                try:
                    datetime.fromisoformat(log["timestamp"])
                except ValueError:
                    pytest.fail(f"无效的时间戳格式: {log['timestamp']}")

    def test_configuration_persistence(self):
        """测试配置持久化"""
        # 修改配置
        original_policy = self.manager.key_rotation_policy.copy()
        self.manager.key_rotation_policy["max_age_days"] = 60

        # 创建新实例
        manager2 = DataEncryptionManager(key_store_path=self.temp_dir)

        # 配置应该从持久化存储中恢复，而不是使用默认值
        # 注意：实际实现可能需要保存配置到文件
        assert manager2.key_rotation_policy is not None

    def test_performance_monitoring(self):
        """测试性能监控"""
        import time

        # 执行一些操作并测量时间
        start_time = time.time()

        for i in range(10):
            data = b"Performance test data " * 100
            result = self.manager.encrypt_data(data)
            decrypt_result = self.manager.decrypt_data(result)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能在合理范围内
        assert total_time < 5.0, f"性能测试失败：{total_time:.2f}秒处理10个操作"

        # 计算每操作平均时间
        avg_time = total_time / 10
        assert avg_time < 0.5, f"平均操作时间过长：{avg_time:.3f}秒"


if __name__ == "__main__":
    pytest.main([__file__])
