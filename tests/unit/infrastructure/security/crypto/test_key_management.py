"""
测试密钥管理组件
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.infrastructure.security.crypto.key_management import (
    KeyMetadata,
    KeyStore,
    KeyGenerator,
    KeyRotationPolicy,
    KeyManager
)


class TestKeyMetadata:
    """测试密钥元数据"""

    def test_key_metadata_creation(self):
        """测试密钥元数据创建"""
        created_at = datetime.now()
        expires_at = created_at + timedelta(days=30)

        metadata = KeyMetadata(
            key_id="test_key_123",
            algorithm="AES-256",
            created_at=created_at,
            expires_at=expires_at,
            status="active",
            usage_count=0,
            max_usage=1000
        )

        assert metadata.key_id == "test_key_123"
        assert metadata.algorithm == "AES-256"
        assert metadata.status == "active"
        assert metadata.usage_count == 0
        assert metadata.max_usage == 1000

    def test_key_metadata_to_dict_from_dict(self):
        """测试密钥元数据的序列化"""
        original = KeyMetadata(
            key_id="test_key",
            algorithm="AES-256",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=1),
            status="active",
            usage_count=5,
            max_usage=100
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = KeyMetadata.from_dict(data)

        assert restored.key_id == original.key_id
        assert restored.algorithm == original.algorithm
        assert restored.status == original.status
        assert restored.usage_count == original.usage_count
        assert restored.max_usage == original.max_usage

    def test_key_metadata_is_expired(self):
        """测试过期检查"""
        # Not expired
        future_date = datetime.now() + timedelta(days=1)
        metadata = KeyMetadata("key1", "AES-256", datetime.now(), future_date)
        assert not metadata.is_expired()

        # Expired
        past_date = datetime.now() - timedelta(days=1)
        metadata = KeyMetadata("key2", "AES-256", datetime.now(), past_date)
        assert metadata.is_expired()

        # No expiration
        metadata = KeyMetadata("key3", "AES-256", datetime.now())
        assert not metadata.is_expired()

    def test_key_metadata_is_usable(self):
        """测试可用性检查"""
        # Active and not expired
        metadata = KeyMetadata("key1", "AES-256", datetime.now(),
                              datetime.now() + timedelta(days=1))
        assert metadata.is_usable()

        # Expired
        metadata = KeyMetadata("key2", "AES-256", datetime.now(),
                              datetime.now() - timedelta(days=1))
        assert not metadata.is_usable()

        # Revoked
        metadata = KeyMetadata("key3", "AES-256", datetime.now(),
                              datetime.now() + timedelta(days=1), status="revoked")
        assert not metadata.is_usable()

        # Exceeded usage limit
        metadata = KeyMetadata("key4", "AES-256", datetime.now(),
                              datetime.now() + timedelta(days=1),
                              usage_count=100, max_usage=100)
        assert not metadata.is_usable()

    def test_record_usage(self):
        """测试记录使用"""
        metadata = KeyMetadata("key1", "AES-256", datetime.now(), usage_count=5)
        metadata.record_usage()
        assert metadata.usage_count == 6


class TestKeyStore:
    """测试密钥存储"""

    def test_key_store_initialization(self, tmp_path):
        """测试密钥存储初始化"""
        store = KeyStore(str(tmp_path / "keys"))
        assert isinstance(store.keys, dict)
        assert isinstance(store.metadata, dict)

    def test_key_store_store_and_get_key(self, tmp_path):
        """测试存储和获取密钥"""
        store = KeyStore(str(tmp_path / "keys"))

        key_id = "test_key"
        key_data = b"secret_key_data"
        metadata = KeyMetadata(key_id, "AES-256", datetime.now())

        # Store key
        store.store_key(key_id, key_data, metadata)

        # Get key
        retrieved = store.get_key(key_id)
        assert retrieved == key_data

    def test_key_store_get_expired_key(self, tmp_path):
        """测试获取过期密钥"""
        store = KeyStore(str(tmp_path / "keys"))

        key_id = "expired_key"
        key_data = b"secret_key_data"
        expired_time = datetime.now() - timedelta(days=1)
        metadata = KeyMetadata(key_id, "AES-256", datetime.now(), expired_time)

        store.store_key(key_id, key_data, metadata)

        # Should not return expired key
        retrieved = store.get_key(key_id)
        assert retrieved is None

    def test_key_store_revoke_key(self, tmp_path):
        """测试撤销密钥"""
        store = KeyStore(str(tmp_path / "keys"))

        key_id = "test_key"
        key_data = b"secret_key_data"
        metadata = KeyMetadata(key_id, "AES-256", datetime.now())

        store.store_key(key_id, key_data, metadata)
        store.revoke_key(key_id)

        # Should not return revoked key
        retrieved = store.get_key(key_id)
        assert retrieved is None

        # Check status
        assert store.metadata[key_id].status == "revoked"

    def test_key_store_list_keys(self, tmp_path):
        """测试列出密钥"""
        store = KeyStore(str(tmp_path / "keys"))

        # Store multiple keys
        keys_data = [
            ("key1", b"data1", "active"),
            ("key2", b"data2", "active"),
            ("key3", b"data3", "revoked")
        ]

        for key_id, key_data, status in keys_data:
            metadata = KeyMetadata(key_id, "AES-256", datetime.now(), status=status)
            store.store_key(key_id, key_data, metadata)

        # List all keys
        all_keys = store.list_keys()
        assert len(all_keys) == 3

        # List active keys only
        active_keys = store.list_keys("active")
        assert len(active_keys) == 2
        assert all(k.status == "active" for k in active_keys)

    def test_key_store_cleanup_expired_keys(self, tmp_path):
        """测试清理过期密钥"""
        store = KeyStore(str(tmp_path / "keys"))

        # Store expired and active keys
        expired_metadata = KeyMetadata("expired", "AES-256", datetime.now(),
                                      datetime.now() - timedelta(days=1))
        active_metadata = KeyMetadata("active", "AES-256", datetime.now(),
                                     datetime.now() + timedelta(days=1))

        store.store_key("expired", b"expired_data", expired_metadata)
        store.store_key("active", b"active_data", active_metadata)

        # Cleanup
        store.cleanup_expired_keys()

        # Expired key should be removed
        assert "expired" not in store.keys
        assert "expired" not in store.metadata

        # Active key should remain
        assert "active" in store.keys
        assert "active" in store.metadata


class TestKeyGenerator:
    """测试密钥生成器"""

    def test_generate_aes_key(self):
        """测试生成AES密钥"""
        # 128-bit key
        key = KeyGenerator.generate_aes_key(128)
        assert len(key) == 16  # 128 bits = 16 bytes

        # 256-bit key
        key = KeyGenerator.generate_aes_key(256)
        assert len(key) == 32  # 256 bits = 32 bytes

    def test_generate_aes_key_invalid_size(self):
        """测试生成无效大小的AES密钥"""
        with pytest.raises(ValueError):
            KeyGenerator.generate_aes_key(64)  # Invalid size

    def test_generate_rsa_keypair(self):
        """测试生成RSA密钥对"""
        private_pem, public_pem = KeyGenerator.generate_rsa_keypair(2048)

        assert isinstance(private_pem, bytes)
        assert isinstance(public_pem, bytes)
        assert b"PRIVATE KEY" in private_pem
        assert b"PUBLIC KEY" in public_pem

    def test_generate_random_key(self):
        """测试生成随机密钥"""
        key = KeyGenerator.generate_random_key(32)
        assert len(key) == 32
        assert isinstance(key, bytes)

        # Different calls should produce different keys
        key2 = KeyGenerator.generate_random_key(32)
        assert key != key2

    def test_derive_key(self):
        """测试密钥派生"""
        password = "test_password"
        salt = b"test_salt"

        key1 = KeyGenerator.derive_key(password, salt)
        key2 = KeyGenerator.derive_key(password, salt)

        # Same inputs should produce same key
        assert key1 == key2
        assert len(key1) == 32  # Default length


class TestKeyRotationPolicy:
    """测试密钥轮换策略"""

    def test_should_rotate_age_based(self):
        """测试基于年龄的轮换"""
        policy = KeyRotationPolicy(max_age_days=30, auto_rotate=True)

        # Recent key
        recent_metadata = KeyMetadata("key1", "AES-256", datetime.now())
        assert not policy.should_rotate(recent_metadata)

        # Old key
        old_metadata = KeyMetadata("key2", "AES-256",
                                  datetime.now() - timedelta(days=60))
        assert policy.should_rotate(old_metadata)

    def test_should_rotate_usage_based(self):
        """测试基于使用次数的轮换"""
        policy = KeyRotationPolicy(max_usage_count=100, auto_rotate=True)

        # Low usage key
        low_usage = KeyMetadata("key1", "AES-256", datetime.now(),
                               usage_count=50, max_usage=100)
        assert not policy.should_rotate(low_usage)

        # High usage key
        high_usage = KeyMetadata("key2", "AES-256", datetime.now(),
                                usage_count=100, max_usage=100)
        assert policy.should_rotate(high_usage)

    def test_should_rotate_disabled(self):
        """测试禁用轮换"""
        policy = KeyRotationPolicy(auto_rotate=False)

        old_metadata = KeyMetadata("key1", "AES-256",
                                  datetime.now() - timedelta(days=100))
        assert not policy.should_rotate(old_metadata)


class TestKeyManager:
    """测试密钥管理器"""

    def test_key_manager_initialization(self, tmp_path):
        """测试密钥管理器初始化"""
        manager = KeyManager(str(tmp_path / "key_manager"))
        assert hasattr(manager, 'key_store')
        assert hasattr(manager, 'key_generator')
        assert hasattr(manager, 'rotation_policy')

    def test_create_aes_key(self, tmp_path):
        """测试创建AES密钥"""
        manager = KeyManager(str(tmp_path / "key_manager"))

        key_id = manager.create_aes_key(key_size=256, expires_in_days=30)

        # Check key was created
        key_data = manager.get_key(key_id)
        assert key_data is not None
        assert len(key_data) == 32  # 256-bit key

        # Check metadata
        keys = manager.list_active_keys()
        assert len(keys) == 1
        assert keys[0].key_id == key_id
        assert keys[0].algorithm == "AES-256"
        assert keys[0].expires_at is not None

    def test_create_rsa_keypair(self, tmp_path):
        """测试创建RSA密钥对"""
        manager = KeyManager(str(tmp_path / "key_manager"))

        key_id = manager.create_rsa_keypair(key_size=2048, expires_in_days=30)

        # Check keys were created
        private_key = manager.get_key(f"{key_id}_private")
        public_key = manager.get_key(f"{key_id}_public")

        assert private_key is not None
        assert public_key is not None
        assert b"PRIVATE KEY" in private_key
        assert b"PUBLIC KEY" in public_key

    def test_revoke_key(self, tmp_path):
        """测试撤销密钥"""
        manager = KeyManager(str(tmp_path / "key_manager"))

        key_id = manager.create_aes_key()

        # Revoke key
        manager.revoke_key(key_id)

        # Should not be able to get revoked key
        key_data = manager.get_key(key_id)
        assert key_data is None

    def test_rotate_key(self, tmp_path):
        """测试轮换密钥"""
        manager = KeyManager(str(tmp_path / "key_manager"))

        old_key_id = manager.create_aes_key()

        # Rotate key
        new_key_id = manager.rotate_key(old_key_id)

        assert new_key_id is not None
        assert new_key_id != old_key_id

        # Old key should be revoked
        old_key = manager.get_key(old_key_id)
        assert old_key is None

        # New key should be available
        new_key = manager.get_key(new_key_id)
        assert new_key is not None

    def test_list_active_keys(self, tmp_path):
        """测试列出活跃密钥"""
        manager = KeyManager(str(tmp_path / "key_manager"))

        # Create multiple keys
        key1 = manager.create_aes_key()
        key2 = manager.create_aes_key()

        active_keys = manager.list_active_keys()
        assert len(active_keys) >= 1  # Should have at least 1 active key

        # Check they are all active
        for key_meta in active_keys:
            assert key_meta.status == "active"

    def test_cleanup_expired_keys(self, tmp_path):
        """测试清理过期密钥"""
        manager = KeyManager(str(tmp_path / "key_manager"))

        # Create a key that expires immediately
        key_id = manager.create_aes_key(expires_in_days=-1)

        # Cleanup
        manager.cleanup_expired_keys()

        # Expired key should be removed
        key_data = manager.get_key(key_id)
        assert key_data is None

    def test_check_key_rotation_needed(self, tmp_path):
        """测试检查需要轮换的密钥"""
        manager = KeyManager(str(tmp_path / "key_manager"))

        # Create a key that needs rotation (old)
        old_key_id = manager.create_aes_key(expires_in_days=-1)  # Already expired

        keys_to_rotate = manager.check_key_rotation_needed()
        assert old_key_id in keys_to_rotate
