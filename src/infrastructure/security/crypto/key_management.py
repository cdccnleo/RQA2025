"""
密钥管理组件
负责密钥的生成、存储、轮换和管理
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from .algorithms import HashAlgorithm, EncryptionAlgorithmFactory


class KeyMetadata:
    """密钥元数据"""

    def __init__(self,
                 key_id: str,
                 algorithm: str,
                 created_at: datetime,
                 expires_at: Optional[datetime] = None,
                 status: str = "active",
                 usage_count: int = 0,
                 max_usage: Optional[int] = None):
        self.key_id = key_id
        self.algorithm = algorithm
        self.created_at = created_at
        self.expires_at = expires_at
        self.status = status  # active, expired, revoked
        self.usage_count = usage_count
        self.max_usage = max_usage

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status,
            "usage_count": self.usage_count,
            "max_usage": self.max_usage
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeyMetadata':
        """从字典创建"""
        return cls(
            key_id=data["key_id"],
            algorithm=data["algorithm"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            status=data.get("status", "active"),
            usage_count=data.get("usage_count", 0),
            max_usage=data.get("max_usage")
        )

    def is_expired(self) -> bool:
        """检查密钥是否过期"""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        return False

    def is_usable(self) -> bool:
        """检查密钥是否可用"""
        if self.status != "active":
            return False
        if self.is_expired():
            return False
        if self.max_usage and self.usage_count >= self.max_usage:
            return False
        return True

    def record_usage(self):
        """记录使用"""
        self.usage_count += 1


class KeyStore:
    """密钥存储"""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.keys: Dict[str, bytes] = {}
        self.metadata: Dict[str, KeyMetadata] = {}
        self._load_keys()

    def _load_keys(self):
        """加载密钥"""
        metadata_file = self.storage_path / "key_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)
                for key_id, data in metadata_data.items():
                    self.metadata[key_id] = KeyMetadata.from_dict(data)

        # Load encrypted keys
        keys_file = self.storage_path / "keys.enc"
        if keys_file.exists():
            # In a real implementation, this would be properly encrypted
            # For demo purposes, we'll load as plain text
            with open(keys_file, 'r', encoding='utf-8') as f:
                keys_data = json.load(f)
                # Convert hex strings back to bytes
                self.keys = {key_id: bytes.fromhex(hex_str) for key_id, hex_str in keys_data.items()}

    def _save_keys(self):
        """保存密钥"""
        # Save metadata
        metadata_data = {key_id: meta.to_dict() for key_id, meta in self.metadata.items()}
        metadata_file = self.storage_path / "key_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_data, f, indent=2, ensure_ascii=False)

        # Save keys (convert bytes to base64 strings for JSON serialization)
        keys_data = {key_id: key.hex() for key_id, key in self.keys.items()}
        keys_file = self.storage_path / "keys.enc"
        with open(keys_file, 'w', encoding='utf-8') as f:
            json.dump(keys_data, f, ensure_ascii=False)

    def store_key(self, key_id: str, key_data: bytes, metadata: KeyMetadata):
        """存储密钥"""
        self.keys[key_id] = key_data
        self.metadata[key_id] = metadata
        self._save_keys()

    def get_key(self, key_id: str) -> Optional[bytes]:
        """获取密钥"""
        metadata = self.metadata.get(key_id)
        if metadata and metadata.is_usable():
            metadata.record_usage()
            self._save_keys()  # Update usage count
            return self.keys.get(key_id)
        return None

    def revoke_key(self, key_id: str):
        """撤销密钥"""
        if key_id in self.metadata:
            self.metadata[key_id].status = "revoked"
            self._save_keys()

    def list_keys(self, status_filter: Optional[str] = None) -> List[KeyMetadata]:
        """列出密钥"""
        keys = list(self.metadata.values())
        if status_filter:
            keys = [k for k in keys if k.status == status_filter]
        return keys

    def cleanup_expired_keys(self):
        """清理过期密钥"""
        expired_keys = [key_id for key_id, meta in self.metadata.items()
                       if meta.is_expired() or not meta.is_usable()]
        for key_id in expired_keys:
            if key_id in self.keys:
                del self.keys[key_id]
            if key_id in self.metadata:
                del self.metadata[key_id]
        if expired_keys:
            self._save_keys()


class KeyGenerator:
    """密钥生成器"""

    @staticmethod
    def generate_aes_key(key_size: int = 256) -> bytes:
        """生成AES密钥"""
        if key_size not in [128, 192, 256]:
            raise ValueError("Invalid AES key size. Must be 128, 192, or 256 bits.")
        return os.urandom(key_size // 8)

    @staticmethod
    def generate_rsa_keypair(key_size: int = 2048):
        """生成RSA密钥对"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_pem, public_pem

    @staticmethod
    def generate_random_key(length: int = 32) -> bytes:
        """生成随机密钥"""
        return os.urandom(length)

    @staticmethod
    def derive_key(password: str, salt: bytes, length: int = 32) -> bytes:
        """从密码派生密钥"""
        return HashAlgorithm.pbkdf2(password, salt)


class KeyRotationPolicy:
    """密钥轮换策略"""

    def __init__(self,
                 max_age_days: int = 90,
                 max_usage_count: int = 10000,
                 auto_rotate: bool = True):
        self.max_age_days = max_age_days
        self.max_usage_count = max_usage_count
        self.auto_rotate = auto_rotate

    def should_rotate(self, metadata: KeyMetadata) -> bool:
        """检查是否应该轮换密钥"""
        if not self.auto_rotate:
            return False

        # Check expiration
        if metadata.expires_at and metadata.expires_at < datetime.now():
            return True

        # Check age
        if metadata.created_at + timedelta(days=self.max_age_days) < datetime.now():
            return True

        # Check usage
        if metadata.max_usage and metadata.usage_count >= metadata.max_usage:
            return True

        return False


class KeyManager:
    """密钥管理器"""

    def __init__(self, storage_path: str):
        self.key_store = KeyStore(storage_path)
        self.key_generator = KeyGenerator()
        self.rotation_policy = KeyRotationPolicy()
        self.hash_algo = HashAlgorithm()

    def create_aes_key(self,
                      key_id: Optional[str] = None,
                      key_size: int = 256,
                      expires_in_days: Optional[int] = None) -> str:
        """创建AES密钥"""
        if key_id is None:
            key_id = f"aes_{self.hash_algo.sha256(str(time.time()))[:16]}"

        key_data = self.key_generator.generate_aes_key(key_size)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        metadata = KeyMetadata(
            key_id=key_id,
            algorithm=f"AES-{key_size}",
            created_at=datetime.now(),
            expires_at=expires_at,
            max_usage=self.rotation_policy.max_usage_count
        )

        self.key_store.store_key(key_id, key_data, metadata)
        return key_id

    def create_rsa_keypair(self,
                          key_id: Optional[str] = None,
                          key_size: int = 2048,
                          expires_in_days: Optional[int] = None) -> str:
        """创建RSA密钥对"""
        if key_id is None:
            key_id = f"rsa_{self.hash_algo.sha256(str(time.time()))[:16]}"

        private_pem, public_pem = self.key_generator.generate_rsa_keypair(key_size)

        # Store both keys
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        # Store private key
        private_metadata = KeyMetadata(
            key_id=f"{key_id}_private",
            algorithm=f"RSA-{key_size}",
            created_at=datetime.now(),
            expires_at=expires_at,
            max_usage=self.rotation_policy.max_usage_count
        )
        self.key_store.store_key(f"{key_id}_private", private_pem, private_metadata)

        # Store public key
        public_metadata = KeyMetadata(
            key_id=f"{key_id}_public",
            algorithm=f"RSA-{key_size}",
            created_at=datetime.now(),
            expires_at=expires_at
        )
        self.key_store.store_key(f"{key_id}_public", public_pem, public_metadata)

        return key_id

    def get_key(self, key_id: str) -> Optional[bytes]:
        """获取密钥"""
        return self.key_store.get_key(key_id)

    def revoke_key(self, key_id: str):
        """撤销密钥"""
        self.key_store.revoke_key(key_id)

    def rotate_key(self, old_key_id: str) -> Optional[str]:
        """轮换密钥"""
        # Get old key metadata
        old_metadata = self.key_store.metadata.get(old_key_id)
        if not old_metadata:
            return None

        # Create new key based on algorithm
        if old_metadata.algorithm.startswith("AES"):
            key_size = int(old_metadata.algorithm.split("-")[1])
            # Generate a unique key ID for rotation
            new_key_id = f"{old_key_id}_rotated_{int(time.time())}"
            new_key_id = self.create_aes_key(key_id=new_key_id, key_size=key_size)
        elif old_metadata.algorithm.startswith("RSA"):
            key_size = int(old_metadata.algorithm.split("-")[1])
            # Generate a unique key ID for rotation
            new_key_id = f"{old_key_id}_rotated_{int(time.time())}"
            new_key_id = self.create_rsa_keypair(key_id=new_key_id, key_size=key_size)
        else:
            return None

        # Revoke old key
        self.revoke_key(old_key_id)

        return new_key_id

    def list_active_keys(self) -> List[KeyMetadata]:
        """列出活跃密钥"""
        return self.key_store.list_keys("active")

    def cleanup_expired_keys(self):
        """清理过期密钥"""
        self.key_store.cleanup_expired_keys()

    def check_key_rotation_needed(self) -> List[str]:
        """检查需要轮换的密钥"""
        keys_needing_rotation = []
        for key_id, metadata in self.key_store.metadata.items():
            if self.rotation_policy.should_rotate(metadata):
                keys_needing_rotation.append(key_id)
        return keys_needing_rotation