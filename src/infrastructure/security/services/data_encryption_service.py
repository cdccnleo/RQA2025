#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据加密管理器

实现端到端的数据加密和解密功能
支持多种加密算法和密钥管理
"""

import os
import base64
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("cryptography库不可用，使用降级加密实现")


@dataclass
class EncryptionKey:

    """加密密钥"""
    key_id: str
    key_data: bytes
    algorithm: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查密钥是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def can_use(self) -> bool:
        """检查密钥是否可以使用"""
        return self.is_active and not self.is_expired()

    def increment_usage(self):
        """增加使用计数"""
        self.usage_count += 1


@dataclass
class EncryptionResult:

    """加密结果"""
    encrypted_data: bytes
    key_id: str
    algorithm: str
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    encrypted_at: datetime = field(default_factory=datetime.now)


@dataclass
class DecryptionResult:

    """解密结果"""
    decrypted_data: bytes
    key_id: str
    algorithm: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    decrypted_at: datetime = field(default_factory=datetime.now)


class DataEncryptionManager:

    """
    数据加密管理器

    提供端到端的数据加密和解密功能：
    - 对称加密（AES）
    - 非对称加密（RSA）
    - 密钥管理和轮换
    - 加密策略管理
    - 安全审计日志
    """

    def __init__(self, key_store_path: Optional[str] = None,


                 enable_audit: bool = True):
        """
        初始化加密管理器

        Args:
            key_store_path: 密钥存储路径
            enable_audit: 是否启用审计日志
        """
        self.key_store_path = Path(key_store_path or "data/security/keys")
        self.key_store_path.mkdir(parents=True, exist_ok=True)

        self.enable_audit = enable_audit
        self.audit_log_path = self.key_store_path / "audit.log"

        # 密钥存储
        self.keys: Dict[str, EncryptionKey] = {}
        self.current_key_id: Optional[str] = None

        # 统计信息
        self.total_encryptions = 0
        self.total_decryptions = 0

        # 加密算法配置
        self.algorithms = {
            'AES-256-GCM': self._encrypt_aes_gcm,
            'AES-256-CBC': self._encrypt_aes_cbc,
            'RSA-OAEP': self._encrypt_rsa_oaep,
            'CHACHA20': self._encrypt_chacha20
        }

        self.decrypt_algorithms = {
            'AES-256-GCM': self._decrypt_aes_gcm,
            'AES-256-CBC': self._decrypt_aes_cbc,
            'RSA-OAEP': self._decrypt_rsa_oaep,
            'CHACHA20': self._decrypt_chacha20
        }

        # 密钥轮换策略
        self.key_rotation_policy = {
            'max_age_days': 90,  # 密钥最大年龄
            'max_usage_count': 10000,  # 最大使用次数
            'rotation_interval_days': 30  # 轮换间隔
        }

        # 初始化
        self._load_keys()
        if not self.keys:  # 只有在没有密钥时才初始化默认密钥
            self._initialize_default_keys()

        if not CRYPTOGRAPHY_AVAILABLE:
            self._initialize_fallback_encryption()

        logging.info("数据加密管理器初始化完成")

    def _initialize_fallback_encryption(self):
        """初始化降级加密实现"""
        logging.warning("使用降级加密实现，请考虑安装cryptography库以获得更好的安全性")

        # 简单的XOR加密作为降级方案

        def simple_xor_encrypt(data: bytes, key: bytes) -> bytes:

            return bytes(a ^ b for a, b in zip(data, key * (len(data) // len(key) + 1)))

        def simple_xor_decrypt(data: bytes, key: bytes) -> bytes:

            return simple_xor_encrypt(data, key)  # XOR是自逆的

        self.algorithms = {
            'SIMPLE - XOR': lambda data, key, **kwargs: simple_xor_encrypt(data, key),
            'AES - 256 - GCM': lambda data, key, **kwargs: simple_xor_encrypt(data, key),
            'AES - 256 - CBC': lambda data, key, **kwargs: simple_xor_encrypt(data, key),
            'RSA - OAEP': lambda data, key, **kwargs: simple_xor_encrypt(data, key),
            'CHACHA20': lambda data, key, **kwargs: simple_xor_encrypt(data, key)
        }

        self.decrypt_algorithms = {
            'SIMPLE - XOR': lambda data, key, **kwargs: simple_xor_decrypt(data, key),
            'AES - 256 - GCM': lambda data, key, **kwargs: simple_xor_decrypt(data, key),
            'AES - 256 - CBC': lambda data, key, **kwargs: simple_xor_decrypt(data, key),
            'RSA - OAEP': lambda data, key, **kwargs: simple_xor_decrypt(data, key),
            'CHACHA20': lambda data, key, **kwargs: simple_xor_decrypt(data, key)
        }

    def encrypt_data(self, data: Union[str, bytes], algorithm: str = "AES-256-GCM",
                     key_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> EncryptionResult:
        """
        加密数据

        Args:
            data: 要加密的数据
            algorithm: 加密算法
            key_id: 密钥ID，如果为None则使用当前活动密钥
            metadata: 元数据

        Returns:
            加密结果
        """
        # 准备数据和密钥
        data_bytes = self._prepare_data_for_encryption(data)
        key = self._get_or_create_encryption_key(key_id, algorithm)

        # 执行加密
        encrypted_data = self._perform_encryption(data_bytes, algorithm, key)

        # 创建结果
        result = self._create_encryption_result(encrypted_data, key.key_id, algorithm, metadata)

        # 后处理
        self._post_encrypt_processing(key, data_bytes, encrypted_data, algorithm)

        return result

    def _prepare_data_for_encryption(self, data: Union[str, bytes]) -> bytes:
        """准备加密数据"""
        return data.encode('utf-8') if isinstance(data, str) else data

    def _get_or_create_encryption_key(self, key_id: Optional[str], algorithm: str) -> EncryptionKey:
        """获取或创建加密密钥"""
        if key_id is None:
            # 查找适合算法的活动密钥
            for key in self.keys.values():
                if key.is_active and key.algorithm in algorithm:
                    return key

            # 如果没有找到，生成新密钥
            key_id = self.generate_key(algorithm)

        if key_id not in self.keys:
            raise ValueError(f"无效的密钥ID: {key_id}")

        key = self.keys[key_id]
        if not key.can_use():
            raise ValueError(f"密钥不可用: {key_id}")

        return key

    def _get_encryption_key(self, key_id: Optional[str]) -> EncryptionKey:
        """获取加密密钥"""
        if key_id is None:
            key_id = self.current_key_id

        if key_id is None or key_id not in self.keys:
            raise ValueError(f"无效的密钥ID: {key_id}")

        key = self.keys[key_id]
        if not key.can_use():
            raise ValueError(f"密钥不可用: {key_id}")

        return key

    def _perform_encryption(self, data_bytes: bytes, algorithm: str, key: EncryptionKey) -> bytes:
        """执行加密操作"""
        if algorithm not in self.algorithms:
            raise ValueError(f"不支持的加密算法: {algorithm}")

        return self.algorithms[algorithm](data_bytes, key.key_data)

    def _create_encryption_result(self, encrypted_data: bytes, key_id: str, algorithm: str, metadata: Optional[Dict[str, Any]]) -> EncryptionResult:
        """创建加密结果"""
        iv = None
        tag = None

        # 解析不同算法的加密结果
        if algorithm == "AES-256-GCM" and len(encrypted_data) >= 28:  # IV(12) + Tag(16) + Ciphertext
            iv = encrypted_data[:12]
            tag = encrypted_data[12:28]
            encrypted_data = encrypted_data[28:]
        elif algorithm == "AES-256-CBC" and len(encrypted_data) >= 16:  # IV(16) + Ciphertext
            iv = encrypted_data[:16]
            encrypted_data = encrypted_data[16:]

        return EncryptionResult(
            encrypted_data=encrypted_data,
            key_id=key_id,
            algorithm=algorithm,
            iv=iv,
            tag=tag,
            metadata=metadata or {}
        )

    def _post_encrypt_processing(self, key: EncryptionKey, data_bytes: bytes, encrypted_data: bytes, algorithm: str) -> None:
        """加密后处理"""
        # 更新密钥使用计数
        key.increment_usage()

        # 更新统计信息
        self.total_encryptions += 1

        # 审计日志
        if self.enable_audit:
            self._audit_encryption(key.key_id, algorithm, len(data_bytes), len(encrypted_data))

        # 检查是否需要轮换密钥
        self._check_key_rotation(key)

    def _audit_encryption(self, key_id: str, algorithm: str, data_size: int, result_size: int) -> None:
        """记录加密审计日志"""
        self._audit_log('encrypt', {
            'key_id': key_id,
            'algorithm': algorithm,
            'data_size': data_size,
            'result_size': result_size
        })

    def decrypt_data(self, encrypted_result: EncryptionResult) -> DecryptionResult:
        """
        解密数据

        Args:
            encrypted_result: 加密结果

        Returns:
            解密结果
        """
        # 获取解密密钥
        key_id = encrypted_result.key_id
        if key_id not in self.keys:
            raise KeyError(f"密钥不存在: {key_id}")

        key = self.keys[key_id]
        if not key.can_use():
            raise ValueError(f"密钥不可用: {key_id}")

        # 执行解密
        algorithm = encrypted_result.algorithm
        if algorithm not in self.decrypt_algorithms:
            raise ValueError(f"不支持的解密算法: {algorithm}")

        try:
            # 对于需要IV和tag的算法，需要传递额外参数
            if algorithm in ["AES-256-GCM", "AES-256-CBC"]:
                decrypted_data = self.decrypt_algorithms[algorithm](
                    encrypted_result.encrypted_data,
                    key.key_data,
                    encrypted_result.iv,
                    encrypted_result.tag
                )
            else:
                decrypted_data = self.decrypt_algorithms[algorithm](
                    encrypted_result.encrypted_data,
                    key.key_data
                )

            # 更新统计信息
            self.total_decryptions += 1

            # 创建解密结果
            result = DecryptionResult(
                decrypted_data=decrypted_data,
                key_id=key_id,
                algorithm=algorithm,
                metadata=encrypted_result.metadata.copy()
            )

            # 审计日志
            if self.enable_audit:
                self._audit_log('decrypt', {
                    'key_id': key_id,
                    'algorithm': algorithm,
                    'encrypted_size': len(encrypted_result.encrypted_data),
                    'decrypted_size': len(decrypted_data)
                })

            return result

        except Exception as e:
            logging.error(f"数据解密失败: {e}")
            raise

    def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> bytes:
        """AES - GCM加密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.algorithms['SIMPLE - XOR'](data, key)

        iv = os.urandom(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # 返回格式: IV(12) + Tag(16) + Ciphertext
        return iv + encryptor.tag + ciphertext

    def _decrypt_aes_gcm(self, ciphertext: bytes, key: bytes, iv: Optional[bytes] = None, tag: Optional[bytes] = None) -> bytes:
        """AES - GCM解密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.decrypt_algorithms['SIMPLE - XOR'](ciphertext, key)

        # 如果没有提供IV和tag，从数据中解析（向后兼容）
        if iv is None:
            iv = ciphertext[:12]
            tag = ciphertext[12:28]
            ciphertext = ciphertext[28:]

        if iv is None or tag is None:
            raise ValueError("AES-GCM解密需要IV和tag")

        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext

    def _encrypt_aes_cbc(self, data: bytes, key: bytes) -> bytes:
        """AES - CBC加密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.algorithms['SIMPLE - XOR'](data, key)

        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # PKCS7填充
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding_length]) * padding_length

        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # 返回格式: IV(16) + Ciphertext
        return iv + ciphertext

    def _decrypt_aes_cbc(self, ciphertext: bytes, key: bytes, iv: Optional[bytes] = None, tag: Optional[bytes] = None) -> bytes:
        """AES - CBC解密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.decrypt_algorithms['SIMPLE - XOR'](ciphertext, key)

        # 如果没有提供IV，从数据中解析（向后兼容）
        if iv is None:
            iv = ciphertext[:16]
            ciphertext = ciphertext[16:]

        if iv is None:
            raise ValueError("AES-CBC解密需要IV")

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        # 移除PKCS7填充
        padding_length = padded_plaintext[-1]
        plaintext = padded_plaintext[:-padding_length]

        return plaintext

    def _encrypt_rsa_oaep(self, data: bytes, key: bytes) -> bytes:
        """RSA - OAEP加密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.algorithms['SIMPLE - XOR'](data, key)

        # key已经是PEM格式的私钥数据，直接加载
        private_key = serialization.load_pem_private_key(
            key, password=None, backend=default_backend())
        public_key = private_key.public_key()

        ciphertext = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return ciphertext

    def _decrypt_rsa_oaep(self, data: bytes, key: bytes) -> bytes:
        """RSA - OAEP解密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.decrypt_algorithms['SIMPLE - XOR'](data, key)

        # 从密钥数据加载RSA私钥
        private_key = serialization.load_pem_private_key(
            key, password=None, backend=default_backend())

        plaintext = private_key.decrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return plaintext

    def _encrypt_chacha20(self, data: bytes, key: bytes) -> bytes:
        """ChaCha20加密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.algorithms['SIMPLE - XOR'](data, key)

        nonce = os.urandom(16)
        algorithm = algorithms.ChaCha20(key, nonce)
        cipher = Cipher(algorithm, mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data)

        # 返回格式: Nonce(16) + Ciphertext
        return nonce + ciphertext

    def _decrypt_chacha20(self, data: bytes, key: bytes) -> bytes:
        """ChaCha20解密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.decrypt_algorithms['SIMPLE - XOR'](data, key)

        nonce = data[:16]
        ciphertext = data[16:]

        algorithm = algorithms.ChaCha20(key, nonce)
        cipher = Cipher(algorithm, mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext)

        return plaintext

    # =========================================================================
    # 密钥管理
    # =========================================================================

    def generate_key(self, algorithm: str = "AES-256", expires_in_days: Optional[int] = None) -> str:
        """
        生成新密钥

        Args:
            algorithm: 密钥算法
            expires_in_days: 过期天数

        Returns:
            密钥ID
        """
        # 生成密钥ID
        key_id = self._generate_key_id()

        # 生成密钥数据
        key_data, key_algorithm = self._generate_key_data(algorithm)

        # 创建密钥对象
        key = self._create_key_object(key_id, key_data, key_algorithm, expires_in_days)

        # 存储密钥
        self._store_key(key)

        # 记录审计日志
        self._audit_key_generation(key)

        logging.info(f"生成新密钥: {key_id} ({key_algorithm})")
        return key_id

    def _generate_key_id(self) -> str:
        """生成密钥ID"""
        import uuid
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_part = str(uuid.uuid4())[:8]  # 使用UUID的前8位确保唯一性
        return f"key_{timestamp}_{unique_part}"

    def _generate_key_data(self, algorithm: str) -> tuple:
        """生成密钥数据"""
        if algorithm.startswith("AES"):
            return self._generate_aes_key_data(algorithm)
        elif algorithm.startswith("RSA"):
            return self._generate_rsa_key_data(algorithm)
        elif algorithm == "CHACHA20":
            return self._generate_chacha20_key_data()
        else:
            raise ValueError(f"不支持的密钥算法: {algorithm}")

    def _generate_aes_key_data(self, algorithm: str) -> tuple:
        """生成AES密钥数据"""
        key_size = 32 if "256" in algorithm else 24 if "192" in algorithm else 16
        key_data = os.urandom(key_size)
        key_algorithm = f"AES-{key_size * 8}"
        return key_data, key_algorithm

    def _generate_rsa_key_data(self, algorithm: str) -> tuple:
        """生成RSA密钥数据"""
        if CRYPTOGRAPHY_AVAILABLE:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )

            # 导出私钥（用于解密）
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            key_algorithm = "RSA-2048"
        else:
            # 降级实现
            key_data = os.urandom(32)
            key_algorithm = "RSA-FAKE"

        return key_data, key_algorithm

    def _generate_chacha20_key_data(self) -> tuple:
        """生成ChaCha20密钥数据"""
        key_data = os.urandom(32)
        return key_data, "CHACHA20"

    def _create_key_object(self, key_id: str, key_data: bytes, algorithm: str, expires_in_days: Optional[int]) -> EncryptionKey:
        """创建密钥对象"""
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        return EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            algorithm=algorithm,
            created_at=datetime.now(),
            expires_at=expires_at
        )

    def _store_key(self, key: EncryptionKey) -> None:
        """存储密钥"""
        self.keys[key.key_id] = key
        self.current_key_id = key.key_id
        self._save_key(key)

    def _audit_key_generation(self, key: EncryptionKey) -> None:
        """记录密钥生成审计日志"""
        if self.enable_audit:
            self._audit_log('generate_key', {
                'key_id': key.key_id,
                'algorithm': key.algorithm
            })

    def rotate_keys(self) -> List[str]:
        """
        轮换密钥

        Returns:
            新生成的密钥ID列表
        """
        rotated_keys = []

        # 为每个活动密钥生成新版本
        active_keys = [k for k in self.keys.values() if k.is_active]

        for old_key in active_keys:
            # 生成新密钥
            new_key_id = self.generate_key(old_key.algorithm.split('-')[0])

            # 标记旧密钥为非活跃
            old_key.is_active = False

            rotated_keys.append(new_key_id)

            logging.info(f"轮换密钥: {old_key.key_id} -> {new_key_id}")

        # 审计日志
        if self.enable_audit:
            self._audit_log('rotate_keys', {
                'rotated_count': len(rotated_keys),
                'new_keys': rotated_keys
            })

        return rotated_keys

    def _check_key_rotation(self, key: EncryptionKey):
        """检查是否需要轮换密钥"""
        should_rotate = False

        # 检查使用次数
        if key.usage_count >= self.key_rotation_policy['max_usage_count']:
            should_rotate = True

        # 检查年龄
        if key.created_at:
            age_days = (datetime.now() - key.created_at).days
            if age_days >= self.key_rotation_policy['max_age_days']:
                should_rotate = True

        if should_rotate:
            logging.info(f"密钥需要轮换: {key.key_id}")
            self.rotate_keys()

    def _load_keys(self):
        """加载密钥"""
        keys_file = self.key_store_path / "keys.json"
        if keys_file.exists():
            try:
                with open(keys_file, 'r') as f:
                    keys_data = json.load(f)

                for key_data in keys_data:
                    key = EncryptionKey(
                        key_id=key_data['key_id'],
                        key_data=base64.b64decode(key_data['key_data']),
                        algorithm=key_data['algorithm'],
                        created_at=datetime.fromisoformat(key_data['created_at']),
                        expires_at=datetime.fromisoformat(
                            key_data['expires_at']) if key_data.get('expires_at') else None,
                        is_active=key_data.get('is_active', True),
                        usage_count=key_data.get('usage_count', 0),
                        metadata=key_data.get('metadata', {})
                    )

                    self.keys[key.key_id] = key

            except Exception as e:
                logging.error(f"加载密钥失败 {keys_file}: {e}")
        else:
            # 兼容旧版本的.key文件格式
            key_files = list(self.key_store_path.glob("*.key"))
            for key_file in key_files:
                try:
                    with open(key_file, 'r') as f:
                        key_data = json.load(f)

                    key = EncryptionKey(
                        key_id=key_data['key_id'],
                        key_data=base64.b64decode(key_data['key_data']),
                        algorithm=key_data['algorithm'],
                        created_at=datetime.fromisoformat(key_data['created_at']),
                        expires_at=datetime.fromisoformat(
                            key_data['expires_at']) if key_data.get('expires_at') else None,
                        is_active=key_data.get('is_active', True),
                        usage_count=key_data.get('usage_count', 0)
                    )

                    self.keys[key.key_id] = key

                except Exception as e:
                    logging.error(f"加载密钥失败 {key_file}: {e}")

    def _save_key(self, key: EncryptionKey):
        """保存密钥"""
        key_file = self.key_store_path / f"{key.key_id}.key"

        key_data = {
            'key_id': key.key_id,
            'key_data': base64.b64encode(key.key_data).decode(),
            'algorithm': key.algorithm,
            'created_at': key.created_at.isoformat(),
            'expires_at': key.expires_at.isoformat() if key.expires_at else None,
            'is_active': key.is_active,
            'usage_count': key.usage_count
        }

        with open(key_file, 'w') as f:
            json.dump(key_data, f, indent=2)

    def _initialize_default_keys(self):
        """初始化默认密钥"""
        if not self.keys:
            # 生成默认AES密钥
            self.generate_key("AES - 256", expires_in_days=365)
            logging.info("生成默认AES密钥")

    # =========================================================================
    # 审计和监控
    # =========================================================================

    def _audit_log(self, operation: str, details: Dict[str, Any]):
        """审计日志"""
        if not self.enable_audit:
            return

        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details
        }

        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logging.error(f"审计日志写入失败: {e}")

    def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取审计日志

        Args:
            limit: 限制条数

        Returns:
            审计日志列表
        """
        if not self.audit_log_path.exists():
            return []

        logs = []
        try:
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    if len(logs) >= limit:
                        break
                    try:
                        logs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logging.error(f"读取审计日志失败: {e}")

        return logs[::-1]  # 返回最新的日志

    def get_encryption_stats(self) -> Dict[str, Any]:
        """
        获取加密统计信息

        Returns:
            统计信息
        """
        total_keys = len(self.keys)
        active_keys = len([k for k in self.keys.values() if k.is_active])
        expired_keys = len([k for k in self.keys.values() if k.is_expired()])

        algorithm_usage = {}
        for key in self.keys.values():
            algorithm_usage[key.algorithm] = algorithm_usage.get(key.algorithm, 0) + 1

        return {
            'total_keys': total_keys,
            'active_keys': active_keys,
            'expired_keys': expired_keys,
            'total_encryptions': self.total_encryptions,
            'total_decryptions': self.total_decryptions,
            'algorithm_usage': algorithm_usage,
            'current_key_id': self.current_key_id,
            'audit_enabled': self.enable_audit,
            'cryptography_available': CRYPTOGRAPHY_AVAILABLE
        }

    # =========================================================================
    # 批量操作
    # =========================================================================

    def encrypt_batch(self, data_list: List[Dict[str, Any]],
                      algorithm: str = "AES-256-GCM") -> List[EncryptionResult]:
        """
        批量加密数据

        Args:
            data_list: 数据列表 [{'data': bytes, 'metadata': dict}]
            algorithm: 加密算法

        Returns:
            加密结果列表
        """
        results = []

        for item in data_list:
            try:
                result = self.encrypt_data(
                    item['data'],
                    algorithm=algorithm,
                    metadata=item.get('metadata', {})
                )
                results.append(result)
            except Exception as e:
                logging.error(f"批量加密失败: {e}")
                # 创建错误结果
                error_result = EncryptionResult(
                    encrypted_data=b'',
                    key_id='error',
                    algorithm=algorithm,
                    metadata={'error': str(e)}
                )
                results.append(error_result)

        logging.info(f"批量加密完成: {len(results)}/{len(data_list)} 成功")
        return results

    def decrypt_batch(self, encrypted_results: List[EncryptionResult]) -> List[DecryptionResult]:
        """
        批量解密数据

        Args:
            encrypted_results: 加密结果列表

        Returns:
            解密结果列表
        """
        results = []

        for encrypted_result in encrypted_results:
            try:
                result = self.decrypt_data(encrypted_result)
                results.append(result)
            except Exception as e:
                logging.error(f"批量解密失败: {e}")
                # 创建错误结果
                error_result = DecryptionResult(
                    decrypted_data=b'',
                    key_id=encrypted_result.key_id,
                    algorithm=encrypted_result.algorithm,
                    metadata={'error': str(e)}
                )
                results.append(error_result)

        logging.info(f"批量解密完成: {len(results)}/{len(encrypted_results)} 成功")
        return results

    # =========================================================================
    # 清理和维护
    # =========================================================================

    def cleanup_expired_keys(self) -> int:
        """
        清理过期密钥

        Returns:
            清理的密钥数量
        """
        expired_keys = [key_id for key_id, key in self.keys.items() if key.is_expired()]
        cleaned_count = 0

        for key_id in expired_keys:
            try:
                # 删除密钥文件
                key_file = self.key_store_path / f"{key_id}.key"
                if key_file.exists():
                    key_file.unlink()

                # 从内存中删除
                del self.keys[key_id]
                cleaned_count += 1

                if self.enable_audit:
                    self._audit_log('cleanup_key', {'key_id': key_id})

            except Exception as e:
                logging.error(f"清理密钥失败 {key_id}: {e}")

        logging.info(f"清理过期密钥: {cleaned_count} 个")
        return cleaned_count

    def export_keys(self, export_path: str, include_private: bool = False):
        """
        导出密钥

        Args:
            export_path: 导出路径
            include_private: 是否包含私钥信息
        """
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'keys': []
        }

        for key in self.keys.values():
            key_data = {
                'key_id': key.key_id,
                'algorithm': key.algorithm,
                'created_at': key.created_at.isoformat(),
                'expires_at': key.expires_at.isoformat() if key.expires_at else None,
                'is_active': key.is_active,
                'usage_count': key.usage_count
            }

            if include_private:
                key_data['key_data'] = base64.b64encode(key.key_data).decode()

            export_data['keys'].append(key_data)

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logging.info(f"密钥已导出到: {export_path}")

    def shutdown(self):
        """关闭加密管理器"""
        # 保存所有密钥到JSON文件
        keys_data = []
        for key in self.keys.values():
            key_data = {
                'key_id': key.key_id,
                'key_data': base64.b64encode(key.key_data).decode(),
                'algorithm': key.algorithm,
                'created_at': key.created_at.isoformat(),
                'expires_at': key.expires_at.isoformat() if key.expires_at else None,
                'is_active': key.is_active,
                'usage_count': key.usage_count,
                'metadata': key.metadata
            }
            keys_data.append(key_data)

        keys_file = self.key_store_path / "keys.json"
        with open(keys_file, 'w') as f:
            json.dump(keys_data, f, indent=2)

        logging.info("数据加密管理器已关闭")
