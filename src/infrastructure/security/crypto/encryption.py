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


@dataclass(init=False)
class EncryptionResult:
    """加密结果"""
    ciphertext: bytes
    key_id: str
    algorithm: str
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    encrypted_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None

    def __init__(
        self,
        *,
        ciphertext: Optional[bytes] = None,
        key_id: str,
        algorithm: str,
        iv: Optional[bytes] = None,
        tag: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None,
        encrypted_at: Optional[datetime] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        encrypted_data: Optional[bytes] = None,
    ) -> None:
        payload = ciphertext if ciphertext is not None else encrypted_data
        if payload is None:
            payload = b""
        self.ciphertext = payload
        self.key_id = key_id
        self.algorithm = algorithm
        self.iv = iv
        self.tag = tag
        self.metadata = dict(metadata) if metadata else {}
        self.encrypted_at = encrypted_at or datetime.now()
        self.success = success
        self.error_message = error_message

    @property
    def encrypted_data(self) -> bytes:
        """兼容旧字段名"""
        return self.ciphertext


@dataclass(init=False)
class DecryptionResult:
    """解密结果"""
    plaintext: bytes
    key_id: str
    algorithm: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    decrypted_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None

    def __init__(
        self,
        *,
        plaintext: Optional[bytes] = None,
        key_id: str,
        algorithm: str,
        metadata: Optional[Dict[str, Any]] = None,
        decrypted_at: Optional[datetime] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        decrypted_data: Optional[bytes] = None,
    ) -> None:
        data = plaintext if plaintext is not None else decrypted_data
        if data is None:
            data = b""
        self.plaintext = data
        self.key_id = key_id
        self.algorithm = algorithm
        self.metadata = dict(metadata) if metadata else {}
        self.decrypted_at = decrypted_at or datetime.now()
        self.success = success
        self.error_message = error_message

    @property
    def decrypted_data(self) -> bytes:
        """兼容旧字段名"""
        return self.plaintext


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

        # 算法别名与加解密实现映射
        self._algorithm_aliases = self._build_encryption_aliases()
        self._key_algorithm_aliases = self._build_key_aliases()

        self.algorithms = {
            'AES-256-GCM': self._encrypt_aes_gcm,
            'AES-256-CBC': self._encrypt_aes_cbc,
            'RSA-OAEP': self._encrypt_rsa_oaep,
            'ChaCha20': self._encrypt_chacha20
        }

        self.decrypt_algorithms = {
            'AES-256-GCM': self._decrypt_aes_gcm,
            'AES-256-CBC': self._decrypt_aes_cbc,
            'RSA-OAEP': self._decrypt_rsa_oaep,
            'ChaCha20': self._decrypt_chacha20
        }

        # 运行统计
        self._stats = {
            "total_encryptions": 0,
            "failed_encryptions": 0,
            "total_decryptions": 0,
            "failed_decryptions": 0,
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

        self._algorithm_aliases.setdefault("SIMPLEXOR", "SIMPLE-XOR")

        # 降级实现直接覆盖加密/解密映射
        fallback_encrypt = lambda data, key, **kwargs: simple_xor_encrypt(data, key)
        fallback_decrypt = lambda data, key, **kwargs: simple_xor_decrypt(data, key)

        self.algorithms.update({
            'SIMPLE-XOR': fallback_encrypt,
            'AES-256-GCM': fallback_encrypt,
            'AES-256-CBC': fallback_encrypt,
            'RSA-OAEP': fallback_encrypt,
            'ChaCha20': fallback_encrypt
        })

        self.decrypt_algorithms.update({
            'SIMPLE-XOR': fallback_decrypt,
            'AES-256-GCM': fallback_decrypt,
            'AES-256-CBC': fallback_decrypt,
            'RSA-OAEP': fallback_decrypt,
            'ChaCha20': fallback_decrypt
        })

    # ------------------------------------------------------------------
    # 算法名称标准化
    # ------------------------------------------------------------------

    def _build_encryption_aliases(self) -> Dict[str, str]:
        return {
            "AES256GCM": "AES-256-GCM",
            "AES256CBC": "AES-256-CBC",
            "RSAOAEP": "RSA-OAEP",
            "CHACHA20": "ChaCha20",
            "SIMPLEXOR": "SIMPLE-XOR",
        }

    def _build_key_aliases(self) -> Dict[str, str]:
        return {
            "AES256": "AES-256",
            "AES192": "AES-192",
            "AES128": "AES-128",
            "RSA2048": "RSA-2048",
            "CHACHA20": "ChaCha20",
        }

    @staticmethod
    def _normalize_alias(name: str) -> str:
        return ''.join(ch for ch in name if ch.isalnum()).upper()

    def _resolve_encryption_algorithm(self, name: str) -> str:
        canonical = self._algorithm_aliases.get(self._normalize_alias(name))
        if not canonical or canonical not in self.algorithms:
            raise ValueError(f"不支持的加密算法: {name}")
        return canonical

    def _resolve_key_algorithm(self, name: str) -> str:
        canonical = self._key_algorithm_aliases.get(self._normalize_alias(name))
        if not canonical:
            raise ValueError(f"不支持的密钥算法: {name}")
        return canonical

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
        # 转换为字节串
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data

        metadata_dict = dict(metadata) if metadata else {}
        display_algorithm = algorithm

        algorithm_name = self._resolve_encryption_algorithm(algorithm)

        # 获取加密密钥
        key_id = key_id or self.current_key_id

        if key_id is None or key_id not in self.keys:
            self._stats["failed_encryptions"] += 1
            raise ValueError(f"无效的密钥ID: {key_id}")

        key = self.keys[key_id]
        if not key.can_use():
            self._stats["failed_encryptions"] += 1
            raise ValueError(f"密钥不可用: {key_id}")

        # 执行加密
        encryptor = self.algorithms.get(algorithm_name)
        if encryptor is None:
            self._stats["failed_encryptions"] += 1
            raise ValueError(f"不支持的加密算法: {algorithm}")

        try:
            ciphertext = encryptor(data_bytes, key.key_data)

            # 更新密钥使用计数
            key.increment_usage()

            # 创建加密结果
            result = EncryptionResult(
                ciphertext=ciphertext,
                key_id=key_id,
                algorithm=display_algorithm,
                metadata=metadata_dict
            )

            self._stats["total_encryptions"] += 1

            # 审计日志
            if self.enable_audit:
                self._audit_log('encrypt_data', {
                    'key_id': key_id,
                    'algorithm': algorithm_name,
                    'data_size': len(data_bytes),
                    'result_size': len(ciphertext)
                })

            # 检查是否需要轮换密钥
            self._check_key_rotation(key)

            return result

        except Exception as e:
            self._stats["failed_encryptions"] += 1
            logging.error(f"数据加密失败: {e}")
            raise

    def decrypt_data(self, encrypted_result: EncryptionResult) -> DecryptionResult:
        """
        解密数据

        Args:
            encrypted_result: 加密结果

        Returns:
            解密结果
        """
        display_algorithm = encrypted_result.algorithm
        algorithm_name = self._resolve_encryption_algorithm(display_algorithm)

        metadata = dict(encrypted_result.metadata) if encrypted_result.metadata else {}

        # 获取解密密钥
        key_id = encrypted_result.key_id
        key = self.keys.get(key_id)
        if key is None:
            self._stats["failed_decryptions"] += 1
            return DecryptionResult(
                plaintext=b'',
                key_id=key_id,
                algorithm=display_algorithm,
                metadata=metadata,
                success=False,
                error_message=f"Key not found: {key_id}"
            )

        if not key.can_use():
            self._stats["failed_decryptions"] += 1
            return DecryptionResult(
                plaintext=b'',
                key_id=key_id,
                algorithm=display_algorithm,
                metadata=metadata,
                success=False,
                error_message=f"Key unavailable: {key_id}"
            )

        decryptor = self.decrypt_algorithms.get(algorithm_name)
        if decryptor is None:
            self._stats["failed_decryptions"] += 1
            raise ValueError(f"不支持的解密算法: {encrypted_result.algorithm}")

        try:
            plaintext = decryptor(
                encrypted_result.ciphertext,
                key.key_data
            )

            result = DecryptionResult(
                plaintext=plaintext,
                key_id=key_id,
                algorithm=display_algorithm,
                metadata=metadata,
                success=True
            )

            self._stats["total_decryptions"] += 1

            if self.enable_audit:
                self._audit_log('decrypt_data', {
                    'key_id': key_id,
                    'algorithm': algorithm_name,
                    'encrypted_size': len(encrypted_result.ciphertext),
                    'decrypted_size': len(plaintext)
                })

            return result

        except Exception as e:
            self._stats["failed_decryptions"] += 1
            logging.error(f"数据解密失败: {e}")
            return DecryptionResult(
                plaintext=b'',
                key_id=key_id,
                algorithm=display_algorithm,
                metadata=metadata,
                success=False,
                error_message=str(e)
            )

    def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> bytes:
        """AES - GCM加密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.algorithms['SIMPLE - XOR'](data, key)

        # 仅在CRYPTOGRAPHY_AVAILABLE为True时使用加密库
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        
        iv = os.urandom(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # 返回格式: IV(12) + Tag(16) + Ciphertext
        return iv + encryptor.tag + ciphertext

    def _decrypt_aes_gcm(self, data: bytes, key: bytes) -> bytes:
        """AES - GCM解密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.decrypt_algorithms['SIMPLE - XOR'](data, key)

        # 仅在CRYPTOGRAPHY_AVAILABLE为True时使用加密库
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        
        iv = data[:12]
        tag = data[12:28]
        ciphertext = data[28:]

        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext

    def _encrypt_aes_cbc(self, data: bytes, key: bytes) -> bytes:
        """AES - CBC加密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.algorithms['SIMPLE - XOR'](data, key)

        # 仅在CRYPTOGRAPHY_AVAILABLE为True时使用加密库
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        
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

    def _decrypt_aes_cbc(self, data: bytes, key: bytes) -> bytes:
        """AES - CBC解密"""
        if not CRYPTOGRAPHY_AVAILABLE:
            return self.decrypt_algorithms['SIMPLE - XOR'](data, key)

        # 仅在CRYPTOGRAPHY_AVAILABLE为True时使用加密库
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        
        iv = data[:16]
        ciphertext = data[16:]

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

        # 仅在CRYPTOGRAPHY_AVAILABLE为True时使用加密库
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        
        # 从密钥数据加载RSA公钥
        public_key = serialization.load_pem_public_key(key, backend=default_backend())
        
        # 确保是RSA公钥
        if not isinstance(public_key, rsa.RSAPublicKey):
            raise ValueError("密钥不是有效的RSA公钥")

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

        # 仅在CRYPTOGRAPHY_AVAILABLE为True时使用加密库
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        
        # 从密钥数据加载RSA私钥
        private_key = serialization.load_pem_private_key(
            key, password=None, backend=default_backend())
        
        # 确保是RSA私钥
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError("密钥不是有效的RSA私钥")

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

        # 仅在CRYPTOGRAPHY_AVAILABLE为True时使用加密库
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        
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

        # 仅在CRYPTOGRAPHY_AVAILABLE为True时使用加密库
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        
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
        key_id = self._generate_key_id()
        key_data, key_algorithm = self._generate_key_data(algorithm)
        key = self._create_encryption_key(key_id, key_data, key_algorithm, expires_in_days)
        
        self.keys[key_id] = key
        self.current_key_id = key_id

        # 保存密钥
        self._save_key(key)

        # 审计日志
        if self.enable_audit:
            self._audit_log('generate_key', {
                'key_id': key_id,
                'algorithm': key_algorithm
            })

        logging.info(f"生成新密钥: {key_id} ({key_algorithm})")
        return key_id

    def _generate_key_id(self) -> str:
        """生成密钥ID"""
        return f"key_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    def _generate_key_data(self, algorithm: str) -> tuple[bytes, str]:
        """
        根据算法生成密钥数据
        
        Args:
            algorithm: 密钥算法
            
        Returns:
            (key_data, key_algorithm) 元组
        """
        canonical = self._resolve_key_algorithm(algorithm)

        if canonical.startswith("AES-"):
            return self._generate_aes_key_data(canonical)
        if canonical.startswith("RSA-"):
            return self._generate_rsa_key_data(canonical)
        if canonical == "ChaCha20":
            return self._generate_chacha20_key_data()
        raise ValueError(f"不支持的密钥算法: {algorithm}")

    def _generate_aes_key_data(self, algorithm: str) -> tuple[bytes, str]:
        """生成AES密钥数据"""
        if algorithm not in {"AES-128", "AES-192", "AES-256"}:
            raise ValueError(f"不支持的AES算法: {algorithm}")

        key_size = int(algorithm.split('-')[1]) // 8
        key_data = os.urandom(key_size)
        return key_data, algorithm

    def _generate_rsa_key_data(self, algorithm: str) -> tuple[bytes, str]:
        """生成RSA密钥数据"""
        if CRYPTOGRAPHY_AVAILABLE:
            # 仅在CRYPTOGRAPHY_AVAILABLE为True时导入加密库
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
            
            key_bits = int(''.join([ch for ch in algorithm if ch.isdigit()]) or "2048")

            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_bits,
                backend=default_backend()
            )

            # 导出公钥
            public_key = private_key.public_key()
            key_data = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            key_algorithm = algorithm
        else:
            # 降级实现
            key_data = os.urandom(32)
            key_algorithm = algorithm
        return key_data, key_algorithm

    def _generate_chacha20_key_data(self) -> tuple[bytes, str]:
        """生成ChaCha20密钥数据"""
        key_data = os.urandom(32)
        key_algorithm = "ChaCha20"
        return key_data, key_algorithm

    def _create_encryption_key(self, key_id: str, key_data: bytes, key_algorithm: str, 
                              expires_in_days: Optional[int]) -> EncryptionKey:
        """
        创建加密密钥对象
        
        Args:
            key_id: 密钥ID
            key_data: 密钥数据
            key_algorithm: 密钥算法
            expires_in_days: 过期天数
            
        Returns:
            EncryptionKey对象
        """
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        return EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            algorithm=key_algorithm,
            created_at=datetime.now(),
            expires_at=expires_at
        )

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
            new_key_id = self.generate_key(old_key.algorithm)

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
        try:
            if not self.key_store_path.exists():
                logging.info(f"密钥存储路径不存在，跳过加载: {self.key_store_path}")
                return

            key_files = list(self.key_store_path.glob("*.key"))
            if not key_files:
                logging.info(f"密钥存储路径为空，跳过加载: {self.key_store_path}")
                return

            for key_file in key_files:
                try:
                    # 添加文件大小检查，避免读取过大的文件
                    if key_file.stat().st_size > 1024 * 1024:  # 1MB限制
                        logging.warning(f"密钥文件过大，跳过: {key_file}")
                        continue

                    with open(key_file, 'r', encoding='utf-8') as f:
                        key_data = json.load(f)

                    algorithm = key_data.get('algorithm', '')
                    try:
                        algorithm = self._resolve_key_algorithm(algorithm)
                    except ValueError:
                        algorithm = algorithm or "UNKNOWN"

                    key = EncryptionKey(
                        key_id=key_data['key_id'],
                        key_data=base64.b64decode(key_data['key_data']),
                        algorithm=algorithm,
                        created_at=datetime.fromisoformat(key_data['created_at']),
                        expires_at=datetime.fromisoformat(
                            key_data['expires_at']) if key_data.get('expires_at') else None,
                        is_active=key_data.get('is_active', True),
                        usage_count=key_data.get('usage_count', 0)
                    )

                    self.keys[key.key_id] = key

                except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
                    logging.error(f"加载密钥失败 {key_file}: {e}")
                    continue

        except Exception as e:
            logging.error(f"扫描密钥目录失败: {e}")

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
            self.generate_key("AES-256", expires_in_days=365)
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

        algorithm_usage: Dict[str, int] = {}
        for key in self.keys.values():
            algorithm_usage[key.algorithm] = algorithm_usage.get(key.algorithm, 0) + 1

        stats = {
            'total_keys': total_keys,
            'active_keys': active_keys,
            'expired_keys': expired_keys,
            'algorithm_usage': algorithm_usage,
            'current_key_id': self.current_key_id,
            'audit_enabled': self.enable_audit,
            'cryptography_available': CRYPTOGRAPHY_AVAILABLE,
            'total_encryptions': self._stats["total_encryptions"],
            'total_decryptions': self._stats["total_decryptions"],
            'failed_encryptions': self._stats["failed_encryptions"],
            'failed_decryptions': self._stats["failed_decryptions"],
        }
        return stats

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
                item_algorithm = item.get('algorithm', algorithm)
                result = self.encrypt_data(
                    item['data'],
                    algorithm=item_algorithm,
                    metadata=item.get('metadata', {})
                )
                results.append(result)
            except Exception as e:
                logging.error(f"批量加密失败: {e}")
                # 创建错误结果
                error_algorithm = item.get('algorithm', algorithm)
                error_result = EncryptionResult(
                    ciphertext=b'',
                    key_id='error',
                    algorithm=error_algorithm,
                    metadata={'error': str(e)},
                    success=False,
                    error_message=str(e)
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
                error_algorithm = encrypted_result.algorithm
                error_result = DecryptionResult(
                    plaintext=b'',
                    key_id=encrypted_result.key_id,
                    algorithm=error_algorithm,
                    metadata={'error': str(e)},
                    success=False,
                    error_message=str(e)
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
        timestamp = datetime.now().isoformat()
        export_data = {
            'export_timestamp': timestamp,
            'exported_at': timestamp,
            'include_private': include_private,
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
        # 保存所有密钥
        for key in self.keys.values():
            self._save_key(key)

        logging.info("数据加密管理器已关闭")
