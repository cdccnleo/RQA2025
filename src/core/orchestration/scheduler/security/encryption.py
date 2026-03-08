"""
任务数据加密模块

提供任务敏感数据的加密存储和解密读取功能
"""

import json
import base64
import hashlib
import os
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# 尝试导入加密库
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None
    hashes = None
    PBKDF2HMAC = None


class EncryptionLevel(Enum):
    """加密级别"""
    NONE = "none"           # 不加密
    PAYLOAD = "payload"     # 仅加密payload
    FULL = "full"          # 加密所有敏感字段


@dataclass
class EncryptionConfig:
    """加密配置"""
    enabled: bool = True
    level: EncryptionLevel = EncryptionLevel.PAYLOAD
    key_file: Optional[str] = None  # 密钥文件路径
    sensitive_fields: Optional[Set[str]] = None  # 敏感字段列表

    def __post_init__(self):
        if self.sensitive_fields is None:
            # 默认敏感字段
            self.sensitive_fields = {
                'api_key', 'api_secret', 'password', 'token',
                'private_key', 'secret_key', 'credentials',
                'account_id', 'trading_password'
            }


class TaskEncryption:
    """
    任务数据加密器

    提供任务敏感数据的加密和解密功能：
    - 支持AES对称加密
    - 可配置加密级别
    - 自动识别敏感字段
    - 密钥管理

    使用场景：
    - API密钥加密存储
    - 交易密码保护
    - 敏感配置加密
    """

    def __init__(self, config: Optional[EncryptionConfig] = None):
        """
        初始化任务加密器

        Args:
            config: 加密配置
        """
        self._config = config or EncryptionConfig()
        self._cipher = None

        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("cryptography库不可用，加密功能将被禁用")
            self._config.enabled = False
            return

        if self._config.enabled:
            self._init_cipher()

    def _init_cipher(self):
        """初始化加密器"""
        try:
            key = self._get_or_create_key()
            self._cipher = Fernet(key)
            logger.info("✅ 任务数据加密已启用")
        except Exception as e:
            logger.error(f"初始化加密器失败: {e}")
            self._config.enabled = False

    def _get_or_create_key(self) -> bytes:
        """
        获取或创建加密密钥

        Returns:
            bytes: 加密密钥
        """
        if self._config.key_file and os.path.exists(self._config.key_file):
            # 从文件读取密钥
            with open(self._config.key_file, 'rb') as f:
                return f.read()

        # 生成新密钥
        key = Fernet.generate_key()

        # 保存密钥到文件
        if self._config.key_file:
            os.makedirs(os.path.dirname(self._config.key_file), exist_ok=True)
            with open(self._config.key_file, 'wb') as f:
                f.write(key)
            # 设置文件权限（仅所有者可读写）
            os.chmod(self._config.key_file, 0o600)
            logger.info(f"加密密钥已保存到: {self._config.key_file}")

        return key

    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> tuple:
        """
        从密码派生密钥

        Args:
            password: 密码
            salt: 盐值

        Returns:
            tuple: (密钥, 盐值)
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt

    def encrypt_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        加密任务payload

        Args:
            payload: 原始payload

        Returns:
            Dict[str, Any]: 加密后的payload
        """
        if not self._config.enabled or not self._cipher:
            return payload

        if self._config.level == EncryptionLevel.NONE:
            return payload

        encrypted_payload = {}

        for key, value in payload.items():
            if self._should_encrypt_field(key, value):
                try:
                    # 序列化并加密值
                    json_value = json.dumps(value)
                    encrypted_value = self._cipher.encrypt(json_value.encode())
                    encrypted_payload[key] = {
                        '__encrypted__': True,
                        'data': base64.b64encode(encrypted_value).decode()
                    }
                except Exception as e:
                    logger.error(f"加密字段 {key} 失败: {e}")
                    encrypted_payload[key] = value
            else:
                encrypted_payload[key] = value

        return encrypted_payload

    def decrypt_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        解密任务payload

        Args:
            payload: 加密的payload

        Returns:
            Dict[str, Any]: 解密后的payload
        """
        if not self._config.enabled or not self._cipher:
            return payload

        decrypted_payload = {}

        for key, value in payload.items():
            if isinstance(value, dict) and value.get('__encrypted__'):
                try:
                    # 解密并反序列化
                    encrypted_data = base64.b64decode(value['data'])
                    decrypted_data = self._cipher.decrypt(encrypted_data)
                    decrypted_payload[key] = json.loads(decrypted_data.decode())
                except Exception as e:
                    logger.error(f"解密字段 {key} 失败: {e}")
                    decrypted_payload[key] = value
            else:
                decrypted_payload[key] = value

        return decrypted_payload

    def _should_encrypt_field(self, key: str, value: Any) -> bool:
        """
        判断字段是否应该加密

        Args:
            key: 字段名
            value: 字段值

        Returns:
            bool: 是否应该加密
        """
        if self._config.level == EncryptionLevel.FULL:
            # 全加密模式：加密所有非基本类型或敏感字段
            return (
                key in self._config.sensitive_fields or
                not isinstance(value, (str, int, float, bool, list, dict)) or
                (isinstance(value, str) and len(value) > 100)  # 长字符串可能是敏感数据
            )

        # 仅加密敏感字段
        return key in self._config.sensitive_fields

    def encrypt_string(self, plaintext: str) -> str:
        """
        加密字符串

        Args:
            plaintext: 明文

        Returns:
            str: 密文（base64编码）
        """
        if not self._config.enabled or not self._cipher:
            return plaintext

        try:
            encrypted = self._cipher.encrypt(plaintext.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"加密字符串失败: {e}")
            return plaintext

    def decrypt_string(self, ciphertext: str) -> str:
        """
        解密字符串

        Args:
            ciphertext: 密文（base64编码）

        Returns:
            str: 明文
        """
        if not self._config.enabled or not self._cipher:
            return ciphertext

        try:
            encrypted = base64.b64decode(ciphertext)
            decrypted = self._cipher.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"解密字符串失败: {e}")
            return ciphertext

    def rotate_key(self, new_key_file: Optional[str] = None) -> bool:
        """
        轮换加密密钥

        Args:
            new_key_file: 新密钥文件路径

        Returns:
            bool: 是否成功
        """
        if not CRYPTOGRAPHY_AVAILABLE or not self._cipher:
            return False

        try:
            # 保存旧密钥
            old_cipher = self._cipher

            # 生成新密钥
            if new_key_file:
                self._config.key_file = new_key_file
            self._init_cipher()

            logger.info("✅ 加密密钥已轮换")
            return True

        except Exception as e:
            logger.error(f"密钥轮换失败: {e}")
            # 恢复旧密钥
            self._cipher = old_cipher
            return False

    def is_enabled(self) -> bool:
        """
        检查加密是否启用

        Returns:
            bool: 是否启用
        """
        return self._config.enabled and self._cipher is not None

    def get_encryption_info(self) -> Dict[str, Any]:
        """
        获取加密信息

        Returns:
            Dict[str, Any]: 加密信息
        """
        return {
            'enabled': self.is_enabled(),
            'level': self._config.level.value,
            'algorithm': 'AES-256-CBC' if CRYPTOGRAPHY_AVAILABLE else 'None',
            'sensitive_fields_count': len(self._config.sensitive_fields),
            'key_file': self._config.key_file
        }


# 简单的XOR加密（备用方案，不推荐用于生产环境）
class SimpleEncryption:
    """简单加密（仅用于演示，生产环境请使用TaskEncryption）"""

    def __init__(self, key: Optional[str] = None):
        self._key = key or "default_key_change_in_production"

    def encrypt(self, plaintext: str) -> str:
        """XOR加密"""
        encrypted = []
        for i, char in enumerate(plaintext):
            key_char = self._key[i % len(self._key)]
            encrypted.append(chr(ord(char) ^ ord(key_char)))
        return base64.b64encode(''.join(encrypted).encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """XOR解密"""
        try:
            encrypted = base64.b64decode(ciphertext).decode()
            decrypted = []
            for i, char in enumerate(encrypted):
                key_char = self._key[i % len(self._key)]
                decrypted.append(chr(ord(char) ^ ord(key_char)))
            return ''.join(decrypted)
        except:
            return ciphertext
