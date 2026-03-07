"""
configencryptionmanager 模块

提供 configencryptionmanager 相关功能和接口。
"""

import os
import logging
import threading
from typing import Optional, Dict

import platform
import platform
import base64
import hashlib
import secrets
import time

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

"""安全配置相关类"""


class ConfigEncryptionManager:
    """配置加密管理器"""

    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._generate_master_key()
        self._key_cache: Dict[str, Fernet] = {}
        self._key_timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()

    def _generate_master_key(self) -> str:
        """生成主密钥"""
        # 使用系统信息和随机数生成密钥（跨平台兼容）
        try:
            try:
                system_info = str(os.uname())
            except AttributeError:
                # Windows系统没有uname，使用替代方案
                system_info = f"{platform.system()} {platform.release()} {platform.machine()}"
        except AttributeError:
            # Windows系统没有uname，使用替代方案
            system_info = f"{platform.system()}{platform.release()}{platform.version()}"

        system_info = f"{system_info}{time.time()}{secrets.token_hex(32)}"
        key_hash = hashlib.sha256(system_info.encode()).digest()
        return base64.urlsafe_b64encode(key_hash[:32]).decode()

    def _derive_key(self, context: str) -> Fernet:
        """派生加密密钥"""
        with self._lock:
            if context in self._key_cache:
                # 检查密钥是否需要轮转
                if time.time() - self._key_timestamps[context] > 30 * 24 * 3600:  # 30天
                    del self._key_cache[context]
                    del self._key_timestamps[context]

            if context not in self._key_cache:
                # 使用PBKDF2派生密钥
                salt = hashlib.sha256(context.encode()).digest()
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
                self._key_cache[context] = Fernet(key)
                self._key_timestamps[context] = time.time()

            return self._key_cache[context]

    def encrypt(self, data: str, context: str = "config") -> str:
        """加密数据"""
        try:
            cipher = self._derive_key(context)
            encrypted = cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"配置加密失败: {e}")
            raise

    def decrypt(self, encrypted_data: str, context: str = "config") -> str:
        """解密数据"""
        try:
            cipher = self._derive_key(context)
            encrypted = base64.urlsafe_b64decode(encrypted_data)
            decrypted = cipher.decrypt(encrypted)
            return decrypted.decode()
        except InvalidToken:
            logger.error("配置解密失败：无效的加密令牌")
            raise ValueError("无效的加密配置数据")
        except Exception as e:
            logger.error(f"配置解密失败: {e}")
            raise


logger = logging.getLogger(__name__)




