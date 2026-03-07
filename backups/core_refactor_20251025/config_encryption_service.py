"""配置加密服务"

提供配置自动加密和解密功能
"""
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from ..security import SecurityManager

logger = logging.getLogger(__name__)


@dataclass
class EncryptionConfig:

    """加密配置"""
    auto_encrypt: bool = True  # 是否自动加密敏感配置
    encrypted_prefix: str = "encrypted:"  # 加密值前缀
    sensitive_patterns: List[str] = None  # 敏感配置模式

    def __post_init__(self):

        if self.sensitive_patterns is None:
            self.sensitive_patterns = [
                "password",
                "secret",
                "key",
                "token",
                "credential",
                "auth",
                "private"
            ]


class ConfigEncryptionService:

    """配置加密服务"

    功能:
    1. 自动识别敏感配置
    2. 加密敏感配置值
    3. 解密配置值
    4. 配置完整性验证
    """

    def __init__(self, security_service: Optional[SecurityManager] = None,


                 encryption_config: Optional[EncryptionConfig] = None):
        """初始化配置加密服务"

        Args:
            security_service: 安全服务实例
            encryption_config: 加密配置
        """
        # 使用默认的安全服务实例
        self._security_service = security_service or SecurityManager()
        self._config = encryption_config or EncryptionConfig()

    def is_sensitive_key(self, key: str) -> bool:
        """检查是否为敏感配置键"

        Args:
            key: 配置键

        Returns:
            bool: 是否为敏感键
        """
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in self._config.sensitive_patterns)

    def encrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """加密配置中的敏感值"

        Args:
            config: 原始配置

        Returns:
            Dict[str, Any]: 加密后的配置
        """
        if not self._config.auto_encrypt:
            return config

        def _encrypt_value(value: Any, key: str) -> Any:
            """递归加密值"""
            if isinstance(value, dict):
                return {k: _encrypt_value(v, f"{key}.{k}" if key else k)
                        for k, v in value.items()}
            elif isinstance(value, list):
                return [_encrypt_value(item, f"{key}[{i}]")
                        for i, item in enumerate(value)]
            elif isinstance(value, str) and self.is_sensitive_key(key):
                try:
                    # 使用SecurityManager的protect_config方法
                    if hasattr(self._security_service, 'protect_config'):
                        # 如果是SecurityManager，使用protect_config
                        # 对于嵌套键，我们需要直接使用加密服务
                        if '.' in key:
                            # 嵌套键，直接使用加密服务
                            encrypted = self._security_service._encryption_service.encrypt(value)
                            return f"{self._config.encrypted_prefix}{encrypted.hex()}"
                        else:
                            # 顶层键，使用protect_config
                            protected = self._security_service.protect_config({key: value})
                            encrypted_value = protected.get(key, value)
                            if encrypted_value != value and encrypted_value.startswith("ENCRYPTED:"):
                                # 将ENCRYPTED:前缀替换为encrypted:前缀
                                return encrypted_value.replace("ENCRYPTED:", f"{self._config.encrypted_prefix}")
                            elif encrypted_value != value:
                                # 如果值被加密但没有ENCRYPTED:前缀，直接使用
                                return f"{self._config.encrypted_prefix}{encrypted_value}"
                    elif hasattr(self._security_service, 'encrypt'):
                        # 如果是SecurityService，使用encrypt
                        encrypted = self._security_service.encrypt(value)
                        return f"{self._config.encrypted_prefix}{encrypted}"
                    return value
                except Exception as e:
                    logger.warning(f"加密失败 {key}: {e}")
                    return value
            else:
                return value

        return _encrypt_value(config, "")

    def decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解密配置中的加密值"

        Args:
            config: 加密的配置

        Returns:
            Dict[str, Any]: 解密后的配置
        """
        def _decrypt_value(value: Any, key: str) -> Any:
            """递归解密值"""
            if isinstance(value, dict):
                return {k: _decrypt_value(v, f"{key}.{k}" if key else k)
                        for k, v in value.items()}
            elif isinstance(value, list):
                return [_decrypt_value(item, f"{key}[{i}]")
                        for i, item in enumerate(value)]
            elif isinstance(value, str) and value.startswith(self._config.encrypted_prefix):
                try:
                    encrypted_data = value[len(self._config.encrypted_prefix):]
                    # 使用SecurityManager的加密服务进行解密
                    if hasattr(self._security_service, '_encryption_service'):
                        # 如果是SecurityManager，使用其加密服务
                        return self._security_service._encryption_service.decrypt(bytes.fromhex(encrypted_data))
                    elif hasattr(self._security_service, 'decrypt'):
                        # 如果是SecurityService，使用decrypt
                        return self._security_service.decrypt(encrypted_data)
                    return value
                except Exception as e:
                    logger.warning(f"解密失败 {key}: {e}")
                    return value
            else:
                return value

        return _decrypt_value(config, "")

    def encrypt_sensitive_values(self, config: Dict[str, Any],


                                 sensitive_keys: List[str]) -> Dict[str, Any]:
        """加密指定的敏感键值"

        Args:
            config: 原始配置
            sensitive_keys: 敏感键列表

        Returns:
            Dict[str, Any]: 加密后的配置
        """
        encrypted_config = config.copy()

        def _encrypt_sensitive(data: Dict[str, Any], path: str = "") -> None:
            """递归加密敏感值"""
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    _encrypt_sensitive(value, current_path)
                elif isinstance(value, str) and current_path in sensitive_keys:
                    try:
                        # 使用SecurityManager的加密服务
                        if hasattr(self._security_service, '_encryption_service'):
                            # 如果是SecurityManager，使用其加密服务
                            encrypted = self._security_service._encryption_service.encrypt(value)
                            data[key] = f"{self._config.encrypted_prefix}{encrypted.hex()}"
                        elif hasattr(self._security_service, 'encrypt'):
                            # 如果是SecurityService，使用encrypt
                            encrypted = self._security_service.encrypt(value)
                            data[key] = f"{self._config.encrypted_prefix}{encrypted}"
                    except Exception as e:
                        logger.warning(f"加密失败 {current_path}: {e}")

        _encrypt_sensitive(encrypted_config)
        return encrypted_config

    def decrypt_sensitive_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解密所有加密的敏感值"

        Args:
            config: 加密的配置

        Returns:
            Dict[str, Any]: 解密后的配置
        """
        decrypted_config = config.copy()

        def _decrypt_sensitive(data: Dict[str, Any]) -> None:
            """递归解密敏感值"""
            for key, value in data.items():
                if isinstance(value, dict):
                    _decrypt_sensitive(value)
                elif isinstance(value, str) and value.startswith(self._config.encrypted_prefix):
                    try:
                        encrypted_data = value[len(self._config.encrypted_prefix):]
                        data[key] = self._security_service.decrypt(encrypted_data)
                    except Exception as e:
                        logger.warning(f"解密失败 {key}: {e}")

        _decrypt_sensitive(decrypted_config)
        return decrypted_config

    def validate_encrypted_config(self, config: Dict[str, Any]) -> bool:
        """验证加密配置的完整性"

        Args:
            config: 配置字典

        Returns:
            bool: 配置是否有效
        """
        try:
            # 检查是否包含无效的加密值

            def _check_encrypted_values(data: Any) -> bool:
                """递归检查加密值"""
                if isinstance(data, dict):
                    for key, value in data.items():
                        if not _check_encrypted_values(value):
                            return False
                elif isinstance(data, list):
                    for item in data:
                        if not _check_encrypted_values(item):
                            return False
                elif isinstance(data, str) and data.startswith(self._config.encrypted_prefix):
                    try:
                        encrypted_data = data[len(self._config.encrypted_prefix):]
                        # 使用SecurityManager的加密服务进行验证
                        if hasattr(self._security_service, '_encryption_service'):
                            # 如果是SecurityManager，使用其加密服务
                            self._security_service._encryption_service.decrypt(
                                bytes.fromhex(encrypted_data))
                        elif hasattr(self._security_service, 'decrypt'):
                            # 如果是SecurityService，使用decrypt
                            self._security_service.decrypt(encrypted_data)
                    except Exception:
                        return False
                return True

            return _check_encrypted_values(config)
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False

    def get_encryption_stats(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """获取加密统计信息"

        Args:
            config: 配置字典

        Returns:
            Dict[str, Any]: 统计信息
        """
        encrypted_count = 0
        sensitive_count = 0

        def _count_values(data: Any, path: str = "") -> None:

            nonlocal encrypted_count, sensitive_count

            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    _count_values(value, current_path)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    _count_values(item, f"{path}[{i}]")
            elif isinstance(data, str):
                if data.startswith(self._config.encrypted_prefix):
                    encrypted_count += 1
                elif self.is_sensitive_key(path):
                    sensitive_count += 1

        _count_values(config)

        return {
            "encrypted_values": encrypted_count,
            "sensitive_values": sensitive_count,
            "encryption_rate": encrypted_count / max(sensitive_count, 1)
        }

    def export_encrypted_config(self, config: Dict[str, Any],


                                file_path: str) -> bool:
        """导出加密配置到文件"

        Args:
            config: 配置字典
            file_path: 文件路径

        Returns:
            bool: 是否成功
        """
        try:
            encrypted_config = self.encrypt_config(config)
            with open(file_path, 'w', encoding='utf - 8') as f:
                json.dump(encrypted_config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"导出加密配置失败: {e}")
            return False

    def import_encrypted_config(self, file_path: str) -> Optional[Dict[str, Any]]:
        """从文件导入加密配置"

        Args:
            file_path: 文件路径

        Returns:
            Optional[Dict[str, Any]]: 解密后的配置
        """
        try:
            with open(file_path, 'r', encoding='utf - 8') as f:
                encrypted_config = json.load(f)

            return self.decrypt_config(encrypted_config)
        except Exception as e:
            logger.error(f"导入加密配置失败: {e}")
            return None
