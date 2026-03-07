
import base64
import time
from contextlib import contextmanager

from cryptography.fernet import Fernet
from pathlib import Path
import json
import logging
import os
import msvcrt
from typing import Dict, Any
"""
安全的邮件配置管理模块
支持环境变量和加密存储，确保敏感信息安全
"""

logger = logging.getLogger(__name__)


class SecureEmailConfig:

    """
secure_config - 配置管理

职责说明：
负责系统配置的统一管理、配置文件的读取、配置验证和配置分发

核心职责：
- 配置文件的读取和解析
- 配置参数的验证
- 配置的热重载
- 配置的分发和同步
- 环境变量管理
- 配置加密和安全

相关接口：
- IConfigComponent
- IConfigManager
- IConfigValidator
""" """安全的邮件配置管理器"""

    def __init__(self, config_path: str = "config/email_config.json"):

        self.config_path = Path(config_path)
        self._key = None
        self._cipher = None
        key_path_env = os.getenv('EMAIL_ENCRYPTION_KEY_FILE')
        self._key_file = Path(key_path_env) if key_path_env else Path('config/.email_key')

    def _get_encryption_key(self) -> bytes:
        """获取或生成加密密钥"""
        if self._key is None:
            # 优先从环境变量获取密钥
            key_env = os.getenv('EMAIL_ENCRYPTION_KEY')
            if key_env:
                self._key = base64.urlsafe_b64decode(key_env)
            else:
                # 从文件获取或生成新密钥
                with self._locked_key_file():
                    key_file = self._key_file
                    if key_file.exists():
                        self._key = key_file.read_bytes()
                    else:
                        # 生成新密钥
                        self._key = Fernet.generate_key()
                        self._write_key_file(key_file, self._key, exclusive=True)
                        logger.warning(f"生成新的加密密钥: {key_file}")

                    if not self._is_valid_key(self._key):
                        # 若密钥无效，重新生成
                        self._key = Fernet.generate_key()
                        self._write_key_file(key_file, self._key, exclusive=False)
        return self._key

    def _is_valid_key(self, key: Any) -> bool:
        return isinstance(key, (bytes, bytearray)) and len(key) == 44

    @contextmanager
    def _locked_key_file(self):
        lock_path = self._key_file.with_suffix(self._key_file.suffix + '.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, 'w+b') as lock_file:
            lock_file.write(b'0')
            lock_file.flush()
            os.fsync(lock_file.fileno())
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    def _write_key_file(self, key_file: Path, key_bytes: bytes, exclusive: bool) -> None:
        key_file.parent.mkdir(parents=True, exist_ok=True)

        if exclusive:
            if key_file.exists():
                return
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            try:
                fd = os.open(str(key_file), flags)
            except FileExistsError:
                return
            with os.fdopen(fd, 'wb') as f:
                f.write(key_bytes)
                f.flush()
                os.fsync(f.fileno())
            return

        for _ in range(3):
            try:
                with open(key_file, 'wb') as f:
                    f.write(key_bytes)
                    f.flush()
                    os.fsync(f.fileno())
                return
            except PermissionError:
                time.sleep(0.05)

    def _get_cipher(self) -> Fernet:
        """获取加密器"""
        if self._cipher is None:
            key = self._get_encryption_key()
            self._cipher = Fernet(key)
        return self._cipher

    def _encrypt_value(self, value: str) -> str:
        """加密敏感值"""
        cipher = self._get_cipher()
        return base64.urlsafe_b64encode(cipher.encrypt(value.encode())).decode()

    def _decrypt_value(self, encrypted_value: str) -> str:
        """解密敏感值"""
        cipher = self._get_cipher()
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
        return cipher.decrypt(encrypted_bytes).decode()

    def load_config(self) -> Dict:
        """加载邮件配置，支持环境变量替换"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"邮件配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 替换环境变量
        resolved_config = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]  # 移除 ${ 和 }
                env_value = os.getenv(env_var)
                if env_value is None:
                    logger.warning(f"环境变量未设置: {env_var}")
                    resolved_config[key] = ""
                else:
                    # 特殊处理端口号
                    if key == "smtp_port":
                        try:
                            resolved_config[key] = int(env_value)
                        except ValueError:
                            logger.warning(f"端口号格式错误: {env_value}，使用默认值25")
                            resolved_config[key] = 25
                    else:
                        resolved_config[key] = env_value
            elif isinstance(value, list):
                # 处理收件人列表
                resolved_list = []
                for item in value:
                    if isinstance(item, str) and item.startswith('${') and item.endswith('}'):
                        env_var = item[2:-1]
                        env_value = os.getenv(env_var)
                        if env_value:
                            # 支持逗号分隔的多个邮箱
                            emails = [email.strip()
                                      for email in env_value.split(',') if email.strip()]
                            resolved_list.extend(emails)
                    else:
                        resolved_list.append(item)
                resolved_config[key] = resolved_list
            else:
                resolved_config[key] = value

        return resolved_config

    def save_encrypted_config(self, config: Dict[str, Any], output_path: str = "config/email_config.encrypted.json"):
        """保存加密的配置文件"""
        encrypted_config = {}
        sensitive_keys = ['username', 'password', 'from_email', 'to_emails']

        for key, value in config.items():
            if key in sensitive_keys:
                if isinstance(value, list):
                    encrypted_config[key] = [self._encrypt_value(str(item)) for item in value]
                else:
                    encrypted_config[key] = self._encrypt_value(str(value))
            else:
                encrypted_config[key] = value

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(encrypted_config, f, indent=2, ensure_ascii=False)

        logger.info(f"加密配置文件已保存: {output_path}")

    def load_encrypted_config(self, config_path: str = "config/email_config.encrypted.json") -> Dict:
        """加载加密的配置文件"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"加密配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            encrypted_config = json.load(f)

        decrypted_config = {}
        sensitive_keys = ['username', 'password', 'from_email', 'to_emails']

        for key, value in encrypted_config.items():
            if key in sensitive_keys:
                if isinstance(value, list):
                    decrypted_config[key] = [self._decrypt_value(item) for item in value]
                else:
                    decrypted_config[key] = self._decrypt_value(value)
            else:
                decrypted_config[key] = value

        return decrypted_config

    def validate_config(self, config: Dict) -> bool:
        """验证配置完整性"""
        required_fields = ['smtp_server', 'smtp_port', 'username', 'password', 'from_email']

        for field in required_fields:
            if field not in config or not config[field]:
                logger.error(f"缺少必需的配置字段: {field}")
                return False

        if 'to_emails' not in config or not config['to_emails']:
            logger.error("缺少收件人配置")
            return False

        return True


def get_email_config() -> Dict:
    """获取邮件配置的便捷函数"""
    config_manager = SecureEmailConfig()

    # 尝试加载加密配置
    try:
        config = config_manager.load_encrypted_config()
        if config_manager.validate_config(config):
            return config
    except FileNotFoundError:
        pass

    # 尝试加载环境变量配置
    try:
        config = config_manager.load_config()
        if config_manager.validate_config(config):
            return config
    except Exception as e:
        logger.error(f"加载邮件配置失败: {e}")

    raise ValueError("无法加载有效的邮件配置")


class SecureConfig:
    """通用安全配置管理器"""

    def __init__(self, config_path: str = "config/secure_config.json"):
        self.config_path = Path(config_path)
        self._key = None
        self._cipher = None
        self._config_cache = {}

    def initialize(self) -> bool:
        """初始化安全配置管理器"""
        try:
            self._get_encryption_key()
            return True
        except Exception as e:
            logger.error(f"初始化安全配置失败: {e}")
            return False

    def encrypt_value(self, value: str) -> str:
        """加密配置值"""
        if self._cipher is None:
            self._get_encryption_key()

        if isinstance(value, str):
            encrypted = self._cipher.encrypt(value.encode())
            return base64.b64encode(encrypted).decode()
        return value

    def decrypt_value(self, encrypted_value: str) -> str:
        """解密配置值"""
        if self._cipher is None:
            self._get_encryption_key()
            # 确保cipher被初始化
            if self._cipher is None and self._key is not None:
                self._cipher = Fernet(self._key)

        try:
            encrypted = base64.b64decode(encrypted_value)
            decrypted = self._cipher.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"解密失败: {e}")
            return encrypted_value

    def get_secure_value(self, key: str, default: str = "") -> str:
        """获取安全配置值"""
        if key in self._config_cache:
            return self._config_cache[key]

        # 从环境变量获取
        env_value = os.getenv(f"SECURE_{key.upper()}")
        if env_value:
            self._config_cache[key] = self.decrypt_value(env_value)
            return self._config_cache[key]

        return default

    def set_secure_value(self, key: str, value: str) -> bool:
        """设置安全配置值"""
        try:
            encrypted_value = self.encrypt_value(value)
            os.environ[f"SECURE_{key.upper()}"] = encrypted_value
            self._config_cache[key] = value
            return True
        except Exception as e:
            logger.error(f"设置安全配置失败: {e}")
            return False

    def _get_encryption_key(self) -> bytes:
        """获取或生成加密密钥"""
        if self._key is None:
            # 优先从环境变量获取
            key_env = os.getenv("CONFIG_ENCRYPTION_KEY")
            if key_env:
                self._key = base64.b64decode(key_env)
            else:
                # 生成新密钥
                self._key = Fernet.generate_key()

            self._cipher = Fernet(self._key)

        return self._key

    def get_status(self) -> Dict:
        """获取安全配置状态"""
        return {
            'initialized': self._cipher is not None,
            'cached_values': len(self._config_cache),
            'config_path': str(self.config_path),
            'encryption_enabled': True
        }


if __name__ == "__main__":
    # 测试配置加载
    try:
        config = get_email_config()
        print("邮件配置加载成功:")
        for key, value in config.items():
            if key in ['password']:
                print(f"  {key}: {'*' * len(str(value))}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"配置加载失败: {e}")




