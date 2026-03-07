"""
RQA2025基础适配器模块

提供适配器的基础类和接口定义
"""

from typing import Any, Dict, List, Optional
import logging
import os
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class SecureConfigManager:

    """安全配置管理器 - 加密存储敏感信息"""

    def __init__(self, key_file: str = None):
        """
        初始化安全配置管理器

        Args:
            key_file: 密钥文件路径，默认为None时自动创建
        """
        if key_file is None:
            # 默认密钥文件路径
            self.key_file = os.path.join(os.path.dirname(__file__), '.encryption_key')
        else:
            self.key_file = key_file

        self.logger = logging.getLogger(self.__class__.__name__)
        self.encryption_key = self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)

    def _get_or_create_key(self) -> bytes:
        """获取或创建加密密钥"""
        try:
            if os.path.exists(self.key_file):
                with open(self.key_file, 'rb') as f:
                    key = f.read()
                self.logger.info("加载现有加密密钥")
                return key
            else:
                # 生成新密钥
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                self.logger.info("创建新的加密密钥")
                return key
        except Exception as e:
            self.logger.error(f"密钥管理失败: {e}")
            # 如果密钥文件操作失败，使用系统生成的密钥
            return Fernet.generate_key()

    def encrypt_sensitive_data(self, data: str) -> str:
        """
        加密敏感数据

        Args:
            data: 待加密的明文数据

        Returns:
            str: 加密后的数据（base64编码）
        """
        try:
            if not data:
                return ""
            encrypted_data = self.cipher.encrypt(data.encode())
            return encrypted_data.decode()
        except Exception as e:
            self.logger.error(f"数据加密失败: {e}")

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        解密敏感数据

        Args:
            encrypted_data: 加密的数据（base64编码）

        Returns:
            str: 解密后的明文数据
        """
        try:
            if not encrypted_data:
                return ""
            decrypted_data = self.cipher.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"数据解密失败: {e}")
            raise

    def secure_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        对配置中的敏感信息进行安全处理

        Args:
            config: 原始配置字典

        Returns:
            Dict[str, Any]: 安全处理后的配置字典
        """
        secure_config = config.copy()
        sensitive_keys = ['password', 'token',
                          'secret', 'key', 'api_key', 'access_token']

        for key, value in secure_config.items():
            # 检查是否为敏感信息
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                if isinstance(value, str) and not self._is_encrypted(value):
                    try:
                        secure_config[key] = self.encrypt_sensitive_data(value)
                        self.logger.info(f"已加密敏感配置项: {key}")
                    except Exception as e:
                        self.logger.warning(f"无法加密配置项 {key}: {e}")
            elif isinstance(value, dict):
                # 递归处理嵌套配置
                secure_config[key] = self.secure_config(value)

        return secure_config

    def unsecure_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        对配置中的敏感信息进行解密处理

        Args:
            config: 加密的配置字典

        Returns:
            Dict[str, Any]: 解密后的配置字典
        """
        unsecure_config = config.copy()
        sensitive_keys = ['password', 'token',
                          'secret', 'key', 'api_key', 'access_token']

        for key, value in unsecure_config.items():
            # 检查是否为敏感信息
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                if isinstance(value, str) and self._is_encrypted(value):
                    try:
                        unsecure_config[key] = self.decrypt_sensitive_data(value)
                        self.logger.info(f"已解密敏感配置项: {key}")
                    except Exception as e:
                        self.logger.warning(f"无法解密配置项 {key}: {e}")
            elif isinstance(value, dict):
                # 递归处理嵌套配置
                unsecure_config[key] = self.unsecure_config(value)

        return unsecure_config

    def _is_encrypted(self, data: str) -> bool:
        """
        检查数据是否已加密

        Args:
            data: 待检查的数据

        Returns:
            bool: 是否已加密
        """
        try:
            # 尝试解密，如果成功则说明是加密数据
            self.cipher.decrypt(data.encode())
            return True
        except BaseException:
            return False

    def rotate_key(self) -> bool:
        """
        轮换加密密钥

        Returns:
            bool: 轮换是否成功
        """
        try:
            # 生成新密钥
            new_key = Fernet.generate_key()
            new_cipher = Fernet(new_key)

            # 读取所有需要重新加密的配置
            # 这里可以扩展为从配置文件或数据库中读取
            config_files = self._find_config_files()

            for config_file in config_files:
                if os.path.exists(config_file):
                    self._re_encrypt_config_file(config_file, self.cipher, new_cipher)

            # 保存新密钥
            with open(self.key_file, 'wb') as f:
                f.write(new_key)

            self.encryption_key = new_key
            self.cipher = new_cipher

            self.logger.info("加密密钥轮换完成")
            return True

        except Exception as e:
            self.logger.error(f"密钥轮换失败: {e}")
            return False

    def _find_config_files(self) -> List[str]:
        """查找需要重新加密的配置文件"""
        config_files = []
        adapters_dir = os.path.dirname(__file__)

        # 查找可能的配置文件
        for root, dirs, files in os.walk(adapters_dir):
            for file in files:
                if file.endswith(('.json', '.yaml', '.yml', '.cfg', '.ini')):
                    config_files.append(os.path.join(root, file))

        return config_files

    def _re_encrypt_config_file(self, file_path: str, old_cipher: Fernet, new_cipher: Fernet):
        """重新加密配置文件"""
        try:
            import json
            import yaml

            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf - 8') as f:
                    config = json.load(f)

                # 重新加密配置
                re_encrypted_config = self._re_encrypt_config(
                    config, old_cipher, new_cipher)

                with open(file_path, 'w', encoding='utf - 8') as f:
                    json.dump(re_encrypted_config, f, indent=2, ensure_ascii=False)

            elif file_path.endswith(('.yaml', '.yml')):
                with open(file_path, 'r', encoding='utf - 8') as f:
                    config = yaml.safe_load(f)

                # 重新加密配置
                re_encrypted_config = self._re_encrypt_config(
                    config, old_cipher, new_cipher)

                with open(file_path, 'w', encoding='utf - 8') as f:
                    yaml.dump(re_encrypted_config, f,
                              default_flow_style=False, allow_unicode=True)

        except Exception as e:
            self.logger.warning(f"重新加密配置文件失败 {file_path}: {e}")

    def _re_encrypt_config(self, config: Dict[str, Any], old_cipher: Fernet, new_cipher: Fernet) -> Dict[str, Any]:
        """重新加密配置字典"""
        if not isinstance(config, dict):
            return config

        re_encrypted = {}
        sensitive_keys = ['password', 'token',
                          'secret', 'key', 'api_key', 'access_token']

        for key, value in config.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                if isinstance(value, str):
                    try:
                        # 先解密旧数据，再用新密钥加密
                        if old_cipher.decrypt(value.encode()):
                            plain_text = old_cipher.decrypt(value.encode()).decode()
                            re_encrypted[key] = new_cipher.encrypt(
                                plain_text.encode()).decode()
                    except BaseException:
                        # 如果解密失败，直接用新密钥加密
                        re_encrypted[key] = new_cipher.encrypt(value.encode()).decode()
                else:
                    re_encrypted[key] = value
            elif isinstance(value, dict):
                re_encrypted[key] = self._re_encrypt_config(
                    value, old_cipher, new_cipher)
            else:
                re_encrypted[key] = value

        return re_encrypted


class BaseAdapter:

    """基础适配器类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        self.config = config or {}
        self.is_connected = False
        self.logger = logging.getLogger(self.__class__.__name__)

        # 初始化安全配置管理器
        self.secure_config_manager = SecureConfigManager()

        # 对敏感配置进行安全处理
        self.config = self._secure_config_initialization(self.config)

    def _secure_config_initialization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        配置安全初始化处理

        Args:
            config: 原始配置字典

        Returns:
            Dict[str, Any]: 安全处理后的配置字典
        """
        try:
            # 对配置进行安全处理（加密敏感信息）
            secure_config = self.secure_config_manager.secure_config(config)

            # 存储安全配置用于后续使用
            self._secure_config = secure_config

            # 为运行时使用准备解密配置
            self._runtime_config = self.secure_config_manager.unsecure_config(
                secure_config)

            return self._runtime_config

        except Exception as e:
            self.logger.warning(f"配置安全初始化失败，使用原始配置: {e}")
            self._secure_config = config.copy()
            self._runtime_config = config.copy()
            return config

    def get_secure_config(self) -> Dict[str, Any]:
        """
        获取安全配置（加密状态）

        Returns:
            Dict[str, Any]: 加密后的配置字典
        """
        return getattr(self, '_secure_config', self.config.copy())

    def get_runtime_config(self) -> Dict[str, Any]:
        """
        获取运行时配置（解密状态）

        Returns:
            Dict[str, Any]: 解密后的配置字典
        """
        return getattr(self, '_runtime_config', self.config.copy())

    def update_secure_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新安全配置

        Args:
            new_config: 新的配置字典

        Returns:
            bool: 更新是否成功
        """
        try:
            # 对新配置进行安全处理
            secure_config = self.secure_config_manager.secure_config(new_config)

            # 更新配置
            self._secure_config = secure_config
            self._runtime_config = self.secure_config_manager.unsecure_config(
                secure_config)
            self.config = self._runtime_config

            self.logger.info("安全配置更新成功")
            return True

        except Exception as e:
            self.logger.error(f"安全配置更新失败: {e}")
            return False

    def connect(self) -> bool:
        """连接到数据源"""
        raise NotImplementedError

    def disconnect(self) -> bool:
        """断开数据源连接"""
        raise NotImplementedError

    def get_data(self, **kwargs) -> Any:
        """从数据源获取数据"""
        raise NotImplementedError

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy" if self.is_connected else "disconnected",
            "adapter_type": self.__class__.__name__
        }


class DataAdapter(BaseAdapter):

    """数据适配器基类"""

    def validate_data(self, data: Any) -> bool:
        """验证数据"""
        return data is not None

    def transform_data(self, data: Any) -> Any:
        """转换数据"""
        return data


class MockAdapter(BaseAdapter):

    """模拟适配器（用于测试）"""

    def connect(self) -> bool:
        """connect 函数的文档字符串"""

        self.is_connected = True
        return True

    def disconnect(self) -> bool:
        """disconnect 函数的文档字符串"""

        self.is_connected = False
        return True

    def get_data(self, **kwargs) -> Dict[str, Any]:
        """get_data 函数的文档字符串"""

        return {
            "mock": True,
            "timestamp": "2024 - 01 - 01T00:00:00Z",
            "data": "mock_data"
        }
