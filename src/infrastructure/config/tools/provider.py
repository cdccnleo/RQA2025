
from abc import ABC, abstractmethod
from typing import Dict, Any
import json
import logging
import os

logger = logging.getLogger(__name__)

# 为了避免导入错误，我们定义一个简单的接口
class IConfigProvider:
    """配置提供者接口"""
    @abstractmethod
    def load(self, source: str) -> Dict[str, Any]:
        """加载配置"""
        pass

    @abstractmethod
    def save(self, config: Dict[str, Any], destination: str) -> bool:
        """保存配置"""
        pass

    @abstractmethod
    def get_default(self) -> Dict[str, Any]:
        """获取默认配置"""
        pass


class ConfigProvider(IConfigProvider, ABC):

    """
provider - 配置管理

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
"""

    @abstractmethod
    def load(self, source: str) -> Dict[str, Any]:
        """加载配置"""

    @abstractmethod
    def save(self, config: Dict[str, Any], destination: str) -> bool:
        """保存配置"""

    @abstractmethod
    def get_default(self) -> Dict[str, Any]:
        """获取默认配置"""


class DefaultConfigProvider(ConfigProvider):

    """默认配置提供者"""

    def __init__(self):

        self._logger = logger
        self._default_config = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "rqa_db"
            },
            "cache": {
                "enabled": False,
                "size": 1000
            }
        }

    def load(self, source: str) -> Dict[str, Any]:
        """加载配置

        Args:
            source: 配置源（文件路径或环境变量前缀）

        Returns:
            Dict[str, Any]: 配置字典
        """
        if source.startswith('env:'):
            # 从环境变量加载
            prefix = source[4:]
            return self._load_from_env(prefix)
        elif os.path.exists(source):
            # 从文件加载
            return self._load_from_file(source)
        else:
            self._logger.warning(f"Config source not found: {source}, using default config")
            return self.get_default()

    def save(self, config: Dict[str, Any], destination: str) -> bool:
        """保存配置"
        Args:
            config: 配置字典
            destination: 目标路径
        Returns:
            bool: 是否保存成功
        """
        try:
            with open(destination, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self._logger.error(f"Failed to save config to {destination}: {str(e)}")
            return False

    def get_default(self) -> Dict[str, Any]:
        """获取默认配置"""
        return self._default_config.copy()

    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """从文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self._logger.error(f"Failed to load config from {file_path}: {str(e)}")
            return self.get_default()

    def _load_from_env(self, prefix: str) -> Dict[str, Any]:
        """从环境变量加载配置"""
        config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                config[config_key] = value
        return config

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置（兼容性实现）"""
        # 从默认配置中获取
        keys = key.split('.')
        config = self._default_config

        for k in keys:
            if isinstance(config, dict) and k in config:
                config = config[k]
            else:
                return default

        return config

    def set_config(self, key: str, value: Any) -> bool:
        """设置配置（兼容性实现）"""
        try:
            keys = key.split('.')
            config = self._default_config

            # 遍历到最后一个键
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            # 设置最后一个键的值
            config[keys[-1]] = value
            return True
        except Exception as e:
            self._logger.error(f"Failed to set config {key}: {str(e)}")
            return False

    def load_config(self, source: str) -> bool:
        """加载配置（兼容性实现）"""
        try:
            loaded_config = self.load(source)
            self._default_config.update(loaded_config)
            return True
        except Exception as e:
            self._logger.error(f"Failed to load config from {source}: {str(e)}")
            return False

    def save_config(self, destination: str) -> bool:
        """保存配置（兼容性实现）"""
        return self.save(self._default_config, destination)




