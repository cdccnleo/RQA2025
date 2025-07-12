from abc import ABC, abstractmethod
from typing import Dict, Any

class ConfigLoaderStrategy(ABC):
    """配置加载策略抽象基类"""
    @abstractmethod
    def load(self, source: str) -> Dict[str, Any]:
        """加载配置源

        Args:
            source: 配置源标识(文件路径/环境变量前缀等)

        Returns:
            解析后的配置字典
        """
        pass

    @abstractmethod
    def can_load(self, source: str) -> bool:
        """判断是否支持该配置源

        Args:
            source: 配置源标识

        Returns:
            bool: 是否支持加载
        """
        pass
