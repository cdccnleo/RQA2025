from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

class IConfigLoader(ABC):
    """配置加载器核心接口，定义统一加载规范"""

    @abstractmethod
    def load(self, source: str) -> Tuple[Dict, Dict]:
        """从指定源加载配置
        
        Args:
            source: 配置源标识符
            
        Returns:
            Tuple[配置数据, 元数据]
            
        Raises:
            ConfigLoadError: 当加载失败时抛出
        """
        pass
        
    @abstractmethod
    def can_load(self, source: str) -> bool:
        """检查是否支持加载该配置源
        
        Args:
            source: 配置源标识符
            
        Returns:
            bool: 是否支持加载
        """
        pass

    @abstractmethod
    def batch_load(self, sources: list[str]) -> Dict[str, Tuple[Dict, Dict]]:
        """批量加载多个配置源
        
        Args:
            sources: 配置源列表
            
        Returns:
            按源标识符索引的(配置数据, 元数据)元组
        """
        pass
