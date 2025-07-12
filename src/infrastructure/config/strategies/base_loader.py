from src.infrastructure.config.interfaces.config_loader import IConfigLoader
from typing import Dict, Any, Tuple

class ConfigLoaderStrategy(IConfigLoader):
    """配置加载策略基类，提供部分默认实现"""
    
    def load(self, source: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """默认加载实现，子类应覆盖"""
        raise NotImplementedError()
        
    def can_load(self, source: str) -> bool:
        """默认支持检查，子类应覆盖"""
        return False
        
    def batch_load(self, sources: list[str]) -> Dict[str, Tuple[Dict, Dict]]:
        """默认批量加载实现，子类可覆盖"""
        return {src: self.load(src) for src in sources}
