from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAdapter(ABC):
    """统一的数据适配器基类"""

    @abstractmethod
    def load(self, source: str) -> Dict[str, Any]:
        """加载原始数据"""
        pass

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """验证数据格式"""
        pass

class GenericAdapter(BaseAdapter):
    """通用适配器扩展"""

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """通用数据转换方法"""
        return {
            'metadata': data.get('meta', {}),
            'payload': data.get('data', {})
        }

class MarketAdapter(BaseAdapter):
    """市场特定适配器基类"""

    @abstractmethod
    def get_market(self) -> str:
        """获取市场标识"""
        pass
