"""
数据层 - 数据加载器接口定义

提供数据加载器的统一接口和基础实现。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from datetime import datetime


class IDataLoader(ABC):
    """数据加载器接口"""

    @abstractmethod
    def load_data(self, symbols: List[str], start_date: datetime,
                  end_date: datetime, **kwargs) -> Dict[str, Any]:
        """加载数据"""

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """获取可用符号列表"""

    @abstractmethod
    def get_data_info(self, symbol: str) -> Dict[str, Any]:
        """获取数据信息"""


class IMarketDataLoader(IDataLoader):
    """市场数据加载器接口"""

    @abstractmethod
    def load_market_data(self, symbols: List[str], data_type: str = "price",
                         **kwargs) -> Dict[str, Any]:
        """加载市场数据"""


# 基础实现类
class BaseDataLoader(IDataLoader):
    """基础数据加载器实现"""

    def __init__(self, name: str):
        self.name = name
        self.is_connected = False

    def load_data(self, symbols: List[str], start_date: datetime,
                  end_date: datetime, **kwargs) -> Dict[str, Any]:
        """基础数据加载实现"""
        return {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "data": {},
            "status": "success"
        }

    def get_available_symbols(self) -> List[str]:
        """获取可用符号"""
        return []

    def get_data_info(self, symbol: str) -> Dict[str, Any]:
        """获取数据信息"""
        return {"symbol": symbol, "available": False}


# 具体的加载器实现
class StockDataLoader(BaseDataLoader):
    """股票数据加载器"""


class CryptoDataLoader(BaseDataLoader):
    """加密货币数据加载器"""


class ForexDataLoader(BaseDataLoader):
    """外汇数据加载器"""


class BondDataLoader(BaseDataLoader):
    """债券数据加载器"""


class OptionsDataLoader(BaseDataLoader):
    """期权数据加载器"""


class MacroDataLoader(BaseDataLoader):
    """宏观经济数据加载器"""


class CommodityDataLoader(BaseDataLoader):
    """商品数据加载器"""


class IndexDataLoader(BaseDataLoader):
    """指数数据加载器"""


# 工厂函数
def get_data_loader(data_type: str) -> IDataLoader:
    """获取数据加载器"""
    loaders = {
        "stock": StockDataLoader,
        "crypto": CryptoDataLoader,
        "forex": ForexDataLoader,
        "bond": BondDataLoader,
        "options": OptionsDataLoader,
        "macro": MacroDataLoader,
        "commodity": CommodityDataLoader,
        "index": IndexDataLoader,
    }

    loader_class = loaders.get(data_type.lower())
    if loader_class:
        return loader_class(f"{data_type}_loader")

    # 默认返回基础加载器
    return BaseDataLoader("default_loader")
