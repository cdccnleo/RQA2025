# -*- coding: utf-8 -*-
"""
市场数据适配器模块
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MarketDataAdapter(ABC):
    """市场数据适配器抽象基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化市场数据适配器

        Args:
            config: 适配器配置
        """
        self.config = config or {}
        self.connected = False
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def connect(self) -> bool:
        """连接数据源

        Returns:
            是否连接成功
        """

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接

        Returns:
            是否断开成功
        """

    @abstractmethod
    def get_market_data(self, symbol: str, data_type: str = "realtime") -> Optional[Dict[str, Any]]:
        """获取市场数据

        Args:
            symbol: 股票代码
            data_type: 数据类型

        Returns:
            市场数据
        """

    @abstractmethod
    def subscribe_market_data(self, symbols: List[str], callback: callable) -> bool:
        """订阅市场数据

        Args:
            symbols: 股票代码列表
            callback: 数据回调函数

        Returns:
            是否订阅成功
        """

    @abstractmethod
    def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """取消订阅市场数据

        Args:
            symbols: 股票代码列表

        Returns:
            是否取消成功
        """

    def is_connected(self) -> bool:
        """检查连接状态

        Returns:
            是否已连接
        """
        return self.connected

    def get_adapter_info(self) -> Dict[str, Any]:
        """获取适配器信息

        Returns:
            适配器信息
        """
        return {
            "adapter_name": self.__class__.__name__,
            "connected": self.connected,
            "config": self.config
        }


class BaseMarketDataAdapter(MarketDataAdapter):
    """基础市场数据适配器实现"""

    def connect(self) -> bool:
        """连接数据源"""
        try:
            # 这里应该实现具体的连接逻辑
            self.connected = True
            self.logger.info("市场数据适配器连接成功")
            return True
        except Exception as e:
            self.logger.error(f"市场数据适配器连接失败: {str(e)}")
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        try:
            self.connected = False
            self.logger.info("市场数据适配器断开连接")
            return True
        except Exception as e:
            self.logger.error(f"市场数据适配器断开连接失败: {str(e)}")
            return False

    def get_market_data(self, symbol: str, data_type: str = "realtime") -> Optional[Dict[str, Any]]:
        """获取市场数据"""
        if not self.connected:
            self.logger.error("适配器未连接")
            return None

        try:
            # 这里应该实现具体的数据获取逻辑
            # 暂时返回模拟数据
            return {
                "symbol": symbol,
                "price": 100.0,
                "volume": 10000,
                "timestamp": datetime.now(),
                "data_type": data_type
            }
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {str(e)}")
            return None

    def subscribe_market_data(self, symbols: List[str], callback: callable) -> bool:
        """订阅市场数据"""
        if not self.connected:
            self.logger.error("适配器未连接")
            return False

        try:
            # 这里应该实现具体的订阅逻辑
            self.logger.info(f"订阅市场数据: {symbols}")
            return True
        except Exception as e:
            self.logger.error(f"订阅市场数据失败: {str(e)}")
            return False

    def unsubscribe_market_data(self, symbols: List[str]) -> bool:
        """取消订阅市场数据"""
        try:
            # 这里应该实现具体的取消订阅逻辑
            self.logger.info(f"取消订阅市场数据: {symbols}")
            return True
        except Exception as e:
            self.logger.error(f"取消订阅市场数据失败: {str(e)}")
            return False


class ChinaMarketDataAdapter(BaseMarketDataAdapter):
    """中国市场数据适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化中国市场数据适配器

        Args:
            config: 配置参数
        """
        super().__init__(config)
        self.market_type = "china"
        self.data_providers = self.config.get("data_providers", ["tushare", "akshare"])

    def get_market_data(self, symbol: str, data_type: str = "realtime") -> Optional[Dict[str, Any]]:
        """获取中国市场数据"""
        # 实现中国市场特定的数据获取逻辑
        data = super().get_market_data(symbol, data_type)
        if data:
            data["market"] = "china"
            data["exchange"] = self._get_exchange(symbol)
        return data

    def _get_exchange(self, symbol: str) -> str:
        """根据股票代码判断交易所"""
        if symbol.startswith("00"):
            return "sz"  # 深圳
        elif symbol.startswith("60"):
            return "sh"  # 上海
        elif symbol.startswith("30"):
            return "cy"  # 创业板
        elif symbol.startswith("68"):
            return "kc"  # 科创板
        else:
            return "unknown"


# 创建默认适配器实例
default_market_data_adapter = ChinaMarketDataAdapter()


def get_market_data_adapter() -> MarketDataAdapter:
    """获取默认市场数据适配器

    Returns:
        市场数据适配器实例
    """
    return default_market_data_adapter
