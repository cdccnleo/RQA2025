"""
数据适配器基础模块
提供数据适配器的核心接口和基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class AdapterType(Enum):

    """适配器类型枚举"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    BOND = "bond"
    COMMODITY = "commodity"
    NEWS = "news"
    INDEX = "index"
    MACRO = "macro"
    OPTIONS = "options"


@dataclass
class AdapterConfig:

    """适配器配置类"""
    name: str = ""
    adapter_type: str = ""
    timeout: int = 30
    max_retries: int = 3
    connection_params: Dict[str, Any] = None
    validation_rules: Dict[str, Any] = None

    def __post_init__(self):

        if self.connection_params is None:
            self.connection_params = {}
        if self.validation_rules is None:
            self.validation_rules = {}


class BaseAdapter(ABC):

    """基础数据适配器抽象类"""

    def __init__(self, config: AdapterConfig):

        self.config = config
        self._is_connected = False
        self._last_error = None

    @abstractmethod
    def connect(self) -> bool:
        """连接到数据源"""

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""

    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._is_connected

    @abstractmethod
    def get_adapter_info(self) -> Dict[str, Any]:
        """获取适配器信息"""

    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置"""


class GenericAdapter(BaseAdapter):

    """通用适配器实现"""

    def connect(self) -> bool:
        """连接到数据源"""
        try:
            # 模拟连接逻辑
            self._is_connected = True
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        try:
            self._is_connected = False
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._is_connected

    def get_adapter_info(self) -> Dict[str, Any]:
        """获取适配器信息"""
        return {
            'name': self.config.name,
            'type': self.config.adapter_type,
            'is_connected': self._is_connected,
            'timeout': self.config.timeout,
            'max_retries': self.config.max_retries
        }

    def validate_config(self) -> bool:
        """验证配置"""
        return True


class MarketAdapter(BaseAdapter):

    """市场数据适配器"""

    def __init__(self, config: AdapterConfig):

        super().__init__(config)
        self.market_data = {}
        self.subscribed_symbols = set()

    def connect(self) -> bool:
        """连接到市场数据源"""
        try:
            self._is_connected = True
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def disconnect(self) -> bool:
        """断开市场数据连接"""
        try:
            self._is_connected = False
            self.subscribed_symbols.clear()
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def is_connected(self) -> bool:
        """检查市场连接状态"""
        return self._is_connected

    def get_adapter_info(self) -> Dict[str, Any]:
        """获取适配器信息"""
        return {
            'name': self.config.name,
            'type': 'market',
            'is_connected': self._is_connected,
            'subscribed_symbols': list(self.subscribed_symbols),
            'market_data_count': len(self.market_data)
        }

    def validate_config(self) -> bool:
        """验证配置"""
        return True

    def subscribe_symbol(self, symbol: str) -> bool:
        """订阅交易对"""
        if self._is_connected:
            self.subscribed_symbols.add(symbol)
            return True
        return False

    def unsubscribe_symbol(self, symbol: str) -> bool:
        """取消订阅交易对"""
        if symbol in self.subscribed_symbols:
            self.subscribed_symbols.remove(symbol)
            return True
        return False

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取市场数据"""
        return self.market_data.get(symbol)
