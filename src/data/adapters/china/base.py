"""
中国市场数据适配器基类

职责定位：
1. 定义中国市场数据适配器的基础接口
2. 提供通用功能和错误处理
3. 支持A股、科创板等市场类型
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class BaseChinaAdapter(ABC):
    """
    中国市场数据适配器基类

    提供中国市场数据适配器的通用功能和接口定义。
    所有中国市场适配器都应继承此类。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化适配器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_connected = False

        # 初始化连接配置
        self._init_connection_config()

    def _init_connection_config(self):
        """初始化连接配置"""
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 3306)
        self.username = self.config.get('username', '')
        self.password = self.config.get('password', '')
        self.database = self.config.get('database', '')

        # 连接池配置
        self.pool_size = self.config.get('pool_size', 10)
        self.max_overflow = self.config.get('max_overflow', 20)
        self.pool_timeout = self.config.get('pool_timeout', 30)

        # 重试配置
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)

    @abstractmethod
    def connect(self) -> bool:
        """
        连接数据源

        Returns:
            bool: 连接是否成功
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开连接

        Returns:
            bool: 断开连接是否成功
        """
        pass

    @abstractmethod
    def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        获取数据

        Args:
            symbol: 股票代码
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 数据字典
        """
        pass

    def is_connected(self) -> bool:
        """
        检查连接状态

        Returns:
            bool: 是否已连接
        """
        return self._is_connected

    def ping(self) -> bool:
        """
        ping测试连接

        Returns:
            bool: 连接是否正常
        """
        try:
            # 基础的连接测试
            if not self._is_connected:
                return False
            # 子类可以重写此方法进行实际的ping测试
            return True
        except Exception as e:
            self.logger.error(f"Ping测试失败: {e}")
            return False

    def validate_symbol(self, symbol: str) -> bool:
        """
        验证股票代码格式

        Args:
            symbol: 股票代码

        Returns:
            bool: 代码格式是否有效
        """
        if not symbol or not isinstance(symbol, str):
            return False

        # 中国A股代码格式验证
        # A股：6位数字，以0,3,6开头
        # 科创板：以688开头
        # 北交所：以4,8开头
        if len(symbol) != 6:
            return False

        try:
            int(symbol)
        except ValueError:
            return False

        # 检查开头数字
        first_digit = int(symbol[0])
        if first_digit not in [0, 3, 4, 6, 8]:
            return False

        return True

    def format_symbol(self, symbol: str) -> str:
        """
        格式化股票代码

        Args:
            symbol: 原始股票代码

        Returns:
            str: 格式化后的股票代码
        """
        if not symbol:
            return ""

        # 移除可能的.SH或.SZ后缀
        symbol = symbol.replace('.SH', '').replace('.SZ', '')

        # 补齐6位数字
        if len(symbol) < 6:
            symbol = symbol.zfill(6)

        return symbol

    def get_market_type(self, symbol: str) -> str:
        """
        获取市场类型

        Args:
            symbol: 股票代码

        Returns:
            str: 市场类型
        """
        if not symbol or len(symbol) != 6:
            return "UNKNOWN"

        first_digit = int(symbol[0])

        if first_digit == 0:
            return "SH"  # 上海主板
        elif first_digit == 3:
            return "SZ"  # 深圳主板
        elif first_digit == 6:
            if symbol.startswith("688"):
                return "STAR"  # 科创板
            else:
                return "SH"  # 上海主板
        elif first_digit == 4 or first_digit == 8:
            return "BJ"  # 北京交易所
        else:
            return "UNKNOWN"

    def get_exchange_suffix(self, symbol: str) -> str:
        """
        获取交易所后缀

        Args:
            symbol: 股票代码

        Returns:
            str: 交易所后缀(.SH或.SZ)
        """
        market_type = self.get_market_type(symbol)

        if market_type in ["SH", "STAR"]:
            return ".SH"
        elif market_type == "SZ":
            return ".SZ"
        elif market_type == "BJ":
            return ".BJ"
        else:
            return ""

    def retry_operation(self, operation, *args, **kwargs):
        """
        重试操作

        Args:
            operation: 要重试的操作函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Any: 操作结果
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"操作失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避

        raise last_exception

    def get_status(self) -> Dict[str, Any]:
        """
        获取适配器状态

        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            "connected": self._is_connected,
            "config": {
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "pool_size": self.pool_size
            },
            "timestamp": datetime.now().isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            connected = self.ping()
            return {
                "status": "healthy" if connected else "unhealthy",
                "connected": connected,
                "timestamp": datetime.now().isoformat(),
                "details": self.get_status()
            }
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
