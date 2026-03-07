"""
RQA2025系统适配器模式实现 - 适配器层核心组件
Adapter Pattern Implementation for RQA2025 System - Adapter Layer Core Component

基于适配器层架构设计，实现适配器模式，用于适配不同接口和实现之间的差异
符合docs\architecture\adapter_layer_architecture_design.md规范
"""

from typing import Any, Dict, List, Type, TypeVar, Callable
from abc import ABC, abstractmethod
import logging
import asyncio
import json
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')

# ==================== 适配器层核心接口 ====================


class BaseAdapter(ABC):
    """
    适配器层基础适配器接口

    所有适配器层的适配器都必须实现的接口
    """

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """适配器ID"""

    @property
    @abstractmethod
    def adapter_type(self) -> str:
        """适配器类型"""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化适配器

        Args:
            config: 配置信息

        Returns:
            初始化是否成功
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """关闭适配器"""

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            健康状态信息
        """

    @abstractmethod
    def get_adapter_info(self) -> Dict[str, Any]:
        """
        获取适配器信息

        Returns:
            适配器信息
        """


class DataSourceAdapter(BaseAdapter, ABC):
    """
    数据源适配器接口

    数据源适配子系统的适配器接口
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        连接到数据源

        Returns:
            连接是否成功
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""

    @abstractmethod
    async def fetch_data(self, **kwargs) -> Any:
        """
        获取数据

        Args:
            **kwargs: 获取参数

        Returns:
            获取的数据
        """

    @abstractmethod
    async def subscribe_data(self, callback: Callable, **kwargs) -> str:
        """
        订阅数据

        Args:
            callback: 数据回调函数
            **kwargs: 订阅参数

        Returns:
            订阅ID
        """

    @abstractmethod
    async def unsubscribe_data(self, subscription_id: str) -> bool:
        """
        取消订阅

        Args:
            subscription_id: 订阅ID

        Returns:
            取消是否成功
        """


class ProtocolAdapter(BaseAdapter, ABC):
    """
    协议适配器接口

    协议转换子系统的适配器接口
    """

    @abstractmethod
    async def convert_protocol(self, data: Any, from_protocol: str, to_protocol: str) -> Any:
        """
        协议转换

        Args:
            data: 要转换的数据
            from_protocol: 源协议
            to_protocol: 目标协议

        Returns:
            转换后的数据
        """

    @abstractmethod
    def supports_protocols(self) -> List[str]:
        """
        获取支持的协议列表

        Returns:
            协议列表
        """


class ConnectionAdapter(BaseAdapter, ABC):
    """
    连接适配器接口

    连接管理子系统的适配器接口
    """

    @abstractmethod
    async def get_connection(self, endpoint: str) -> Any:
        """
        获取连接

        Args:
            endpoint: 端点

        Returns:
            连接对象
        """

    @abstractmethod
    async def release_connection(self, connection: Any) -> None:
        """
        释放连接

        Args:
            connection: 连接对象
        """

    @abstractmethod
    async def get_connection_stats(self) -> Dict[str, Any]:
        """
        获取连接统计信息

        Returns:
            统计信息
        """


# ==================== 适配器层核心实现 ====================

class MarketDataAdapter(DataSourceAdapter):
    """
    市场数据适配器

    适配不同市场数据源到统一的接口
    """

    def __init__(self, adapter_id: str, market_type: str):
        self._adapter_id = adapter_id
        self._market_type = market_type
        self._config = {}
        self._connection_manager = None
        self._data_converter = None
        self._cache_manager = None
        self._is_connected = False
        self._subscriptions = {}

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_type(self) -> str:
        return "market_data"

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化市场数据适配器"""
        try:
            self._config = config
            # 初始化连接管理器
            self._connection_manager = ConnectionManager(config.get('connection', {}))
            # 初始化数据转换器
            self._data_converter = DataConverter(config.get('data_conversion', {}))
            # 初始化缓存管理器
            self._cache_manager = CacheManager(config.get('cache', {}))

            logger.info(f"市场数据适配器 {self._adapter_id} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"市场数据适配器 {self._adapter_id} 初始化失败: {e}")
            return False

    async def shutdown(self) -> None:
        """关闭适配器"""
        if self._is_connected:
            await self.disconnect()

        # 清理订阅
        for subscription_id in list(self._subscriptions.keys()):
            await self.unsubscribe_data(subscription_id)

        # 关闭组件
        if self._connection_manager:
            await self._connection_manager.shutdown()
        if self._cache_manager:
            await self._cache_manager.shutdown()

        logger.info(f"市场数据适配器 {self._adapter_id} 已关闭")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'adapter_id': self._adapter_id,
            'adapter_type': self.adapter_type,
            'connected': self._is_connected,
            'subscriptions': len(self._subscriptions),
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_adapter_info(self) -> Dict[str, Any]:
        """获取适配器信息"""
        return {
            'adapter_id': self._adapter_id,
            'adapter_type': self.adapter_type,
            'market_type': self._market_type,
            'config': self._config,
            'connected': self._is_connected
        }

    async def connect(self) -> bool:
        """连接到市场数据源"""
        try:
            if self._connection_manager:
                self._is_connected = await self._connection_manager.connect()
                if self._is_connected:
                    logger.info(f"市场数据适配器 {self._adapter_id} 连接成功")
                return self._is_connected
            return False
        except Exception as e:
            logger.error(f"市场数据适配器 {self._adapter_id} 连接失败: {e}")
            return False

    async def disconnect(self) -> None:
        """断开连接"""
        if self._connection_manager:
            await self._connection_manager.disconnect()
        self._is_connected = False
        logger.info(f"市场数据适配器 {self._adapter_id} 已断开连接")

    async def fetch_data(self, **kwargs) -> Any:
        """获取市场数据"""
        symbol = kwargs.get('symbol')
        data_type = kwargs.get('data_type', 'realtime')

        if not symbol:
            raise ValueError("必须提供symbol参数")

        # 检查缓存
        cache_key = f"{self._market_type}:{symbol}:{data_type}"
        if self._cache_manager:
            cached_data = await self._cache_manager.get(cache_key)
            if cached_data:
                return cached_data

        # 获取原始数据
        if not self._is_connected:
            await self.connect()

        raw_data = await self._fetch_raw_data(symbol, data_type)

        # 数据转换
        if self._data_converter:
            converted_data = await self._data_converter.convert(raw_data, self._market_type)
        else:
            converted_data = raw_data

        # 缓存数据
        if self._cache_manager:
            await self._cache_manager.set(cache_key, converted_data, ttl=300)

        return converted_data

    async def _fetch_raw_data(self, symbol: str, data_type: str) -> Dict[str, Any]:
        """获取原始数据（子类实现）"""
        raise NotImplementedError("子类必须实现_fetch_raw_data方法")

    async def subscribe_data(self, callback: Callable, **kwargs) -> str:
        """订阅市场数据"""
        symbol = kwargs.get('symbol')
        if not symbol:
            raise ValueError("必须提供symbol参数")

        import uuid
        subscription_id = str(uuid.uuid4())

        if not self._is_connected:
            await self.connect()

        # 创建订阅
        self._subscriptions[subscription_id] = {
            'symbol': symbol,
            'callback': callback,
            'kwargs': kwargs
        }

        # 启动数据流
        asyncio.create_task(self._start_data_stream(subscription_id))

        logger.info(f"订阅市场数据: {symbol}, 订阅ID: {subscription_id}")
        return subscription_id

    async def unsubscribe_data(self, subscription_id: str) -> bool:
        """取消订阅"""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.info(f"取消订阅: {subscription_id}")
            return True
        return False

    async def _start_data_stream(self, subscription_id: str) -> None:
        """启动数据流（子类实现）"""
        raise NotImplementedError("子类必须实现_start_data_stream方法")


class TradingAdapter(DataSourceAdapter):
    """
    交易接口适配器

    适配不同券商交易接口到统一的交易接口
    """

    def __init__(self, adapter_id: str, broker_type: str):
        self._adapter_id = adapter_id
        self._broker_type = broker_type
        self._config = {}
        self._connection_manager = None
        self._order_converter = None
        self._execution_monitor = None
        self._is_connected = False

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_type(self) -> str:
        return "trading"

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化交易适配器"""
        try:
            self._config = config
            self._connection_manager = ConnectionManager(config.get('connection', {}))
            self._order_converter = OrderConverter(config.get('order_conversion', {}))
            self._execution_monitor = ExecutionMonitor(config.get('execution_monitoring', {}))

            logger.info(f"交易适配器 {self._adapter_id} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"交易适配器 {self._adapter_id} 初始化失败: {e}")
            return False

    async def shutdown(self) -> None:
        """关闭适配器"""
        if self._is_connected:
            await self.disconnect()

        if self._execution_monitor:
            await self._execution_monitor.shutdown()
        if self._connection_manager:
            await self._connection_manager.shutdown()

        logger.info(f"交易适配器 {self._adapter_id} 已关闭")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'adapter_id': self._adapter_id,
            'adapter_type': self.adapter_type,
            'connected': self._is_connected,
            'broker_type': self._broker_type,
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_adapter_info(self) -> Dict[str, Any]:
        """获取适配器信息"""
        return {
            'adapter_id': self._adapter_id,
            'adapter_type': self.adapter_type,
            'broker_type': self._broker_type,
            'config': self._config,
            'connected': self._is_connected
        }

    async def connect(self) -> bool:
        """连接到交易接口"""
        try:
            if self._connection_manager:
                self._is_connected = await self._connection_manager.connect()
                return self._is_connected
            return False
        except Exception as e:
            logger.error(f"交易适配器 {self._adapter_id} 连接失败: {e}")
            return False

    async def disconnect(self) -> None:
        """断开连接"""
        if self._connection_manager:
            await self._connection_manager.disconnect()
        self._is_connected = False

    async def fetch_data(self, **kwargs) -> Any:
        """获取交易数据（账户信息、持仓等）"""
        data_type = kwargs.get('data_type', 'account')

        if not self._is_connected:
            await self.connect()

        if data_type == 'account':
            return await self._get_account_info()
        elif data_type == 'positions':
            return await self._get_positions()
        elif data_type == 'orders':
            return await self._get_orders()
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

    async def _get_account_info(self) -> Dict[str, Any]:
        """获取账户信息（子类实现）"""
        raise NotImplementedError("子类必须实现_get_account_info方法")

    async def _get_positions(self) -> List[Dict[str, Any]]:
        """获取持仓信息（子类实现）"""
        raise NotImplementedError("子类必须实现_get_positions方法")

    async def _get_orders(self) -> List[Dict[str, Any]]:
        """获取订单信息（子类实现）"""
        raise NotImplementedError("子类必须实现_get_orders方法")

    async def subscribe_data(self, callback: Callable, **kwargs) -> str:
        """订阅交易数据（执行报告等）"""
        import uuid
        subscription_id = str(uuid.uuid4())

        if not self._is_connected:
            await self.connect()

        # 启动执行监控
        if self._execution_monitor:
            await self._execution_monitor.start_monitoring(subscription_id, callback)

        logger.info(f"订阅交易数据，订阅ID: {subscription_id}")
        return subscription_id

    async def unsubscribe_data(self, subscription_id: str) -> bool:
        """取消订阅"""
        if self._execution_monitor:
            return await self._execution_monitor.stop_monitoring(subscription_id)
        return True


class ProtocolConverter(ProtocolAdapter):
    """
    协议转换器

    实现协议转换子系统的核心功能
    """

    def __init__(self, adapter_id: str):
        self._adapter_id = adapter_id
        self._converters = {}
        self._config = {}

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_type(self) -> str:
        return "protocol_converter"

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化协议转换器"""
        try:
            self._config = config
            # 初始化各种协议转换器
            self._converters = {
                'http': HTTPConverter(),
                'websocket': WebSocketConverter(),
                'fix': FIXConverter(),
                'tcp': TCPConverter(),
                'udp': UDPConverter(),
                'json': JSONConverter(),
                'xml': XMLConverter(),
                'protobuf': ProtobufConverter()
            }
            logger.info(f"协议转换器 {self._adapter_id} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"协议转换器 {self._adapter_id} 初始化失败: {e}")
            return False

    async def shutdown(self) -> None:
        """关闭协议转换器"""
        self._converters.clear()
        logger.info(f"协议转换器 {self._adapter_id} 已关闭")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'adapter_id': self._adapter_id,
            'adapter_type': self.adapter_type,
            'converters': len(self._converters),
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_adapter_info(self) -> Dict[str, Any]:
        """获取适配器信息"""
        return {
            'adapter_id': self._adapter_id,
            'adapter_type': self.adapter_type,
            'supported_protocols': self.supports_protocols(),
            'config': self._config
        }

    async def convert_protocol(self, data: Any, from_protocol: str, to_protocol: str) -> Any:
        """协议转换"""
        if from_protocol == to_protocol:
            return data

        # 获取转换器
        from_converter = self._converters.get(from_protocol)
        to_converter = self._converters.get(to_protocol)

        if not from_converter or not to_converter:
            raise ValueError(f"不支持的协议转换: {from_protocol} -> {to_protocol}")

        try:
            # 解析源协议数据
            parsed_data = await from_converter.parse(data)

            # 转换为目标协议格式
            converted_data = await to_converter.format(parsed_data)

            return converted_data

        except Exception as e:
            logger.error(f"协议转换失败 {from_protocol} -> {to_protocol}: {e}")
            raise

    def supports_protocols(self) -> List[str]:
        """获取支持的协议列表"""
        return list(self._converters.keys())


class ConnectionManager(ConnectionAdapter):
    """
    连接管理器

    实现连接管理子系统的核心功能
    """

    def __init__(self, adapter_id: str):
        self._adapter_id = adapter_id
        self._pools = {}
        self._health_checks = {}
        self._config = {}

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_type(self) -> str:
        return "connection_manager"

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化连接管理器"""
        try:
            self._config = config
            # 初始化连接池
            for endpoint, endpoint_config in config.get('endpoints', {}).items():
                await self._create_connection_pool(endpoint)
            logger.info(f"连接管理器 {self._adapter_id} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"连接管理器 {self._adapter_id} 初始化失败: {e}")
            return False

    async def connect(self) -> bool:
        """连接管理器连接"""
        try:
            # 这里可以添加全局连接逻辑
            logger.info(f"连接管理器 {self._adapter_id} 已连接")
            return True
        except Exception as e:
            logger.error(f"连接管理器 {self._adapter_id} 连接失败: {e}")
            return False

    async def disconnect(self) -> None:
        """断开连接管理器"""
        try:
            # 关闭所有连接池
            for pool in self._pools.values():
                await pool.shutdown()
            self._pools.clear()
            logger.info(f"连接管理器 {self._adapter_id} 已断开连接")
        except Exception as e:
            logger.error(f"连接管理器 {self._adapter_id} 断开连接失败: {e}")

    async def shutdown(self) -> None:
        """关闭连接管理器"""
        # 关闭所有连接池
        for pool in self._pools.values():
            await pool.shutdown()
        self._pools.clear()
        self._health_checks.clear()
        logger.info(f"连接管理器 {self._adapter_id} 已关闭")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        stats = await self.get_connection_stats()
        return {
            'adapter_id': self._adapter_id,
            'adapter_type': self.adapter_type,
            'pools': len(self._pools),
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_adapter_info(self) -> Dict[str, Any]:
        """获取适配器信息"""
        return {
            'adapter_id': self._adapter_id,
            'adapter_type': self.adapter_type,
            'pools': len(self._pools),
            'config': self._config
        }

    async def get_connection(self, endpoint: str) -> Any:
        """获取连接"""
        if endpoint not in self._pools:
            await self._create_connection_pool(endpoint)

        pool = self._pools[endpoint]
        connection = await pool.get_connection()

        # 注册健康检查
        if endpoint not in self._health_checks:
            self._health_checks[endpoint] = lambda: self._check_connection_health(connection)

        return connection

    async def release_connection(self, connection: Any) -> None:
        """释放连接"""
        # 找到对应的连接池
        for pool in self._pools.values():
            if await pool.release_connection(connection):
                return

    async def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        stats = {}
        for endpoint, pool in self._pools.items():
            pool_stats = await pool.get_stats()
            stats[endpoint] = pool_stats
        return stats

    async def _create_connection_pool(self, endpoint: str) -> None:
        """创建连接池"""
        endpoint_config = self._config.get('endpoints', {}).get(endpoint, {})
        pool_type = endpoint_config.get('type', 'http')

        if pool_type == 'http':
            pool = HTTPConnectionPool(endpoint, **endpoint_config)
        elif pool_type == 'websocket':
            pool = WebSocketConnectionPool(endpoint, **endpoint_config)
        else:
            raise ValueError(f"不支持的连接池类型: {pool_type}")

        await pool.initialize()
        self._pools[endpoint] = pool

    async def _check_connection_health(self, connection: Any) -> bool:
        """检查连接健康"""
        try:
            # 简单的ping操作
            await connection.ping()
            return True
        except Exception:
            return False


# ==================== 辅助组件 ====================

# 简化的辅助组件实现（实际应该从各自模块导入）
class HTTPConverter:
    async def parse(self, data): return data
    async def format(self, data): return data


class WebSocketConverter:
    async def parse(self, data): return data
    async def format(self, data): return data


class FIXConverter:
    async def parse(self, data): return data
    async def format(self, data): return data


class TCPConverter:
    async def parse(self, data): return data
    async def format(self, data): return data


class UDPConverter:
    async def parse(self, data): return data
    async def format(self, data): return data


class JSONConverter:
    async def parse(self, data): return data if isinstance(data, dict) else json.loads(data)
    async def format(self, data): return json.dumps(data)


class XMLConverter:
    async def parse(self, data): return data
    async def format(self, data): return data


class ProtobufConverter:
    async def parse(self, data): return data
    async def format(self, data): return data


class DataConverter:
    def __init__(self, config): self.config = config
    async def convert(self, data, market_type): return data


class CacheManager:
    def __init__(self, config): self.config = config
    async def get(self, key): return None
    async def set(self, key, value, ttl=None): pass
    async def shutdown(self): pass


class OrderConverter:
    def __init__(self, config): self.config = config


class ExecutionMonitor:
    def __init__(self, config): self.config = config
    async def start_monitoring(self, subscription_id, callback): pass
    async def stop_monitoring(self, subscription_id): return True
    async def shutdown(self): pass


class HTTPConnectionPool:
    def __init__(self, endpoint, **kwargs):
        self.endpoint = endpoint
        self.config = kwargs
        self.connections = []

    async def initialize(self): pass
    async def get_connection(self): return {"type": "http", "endpoint": self.endpoint}
    async def release_connection(self, connection): return True
    async def shutdown(self): pass
    async def get_stats(self): return {"total": len(self.connections), "active": 0, "idle": 0}


class WebSocketConnectionPool:
    def __init__(self, endpoint, **kwargs):
        self.endpoint = endpoint
        self.config = kwargs
        self.connections = []

    async def initialize(self): pass
    async def get_connection(self): return {"type": "websocket", "endpoint": self.endpoint}
    async def release_connection(self, connection): return True
    async def shutdown(self): pass
    async def get_stats(self): return {"total": len(self.connections), "active": 0, "idle": 0}


# ==================== 适配器工厂 ====================

class AdapterFactory:
    """
    适配器工厂

    创建和管理各种适配器实例
    """

    def __init__(self):
        self._adapter_types: Dict[str, Type[BaseAdapter]] = {}

    def register_adapter_type(self, adapter_type: str, adapter_class: Type[BaseAdapter]) -> None:
        """
        注册适配器类型

        Args:
            adapter_type: 适配器类型名称
            adapter_class: 适配器类
        """
        self._adapter_types[adapter_type] = adapter_class

    def create_adapter(self, adapter_type: str, *args, **kwargs) -> BaseAdapter:
        """
        创建适配器实例

        Args:
            adapter_type: 适配器类型
            adaptee: 被适配对象

        Returns:
            适配器实例
        """
        if adapter_type not in self._adapter_types:
            raise ValueError(f"未注册的适配器类型: {adapter_type}")

        adapter_class = self._adapter_types[adapter_type]
        return adapter_class(*args, **kwargs)

    def get_supported_adapter_types(self) -> List[str]:
        """
        获取支持的适配器类型

        Returns:
            适配器类型列表
        """
        return list(self._adapter_types.keys())


# 全局适配器工厂实例
global_adapter_factory = AdapterFactory()

# 注册默认适配器类型
# global_adapter_factory.register_adapter_type('database', DatabaseAdapter)
# global_adapter_factory.register_adapter_type('cache', CacheAdapter)
# global_adapter_factory.register_adapter_type('configuration', ConfigurationAdapter)


# ==================== 便捷函数 ====================

def create_adapter(adapter_type: str, *args, **kwargs) -> BaseAdapter:
    """
    创建适配器（使用全局工厂）

    Args:
        adapter_type: 适配器类型
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        适配器实例
    """
    return global_adapter_factory.create_adapter(adapter_type, *args, **kwargs)


def adapt_operation(adapter: BaseAdapter, operation: str, *args, **kwargs) -> Any:
    """
    执行适配操作

    Args:
        adapter: 适配器实例
        operation: 操作名称
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        操作结果
    """
    return adapter.adapt(operation, *args, **kwargs)
