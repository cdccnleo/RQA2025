#!/usr/bin/env python3
"""
统一数据加载器接口

定义数据管理层数据加载的统一接口，确保所有数据加载器实现统一的API。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterator, AsyncIterator
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class DataSource(Enum):
    """数据源类型"""
    STOCK = "stock"
    FUTURES = "futures"
    OPTIONS = "options"
    INDEX = "index"
    BONDS = "bonds"
    FOREX = "forex"
    CRYPTO = "crypto"
    NEWS = "news"
    MACRO = "macro"
    COMMODITY = "commodity"


class DataFormat(Enum):
    """数据格式"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    BINARY = "binary"
    PROTOBUF = "protobuf"
    AVRO = "avro"


class LoadingMode(Enum):
    """加载模式"""
    BATCH = "batch"  # 批量加载
    STREAM = "stream"  # 流式加载
    INCREMENTAL = "incremental"  # 增量加载
    HISTORICAL = "historical"  # 历史数据加载


class LoadingPriority(Enum):
    """加载优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DataRequest:
    """
    数据请求

    表示对数据的加载请求。
    """
    source: DataSource
    symbols: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    data_format: DataFormat = DataFormat.JSON
    loading_mode: LoadingMode = LoadingMode.BATCH
    priority: LoadingPriority = LoadingPriority.NORMAL
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    batch_size: int = 1000
    timeout: int = 300  # 秒


@dataclass
class DataResponse:
    """
    数据响应

    表示数据加载的结果。
    """
    request: DataRequest
    data: Any
    metadata: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None
    load_time: Optional[float] = None
    record_count: int = 0


class IDataLoader(ABC):
    """
    数据加载器统一接口

    所有数据加载器实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def load_data(self, request: DataRequest) -> DataResponse:
        """
        加载数据

        Args:
            request: 数据请求

        Returns:
            数据响应
        """

    @abstractmethod
    async def load_data_async(self, request: DataRequest) -> DataResponse:
        """
        异步加载数据

        Args:
            request: 数据请求

        Returns:
            数据响应
        """

    @abstractmethod
    def load_data_stream(self, request: DataRequest) -> Iterator[DataResponse]:
        """
        流式加载数据

        Args:
            request: 数据请求

        Yields:
            数据响应迭代器
        """

    @abstractmethod
    async def load_data_stream_async(self, request: DataRequest) -> AsyncIterator[DataResponse]:
        """
        异步流式加载数据

        Args:
            request: 数据请求

        Yields:
            数据响应异步迭代器
        """

    @abstractmethod
    def validate_request(self, request: DataRequest) -> Dict[str, Any]:
        """
        验证请求

        Args:
            request: 数据请求

        Returns:
            验证结果 {'valid': bool, 'errors': List[str], 'warnings': List[str]}
        """

    @abstractmethod
    def get_supported_sources(self) -> List[DataSource]:
        """
        获取支持的数据源

        Returns:
            支持的数据源列表
        """

    @abstractmethod
    def get_supported_formats(self) -> List[DataFormat]:
        """
        获取支持的数据格式

        Returns:
            支持的数据格式列表
        """

    @abstractmethod
    def get_loading_modes(self) -> List[LoadingMode]:
        """
        获取支持的加载模式

        Returns:
            支持的加载模式列表
        """

    @abstractmethod
    def estimate_load_time(self, request: DataRequest) -> float:
        """
        估算加载时间

        Args:
            request: 数据请求

        Returns:
            预估加载时间(秒)
        """

    @abstractmethod
    def get_cache_status(self, request: DataRequest) -> Dict[str, Any]:
        """
        获取缓存状态

        Args:
            request: 数据请求

        Returns:
            缓存状态信息
        """

    @abstractmethod
    def prefetch_data(self, request: DataRequest) -> bool:
        """
        预取数据

        Args:
            request: 数据请求

        Returns:
            是否预取成功
        """

    @abstractmethod
    def cancel_loading(self, request_id: str) -> bool:
        """
        取消加载

        Args:
            request_id: 请求ID

        Returns:
            是否取消成功
        """

    @abstractmethod
    def get_loading_status(self, request_id: str) -> Dict[str, Any]:
        """
        获取加载状态

        Args:
            request_id: 请求ID

        Returns:
            加载状态信息
        """

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        配置加载器

        Args:
            config: 配置字典

        Returns:
            是否配置成功
        """

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查

        Returns:
            健康检查结果
        """

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标

        Returns:
            性能指标字典
        """

    @abstractmethod
    def optimize_loading(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化加载性能

        Args:
            metrics: 当前性能指标

        Returns:
            优化建议和结果
        """


class IDataLoaderFactory(ABC):
    """
    数据加载器工厂接口
    """

    @abstractmethod
    def create_loader(self, source: DataSource, config: Dict[str, Any]) -> IDataLoader:
        """
        创建数据加载器

        Args:
            source: 数据源
            config: 配置字典

        Returns:
            数据加载器实例
        """

    @abstractmethod
    def get_supported_sources(self) -> List[DataSource]:
        """
        获取支持的数据源

        Returns:
            支持的数据源列表
        """

    @abstractmethod
    def get_loader_config_template(self, source: DataSource) -> Dict[str, Any]:
        """
        获取加载器配置模板

        Args:
            source: 数据源

        Returns:
            配置模板字典
        """


class IDataLoaderManager(ABC):
    """
    数据加载器管理器接口
    """

    @abstractmethod
    def register_loader(self, source: DataSource, loader: IDataLoader) -> bool:
        """
        注册数据加载器

        Args:
            source: 数据源
            loader: 数据加载器

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister_loader(self, source: DataSource) -> bool:
        """
        注销数据加载器

        Args:
            source: 数据源

        Returns:
            是否注销成功
        """

    @abstractmethod
    def get_loader(self, source: DataSource) -> Optional[IDataLoader]:
        """
        获取数据加载器

        Args:
            source: 数据源

        Returns:
            数据加载器实例
        """

    @abstractmethod
    def load_data(self, request: DataRequest) -> DataResponse:
        """
        加载数据（统一入口）

        Args:
            request: 数据请求

        Returns:
            数据响应
        """

    @abstractmethod
    async def load_data_async(self, request: DataRequest) -> DataResponse:
        """
        异步加载数据（统一入口）

        Args:
            request: 数据请求

        Returns:
            数据响应
        """

    @abstractmethod
    def get_all_loaders(self) -> Dict[str, IDataLoader]:
        """
        获取所有加载器

        Returns:
            加载器字典 {source: loader}
        """

    @abstractmethod
    def get_loading_statistics(self) -> Dict[str, Any]:
        """
        获取加载统计信息

        Returns:
            统计信息字典
        """

    @abstractmethod
    def optimize_loaders(self) -> Dict[str, Any]:
        """
        优化所有加载器

        Returns:
            优化结果字典
        """


class IDataCache(ABC):
    """
    数据缓存接口
    """

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存数据

        Args:
            key: 缓存键
            default: 默认值

        Returns:
            缓存数据
        """

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存数据

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间(秒)

        Returns:
            是否设置成功
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除缓存数据

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """

    @abstractmethod
    def clear(self) -> bool:
        """
        清空缓存

        Returns:
            是否清空成功
        """

    @abstractmethod
    def has(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """

    @abstractmethod
    def get_cache_info(self, key: str) -> Dict[str, Any]:
        """
        获取缓存信息

        Args:
            key: 缓存键

        Returns:
            缓存信息字典
        """


class IDataValidator(ABC):
    """
    数据验证器接口
    """

    @abstractmethod
    def validate_data(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证数据

        Args:
            data: 要验证的数据
            schema: 验证模式

        Returns:
            验证结果 {'valid': bool, 'errors': List[str], 'warnings': List[str]}
        """

    @abstractmethod
    def validate_request(self, request: DataRequest) -> Dict[str, Any]:
        """
        验证请求

        Args:
            request: 数据请求

        Returns:
            验证结果
        """

    @abstractmethod
    def get_validation_schema(self, source: DataSource) -> Dict[str, Any]:
        """
        获取验证模式

        Args:
            source: 数据源

        Returns:
            验证模式字典
        """
