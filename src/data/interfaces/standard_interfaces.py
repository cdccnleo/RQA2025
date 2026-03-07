"""
RQA2025 数据层标准接口定义

本模块定义数据层的标准接口规范，
包括数据验证器和数据注册表等标准接口。
"""

from abc import abstractmethod
from typing import Dict, List, Any, Optional, Protocol


class IDataValidator(Protocol):
    """数据验证器接口"""

    @abstractmethod
    def validate(self, data: Any, data_type: str) -> Dict[str, Any]:
        """验证数据"""

    @abstractmethod
    def get_validation_rules(self, data_type: str) -> Dict[str, Any]:
        """获取验证规则"""


class IDataRegistry(Protocol):
    """数据注册表接口"""

    @abstractmethod
    def register_data_source(self, name: str, source_config: Dict[str, Any]) -> bool:
        """注册数据源"""

    @abstractmethod
    def get_data_source(self, name: str) -> Optional[Dict[str, Any]]:
        """获取数据源配置"""

    @abstractmethod
    def list_data_sources(self) -> List[str]:
        """列出所有数据源"""

    @abstractmethod
    def unregister_data_source(self, name: str) -> bool:
        """注销数据源"""


class IDataAdapter(Protocol):
    """数据适配器接口"""

    @abstractmethod
    def connect(self) -> bool:
        """连接数据源"""
        ...

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        ...

    @abstractmethod
    def fetch_data(self, request: 'DataRequest') -> 'DataResponse':
        """获取数据"""
        ...

    @abstractmethod
    def validate_connection(self) -> bool:
        """验证连接"""
        ...


class IDataCache(Protocol):
    """缓存接口"""

    @abstractmethod
    def get(self, key: str) -> Any:
        """获取缓存数据"""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存数据"""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存数据"""

    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""


# 数据请求相关类
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class DataRequest:
    """数据请求数据类"""
    source: str
    query: Dict[str, Any]
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.request_id is None:
            import uuid
            self.request_id = str(uuid.uuid4())


@dataclass
class DataResponse:
    """数据响应数据类"""
    data: Any
    status: str = "success"
    message: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# 数据源类型枚举
from enum import Enum

# 强制定义完整的 DataSourceType，避免外部旧版本残缺导致 AttributeError
DataSourceType = Enum(
    "DataSourceType",
    {
        "DATABASE": "database",
        "API": "api",
        "FILE": "file",
        "STREAM": "stream",
        "CACHE": "cache",
        "STOCK": "stock",
        "CRYPTO": "crypto",
        "NEWS": "news",
        "MACRO": "macro",
    },
)

__all__ = [
    'IDataValidator',
    'IDataRegistry',
    'IDataAdapter',
    'IDataCache',
    'DataRequest',
    'DataResponse',
    'DataSourceType'
]
