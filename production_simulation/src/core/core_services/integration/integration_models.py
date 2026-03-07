"""
服务集成数据模型

包含服务集成相关的数据模型定义。
"""

from dataclasses import dataclass
from typing import Dict, Any

from src.core.constants import DEFAULT_TIMEOUT, DEFAULT_BATCH_SIZE


@dataclass
class ServiceCall:
    """服务调用信息"""
    service_name: str
    method_name: str
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: float = float(DEFAULT_TIMEOUT)
    retry_count: int = 3


@dataclass
class ServiceEndpoint:
    """服务端点信息"""
    service_name: str
    endpoint_url: str
    protocol: str = "http"
    connection_pool_size: int = DEFAULT_BATCH_SIZE
    timeout: float = float(DEFAULT_TIMEOUT)

