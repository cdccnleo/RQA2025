"""
服务通信数据模型

包含服务通信相关的数据模型定义。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


@dataclass
class ServiceEndpoint:
    """服务端点信息"""
    service_name: str
    endpoint_url: str
    protocol: str = "http"
    version: str = "1.0.0"
    timeout: int = 30
    retry_count: int = 3
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"


@dataclass
class Message:
    """通信消息"""
    message_id: str
    message_type: MessageType
    sender: str
    receiver: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    correlation_id: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class CloudNativeConfig:
    """云原生配置"""
    enable_service_mesh: bool = False
    enable_circuit_breaker: bool = True
    enable_adaptive_timeout: bool = True
    enable_performance_optimization: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    adaptive_timeout_percentile: float = 0.95
    max_timeout: int = 120
    min_timeout: int = 5

