"""
standard_interfaces 模块

提供 standard_interfaces 相关功能和接口。
"""

import logging


from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Callable
"""
RQA2025 Infrastructure Standard Interfaces

Standard interfaces for infrastructure components.
"""

logger = logging.getLogger(__name__)

# 服务状态枚举


class ServiceStatus(Enum):
    """服务状态枚举"""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"

# 数据请求接口


@dataclass
class DataRequest:

    """Data request structure"""
    symbol: str
    market: str = "CN"
    data_type: str = "stock"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: str = "1d"
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'market': self.market,
            'data_type': self.data_type,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'interval': self.interval,
            'params': self.params or {}
        }


@dataclass
class DataResponse:

    """Data response structure"""
    request: DataRequest
    data: Any
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()


class IServiceProvider(Protocol):

    """Service provider interface"""

    def get_service(self, service_name: str) -> Any:
        """Get service instance"""
        ...

    def register_service(self, service_name: str, service_instance: Any) -> bool:
        """Register service instance"""
        ...

    def unregister_service(self, service_name: str) -> bool:
        """Unregister service instance"""
        ...


class ICacheProvider(Protocol):

    """Cache provider interface"""

    def get(self, key: str) -> Any:
        """Get value from cache"""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        ...

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        ...

    def clear(self) -> bool:
        """Clear all cache"""
        ...


class ILogger(Protocol):

    """Logger interface"""

    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        ...

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        ...

    def error(self, message: str, **kwargs) -> None:
        """Log error message"""
        ...

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        ...


class IConfigProvider(Protocol):

    """Configuration provider interface"""

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        ...

    def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        ...

    def load_config(self, config_file: str) -> bool:
        """Load configuration from file"""
        ...

    def save_config(self, config_file: str) -> bool:
        """Save configuration to file"""
        ...


class IHealthCheck(Protocol):

    """Health check interface"""

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        ...

    def is_healthy(self) -> bool:
        """Check if component is healthy"""
        ...

# 事件相关接口


@dataclass
class Event:

    """Event structure"""
    event_type: str
    data: Optional[Dict[str, Any]] = None
    source: str = "system"
    timestamp: float = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()


class IEventBus(Protocol):

    """Event bus interface"""

    def publish(self, event: Event) -> str:
        """Publish event"""
        ...

    def subscribe(self, event_type: str, handler: callable) -> bool:
        """Subscribe to event"""
        ...

    def unsubscribe(self, event_type: str, handler: callable) -> bool:
        """Unsubscribe from event"""
        ...


class IConfigEventBus(Protocol):

    """Configuration event bus interface"""

    def publish(self, event_type: str, payload: Dict) -> None:
        """Publish configuration event"""
        ...

    def subscribe(self, event_type: str, handler: Callable[[Dict], None],

                  filter_func: Optional[Callable[[Dict], bool]] = None) -> str:
        """Subscribe to configuration event"""
        ...

    def unsubscribe(self, event_type: str, subscription_id: str) -> bool:
        """Unsubscribe from configuration event"""
        ...

    def get_subscribers(self, event: str) -> Dict[str, Callable]:
        """Get event subscribers"""
        ...

    def notify_config_updated(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify configuration update"""
        ...

    def notify_config_error(self, error: str, details: Dict[str, Any]) -> None:
        """Notify configuration error"""
        ...

    def notify_config_loaded(self, source: str, config: Dict[str, Any]) -> None:
        """Notify configuration loaded"""
        ...

    def get_dead_letters(self) -> List[Dict]:
        """Get dead letter queue"""
        ...

    def clear_dead_letters(self) -> None:
        """Clear dead letter queue"""
        ...

    def emit_config_changed(self, key: str, old_value: Any, new_value: Any) -> None:
        """Emit configuration changed event"""
        ...

    def emit_config_loaded(self, source: str) -> None:
        """Emit configuration loaded event"""
        ...


class IConfigVersionManager(Protocol):

    """Configuration version manager interface"""

    def get_latest_version(self) -> Optional[str]:
        """Get latest version"""
        ...

    @property
    def _versions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get versions data"""
        ...


class IEventSubscriber(Protocol):

    """Event subscriber interface"""

    def __init__(self, event_bus: IConfigEventBus):
        """Initialize subscriber"""
        ...

    def handle_event(self, event: Dict) -> bool:
        """Handle event"""
        ...

# 监控接口


class IMonitor(Protocol):

    """Monitor interface"""

    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None) -> None:
        """Record metric"""
        ...

    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metric"""
        ...

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        ...

# 特征处理相关接口


@dataclass
class FeatureRequest:

    """Feature processing request"""
    data: Any
    feature_names: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'data': self.data,
            'feature_names': self.feature_names or [],
            'config': self.config or {},
            'metadata': self.metadata or {}
        }


@dataclass
class FeatureResponse:

    """Feature processing response"""
    features: Any
    feature_names: List[str]
    metadata: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None


class IFeatureProcessor(Protocol):

    """Feature processor interface"""

    def process(self, request: FeatureRequest) -> FeatureResponse:
        """Process features"""
        ...

    def get_supported_features(self) -> List[str]:
        """Get list of supported features"""
        ...

    def validate_data(self, data: Any) -> bool:
        """Validate input data"""
        ...


# 类型别名
FeatureProcessor = IFeatureProcessor

# 导出所有接口
__all__ = [
    'DataRequest',
    'DataResponse',
    'IServiceProvider',
    'ICacheProvider',
    'ILogger',
    'IConfigProvider',
    'IHealthCheck',
    'Event',
    'IEventBus',
    'IConfigEventBus',
    'IConfigVersionManager',
    'IEventSubscriber',
    'IMonitor',
    'FeatureRequest',
    'FeatureResponse',
    'IFeatureProcessor',
    'FeatureProcessor',
    'TradingStrategy'
]

# 交易策略接口


class TradingStrategy(ABC):

    """交易策略接口"""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化策略"""

    @abstractmethod
    def generate_signals(self, data: Any) -> List[Dict[str, Any]]:
        """生成交易信号"""

    @abstractmethod
    def execute_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行交易信号"""

    @abstractmethod
    def evaluate_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估策略性能"""

    @abstractmethod
    def update_parameters(self, new_params: Dict[str, Any]) -> bool:
        """更新策略参数"""

    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
