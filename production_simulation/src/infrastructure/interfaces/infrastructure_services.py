"""
RQA2025 基础设施层 - 基础设施服务接口定义

本模块定义基础设施层提供的纯粹技术服务接口，
这些接口不依赖任何业务层，而是为所有上层业务提供技术支撑。

基础设施层职责：
1. 配置管理 - 统一配置服务
2. 缓存服务 - 多级缓存管理
3. 日志服务 - 结构化日志记录
4. 监控服务 - 系统和应用监控
5. 安全服务 - 身份认证和授权
6. 健康检查 - 服务健康状态监控
7. 资源管理 - 系统资源配额管理
8. 事件总线 - 异步事件处理
9. 依赖注入 - 服务容器管理
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# 基础设施服务状态枚举
# =============================================================================

class InfrastructureServiceStatus(Enum):
    """基础设施服务状态"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    ERROR = "error"


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# =============================================================================
# 配置管理服务接口
# =============================================================================

class IConfigManager(Protocol):
    """配置管理器接口 - 提供统一的配置管理服务"""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""

    @abstractmethod
    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""

    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置节"""

    @abstractmethod
    def reload(self) -> bool:
        """重新加载配置"""

    @abstractmethod
    def validate_config(self) -> List[str]:
        """验证配置有效性，返回错误列表"""


# =============================================================================
# 缓存服务接口
# =============================================================================

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: datetime = None
    accessed_at: datetime = None
    access_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.accessed_at is None:
            self.accessed_at = datetime.now()


class ICacheService(Protocol):
    """缓存服务接口 - 提供统一的缓存管理"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存值"""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查缓存键是否存在"""

    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""


class IMultiLevelCache(Protocol):
    """多级缓存接口 - 支持L1/L2/L3多级缓存"""

    @abstractmethod
    def get_from_level(self, level: int, key: str) -> Optional[Any]:
        """从指定缓存级别获取值"""

    @abstractmethod
    def set_to_level(self, level: int, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置指定缓存级别的值"""

    @abstractmethod
    def invalidate_level(self, level: int, key: str) -> bool:
        """使指定缓存级别的键无效"""

    @abstractmethod
    def get_cache_levels(self) -> List[str]:
        """获取缓存级别列表"""


# =============================================================================
# 日志服务接口
# =============================================================================

@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None
    exception: Optional[Exception] = None
    extra_data: Optional[Dict[str, Any]] = None


class ILogger(Protocol):
    """日志器接口 - 提供结构化日志记录"""

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """记录调试日志"""

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """记录信息日志"""

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """记录警告日志"""

    @abstractmethod
    def error(self, message: str, exc: Optional[Exception] = None, **kwargs) -> None:
        """记录错误日志"""

    @abstractmethod
    def critical(self, message: str, exc: Optional[Exception] = None, **kwargs) -> None:
        """记录严重错误日志"""

    @abstractmethod
    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """记录指定级别日志"""

    @abstractmethod
    def is_enabled_for(self, level: LogLevel) -> bool:
        """检查指定级别是否启用"""


class ILogManager(Protocol):
    """日志管理器接口 - 管理多个日志器"""

    @abstractmethod
    def get_logger(self, name: str) -> ILogger:
        """获取指定名称的日志器"""

    @abstractmethod
    def configure_logger(self, name: str, config: Dict[str, Any]) -> bool:
        """配置日志器"""

    @abstractmethod
    def get_all_loggers(self) -> Dict[str, ILogger]:
        """获取所有日志器"""


# =============================================================================
# 监控服务接口
# =============================================================================

@dataclass
class MetricData:
    """监控指标数据"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


class IMonitor(Protocol):
    """监控器接口 - 系统和应用监控"""

    @abstractmethod
    def record_metric(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标"""

    @abstractmethod
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """增加计数器"""

    @abstractmethod
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录直方图"""

    @abstractmethod
    def start_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """开始计时器，返回计时器ID"""

    @abstractmethod
    def stop_timer(self, timer_id: str) -> float:
        """停止计时器，返回耗时(秒)"""

    @abstractmethod
    def get_metrics(self, pattern: Optional[str] = None) -> List[MetricData]:
        """获取指标数据"""


# =============================================================================
# 安全服务接口
# =============================================================================

@dataclass
class UserCredentials:
    """用户凭据"""
    username: str
    password_hash: str
    salt: str
    roles: List[str]
    permissions: List[str]
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SecurityToken:
    """安全令牌"""
    token: str
    user_id: str
    issued_at: datetime
    expires_at: datetime
    permissions: List[str]


class ISecurityManager(Protocol):
    """安全管理器接口 - 身份认证和授权"""

    @abstractmethod
    def authenticate(self, username: str, password: str) -> Optional[SecurityToken]:
        """用户认证"""

    @abstractmethod
    def validate_token(self, token: str) -> Optional[SecurityToken]:
        """验证令牌"""

    @abstractmethod
    def authorize(self, token: str, resource: str, action: str) -> bool:
        """权限验证"""

    @abstractmethod
    def create_user(self, credentials: UserCredentials) -> bool:
        """创建用户"""

    @abstractmethod
    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """更新用户信息"""

    @abstractmethod
    def delete_user(self, username: str) -> bool:
        """删除用户"""

    @abstractmethod
    def get_user_permissions(self, username: str) -> List[str]:
        """获取用户权限"""


# =============================================================================
# 健康检查服务接口
# =============================================================================

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    service_name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time: float
    timestamp: datetime = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class IHealthChecker(Protocol):
    """健康检查器接口 - 服务健康状态监控"""

    @abstractmethod
    def check_health(self) -> HealthCheckResult:
        """执行健康检查"""

    @abstractmethod
    def is_healthy(self) -> bool:
        """快速健康状态检查"""

    @abstractmethod
    def get_health_history(self, limit: int = 10) -> List[HealthCheckResult]:
        """获取健康检查历史"""

    @abstractmethod
    def get_detailed_status(self) -> Dict[str, Any]:
        """获取详细状态信息"""


# =============================================================================
# 资源管理服务接口
# =============================================================================

@dataclass
class ResourceQuota:
    """资源配额"""
    resource_type: str  # 'cpu', 'memory', 'disk', 'network'
    limit: Union[int, float]
    used: Union[int, float]
    unit: str
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


class IResourceManager(Protocol):
    """资源管理器接口 - 系统资源配额管理"""

    @abstractmethod
    def get_resource_usage(self, resource_type: str) -> Optional[ResourceQuota]:
        """获取资源使用情况"""

    @abstractmethod
    def set_resource_limit(self, resource_type: str, limit: Union[int, float], unit: str) -> bool:
        """设置资源限制"""

    @abstractmethod
    def check_resource_available(self, resource_type: str, required: Union[int, float]) -> bool:
        """检查资源可用性"""

    @abstractmethod
    def allocate_resource(self, resource_type: str, amount: Union[int, float]) -> bool:
        """分配资源"""

    @abstractmethod
    def release_resource(self, resource_type: str, amount: Union[int, float]) -> bool:
        """释放资源"""

    @abstractmethod
    def get_all_resource_quotas(self) -> Dict[str, ResourceQuota]:
        """获取所有资源配额"""


# =============================================================================
# 事件总线服务接口
# =============================================================================

@dataclass
class Event:
    """事件对象"""
    event_id: str
    event_type: str
    payload: Dict[str, Any]
    source: str
    timestamp: datetime = None
    correlation_id: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventHandler(Protocol):
    """事件处理器协议"""

    def handle_event(self, event: Event) -> None:
        """处理事件"""


class IEventBus(Protocol):
    """事件总线接口 - 异步事件处理"""

    @abstractmethod
    def publish(self, event: Event) -> bool:
        """发布事件"""

    @abstractmethod
    def subscribe(self, event_type: str, handler: EventHandler) -> str:
        """订阅事件，返回订阅ID"""

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""

    @abstractmethod
    def publish_async(self, event: Event) -> str:
        """异步发布事件，返回任务ID"""

    @abstractmethod
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """获取事件历史"""


# =============================================================================
# 依赖注入容器接口
# =============================================================================

class IServiceContainer(Protocol):
    """服务容器接口 - 依赖注入管理"""

    @abstractmethod
    def register(self, interface: type, implementation: type, singleton: bool = True) -> None:
        """注册服务"""

    @abstractmethod
    def register_instance(self, interface: type, instance: Any) -> None:
        """注册服务实例"""

    @abstractmethod
    def resolve(self, interface: type) -> Any:
        """解析服务实例"""

    @abstractmethod
    def has_service(self, interface: type) -> bool:
        """检查服务是否已注册"""

    @abstractmethod
    def unregister(self, interface: type) -> bool:
        """注销服务"""

    @abstractmethod
    def get_registered_services(self) -> List[type]:
        """获取已注册的服务类型"""


# =============================================================================
# 基础设施服务提供者接口
# =============================================================================

class IInfrastructureServiceProvider(Protocol):
    """基础设施服务提供者接口 - 统一访问所有基础设施服务"""

    @property
    def config_manager(self) -> IConfigManager:
        """配置管理器"""

    @property
    def cache_service(self) -> ICacheService:
        """缓存服务"""

    @property
    def logger(self) -> ILogger:
        """日志器"""

    @property
    def monitor(self) -> IMonitor:
        """监控器"""

    @property
    def security_manager(self) -> ISecurityManager:
        """安全管理器"""

    @property
    def health_checker(self) -> IHealthChecker:
        """健康检查器"""

    @property
    def resource_manager(self) -> IResourceManager:
        """资源管理器"""

    @property
    def event_bus(self) -> IEventBus:
        """事件总线"""

    @property
    def service_container(self) -> IServiceContainer:
        """服务容器"""

    @abstractmethod
    def get_service_status(self) -> InfrastructureServiceStatus:
        """获取基础设施服务整体状态"""

    @abstractmethod
    def initialize_all_services(self) -> bool:
        """初始化所有基础设施服务"""

    @abstractmethod
    def shutdown_all_services(self) -> bool:
        """关闭所有基础设施服务"""

    @abstractmethod
    def get_service_health_report(self) -> Dict[str, HealthCheckResult]:
        """获取所有服务的健康报告"""
