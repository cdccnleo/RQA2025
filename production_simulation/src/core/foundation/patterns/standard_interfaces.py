"""
RQA2025系统标准接口协议
Standard Interface Protocols for RQA2025 System

定义系统各层级的标准接口协议，确保接口设计的一致性和可扩展性
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Protocol, Union, Callable, TypeVar, Generic
from datetime import datetime

try:
    from ...unified_exceptions import handle_infrastructure_exceptions
except ImportError:
    # Fallback: 简单的装饰器
    def handle_infrastructure_exceptions(func):
        return func

# 类型变量
T = TypeVar('T')

# ==================== 基础接口协议 ====================


class StandardServiceInterface(ABC):
    """
    标准服务接口协议

    所有服务组件应遵循的标准接口协议
    """

    @property
    @abstractmethod
    def service_name(self) -> str:
        """服务名称"""

    @property
    @abstractmethod
    def service_version(self) -> str:
        """服务版本"""

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        初始化服务

        Args:
            config: 初始化配置

        Returns:
            bool: 初始化是否成功
        """

    @abstractmethod
    def shutdown(self) -> None:
        """关闭服务"""

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        获取服务状态

        Returns:
            Dict[str, Any]: 状态信息
        """

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取服务指标

        Returns:
            Dict[str, Any]: 性能指标
        """


class DataAccessInterface(ABC):
    """
    数据访问接口协议

    标准的数据访问层接口
    """

    @abstractmethod
    def connect(self, connection_string: str) -> bool:
        """建立连接"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""

    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""

    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行查询"""

    @abstractmethod
    def execute_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> int:
        """执行命令"""

    @abstractmethod
    def begin_transaction(self) -> 'TransactionInterface':
        """开始事务"""

    @abstractmethod
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""


class TransactionInterface(ABC):
    """事务接口协议"""

    @abstractmethod
    def commit(self) -> None:
        """提交事务"""

    @abstractmethod
    def rollback(self) -> None:
        """回滚事务"""

    @abstractmethod
    def is_active(self) -> bool:
        """检查事务是否活跃"""


class CacheInterface(ABC):
    """
    缓存接口协议

    统一的缓存操作接口
    """

    @abstractmethod
    def get(self, key: str) -> Any:
        """获取缓存值"""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存值"""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查键是否存在"""

    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""

    @abstractmethod
    def get_keys(self, pattern: str = "*") -> List[str]:
        """获取匹配的键列表"""


class ConfigurationInterface(ABC):
    """
    配置接口协议

    标准配置管理接口
    """

    @abstractmethod
    def load(self, source: str) -> bool:
        """加载配置"""

    @abstractmethod
    def save(self, target: str) -> bool:
        """保存配置"""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""

    @abstractmethod
    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""

    @abstractmethod
    def has(self, key: str) -> bool:
        """检查配置是否存在"""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除配置"""

    @abstractmethod
    def validate(self) -> List[str]:
        """验证配置有效性"""

    @abstractmethod
    def reload(self) -> bool:
        """重新加载配置"""


class LoggingInterface(ABC):
    """
    日志接口协议

    标准日志记录接口
    """

    @abstractmethod
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录调试信息"""

    @abstractmethod
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录信息"""

    @abstractmethod
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录警告"""

    @abstractmethod
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录错误"""

    @abstractmethod
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录严重错误"""

    @abstractmethod
    def set_level(self, level: str) -> None:
        """设置日志级别"""

    @abstractmethod
    def add_handler(self, handler: Any) -> None:
        """添加日志处理器"""


class MonitoringInterface(ABC):
    """
    监控接口协议

    标准监控和指标收集接口
    """

    @abstractmethod
    def record_metric(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标"""

    @abstractmethod
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """增加计数器"""

    @abstractmethod
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录计时器"""

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标数据"""

    @abstractmethod
    def reset_metrics(self) -> None:
        """重置指标"""


class EventInterface(ABC):
    """
    事件接口协议

    标准事件发布和订阅接口
    """

    @abstractmethod
    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """发布事件"""

    @abstractmethod
    def subscribe(self, event_type: str, handler: Callable) -> str:
        """订阅事件"""

    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""

    @abstractmethod
    def get_subscriptions(self) -> List[str]:
        """获取订阅列表"""


class QueueInterface(ABC):
    """
    队列接口协议

    标准消息队列接口
    """

    @abstractmethod
    def enqueue(self, message: Any, priority: int = 0) -> bool:
        """入队消息"""

    @abstractmethod
    def dequeue(self) -> Optional[Any]:
        """出队消息"""

    @abstractmethod
    def peek(self) -> Optional[Any]:
        """查看队首消息"""

    @abstractmethod
    def size(self) -> int:
        """获取队列大小"""

    @abstractmethod
    def is_empty(self) -> bool:
        """检查队列是否为空"""

    @abstractmethod
    def clear(self) -> None:
        """清空队列"""


class NotificationInterface(ABC):
    """
    通知接口协议

    标准通知发送接口
    """

    @abstractmethod
    def send_email(self, to: Union[str, List[str]], subject: str, body: str,
                   html: bool = False) -> bool:
        """发送邮件"""

    @abstractmethod
    def send_sms(self, to: Union[str, List[str]], message: str) -> bool:
        """发送短信"""

    @abstractmethod
    def send_webhook(self, url: str, payload: Dict[str, Any],
                     headers: Optional[Dict[str, str]] = None) -> bool:
        """发送Webhook"""

    @abstractmethod
    def get_delivery_status(self, message_id: str) -> Dict[str, Any]:
        """获取发送状态"""


# ==================== 业务层接口协议 ====================

class BusinessServiceInterface(StandardServiceInterface):
    """
    业务服务接口协议

    业务层服务的标准接口
    """

    @abstractmethod
    def validate_business_rules(self, data: Dict[str, Any]) -> List[str]:
        """
        验证业务规则

        Args:
            data: 待验证的数据

        Returns:
            List[str]: 验证错误列表
        """

    @abstractmethod
    def execute_business_logic(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行业务逻辑

        Args:
            operation: 操作类型
            params: 操作参数

        Returns:
            Dict[str, Any]: 执行结果
        """

    @abstractmethod
    def get_business_metrics(self) -> Dict[str, Any]:
        """
        获取业务指标

        Returns:
            Dict[str, Any]: 业务指标数据
        """


class UserServiceInterface(BusinessServiceInterface):
    """用户服务接口协议"""

    @abstractmethod
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建用户"""

    @abstractmethod
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""

    @abstractmethod
    def update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """更新用户信息"""

    @abstractmethod
    def delete_user(self, user_id: str) -> bool:
        """删除用户"""

    @abstractmethod
    def authenticate_user(self, credentials: Dict[str, Any]) -> Optional[str]:
        """用户认证"""


class TradingServiceInterface(BusinessServiceInterface):
    """交易服务接口协议"""

    @abstractmethod
    def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """下单"""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态"""

    @abstractmethod
    def get_portfolio(self, user_id: str) -> Dict[str, Any]:
        """获取投资组合"""

    @abstractmethod
    def execute_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行交易"""


class RiskServiceInterface(BusinessServiceInterface):
    """风险控制服务接口协议"""

    @abstractmethod
    def assess_risk(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """风险评估"""

    @abstractmethod
    def check_limits(self, user_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """检查限额"""

    @abstractmethod
    def calculate_var(self, portfolio_data: Dict[str, Any]) -> float:
        """计算VaR"""

    @abstractmethod
    def get_risk_metrics(self, user_id: str) -> Dict[str, Any]:
        """获取风险指标"""


# ==================== 适配器接口协议 ====================

class AdapterInterface(Generic[T], ABC):
    """
    适配器接口协议

    用于适配不同实现的通用接口
    """

    def __init__(self, adaptee: T):
        self.adaptee = adaptee

    @abstractmethod
    def adapt_operation(self, operation: str, *args, **kwargs) -> Any:
        """
        执行适配操作

        Args:
            operation: 操作名称
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            Any: 操作结果
        """


# ==================== 协议定义 ====================

class ServiceProtocol(Protocol):
    """服务协议"""

    service_name: str
    service_version: str

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool: ...
    def shutdown(self) -> None: ...
    def health_check(self) -> Dict[str, Any]: ...
    def get_status(self) -> Dict[str, Any]: ...
    def get_metrics(self) -> Dict[str, Any]: ...


class DataAccessProtocol(Protocol):
    """数据访问协议"""

    def connect(self, connection_string: str) -> bool: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def execute_query(self, query: str,
                      params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...

    def execute_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> int: ...


class CacheProtocol(Protocol):
    """缓存协议"""

    def get(self, key: str) -> Any: ...
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool: ...
    def delete(self, key: str) -> bool: ...
    def exists(self, key: str) -> bool: ...
    def clear(self) -> bool: ...


# ==================== 工厂和注册器 ====================

class InterfaceRegistry:
    """
    接口注册器

    管理系统中所有的接口实现
    """

    def __init__(self):
        self._interfaces: Dict[str, Dict[str, Any]] = {}
        self._implementations: Dict[str, List[Any]] = {}

    @handle_infrastructure_exceptions
    def register_interface(self, interface_name: str, interface_class: type) -> None:
        """
        注册接口

        Args:
            interface_name: 接口名称
            interface_class: 接口类
        """
        if interface_name not in self._interfaces:
            self._interfaces[interface_name] = {}

        self._interfaces[interface_name]['interface'] = interface_class
        self._interfaces[interface_name]['implementations'] = []

    @handle_infrastructure_exceptions
    def register_implementation(self, interface_name: str, implementation_class: type,
                                version: str = "1.0.0") -> None:
        """
        注册接口实现

        Args:
            interface_name: 接口名称
            implementation_class: 实现类
            version: 版本号
        """
        if interface_name not in self._interfaces:
            raise ValueError(f"接口 {interface_name} 未注册")

        implementation_info = {
            'class': implementation_class,
            'version': version,
            'registered_at': datetime.now().isoformat()
        }

        self._interfaces[interface_name]['implementations'].append(implementation_info)

    def get_interface(self, interface_name: str) -> Optional[type]:
        """获取接口类"""
        return self._interfaces.get(interface_name, {}).get('interface')

    def get_implementations(self, interface_name: str) -> List[Dict[str, Any]]:
        """获取接口的所有实现"""
        return self._interfaces.get(interface_name, {}).get('implementations', [])

    def get_default_implementation(self, interface_name: str) -> Optional[type]:
        """获取接口的默认实现"""
        implementations = self.get_implementations(interface_name)
        if implementations:
            # 返回最新版本的实现
            return max(implementations, key=lambda x: x['version'])['class']
        return None

    def list_interfaces(self) -> List[str]:
        """列出所有注册的接口"""
        return list(self._interfaces.keys())


# 全局接口注册器实例
global_interface_registry = InterfaceRegistry()


# ==================== 便捷函数 ====================

def register_standard_interface(interface_name: str, interface_class: type) -> None:
    """
    注册标准接口

    Args:
        interface_name: 接口名称
        interface_class: 接口类
    """
    global_interface_registry.register_interface(interface_name, interface_class)


def register_interface_implementation(interface_name: str, implementation_class: type,
                                      version: str = "1.0.0") -> None:
    """
    注册接口实现

    Args:
        interface_name: 接口名称
        implementation_class: 实现类
        version: 版本号
    """
    global_interface_registry.register_implementation(interface_name, implementation_class, version)


def get_interface_implementations(interface_name: str) -> List[Dict[str, Any]]:
    """
    获取接口实现

    Args:
        interface_name: 接口名称

    Returns:
        List[Dict[str, Any]]: 实现列表
    """
    return global_interface_registry.get_implementations(interface_name)


def get_default_implementation(interface_name: str) -> Optional[type]:
    """
    获取默认实现

    Args:
        interface_name: 接口名称

    Returns:
        Optional[type]: 默认实现类
    """
    return global_interface_registry.get_default_implementation(interface_name)


# 初始化标准接口
def initialize_standard_interfaces() -> None:
    """初始化标准接口注册"""
    # 基础服务接口
    register_standard_interface("StandardService", StandardServiceInterface)
    register_standard_interface("DataAccess", DataAccessInterface)
    register_standard_interface("Cache", CacheInterface)
    register_standard_interface("Configuration", ConfigurationInterface)
    register_standard_interface("Logging", LoggingInterface)
    register_standard_interface("Monitoring", MonitoringInterface)

    # 业务服务接口
    register_standard_interface("BusinessService", BusinessServiceInterface)
    register_standard_interface("UserService", UserServiceInterface)
    register_standard_interface("TradingService", TradingServiceInterface)
    register_standard_interface("RiskService", RiskServiceInterface)

    # 适配器接口
    register_standard_interface("Adapter", AdapterInterface)


# 自动初始化
initialize_standard_interfaces()


# 兼容性别名
BusinessLogicInterface = BusinessServiceInterface
PresentationInterface = StandardServiceInterface
MessageQueueInterface = QueueInterface
CachingInterface = CacheInterface
SecurityInterface = StandardServiceInterface


__all__ = [
    'StandardServiceInterface',
    'DataAccessInterface',
    'BusinessLogicInterface',
    'PresentationInterface',
    'MonitoringInterface',
    'ConfigurationInterface',
    'SecurityInterface',
    'LoggingInterface',
    'CachingInterface',
    'MessageQueueInterface',
    'TransactionInterface',
    'EventInterface',
    'NotificationInterface',
    'BusinessServiceInterface',
    'UserServiceInterface',
    'TradingServiceInterface',
    'RiskServiceInterface',
    'AdapterInterface',
    'InterfaceRegistry',
]
