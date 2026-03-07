
# =============================================================================
# 数据类型定义

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
"""
统一资源管理接口标准

Phase 2: 结构优化 - 统一接口标准

定义统一的资源管理接口规范，为整个资源管理系统提供一致的抽象层。
"""


@dataclass
class ResourceInfo:
    """资源信息"""
    resource_id: str
    resource_type: str
    name: str
    description: Optional[str] = None
    status: str = "unknown"
    capacity: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class ResourceRequest:
    """资源请求"""
    request_id: str
    resource_type: str
    requester_id: str
    requirements: Dict[str, Any]
    priority: int = 1
    timeout: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ResourceAllocation:
    """资源分配"""
    allocation_id: str
    request_id: str
    resource_id: str
    allocated_resources: Dict[str, Any]
    allocated_at: datetime = None
    expires_at: Optional[datetime] = None
    status: str = "active"

    def __post_init__(self):
        if self.allocated_at is None:
            self.allocated_at = datetime.now()


@dataclass
class ResourceMetrics:
    """资源指标"""
    resource_id: str
    resource_type: str
    timestamp: float
    metrics: Dict[str, Any]
    collected_at: datetime = None

    def __post_init__(self):
        if self.collected_at is None:
            self.collected_at = datetime.now()

# =============================================================================
# 核心接口定义
# =============================================================================


class IResourceProvider(ABC):
    """资源提供者接口

    定义资源提供者的标准行为规范。
    资源提供者负责管理和分配特定类型的资源。
    """

    @property
    @abstractmethod
    def resource_type(self) -> str:
        """资源类型"""

    @abstractmethod
    def get_available_resources(self) -> List[ResourceInfo]:
        """获取可用资源列表"""

    @abstractmethod
    def allocate_resource(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """分配资源"""

    @abstractmethod
    def release_resource(self, allocation_id: str) -> bool:
        """释放资源"""

    @abstractmethod
    def get_resource_status(self, resource_id: str) -> Optional[ResourceInfo]:
        """获取资源状态"""

    @abstractmethod
    def optimize_resources(self) -> Dict[str, Any]:
        """优化资源使用"""


class IResourceConsumer(ABC):
    """资源消费者接口

    定义资源消费者的标准行为规范。
    资源消费者负责请求和使用资源。
    """

    @abstractmethod
    def request_resource(self, resource_type: str, requirements: Dict[str, Any],
                         priority: int = 1, timeout: Optional[float] = None) -> Optional[str]:
        """请求资源"""

    @abstractmethod
    def release_resource(self, allocation_id: str) -> bool:
        """释放资源"""

    @abstractmethod
    def get_consumed_resources(self) -> List[ResourceAllocation]:
        """获取已消费的资源"""

    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""


class IResourceMonitor(ABC):
    """资源监控器接口

    定义资源监控器的标准行为规范。
    资源监控器负责收集和分析资源使用情况。
    """

    @abstractmethod
    def start_monitoring(self) -> bool:
        """开始监控"""

    @abstractmethod
    def stop_monitoring(self) -> bool:
        """停止监控"""

    @abstractmethod
    def collect_metrics(self) -> List[ResourceMetrics]:
        """收集指标"""

    @abstractmethod
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""

    @abstractmethod
    def get_metrics_history(self, resource_id: Optional[str] = None,
                            since: Optional[float] = None) -> List[ResourceMetrics]:
        """获取指标历史"""

    @abstractmethod
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常"""


class IResourceScheduler(ABC):
    """资源调度器接口

    定义资源调度器的标准行为规范。
    资源调度器负责智能调度和分配资源。
    """

    @abstractmethod
    def schedule_request(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """调度资源请求"""

    @abstractmethod
    def prioritize_requests(self, requests: List[ResourceRequest]) -> List[ResourceRequest]:
        """优先级排序"""

    @abstractmethod
    def balance_load(self) -> Dict[str, Any]:
        """负载均衡"""

    @abstractmethod
    def predict_demand(self, time_window: float) -> Dict[str, Any]:
        """预测需求"""

    @abstractmethod
    def get_schedule_status(self) -> Dict[str, Any]:
        """获取调度状态"""


class IResourceManager(ABC):
    """统一资源管理器接口

    定义统一资源管理器的标准行为规范。
    这是整个资源管理系统的核心接口。
    """

    @abstractmethod
    def register_provider(self, provider: IResourceProvider) -> bool:
        """注册资源提供者"""

    @abstractmethod
    def unregister_provider(self, resource_type: str) -> bool:
        """注销资源提供者"""

    @abstractmethod
    def get_providers(self) -> List[IResourceProvider]:
        """获取所有提供者"""

    @abstractmethod
    def register_consumer(self, consumer: IResourceConsumer) -> bool:
        """注册资源消费者"""

    @abstractmethod
    def unregister_consumer(self, consumer_id: str) -> bool:
        """注销资源消费者"""

    @abstractmethod
    def get_consumers(self) -> List[IResourceConsumer]:
        """获取所有消费者"""

    @abstractmethod
    def request_resource(self, consumer_id: str, resource_type: str,
                         requirements: Dict[str, Any], priority: int = 1) -> Optional[str]:
        """请求资源"""

    @abstractmethod
    def release_resource(self, allocation_id: str) -> bool:
        """释放资源"""

    @abstractmethod
    def get_resource_status(self) -> Dict[str, Any]:
        """获取资源状态"""

    @abstractmethod
    def optimize_resources(self) -> Dict[str, Any]:
        """优化资源使用"""

    @abstractmethod
    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""


class IResourceEventHandler(ABC):
    """资源事件处理器接口

    定义资源事件处理的标准行为规范。
    用于处理资源相关的事件。
    """

    @abstractmethod
    def on_resource_allocated(self, allocation: ResourceAllocation):
        """资源分配事件"""

    @abstractmethod
    def on_resource_released(self, allocation_id: str):
        """资源释放事件"""

    @abstractmethod
    def on_resource_failed(self, request: ResourceRequest, reason: str):
        """资源分配失败事件"""

    @abstractmethod
    def on_resource_anomaly(self, resource_id: str, anomaly: Dict[str, Any]):
        """资源异常事件"""


class IResourcePolicy(ABC):
    """资源策略接口

    定义资源策略的标准行为规范。
    资源策略用于定义资源分配和管理规则。
    """

    @abstractmethod
    def evaluate_request(self, request: ResourceRequest) -> Dict[str, Any]:
        """评估资源请求"""

    @abstractmethod
    def check_constraints(self, allocation: ResourceAllocation) -> bool:
        """检查约束条件"""

    @abstractmethod
    def calculate_priority(self, request: ResourceRequest) -> int:
        """计算优先级"""

    @abstractmethod
    def get_policy_config(self) -> Dict[str, Any]:
        """获取策略配置"""

# =============================================================================
# 工厂接口
# =============================================================================


class IResourceManagerFactory(ABC):
    """资源管理器工厂接口"""

    @abstractmethod
    def create_resource_manager(self, config: Dict[str, Any]) -> IResourceManager:
        """创建资源管理器"""

    @abstractmethod
    def create_provider(self, resource_type: str, config: Dict[str, Any]) -> IResourceProvider:
        """创建资源提供者"""

    @abstractmethod
    def create_consumer(self, consumer_type: str, config: Dict[str, Any]) -> IResourceConsumer:
        """创建资源消费者"""

    @abstractmethod
    def create_monitor(self, monitor_type: str, config: Dict[str, Any]) -> IResourceMonitor:
        """创建资源监控器"""

    @abstractmethod
    def create_scheduler(self, scheduler_type: str, config: Dict[str, Any]) -> IResourceScheduler:
        """创建资源调度器"""

# =============================================================================
# 适配器接口
# =============================================================================


class IResourceAdapter(ABC):
    """资源适配器接口

    定义资源适配器的标准行为规范。
    用于将不同类型的资源适配到统一接口。
    """

    @abstractmethod
    def adapt_resource_info(self, raw_info: Any) -> ResourceInfo:
        """适配资源信息"""

    @abstractmethod
    def adapt_request(self, raw_request: Any) -> ResourceRequest:
        """适配资源请求"""

    @abstractmethod
    def adapt_allocation(self, raw_allocation: Any) -> ResourceAllocation:
        """适配资源分配"""

    @abstractmethod
    def adapt_metrics(self, raw_metrics: Any) -> ResourceMetrics:
        """适配资源指标"""

# =============================================================================
# 配置接口
# =============================================================================


class IResourceConfig(Protocol):
    """资源配置协议"""

    def get_resource_config(self, resource_type: str) -> Dict[str, Any]:
        """获取资源配置"""
        ...

    def update_resource_config(self, resource_type: str, config: Dict[str, Any]) -> bool:
        """更新资源配置"""
        ...

    def validate_resource_config(self, resource_type: str, config: Dict[str, Any]) -> List[str]:
        """验证资源配置"""
        ...

# =============================================================================
# 异常定义
# =============================================================================


class ResourceError(Exception):
    """资源相关异常基类"""


class ResourceNotFoundError(ResourceError):
    """资源未找到异常"""


class ResourceAllocationError(ResourceError):
    """资源分配异常"""


class ResourceReleaseError(ResourceError):
    """资源释放异常"""


class ResourceQuotaExceededError(ResourceError):
    """资源配额超限异常"""


class ResourceTimeoutError(ResourceError):
    """资源超时异常"""


class ResourceUnavailableError(ResourceError):
    """资源不可用异常"""
