
# ==================== 枚举定义 ====================

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List
"""
云原生环境配置类

定义云原生环境相关的配置数据结构
"""


class ServiceMeshType(Enum):
    """服务网格类型"""
    ISTIO = "istio"
    LINKERD = "linkerd"
    CONSUL = "consul"
    AWS_APP_MESH = "aws_app_mesh"
    KUMA = "kuma"


class CloudProvider(Enum):
    """云提供商"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"
    TENCENT = "tencent"
    HUAWEI = "huawei"


class ScalingPolicy(Enum):
    """自动伸缩策略"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    CUSTOM_METRIC = "custom_metric"
    SCHEDULED = "scheduled"

# ==================== 配置类定义 ====================


@dataclass
class ServiceMeshConfig:
    """服务网格配置"""

    enabled: bool = True
    mesh_type: ServiceMeshType = ServiceMeshType.ISTIO
    namespace: str = "istio-system"
    version: str = "1.20.0"
    enable_mtls: bool = True
    enable_tracing: bool = True
    enable_metrics: bool = True
    custom_annotations: Dict[str, str] = field(default_factory=dict)
    custom_labels: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """后初始化处理"""
        if not isinstance(self.mesh_type, ServiceMeshType):
            self.mesh_type = ServiceMeshType(self.mesh_type)


@dataclass
class MultiCloudConfig:
    """多云配置"""

    enabled: bool = False
    primary_provider: CloudProvider = CloudProvider.AWS
    secondary_providers: List[CloudProvider] = field(default_factory=list)
    region_mapping: Dict[str, str] = field(default_factory=dict)
    failover_enabled: bool = True
    load_balancing_strategy: str = "round_robin"
    health_check_interval: int = 30
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """后初始化处理"""
        if not isinstance(self.primary_provider, CloudProvider):
            self.primary_provider = CloudProvider(self.primary_provider)

        self.secondary_providers = [
            CloudProvider(p) if not isinstance(p, CloudProvider) else p
            for p in self.secondary_providers
        ]


@dataclass
class AutoScalingConfig:
    """自动伸缩配置"""

    enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    scale_up_threshold: int = 80
    scale_down_threshold: int = 30
    stabilization_window_seconds: int = 300
    scaling_policy: ScalingPolicy = ScalingPolicy.CPU_UTILIZATION
    custom_metrics: List[str] = field(default_factory=list)
    cooldown_period_seconds: int = 60

    def __post_init__(self):
        """后初始化处理"""
        if not isinstance(self.scaling_policy, ScalingPolicy):
            self.scaling_policy = ScalingPolicy(self.scaling_policy)


@dataclass
class CloudNativeMonitoringConfig:
    """云原生监控配置"""

    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    alerting_enabled: bool = True
    log_aggregation: bool = True
    custom_dashboards: List[str] = field(default_factory=list)
    metrics_retention_days: int = 30
    enable_tracing: bool = True
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """后初始化处理"""
        if self.custom_dashboards is None:
            self.custom_dashboards = []
        if self.custom_metrics is None:
            self.custom_metrics = {}




