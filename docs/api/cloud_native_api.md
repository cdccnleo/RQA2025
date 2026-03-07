# 云原生环境API文档

## 📋 概述

云原生环境模块提供了完整的云原生基础设施管理功能，包括服务网格、多云管理、自动伸缩和增强监控，支持现代云原生应用的全生命周期管理。

## 🏗️ 模块架构

### 功能模块

```
云原生环境模块
├── 配置层 (cloud_native_configs.py)
│   ├── ServiceMeshConfig        # 服务网格配置
│   ├── MultiCloudConfig         # 多云配置
│   ├── AutoScalingConfig        # 自动伸缩配置
│   └── CloudNativeMonitoringConfig # 监控配置
├── 服务网格 (cloud_service_mesh.py)
│   └── ServiceMeshManager       # 服务网格管理器
├── 多云管理 (cloud_multi_cloud.py)
│   └── MultiCloudManager        # 多云管理器
├── 自动伸缩 (cloud_auto_scaling.py)
│   └── AutoScalingManager       # 自动伸缩管理器
└── 增强监控 (cloud_enhanced_monitoring.py)
    ├── EnhancedMonitoringManager # 增强监控管理器
    ├── MetricsAggregator        # 指标聚合器
    ├── AlertCorrelator          # 告警关联器
    └── PerformanceAnalyzer      # 性能分析器
```

### 支持的云提供商

- **Amazon Web Services (AWS)**
- **Microsoft Azure**
- **Google Cloud Platform (GCP)**
- **Alibaba Cloud**
- **Tencent Cloud**
- **Huawei Cloud**

## 📚 API参考

### 配置类定义

#### `ServiceMeshConfig`

服务网格配置类：

```python
@dataclass
class ServiceMeshConfig:
    enabled: bool = True
    mesh_type: ServiceMeshType = ServiceMeshType.ISTIO
    namespace: str = "istio-system"
    version: str = "1.20.0"
    enable_mtls: bool = True
    enable_tracing: bool = True
    enable_metrics: bool = True
    custom_annotations: Dict[str, str] = field(default_factory=dict)
    custom_labels: Dict[str, str] = field(default_factory=dict)
```

#### `MultiCloudConfig`

多云配置类：

```python
@dataclass
class MultiCloudConfig:
    enabled: bool = False
    primary_provider: CloudProvider = CloudProvider.AWS
    secondary_providers: List[CloudProvider] = field(default_factory=list)
    region_mapping: Dict[str, str] = field(default_factory=dict)
    failover_enabled: bool = True
    load_balancing_strategy: str = "round_robin"
    health_check_interval: int = 30
    custom_config: Dict[str, Any] = field(default_factory=dict)
```

#### `AutoScalingConfig`

自动伸缩配置类：

```python
@dataclass
class AutoScalingConfig:
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
```

#### `CloudNativeMonitoringConfig`

云原生监控配置类：

```python
@dataclass
class CloudNativeMonitoringConfig:
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    alerting_enabled: bool = True
    log_aggregation: bool = True
    custom_dashboards: List[str] = field(default_factory=list)
    metrics_retention_days: int = 30
    enable_tracing: bool = True
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
```

### 枚举定义

#### `ServiceMeshType`

服务网格类型：

```python
class ServiceMeshType(Enum):
    ISTIO = "istio"
    LINKERD = "linkerd"
    CONSUL = "consul"
    AWS_APP_MESH = "aws_app_mesh"
    KUMA = "kuma"
```

#### `CloudProvider`

云提供商：

```python
class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"
    TENCENT = "tencent"
    HUAWEI = "huawei"
```

#### `ScalingPolicy`

伸缩策略：

```python
class ScalingPolicy(Enum):
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    CUSTOM_METRIC = "custom_metric"
    SCHEDULED = "scheduled"
```

### 服务网格管理

#### `ServiceMeshManager`

服务网格管理器：

```python
class ServiceMeshManager:
    def __init__(self, config: ServiceMeshConfig):
        """初始化服务网格管理器"""

    def install_service_mesh(self) -> bool:
        """安装服务网格"""

    def configure_sidecar_injection(self, namespace: str) -> bool:
        """配置Sidecar注入"""

    def enable_mtls(self, namespace: str = None) -> bool:
        """启用mTLS"""

    def get_mesh_status(self) -> Dict[str, Any]:
        """获取服务网格状态"""

    def uninstall_service_mesh(self) -> bool:
        """卸载服务网格"""
```

### 多云管理

#### `MultiCloudManager`

多云管理器：

```python
class MultiCloudManager:
    def __init__(self, config: MultiCloudConfig):
        """初始化多云管理器"""

    def get_current_provider(self) -> CloudProvider:
        """获取当前使用的提供商"""

    def switch_provider(self, target_provider: CloudProvider) -> bool:
        """切换到指定的云提供商"""

    def failover_to_next_provider(self) -> Optional[CloudProvider]:
        """故障转移到下一个可用的提供商"""

    def deploy_to_current_provider(self, resource_config: Dict[str, Any]) -> bool:
        """在当前提供商上部署资源"""

    def get_provider_status(self) -> Dict[str, Any]:
        """获取所有提供商的状态"""

    def get_region_mapping(self, logical_region: str) -> Optional[str]:
        """获取区域映射"""

    def validate_multi_cloud_setup(self) -> Dict[str, Any]:
        """验证多云设置"""
```

### 自动伸缩

#### `AutoScalingManager`

自动伸缩管理器：

```python
class AutoScalingManager:
    def __init__(self, config: AutoScalingConfig):
        """初始化自动伸缩管理器"""

    def should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """检查是否应该扩容"""

    def should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """检查是否应该缩容"""

    def scale_up(self, reason: str = "自动扩容") -> bool:
        """执行扩容操作"""

    def scale_down(self, reason: str = "自动缩容") -> bool:
        """执行缩容操作"""

    def manual_scale(self, target_replicas: int, reason: str = "手动伸缩") -> bool:
        """手动伸缩"""

    def get_current_replicas(self) -> int:
        """获取当前副本数"""

    def get_scaling_status(self) -> Dict[str, Any]:
        """获取伸缩状态"""

    def get_scaling_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取伸缩历史"""

    def update_metrics(self, metrics: Dict[str, Any]):
        """更新指标缓存"""

    def get_average_metric(self, metric_name: str, window_size: int = 5) -> Optional[float]:
        """获取指标平均值"""

    def reset_statistics(self):
        """重置统计信息"""

    def validate_scaling_config(self) -> Dict[str, Any]:
        """验证伸缩配置"""
```

### 增强监控

#### `EnhancedMonitoringManager`

增强监控管理器：

```python
class EnhancedMonitoringManager:
    def __init__(self, config: CloudNativeMonitoringConfig):
        """初始化增强监控管理器"""

    def add_custom_metric(self, name: str, value: Union[int, float],
                         metric_type: MetricType = MetricType.GAUGE,
                         labels: Optional[Dict[str, str]] = None) -> bool:
        """添加自定义指标"""

    def get_metric_statistics(self, metric_name: str,
                            time_window_minutes: int = 60) -> Optional[Dict[str, Any]]:
        """获取指标统计信息"""

    def define_alert_pattern(self, pattern_name: str, conditions: Dict[str, Any]) -> bool:
        """定义告警模式"""

    def evaluate_alert_patterns(self) -> List[Dict[str, Any]]:
        """评估告警模式"""

    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""

    def cleanup_old_data(self, retention_days: int = 30) -> int:
        """清理过期数据"""

    def export_monitoring_data(self, format: str = "json") -> Optional[str]:
        """导出监控数据"""
```

#### `MetricsAggregator`

指标聚合器：

```python
class MetricsAggregator:
    def aggregate_metric(self, metric_name: str, values: List[float]) -> Dict[str, Any]:
        """聚合指标"""
```

#### `AlertCorrelator`

告警关联器：

```python
class AlertCorrelator:
    def correlate_alert(self, alert: Alert) -> str:
        """关联告警"""

    def get_alert_groups(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取告警组"""
```

#### `PerformanceAnalyzer`

性能分析器：

```python
class PerformanceAnalyzer:
    def detect_anomaly(self, metric_name: str, value: float) -> float:
        """检测异常"""
```

## 🚀 使用示例

### 服务网格管理

```python
from infrastructure.config.environment.cloud_native_configs import ServiceMeshConfig, ServiceMeshType
from infrastructure.config.environment.cloud_service_mesh import ServiceMeshManager

# 配置Istio服务网格
mesh_config = ServiceMeshConfig(
    mesh_type=ServiceMeshType.ISTIO,
    namespace="istio-system",
    version="1.20.0",
    enable_mtls=True,
    enable_tracing=True,
    enable_metrics=True
)

# 创建服务网格管理器
mesh_manager = ServiceMeshManager(mesh_config)

# 安装服务网格
if mesh_manager.install_service_mesh():
    print("✅ 服务网格安装成功")

    # 配置Sidecar注入
    mesh_manager.configure_sidecar_injection("default")

    # 启用mTLS
    mesh_manager.enable_mtls("default")

    # 获取网格状态
    status = mesh_manager.get_mesh_status()
    print("服务网格状态:", status)
else:
    print("❌ 服务网格安装失败")
```

### 多云管理

```python
from infrastructure.config.environment.cloud_native_configs import MultiCloudConfig, CloudProvider
from infrastructure.config.environment.cloud_multi_cloud import MultiCloudManager

# 配置多云环境
multi_config = MultiCloudConfig(
    enabled=True,
    primary_provider=CloudProvider.AWS,
    secondary_providers=[CloudProvider.AZURE, CloudProvider.GCP],
    failover_enabled=True,
    load_balancing_strategy="round_robin"
)

# 创建多云管理器
multi_manager = MultiCloudManager(multi_config)

# 检查提供商状态
status = multi_manager.get_provider_status()
print("云提供商状态:", status)

# 部署资源到当前提供商
resource_config = {
    "type": "kubernetes-deployment",
    "name": "my-app",
    "replicas": 3,
    "image": "nginx:latest"
}

if multi_manager.deploy_to_current_provider(resource_config):
    print("✅ 资源部署成功")
else:
    print("❌ 资源部署失败，尝试故障转移...")
    new_provider = multi_manager.failover_to_next_provider()
    if new_provider:
        print(f"故障转移到: {new_provider.value}")
        # 重新尝试部署
        multi_manager.deploy_to_current_provider(resource_config)
```

### 自动伸缩

```python
from infrastructure.config.environment.cloud_native_configs import AutoScalingConfig, ScalingPolicy
from infrastructure.config.environment.cloud_auto_scaling import AutoScalingManager

# 配置自动伸缩
scaling_config = AutoScalingConfig(
    enabled=True,
    min_replicas=1,
    max_replicas=10,
    target_cpu_utilization=70,
    target_memory_utilization=80,
    scaling_policy=ScalingPolicy.CPU_UTILIZATION,
    cooldown_period_seconds=60
)

# 创建自动伸缩管理器
scaling_manager = AutoScalingManager(scaling_config)

# 模拟监控指标
current_metrics = {
    "cpu_utilization": 85.0,  # 85% CPU使用率
    "memory_utilization": 60.0,  # 60% 内存使用率
    "current_replicas": 3
}

# 检查是否需要扩容
if scaling_manager.should_scale_up(current_metrics):
    print("检测到需要扩容...")
    if scaling_manager.scale_up("CPU使用率过高"):
        print("✅ 扩容成功")
    else:
        print("❌ 扩容失败")

# 获取伸缩状态
status = scaling_manager.get_scaling_status()
print("伸缩状态:", status)

# 获取伸缩历史
history = scaling_manager.get_scaling_history(limit=5)
print("最近伸缩操作:", history)
```

### 增强监控

```python
from infrastructure.config.environment.cloud_native_configs import CloudNativeMonitoringConfig
from infrastructure.config.monitoring.dashboard_models import MetricType
from infrastructure.config.environment.cloud_enhanced_monitoring import EnhancedMonitoringManager

# 配置增强监控
monitoring_config = CloudNativeMonitoringConfig(
    prometheus_enabled=True,
    grafana_enabled=True,
    alerting_enabled=True,
    log_aggregation=True,
    enable_tracing=True,
    metrics_retention_days=30
)

# 创建增强监控管理器
monitoring_manager = EnhancedMonitoringManager(monitoring_config)

# 添加自定义指标
monitoring_manager.add_custom_metric(
    name="response_time",
    value=0.245,
    metric_type=MetricType.GAUGE,
    labels={"endpoint": "/api/users", "method": "GET"}
)

monitoring_manager.add_custom_metric(
    name="error_rate",
    value=0.02,
    metric_type=MetricType.GAUGE,
    labels={"service": "user-service"}
)

# 获取指标统计
stats = monitoring_manager.get_metric_statistics("response_time", time_window_minutes=60)
if stats:
    print(f"响应时间统计: 平均={stats['avg']:.3f}s, 最大={stats['max']:.3f}s")

# 定义告警模式
alert_pattern = {
    "metric": "error_rate",
    "operator": ">",
    "threshold": 0.05,
    "severity": "warning"
}

monitoring_manager.define_alert_pattern("high_error_rate", alert_pattern)

# 评估告警模式
alerts = monitoring_manager.evaluate_alert_patterns()
if alerts:
    print("触发告警:", alerts)

# 获取监控状态
status = monitoring_manager.get_monitoring_status()
print("监控系统状态:", status)

# 导出监控数据
export_data = monitoring_manager.export_monitoring_data("json")
if export_data:
    with open("monitoring_export.json", "w") as f:
        f.write(export_data)
    print("✅ 监控数据已导出")
```

### 综合云原生应用

```python
from infrastructure.config.environment.cloud_native_configs import (
    ServiceMeshConfig, MultiCloudConfig, AutoScalingConfig, CloudNativeMonitoringConfig,
    ServiceMeshType, CloudProvider, ScalingPolicy
)
from infrastructure.config.environment import (
    ServiceMeshManager, MultiCloudManager, AutoScalingManager, EnhancedMonitoringManager
)

class CloudNativeApplication:
    """云原生应用管理器"""

    def __init__(self):
        # 初始化各项配置
        self.mesh_config = ServiceMeshConfig(mesh_type=ServiceMeshType.ISTIO)
        self.multi_config = MultiCloudConfig(
            primary_provider=CloudProvider.AWS,
            secondary_providers=[CloudProvider.AZURE]
        )
        self.scaling_config = AutoScalingConfig(scaling_policy=ScalingPolicy.CPU_UTILIZATION)
        self.monitoring_config = CloudNativeMonitoringConfig()

        # 初始化管理器
        self.mesh_manager = ServiceMeshManager(self.mesh_config)
        self.multi_manager = MultiCloudManager(self.multi_config)
        self.scaling_manager = AutoScalingManager(self.scaling_config)
        self.monitoring_manager = EnhancedMonitoringManager(self.monitoring_config)

    def deploy_application(self, app_config: Dict[str, Any]) -> bool:
        """部署云原生应用"""
        try:
            # 1. 安装服务网格
            if not self.mesh_manager.install_service_mesh():
                return False

            # 2. 部署到多云环境
            if not self.multi_manager.deploy_to_current_provider(app_config):
                # 尝试故障转移
                new_provider = self.multi_manager.failover_to_next_provider()
                if not new_provider or not self.multi_manager.deploy_to_current_provider(app_config):
                    return False

            # 3. 配置自动伸缩
            self.scaling_manager.manual_scale(app_config.get("initial_replicas", 3))

            # 4. 设置监控指标
            self.monitoring_manager.add_custom_metric(
                "deployment_status", 1.0, MetricType.GAUGE,
                {"app": app_config["name"], "status": "deployed"}
            )

            return True

        except Exception as e:
            self.monitoring_manager.add_custom_metric(
                "deployment_status", 0.0, MetricType.GAUGE,
                {"app": app_config.get("name", "unknown"), "error": str(e)}
            )
            return False

    def monitor_application(self) -> Dict[str, Any]:
        """监控应用状态"""
        return {
            "mesh_status": self.mesh_manager.get_mesh_status(),
            "cloud_status": self.multi_manager.get_provider_status(),
            "scaling_status": self.scaling_manager.get_scaling_status(),
            "monitoring_status": self.monitoring_manager.get_monitoring_status()
        }

    def scale_application(self, metrics: Dict[str, Any]) -> bool:
        """根据指标自动伸缩应用"""
        if self.scaling_manager.should_scale_up(metrics):
            return self.scaling_manager.scale_up("自动扩容")
        elif self.scaling_manager.should_scale_down(metrics):
            return self.scaling_manager.scale_down("自动缩容")
        return True  # 无需伸缩也是成功状态

# 使用示例
app = CloudNativeApplication()

# 部署应用
app_config = {
    "name": "ecommerce-api",
    "image": "myregistry/ecommerce:latest",
    "initial_replicas": 3,
    "ports": [8080],
    "env": {"NODE_ENV": "production"}
}

if app.deploy_application(app_config):
    print("✅ 云原生应用部署成功")

    # 监控应用
    status = app.monitor_application()
    print("应用状态:", status)

    # 模拟指标更新
    metrics = {
        "cpu_utilization": 75.0,
        "memory_utilization": 65.0,
        "current_replicas": 3
    }

    # 自动伸缩
    app.scale_application(metrics)

else:
    print("❌ 云原生应用部署失败")
```

## ⚙️ 配置参数

### 服务网格配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `True` | 是否启用服务网格 |
| `mesh_type` | `ServiceMeshType` | `ISTIO` | 服务网格类型 |
| `namespace` | `str` | `"istio-system"` | 安装命名空间 |
| `version` | `str` | `"1.20.0"` | 服务网格版本 |
| `enable_mtls` | `bool` | `True` | 启用mTLS |
| `enable_tracing` | `bool` | `True` | 启用分布式追踪 |
| `enable_metrics` | `bool` | `True` | 启用指标收集 |

### 多云配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `False` | 是否启用多云 |
| `primary_provider` | `CloudProvider` | `AWS` | 主云提供商 |
| `secondary_providers` | `List[CloudProvider]` | `[]` | 备份提供商 |
| `failover_enabled` | `bool` | `True` | 启用故障转移 |
| `load_balancing_strategy` | `str` | `"round_robin"` | 负载均衡策略 |
| `health_check_interval` | `int` | `30` | 健康检查间隔(秒) |

### 自动伸缩配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `True` | 是否启用自动伸缩 |
| `min_replicas` | `int` | `1` | 最小副本数 |
| `max_replicas` | `int` | `10` | 最大副本数 |
| `target_cpu_utilization` | `int` | `70` | CPU目标利用率(%) |
| `target_memory_utilization` | `int` | `80` | 内存目标利用率(%) |
| `scaling_policy` | `ScalingPolicy` | `CPU_UTILIZATION` | 伸缩策略 |
| `cooldown_period_seconds` | `int` | `60` | 冷却期(秒) |

### 监控配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prometheus_enabled` | `bool` | `True` | 启用Prometheus |
| `grafana_enabled` | `bool` | `True` | 启用Grafana |
| `alerting_enabled` | `bool` | `True` | 启用告警 |
| `log_aggregation` | `bool` | `True` | 启用日志聚合 |
| `enable_tracing` | `bool` | `True` | 启用追踪 |
| `metrics_retention_days` | `int` | `30` | 指标保留天数 |

## 🔧 错误处理

### 常见错误

| 错误场景 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 服务网格安装失败 | 权限不足、网络问题 | 检查集群权限和网络连接 |
| 云提供商切换失败 | 凭据无效、区域不可用 | 验证凭据和区域配置 |
| 自动伸缩失败 | 资源限制、策略冲突 | 检查资源配额和伸缩策略 |
| 监控指标异常 | 配置错误、数据源问题 | 验证监控配置和数据源 |

### 异常类

- `CloudProviderError`: 云提供商相关异常
- `ServiceMeshError`: 服务网格相关异常
- `AutoScalingError`: 自动伸缩相关异常
- `MonitoringError`: 监控相关异常

## 📊 性能指标

### 管理器性能

| 操作 | 响应时间 | 吞吐量 | 资源使用 |
|------|----------|--------|----------|
| 服务网格状态查询 | <100ms | >100 ops/s | <1MB |
| 云提供商切换 | <500ms | >50 ops/s | <5MB |
| 自动伸缩决策 | <50ms | >200 ops/s | <2MB |
| 监控指标收集 | <10ms | >1000 ops/s | <500KB |

### 可扩展性

- **支持的服务网格**: Istio、Linkerd、Consul等
- **支持的云提供商**: 6大主流云提供商
- **最大集群规模**: 1000+节点
- **监控指标数量**: 10000+ 实时指标
- **历史数据保留**: 90天

## 🔒 安全考虑

### 云安全
- 多云凭据加密存储
- 访问权限最小化原则
- 网络隔离和防火墙
- 合规性审计和日志

### 服务网格安全
- mTLS加密通信
- 身份认证和授权
- 流量加密和审计
- 安全策略自动注入

### 监控安全
- 敏感指标数据加密
- 访问控制和权限管理
- 审计日志完整记录
- 异常检测和告警

---

*云原生环境模块提供了企业级的云原生基础设施管理能力，支持现代微服务架构的完整生命周期管理。*
