"""
应用监控配置

定义应用监控相关的配置类
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, replace


@dataclass
class AlertHandler:
    """告警处理器配置"""
    name: str
    handler_type: str = "email"
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    handler: Optional[Any] = None


@dataclass
class InfluxDBConfig:
    """InfluxDB配置"""
    host: str = "localhost"
    port: int = 8086
    database: str = "health_metrics"
    username: Optional[str] = None
    password: Optional[str] = None
    ssl: bool = False
    timeout: int = 10
    enabled: bool = True


@dataclass
class PrometheusConfig:
    """Prometheus配置"""
    gateway_url: str = "http://localhost:9091"
    job_name: str = "health_monitor"
    instance: str = "localhost:8000"
    timeout: int = 10
    labels: Dict[str, str] = field(default_factory=dict)
    registry: Optional[Any] = None

    def with_registry(self, registry: Any) -> "PrometheusConfig":
        """返回带有新的 registry 的 Prometheus 配置"""
        return replace(self, registry=registry)


@dataclass
class ApplicationMonitorConfig:
    """应用监控配置"""
    app_name: str = "rqa2025"
    service_name: str = "rqa2025"
    check_interval: int = 60
    enabled_checks: List[str] = field(default_factory=lambda: ["cpu", "memory", "disk"])
    alert_handlers: List[AlertHandler] = field(default_factory=list)
    influx_config: Optional[InfluxDBConfig] = None
    prometheus_config: PrometheusConfig = field(default_factory=PrometheusConfig)
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "cpu": {"warning": 70.0, "critical": 90.0},
        "memory": {"warning": 80.0, "critical": 95.0},
        "disk": {"warning": 85.0, "critical": 95.0}
    })
    custom_checks: Dict[str, Callable] = field(default_factory=dict)
    log_level: str = "INFO"
    enable_metrics: bool = True
    sample_rate: float = 1.0
    retention_policy: str = "30d"
    influx_client_mock: Optional[Any] = None
    skip_thread: bool = False

    def __post_init__(self) -> None:
        # 兼容旧字段：如果只提供 service_name 未提供 app_name，则保持一致
        if not self.app_name:
            self.app_name = self.service_name
        if not isinstance(self.alert_handlers, list):
            self.alert_handlers = list(self.alert_handlers or [])
        if self.prometheus_config is None:
            self.prometheus_config = PrometheusConfig()

    @classmethod
    def create_default(cls, **overrides: Any) -> "ApplicationMonitorConfig":
        """创建默认配置，支持传入覆盖参数"""
        registry = overrides.pop("registry", None)
        prometheus_config = overrides.pop("prometheus_config", None)
        if prometheus_config is None:
            prometheus_config = PrometheusConfig()
        if registry is not None:
            prometheus_config = prometheus_config.with_registry(registry)
        if "service_name" not in overrides and "app_name" in overrides:
            overrides["service_name"] = overrides["app_name"]
        defaults = {
            "app_name": "rqa2025",
            "service_name": "rqa2025",
        }
        defaults.update(overrides)
        return cls(prometheus_config=prometheus_config, **defaults)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于调试或序列化"""
        return {
            "app_name": self.app_name,
            "service_name": self.service_name,
            "check_interval": self.check_interval,
            "enabled_checks": list(self.enabled_checks),
            "alert_handlers": [handler.__dict__ for handler in self.alert_handlers],
            "influxdb_config": self.influx_config.__dict__ if self.influx_config else None,
            "prometheus_config": self.prometheus_config.__dict__ if self.prometheus_config else None,
            "thresholds": self.thresholds,
            "custom_checks": list(self.custom_checks.keys()),
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "sample_rate": self.sample_rate,
            "retention_policy": self.retention_policy,
            "skip_thread": self.skip_thread,
        }
