"""
prometheus_exporter 模块

提供 prometheus_exporter 相关功能和接口。
"""

import json
import logging
import os

import socket
# 条件导入prometheus_client以支持Windows兼容性
import threading
import time

# from prometheus_client import (
#     # Prometheus metrics imports will be added here
# )

from ..core.interfaces import IUnifiedInfrastructureInterface
from dataclasses import dataclass
from datetime import datetime
from ..core.adapters import InfrastructureAdapterFactory
from typing import Dict, Any, List, Optional
from .prometheus_integration import check_health, check_integration_class, check_prometheus_client, check_metric_system
"""
基础设施层 - 日志系统组件

prometheus_exporter 模块

日志系统相关的文件
提供日志系统相关的功能实现。

Prometheus指标导出器
将健康检查指标导出为Prometheus格式，支持Grafana监控
增强版：支持更多指标类型和Grafana仪表板配置
"""

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
        REGISTRY, CollectorRegistry, Info
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # 创建模拟类以避免导入错误
    Counter = None
    Gauge = None
    Histogram = None
    Summary = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None
    REGISTRY = None
    CollectorRegistry = None
    Info = None

    class MockMetric:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):

            pass

        def observe(self, *args, **kwargs):
            pass

    Counter = Gauge = Histogram = Summary = Info = MockMetric

logger = logging.getLogger(__name__)


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    description: str
    labels: List[str]
    metric_type: str  # counter, gauge, histogram, summary, info


@dataclass
class GrafanaDashboard:
    """Grafana仪表板配置"""
    title: str
    description: str
    panels: List[Dict[str, Any]]
    variables: List[Dict[str, Any]]
    refresh: str = "30s"


class HealthCheckPrometheusExporter(IUnifiedInfrastructureInterface):
    """健康检查Prometheus指标导出器"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        初始化导出器

        Args:
            registry: Prometheus注册表，None使用默认注册表
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, using mock metrics")

        self.registry = registry or REGISTRY
        self.metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()

        # 初始化适配器支持
        self._adapters: Dict[str, Any] = {}
        self._init_adapters()

        # 初始化指标
        self._init_metrics()

        # Grafana仪表板配置
        self._init_grafana_dashboard()

        self._initialized = False
        self._export_count = 0
        self._last_export_time = None

        logger.info("Prometheus exporter initialized with enhanced metrics and adapter support")

    def _init_metrics(self) -> None:
        """初始化Prometheus指标"""
        try:
            # 如果Prometheus不可用，提前返回，不初始化指标
            if not PROMETHEUS_AVAILABLE:
                logger.warning("Prometheus not available, skipping metric initialization")
                return

            # 按功能模块化初始化指标
            self._init_health_check_metrics()
            self._init_system_resource_metrics()
            self._init_cache_metrics()
            self._init_alert_metrics()
            self._init_dependency_metrics()
            self._init_system_info_metrics()

            logger.info("Enhanced Prometheus metrics initialized")

        except Exception as e:
            logger.error(f"Error initializing Prometheus metrics: {e}")

    def _init_health_check_metrics(self) -> None:
        """初始化健康检查相关指标"""
        # 健康检查状态指标
        self.metrics['health_status'] = Gauge(
            'rqa_health_status',
            'Health check status (1=healthy, 0=unhealthy, -1=error)',
            ['service', 'check_type', 'instance'],
            registry=self.registry
        )

        # 健康检查响应时间指标
        self.metrics['health_response_time'] = Histogram(
            'rqa_health_response_time_seconds',
            'Health check response time in seconds',
            ['service', 'check_type', 'instance'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )

        # 健康检查失败计数
        self.metrics['health_failures'] = Counter(
            'rqa_health_failures_total',
            'Total number of health check failures',
            ['service', 'check_type', 'error_code', 'instance'],
            registry=self.registry
        )

        # 健康检查成功计数
        self.metrics['health_successes'] = Counter(
            'rqa_health_successes_total',
            'Total number of successful health checks',
            ['service', 'check_type', 'instance'],
            registry=self.registry
        )

        # 健康检查频率指标
        self.metrics['health_check_frequency'] = Counter(
            'rqa_health_check_frequency_total',
            'Total number of health checks performed',
            ['service', 'check_type', 'instance'],
            registry=self.registry
        )

    def _init_system_resource_metrics(self) -> None:
        """初始化系统资源指标"""
        # CPU使用率
        self.metrics['system_cpu_usage'] = Gauge(
            'rqa_system_cpu_usage_percent',
            'System CPU usage percentage',
            ['instance', 'core'],
            registry=self.registry
        )

        # 内存使用情况
        self.metrics['system_memory_usage'] = Gauge(
            'rqa_system_memory_usage_bytes',
            'System memory usage in bytes',
            ['instance', 'type'],
            registry=self.registry
        )

        # 磁盘使用情况
        self.metrics['system_disk_usage'] = Gauge(
            'rqa_system_disk_usage_percent',
            'System disk usage percentage',
            ['instance', 'mountpoint', 'filesystem'],
            registry=self.registry
        )

    def _init_cache_metrics(self) -> None:
        """初始化缓存性能指标"""
        # 缓存命中率
        self.metrics['cache_hit_rate'] = Gauge(
            'rqa_cache_hit_rate',
            'Cache hit rate percentage',
            ['cache_type', 'instance'],
            registry=self.registry
        )

        # 缓存条目数量
        self.metrics['cache_entries'] = Gauge(
            'rqa_cache_entries_total',
            'Total number of cache entries',
            ['cache_type', 'instance'],
            registry=self.registry
        )

        # 缓存驱逐计数
        self.metrics['cache_evictions'] = Counter(
            'rqa_cache_evictions_total',
            'Total number of cache evictions',
            ['cache_type', 'policy', 'instance'],
            registry=self.registry
        )

    def _init_alert_metrics(self) -> None:
        """初始化告警指标"""
        # 活跃告警数量
        self.metrics['active_alerts'] = Gauge(
            'rqa_active_alerts',
            'Number of active alerts',
            ['severity', 'instance'],
            registry=self.registry
        )

        # 告警触发计数
        self.metrics['alert_triggered'] = Counter(
            'rqa_alerts_triggered_total',
            'Total number of alerts triggered',
            ['severity', 'rule_name', 'instance'],
            registry=self.registry
        )

    def _init_dependency_metrics(self) -> None:
        """初始化服务依赖指标"""
        # 依赖服务状态
        self.metrics['dependency_status'] = Gauge(
            'rqa_dependency_status',
            'Dependency service status (1=healthy, 0=unhealthy)',
            ['service', 'dependency', 'instance'],
            registry=self.registry
        )

        # 依赖服务响应时间
        self.metrics['dependency_response_time'] = Histogram(
            'rqa_dependency_response_time_seconds',
            'Dependency service response time',
            ['service', 'dependency', 'instance'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )

    def _init_system_info_metrics(self) -> None:
        """初始化系统信息指标"""
        # 系统信息
        self.metrics['system_info'] = Info(
            'rqa_system_info',
            'System information',
            ['instance', 'version', 'environment'],
            registry=self.registry
        )

    def _init_grafana_dashboard(self) -> None:
        """初始化Grafana仪表板配置"""
        panels = self._create_dashboard_panels()
        variables = self._create_dashboard_variables()

        self.grafana_dashboard = GrafanaDashboard(
            title="RQA2025 Health Check Dashboard",
            description="Comprehensive health check monitoring dashboard",
            refresh="30s",
            panels=panels,
            variables=variables
        )

    def _create_dashboard_panels(self) -> List[Dict[str, Any]]:
        """创建仪表板面板配置"""
        return [
            self._create_system_health_panel(),
            self._create_response_time_panel(),
            self._create_cache_performance_panel(),
            self._create_active_alerts_panel()
        ]

    def _create_system_health_panel(self) -> Dict[str, Any]:
        """创建系统健康概览面板"""
        return {
            "title": "System Health Overview",
            "type": "stat",
            "targets": [
                {"expr": "rqa_health_status", "legendFormat": "{{service}} - {{check_type}}"}
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "green", "value": 1}
                        ]
                    }
                }
            }
        }

    def _create_response_time_panel(self) -> Dict[str, Any]:
        """创建响应时间分布面板"""
        return {
            "title": "Response Time Distribution",
            "type": "heatmap",
            "targets": [
                {"expr": "rate(rqa_health_response_time_seconds_bucket[5m])",
                 "legendFormat": "{{le}}"}
            ]
        }

    def _create_cache_performance_panel(self) -> Dict[str, Any]:
        """创建缓存性能面板"""
        return {
            "title": "Cache Performance",
            "type": "timeseries",
            "targets": [
                {"expr": "rqa_cache_hit_rate", "legendFormat": "{{cache_type}} Hit Rate"}
            ]
        }

    def _create_active_alerts_panel(self) -> Dict[str, Any]:
        """创建活跃告警面板"""
        return {
            "title": "Active Alerts",
            "type": "stat",
            "targets": [
                {"expr": "rqa_active_alerts", "legendFormat": "{{severity}} Alerts"}
            ]
        }

    def _create_dashboard_variables(self) -> List[Dict[str, Any]]:
        """创建仪表板变量配置"""
        return [
            self._create_instance_variable(),
            self._create_service_variable()
        ]

    def _create_instance_variable(self) -> Dict[str, Any]:
        """创建实例变量"""
        return {
            "name": "instance",
            "type": "query",
            "query": "label_values(rqa_health_status, instance)",
            "label": "Instance"
        }

    def _create_service_variable(self) -> Dict[str, Any]:
        """创建服务变量"""
        return {
            "name": "service",
            "type": "query",
            "query": "label_values(rqa_health_status, service)",
            "label": "Service"
        }

    def record_health_check(
        self,
        service: str,
        check_type: str,
        status: str,
        response_time: float,
        error_code: Optional[str] = None,
        instance: str = "default"
    ):
        """
        记录健康检查指标

        Args:
            service: 服务名称
            check_type: 检查类型
            status: 检查状态
            response_time: 响应时间
            error_code: 错误代码
            instance: 实例标识
        """
        try:
            self._record_health_status(service, check_type, status, instance)
            self._record_response_time(service, check_type, response_time, instance)
            self._record_success_failure_count(service, check_type, status, error_code, instance)
            self._record_check_frequency(service, check_type, instance)

        except Exception as e:
            logger.error(f"Error recording health check metrics: {e}")

    def _record_health_status(self, service: str, check_type: str, status: str, instance: str) -> None:
        """记录健康状态指标"""
        status_value = 1 if status == "healthy" else (0 if status == "unhealthy" else -1)
        self.metrics['health_status'].labels(
            service=service,
            check_type=check_type,
            instance=instance
        ).set(status_value)

    def _record_response_time(self, service: str, check_type: str, response_time: float, instance: str) -> None:
        """记录响应时间指标"""
        self.metrics['health_response_time'].labels(
            service=service,
            check_type=check_type,
            instance=instance
        ).observe(response_time)

    def _record_success_failure_count(self, service: str, check_type: str, status: str,
                                      error_code: Optional[str], instance: str) -> None:
        """记录成功/失败计数指标"""
        if status == "healthy":
            self.metrics['health_successes'].labels(
                service=service,
                check_type=check_type,
                instance=instance
            ).inc()
        else:
            self.metrics['health_failures'].labels(
                service=service,
                check_type=check_type,
                error_code=error_code or "unknown",
                instance=instance
            ).inc()

    def _record_check_frequency(self, service: str, check_type: str, instance: str) -> None:
        """记录检查频率指标"""
        self.metrics['health_check_frequency'].labels(
            service=service,
            check_type=check_type,
            instance=instance
        ).inc()

    def record_system_metrics(self, host: str, cpu_percent: float,
                              memory_bytes: int, disk_usage: Dict[str, float],
                              instance: str = "default"):
        """
        记录系统指标

        Args:
            host: 主机名
            cpu_percent: CPU使用率
            memory_bytes: 内存使用量
            disk_usage: 磁盘使用情况
            instance: 实例标识
        """
        try:
            # CPU指标
            self.metrics['system_cpu_usage'].labels(
                instance=instance,
                core="total"
            ).set(cpu_percent)

            # 内存指标
            self.metrics['system_memory_usage'].labels(
                instance=instance,
                type="used"
            ).set(memory_bytes)

            # 磁盘指标
            for mountpoint, usage in disk_usage.items():
                self.metrics['system_disk_usage'].labels(
                    instance=instance,
                    mountpoint=mountpoint,
                    filesystem="ext4"  # 默认文件系统类型
                ).set(usage)

            # 系统信息
            self.metrics['system_info'].labels(
                instance=instance,
                version="2.1.0",
                environment="production"
            ).info({"host": host})

        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")

    def record_dependency_status(self, service: str, dependency: str,
                                 status: str, response_time: Optional[float] = None,
                                 instance: str = "default"):
        """
        记录依赖服务状态

        Args:
            service: 主服务名称
            dependency: 依赖服务名称
            status: 依赖状态
            response_time: 响应时间
            instance: 实例标识
        """
        try:
            # 状态指标
            status_value = 1 if status == "healthy" else 0
            self.metrics['dependency_status'].labels(
                service=service,
                dependency=dependency,
                instance=instance
            ).set(status_value)

            # 响应时间指标
            if response_time is not None:
                self.metrics['dependency_response_time'].labels(
                    service=service,
                    dependency=dependency,
                    instance=instance
                ).observe(response_time)

        except Exception as e:
            logger.error(f"Error recording dependency metrics: {e}")

    def record_cache_metrics(self, cache_type: str, hit_rate: float,
                             total_entries: int, evictions: int,
                             policy: str = "lru", instance: str = "default"):
        """
        记录缓存指标

        Args:
            cache_type: 缓存类型
            hit_rate: 命中率
            total_entries: 总条目数
            evictions: 驱逐次数
            policy: 缓存策略
            instance: 实例标识
        """
        try:
            # 命中率指标
            self.metrics['cache_hit_rate'].labels(
                cache_type=cache_type,
                instance=instance
            ).set(hit_rate)

            # 条目数指标
            self.metrics['cache_entries'].labels(
                cache_type=cache_type,
                instance=instance
            ).set(total_entries)

            # 驱逐次数指标
            self.metrics['cache_evictions'].labels(
                cache_type=cache_type,
                policy=policy,
                instance=instance
            ).inc(evictions)

        except Exception as e:
            logger.error(f"Error recording cache metrics: {e}")

    def record_alert_metrics(self, severity: str, rule_name: str,
                             active_count: int, instance: str = "default"):
        """
        记录告警指标

        Args:
            severity: 告警严重程度
            rule_name: 规则名称
            active_count: 活跃告警数量
            instance: 实例标识
        """
        try:
            # 活跃告警数量
            self.metrics['active_alerts'].labels(
                severity=severity,
                instance=instance
            ).set(active_count)

            # 告警触发计数
            if active_count > 0:
                self.metrics['alert_triggered'].labels(
                    severity=severity,
                    rule_name=rule_name,
                    instance=instance
                ).inc()

        except Exception as e:
            logger.error(f"Error recording alert metrics: {e}")

    def record_performance_metric(self, name: str, value: float,
                                  metadata: Optional[Dict[str, Any]] = None,
                                  instance: str = "default"):
        """
        记录性能指标

        Args:
            name: 指标名称
            value: 指标值
            metadata: 元数据
            instance: 实例标识
        """
        try:
            self._ensure_performance_metrics_initialized()
            self._record_main_performance_metric(name, value, instance)

            if metadata:
                self._record_performance_metadata(name, metadata, instance)

            logger.debug(f"Recorded performance metric: {name}={value}")

        except Exception as e:
            logger.error(f"Error recording performance metric {name}: {e}")

    def _ensure_performance_metrics_initialized(self) -> None:
        """确保性能指标已初始化"""
        if 'performance_metrics' not in self.metrics:
            self._initialize_performance_metrics_gauge()

    def _initialize_performance_metrics_gauge(self) -> None:
        """初始化性能指标Gauge"""
        try:
            # 使用唯一的指标名称避免冲突
            metric_name = f'rqa_performance_metrics_{id(self)}'
            self.metrics['performance_metrics'] = Gauge(
                metric_name,
                'Performance metrics',
                ['metric_name', 'instance'],
                registry=self.registry
            )
        except ValueError as ve:
            if "Duplicated timeseries" in str(ve):
                # 如果指标已存在，创建模拟指标避免重复注册
                self.metrics['performance_metrics'] = self._create_mock_gauge()
                logger.warning(f"Performance metrics gauge already exists, using mock: {ve}")
            else:
                raise ve

    def _create_mock_gauge(self):
        """创建模拟Gauge对象"""
        return type('MockGauge', (), {
            'labels': lambda **kwargs: type('MockLabels', (), {'set': lambda v: None})(),
            'set': lambda v: None
        })

    def _record_main_performance_metric(self, name: str, value: float, instance: str) -> None:
        """记录主要的性能指标"""
        self.metrics['performance_metrics'].labels(
            metric_name=name,
            instance=instance
        ).set(value)

    def _record_performance_metadata(self, name: str, metadata: Dict[str, Any], instance: str) -> None:
        """记录性能指标元数据"""
        for key, val in metadata.items():
            if isinstance(val, (str, int, float)):
                self._record_single_metadata_item(name, key, val, instance)

    def _record_single_metadata_item(self, metric_name: str, key: str, val: Any, instance: str) -> None:
        """记录单个元数据项"""
        metric_key = f"performance_metrics_{key}"

        if metric_key not in self.metrics:
            self._initialize_metadata_metric(metric_key, key)

        self.metrics[metric_key].labels(
            metric_name=metric_name,
            instance=instance
        ).set(float(val) if isinstance(val, (int, float)) else 0.0)

    def _initialize_metadata_metric(self, metric_key: str, key: str) -> None:
        """初始化元数据指标"""
        try:
            # 使用唯一的指标名称避免冲突
            unique_metric_name = f'rqa_{metric_key}_{id(self)}'
            self.metrics[metric_key] = Gauge(
                unique_metric_name,
                f'Performance metric metadata: {key}',
                ['metric_name', 'instance'],
                registry=self.registry
            )
        except ValueError as ve:
            if "Duplicated timeseries" in str(ve):
                # 创建模拟指标避免重复注册
                self.metrics[metric_key] = self._create_mock_gauge()
                logger.warning(f"Metadata metric gauge already exists, using mock: {ve}")
            else:
                raise ve

    def record_metric(self, name: str, value: float,
                      tags: Optional[Dict[str, str]] = None,
                      instance: str = "default"):
        """
        记录通用指标（兼容性方法）

        Args:
            name: 指标名称
            value: 指标值
            tags: 标签
            instance: 实例标识
        """
        self.record_performance_metric(name, value, tags, instance)

    def generate_metrics(self) -> bytes:
        """
        生成Prometheus指标数据

        Returns:
            Prometheus格式的指标数据
        """
        try:
            return generate_latest(self.registry)
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return b"# Error generating metrics\n"

    def get_metrics_content_type(self) -> str:
        """获取指标内容类型"""
        return CONTENT_TYPE_LATEST

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要

        Returns:
            指标摘要信息
        """
        try:
            summary = {}
            for metric_name, metric in self.metrics.items():
                if hasattr(metric, '_metrics'):
                    # 对于有标签的指标，统计标签组合数量
                    summary[metric_name] = {
                        'type': type(metric).__name__,
                        'label_combinations': len(metric._metrics)
                    }
                else:
                    summary[metric_name] = {
                        'type': type(metric).__name__,
                        'label_combinations': 1
                    }
            return {
                'total_metrics': len(self.metrics),
                'metrics': summary,
                'registry': str(self.registry),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {'error': str(e)}

    def export_grafana_dashboard(self, filepath: Optional[str] = None) -> str:
        """
        导出Grafana仪表板配置

        Args:
            filepath: 导出文件路径，None则返回JSON字符串

        Returns:
            仪表板配置JSON字符串
        """
        try:
            dashboard_json = {
                "dashboard": {
                    "title": self.grafana_dashboard.title,
                    "description": self.grafana_dashboard.description,
                    "refresh": self.grafana_dashboard.refresh,
                    "panels": self.grafana_dashboard.panels,
                    "templating": {},
                    "list": self.grafana_dashboard.variables,
                    "time": {},
                    "from": "now - 1h",
                    "to": "now",
                    "folderId": 0,
                    "overwrite": True,
                }
            }
            if filepath:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(dashboard_json, f, indent=2, ensure_ascii=False)
                logger.info(f"Grafana dashboard exported to: {filepath}")
                return filepath
            else:
                return json.dumps(dashboard_json, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error exporting Grafana dashboard: {e}")
            return ""

    def reset_metrics(self) -> None:
        """重置所有指标"""
        try:
            # 重新初始化指标
            self._init_metrics()
            logger.info("All metrics reset")
        except Exception as e:
            logger.error(f"Error resetting metrics: {e}")

    def get_instance_identifier(self) -> str:
        """
        获取实例标识符

        Returns:
            实例标识符
        """
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

    # =========================================================================
    # 适配器模式集成
    # =========================================================================

    def _init_adapters(self) -> None:
        """初始化适配器支持"""
        try:
            # 尝试创建监控适配器
            if InfrastructureAdapterFactory.has_adapter("monitoring"):
                self._adapters["monitoring"] = InfrastructureAdapterFactory.create_adapter(
                    "monitoring",
                    config={"exporter": "prometheus"}
                )
                logger.info("监控适配器初始化成功")
            else:
                logger.warning("监控适配器不可用")

            # 尝试创建缓存适配器
            if InfrastructureAdapterFactory.has_adapter("cache"):
                self._adapters["cache"] = InfrastructureAdapterFactory.create_adapter(
                    "cache",
                    config={"backend": "prometheus_metrics"}
                )
                logger.info("缓存适配器初始化成功")
            else:
                logger.warning("缓存适配器不可用")

        except Exception as e:
            logger.error(f"适配器初始化失败: {str(e)}")

    def get_adapter(self, adapter_type: str) -> Optional[Any]:
        """获取指定类型的适配器"""
        return self._adapters.get(adapter_type)

    def execute_adapter_operation(self, adapter_type: str, operation: str, **kwargs) -> Any:
        """通过适配器执行操作"""
        try:
            adapter = self.get_adapter(adapter_type)
            if not adapter:
                logger.warning(f"适配器不存在: {adapter_type}")
                return {"error": f"Adapter not found: {adapter_type}"}

            logger.debug(f"通过适配器执行操作: {adapter_type}.{operation}")
            return adapter.execute_operation(operation, **kwargs)

        except Exception as e:
            logger.error(f"适配器操作执行失败 {adapter_type}.{operation}: {str(e)}")
            return {"error": str(e)}

    async def execute_adapter_operation_async(self, adapter_type: str, operation: str, **kwargs) -> Any:
        """异步通过适配器执行操作"""
        try:
            adapter = self.get_adapter(adapter_type)
            if not adapter:
                logger.warning(f"适配器不存在: {adapter_type}")
                return {"error": f"Adapter not found: {adapter_type}"}

            logger.debug(f"异步通过适配器执行操作: {adapter_type}.{operation}")
            return await adapter.execute_operation_async(operation, **kwargs)

        except Exception as e:
            logger.error(f"异步适配器操作执行失败 {adapter_type}.{operation}: {str(e)}")
            return {"error": str(e)}

    def get_available_adapters(self) -> List[str]:
        """获取所有可用适配器类型"""
        return list(self._adapters.keys())

    def is_adapter_available(self, adapter_type: str) -> bool:
        """检查适配器是否可用"""
        adapter = self.get_adapter(adapter_type)
        return adapter is not None and adapter.is_service_available()

    def get_adapter_status(self, adapter_type: str) -> Dict[str, Any]:
        """获取适配器状态"""
        try:
            adapter = self.get_adapter(adapter_type)
            if not adapter:
                return {"available": False, "error": "Adapter not found"}

            status = adapter.get_service_status()
            return {
                "adapter_type": adapter_type,
                "available": True,
                "status": status
            }

        except Exception as e:
            logger.error(f"获取适配器状态失败 {adapter_type}: {str(e)}")
            return {
                "adapter_type": adapter_type,
                "available": False,
                "error": str(e)
            }


# 全局导出器实例
_prometheus_exporter: Optional[HealthCheckPrometheusExporter] = None
_exporter_lock = threading.Lock()


def monitor_prometheus_exporter() -> Dict[str, Any]:
    """监控Prometheus导出器状态"""
    try:
        health_check = check_health()
        exporter_efficiency = 1.0 if health_check["healthy"] else 0.0
        return {
            "healthy": health_check["healthy"],
            "exporter_metrics": {
                "service_name": "prometheus_exporter",
                "exporter_efficiency": exporter_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            }
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def validate_prometheus_exporter() -> Dict[str, Any]:
    """验证Prometheus导出器"""
    try:
        validation_results = {
            "exporter_validation": check_exporter_class(),
            "prometheus_validation": check_prometheus_integration(),
            "metrics_validation": check_metrics_system()
        }
        overall_valid = all(result.get("valid", False) for result in validation_results.values())
        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
