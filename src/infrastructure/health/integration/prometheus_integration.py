"""
prometheus_integration 模块

提供 prometheus_integration 相关功能和接口。
"""

import json
import logging

import threading
import time

from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry
# from prometheus_client import (
#     # Prometheus integration imports will be added here
# )

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
"""
基础设施层 - 日志系统组件

prometheus_integration 模块

日志系统相关的文件
提供日志系统相关的功能实现。

Prometheus监控集成模块

为健康检查模块提供Prometheus格式的指标导出，支持自定义指标、标签和告警规则。
"""

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):

    """指标类型枚举"""
    COUNTER = "counter"      # 计数器
    GAUGE = "gauge"          # 仪表盘
    HISTOGRAM = "histogram"  # 直方图
    SUMMARY = "summary"      # 摘要


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # 直方图分桶
    quantiles: Optional[List[float]] = None  # 摘要分位数


class PrometheusIntegration:
    """Prometheus监控集成"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Prometheus集成

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 初始化验证
        if not self._validate_prometheus_availability():
            return

        # 初始化基础组件
        self._initialize_basic_components()

        # 初始化健康检查指标
        self._initialize_health_metrics()

        # 初始化自定义指标存储
        self._initialize_custom_metrics_storage()

        # 启动指标更新线程
        self._start_update_thread()

        logger.info("Prometheus监控集成已初始化")

    def _validate_prometheus_availability(self) -> bool:
        """验证Prometheus可用性"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus客户端库未安装，监控功能将不可用")
            self._enabled = False
            return False

        self._enabled = True
        return True

    def _initialize_basic_components(self) -> None:
        """初始化基础组件"""
        # 指标注册表
        self._registry = CollectorRegistry()

        # 指标存储
        self._metrics: Dict[str, Any] = {}

        # 指标更新线程
        self._update_thread = None
        self._running = False

    def _initialize_health_metrics(self) -> None:
        """初始化健康检查指标"""
        self._health_metrics = {}

        # 基础健康指标
        self._init_basic_health_metrics()

        # 缓存相关指标
        self._init_cache_metrics()

        # 性能指标
        self._init_performance_metrics()

        # 告警指标
        self._init_alert_metrics()

    def _init_basic_health_metrics(self) -> None:
        """初始化基础健康指标"""
        self._health_metrics.update({
            'health_check_total': Counter(
                'rqa2025_health_check_total',
                'Total number of health checks',
                ['service_name', 'status', 'check_type'],
                registry=self._registry
            ),

            'health_check_duration_seconds': Histogram(
                'rqa2025_health_check_duration_seconds',
                'Health check duration in seconds',
                ['service_name', 'check_type'],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                registry=self._registry
            ),

            'health_status': Gauge(
                'rqa2025_health_status',
                'Current health status (1=healthy, 0=unhealthy)',
                ['service_name', 'check_type'],
                registry=self._registry
            ),

            'service_response_time_seconds': Histogram(
                'rqa2025_service_response_time_seconds',
                'Service response time in seconds',
                ['service_name', 'endpoint'],
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                registry=self._registry
            )
        })

    def _init_cache_metrics(self) -> None:
        """初始化缓存相关指标"""
        self._health_metrics.update({
            'cache_hit_ratio': Gauge(
                'rqa2025_cache_hit_ratio',
                'Cache hit ratio (0.0 to 1.0)',
                ['cache_level', 'cache_name'],
                registry=self._registry
            ),

            'cache_entries_total': Gauge(
                'rqa2025_cache_entries_total',
                'Total number of cache entries',
                ['cache_level', 'cache_name'],
                registry=self._registry
            ),

            'cache_memory_bytes': Gauge(
                'rqa2025_cache_memory_bytes',
                'Cache memory usage in bytes',
                ['cache_level', 'cache_name'],
                registry=self._registry
            )
        })

    def _init_performance_metrics(self) -> None:
        """初始化性能指标"""
        self._health_metrics['performance_score'] = Gauge(
            'rqa2025_performance_score',
            'Performance score (0.0 to 1.0)',
            ['service_name', 'metric_type'],
            registry=self._registry
        )

    def _init_alert_metrics(self) -> None:
        """初始化告警指标"""
        self._health_metrics['alert_count'] = Counter(
            'rqa2025_alert_count_total',
            'Total number of alerts',
            ['alert_level', 'alert_type', 'service_name'],
            registry=self._registry
        )

    def _initialize_custom_metrics_storage(self) -> None:
        """初始化自定义指标存储"""
        self._custom_metrics: Dict[str, Any] = {}

    def register_custom_metric(self, definition: MetricDefinition) -> bool:
        """
        注册自定义指标

        Args:
            definition: 指标定义

        Returns:
            是否注册成功
        """
        if not self._validate_registration_request(definition):
            return False

        try:
            metric_name = self._generate_metric_name(definition.name)
            metric = self._create_metric_by_type(definition, metric_name)

            if metric is None:
                return False

            self._store_custom_metric(metric_name, metric)
            logger.info(f"自定义指标已注册: {metric_name}")
            return True

        except Exception as e:
            logger.error(f"注册自定义指标失败: {e}")
            return False

    def _validate_registration_request(self, definition: MetricDefinition) -> bool:
        """验证注册请求"""
        if not self._enabled:
            return False

        if not definition or not definition.name:
            logger.error("指标定义无效或缺少名称")
            return False

        return True

    def _generate_metric_name(self, base_name: str) -> str:
        """生成指标名称"""
        return f"rqa2025_custom_{base_name}"

    def _create_metric_by_type(self, definition: MetricDefinition, metric_name: str) -> Optional[Any]:
        """根据类型创建指标"""
        metric_type = definition.metric_type

        if metric_type == MetricType.COUNTER:
            return self._create_counter_metric(definition, metric_name)
        elif metric_type == MetricType.GAUGE:
            return self._create_gauge_metric(definition, metric_name)
        elif metric_type == MetricType.HISTOGRAM:
            return self._create_histogram_metric(definition, metric_name)
        elif metric_type == MetricType.SUMMARY:
            return self._create_summary_metric(definition, metric_name)
        else:
            logger.error(f"不支持的指标类型: {metric_type}")
            return None

    def _create_counter_metric(self, definition: MetricDefinition, metric_name: str) -> Counter:
        """创建计数器指标"""
        return Counter(
            metric_name,
            definition.description,
            definition.labels,
            registry=self._registry
        )

    def _create_gauge_metric(self, definition: MetricDefinition, metric_name: str) -> Gauge:
        """创建仪表盘指标"""
        return Gauge(
            metric_name,
            definition.description,
            definition.labels,
            registry=self._registry
        )

    def _create_histogram_metric(self, definition: MetricDefinition, metric_name: str) -> Histogram:
        """创建直方图指标"""
        return Histogram(
            metric_name,
            definition.description,
            definition.labels,
            buckets=definition.buckets or [0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self._registry
        )

    def _create_summary_metric(self, definition: MetricDefinition, metric_name: str) -> Summary:
        """创建摘要指标"""
        return Summary(
            metric_name,
            definition.description,
            definition.labels,
            quantiles=definition.quantiles or [0.5, 0.9, 0.95, 0.99],
            registry=self._registry
        )

    def _store_custom_metric(self, metric_name: str, metric: Any) -> None:
        """存储自定义指标"""
        self._custom_metrics[metric_name] = metric

    def record_health_check(self, service_name: str, status: str,
                            check_type: str, duration: float):
        """
        记录健康检查指标

        Args:
            service_name: 服务名称
            status: 检查状态
            check_type: 检查类型
            duration: 检查耗时（秒）
        """
        if not self._enabled:
            return

        try:
            # 记录总次数
            self._health_metrics['health_check_total'].labels(
                service_name=service_name,
                status=status,
                check_type=check_type
            ).inc()

            # 记录耗时
            self._health_metrics['health_check_duration_seconds'].labels(
                service_name=service_name,
                check_type=check_type
            ).observe(duration)

            # 更新状态
            status_value = 1 if status == 'healthy' else 0
            self._health_metrics['health_status'].labels(
                service_name=service_name,
                check_type=check_type
            ).set(status_value)

        except Exception as e:
            logger.error(f"记录健康检查指标失败: {e}")

    def record_service_response_time(self, service_name: str, endpoint: str,

                                     response_time: float):
        """
        记录服务响应时间

        Args:
            service_name: 服务名称
            endpoint: 端点
            response_time: 响应时间（秒）
        """
        if not self._enabled:
            return

        try:
            self._health_metrics['service_response_time_seconds'].labels(
                service_name=service_name,
                endpoint=endpoint
            ).observe(response_time)

        except Exception as e:
            logger.error(f"记录服务响应时间失败: {e}")

    def update_cache_metrics(self, cache_level: str, cache_name: str,

                             hit_ratio: float, entries_count: int, memory_bytes: int):
        """
        更新缓存指标

        Args:
            cache_level: 缓存级别
            cache_name: 缓存名称
            hit_ratio: 命中率
            entries_count: 条目数量
            memory_bytes: 内存使用（字节）
        """
        if not self._enabled:
            return

        try:
            self._health_metrics['cache_hit_ratio'].labels(
                cache_level=cache_level,
                cache_name=cache_name
            ).set(hit_ratio)

            self._health_metrics['cache_entries_total'].labels(
                cache_level=cache_level,
                cache_name=cache_name
            ).set(entries_count)

            self._health_metrics['cache_memory_bytes'].labels(
                cache_level=cache_level,
                cache_name=cache_name
            ).set(memory_bytes)

        except Exception as e:
            logger.error(f"更新缓存指标失败: {e}")

    def update_performance_score(self, service_name: str, metric_type: str, score: float):
        """
        更新性能评分

        Args:
            service_name: 服务名称
            metric_type: 指标类型
            score: 评分（0.0 - 1.0）
        """
        if not self._enabled:
            return

        try:
            self._health_metrics['performance_score'].labels(
                service_name=service_name,
                metric_type=metric_type
            ).set(score)

        except Exception as e:
            logger.error(f"更新性能评分失败: {e}")

    def record_alert(self, alert_level: str, alert_type: str, service_name: str):
        """
        记录告警

        Args:
            alert_level: 告警级别
            alert_type: 告警类型
            service_name: 服务名称
        """
        if not self._enabled:
            return

        try:
            self._health_metrics['alert_count'].labels(
                alert_level=alert_level,
                alert_type=alert_type,
                service_name=service_name
            ).inc()

        except Exception as e:
            logger.error(f"记录告警失败: {e}")

    def update_custom_metric(self, metric_name: str, value: float,

                             labels: Optional[Dict[str, str]] = None):
        """
        更新自定义指标

        Args:
            metric_name: 指标名称
            value: 指标值
            labels: 标签
        """
        if not self._enabled or metric_name not in self._custom_metrics:
            return

        try:
            metric = self._custom_metrics[metric_name]
            labels = labels or {}

            if hasattr(metric, 'set'):
                # Gauge类型
                metric.labels(**labels).set(value)
            elif hasattr(metric, 'observe'):
                # Histogram / Summary类型
                metric.labels(**labels).observe(value)
            elif hasattr(metric, 'inc'):
                # Counter类型
                metric.labels(**labels).inc(value)

        except Exception as e:
            logger.error(f"更新自定义指标失败: {e}")

    def get_metrics(self) -> str:
        """
        获取Prometheus格式的指标

        Returns:
            Prometheus格式的指标字符串
        """
        if not self._enabled:
            return ""

        try:
            return generate_latest(self._registry)
        except Exception as e:
            logger.error(f"生成指标失败: {e}")
            return ""

    def get_metrics_content_type(self) -> str:
        """获取指标内容类型"""
        return CONTENT_TYPE_LATEST

    def _start_update_thread(self):
        """启动指标更新线程"""
        if self._update_thread and self._update_thread.is_alive():
            return

        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logger.info("指标更新线程已启动")

    def _update_loop(self):
        """指标更新循环"""
        while self._running:
            try:
                time.sleep(30)  # 每30秒更新一次
                self._update_system_metrics()
            except Exception as e:
                logger.error(f"指标更新失败: {e}")

    def _update_system_metrics(self):
        """更新系统指标"""
        try:
            # 这里可以添加系统级别的指标更新
            # 例如：CPU使用率、内存使用率等
            pass
        except Exception as e:
            logger.error(f"更新系统指标失败: {e}")

    def stop(self):
        """停止监控集成"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)

        logger.info("Prometheus监控集成已停止")

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.stop()

# 模块级健康检查函数


def check_health() -> Dict[str, Any]:
    """执行整体健康检查

    Returns:
        Dict[str, Any]: 健康检查结果
    """
    try:
        logger.info("开始Prometheus集成模块健康检查")

        health_checks = {
            "prometheus_client": check_prometheus_client(),
            "integration_class": check_integration_class(),
            "metric_system": check_metric_system()
        }

        # 综合健康状态
        overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

        result = {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "service": "prometheus_integration",
            "checks": health_checks
        }

        if not overall_healthy:
            logger.warning("Prometheus集成模块健康检查发现问题")
            result["issues"] = [
                name for name, check in health_checks.items()
                if not check.get("healthy", False)
            ]

        logger.info(f"Prometheus集成模块健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result

    except Exception as e:
        logger.error(f"Prometheus集成模块健康检查失败: {str(e)}", exc_info=True)
        return {
            "healthy": False,
            "timestamp": datetime.now().isoformat(),
            "service": "prometheus_integration",
            "error": str(e)
        }


def check_prometheus_client() -> Dict[str, Any]:
    """检查Prometheus客户端

    Returns: Dict[str, Any]rometheus客户端检查结果
    """
    try:
        # 检查Prometheus客户端是否可用
        prometheus_available = PROMETHEUS_AVAILABLE

        if not prometheus_available:
            return {
                "healthy": False,
                "prometheus_available": False,
                "error": "prometheus_client library not installed"
            }

        # 检查必需的类是否可以导入
        try:
            required_classes_available = all([
                Counter is not None,
                Gauge is not None,
                Histogram is not None,
                Summary is not None,
                CollectorRegistry is not None
            ])
        except ImportError as e:
            required_classes_available = False
            import_error = str(e)
        else:
            import_error = None

        return {
            "healthy": prometheus_available and required_classes_available,
            "prometheus_available": prometheus_available,
            "required_classes_available": required_classes_available,
            "import_error": import_error
        }
    except Exception as e:
        logger.error(f"Prometheus客户端检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_integration_class() -> Dict[str, Any]:
    """检查集成类定义

    Returns:
        Dict[str, Any]: 集成类检查结果
    """
    try:
        # 检查PrometheusIntegration类存在
        integration_class_exists = 'PrometheusIntegration' in globals()

        if not integration_class_exists:
            return {"healthy": False, "error": "PrometheusIntegration class not found"}

        # 检查必需的方法
        required_methods = ['__init__', 'register_custom_metric',
                            'record_health_check', 'get_metrics', 'stop']
        existing_methods = [method for method in dir(
            PrometheusIntegration) if not method.startswith('_')]

        methods_complete = all(method in existing_methods for method in required_methods)

        # 测试类实例化
        instantiation_works = False
        if PROMETHEUS_AVAILABLE:
            try:
                integration = PrometheusIntegration()
                instantiation_works = integration is not None
            except Exception:
                instantiation_works = False

        return {
            "healthy": integration_class_exists and methods_complete and (not PROMETHEUS_AVAILABLE or instantiation_works),
            "integration_class_exists": integration_class_exists,
            "methods_complete": methods_complete,
            "instantiation_works": instantiation_works if PROMETHEUS_AVAILABLE else "N/A (Prometheus not available)",
            "existing_methods": existing_methods
        }
    except Exception as e:
        logger.error(f"集成类检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_metric_system() -> Dict[str, Any]:
    """检查指标系统

    Returns:
        Dict[str, Any]: 指标系统检查结果
    """
    try:
        # 检查MetricType枚举
        metric_type_enum_exists = 'MetricType' in globals()

        if not metric_type_enum_exists:
            return {"healthy": False, "error": "MetricType enum not found"}

        # 检查枚举值
        expected_metric_types = ["counter", "gauge", "histogram", "summary"]
        actual_metric_types = [mt.value for mt in MetricType]

        metric_types_complete = set(actual_metric_types) == set(expected_metric_types)

        # 检查MetricDefinition数据类
        metric_definition_exists = 'MetricDefinition' in globals()

        return {
            "healthy": metric_type_enum_exists and metric_types_complete and metric_definition_exists,
            "metric_type_enum_exists": metric_type_enum_exists,
            "metric_types_complete": metric_types_complete,
            "metric_definition_exists": metric_definition_exists,
            "expected_metric_types": expected_metric_types,
            "actual_metric_types": actual_metric_types
        }
    except Exception as e:
        logger.error(f"指标系统检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def health_status() -> Dict[str, Any]:
    """获取健康状态摘要

    Returns:
        Dict[str, Any]: 健康状态摘要
    """
    try:
        health_check = check_health()

        return {
            "status": "healthy" if health_check["healthy"] else "unhealthy",
            "service": "prometheus_integration",
            "health_check": health_check,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康状态摘要失败: {str(e)}")
        return {"status": "error", "error": str(e)}


def health_summary() -> Dict[str, Any]:
    """获取健康摘要报告

    Returns:
        Dict[str, Any]: 健康摘要报告
    """
    try:
        health_check = check_health()

        # 统计指标类型信息
        metric_types_count = len(MetricType) if 'MetricType' in globals() else 0

        return {
            "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
            "prometheus_integration_module_info": {
                "service_name": "prometheus_integration",
                "purpose": "Prometheus监控集成",
                "operational": health_check["healthy"]
            },
            "prometheus_status": {
                "client_available": PROMETHEUS_AVAILABLE,
                "integration_class_working": health_check["checks"]["integration_class"]["healthy"],
                "metric_system_complete": health_check["checks"]["metric_system"]["healthy"]
            },
            "metrics_capabilities": {
                "metric_types_supported": metric_types_count,
                "custom_metrics_supported": True,
                "health_metrics_enabled": PROMETHEUS_AVAILABLE
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康摘要报告失败: {str(e)}")
        return {"overall_health": "error", "error": str(e)}


def monitor_prometheus_integration() -> Dict[str, Any]:
    """监控Prometheus集成状态

    Returns:
        Dict[str, Any]: 集成监控结果
    """
    try:
        health_check = check_health()

        # 计算集成效率指标
        integration_efficiency = 1.0 if health_check["healthy"] else 0.0

        return {
            "healthy": health_check["healthy"],
            "integration_metrics": {
                "service_name": "prometheus_integration",
                "integration_efficiency": integration_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            },
            "prometheus_metrics": {
                "client_available": PROMETHEUS_AVAILABLE,
                "metric_types_supported": len(MetricType) if 'MetricType' in globals() else 0,
                "integration_class_working": health_check["checks"]["integration_class"]["healthy"]
            }
        }
    except Exception as e:
        logger.error(f"Prometheus集成监控失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def validate_prometheus_config() -> Dict[str, Any]:
    """验证Prometheus配置

    Returns:
        Dict[str, Any]: 配置验证结果
    """
    try:
        validation_results = {
            "client_validation": _validate_prometheus_client(),
            "class_validation": _validate_integration_classes(),
            "enum_validation": _validate_metric_enums()
        }

        overall_valid = all(result.get("valid", False) for result in validation_results.values())

        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prometheus配置验证失败: {str(e)}")
        return {"valid": False, "error": str(e)}


def _validate_prometheus_client() -> Dict[str, Any]:
    """验证Prometheus客户端"""
    try:
        # 检查全局可用性标志
        client_available = PROMETHEUS_AVAILABLE

        # 尝试导入验证
        import_tests = {}
        if client_available:
            try:
                import_tests = {
                    "Counter": Counter is not None,
                    "Gauge": Gauge is not None,
                    "Histogram": Histogram is not None,
                    "Summary": Summary is not None,
                    "CollectorRegistry": CollectorRegistry is not None,
                    "generate_latest": callable(generate_latest)
                }
            except ImportError as e:
                import_tests = {"import_error": str(e)}

        all_imports_successful = client_available and all(
            import_tests.values()) if import_tests and not "import_error" in import_tests else False

        return {
            "valid": all_imports_successful,
            "client_available": client_available,
            "all_imports_successful": all_imports_successful,
            "import_tests": import_tests
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_integration_classes() -> Dict[str, Any]:
    """验证集成类"""
    try:
        # 检查必需的类
        required_classes = ['PrometheusIntegration', 'MetricDefinition', 'MetricType']
        classes_exist = all(cls in globals() for cls in required_classes)

        # 检查类是否可以实例化
        instantiation_tests = {}
        for cls_name in required_classes:
            if cls_name in globals():
                try:
                    cls = globals()[cls_name]
                    if cls_name == 'PrometheusIntegration':
                        # 只在Prometheus可用的情况下测试实例化
                        if PROMETHEUS_AVAILABLE:
                            instance = cls()
                            instantiation_tests[cls_name] = {"success": True}
                        else:
                            instantiation_tests[cls_name] = {
                                "success": "N/A (Prometheus not available)"}
                    else:
                        # 对于数据类和枚举，尝试创建实例
                        if cls_name == 'MetricDefinition':
                            instance = cls(name="test_metric", type=MetricType.COUNTER,
                                           description="Test metric")
                        elif cls_name == 'MetricType':
                            # 枚举不需要实例化
                            instantiation_tests[cls_name] = {"success": True}
                        instantiation_tests[cls_name] = {"success": True}
                except Exception as e:
                    instantiation_tests[cls_name] = {"success": False, "error": str(e)}
            else:
                instantiation_tests[cls_name] = {"success": False, "error": "Class not found"}

        all_instantiable = all(
            test["success"] == True or test["success"] == "N/A (Prometheus not available)"
            for test in instantiation_tests.values()
        )

        return {
            "valid": classes_exist and all_instantiable,
            "classes_exist": classes_exist,
            "all_instantiable": all_instantiable,
            "instantiation_tests": instantiation_tests,
            "required_classes": required_classes
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_metric_enums() -> Dict[str, Any]:
    """验证指标枚举"""
    try:
        # 检查MetricType枚举
        metric_type_exists = 'MetricType' in globals()

        if not metric_type_exists:
            return {"valid": False, "error": "MetricType not found"}

        # 检查枚举值
        expected_values = ["counter", "gauge", "histogram", "summary"]
        actual_values = [mt.value for mt in MetricType]

        values_match = set(actual_values) == set(expected_values)

        # 检查枚举方法
        has_members = hasattr(MetricType, '__members__')

        return {
            "valid": metric_type_exists and values_match and has_members,
            "metric_type_exists": metric_type_exists,
            "values_match": values_match,
            "has_members": has_members,
            "expected_values": expected_values,
            "actual_values": actual_values
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
