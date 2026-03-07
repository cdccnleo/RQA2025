"""
metrics_collectors 模块

提供 metrics_collectors 相关功能和接口。
"""

import logging

import GPUtil
import platform
import psutil
import datetime
# 导入常量
from .constants import (
    GPU_PERCENTAGE_MULTIPLIER, QUALITY_SCORE_PENALTY_PER_ERROR, SECONDS_TO_HOURS
)
# 导入标准化工具
import psutil

from ..core.interfaces import IUnifiedInfrastructureInterface
# from .constants import (
#     # Constants will be imported here
# )
from .standardization import standardize_metrics_format, handle_metrics_exceptions
from datetime import datetime
from typing import Dict, Any, Optional
"""
系统指标收集器组件

将SystemMetricsCollector拆分为更小的专用收集器组件。
"""

logger = logging.getLogger(__name__)


class CPUCollector:
    """CPU指标收集器 (标准化接口)"""

    @staticmethod
    @handle_metrics_exceptions
    def collect() -> Dict[str, Any]:
        """
        收集CPU指标 (标准化数据格式)

        Returns:
            Dict[str, Any]: 标准化的CPU指标数据
        """
        try:
            raw_data = {
                'usage_percent': psutil.cpu_percent(interval=0.1),
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True),
                'freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'freq_min': psutil.cpu_freq().min if psutil.cpu_freq() else None,
                'freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None
            }

            # 返回标准化格式
            return standardize_metrics_format({
                "timestamp": datetime.now().isoformat(),
                "source": "system",
                "metrics": {"cpu": raw_data}
            })

        except Exception as e:
            # 异常已由装饰器处理，返回错误格式
            return {'error': str(e)}


class MemoryCollector:
    """内存指标收集器"""

    @staticmethod
    def collect() -> Dict[str, Any]:
        """收集内存指标"""
        try:
            mem = psutil.virtual_memory()
            return {
                'total': mem.total,
                'available': mem.available,
                'percent': mem.percent,
                'used': mem.used,
                'free': mem.free
            }
        except Exception as e:
            return {'error': str(e)}


class DiskCollector:
    """磁盘指标收集器"""

    @staticmethod
    def collect() -> Dict[str, Any]:
        """收集磁盘指标"""
        try:
            disk = psutil.disk_usage('/')
            return {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }
        except Exception as e:
            return {'error': str(e)}


class NetworkCollector:
    """网络指标收集器"""

    @staticmethod
    def collect() -> Dict[str, Any]:
        """收集网络指标"""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout
            }
        except Exception as e:
            return {'error': str(e)}


class GPUCollector:
    """GPU指标收集器"""

    @staticmethod
    def collect() -> Dict[str, Any]:
        """收集GPU指标"""
        try:
            # 动态导入以支持可选依赖
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 取第一个GPU
                return {
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * GPU_PERCENTAGE_MULTIPLIER,  # GPU负载转换为百分比
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'memory_util': gpu.memoryUtil * GPU_PERCENTAGE_MULTIPLIER,  # GPU内存利用率转换为百分比
                    'temperature': gpu.temperature
                }
            else:
                return {'available': False}
        except Exception as e:
            return {'error': str(e)}


class SystemInfoCollector:
    """系统信息收集器"""

    @staticmethod
    def collect() -> Dict[str, Any]:
        """收集系统信息"""
        try:
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'python_version': platform.python_version(),
                'boot_time': psutil.boot_time()
            }
        except Exception as e:
            return {'error': str(e)}


class MetricsAggregator(IUnifiedInfrastructureInterface):
    """指标聚合器 (标准化数据格式)"""

    def __init__(self):
        self.collectors = {
            'cpu': CPUCollector,
            'memory': MemoryCollector,
            'disk': DiskCollector,
            'network': NetworkCollector,
            'gpu': GPUCollector,
            'system': SystemInfoCollector
        }
        self._initialized = False
        self._collection_count = 0
        self._last_collection_time = None

    @handle_metrics_exceptions
    def collect_all(self) -> Dict[str, Any]:
        """
        收集所有系统指标 (标准化格式聚合)

        聚合所有专用收集器的标准化输出，返回统一的指标集合。

        Returns:
            Dict[str, Any]: 标准化的完整指标数据
        """
        timestamp = datetime.now().isoformat()
        aggregated_metrics = {}
        collection_errors = []

        # 收集各个指标类型
        for name, collector_class in self.collectors.items():
            try:
                collector_result = collector_class.collect()

                # 检查是否为错误结果
                if isinstance(collector_result, dict) and 'error' in collector_result:
                    collection_errors.append(f"{name}: {collector_result['error']}")
                    continue

                # 提取指标数据
                if 'metrics' in collector_result and isinstance(collector_result['metrics'], dict):
                    # 合并到聚合指标中
                    for metric_type, metric_data in collector_result['metrics'].items():
                        if metric_type not in aggregated_metrics:
                            aggregated_metrics[metric_type] = metric_data
                        else:
                            # 如果已有相同类型，合并数据
                            if isinstance(aggregated_metrics[metric_type], dict) and isinstance(metric_data, dict):
                                aggregated_metrics[metric_type].update(metric_data)

            except Exception as e:
                logger.error(f"收集{name}指标失败: {e}")
                collection_errors.append(f"{name}: {str(e)}")

        # 构建标准化输出
        result = {
            "timestamp": timestamp,
            "source": "system_aggregator",
            "version": "1.0",
            "metrics": aggregated_metrics,
            "metadata": {
                "collection_time": timestamp,
                "collectors_used": list(self.collectors.keys()),
                # 根据错误数量降低质量评分
                "quality_score": 1.0 - (len(collection_errors) * QUALITY_SCORE_PENALTY_PER_ERROR)
            }
        }

        # 如果有错误，添加到元数据中
        if collection_errors:
            result["metadata"]["errors"] = collection_errors
            result["metadata"]["error_count"] = len(collection_errors)

        self._collection_count += 1
        self._last_collection_time = timestamp

        return result

    def collect_specific(self, metric_types: list) -> Dict[str, Any]:
        """
        收集指定的指标类型

        Args:
            metric_types: 要收集的指标类型列表

        Returns:
            包含指定指标的字典
        """
        timestamp = datetime.datetime.now().isoformat()

        metrics = {'timestamp': timestamp}

        for metric_type in metric_types:
            if metric_type in self.collectors:
                try:
                    collector_class = self.collectors[metric_type]
                    metrics[metric_type] = collector_class.collect()
                except Exception as e:
                    logger.error(f"收集{metric_type}指标失败: {e}")
                    metrics[metric_type] = {'error': str(e)}
            else:
                logger.warning(f"未知的指标类型: {metric_type}")

        return metrics

    # IUnifiedInfrastructureInterface 实现
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化指标聚合器

        Args:
            config: 配置参数

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("开始初始化MetricsAggregator")

            if config:
                # 可以在这里处理配置参数，如设置收集间隔等
                pass

            self._initialized = True
            logger.info("MetricsAggregator初始化成功")
            return True

        except Exception as e:
            logger.error(f"MetricsAggregator初始化失败: {str(e)}", exc_info=True)
            self._initialized = False
            return False

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """
        try:
            logger.debug("获取MetricsAggregator组件信息")

            return {
                "component_type": "MetricsAggregator",
                "initialized": self._initialized,
                "available_collectors": list(self.collectors.keys()),
                "collector_count": len(self.collectors),
                "collection_count": self._collection_count,
                "last_collection_time": self._last_collection_time.isoformat() if self._last_collection_time else None
            }
        except Exception as e:
            logger.error(f"获取MetricsAggregator组件信息失败: {str(e)}")
            return {"error": str(e)}

    def is_healthy(self) -> bool:
        """检查组件健康状态

        Returns:
            bool: 组件是否健康
        """
        try:
            logger.debug("检查MetricsAggregator组件健康状态")

            # 检查基本状态
            if not self._initialized:
                logger.warning("MetricsAggregator未初始化")
                return False

            # 检查是否有收集器
            if not self.collectors:
                logger.warning("MetricsAggregator没有配置收集器")
                return False

            # 检查是否能够执行收集操作
            try:
                # 执行一次收集来验证功能（使用指定的指标类型以避免过多开销）
                result = self.collect_specific(['cpu'])
                return 'cpu' in result and 'error' not in result.get('cpu', {})
            except Exception as e:
                logger.error(f"MetricsAggregator健康检查失败: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"MetricsAggregator健康检查异常: {str(e)}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标

        Returns:
            Dict[str, Any]: 组件指标数据
        """
        try:
            logger.debug("获取MetricsAggregator组件指标")

            return {
                "component_metrics": {
                    "initialized": self._initialized,
                    "collector_count": len(self.collectors),
                    "collection_count": self._collection_count,
                    "last_collection_time": self._last_collection_time.isoformat() if self._last_collection_time else None,
                    "uptime_seconds": (datetime.now() - self._last_collection_time).total_seconds() if self._last_collection_time else 0
                },
                "collector_metrics": {
                    "available_collectors": list(self.collectors.keys()),
                    "total_collectors": len(self.collectors)
                },
                "performance_metrics": {
                    "collections_per_hour": self._collection_count / max(1, (datetime.now() - (self._last_collection_time or datetime.now())).total_seconds() / SECONDS_TO_HOURS)
                }
            }
        except Exception as e:
            logger.error(f"获取MetricsAggregator指标失败: {str(e)}")
            return {"error": str(e)}

    def cleanup(self) -> bool:
        """清理组件资源

        Returns:
            bool: 清理是否成功
        """
        try:
            logger.info("开始清理MetricsAggregator资源")

            # 清理收集器引用
            self.collectors.clear()

            # 重置计数器
            self._collection_count = 0
            self._last_collection_time = None

            # 保持初始化状态，但清理运行时数据
            logger.info("MetricsAggregator资源清理完成")
            return True

        except Exception as e:
            logger.error(f"MetricsAggregator资源清理失败: {str(e)}", exc_info=True)
            return False

# 模块级健康检查函数


def check_health() -> Dict[str, Any]:
    """执行整体健康检查"""
    try:
        logger.info("开始指标收集器模块健康检查")

        health_checks = {
            "collector_classes": check_collector_classes(),
            "metrics_system": check_metrics_system(),
            "dependencies": check_dependencies()
        }

        overall_healthy = all(check.get("healthy", False) for check in health_checks.values())
        result = {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "service": "metrics_collectors",
            "checks": health_checks
        }

        if not overall_healthy:
            logger.warning("指标收集器模块健康检查发现问题")
            result["issues"] = [name for name, check in health_checks.items()
                                if not check.get("healthy", False)]

        logger.info(f"指标收集器模块健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result
    except Exception as e:
        logger.error(f"指标收集器模块健康检查失败: {str(e)}", exc_info=True)
        return {
            "healthy": False,
            "timestamp": datetime.now().isoformat(),
            "service": "metrics_collectors",
            "error": str(e)
        }


def check_collector_classes() -> Dict[str, Any]:
    """检查收集器类"""
    try:
        # 检查主要的收集器类
        collector_classes = ['CPUCollector', 'MemoryCollector', 'DiskCollector',
                             'NetworkCollector', 'GPUCollector', 'SystemInfoCollector', 'MetricsAggregator']
        classes_exist = all(cls in globals() for cls in collector_classes)

        return {
            "healthy": classes_exist,
            "classes_exist": classes_exist,
            "collector_classes": collector_classes,
            "missing_classes": [cls for cls in collector_classes if cls not in globals()]
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def check_metrics_system() -> Dict[str, Any]:
    """检查指标系统"""
    try:
        # 检查聚合器类
        aggregator_exists = 'MetricsAggregator' in globals()

        if not aggregator_exists:
            return {"healthy": False, "error": "MetricsAggregator class not found"}

        # 检查必需的方法
        required_methods = ['initialize', 'get_component_info',
                            'is_healthy', 'get_metrics', 'cleanup']
        methods_exist = all(hasattr(MetricsAggregator, method) for method in required_methods)

        return {
            "healthy": aggregator_exists and methods_exist,
            "aggregator_exists": aggregator_exists,
            "methods_exist": methods_exist,
            "required_methods": required_methods
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def check_dependencies() -> Dict[str, Any]:
    """检查依赖"""
    try:
        # 检查psutil依赖
        psutil_available = True
        try:
            import psutil
        except ImportError:
            psutil_available = False

        return {
            "healthy": psutil_available,
            "psutil_available": psutil_available
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def health_status() -> Dict[str, Any]:
    """获取健康状态摘要"""
    try:
        health_check = check_health()
        return {
            "status": "healthy" if health_check["healthy"] else "unhealthy",
            "service": "metrics_collectors",
            "health_check": health_check,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def health_summary() -> Dict[str, Any]:
    """获取健康摘要报告"""
    try:
        health_check = check_health()
        return {
            "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
            "metrics_collectors_module_info": {
                "service_name": "metrics_collectors",
                "purpose": "系统指标收集器",
                "operational": health_check["healthy"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"overall_health": "error", "error": str(e)}


def monitor_metrics_collectors() -> Dict[str, Any]:
    """监控指标收集器状态"""
    try:
        health_check = check_health()
        collector_efficiency = 1.0 if health_check["healthy"] else 0.0
        return {
            "healthy": health_check["healthy"],
            "collector_metrics": {
                "service_name": "metrics_collectors",
                "collector_efficiency": collector_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            }
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def validate_metrics_collectors() -> Dict[str, Any]:
    """验证指标收集器"""
    try:
        validation_results = {
            "collector_validation": check_collector_classes(),
            "metrics_validation": check_metrics_system(),
            "dependency_validation": check_dependencies()
        }
        overall_valid = all(result.get("valid", False) for result in validation_results.values())
        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
