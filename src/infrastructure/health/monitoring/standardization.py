"""
standardization 模块

提供 standardization 相关功能和接口。
"""

import logging

# 移除循环导入 - StandardizedMetricsInterface和standardize_metrics_format在同一文件中定义
from .constants import (
    SOURCE_SYSTEM, STATUS_UNKNOWN, STATUS_HEALTHY, STATUS_CRITICAL, STATUS_WARNING
)
from datetime import datetime
from typing import Dict, Any, Optional
"""
指标收集体系标准化工具

根据Phase 8.2.3接口标准化优化，提供统一的数据格式、接口命名和异常处理。

标准化体系包括：
- StandardizedMetricsInterface: 统一的指标收集接口
- 数据格式标准化函数: standardize_metrics_format()
- 数据验证函数: validate_metrics_data()
- 异常处理装饰器: handle_metrics_exceptions()

数据格式标准：
{
    "timestamp": "ISO格式时间戳",
    "source": "数据源标识",
    "version": "1.0",
    "metrics": {
        "cpu": {...},
        "memory": {...},
        "disk": {...},
        "network": {...}
    },
    "metadata": {
        "quality_score": 0.0-1.0,
        "collection_duration": float
    }
}

使用示例：
    class MyCollector(StandardizedMetricsInterface):
        def collect_metrics(self):
            raw_data = {...}  # 原始数据
            return standardize_metrics_format(raw_data)
"""

logger = logging.getLogger(__name__)


def standardize_metrics_format(raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化指标数据格式

    将各种来源的指标数据转换为统一的格式标准。

    Args:
        raw_metrics: 原始指标数据字典

    Returns:
        Dict[str, Any]: 标准化的指标数据
    """
    try:
        # 确保基本字段存在
        standardized = {
            "timestamp": raw_metrics.get("timestamp", datetime.now().isoformat()),
            "source": raw_metrics.get("source", SOURCE_SYSTEM),
            "version": "1.0",  # 数据格式版本
            "metadata": raw_metrics.get("metadata", {}),
            "metrics": {}
        }

        # 标准化各个指标类型
        raw_data = raw_metrics.get("metrics", raw_metrics)

        # CPU指标标准化
        if "cpu" in raw_data:
            standardized["metrics"]["cpu"] = _standardize_cpu_metrics(raw_data["cpu"])

        # 内存指标标准化
        if "memory" in raw_data:
            standardized["metrics"]["memory"] = _standardize_memory_metrics(raw_data["memory"])

        # 磁盘指标标准化
        if "disk" in raw_data:
            standardized["metrics"]["disk"] = _standardize_disk_metrics(raw_data["disk"])

        # 网络指标标准化
        if "network" in raw_data:
            standardized["metrics"]["network"] = _standardize_network_metrics(raw_data["network"])

        # GPU指标标准化
        if "gpu" in raw_data:
            standardized["metrics"]["gpu"] = _standardize_gpu_metrics(raw_data["gpu"])

        # 系统指标标准化
        if "system" in raw_data:
            standardized["metrics"]["system"] = _standardize_system_metrics(raw_data["system"])

        return standardized

    except Exception as e:
        logger.error(f"指标数据格式标准化失败: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "source": SOURCE_SYSTEM,
            "version": "1.0",
            "error": str(e),
            "original_data": raw_metrics
        }


def _standardize_cpu_metrics(cpu_data: Dict[str, Any]) -> Dict[str, Any]:
    """标准化CPU指标"""
    return {
        "usage_percent": float(cpu_data.get("usage_percent", 0.0)),
        "count": int(cpu_data.get("count", cpu_data.get("cpu_count", 0))),
        "count_logical": int(cpu_data.get("count_logical", cpu_data.get("cpu_count_logical", 0))),
        "freq_current": cpu_data.get("freq_current"),
        "freq_min": cpu_data.get("freq_min"),
        "freq_max": cpu_data.get("freq_max"),
        "load_average": cpu_data.get("load_average"),
        "status": _determine_cpu_status(cpu_data)
    }


def _standardize_memory_metrics(memory_data: Dict[str, Any]) -> Dict[str, Any]:
    """标准化内存指标"""
    return {
        "total": int(memory_data.get("total", 0)),
        "available": int(memory_data.get("available", 0)),
        "used": int(memory_data.get("used", 0)),
        "free": int(memory_data.get("free", 0)),
        "percent": float(memory_data.get("percent", 0.0)),
        "buffers": int(memory_data.get("buffers", 0)),
        "cached": int(memory_data.get("cached", 0)),
        "status": _determine_memory_status(memory_data)
    }


def _standardize_disk_metrics(disk_data: Dict[str, Any]) -> Dict[str, Any]:
    """标准化磁盘指标"""
    return {
        "total": int(disk_data.get("total", 0)),
        "used": int(disk_data.get("used", 0)),
        "free": int(disk_data.get("free", 0)),
        "percent": float(disk_data.get("percent", 0.0)),
        "mount_point": disk_data.get("mount_point", "/"),
        "fstype": disk_data.get("fstype"),
        "status": _determine_disk_status(disk_data)
    }


def _standardize_network_metrics(network_data: Dict[str, Any]) -> Dict[str, Any]:
    """标准化网络指标"""
    return {
        "bytes_sent": int(network_data.get("bytes_sent", 0)),
        "bytes_recv": int(network_data.get("bytes_recv", 0)),
        "packets_sent": int(network_data.get("packets_sent", 0)),
        "packets_recv": int(network_data.get("packets_recv", 0)),
        "errin": int(network_data.get("errin", 0)),
        "errout": int(network_data.get("errout", 0)),
        "dropin": int(network_data.get("dropin", 0)),
        "dropout": int(network_data.get("dropout", 0)),
        "status": _determine_network_status(network_data)
    }


def _standardize_gpu_metrics(gpu_data: Dict[str, Any]) -> Dict[str, Any]:
    """标准化GPU指标"""
    if isinstance(gpu_data, list):
        # 多GPU情况
        return {
            "count": len(gpu_data),
            "gpus": [_standardize_single_gpu(gpu) for gpu in gpu_data],
            "status": _determine_gpu_status(gpu_data)
        }
    else:
        # 单GPU情况
        return _standardize_single_gpu(gpu_data)


def _standardize_single_gpu(gpu_data: Dict[str, Any]) -> Dict[str, Any]:
    """标准化单个GPU指标"""
    return {
        "id": gpu_data.get("id", 0),
        "name": gpu_data.get("name", "Unknown"),
        "memory_total": int(gpu_data.get("memory_total", 0)),
        "memory_used": int(gpu_data.get("memory_used", 0)),
        "memory_free": int(gpu_data.get("memory_free", 0)),
        "memory_percent": float(gpu_data.get("memory_percent", 0.0)),
        "temperature": gpu_data.get("temperature"),
        "utilization": gpu_data.get("utilization"),
        "status": gpu_data.get("status", STATUS_UNKNOWN)
    }


def _standardize_system_metrics(system_data: Dict[str, Any]) -> Dict[str, Any]:
    """标准化系统指标"""
    return {
        "uptime": system_data.get("uptime"),
        "boot_time": system_data.get("boot_time"),
        "process_count": int(system_data.get("process_count", 0)),
        "thread_count": int(system_data.get("thread_count", 0)),
        "load_average": system_data.get("load_average"),
        "hostname": system_data.get("hostname"),
        "platform": system_data.get("platform"),
        "status": system_data.get("status", STATUS_HEALTHY)
    }


def _determine_cpu_status(cpu_data: Dict[str, Any]) -> str:
    """根据CPU数据确定状态"""
    usage = cpu_data.get("usage_percent", 0.0)
    if usage >= 95.0:
        return STATUS_CRITICAL
    elif usage >= 80.0:
        return STATUS_WARNING
    else:
        return STATUS_HEALTHY


def _determine_memory_status(memory_data: Dict[str, Any]) -> str:
    """根据内存数据确定状态"""
    percent = memory_data.get("percent", 0.0)
    if percent >= 95.0:
        return STATUS_CRITICAL
    elif percent >= 85.0:
        return STATUS_WARNING
    else:
        return STATUS_HEALTHY


def _determine_disk_status(disk_data: Dict[str, Any]) -> str:
    """根据磁盘数据确定状态"""
    percent = disk_data.get("percent", 0.0)
    if percent >= 95.0:
        return STATUS_CRITICAL
    elif percent >= 85.0:
        return STATUS_WARNING
    else:
        return STATUS_HEALTHY


def _determine_network_status(network_data: Dict[str, Any]) -> str:
    """根据网络数据确定状态"""
    errin = network_data.get("errin", 0)
    errout = network_data.get("errout", 0)
    total_errors = errin + errout

    if total_errors > 1000:  # 错误包太多
        return STATUS_CRITICAL
    elif total_errors > 100:
        return STATUS_WARNING
    else:
        return STATUS_HEALTHY


def _determine_gpu_status(gpu_data: Any) -> str:
    """根据GPU数据确定状态"""
    if isinstance(gpu_data, list):
        # 多GPU：如果任一GPU有问题则警告
        for gpu in gpu_data:
            if gpu.get("status") == STATUS_CRITICAL:
                return STATUS_CRITICAL
            elif gpu.get("status") == STATUS_WARNING:
                return STATUS_WARNING
        return STATUS_HEALTHY
    else:
        # 单GPU
        return gpu_data.get("status", STATUS_UNKNOWN)


class StandardizedMetricsInterface:
    """
    标准化指标收集接口

    为所有指标收集组件提供统一的接口方法命名和行为标准。
    任何实现此接口的类都必须提供以下标准方法：

    核心方法：
    - collect_metrics(): 收集指标数据
    - get_latest_metrics(): 获取最新指标
    - get_metrics_history(): 获取历史指标
    - validate_metrics_health(): 验证指标健康状态
    - get_metrics_summary(): 获取指标摘要

    数据格式标准：
    所有方法返回的数据都应符合标准化的JSON格式，包括：
    - timestamp: ISO格式时间戳
    - source: 数据源标识
    - version: 格式版本
    - metrics: 具体的指标数据
    - metadata: 元数据信息

    实现此接口的类示例：
    - SystemMetricsCollector: 系统指标收集器
    - PerformanceMonitor: 性能监控器
    - DatabaseMetricsCollector: 数据库指标收集器

    使用方式：
        collector = SomeCollector()
        assert isinstance(collector, StandardizedMetricsInterface)

        # 收集指标
        metrics = collector.collect_metrics()

        # 获取历史数据
        history = collector.get_metrics_history(hours=24)

        # 验证健康状态
        health = collector.validate_metrics_health()
    """

    def collect_metrics(self) -> Dict[str, Any]:
        """
        收集指标 (标准接口方法)

        Returns:
            Dict[str, Any]: 标准化的指标数据
        """
        raise NotImplementedError

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        获取最新指标 (标准接口方法)

        Returns:
            Optional[Dict[str, Any]]: 最新的指标数据
        """
        raise NotImplementedError

    def get_metrics_history(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取指标历史 (标准接口方法)

        Args:
            hours: 历史时间范围(小时)

        Returns:
            Dict[str, Any]: 历史指标数据
        """
        raise NotImplementedError

    def validate_metrics_health(self) -> Dict[str, Any]:
        """
        验证指标健康状态 (标准接口方法)

        Returns:
            Dict[str, Any]: 健康验证结果
        """
        raise NotImplementedError

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要 (标准接口方法)

        Returns:
            Dict[str, Any]: 指标摘要信息
        """
        raise NotImplementedError


def handle_metrics_exceptions(func):
    """
    指标收集异常处理装饰器

    统一处理指标收集过程中的异常，提供标准化的错误处理和日志记录。

    处理的异常类型：
    - PermissionError: 权限不足
    - ConnectionError: 连接失败
    - TimeoutError: 操作超时
    - Exception: 其他所有异常

    返回的错误格式：
    {
        "error": "错误类型描述",
        "function": "出错的函数名",
        "timestamp": "ISO格式时间戳"
    }

    使用示例：
        @handle_metrics_exceptions
        def collect_cpu_metrics(self):
            # 收集逻辑
            return metrics_data

        # 如果发生异常，会自动返回标准化的错误格式
        # 而不是抛出异常
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PermissionError:
            logger.warning(f"权限不足: {func.__name__}")
            return {
                "error": "permission_denied",
                "function": func.__name__,
                "timestamp": datetime.now().isoformat()
            }
        except ConnectionError:
            logger.warning(f"连接失败: {func.__name__}")
            return {
                "error": "connection_failed",
                "function": func.__name__,
                "timestamp": datetime.now().isoformat()
            }
        except TimeoutError:
            logger.warning(f"操作超时: {func.__name__}")
            return {
                "error": "timeout",
                "function": func.__name__,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"指标收集异常 {func.__name__}: {e}")
            return {
                "error": str(e),
                "function": func.__name__,
                "timestamp": datetime.now().isoformat()
            }
    return wrapper


def validate_metrics_data(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证指标数据完整性和质量

    对指标数据进行全面验证，包括：
    - 必要字段检查（timestamp, source, metrics）
    - 时间戳格式验证
    - 数据类型验证
    - 数值范围检查
    - 质量评分计算

    Args:
        metrics: 要验证的指标数据，必须是标准化的格式

    Returns:
        Dict[str, Any]: 验证结果，包含：
        - valid: bool，数据是否有效
        - errors: List[str]，错误列表
        - warnings: List[str]，警告列表
        - quality_score: float，质量评分(0.0-1.0)

    示例：
        metrics = {
            "timestamp": "2025-09-28T10:00:00",
            "source": "system",
            "metrics": {"cpu": {"usage_percent": 85.5}}
        }

        result = validate_metrics_data(metrics)
        # result = {
        #     "valid": True,
        #     "errors": [],
        #     "warnings": [],
        #     "quality_score": 1.0
        # }
    """
    errors = []
    warnings = []

    # 检查必要字段
    required_fields = ["timestamp", "source", "metrics"]
    for field in required_fields:
        if field not in metrics:
            errors.append(f"缺少必要字段: {field}")

    # 检查时间戳格式
    if "timestamp" in metrics:
        try:
            datetime.fromisoformat(metrics["timestamp"].replace('Z', '+00:00'))
        except (ValueError, TypeError):
            errors.append("时间戳格式无效")

    # 检查指标数据
    if "metrics" in metrics and isinstance(metrics["metrics"], dict):
        metrics_data = metrics["metrics"]

        # 检查各指标类型的合理性
        for metric_type, metric_data in metrics_data.items():
            if metric_type == "cpu":
                _validate_cpu_metrics(metric_data, errors, warnings)
            elif metric_type == "memory":
                _validate_memory_metrics(metric_data, errors, warnings)
            elif metric_type == "disk":
                _validate_disk_metrics(metric_data, errors, warnings)
            elif metric_type == "network":
                _validate_network_metrics(metric_data, errors, warnings)
    else:
        errors.append("metrics字段缺失或格式错误")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "quality_score": max(0.0, 1.0 - (len(errors) * 0.2 + len(warnings) * 0.1))
    }


def _validate_cpu_metrics(cpu_data: Dict[str, Any], errors: list, warnings: list):
    """验证CPU指标"""
    if not isinstance(cpu_data, dict):
        errors.append("CPU指标必须是字典类型")
        return

    # 检查必要字段
    if "usage_percent" not in cpu_data:
        errors.append("CPU指标缺少usage_percent字段")
    elif not isinstance(cpu_data["usage_percent"], (int, float)):
        errors.append("CPU usage_percent必须是数值类型")
    elif not (0 <= cpu_data["usage_percent"] <= 100):
        warnings.append("CPU usage_percent超出合理范围(0-100)")


def _validate_memory_metrics(memory_data: Dict[str, Any], errors: list, warnings: list):
    """验证内存指标"""
    if not isinstance(memory_data, dict):
        errors.append("内存指标必须是字典类型")
        return

    # 检查数值字段
    numeric_fields = ["total", "used", "free", "percent"]
    for field in numeric_fields:
        if field in memory_data:
            if not isinstance(memory_data[field], (int, float)):
                errors.append(f"内存指标{field}必须是数值类型")
            elif field == "percent" and not (0 <= memory_data[field] <= 100):
                warnings.append("内存百分比超出合理范围(0-100)")


def _validate_disk_metrics(disk_data: Dict[str, Any], errors: list, warnings: list):
    """验证磁盘指标"""
    if not isinstance(disk_data, dict):
        errors.append("磁盘指标必须是字典类型")
        return

    # 检查数值字段
    numeric_fields = ["total", "used", "free", "percent"]
    for field in numeric_fields:
        if field in disk_data:
            if not isinstance(disk_data[field], (int, float)):
                errors.append(f"磁盘指标{field}必须是数值类型")
            elif field == "percent" and not (0 <= disk_data[field] <= 100):
                warnings.append("磁盘百分比超出合理范围(0-100)")


def _validate_network_metrics(network_data: Dict[str, Any], errors: list, warnings: list):
    """验证网络指标"""
    if not isinstance(network_data, dict):
        errors.append("网络指标必须是字典类型")
        return

    # 检查计数器字段
    counter_fields = ["bytes_sent", "bytes_recv", "packets_sent", "packets_recv"]
    for field in counter_fields:
        if field in network_data and not isinstance(network_data[field], (int, float)):
            errors.append(f"网络指标{field}必须是数值类型")
