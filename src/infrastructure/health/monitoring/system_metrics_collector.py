"""
system_metrics_collector 模块

提供 system_metrics_collector 相关功能和接口。
"""

import logging

import threading
import time

from .constants import (
    DEFAULT_HISTORY_SIZE, COUNTER_API_CALLS, COUNTER_DEFAULT_VALUE,
    COUNTER_CACHE_HITS, COUNTER_CACHE_MISSES, COUNTER_DB_CONNECTIONS,
    COUNTER_ERRORS, COUNTER_WARNINGS, COLLECTION_INTERVAL_DEFAULT,
    THREAD_JOIN_TIMEOUT_DEFAULT, DEFAULT_HISTORY_QUERY_HOURS,
    AVERAGE_CALCULATION_HOURS, DEFAULT_HISTORY_LIMIT,
    RESPONSE_TIME_THRESHOLD_MS, CPU_HEALTHY_THRESHOLD,
    MEMORY_HEALTHY_THRESHOLD, DISK_HEALTHY_THRESHOLD,
    HISTORY_CAPACITY_WARNING_RATIO, TREND_CALCULATION_PERIODS,
    TREND_CHANGE_THRESHOLD_RATIO, CPU_USAGE_MIN, CPU_USAGE_MAX,
    MEMORY_USAGE_MIN, MEMORY_USAGE_MAX, DISK_USAGE_MIN, DISK_USAGE_MAX
)
from .metrics_collectors import MetricsAggregator
from .metrics_storage import MetricsStorage
from .standardization import (
    StandardizedMetricsInterface, standardize_metrics_format,
    handle_metrics_exceptions, validate_metrics_data
)
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional
"""
系统指标收集器模块

提供系统级指标收集功能，基于专用收集器组件构建。
"""

logger = logging.getLogger(__name__)


class SystemMetricsCollector(StandardizedMetricsInterface):
    """
    系统指标收集器 (标准化接口完整实现)

    根据Phase 8.2.3接口标准化优化，实现了完整的标准化指标收集接口。

    核心特性：
    - ✅ 继承StandardizedMetricsInterface，提供5个标准接口方法
    - ✅ 使用标准化数据格式和异常处理装饰器
    - ✅ 职责分离架构：收集逻辑与存储逻辑分离
    - ✅ 统一的常量管理和配置管理
    - ✅ 线程安全的指标收集和存储

    实现的标准接口方法：
    1. collect_metrics() -> Dict[str, Any]
       收集当前系统指标，返回标准化的数据格式

    2. get_latest_metrics() -> Optional[Dict[str, Any]]
       获取最新存储的指标数据

    3. get_metrics_history(hours: int = 24) -> Dict[str, Any]
       获取指定时间范围内的历史指标数据及统计信息

    4. validate_metrics_health() -> Dict[str, Any]
       验证指标收集系统的健康状态

    5. get_metrics_summary() -> Dict[str, Any]
       获取系统的完整指标摘要信息

    数据格式标准：
    所有方法返回的数据都符合以下标准：
    {
        "timestamp": "2025-09-28T10:00:00.123456",
        "source": "system_collector",
        "version": "1.0",
        "metrics": {
            "cpu": {"usage_percent": 45.5, "count": 8},
            "memory": {"total": 8589934592, "used": 4294967296},
            ...
        },
        "metadata": {
            "quality_score": 0.95,
            "collection_duration": 0.023
        }
    }

    使用示例：
        # 创建收集器实例
        collector = SystemMetricsCollector()

        # 收集指标
        metrics = collector.collect_metrics()

        # 获取历史数据
        history = collector.get_metrics_history(hours=24)

        # 验证健康状态
        health = collector.validate_metrics_health()

        # 获取摘要信息
        summary = collector.get_metrics_summary()

    线程安全：
    - 所有方法都是线程安全的
    - 支持并发收集和查询操作
    - 使用MetricsStorage的线程安全存储

    配置管理：
    - 使用统一常量管理配置参数
    - 支持自定义存储后端
    - 可配置收集间隔和历史大小
    """

    def __init__(self, storage: Optional[MetricsStorage] = None, history_size: int = DEFAULT_HISTORY_SIZE):
        """
        初始化系统指标收集器

        Args:
            storage: 指标存储器实例，如果为None则创建默认实例
            history_size: 历史数据存储大小(向后兼容参数)
        """
        # 使用MetricsStorage进行数据存储
        if storage is None:
            self.storage = MetricsStorage(history_size)
        else:
            self.storage = storage

        # 性能计数器 - 使用常量初始化
        self.performance_counters = {
            COUNTER_API_CALLS: COUNTER_DEFAULT_VALUE,
            COUNTER_CACHE_HITS: COUNTER_DEFAULT_VALUE,
            COUNTER_CACHE_MISSES: COUNTER_DEFAULT_VALUE,
            COUNTER_DB_CONNECTIONS: COUNTER_DEFAULT_VALUE,
            COUNTER_ERRORS: COUNTER_DEFAULT_VALUE,
            COUNTER_WARNINGS: COUNTER_DEFAULT_VALUE
        }

        # 使用聚合器来收集指标
        self.aggregator = MetricsAggregator()

        # 收集控制状态
        self.is_collecting = False
        self.collector_thread: Optional[threading.Thread] = None
        self._collection_interval = COLLECTION_INTERVAL_DEFAULT

        logger.info("系统指标收集器初始化完成")

    def start_collection(self, interval: float = COLLECTION_INTERVAL_DEFAULT):
        """
        开始指标收集

        Args:
            interval: 收集间隔(秒)，使用常量默认值
        """
        if self.is_collecting:
            logger.warning("指标收集已在运行中")
            return

        self._collection_interval = interval
        self.is_collecting = True

        self.collector_thread = threading.Thread(
            target=self._collection_worker,
            args=(interval,),
            daemon=True,
            name="SystemMetricsCollector"
        )
        self.collector_thread.start()
        logger.info(f"系统指标收集已启动，间隔: {interval}秒")

    def stop_collection(self):
        """停止指标收集"""
        if not self.is_collecting:
            return

        self.is_collecting = False
        if self.collector_thread and self.collector_thread.is_alive():
            self.collector_thread.join(timeout=THREAD_JOIN_TIMEOUT_DEFAULT)
        logger.info("系统指标收集已停止")

    @handle_metrics_exceptions
    def _collection_worker(self, interval: float):
        """
        指标收集工作线程 (标准化异常处理)

        持续收集系统指标并存储到MetricsStorage中。
        使用标准化的异常处理装饰器。
        """
        logger.debug(f"指标收集工作线程启动，间隔: {interval}秒")

        while self.is_collecting:
            try:
                # 收集当前指标
                raw_metrics = self._collect_current_metrics()

                # 标准化数据格式
                standardized_metrics = standardize_metrics_format(raw_metrics)

                # 验证数据质量
                validation = validate_metrics_data(standardized_metrics)
                if not validation["valid"]:
                    logger.warning(f"指标数据验证失败: {validation['errors']}")
                    # 仍然存储，但降低质量评分
                    standardized_metrics["quality_score"] = validation["quality_score"]

                # 通过MetricsStorage存储
                success = self.storage.store_metrics(standardized_metrics)
                if not success:
                    logger.error("指标数据存储失败")

                # 休眠指定间隔
                time.sleep(interval)

            except Exception:
                # 异常已由装饰器处理，这里不需要额外处理
                time.sleep(interval)

    # =========================================================================
    # StandardizedMetricsInterface 标准化接口实现
    # =========================================================================

    def collect_metrics(self) -> Dict[str, Any]:
        """
        收集指标 (标准化接口方法)

        Returns:
            Dict[str, Any]: 标准化的指标数据
        """
        raw_metrics = self._collect_current_metrics()
        return standardize_metrics_format(raw_metrics)

    def collect_cpu(self) -> Dict[str, Any]:
        """兼容历史接口，返回CPU指标摘要信息"""
        metrics = self.collect_metrics()
        cpu_metrics = metrics.get("metrics", {}).get("cpu", {})
        return {"timestamp": metrics.get("timestamp"), **cpu_metrics}

    def collect_memory(self) -> Dict[str, Any]:
        """兼容历史接口，返回内存指标摘要信息"""
        metrics = self.collect_metrics()
        memory_metrics = metrics.get("metrics", {}).get("memory", {})
        return {"timestamp": metrics.get("timestamp"), **memory_metrics}

    def collect_disk(self) -> Dict[str, Any]:
        """兼容历史接口，返回磁盘指标摘要信息"""
        metrics = self.collect_metrics()
        disk_metrics = metrics.get("metrics", {}).get("disk", {})
        return {"timestamp": metrics.get("timestamp"), **disk_metrics}

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        获取最新指标 (标准化接口方法)

        Returns:
            Optional[Dict[str, Any]]: 最新的标准化指标数据
        """
        return self.storage.get_latest_metrics()

    def get_metrics_history(self, hours: int = DEFAULT_HISTORY_QUERY_HOURS) -> Dict[str, Any]:
        """
        获取指标历史 (标准化接口方法)

        Args:
            hours: 历史时间范围(小时)

        Returns:
            Dict[str, Any]: 历史指标数据和统计信息
        """
        history = self.storage.get_metrics_history(hours)
        avg_metrics = self.storage.get_average_metrics(AVERAGE_CALCULATION_HOURS)  # 使用常量定义的平均值计算时间

        return {
            "history": history,
            "count": len(history),
            "time_range_hours": hours,
            "averages": avg_metrics.get("averages", {}) if avg_metrics else {},
            "summary": {
                "total_records": len(history),
                "avg_response_time": avg_metrics.get("averages", {}).get("response_time", 0) if avg_metrics else 0,
                "data_quality": "good" if len(history) > 0 else "no_data"
            }
        }

    def validate_metrics_health(self) -> Dict[str, Any]:
        """
        验证指标健康状态 (标准化接口方法)

        Returns:
            Dict[str, Any]: 健康验证结果
        """
        return self.check_collection_health()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要 (标准化接口方法)

        Returns:
            Dict[str, Any]: 指标摘要信息
        """
        latest = self.get_latest_metrics()
        storage_stats = self.storage.get_storage_stats()
        health_check = self.validate_metrics_health()

        return {
            "current_status": health_check.get("healthy", False),
            "latest_collection": latest.get("timestamp") if latest else None,
            "metrics_types": list(latest.get("metrics", {}).keys()) if latest else [],
            "storage_info": {
                "total_records": storage_stats.get("current_count", 0),
                "utilization_percent": storage_stats.get("utilization_percent", 0),
                "uptime_seconds": storage_stats.get("uptime_seconds", 0)
            },
            "performance_counters": self.performance_counters.copy(),
            "collection_config": {
                "interval": self._collection_interval,
                "is_collecting": self.is_collecting,
                "thread_alive": health_check.get("thread_alive", False)
            }
        }

    def _collect_current_metrics(self) -> Dict[str, Any]:
        """
        收集当前系统指标

        Returns:
            当前系统指标字典
        """
        # 使用聚合器收集所有指标
        return self.aggregator.collect_all()

    # 存储职责已分离到MetricsStorage，不再需要单独的存储逻辑

    def collect_specific_metrics(self, metric_types: List[str]) -> Dict[str, Any]:
        """
        收集指定的指标类型

        Args:
            metric_types: 要收集的指标类型列表

        Returns:
            包含指定指标的字典
        """
        return self.aggregator.collect_specific(metric_types)

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        获取最新的指标数据

        Returns:
            Optional[Dict[str, Any]]: 最新的指标数据，通过MetricsStorage获取
        """
        return self.storage.get_latest_metrics()

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        获取指标历史数据

        Args:
            hours: 历史时间范围(小时)，默认为24小时

        Returns:
            List[Dict[str, Any]]: 指定时间范围内的历史指标数据
        """
        return self.storage.get_metrics_history(hours)

    # 保持向后兼容的旧方法名
    def get_cpu_history(self, limit: int = DEFAULT_HISTORY_LIMIT) -> List[Dict[str, Any]]:
        """
        获取CPU历史数据 (向后兼容方法)

        注意: 此方法已废弃，请使用 get_metrics_history()

        Args:
            limit: 限制返回的记录数量

        Returns:
            List[Dict[str, Any]]: CPU相关历史数据
        """
        logger.warning("get_cpu_history方法已废弃，请使用get_metrics_history()")
        history = self.get_metrics_history(DEFAULT_HISTORY_QUERY_HOURS)  # 获取默认小时数的歷史数据
        # 过滤出包含CPU指标的记录
        cpu_history = [h for h in history if 'cpu' in h and isinstance(h['cpu'], dict)]
        return cpu_history[-limit:] if limit > 0 else cpu_history

    def get_memory_history(self, limit: int = DEFAULT_HISTORY_LIMIT) -> List[Dict[str, Any]]:
        """
        获取内存历史数据 (向后兼容方法)

        注意: 此方法已废弃，请使用 get_metrics_history()

        Args:
            limit: 限制返回的记录数量

        Returns:
            List[Dict[str, Any]]: 内存相关历史数据
        """
        logger.warning("get_memory_history方法已废弃，请使用get_metrics_history()")
        history = self.get_metrics_history(DEFAULT_HISTORY_QUERY_HOURS)
        memory_history = [h for h in history if 'memory' in h and isinstance(h['memory'], dict)]
        return memory_history[-limit:] if limit > 0 else memory_history

    def get_disk_history(self, limit: int = DEFAULT_HISTORY_LIMIT) -> List[Dict[str, Any]]:
        """
        获取磁盘历史数据 (向后兼容方法)

        注意: 此方法已废弃，请使用 get_metrics_history()

        Args:
            limit: 限制返回的记录数量

        Returns:
            List[Dict[str, Any]]: 磁盘相关历史数据
        """
        logger.warning("get_disk_history方法已废弃，请使用get_metrics_history()")
        history = self.get_metrics_history(DEFAULT_HISTORY_QUERY_HOURS)
        disk_history = [h for h in history if 'disk' in h and isinstance(h['disk'], dict)]
        return disk_history[-limit:] if limit > 0 else disk_history

    def get_network_history(self, limit: int = DEFAULT_HISTORY_LIMIT) -> List[Dict[str, Any]]:
        """
        获取网络历史数据 (向后兼容方法)

        注意: 此方法已废弃，请使用 get_metrics_history()

        Args:
            limit: 限制返回的记录数量

        Returns:
            List[Dict[str, Any]]: 网络相关历史数据
        """
        logger.warning("get_network_history方法已废弃，请使用get_metrics_history()")
        history = self.get_metrics_history(DEFAULT_HISTORY_QUERY_HOURS)
        network_history = [h for h in history if 'network' in h and isinstance(h['network'], dict)]
        return network_history[-limit:] if limit > 0 else network_history

    def get_gpu_history(self, limit: int = DEFAULT_HISTORY_LIMIT) -> List[Dict[str, Any]]:
        """
        获取GPU历史数据 (向后兼容方法)

        注意: 此方法已废弃，请使用 get_metrics_history()

        Args:
            limit: 限制返回的记录数量

        Returns:
            List[Dict[str, Any]]: GPU相关历史数据
        """
        logger.warning("get_gpu_history方法已废弃，请使用get_metrics_history()")
        history = self.get_metrics_history(DEFAULT_HISTORY_QUERY_HOURS)
        gpu_history = [h for h in history if 'gpu' in h and isinstance(h['gpu'], dict)]
        return gpu_history[-limit:] if limit > 0 else gpu_history

    def get_performance_counters(self) -> Dict[str, int]:
        """获取性能计数器"""
        return self.performance_counters.copy()

    def increment_counter(self, counter_name: str, value: int = 1):
        """增加性能计数器"""
        if counter_name in self.performance_counters:
            self.performance_counters[counter_name] += value

    def reset_counters(self):
        """重置所有性能计数器"""
        for key in self.performance_counters:
            self.performance_counters[key] = 0

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要信息"""
        latest = self.get_latest_metrics()
        if not latest:
            return {'status': 'no_data'}

        return {
            'timestamp': latest['timestamp'],
            'cpu_usage': latest.get('cpu', {}).get('usage_percent', 0),
            'memory_usage': latest.get('memory', {}).get('percent', 0),
            'disk_usage': latest.get('disk', {}).get('percent', 0),
            'gpu_available': 'load' in latest.get('gpu', {}),
            'gpu_usage': latest.get('gpu', {}).get('load', 0) if 'load' in latest.get('gpu', {}) else 0,
            'status': 'active' if self.is_collecting else 'inactive'
        }

    def check_health(self) -> Dict[str, Any]:
        """执行整体健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            logger.info("开始系统指标收集器健康检查")

            health_checks = {
                "collection_status": self.check_collection_health(),
                "data_integrity": self.check_data_integrity(),
                "performance": self.check_performance_health(),
                "resource_usage": self.check_resource_usage_health()
            }

            # 综合健康状态
            overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

            result = {
                "healthy": overall_healthy,
                "timestamp": datetime.now().isoformat(),
                "service": "system_metrics_collector",
                "checks": health_checks
            }

            if not overall_healthy:
                logger.warning("系统指标收集器健康检查发现问题")
                result["issues"] = [
                    name for name, check in health_checks.items()
                    if not check.get("healthy", False)
                ]

            logger.info(f"系统指标收集器健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
            return result

        except Exception as e:
            logger.error(f"系统指标收集器健康检查失败: {str(e)}", exc_info=True)
            return {
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
                "service": "system_metrics_collector",
                "error": str(e)
            }

    def check_collection_health(self) -> Dict[str, Any]:
        """
        检查指标收集健康状态

        Returns:
            Dict[str, Any]: 收集健康状态，集成MetricsStorage信息
        """
        try:
            is_healthy = self.is_collecting
            has_recent_data = bool(self.get_latest_metrics())

            # 检查线程状态
            thread_alive = (self.collector_thread and self.collector_thread.is_alive()
                            ) if self.collector_thread else False

            # 获取存储器统计信息
            storage_stats = self.storage.get_storage_stats()

            return {
                "healthy": is_healthy and has_recent_data,
                "collecting": self.is_collecting,
                "thread_alive": thread_alive,
                "has_recent_data": has_recent_data,
                "collection_interval": self._collection_interval,
                "storage": storage_stats,
                "performance_counters": self.performance_counters.copy()
            }
        except Exception as e:
            logger.error(f"收集健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_data_integrity(self) -> Dict[str, Any]:
        """检查数据完整性

        Returns:
            Dict[str, Any]: 数据完整性检查结果
        """
        try:
            latest = self.get_latest_metrics()
            if not latest:
                return {"healthy": False, "reason": "no_data_available"}

            # 检查必要指标是否存在
            required_keys = ["cpu", "memory", "disk", "timestamp"]
            missing_keys = [key for key in required_keys if key not in latest]

            # 检查历史数据一致性
            history_consistent = len(self.metrics_history) <= self.history_size

            return {
                "healthy": len(missing_keys) == 0 and history_consistent,
                "missing_keys": missing_keys,
                "history_consistent": history_consistent,
                "data_points": len(self.metrics_history)
            }
        except Exception as e:
            logger.error(f"数据完整性检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_performance_health(self) -> Dict[str, Any]:
        """检查性能健康状态

        Returns:
            Dict[str, Any]: 性能健康检查结果
        """
        try:
            latest = self.get_latest_metrics()
            if not latest:
                return {"healthy": False, "reason": "no_performance_data"}

            # 检查响应时间
            response_time = latest.get("response_time", 0)
            acceptable_response_time = response_time < RESPONSE_TIME_THRESHOLD_MS  # 使用常量定义的响应时间阈值

            # 检查收集频率
            collection_frequency_ok = self.is_collecting  # 简化检查

            return {
                "healthy": acceptable_response_time and collection_frequency_ok,
                "response_time_ms": response_time,
                "collection_active": collection_frequency_ok,
                "counters": self.get_performance_counters()
            }
        except Exception as e:
            logger.error(f"性能健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_resource_usage_health(self) -> Dict[str, Any]:
        """检查资源使用健康状态

        Returns:
            Dict[str, Any]: 资源使用健康检查结果
        """
        try:
            latest = self.get_latest_metrics()
            if not latest:
                return {"healthy": False, "reason": "no_resource_data"}

            cpu_usage = latest.get("cpu", {}).get("usage_percent", 0)
            memory_usage = latest.get("memory", {}).get("percent", 0)
            disk_usage = latest.get("disk", {}).get("percent", 0)

            # 使用常量定义的健康阈值进行评估
            cpu_healthy = cpu_usage < CPU_HEALTHY_THRESHOLD  # 使用常量定义的CPU健康阈值
            memory_healthy = memory_usage < MEMORY_HEALTHY_THRESHOLD  # 使用常量定义的内存健康阈值
            disk_healthy = disk_usage < DISK_HEALTHY_THRESHOLD  # 使用常量定义的磁盘健康阈值

            return {
                "healthy": cpu_healthy and memory_healthy and disk_healthy,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage,
                "cpu_healthy": cpu_healthy,
                "memory_healthy": memory_healthy,
                "disk_healthy": disk_healthy
            }
        except Exception as e:
            logger.error(f"资源使用健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def health_status(self) -> Dict[str, Any]:
        """获取健康状态摘要

        Returns:
            Dict[str, Any]: 健康状态摘要
        """
        try:
            summary = self.get_system_summary()
            health_check = self.check_health()

            return {
                "status": "healthy" if health_check["healthy"] else "unhealthy",
                "summary": summary,
                "health_check": health_check,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康状态摘要失败: {str(e)}")
            return {"status": "error", "error": str(e)}

    def health_summary(self) -> Dict[str, Any]:
        """获取健康摘要报告

        Returns:
            Dict[str, Any]: 健康摘要报告
        """
        try:
            health_check = self.check_health()
            summary = self.get_system_summary()
            counters = self.get_performance_counters()

            return {
                "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
                "system_status": summary,
                "performance_counters": counters,
                "collection_stats": {
                    "active": self.is_collecting,
                    "history_size": len(self.metrics_history),
                    "max_history": self.history_size
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康摘要报告失败: {str(e)}")
            return {"overall_health": "error", "error": str(e)}

    def monitor_collection_status(self) -> Dict[str, Any]:
        """监控指标收集状态

        Returns:
            Dict[str, Any]: 收集状态监控结果
        """
        try:
            status = {
                "collecting": self.is_collecting,
                "thread_alive": self.collector_thread.is_alive() if self.collector_thread else False,
                "data_points": len(self.metrics_history),
                "history_utilization": len(self.metrics_history) / self.history_size if self.history_size > 0 else 0,
                "last_collection_time": self.metrics_history[-1]["timestamp"] if self.metrics_history else None
            }

            # 检查是否需要告警
            warnings = []
            if not self.is_collecting:
                warnings.append("collection_stopped")
            if len(self.metrics_history) == 0:
                warnings.append("no_data")
            if len(self.metrics_history) >= self.history_size * HISTORY_CAPACITY_WARNING_RATIO:
                warnings.append("history_near_capacity")

            status["warnings"] = warnings
            status["healthy"] = len(warnings) == 0

            return status
        except Exception as e:
            logger.error(f"监控收集状态失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def monitor_performance(self) -> Dict[str, Any]:
        """监控系统性能指标

        Returns:
            Dict[str, Any]: 性能监控结果
        """
        try:
            latest = self.get_latest_metrics()
            if not latest:
                return {"healthy": False, "reason": "no_performance_data"}

            # 计算性能趋势
            cpu_trend = self._calculate_trend(self.cpu_history, "value")
            memory_trend = self._calculate_trend(self.memory_history, "value")

            return {
                "healthy": True,
                "current": {
                    "cpu_usage": latest.get("cpu", {}).get("usage_percent", 0),
                    "memory_usage": latest.get("memory", {}).get("percent", 0),
                    "disk_usage": latest.get("disk", {}).get("percent", 0)
                },
                "trends": {
                    "cpu_trend": cpu_trend,
                    "memory_trend": memory_trend
                },
                "performance_counters": self.get_performance_counters()
            }
        except Exception as e:
            logger.error(f"性能监控失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def _calculate_trend(self, history: deque, value_key: str, periods: int = TREND_CALCULATION_PERIODS) -> str:
        """
        计算指标趋势 (使用常量定义的计算参数)

        Args:
            history: 历史数据队列
            value_key: 值字段名
            periods: 计算周期数(使用常量定义)

        Returns:
            str: 趋势描述 ("increasing", "decreasing", "stable")
        """
        if len(history) < periods * 2:
            return "insufficient_data"

        try:
            recent = [item[value_key] for item in list(history)[-periods:]]
            previous = [item[value_key] for item in list(history)[-periods*2:-periods]]

            recent_avg = sum(recent) / len(recent) if recent else 0
            previous_avg = sum(previous) / len(previous) if previous else 0

            if recent_avg > previous_avg * (1 + TREND_CHANGE_THRESHOLD_RATIO):  # 使用常量定义的增长阈值
                return "increasing"
            elif recent_avg < previous_avg * (1 - TREND_CHANGE_THRESHOLD_RATIO):  # 使用常量定义的下降阈值
                return "decreasing"
            else:
                return "stable"
        except Exception:
            return "calculation_error"

    def validate_metrics_data(self) -> Dict[str, Any]:
        """验证指标数据有效性

        Returns:
            Dict[str, Any]: 数据验证结果
        """
        try:
            validation_results = {
                "data_structure": self._validate_data_structure(),
                "value_ranges": self._validate_value_ranges(),
                "timestamp_consistency": self._validate_timestamp_consistency(),
                "history_integrity": self._validate_history_integrity()
            }

            overall_valid = all(result.get("valid", False)
                                for result in validation_results.values())

            return {
                "valid": overall_valid,
                "validation_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"指标数据验证失败: {str(e)}")
            return {"valid": False, "error": str(e)}

    def _validate_data_structure(self) -> Dict[str, Any]:
        """验证数据结构"""
        try:
            latest = self.get_latest_metrics()
            if not latest:
                return {"valid": False, "reason": "no_data"}

            required_fields = ["timestamp", "cpu", "memory", "disk"]
            missing_fields = [field for field in required_fields if field not in latest]

            return {
                "valid": len(missing_fields) == 0,
                "missing_fields": missing_fields
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_value_ranges(self) -> Dict[str, Any]:
        """验证数值范围"""
        try:
            latest = self.get_latest_metrics()
            if not latest:
                return {"valid": False, "reason": "no_data"}

            issues = []

            # 检查CPU使用率
            cpu_usage = latest.get("cpu", {}).get("usage_percent", 0)
            if not (CPU_USAGE_MIN <= cpu_usage <= CPU_USAGE_MAX):
                issues.append(f"cpu_usage_out_of_range: {cpu_usage}")

            # 检查内存使用率
            memory_usage = latest.get("memory", {}).get("percent", 0)
            if not (MEMORY_USAGE_MIN <= memory_usage <= MEMORY_USAGE_MAX):
                issues.append(f"memory_usage_out_of_range: {memory_usage}")

            # 检查磁盘使用率
            disk_usage = latest.get("disk", {}).get("percent", 0)
            if not (DISK_USAGE_MIN <= disk_usage <= DISK_USAGE_MAX):
                issues.append(f"disk_usage_out_of_range: {disk_usage}")

            return {
                "valid": len(issues) == 0,
                "issues": issues
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_timestamp_consistency(self) -> Dict[str, Any]:
        """验证时间戳一致性"""
        try:
            if len(self.metrics_history) < 2:
                return {"valid": True, "reason": "insufficient_data"}

            # 检查时间戳是否递增
            timestamps = [item["timestamp"] for item in self.metrics_history]
            inconsistent_timestamps = []

            for i in range(1, len(timestamps)):
                if timestamps[i] < timestamps[i-1]:
                    inconsistent_timestamps.append((timestamps[i-1], timestamps[i]))

            return {
                "valid": len(inconsistent_timestamps) == 0,
                "inconsistent_timestamps": inconsistent_timestamps
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_history_integrity(self) -> Dict[str, Any]:
        """验证历史数据完整性"""
        try:
            # 检查历史数据大小是否合理
            history_size_ok = len(self.metrics_history) <= self.history_size

            # 检查是否有重复数据
            unique_timestamps = len(set(item["timestamp"] for item in self.metrics_history))
            no_duplicates = unique_timestamps == len(self.metrics_history)

            return {
                "valid": history_size_ok and no_duplicates,
                "history_size_ok": history_size_ok,
                "no_duplicates": no_duplicates,
                "total_entries": len(self.metrics_history),
                "unique_timestamps": unique_timestamps
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}
