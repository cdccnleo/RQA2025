#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据层性能优化管理器

整合所有性能优化措施，提供统一的性能优化管理：
- 缓存优化策略
- 并发处理优化
- 内存管理优化
- 异步处理优化
- 性能监控和自动调优
"""

import gc
import psutil
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

try:
    from ..infrastructure_integration_manager import (
        get_data_integration_manager,
        log_data_operation, record_data_metric, publish_data_event
    )
except ImportError:
    # 降级处理
    def get_data_integration_manager():
        return Mock()
    
    def log_data_operation(*args, **kwargs):
        pass
    
    def record_data_metric(*args, **kwargs):
        pass
    
    def publish_data_event(*args, **kwargs):
        pass
    
    from unittest.mock import Mock

try:
    from ..interfaces.standard_interfaces import DataSourceType
except ImportError:
    # 降级处理
    from enum import Enum
    class DataSourceType(Enum):
        STOCK = "stock"
        FOREX = "forex"
        CRYPTO = "crypto"

# 兼容缺失枚举成员的环境
TYPE_STOCK = getattr(DataSourceType, "STOCK", type("Compat", (), {"value": "stock"})())


@dataclass
class PerformanceMetrics:

    """性能指标"""
    response_time: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceConfig:

    """性能配置"""
    enable_memory_monitoring: bool = True
    enable_gc_optimization: bool = True
    enable_connection_pooling: bool = True
    enable_object_pooling: bool = True
    memory_threshold: float = 0.8  # 内存使用率阈值
    gc_threshold: int = 1000  # GC阈值
    max_connections: int = 100
    connection_timeout: int = 30
    enable_performance_monitoring: bool = True
    monitoring_interval: int = 60  # 监控间隔（秒）


class DataPerformanceOptimizer:

    """
    数据层性能优化管理器

    提供全面的性能优化功能：
    - 内存管理优化
    - 连接池管理
    - 对象池化
    - GC优化
    - 性能监控
    - 自动调优
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        """
        初始化性能优化器

        Args:
            config: 性能配置
        """
        # 使用基础设施集成管理器获取配置
        self.config_obj = config or PerformanceConfig()
        merged_config = self._load_config_from_integration_manager()
        self.config = PerformanceConfig(**merged_config)

        # 初始化基础设施集成管理器
        self.integration_manager = get_data_integration_manager()
        if not self.integration_manager._initialized:
            self.integration_manager.initialize()

        # 性能监控
        self.performance_history: Dict[DataSourceType, List[PerformanceMetrics]] = {}
        self._performance_lock = threading.Lock()

        # 内存管理
        self.memory_monitor_thread = None
        self._stop_memory_monitor = False

        # 连接池
        self.connection_pools = {}

        # 对象池
        self.object_pools = {}

        # 性能统计
        self.stats = {
            'optimizations_applied': 0,
            'memory_cleanups': 0,
            'gc_cycles': 0,
            'connection_recycles': 0,
            'objects_recycled': 0
        }

        # 启动性能监控
        self._start_performance_monitoring()

        # 注册健康检查
        self._register_health_checks()

        log_data_operation("performance_optimizer_init", TYPE_STOCK,
                           {"config": self.config.__dict__}, "info")

    def _load_config_from_integration_manager(self) -> Dict[str, Any]:
        """从基础设施集成管理器加载配置"""
        try:
            merged_config = self.config_obj.__dict__.copy()

            # 从基础设施集成管理器获取配置
            if hasattr(self.integration_manager, '_integration_config'):
                infra_config = self.integration_manager._integration_config
                merged_config.update({
                    'enable_memory_monitoring': infra_config.get('enable_memory_monitoring', self.config_obj.enable_memory_monitoring),
                    'enable_gc_optimization': infra_config.get('enable_gc_optimization', self.config_obj.enable_gc_optimization),
                    'memory_threshold': infra_config.get('memory_threshold', self.config_obj.memory_threshold),
                    'max_connections': infra_config.get('max_connections', self.config_obj.max_connections)
                })

            return merged_config

        except Exception as e:
            # 如果集成管理器不可用，使用默认配置
            return self.config_obj.__dict__.copy()

    def _register_health_checks(self) -> None:
        """注册健康检查"""
        try:
            health_bridge = self.integration_manager.get_health_check_bridge()
            if health_bridge:
                # 注册性能优化器健康检查
                    health_bridge.register_data_health_check(
                    "performance_optimizer",
                    self._performance_optimizer_health_check,
                    TYPE_STOCK
                )

        except Exception as e:
            log_data_operation("performance_optimizer_health_check_registration_error", TYPE_STOCK,
                               {"error": str(e)}, "warning")

    def _performance_optimizer_health_check(self) -> Dict[str, Any]:
        """性能优化器健康检查"""
        try:
            health_status = {
                'component': 'DataPerformanceOptimizer',
                'status': 'healthy',
                'memory_monitoring_active': self.memory_monitor_thread is not None and self.memory_monitor_thread.is_alive(),
                'performance_history_records': sum(len(history) for history in self.performance_history.values()),
                'connection_pools_count': len(self.connection_pools),
                'object_pools_count': len(self.object_pools),
                'optimizations_applied': self.stats['optimizations_applied'],
                'timestamp': datetime.now().isoformat()
            }

            # 检查关键指标
            process = psutil.Process()
            memory_percent = process.memory_percent()

            if memory_percent > self.config.memory_threshold * 100:
                health_status['status'] = 'warning'
                health_status['message'] = f'内存使用率过高: {memory_percent:.1f}%'

            return health_status

        except Exception as e:
            return {
                'component': 'DataPerformanceOptimizer',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _start_performance_monitoring(self) -> None:
        """启动性能监控"""
        if not self.config.enable_performance_monitoring:
            return

        def performance_monitor_worker():

            while not self._stop_memory_monitor:
                try:
                    self._collect_performance_metrics()
                    self._apply_performance_optimizations()

                    # 等待下一个监控周期
                    for _ in range(self.config.monitoring_interval):
                        if self._stop_memory_monitor:
                            break
                        threading.Event().wait(1)

                except Exception as e:
                    log_data_operation("performance_monitor_error", TYPE_STOCK,
                                       {"error": str(e)}, "error")

        self.memory_monitor_thread = threading.Thread(
            target=performance_monitor_worker,
            daemon=True,
            name="PerformanceOptimizer"
        )
        self.memory_monitor_thread.start()

        log_data_operation("performance_monitoring_started", TYPE_STOCK, {}, "info")

    def _collect_performance_metrics(self) -> None:
        """收集性能指标"""
        try:
            with self._performance_lock:
                process = psutil.Process()

                # 收集系统级性能指标
                memory_percent = process.memory_percent()
                cpu_percent = process.cpu_percent()

                # 为每个数据类型创建性能指标
                for data_type in DataSourceType:
                    if data_type not in self.performance_history:
                        self.performance_history[data_type] = []

                    metrics = PerformanceMetrics(
                        memory_usage=memory_percent,
                        cpu_usage=cpu_percent,
                        timestamp=datetime.now()
                    )

                    self.performance_history[data_type].append(metrics)

                    # 保持历史记录数量
                    if len(self.performance_history[data_type]) > 100:
                        self.performance_history[data_type] = self.performance_history[data_type][-100:]

                    # 记录到基础设施监控
                    record_data_metric("memory_usage", memory_percent, data_type)
                    record_data_metric("cpu_usage", cpu_percent, data_type)

        except Exception as e:
            log_data_operation("performance_metrics_collection_error", TYPE_STOCK,
                               {"error": str(e)}, "error")

    def _apply_performance_optimizations(self) -> None:
        """应用性能优化措施"""
        try:
            process = psutil.Process()
            memory_percent = process.memory_percent()

            # 内存优化
            if memory_percent > self.config.memory_threshold * 100:
                self._optimize_memory_usage()
                self.stats['optimizations_applied'] += 1

            # GC优化
            if self.config.enable_gc_optimization:
                self._optimize_gc()
                self.stats['gc_cycles'] += 1

            # 连接池优化
            if self.config.enable_connection_pooling:
                self._optimize_connection_pools()

            # 对象池优化
            if self.config.enable_object_pooling:
                self._optimize_object_pools()

        except Exception as e:
            log_data_operation("performance_optimization_error", TYPE_STOCK,
                               {"error": str(e)}, "error")

    def _optimize_memory_usage(self) -> None:
        """优化内存使用"""
        try:
            # 手动触发垃圾回收
            collected = gc.collect()
            self.stats['memory_cleanups'] += 1

            # 记录内存清理结果
            log_data_operation("memory_cleanup", TYPE_STOCK,
                               {"objects_collected": collected}, "info")

            # 发布内存优化事件
            publish_data_event("memory_optimized", {
                "objects_collected": collected,
                "memory_before": psutil.Process().memory_percent()
            }, TYPE_STOCK, "normal")

        except Exception as e:
            log_data_operation("memory_optimization_error", TYPE_STOCK,
                               {"error": str(e)}, "error")

    def _optimize_gc(self) -> None:
        """优化垃圾回收"""
        try:
            # 获取GC统计信息
            gc_stats = {}
            for gen in range(3):
                gc_stats[f'gen_{gen}_count'] = gc.get_count()[gen]
                gc_stats[f'gen_{gen}_threshold'] = gc.get_threshold()[gen]

            # 根据配置调整GC阈值
            if self.config.gc_threshold > 0:
                current_threshold = gc.get_threshold()
                new_threshold = tuple(min(threshold, self.config.gc_threshold)
                                      for threshold in current_threshold)
                gc.set_threshold(*new_threshold)

                log_data_operation("gc_optimized", TYPE_STOCK,
                                   {"old_threshold": current_threshold, "new_threshold": new_threshold}, "info")

        except Exception as e:
            log_data_operation("gc_optimization_error", TYPE_STOCK,
                               {"error": str(e)}, "error")

    def _optimize_connection_pools(self) -> None:
        """优化连接池"""
        try:
            # 检查连接池状态
            for pool_name, pool in self.connection_pools.items():
                if hasattr(pool, 'get_stats'):
                    stats = pool.get_stats()

                    # 如果连接数过多，清理空闲连接
                    if stats.get('active_connections', 0) > self.config.max_connections:
                        if hasattr(pool, 'cleanup_idle'):
                            cleaned = pool.cleanup_idle()
                            self.stats['connection_recycles'] += cleaned

                            log_data_operation("connection_pool_optimized", TYPE_STOCK,
                                               {"pool": pool_name, "cleaned_connections": cleaned}, "info")

        except Exception as e:
            log_data_operation("connection_pool_optimization_error", TYPE_STOCK,
                               {"error": str(e)}, "error")

    def _optimize_object_pools(self) -> None:
        """优化对象池"""
        try:
            # 检查对象池状态
            for pool_name, pool in self.object_pools.items():
                if hasattr(pool, 'get_stats'):
                    stats = pool.get_stats()

                    # 如果对象数过多，清理过期对象
                    if stats.get('active_objects', 0) > 1000:  # 阈值
                        if hasattr(pool, 'cleanup_expired'):
                            cleaned = pool.cleanup_expired()
                            self.stats['objects_recycled'] += cleaned

                            log_data_operation("object_pool_optimized", TYPE_STOCK,
                                               {"pool": pool_name, "cleaned_objects": cleaned}, "info")

        except Exception as e:
            log_data_operation("object_pool_optimization_error", TYPE_STOCK,
                               {"error": str(e)}, "error")

    def register_connection_pool(self, name: str, pool: Any) -> None:
        """
        注册连接池

        Args:
            name: 连接池名称
            pool: 连接池对象
        """
        try:
            self.connection_pools[name] = pool
            log_data_operation("connection_pool_registered", TYPE_STOCK,
                               {"pool_name": name}, "info")

        except Exception as e:
            log_data_operation("connection_pool_registration_error", TYPE_STOCK,
                               {"pool_name": name, "error": str(e)}, "error")

    def register_object_pool(self, name: str, pool: Any) -> None:
        """
        注册对象池

        Args:
            name: 对象池名称
            pool: 对象池对象
        """
        try:
            self.object_pools[name] = pool
            log_data_operation("object_pool_registered", TYPE_STOCK,
                               {"pool_name": name}, "info")

        except Exception as e:
            log_data_operation("object_pool_registration_error", TYPE_STOCK,
                               {"pool_name": name, "error": str(e)}, "error")

    def get_performance_report(self, data_type: Optional[DataSourceType] = None) -> Dict[str, Any]:
        """
        获取性能报告

        Args:
            data_type: 数据类型，如果为None则返回所有类型

        Returns:
            性能报告
        """
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'config': self.config.__dict__,
                'stats': self.stats.copy(),
                'current_memory_usage': psutil.Process().memory_percent(),
                'current_cpu_usage': psutil.cpu_percent(),
                'data_types': {}
            }

            target_types = [data_type] if data_type else list(self.performance_history.keys())

            for dt in target_types:
                if dt in self.performance_history and self.performance_history[dt]:
                    history = self.performance_history[dt]

                    # 计算平均值
                    avg_memory = sum(h.memory_usage for h in history) / len(history)
                    avg_cpu = sum(h.cpu_usage for h in history) / len(history)
                    avg_response_time = sum(h.response_time for h in history) / \
                        len(history) if any(h.response_time > 0 for h in history) else 0

                    report['data_types'][dt.value] = {
                        'avg_memory_usage': avg_memory,
                        'avg_cpu_usage': avg_cpu,
                        'avg_response_time': avg_response_time,
                        'history_records': len(history),
                        'latest_metrics': history[-1].__dict__ if history else None
                    }

            return report

        except Exception as e:
            log_data_operation("performance_report_error", TYPE_STOCK,
                               {"error": str(e)}, "error")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }

    def apply_manual_optimization(self, optimization_type: str, **kwargs) -> bool:
        """
        手动应用优化措施

        Args:
            optimization_type: 优化类型
            **kwargs: 优化参数

        Returns:
            是否应用成功
        """
        try:
            if optimization_type == 'memory_cleanup':
                collected = gc.collect()
                self.stats['memory_cleanups'] += 1
                log_data_operation("manual_memory_cleanup", TYPE_STOCK,
                                   {"objects_collected": collected}, "info")
                return True

            elif optimization_type == 'gc_optimization':
                self._optimize_gc()
                log_data_operation("manual_gc_optimization", TYPE_STOCK, {}, "info")
                return True

            elif optimization_type == 'connection_pool_cleanup':
                total_cleaned = 0
                for pool in self.connection_pools.values():
                    if hasattr(pool, 'cleanup_idle'):
                        total_cleaned += pool.cleanup_idle()

                self.stats['connection_recycles'] += total_cleaned
                log_data_operation("manual_connection_cleanup", TYPE_STOCK,
                                   {"cleaned_connections": total_cleaned}, "info")
                return True

            else:
                log_data_operation("unknown_optimization_type", TYPE_STOCK,
                                   {"optimization_type": optimization_type}, "warning")
                return False

        except Exception as e:
            log_data_operation("manual_optimization_error", TYPE_STOCK,
                               {"optimization_type": optimization_type, "error": str(e)}, "error")
            return False

    def shutdown(self) -> None:
        """关闭性能优化器"""
        try:
            # 停止监控线程
            self._stop_memory_monitor = True

            if self.memory_monitor_thread and self.memory_monitor_thread.is_alive():
                self.memory_monitor_thread.join(timeout=5.0)

            # 清理资源
            self.connection_pools.clear()
            self.object_pools.clear()

            log_data_operation("performance_optimizer_shutdown", TYPE_STOCK,
                               {"final_stats": self.stats}, "info")

        except Exception as e:
            log_data_operation("performance_optimizer_shutdown_error", TYPE_STOCK,
                               {"error": str(e)}, "error")


# 全局单例实例
_performance_optimizer = None


def get_performance_optimizer() -> DataPerformanceOptimizer:
    """
    获取性能优化器单例实例

    Returns:
        性能优化器实例
    """
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = DataPerformanceOptimizer()
    return _performance_optimizer


__all__ = [
    'PerformanceMetrics',
    'PerformanceConfig',
    'DataPerformanceOptimizer',
    'get_performance_optimizer'
]
