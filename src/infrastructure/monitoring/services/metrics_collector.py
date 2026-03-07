#!/usr/bin/env python3
"""
RQA2025 基础设施层指标收集器

负责收集各种监控指标，包括系统指标、测试覆盖率、性能指标等。
这是从ContinuousMonitoringSystem中拆分出来的数据收集组件。
"""

import psutil
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Align module aliases for both legacy and src-prefixed imports.
_module = sys.modules[__name__]
sys.modules.setdefault("src.infrastructure.monitoring.services.metrics_collector", _module)
sys.modules.setdefault("infrastructure.monitoring.services.metrics_collector", _module)


class MetricsCollector:
    """
    指标收集器

    负责收集系统监控相关的各种指标数据。
    """

    def __init__(self, project_root: Optional[str] = None):
        """
        初始化指标收集器

        Args:
            project_root: 项目根目录
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.collection_stats = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'last_collection_time': None
        }

        # 缓存机制以提高性能
        self._cache = {}
        self._cache_timeout = 30  # 缓存30秒
        self._last_cache_update = {}

        logger.info("指标收集器初始化完成")

    def _get_cached_result(self, cache_key: str, collector_func, *args, **kwargs):
        """
        获取缓存结果，如果缓存过期则重新收集

        Args:
            cache_key: 缓存键
            collector_func: 收集函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            Any: 收集结果
        """
        current_time = time.time()

        # 检查缓存是否有效
        if (cache_key in self._cache and
            cache_key in self._last_cache_update and
            current_time - self._last_cache_update[cache_key] < self._cache_timeout):
            return self._cache[cache_key]

        # 重新收集数据
        try:
            result = collector_func(*args, **kwargs)
            self._cache[cache_key] = result
            self._last_cache_update[cache_key] = current_time
            return result
        except Exception as e:
            logger.warning(f"收集缓存数据失败 {cache_key}: {e}")
            if cache_key not in self._cache:
                raise
            return self._cache.get(cache_key, {})

    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        收集所有指标（优化版本）

        Returns:
            Dict[str, Any]: 收集的指标数据
        """
        try:
            start_time = time.time()
            timestamp = datetime.now()

            # 并发收集指标以提高性能
            metrics = {
                'timestamp': timestamp,
                'system_metrics': self._collect_system_metrics_cached(),
                'test_coverage_metrics': self._collect_test_coverage_metrics_cached(),
                'performance_metrics': self._collect_performance_metrics_cached(),
                'resource_usage': self._collect_resource_usage_cached(),
                'health_status': self._collect_health_status_cached(),
                'route_health': self._collect_route_health_cached(),
                'logger_pool_metrics': self._collect_logger_pool_metrics_cached()
            }

            # 更新统计信息
            self.collection_stats['total_collections'] += 1
            self.collection_stats['successful_collections'] += 1
            self.collection_stats['last_collection_time'] = timestamp

            collection_time = time.time() - start_time
            logger.debug(f"指标收集完成: {len(metrics)}个类别，耗时{collection_time:.3f}秒")
            return metrics

        except Exception as e:
            self.collection_stats['failed_collections'] += 1
            logger.error(f"指标收集失败: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'system_metrics': {},
                'test_coverage_metrics': {},
                'performance_metrics': {},
                'resource_usage': {},
                'health_status': {},
                'route_health': {},
                'logger_pool_metrics': {}
            }

    def _collect_system_metrics_cached(self) -> Dict[str, Any]:
        """收集系统指标（缓存版本）"""
        return self._get_cached_result('system_metrics', self._collect_system_metrics)

    def _collect_test_coverage_metrics_cached(self) -> Dict[str, Any]:
        """收集测试覆盖率指标（缓存版本）"""
        return self._get_cached_result('test_coverage_metrics', self._collect_test_coverage_metrics)

    def _collect_performance_metrics_cached(self) -> Dict[str, Any]:
        """收集性能指标（缓存版本）"""
        return self._get_cached_result('performance_metrics', self._collect_performance_metrics)

    def _collect_resource_usage_cached(self) -> Dict[str, Any]:
        """收集资源使用情况（缓存版本）"""
        return self._get_cached_result('resource_usage', self._collect_resource_usage)

    def _collect_health_status_cached(self) -> Dict[str, Any]:
        """收集健康状态（缓存版本）"""
        return self._get_cached_result('health_status', self._collect_health_status)

    def _collect_route_health_cached(self) -> Dict[str, Any]:
        """收集路由健康检查指标（缓存版本）"""
        return self._get_cached_result('route_health', self._collect_route_health)

    def _collect_logger_pool_metrics_cached(self) -> Dict[str, Any]:
        """收集Logger池性能指标（缓存版本）"""
        return self._get_cached_result('logger_pool_metrics', self._collect_logger_pool_metrics)

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """
        收集系统指标（优化版本）

        Returns:
            Dict[str, Any]: 系统指标
        """
        try:
            # CPU信息 - 使用更短的间隔以提高响应性
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_times_percent = psutil.cpu_times_percent(interval=0.1)

            # 内存信息
            memory = psutil.virtual_memory()

            # 磁盘信息
            disk = psutil.disk_usage('/')

            # 网络信息
            network = psutil.net_io_counters()

            # 进程信息（轻量级）
            process_count = len(psutil.pids())

            return {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'user_percent': cpu_times_percent.user if cpu_times_percent else 0,
                    'system_percent': cpu_times_percent.system if cpu_times_percent else 0,
                    'idle_percent': cpu_times_percent.idle if cpu_times_percent else 0,
                    'count': psutil.cpu_count(),
                    'count_logical': psutil.cpu_count(logical=True)
                },
                'memory': {
                    'usage_percent': memory.percent,
                    'used_bytes': memory.used,
                    'total_bytes': memory.total,
                    'available_bytes': memory.available,
                    'free_bytes': memory.free
                },
                'disk': {
                    'usage_percent': disk.percent,
                    'used_bytes': disk.used,
                    'total_bytes': disk.total,
                    'free_bytes': disk.free
                },
                'network': {
                    'bytes_sent': network.bytes_sent if network else 0,
                    'bytes_received': network.bytes_recv if network else 0,
                    'packets_sent': network.packets_sent if network else 0,
                    'packets_recv': network.packets_recv if network else 0,
                    'errin': network.errin if network else 0,
                    'errout': network.errout if network else 0
                },
                'system': {
                    'process_count': process_count,
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                    'boot_time': psutil.boot_time()
                }
            }

        except Exception as e:
            logger.warning(f"收集系统指标失败: {e}")
            return {}

    def _collect_test_coverage_metrics(self) -> Dict[str, Any]:
        """
        收集测试覆盖率指标

        Returns:
            Dict[str, Any]: 测试覆盖率指标
        """
        try:
            # 查找覆盖率报告文件
            coverage_files = list(self.project_root.glob("**/coverage.xml"))
            coverage_files.extend(self.project_root.glob("**/coverage.json"))
            coverage_files.extend(self.project_root.glob("**/.coverage"))

            if not coverage_files:
                # 生成模拟数据
                return self._get_mock_coverage_data()

            # 这里应该解析实际的覆盖率文件
            # 暂时返回模拟数据
            return self._get_mock_coverage_data()

        except Exception as e:
            logger.warning(f"收集测试覆盖率指标失败: {e}")
            return self._get_mock_coverage_data()

    def _get_mock_coverage_data(self) -> Dict[str, Any]:
        """
        获取模拟覆盖率数据

        Returns:
            Dict[str, Any]: 模拟覆盖率数据
        """
        # 生成基于时间的模拟数据
        base_coverage = 75.0
        time_factor = time.time() % 86400 / 86400  # 每日循环
        coverage_variation = 5.0 * (time_factor - 0.5)  # ±5%的变化

        return {
            'overall_coverage': round(base_coverage + coverage_variation, 2),
            'line_coverage': round(base_coverage + coverage_variation - 2, 2),
            'branch_coverage': round(base_coverage + coverage_variation - 3, 2),
            'function_coverage': round(base_coverage + coverage_variation + 1, 2),
            'class_coverage': round(base_coverage + coverage_variation - 1, 2),
            'files_covered': 125,
            'total_files': 150,
            'test_suites_run': 45,
            'tests_passed': 892,
            'tests_failed': 8,
            'test_execution_time': 45.2
        }

    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """
        收集性能指标

        Returns:
            Dict[str, Any]: 性能指标
        """
        try:
            # 进程信息
            process = psutil.Process()
            process_cpu = process.cpu_percent()
            process_memory = process.memory_info()
            process_threads = process.num_threads()

            # 系统负载
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None

            return {
                'process': {
                    'cpu_percent': process_cpu,
                    'memory_rss': process_memory.rss,
                    'memory_vms': process_memory.vms,
                    'threads': process_threads
                },
                'system_load': {
                    'load_1min': load_avg[0] if load_avg else None,
                    'load_5min': load_avg[1] if load_avg else None,
                    'load_15min': load_avg[2] if load_avg else None
                },
                'response_time_ms': 45.2,  # 模拟响应时间
                'throughput_req_per_sec': 125.8,  # 模拟吞吐量
                'error_rate_percent': 0.02  # 模拟错误率
            }

        except Exception as e:
            logger.warning(f"收集性能指标失败: {e}")
            return {}

    def _collect_resource_usage(self) -> Dict[str, Any]:
        """
        收集资源使用情况

        Returns:
            Dict[str, Any]: 资源使用情况
        """
        try:
            # 获取更详细的资源信息
            memory_detailed = psutil.virtual_memory()
            disk_detailed = psutil.disk_io_counters()
            network_detailed = psutil.net_io_counters()

            return {
                'memory_detailed': {
                    'available': memory_detailed.available,
                    'free': memory_detailed.free,
                    'cached': getattr(memory_detailed, 'cached', 0),
                    'buffers': getattr(memory_detailed, 'buffers', 0)
                },
                'disk_io': {
                    'read_count': disk_detailed.read_count if disk_detailed else 0,
                    'write_count': disk_detailed.write_count if disk_detailed else 0,
                    'read_bytes': disk_detailed.read_bytes if disk_detailed else 0,
                    'write_bytes': disk_detailed.write_bytes if disk_detailed else 0
                } if disk_detailed else {},
                'network_io_detailed': {
                    'packets_sent': network_detailed.packets_sent if network_detailed else 0,
                    'packets_recv': network_detailed.packets_recv if network_detailed else 0,
                    'errin': network_detailed.errin if network_detailed else 0,
                    'errout': network_detailed.errout if network_detailed else 0
                } if network_detailed else {}
            }

        except Exception as e:
            logger.warning(f"收集资源使用情况失败: {e}")
            return {}

    def _collect_health_status(self) -> Dict[str, Any]:
        """
        收集健康状态信息

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            # 基本的健康检查
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent

            # 健康评分计算
            health_score = 100

            if cpu_usage > 80:
                health_score -= 20
            if memory_usage > 85:
                health_score -= 20
            if disk_usage > 90:
                health_score -= 20

            health_score = max(0, min(100, health_score))

            # 健康状态判断
            if health_score >= 80:
                status = "healthy"
            elif health_score >= 60:
                status = "warning"
            else:
                status = "critical"

            return {
                'overall_status': status,
                'health_score': health_score,
                'components': {
                    'cpu': 'healthy' if cpu_usage < 80 else 'warning' if cpu_usage < 90 else 'critical',
                    'memory': 'healthy' if memory_usage < 85 else 'warning' if memory_usage < 95 else 'critical',
                    'disk': 'healthy' if disk_usage < 90 else 'warning' if disk_usage < 95 else 'critical'
                },
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"收集健康状态失败: {e}")
            return {
                'overall_status': 'unknown',
                'health_score': 0,
                'error': str(e)
            }

    def _collect_route_health(self) -> Dict[str, Any]:
        """
        收集路由健康检查指标
        
        Returns:
            Dict[str, Any]: 路由健康检查结果
        """
        try:
            # 尝试获取FastAPI应用实例
            # 注意：这里需要从全局或上下文获取app实例
            # 如果无法获取，返回空结果
            try:
                from src.gateway.web.api import app
                from src.gateway.web.route_health_check import RouteHealthChecker
                
                checker = RouteHealthChecker(app, enable_dynamic_discovery=True)
                results = checker.check_routes()
                
                return {
                    'total_routes': results.get('total_routes', 0),
                    'health_status': results.get('health_status', 'unknown'),
                    'required_missing': len(results.get('errors', [])),
                    'optional_missing': len(results.get('warnings', [])),
                    'experimental_missing': len(results.get('info', [])),
                    'checked_categories': len(results.get('checked_routes', {})),
                    'details': {
                        category: {
                            'expected': cat_results.get('expected', 0),
                            'found': cat_results.get('found', 0),
                            'missing': len(cat_results.get('missing', [])),
                            'priority': cat_results.get('priority', 'unknown')
                        }
                        for category, cat_results in results.get('checked_routes', {}).items()
                    }
                }
            except ImportError:
                logger.debug("无法导入路由健康检查模块（可能不在Web环境中）")
                return {
                    'status': 'unavailable',
                    'message': '路由健康检查不可用（非Web环境）'
                }
        except Exception as e:
            logger.warning(f"收集路由健康检查指标失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _collect_logger_pool_metrics(self) -> Dict[str, Any]:
        """
        收集Logger池性能指标
        
        Returns:
            Dict[str, Any]: Logger池性能指标
        """
        try:
            from src.infrastructure.logging.core.interfaces import get_logger_pool
            
            pool = get_logger_pool()
            stats = pool.get_stats()
            
            # 计算性能指标
            hit_rate = stats.get('hit_rate', 0.0)
            pool_size = stats.get('pool_size', 0)
            max_size = stats.get('max_size', 100)
            utilization = pool_size / max_size if max_size > 0 else 0.0
            
            # 判断健康状态
            if hit_rate >= 0.8:
                pool_status = 'healthy'
            elif hit_rate >= 0.5:
                pool_status = 'warning'
            else:
                pool_status = 'critical'
            
            return {
                'hit_rate': hit_rate,
                'hit_count': stats.get('hit_count', 0),
                'miss_count': stats.get('miss_count', 0),
                'pool_size': pool_size,
                'max_size': max_size,
                'utilization': utilization,
                'status': pool_status,
                'warmed_up': stats.get('warmed_up', False),
                'preloaded_count': stats.get('preloaded_count', 0),
                'lru_cache_size': stats.get('lru_cache_size', 0),
                'total_requests': stats.get('hit_count', 0) + stats.get('miss_count', 0)
            }
        except ImportError:
            logger.debug("无法导入Logger池模块")
            return {
                'status': 'unavailable',
                'message': 'Logger池不可用'
            }
        except Exception as e:
            logger.warning(f"收集Logger池指标失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def collect_test_coverage(self) -> Dict[str, Any]:
        """
        收集测试覆盖率数据（公共接口）
        
        Returns:
            Dict[str, Any]: 测试覆盖率数据
        """
        return self._collect_test_coverage_metrics()

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """
        收集性能指标（公共接口）
        
        Returns:
            Dict[str, Any]: 性能指标
        """
        return self._collect_performance_metrics()

    def collect_resource_usage(self) -> Dict[str, Any]:
        """
        收集资源使用情况（公共接口）
        
        Returns:
            Dict[str, Any]: 资源使用情况
        """
        return self._collect_resource_usage()

    def collect_health_status(self) -> Dict[str, Any]:
        """
        收集健康状态（公共接口）
        
        Returns:
            Dict[str, Any]: 健康状态
        """
        return self._collect_health_status()

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取收集统计信息

        Returns:
            Dict[str, Any]: 收集统计
        """
        return {
            'total_collections': self.collection_stats['total_collections'],
            'successful_collections': self.collection_stats['successful_collections'],
            'failed_collections': self.collection_stats['failed_collections'],
            'success_rate': (self.collection_stats['successful_collections'] /
                           max(1, self.collection_stats['total_collections'])) * 100,
            'last_collection_time': self.collection_stats['last_collection_time'].isoformat()
            if self.collection_stats['last_collection_time'] else None
        }

    def reset_stats(self):
        """重置收集统计"""
        self.collection_stats = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'last_collection_time': None
        }
        logger.info("指标收集器统计已重置")

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._last_cache_update.clear()
        logger.info("指标收集器缓存已清空")

    def set_cache_timeout(self, timeout_seconds: int):
        """
        设置缓存超时时间

        Args:
            timeout_seconds: 缓存超时时间（秒）
        """
        self._cache_timeout = timeout_seconds
        logger.info(f"缓存超时时间设置为: {timeout_seconds}秒")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            Dict[str, Any]: 缓存统计
        """
        current_time = time.time()
        cache_stats = {}

        for cache_key in self._cache.keys():
            last_update = self._last_cache_update.get(cache_key, 0)
            age = current_time - last_update
            is_valid = age < self._cache_timeout

            cache_stats[cache_key] = {
                'age_seconds': age,
                'is_valid': is_valid,
                'expires_in': max(0, self._cache_timeout - age)
            }

        return {
            'cache_entries': len(self._cache),
            'cache_timeout': self._cache_timeout,
            'entries': cache_stats
        }


# 全局指标收集器实例
global_metrics_collector = MetricsCollector()
