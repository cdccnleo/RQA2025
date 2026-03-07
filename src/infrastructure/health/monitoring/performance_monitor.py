"""
performance_monitor 模块

提供 performance_monitor 相关功能和接口。
"""

import logging

import gc
import tracemalloc

from datetime import datetime, timedelta
from src.infrastructure.health.core.interfaces import IUnifiedInfrastructureInterface
from typing import Dict, List, Any, Optional
"""
性能监控器模块

提供系统性能监控和内存追踪功能，包括内存使用分析、性能指标收集和告警机制。
"""

logger = logging.getLogger(__name__)

# 常量定义 - 清理魔法数字
TOP_STATS_LIMIT = 10  # 顶部统计显示数量
STATS_DISPLAY_LIMIT = 20  # 统计显示限制
DEFAULT_MEMORY_LEAK_THRESHOLD_MB = 50.0  # 默认内存泄漏阈值(MB)
DEFAULT_ALERT_HOURS = 24  # 默认告警时间范围(小时)
MEMORY_UNIT_CONVERSION = 1024 * 1024  # MB到字节的转换


class PerformanceMonitor(IUnifiedInfrastructureInterface):
    """
    性能监控器

    负责监控系统性能，包括内存使用情况、性能指标收集、内存泄漏检测等。
    提供性能分析和告警功能。
    """

    def __init__(self):
        """初始化性能监控器"""
        self.tracemalloc_started = False
        self.performance_data = {}
        self.alerts = []
        self.snapshots = []

    def record(self, operation: str, duration: float, tags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        记录单次操作的性能数据，并返回更新后的摘要信息。
        该方法为测试保留的兼容接口，确保在高频调用时仍然安全。
        """
        operation_name = operation or "unknown_operation"
        duration_value = float(duration) if duration is not None else 0.0
        tags = tags or {}

        stats = self.performance_data.setdefault(
            operation_name,
            {
                "count": 0,
                "total_duration": 0.0,
                "max_duration": 0.0,
                "history": [],
            },
        )

        stats["count"] += 1
        stats["total_duration"] += duration_value
        stats["max_duration"] = max(stats["max_duration"], duration_value)
        stats["history"].append(
            {"timestamp": datetime.now().isoformat(), "duration": duration_value, "tags": tags}
        )
        if len(stats["history"]) > 1000:
            stats["history"] = stats["history"][-1000:]

        return {
            "operation": operation_name,
            "count": stats["count"],
            "duration": duration_value,
            "average_duration": stats["total_duration"] / max(stats["count"], 1),
        }

    def start_memory_tracing(self):
        """开始内存追踪"""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
            # 记录初始快照
            self.snapshots.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'start',
                'snapshot': tracemalloc.take_snapshot()
            })
            logger.info("内存追踪已启动")

    def stop_memory_tracing(self):
        """停止内存追踪"""
        if self.tracemalloc_started:
            # 记录结束快照
            self.snapshots.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'end',
                'snapshot': tracemalloc.take_snapshot()
            })
            tracemalloc.stop()
            self.tracemalloc_started = False
            logger.info("内存追踪已停止")

    def take_memory_snapshot(self) -> Dict[str, Any]:
        """
        获取内存快照

        Returns:
            内存快照信息字典
        """
        if not self.tracemalloc_started:
            return {'error': 'Memory tracing not started'}

        try:
            snapshot = tracemalloc.take_snapshot()

            # 记录快照
            self.snapshots.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'manual',
                'snapshot': snapshot
            })

            # 分析快照
            top_stats = snapshot.statistics('lineno')
            total_size = sum(stat.size for stat in top_stats)

            memory_info = {
                'timestamp': datetime.now().isoformat(),
                'total_size': total_size,
                'total_count': sum(stat.count for stat in top_stats),
                'top_allocations': []
            }

            # 获取前N个最大内存分配
            for stat in top_stats[:TOP_STATS_LIMIT]:
                memory_info['top_allocations'].append({
                    'size': stat.size,
                    'count': stat.count,
                    'average': stat.size / stat.count if stat.count > 0 else 0,
                    'traceback': str(stat.traceback)
                })

            return memory_info

        except Exception as e:
            logger.error(f"获取内存快照失败: {e}")
            return {'error': str(e)}

    def compare_memory_snapshots(self, snapshot1_index: int = -2,
                                 snapshot2_index: int = -1) -> Dict[str, Any]:
        """
        比较两个内存快照

        Args:
            snapshot1_index: 第一个快照索引
            snapshot2_index: 第二个快照索引

        Returns:
            快照比较结果
        """
        if len(self.snapshots) < 2:
            return {'error': '需要至少两个快照进行比较'}

        try:
            # 验证和调整快照索引
            valid_indices = self._validate_snapshot_indices(snapshot1_index, snapshot2_index)
            if 'error' in valid_indices:
                return valid_indices

            snapshot1_index, snapshot2_index = valid_indices['indices']

            # 提取快照数据
            snapshot_data = self._extract_snapshot_data(snapshot1_index, snapshot2_index)
            if 'error' in snapshot_data:
                return snapshot_data

            snapshot1, snapshot2 = snapshot_data['snapshots']

            # 执行快照比较
            stats = snapshot2.compare_to(snapshot1, 'lineno')

            # 格式化比较结果
            return self._format_comparison_results(snapshot1_index, snapshot2_index, stats)

        except Exception as e:
            logger.error(f"比较内存快照失败: {e}")
            return {'error': str(e)}

    def _validate_snapshot_indices(self, snapshot1_index: int,
                                   snapshot2_index: int) -> Dict[str, Any]:
        """
        验证和调整快照索引

        Args:
            snapshot1_index: 第一个快照索引
            snapshot2_index: 第二个快照索引

        Returns:
            验证结果和调整后的索引
        """
        if snapshot1_index < 0:
            snapshot1_index += len(self.snapshots)
        if snapshot2_index < 0:
            snapshot2_index += len(self.snapshots)

        if not (0 <= snapshot1_index < len(self.snapshots)
                and 0 <= snapshot2_index < len(self.snapshots)):
            return {'error': '快照索引超出范围'}

        return {'indices': (snapshot1_index, snapshot2_index)}

    def _extract_snapshot_data(self, snapshot1_index: int,
                               snapshot2_index: int) -> Dict[str, Any]:
        """
        提取快照数据

        Args:
            snapshot1_index: 第一个快照索引
            snapshot2_index: 第二个快照索引

        Returns:
            快照数据
        """
        try:
            snapshot1 = self.snapshots[snapshot1_index]['snapshot']
            snapshot2 = self.snapshots[snapshot2_index]['snapshot']
            return {'snapshots': (snapshot1, snapshot2)}
        except (IndexError, KeyError) as e:
            return {'error': f'提取快照数据失败: {e}'}

    def _format_comparison_results(self, snapshot1_index: int, snapshot2_index: int,
                                   stats) -> Dict[str, Any]:
        """
        格式化比较结果

        Args:
            snapshot1_index: 第一个快照索引
            snapshot2_index: 第二个快照索引
            stats: 比较统计数据

        Returns:
            格式化的比较结果
        """
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'snapshot1_time': self.snapshots[snapshot1_index]['timestamp'],
            'snapshot2_time': self.snapshots[snapshot2_index]['timestamp'],
            'changes': []
        }

        for stat in stats[:STATS_DISPLAY_LIMIT]:  # 只显示前N个变化
            comparison['changes'].append({
                'size_diff': stat.size_diff,
                'count_diff': stat.count_diff,
                'size': stat.size,
                'count': stat.count,
                'traceback': str(stat.traceback)
            })

        return comparison

    def get_memory_usage_trend(self) -> Dict[str, Any]:
        """
        获取内存使用趋势

        Returns:
            内存使用趋势分析
        """
        if not self.snapshots:
            return {'error': '没有可用的内存快照'}

        try:
            trend_data = []

            for snapshot_info in self.snapshots:
                snapshot = snapshot_info['snapshot']
                stats = snapshot.statistics('lineno')
                total_size = sum(stat.size for stat in stats)
                total_count = sum(stat.count for stat in stats)

                trend_data.append({
                    'timestamp': snapshot_info['timestamp'],
                    'total_size': total_size,
                    'total_count': total_count,
                    'type': snapshot_info['type']
                })

            return {
                'trend': trend_data,
                'summary': {
                    'total_snapshots': len(trend_data),
                    'start_time': trend_data[0]['timestamp'] if trend_data else None,
                    'end_time': trend_data[-1]['timestamp'] if trend_data else None,
                    'max_memory': max((d['total_size'] for d in trend_data), default=0),
                    'min_memory': min((d['total_size'] for d in trend_data), default=0)
                }
            }

        except Exception as e:
            logger.error(f"获取内存使用趋势失败: {e}")
            return {'error': str(e)}

    def detect_memory_leaks(self, threshold_mb: float = DEFAULT_MEMORY_LEAK_THRESHOLD_MB) -> Dict[str, Any]:
        """
        检测内存泄漏

        Args:
            threshold_mb: 内存增长阈值(MB)

        Returns:
            内存泄漏检测结果
        """
        validation_result = self._validate_leak_detection_requirements()
        if 'error' in validation_result:
            return validation_result

        try:
            # 计算内存增长
            growth_data = self._calculate_memory_growth()
            if 'error' in growth_data:
                return growth_data

            # 识别显著增长
            significant_growth = self._identify_significant_growth(
                growth_data['stats'], threshold_mb
            )

            # 格式化泄漏信息
            return self._format_leak_info(significant_growth, threshold_mb)

        except Exception as e:
            logger.error(f"内存泄漏检测失败: {e}")
            return {'error': str(e)}

    def _validate_leak_detection_requirements(self) -> Dict[str, Any]:
        """
        验证内存泄漏检测的要求

        Returns:
            验证结果
        """
        if len(self.snapshots) < 2:
            return {'error': '需要至少两个快照进行内存泄漏检测'}
        return {}

    def _calculate_memory_growth(self) -> Dict[str, Any]:
        """
        计算内存增长

        Returns:
            内存增长数据
        """
        try:
            # 获取第一个和最后一个快照
            first_snapshot = self.snapshots[0]['snapshot']
            last_snapshot = self.snapshots[-1]['snapshot']

            # 比较快照
            stats = last_snapshot.compare_to(first_snapshot, 'lineno')

            return {'stats': stats}
        except Exception as e:
            return {'error': f'计算内存增长失败: {e}'}

    def _identify_significant_growth(self, stats, threshold_mb: float):
        """
        识别显著的内存增长

        Args:
            stats: 比较统计数据
            threshold_mb: 内存增长阈值(MB)

        Returns:
            显著增长的统计数据列表
        """
        threshold_bytes = threshold_mb * MEMORY_UNIT_CONVERSION
        return [
            stat for stat in stats
            if stat.size_diff > threshold_bytes
        ]

    def _format_leak_info(self, significant_growth, threshold_mb: float) -> Dict[str, Any]:
        """
        格式化内存泄漏信息

        Args:
            significant_growth: 显著增长的统计数据
            threshold_mb: 内存增长阈值(MB)

        Returns:
            格式化的泄漏信息
        """
        leak_info = {
            'timestamp': datetime.now().isoformat(),
            'duration_hours': (
                datetime.fromisoformat(self.snapshots[-1]['timestamp'])
                - datetime.fromisoformat(self.snapshots[0]['timestamp'])
            ).total_seconds() / 3600,  # 转换为小时
            'total_snapshots': len(self.snapshots),
            'threshold_mb': threshold_mb,
            'potential_leaks': []
        }

        for stat in significant_growth[:TOP_STATS_LIMIT]:  # 只报告前N个潜在泄漏
            leak_info['potential_leaks'].append({
                'size_increase': stat.size_diff,
                'size_increase_mb': stat.size_diff / MEMORY_UNIT_CONVERSION,
                'count_increase': stat.count_diff,
                'traceback': str(stat.traceback)
            })

        leak_info['leak_count'] = len(leak_info['potential_leaks'])
        leak_info['total_leak_mb'] = sum(
            leak['size_increase_mb'] for leak in leak_info['potential_leaks']
        )

        return leak_info

    def add_performance_alert(self, alert_type: str, message: str,
                              severity: str = "info", metadata: Optional[Dict[str, Any]] = None):
        """
        添加性能告警

        Args:
            alert_type: 告警类型
            message: 告警消息
            severity: 严重程度 (info, warning, error, critical)
            metadata: 额外元数据
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity,
            'metadata': metadata or {}
        }

        self.alerts.append(alert)
        logger.warning(f"性能告警: {alert_type} - {message}")

    def get_performance_alerts(self, hours: int = DEFAULT_ALERT_HOURS) -> List[Dict[str, Any]]:
        """
        获取性能告警

        Args:
            hours: 获取最近几小时的告警

        Returns:
            性能告警列表
        """
        if not self.alerts:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]

    def clear_old_alerts(self, hours: int = 168):  # 7天
        """
        清理旧的告警

        Args:
            hours: 保留最近几小时的告警
        """
        if not self.alerts:
            return

        cutoff_time = datetime.now() - timedelta(hours=hours)
        original_count = len(self.alerts)
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]

        removed_count = original_count - len(self.alerts)
        if removed_count > 0:
            logger.info(f"清理了 {removed_count} 个旧的性能告警")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要

        Returns:
            性能监控摘要信息
        """
        return {
            'memory_tracing_active': self.tracemalloc_started,
            'total_snapshots': len(self.snapshots),
            'total_alerts': len(self.alerts),
            'recent_alerts': len(self.get_performance_alerts(24)),
            'memory_info': self.take_memory_snapshot() if self.tracemalloc_started else None
        }

    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        强制垃圾回收

        Returns:
            垃圾回收结果
        """
        try:
            collected_objects = gc.collect()
            gc_stats = {
                'collected_objects': collected_objects,
                'generations': gc.get_stats(),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"强制垃圾回收完成，回收了 {collected_objects} 个对象")
            return gc_stats

        except Exception as e:
            logger.error(f"强制垃圾回收失败: {e}")
            return {'error': str(e)}

    def monitor_function_performance(self, func_name: str, execution_time: float,
                                     threshold: float = 1.0):
        """
        监控函数性能

        Args:
            func_name: 函数名称
            execution_time: 执行时间(秒)
            threshold: 性能阈值(秒)
        """
        if execution_time > threshold:
            self.add_performance_alert(
                'slow_function',
                f"函数 {func_name} 执行时间过长: {execution_time:.3f}秒",
                'warning',
                {
                    'function': func_name,
                    'execution_time': execution_time,
                    'threshold': threshold
                }
            )

    # =========================================================================
    # 统一基础设施接口实现
    # =========================================================================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化组件"""
        try:
            if config:
                # 可以在这里处理配置更新
                pass

            # 确保tracemalloc已启动
            if not tracemalloc.is_tracing():
                tracemalloc.start()

            logger.info("PerformanceMonitor 初始化成功")
            return True
        except Exception as e:
            logger.error(f"PerformanceMonitor 初始化失败: {e}")
            return False

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "component_type": "PerformanceMonitor",
            "version": "2.0.0",
            "capabilities": ["memory_monitoring", "performance_tracking", "leak_detection"],
            "tracemalloc_enabled": tracemalloc.is_tracing(),
            "snapshots_count": len(self.snapshots) if hasattr(self, 'snapshots') else 0,
            "alerts_count": len(self.alerts) if hasattr(self, 'alerts') else 0
        }

    def is_healthy(self) -> bool:
        """检查组件健康状态"""
        try:
            # 检查tracemalloc状态
            if not tracemalloc.is_tracing():
                return False

            # 检查快照数量是否合理
            if hasattr(self, 'snapshots') and len(self.snapshots) > 1000:  # 防止内存溢出
                return False

            return True
        except Exception:
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标"""
        try:
            current, peak = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)

            return {
                "current_memory_mb": current / (1024 * 1024),
                "peak_memory_mb": peak / (1024 * 1024),
                "snapshots_count": len(self.snapshots) if hasattr(self, 'snapshots') else 0,
                "alerts_count": len(self.alerts) if hasattr(self, 'alerts') else 0,
                "tracemalloc_active": tracemalloc.is_tracing(),
                "gc_stats": {
                    "collections": gc.get_stats(),
                    "objects_count": len(gc.get_objects())
                }
            }
        except Exception as e:
            logger.error(f"获取指标失败: {e}")
            return {"error": str(e)}

    def cleanup(self) -> bool:
        """清理组件资源"""
        try:
            # 清理快照
            if hasattr(self, 'snapshots'):
                self.snapshots.clear()

            # 清理告警
            if hasattr(self, 'alerts'):
                self.alerts.clear()

            # 停止tracemalloc
            if tracemalloc.is_tracing():
                tracemalloc.stop()

            logger.info("PerformanceMonitor 资源清理完成")
            return True
        except Exception as e:
            logger.error(f"PerformanceMonitor 资源清理失败: {e}")
            return False

    # ============================================================================
    # 标准化健康检查方法
    # ============================================================================

    def check_health(self) -> Dict[str, Any]:
        """检查性能监控器整体健康状态

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            logger.info("开始执行性能监控器健康检查")

            # 检查内存追踪状态
            memory_status = self.check_memory_health()

            # 检查性能数据状态
            performance_status = self.check_performance_data_health()

            # 检查告警系统状态
            alert_status = self.check_alert_system_health()

            # 综合判断整体健康状态
            overall_healthy = all([
                memory_status.get('healthy', False),
                performance_status.get('healthy', False),
                alert_status.get('healthy', False)
            ])

            result = {
                'healthy': overall_healthy,
                'timestamp': datetime.now().isoformat(),
                'component': 'performance_monitor',
                'details': {
                    'memory_status': memory_status,
                    'performance_status': performance_status,
                    'alert_status': alert_status
                }
            }

            logger.info(f"性能监控器健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
            return result

        except Exception as e:
            logger.error(f"性能监控器健康检查失败: {e}")
            return {
                'healthy': False,
                'timestamp': datetime.now().isoformat(),
                'component': 'performance_monitor',
                'error': str(e)
            }

    def check_memory_health(self) -> Dict[str, Any]:
        """检查内存监控健康状态

        Returns:
            Dict[str, Any]: 内存监控健康检查结果
        """
        try:
            issues = []

            # 检查tracemalloc状态
            if not tracemalloc.is_tracing():
                issues.append("内存追踪未启动")
            else:
                # 检查内存使用情况
                current, peak = tracemalloc.get_traced_memory()
                current_mb = current / MEMORY_UNIT_CONVERSION
                peak_mb = peak / MEMORY_UNIT_CONVERSION

                if current_mb > 500:  # 内存使用超过500MB
                    issues.append(f"当前内存使用过高: {current_mb:.1f}MB")

                if peak_mb > 1000:  # 峰值内存超过1GB
                    issues.append(f"峰值内存使用过高: {peak_mb:.1f}MB")

            # 检查快照数量
            snapshot_count = len(self.snapshots) if hasattr(self, 'snapshots') else 0
            if snapshot_count > 100:  # 快照数量过多
                issues.append(f"内存快照数量过多: {snapshot_count}个")

            return {
                'healthy': len(issues) == 0,
                'issues': issues,
                'tracemalloc_active': tracemalloc.is_tracing(),
                'snapshots_count': snapshot_count,
                'memory_usage_mb': current_mb if tracemalloc.is_tracing() else 0,
                'peak_memory_mb': peak_mb if tracemalloc.is_tracing() else 0
            }

        except Exception as e:
            logger.error(f"内存监控健康检查失败: {e}")
            return {
                'healthy': False,
                'issues': [f"检查过程异常: {str(e)}"]
            }

    def check_performance_data_health(self) -> Dict[str, Any]:
        """检查性能数据健康状态

        Returns:
            Dict[str, Any]: 性能数据健康检查结果
        """
        try:
            issues = []

            # 检查性能数据存储
            if not hasattr(self, 'performance_data') or not isinstance(self.performance_data, dict):
                issues.append("性能数据存储异常")
            else:
                # 检查数据量
                total_entries = sum(len(data) for data in self.performance_data.values()
                                    if isinstance(data, list))
                if total_entries > 10000:  # 数据量过大
                    issues.append(f"性能数据量过大: {total_entries}条记录")

                # 检查数据类型分布
                data_types = list(self.performance_data.keys())
                if len(data_types) == 0:
                    issues.append("缺少性能数据类型")

            return {
                'healthy': len(issues) == 0,
                'issues': issues,
                'total_performance_entries': total_entries if 'total_entries' in locals() else 0,
                'data_types': data_types if 'data_types' in locals() else [],
                'data_structure_valid': isinstance(self.performance_data, dict)
            }

        except Exception as e:
            logger.error(f"性能数据健康检查失败: {e}")
            return {
                'healthy': False,
                'issues': [f"检查过程异常: {str(e)}"]
            }

    def check_alert_system_health(self) -> Dict[str, Any]:
        """检查告警系统健康状态

        Returns:
            Dict[str, Any]: 告警系统健康检查结果
        """
        try:
            issues = []

            # 检查告警数据存储
            if not hasattr(self, 'alerts') or not isinstance(self.alerts, list):
                issues.append("告警数据存储异常")
            else:
                alert_count = len(self.alerts)
                if alert_count > 1000:  # 告警数量过多
                    issues.append(f"告警数量过多: {alert_count}个告警")

                # 检查近期告警
                recent_alerts = [a for a in self.alerts
                                 if isinstance(a, dict) and 'timestamp' in a
                                 and (datetime.now() - datetime.fromisoformat(a['timestamp'])).days < 1]
                if len(recent_alerts) > 50:  # 24小时内告警过多
                    issues.append(f"24小时内告警过多: {len(recent_alerts)}个")

            return {
                'healthy': len(issues) == 0,
                'issues': issues,
                'alerts_count': len(self.alerts) if hasattr(self, 'alerts') else 0,
                'recent_alerts_count': len(recent_alerts) if 'recent_alerts' in locals() else 0,
                'alert_system_active': hasattr(self, 'alerts')
            }

        except Exception as e:
            logger.error(f"告警系统健康检查失败: {e}")
            return {
                'healthy': False,
                'issues': [f"检查过程异常: {str(e)}"]
            }

    def health_check(self) -> Dict[str, Any]:
        """执行健康检查（别名方法）

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        return self.check_health()

    def monitor_performance_status(self) -> Dict[str, Any]:
        """监控性能监控器状态

        Returns:
            Dict[str, Any]: 性能监控器状态信息
        """
        try:
            current, peak = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)

            return {
                'component': 'performance_monitor',
                'status': 'active',
                'timestamp': datetime.now().isoformat(),
                'memory': {
                    'current_mb': current / MEMORY_UNIT_CONVERSION,
                    'peak_mb': peak / MEMORY_UNIT_CONVERSION,
                    'tracing_active': tracemalloc.is_tracing()
                },
                'data': {
                    'performance_entries': sum(len(data) for data in self.performance_data.values()
                                               if isinstance(data, list)),
                    'alerts_count': len(self.alerts),
                    'snapshots_count': len(self.snapshots)
                },
                'health': self.check_health()
            }
        except Exception as e:
            logger.error(f"获取性能监控器状态失败: {e}")
            return {
                'component': 'performance_monitor',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def validate_performance_config(self) -> Dict[str, Any]:
        """验证性能监控配置有效性

        Returns:
            Dict[str, Any]: 配置验证结果
        """
        try:
            # 检查基本配置有效性
            issues = []

            # 验证内存泄漏阈值
            if DEFAULT_MEMORY_LEAK_THRESHOLD_MB <= 0:
                issues.append("内存泄漏阈值配置无效")

            # 验证告警时间范围
            if DEFAULT_ALERT_HOURS <= 0:
                issues.append("告警时间范围配置无效")

            # 验证统计显示限制
            if TOP_STATS_LIMIT <= 0 or STATS_DISPLAY_LIMIT <= 0:
                issues.append("统计显示限制配置无效")

            return {
                'healthy': len(issues) == 0,
                'issues': issues,
                'config': {
                    'memory_leak_threshold_mb': DEFAULT_MEMORY_LEAK_THRESHOLD_MB,
                    'alert_hours': DEFAULT_ALERT_HOURS,
                    'top_stats_limit': TOP_STATS_LIMIT,
                    'stats_display_limit': STATS_DISPLAY_LIMIT
                }
            }

        except Exception as e:
            logger.error(f"性能监控配置验证失败: {e}")
            return {
                'healthy': False,
                'issues': [f"验证过程异常: {str(e)}"]
            }
