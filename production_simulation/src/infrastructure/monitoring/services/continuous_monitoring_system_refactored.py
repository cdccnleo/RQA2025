#!/usr/bin/env python3
"""
RQA2025 基础设施层连续监控和优化系统 (重构版)

提供连续监控和优化功能，支持实时监控、性能分析和自动化优化建议。

重构说明:
- 拆分为多个职责单一的组件
- 使用参数对象模式替换长参数列表
- 提高代码可维护性和可测试性
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import threading

from ..core.parameter_objects import (
    MonitoringConfig,
    PerformanceMetricsConfig,
    CoverageCollectionConfig,
    ResourceUsageConfig,
    OptimizationSuggestionConfig,
    DataPersistenceConfig
)
from ..components.monitoring_coordinator import MonitoringCoordinator
from ..components.alert_manager import AlertManager
from ..components.data_persistor import DataPersistor
from ..core.health_check_interface import HealthCheckInterface


class PerformanceMetricsCollector:
    """
    性能指标收集器

    专门负责收集各种性能指标的组件。
    """

    def __init__(self, config: PerformanceMetricsConfig):
        """
        初始化性能指标收集器

        Args:
            config: 性能指标收集配置
        """
        self.config = config

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """
        收集性能指标

        Returns:
            Dict[str, Any]: 性能指标数据
        """
        # 这里应该实现实际的性能指标收集逻辑
        # 现在返回模拟数据
        return {
            'timestamp': datetime.now(),
            'response_time_ms': 4.20,
            'throughput_tps': 2000,
            'memory_usage_mb': 512.0,
            'cpu_usage_percent': 45.5,
            'disk_usage_percent': 50.0,
            'network_io': {
                'bytes_sent': 1024000,
                'bytes_recv': 2048000
            }
        }

    def collect_test_coverage_metrics(self) -> Dict[str, Any]:
        """
        收集测试覆盖率指标

        Returns:
            Dict[str, Any]: 测试覆盖率数据
        """
        return {
            'timestamp': datetime.now(),
            'coverage_percent': 85.5,
            'lines_covered': 1250,
            'lines_total': 1500,
            'branches_covered': 450,
            'branches_total': 500,
            'functions_covered': 95,
            'functions_total': 100
        }

    def collect_resource_usage_metrics(self) -> Dict[str, Any]:
        """
        收集资源使用指标

        Returns:
            Dict[str, Any]: 资源使用数据
        """
        return {
            'timestamp': datetime.now(),
            'cpu_percent': 45.5,
            'memory_percent': 65.2,
            'disk_percent': 50.0,
            'network_bytes_sent': 1024000,
            'network_bytes_recv': 2048000,
            'process_count': 25,
            'thread_count': 150
        }


class OptimizationAdvisor:
    """
    优化建议顾问

    基于监控数据生成优化建议的组件。
    """

    def __init__(self, config: OptimizationSuggestionConfig):
        """
        初始化优化建议顾问

        Args:
            config: 优化建议配置
        """
        self.config = config

    def generate_optimization_suggestions(self, performance_data: Dict[str, Any],
                                       coverage_data: Dict[str, Any],
                                       resource_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成优化建议

        Args:
            performance_data: 性能数据
            coverage_data: 覆盖率数据
            resource_data: 资源数据

        Returns:
            List[Dict[str, Any]]: 优化建议列表
        """
        suggestions = []

        # 性能优化建议
        perf_suggestions = self._generate_performance_suggestions(performance_data)
        suggestions.extend(perf_suggestions)

        # 覆盖率优化建议
        coverage_suggestions = self._generate_coverage_suggestions(coverage_data)
        suggestions.extend(coverage_suggestions)

        # 资源优化建议
        resource_suggestions = self._generate_resource_suggestions(resource_data)
        suggestions.extend(resource_suggestions)

        return suggestions

    def _generate_performance_suggestions(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成性能优化建议"""
        suggestions = []

        response_time = performance_data.get('response_time_ms', 0)
        if response_time > 10:
            suggestions.append({
                'type': 'performance',
                'priority': 'high',
                'title': '响应时间优化',
                'description': f'当前响应时间{response_time:.1f}ms过高，建议优化数据库查询',
                'actions': [
                    '检查数据库查询性能',
                    '添加适当的索引',
                    '考虑使用缓存'
                ],
                'timestamp': datetime.now()
            })

        return suggestions

    def _generate_coverage_suggestions(self, coverage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成覆盖率优化建议"""
        suggestions = []

        coverage_percent = coverage_data.get('coverage_percent', 0)
        if coverage_percent < 80:
            suggestions.append({
                'type': 'coverage',
                'priority': 'medium',
                'title': '提升测试覆盖率',
                'description': f'当前覆盖率仅为{coverage_percent:.1f}%，建议补充单元测试',
                'actions': [
                    '为未覆盖的代码添加单元测试',
                    '实现边界条件测试',
                    '完善集成测试'
                ],
                'timestamp': datetime.now()
            })

        return suggestions

    def _generate_resource_suggestions(self, resource_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成资源优化建议"""
        suggestions = []

        memory_percent = resource_data.get('memory_percent', 0)
        if memory_percent > 80:
            suggestions.append({
                'type': 'resource',
                'priority': 'high',
                'title': '内存使用优化',
                'description': f'内存使用率{memory_percent:.1f}%过高，建议优化内存管理',
                'actions': [
                    '检查内存泄漏',
                    '优化数据结构',
                    '考虑使用内存池'
                ],
                'timestamp': datetime.now()
            })

        return suggestions


class ContinuousMonitoringSystemRefactored(HealthCheckInterface):
    """
    连续监控和优化系统 (重构版)

    使用组件化架构的连续监控系统，提供：
    - 模块化的监控组件
    - 智能的优化建议
    - 统一的数据持久化
    """

    def __init__(self, project_root: Optional[str] = None,
                 monitoring_config: Optional[MonitoringConfig] = None,
                 performance_config: Optional[PerformanceMetricsConfig] = None,
                 coverage_config: Optional[CoverageCollectionConfig] = None,
                 resource_config: Optional[ResourceUsageConfig] = None,
                 optimization_config: Optional[OptimizationSuggestionConfig] = None,
                 persistence_config: Optional[DataPersistenceConfig] = None):
        """
        初始化连续监控系统

        Args:
            project_root: 项目根目录
            monitoring_config: 监控配置
            performance_config: 性能配置
            coverage_config: 覆盖率配置
            resource_config: 资源配置
            optimization_config: 优化配置
            persistence_config: 持久化配置
        """
        # 服务信息 (实现HealthCheckInterface)
        self._service_name = "continuous_monitoring_system_refactored"
        self._service_version = "2.1.0"

        # 监控状态
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.start_time: Optional[datetime] = None

        # 使用默认配置
        self.monitoring_config = monitoring_config or MonitoringConfig()
        self.performance_config = performance_config or PerformanceMetricsConfig()
        self.coverage_config = coverage_config or CoverageCollectionConfig()
        self.resource_config = resource_config or ResourceUsageConfig()
        self.optimization_config = optimization_config or OptimizationSuggestionConfig()
        self.persistence_config = persistence_config or DataPersistenceConfig()

        # 初始化组件
        self._init_components(project_root)

    def _init_components(self, project_root: Optional[str]):
        """初始化组件"""
        # 性能指标收集器
        self.metrics_collector = PerformanceMetricsCollector(self.performance_config)

        # 告警管理器
        self.alert_manager = AlertManager("continuous_monitoring", self.monitoring_config.alert_thresholds)

        # 优化建议顾问
        self.optimization_advisor = OptimizationAdvisor(self.optimization_config)

        # 数据持久化器
        self.data_persistor = DataPersistor("continuous_monitoring", self.persistence_config)

        # 监控协调器
        self.monitoring_coordinator = MonitoringCoordinator("continuous_monitoring", self.monitoring_config)
        self.monitoring_coordinator.set_components(
            self.metrics_collector,
            self.alert_manager,
            None  # 这个系统主要关注数据收集，不需要指标导出器
        )

    def start_monitoring(self) -> bool:
        """
        启动监控

        Returns:
            bool: 是否成功启动
        """
        try:
            if self.monitoring_active:
                return True

            self.monitoring_active = True
            self.start_time = datetime.now()

            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="ContinuousMonitoringSystem",
                daemon=True
            )
            self.monitoring_thread.start()

            print(f"✅ 连续监控系统已启动")
            return True

        except Exception as e:
            self.monitoring_active = False
            print(f"❌ 启动连续监控系统失败: {e}")
            return False

    def stop_monitoring(self) -> bool:
        """
        停止监控

        Returns:
            bool: 是否成功停止
        """
        try:
            if not self.monitoring_active:
                return True

            self.monitoring_active = False

            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)

            print(f"✅ 连续监控系统已停止")
            return True

        except Exception as e:
            print(f"❌ 停止连续监控系统失败: {e}")
            return False

    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 执行监控周期
                self._execute_monitoring_cycle()

                # 等待下一个周期
                import time
                time.sleep(self.monitoring_config.collection_interval)

            except Exception as e:
                print(f"监控循环异常: {e}")
                import time
                time.sleep(self.monitoring_config.collection_interval)

    def _execute_monitoring_cycle(self):
        """执行监控周期"""
        try:
            # 收集各种指标
            performance_data = self.metrics_collector.collect_performance_metrics()
            coverage_data = self.metrics_collector.collect_test_coverage_metrics()
            resource_data = self.metrics_collector.collect_resource_usage_metrics()

            # 合并所有监控数据
            monitoring_data = {
                'performance': performance_data,
                'coverage': coverage_data,
                'resource': resource_data,
                'timestamp': datetime.now()
            }

            # 持久化数据
            self.data_persistor.persist_data(monitoring_data)

            # 生成优化建议
            suggestions = self.optimization_advisor.generate_optimization_suggestions(
                performance_data, coverage_data, resource_data
            )

            # 存储建议
            if suggestions:
                suggestions_data = {
                    'suggestions': suggestions,
                    'timestamp': datetime.now(),
                    'type': 'optimization_suggestions'
                }
                self.data_persistor.persist_data(suggestions_data)

        except Exception as e:
            print(f"执行监控周期失败: {e}")

    def get_monitoring_report(self) -> Dict[str, Any]:
        """
        获取监控报告

        Returns:
            Dict[str, Any]: 监控报告
        """
        try:
            # 获取最新数据
            recent_data = self.data_persistor.retrieve_data(limit=10)

            # 生成报告
            report = {
                'service_name': self._service_name,
                'service_version': self._service_version,
                'monitoring_active': self.monitoring_active,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'data_statistics': self.data_persistor.get_data_statistics(),
                'alert_statistics': self.alert_manager.get_alert_statistics(),
                'recent_data': recent_data[-5:] if recent_data else [],
                'generated_at': datetime.now().isoformat()
            }

            return report

        except Exception as e:
            print(f"生成监控报告失败: {e}")
            return {'error': str(e)}

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能汇总

        Returns:
            Dict[str, Any]: 性能汇总数据
        """
        try:
            # 获取性能相关数据
            performance_entries = []
            for entry in self.data_persistor.retrieve_data(limit=100):
                if 'performance' in entry.get('data', {}):
                    performance_entries.append(entry)

            if not performance_entries:
                return {'message': '暂无性能数据'}

            # 计算汇总统计
            response_times = []
            memory_usage = []
            cpu_usage = []

            for entry in performance_entries[-20:]:  # 最近20条记录
                perf_data = entry['data']['performance']
                response_times.append(perf_data.get('response_time_ms', 0))
                memory_usage.append(perf_data.get('memory_usage_mb', 0))
                cpu_usage.append(perf_data.get('cpu_usage_percent', 0))

            return {
                'response_time_avg': sum(response_times) / len(response_times) if response_times else 0,
                'response_time_max': max(response_times) if response_times else 0,
                'response_time_min': min(response_times) if response_times else 0,
                'memory_usage_avg': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'cpu_usage_avg': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                'data_points': len(performance_entries),
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"获取性能汇总失败: {e}")
            return {'error': str(e)}

    def get_optimization_suggestions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取优化建议

        Args:
            limit: 返回的最大建议数量

        Returns:
            List[Dict[str, Any]]: 优化建议列表
        """
        try:
            suggestions_entries = []
            for entry in self.data_persistor.retrieve_data(limit=100):
                if entry.get('data', {}).get('type') == 'optimization_suggestions':
                    suggestions_entries.extend(entry['data'].get('suggestions', []))

            # 按优先级排序
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            suggestions_entries.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 3))

            return suggestions_entries[:limit]

        except Exception as e:
            print(f"获取优化建议失败: {e}")
            return []

    def export_monitoring_data(self, file_path: str, format_type: str = 'json') -> bool:
        """
        导出监控数据

        Args:
            file_path: 导出文件路径
            format_type: 导出格式

        Returns:
            bool: 是否成功导出
        """
        return self.data_persistor.export_data(file_path, format_type)

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        清理旧数据

        Args:
            days_to_keep: 保留天数

        Returns:
            int: 删除的记录数量
        """
        return self.data_persistor.cleanup_old_data(days_to_keep)

    # HealthCheckInterface 实现
    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        return {
            'service': self._service_name,
            'version': self._service_version,
            'status': 'healthy' if self.monitoring_active else 'stopped',
            'monitoring_active': self.monitoring_active,
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'last_check': datetime.now().isoformat()
        }

    # 上下文管理器支持
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
