"""
Phase 3.3: 性能和监控优化工具

基于统一接口的性能监控和架构优化
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from functools import wraps


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    component_name: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """系统指标数据类"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_connections: int
    thread_count: int
    process_count: int


class UnifiedPerformanceMonitor:
    """统一性能监控器"""

    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.system_metrics_history: List[SystemMetrics] = []
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = False
        self.logger = logging.getLogger(__name__)

        # 配置
        self.max_history_size = 10000
        self.system_monitoring_interval = 5.0  # 5秒间隔

    def start_system_monitoring(self):
        """启动系统监控"""
        if self._monitoring_thread is None:
            self._stop_monitoring = False
            self._monitoring_thread = threading.Thread(
                target=self._system_monitoring_loop, daemon=True)
            self._monitoring_thread.start()
            self.logger.info("系统性能监控已启动")

    def stop_system_monitoring(self):
        """停止系统监控"""
        if self._monitoring_thread:
            self._stop_monitoring = True
            self._monitoring_thread.join(timeout=5.0)
            self.logger.info("系统性能监控已停止")

    def _system_monitoring_loop(self):
        """系统监控循环"""
        while not self._stop_monitoring:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)

                # 限制历史记录大小
                if len(self.system_metrics_history) > self.max_history_size:
                    self.system_metrics_history = self.system_metrics_history[-self.max_history_size:]

            except Exception as e:
                self.logger.error(f"收集系统指标失败: {e}")

            time.sleep(self.system_monitoring_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        timestamp = time.time()

        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1.0)

            # 内存使用
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / 1024 / 1024

            # 磁盘使用
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            # 网络连接
            network_connections = len(psutil.net_connections())

            # 进程和线程
            thread_count = threading.active_count()
            process_count = len(psutil.pids())

        except Exception as e:
            self.logger.error(f"收集系统指标时出错: {e}")
            # 返回默认值
            cpu_percent = memory_percent = memory_used_mb = disk_usage_percent = 0.0
            network_connections = thread_count = process_count = 0

        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            disk_usage_percent=disk_usage_percent,
            network_connections=network_connections,
            thread_count=thread_count,
            process_count=process_count
        )

    def measure_performance(self, component_name: str, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> Callable:
        """性能测量装饰器"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                error_message = None

                try:
                    # 记录开始时的内存使用
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024

                    result = func(*args, **kwargs)

                    # 记录结束时的内存使用
                    end_memory = process.memory_info().rss / 1024 / 1024
                    memory_usage_mb = end_memory - start_memory

                    return result

                except Exception as e:
                    success = False
                    error_message = str(e)
                    memory_usage_mb = 0.0
                    raise

                finally:
                    end_time = time.time()
                    duration_ms = (end_time - start_time) * 1000

                    # 创建性能指标
                    metrics = PerformanceMetrics(
                        component_name=component_name,
                        operation_name=operation_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration_ms,
                        memory_usage_mb=memory_usage_mb,
                        success=success,
                        error_message=error_message,
                        metadata=metadata or {}
                    )

                    self.record_metrics(metrics)

            return wrapper
        return decorator

    def record_metrics(self, metrics: PerformanceMetrics):
        """记录性能指标"""
        self.metrics_history.append(metrics)

        # 限制历史记录大小
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]

        # 记录到日志
        log_level = logging.INFO if metrics.success else logging.WARNING
        self.logger.log(log_level,
                        f"性能指标: {metrics.component_name}.{metrics.operation_name} - "
                        f"耗时: {metrics.duration_ms:.2f}ms, "
                        f"内存: {metrics.memory_usage_mb:.2f}MB, "
                        f"成功: {metrics.success}")

    def get_performance_summary(self, component_name: Optional[str] = None, hours: int = 1) -> Dict[str, Any]:
        """获取性能摘要"""
        cutoff_time = time.time() - (hours * 3600)

        # 过滤相关指标
        relevant_metrics = [
            m for m in self.metrics_history
            if m.end_time and m.end_time >= cutoff_time and
            (component_name is None or m.component_name == component_name)
        ]

        if not relevant_metrics:
            return {"error": "没有找到相关性能指标"}

        # 计算统计信息
        total_operations = len(relevant_metrics)
        successful_operations = sum(1 for m in relevant_metrics if m.success)
        failed_operations = total_operations - successful_operations

        durations = [m.duration_ms for m in relevant_metrics if m.duration_ms is not None]
        memory_usages = [
            m.memory_usage_mb for m in relevant_metrics if m.memory_usage_mb is not None]

        summary = {
            "time_range_hours": hours,
            "component_filter": component_name,
            "total_operations": total_operations,
            "success_rate": successful_operations / total_operations * 100 if total_operations > 0 else 0,
            "failure_rate": failed_operations / total_operations * 100 if total_operations > 0 else 0,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "avg_memory_mb": sum(memory_usages) / len(memory_usages) if memory_usages else 0,
            "max_memory_mb": max(memory_usages) if memory_usages else 0,
            "recent_operations": [
                {
                    "component": m.component_name,
                    "operation": m.operation_name,
                    "duration_ms": m.duration_ms,
                    "memory_mb": m.memory_usage_mb,
                    "success": m.success,
                    "timestamp": datetime.fromtimestamp(m.end_time).isoformat()
                }
                for m in relevant_metrics[-10:]  # 最近10个操作
            ]
        }

        return summary

    def get_system_health_report(self) -> Dict[str, Any]:
        """获取系统健康报告"""
        if not self.system_metrics_history:
            return {"error": "没有系统监控数据"}

        recent_metrics = self.system_metrics_history[-10:]  # 最近10个数据点

        # 计算平均值
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics)

        # 健康评分 (0-100, 越高越好)
        cpu_score = max(0, 100 - avg_cpu * 2)  # CPU使用率每增加1%, 扣2分
        memory_score = max(0, 100 - avg_memory * 1.5)  # 内存使用率每增加1%, 扣1.5分
        disk_score = max(0, 100 - avg_disk * 3)  # 磁盘使用率每增加1%, 扣3分

        overall_score = (cpu_score + memory_score + disk_score) / 3

        # 健康状态
        if overall_score >= 80:
            health_status = "excellent"
        elif overall_score >= 60:
            health_status = "good"
        elif overall_score >= 40:
            health_status = "warning"
        else:
            health_status = "critical"

        return {
            "health_status": health_status,
            "overall_score": round(overall_score, 2),
            "cpu_score": round(cpu_score, 2),
            "memory_score": round(memory_score, 2),
            "disk_score": round(disk_score, 2),
            "current_metrics": {
                "cpu_percent": recent_metrics[-1].cpu_percent,
                "memory_percent": recent_metrics[-1].memory_percent,
                "memory_used_mb": recent_metrics[-1].memory_used_mb,
                "disk_usage_percent": recent_metrics[-1].disk_usage_percent,
                "network_connections": recent_metrics[-1].network_connections,
                "thread_count": recent_metrics[-1].thread_count,
                "process_count": recent_metrics[-1].process_count,
                "timestamp": datetime.fromtimestamp(recent_metrics[-1].timestamp).isoformat()
            },
            "averages_last_10": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2),
                "disk_usage_percent": round(avg_disk, 2)
            }
        }


class ArchitecturePerformanceOptimizer:
    """架构性能优化器"""

    def __init__(self, monitor: UnifiedPerformanceMonitor):
        self.monitor = monitor
        self.optimization_suggestions = []

    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """分析性能瓶颈"""
        bottlenecks = []

        # 分析慢操作
        all_metrics = self.monitor.metrics_history
        if not all_metrics:
            return bottlenecks

        # 计算平均响应时间
        component_stats = {}
        for metrics in all_metrics:
            key = metrics.component_name
            if key not in component_stats:
                component_stats[key] = {'durations': [], 'errors': 0, 'total': 0}

            if metrics.duration_ms is not None:
                component_stats[key]['durations'].append(metrics.duration_ms)
            if not metrics.success:
                component_stats[key]['errors'] += 1
            component_stats[key]['total'] += 1

        # 识别瓶颈
        for component, stats in component_stats.items():
            if stats['durations']:
                avg_duration = sum(stats['durations']) / len(stats['durations'])
                max_duration = max(stats['durations'])
                error_rate = stats['errors'] / stats['total'] * 100

                # 慢操作瓶颈
                if avg_duration > 1000:  # 平均超过1秒
                    bottlenecks.append({
                        'type': 'slow_operation',
                        'component': component,
                        'severity': 'high',
                        'avg_duration_ms': round(avg_duration, 2),
                        'max_duration_ms': round(max_duration, 2),
                        'recommendation': f'优化 {component} 的性能，考虑异步处理或缓存'
                    })

                # 高错误率瓶颈
                if error_rate > 10:  # 错误率超过10%
                    bottlenecks.append({
                        'type': 'high_error_rate',
                        'component': component,
                        'severity': 'high',
                        'error_rate_percent': round(error_rate, 2),
                        'recommendation': f'调查 {component} 的错误原因并修复'
                    })

        return bottlenecks

    def generate_optimization_plan(self) -> Dict[str, Any]:
        """生成优化计划"""
        bottlenecks = self.analyze_bottlenecks()

        # 系统健康分析
        health_report = self.monitor.get_system_health_report()

        # 生成优化建议
        optimizations = []

        # 基于瓶颈的优化
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'slow_operation':
                optimizations.append({
                    'priority': 'high',
                    'category': 'performance',
                    'component': bottleneck['component'],
                    'title': f'优化 {bottleneck["component"]} 响应时间',
                    'description': f'平均响应时间 {bottleneck["avg_duration_ms"]}ms 过高',
                    'actions': [
                        '实现异步处理',
                        '添加缓存层',
                        '优化数据库查询',
                        '使用连接池',
                        '实现请求合并'
                    ]
                })
            elif bottleneck['type'] == 'high_error_rate':
                optimizations.append({
                    'priority': 'high',
                    'category': 'reliability',
                    'component': bottleneck['component'],
                    'title': f'修复 {bottleneck["component"]} 错误',
                    'description': f'错误率 {bottleneck["error_rate_percent"]}% 过高',
                    'actions': [
                        '添加错误重试机制',
                        '改进错误处理',
                        '添加健康检查',
                        '实现熔断器模式',
                        '增加监控和告警'
                    ]
                })

        # 基于系统健康的优化
        if health_report.get('health_status') in ['warning', 'critical']:
            optimizations.append({
                'priority': 'medium',
                'category': 'system',
                'component': 'system',
                'title': '优化系统资源使用',
                'description': f'系统健康状态: {health_report.get("health_status", "unknown")}',
                'actions': [
                    '增加系统内存',
                    '优化CPU密集型任务',
                    '清理磁盘空间',
                    '减少并发连接数',
                    '实现资源限制'
                ]
            })

        # 架构层面的优化建议
        architecture_optimizations = [
            {
                'priority': 'medium',
                'category': 'architecture',
                'component': 'all',
                'title': '实现统一的监控接口',
                'description': '所有组件应该实现统一的性能监控接口',
                'actions': [
                    '继承BasePerformanceMonitorable接口',
                    '实现标准性能指标收集',
                    '添加组件健康检查',
                    '实现性能阈值告警'
                ]
            },
            {
                'priority': 'low',
                'category': 'caching',
                'component': 'cache',
                'title': '优化缓存策略',
                'description': '基于使用模式优化缓存失效和预加载策略',
                'actions': [
                    '实现智能缓存预加载',
                    '优化缓存键策略',
                    '添加缓存性能监控',
                    '实现分布式缓存',
                    '添加缓存压缩'
                ]
            },
            {
                'priority': 'low',
                'category': 'database',
                'component': 'database',
                'title': '数据库性能优化',
                'description': '优化数据库连接和查询性能',
                'actions': [
                    '实现连接池',
                    '添加查询缓存',
                    '优化索引策略',
                    '实现读写分离',
                    '添加查询性能监控'
                ]
            }
        ]

        optimizations.extend(architecture_optimizations)

        return {
            'bottlenecks': bottlenecks,
            'health_report': health_report,
            'optimizations': optimizations,
            'implementation_priority': sorted(optimizations, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
        }


class PerformanceMonitoringSystem:
    """性能监控系统"""

    def __init__(self):
        self.monitor = UnifiedPerformanceMonitor()
        self.optimizer = ArchitecturePerformanceOptimizer(self.monitor)

    def start_monitoring(self):
        """启动监控系统"""
        self.monitor.start_system_monitoring()
        print("✅ 性能监控系统已启动")

    def stop_monitoring(self):
        """停止监控系统"""
        self.monitor.stop_system_monitoring()
        print("✅ 性能监控系统已停止")

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板数据"""
        return {
            'system_health': self.monitor.get_system_health_report(),
            'performance_summary': self.monitor.get_performance_summary(),
            'bottlenecks': self.optimizer.analyze_bottlenecks(),
            'optimization_plan': self.optimizer.generate_optimization_plan()
        }

    def export_monitoring_data(self, filename: str = 'performance_monitoring_data.json'):
        """导出监控数据"""
        data = {
            'export_time': datetime.now().isoformat(),
            'system_metrics': [
                {
                    'timestamp': datetime.fromtimestamp(m.timestamp).isoformat(),
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'memory_used_mb': m.memory_used_mb,
                    'disk_usage_percent': m.disk_usage_percent,
                    'network_connections': m.network_connections,
                    'thread_count': m.thread_count,
                    'process_count': m.process_count
                }
                for m in self.monitor.system_metrics_history[-100:]  # 最近100个数据点
            ],
            'performance_metrics': [
                {
                    'component_name': m.component_name,
                    'operation_name': m.operation_name,
                    'start_time': datetime.fromtimestamp(m.start_time).isoformat(),
                    'end_time': datetime.fromtimestamp(m.end_time).isoformat() if m.end_time else None,
                    'duration_ms': m.duration_ms,
                    'memory_usage_mb': m.memory_usage_mb,
                    'success': m.success,
                    'error_message': m.error_message,
                    'metadata': m.metadata
                }
                for m in self.monitor.metrics_history[-1000:]  # 最近1000个操作
            ]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ 监控数据已导出到: {filename}")


def create_monitoring_integration():
    """创建监控集成代码"""
    integration_code = '''"""
基础设施层性能监控集成

将性能监控集成到现有组件中
"""

from phase3_performance_monitoring import UnifiedPerformanceMonitor
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class IPerformanceMonitorable(ABC):
    """性能监控接口"""

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        pass

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        pass


class BasePerformanceMonitorable(IPerformanceMonitorable):
    """基础性能监控类"""

    def __init__(self, component_name: str):
        self._component_name = component_name
        self._performance_monitor = UnifiedPerformanceMonitor()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self._performance_monitor.get_performance_summary(self._component_name)

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        # 实现基本的健康检查逻辑
        try:
            # 检查组件是否能正常工作
            self._perform_health_check()
            return {
                "component": self._component_name,
                "status": "healthy",
                "timestamp": "2025-01-21T12:00:00Z"
            }
        except Exception as e:
            return {
                "component": self._component_name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2025-01-21T12:00:00Z"
            }

    def _perform_health_check(self):
        """执行健康检查"""
        # 子类应该重写此方法
        pass

    def measure_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """操作性能测量装饰器"""
        return self._performance_monitor.measure_performance(
            self._component_name,
            operation_name,
            metadata
        )


# 全局监控实例
_global_monitor = UnifiedPerformanceMonitor()

def get_global_monitor() -> UnifiedPerformanceMonitor:
    """获取全局监控实例"""
    return _global_monitor

def start_global_monitoring():
    """启动全局监控"""
    _global_monitor.start_system_monitoring()

def stop_global_monitoring():
    """停止全局监控"""
    _global_monitor.stop_system_monitoring()

def measure_performance(component_name: str, operation_name: str):
    """性能测量装饰器"""
    return _global_monitor.measure_performance(component_name, operation_name)
'''

    with open('performance_monitoring_integration.py', 'w', encoding='utf-8') as f:
        f.write(integration_code)

    print("✅ 性能监控集成代码已生成: performance_monitoring_integration.py")


def main():
    """主函数"""
    print('🚀 开始Phase 3.3: 性能和监控优化')
    print('=' * 50)

    # 创建监控系统
    monitoring_system = PerformanceMonitoringSystem()

    # 启动监控
    monitoring_system.start_monitoring()

    try:
        # 等待一些监控数据收集
        print('📊 收集监控数据中...')
        import time
        time.sleep(10)  # 收集10秒的数据

        # 获取监控仪表板
        dashboard = monitoring_system.get_monitoring_dashboard()

        # 导出监控数据
        monitoring_system.export_monitoring_data()

        # 创建监控集成
        create_monitoring_integration()

        # 生成优化报告
        optimization_plan = dashboard['optimization_plan']

        with open('performance_optimization_plan.json', 'w', encoding='utf-8') as f:
            json.dump(optimization_plan, f, indent=2, ensure_ascii=False)

        print('✅ 性能优化计划已生成: performance_optimization_plan.json')

        # 输出关键指标
        health = dashboard['system_health']
        print(f'\\n📊 系统健康状态: {health.get("health_status", "unknown").upper()}')
        print(f'   总体评分: {health.get("overall_score", 0)}/100')
        print(f'   CPU使用率: {health.get("current_metrics", {}).get("cpu_percent", 0):.1f}%')
        print(f'   内存使用率: {health.get("current_metrics", {}).get("memory_percent", 0):.1f}%')

        bottlenecks = dashboard['bottlenecks']
        if bottlenecks:
            print(f'\\n⚠️ 发现 {len(bottlenecks)} 个性能瓶颈')
            for bottleneck in bottlenecks[:3]:  # 显示前3个
                print(
                    f'   - {bottleneck["component"]}: {bottleneck["type"]} (严重程度: {bottleneck["severity"]})')

        optimizations = optimization_plan.get('optimizations', [])
        if optimizations:
            print(f'\\n💡 建议优化项目: {len(optimizations)} 个')
            high_priority = [opt for opt in optimizations if opt['priority'] == 'high']
            print(f'   高优先级: {len(high_priority)} 个')

    finally:
        # 停止监控
        monitoring_system.stop_monitoring()

    print('\\n✅ Phase 3.3 性能和监控优化完成！')
    print('生成的文件:')
    print('  - performance_monitoring_data.json')
    print('  - performance_optimization_plan.json')
    print('  - performance_monitoring_integration.py')


if __name__ == "__main__":
    main()
