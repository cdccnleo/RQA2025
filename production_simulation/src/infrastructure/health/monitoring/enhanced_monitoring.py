"""
enhanced_monitoring 模块

提供 enhanced_monitoring 相关功能和接口。
"""

import logging

import threading
import time

from .health_checker import HealthChecker
from .performance_monitor import PerformanceMonitor
from .system_metrics_collector import SystemMetricsCollector
from datetime import datetime
from typing import Dict, Any
"""
增强监控系统 - 协调器模块

提供统一的监控系统接口，协调各个监控组件的工作。
"""

logger = logging.getLogger(__name__)


class EnhancedMonitoringSystem:
    """
    增强监控系统

    作为监控系统的统一入口，协调各个监控组件（指标收集、健康检查、性能监控）的工作。
    提供统一的监控接口和系统状态管理。
    """

    def __init__(self):
        """初始化增强监控系统"""

        self.metrics_collector = SystemMetricsCollector()
        self.health_checker = HealthChecker(self.metrics_collector)
        self.performance_monitor = PerformanceMonitor()

        self.is_monitoring = False
        self.monitoring_thread = None
        self._stop_event = threading.Event()
        self._running = True

        # 监控配置
        self.config = {
            'metrics_collection_interval': 1.0,  # 1秒
            'health_check_interval': 30.0,      # 30秒
            'alert_check_interval': 60.0,       # 60秒
            'enable_memory_tracing': False
        }

    def start_monitoring(self):
        """启动监控系统"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self._stop_event.clear()

        # 启动内存追踪（如果启用）
        if self.config['enable_memory_tracing']:
            self.performance_monitor.start_memory_tracing()

        # 启动指标收集
        self.metrics_collector.start_collection(self.config['metrics_collection_interval'])

        # 启动监控线程
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True
        )
        self.monitoring_thread.start()

        logger.info("增强监控系统已启动")

    def stop_monitoring(self):
        """停止监控系统"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self._stop_event.set()

        # 停止指标收集
        self.metrics_collector.stop_collection()

        # 停止内存追踪
        if self.config['enable_memory_tracing']:
            self.performance_monitor.stop_memory_tracing()

        # 等待监控线程结束
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        logger.info("增强监控系统已停止")

    def _monitoring_worker(self):
        """监控工作线程"""
        last_health_check = 0
        last_alert_check = 0

        while self._running and not self._stop_event.is_set():
            try:
                current_time = time.time()

                # 定期健康检查
                if current_time - last_health_check >= self.config['health_check_interval']:
                    self._perform_health_check()
                    last_health_check = current_time

                # 定期告警检查
                if current_time - last_alert_check >= self.config['alert_check_interval']:
                    self._perform_alert_check()
                    last_alert_check = current_time

                time.sleep(1.0)  # 每秒检查一次

            except Exception as e:
                logger.error(f"监控工作线程异常: {e}")
                time.sleep(5.0)  # 异常后等待5秒

    def _perform_health_check(self):
        """执行健康检查"""
        try:
            results = self.health_checker.run_health_checks()

            # 如果整体状态不健康，记录性能告警
            if results.get('overall_status') != 'healthy':
                self.performance_monitor.add_performance_alert(
                    'health_check_failed',
                    f"系统健康检查失败: {results.get('overall_status')}",
                    'warning',
                    results
                )

        except Exception as e:
            logger.error(f"健康检查执行失败: {e}")

    def _perform_alert_check(self):
        """执行告警检查"""
        try:
            # 这里可以添加额外的告警检查逻辑
            # 例如：检查性能指标是否超出阈值
            pass
        except Exception as e:
            logger.error(f"告警检查执行失败: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统整体状态

        Returns:
            系统状态信息字典
        """
        try:
            # 获取各项指标
            metrics_summary = self.metrics_collector.get_system_summary()
            health_results = self.health_checker.run_health_checks()
            performance_summary = self.performance_monitor.get_performance_summary()

            # 综合状态评估
            overall_status = self._calculate_overall_status(
                metrics_summary, health_results, performance_summary
            )

            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': overall_status,
                'metrics': metrics_summary,
                'health': health_results,
                'performance': performance_summary,
                'is_monitoring': self.is_monitoring
            }

        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e),
                'is_monitoring': self.is_monitoring
            }

    def _calculate_overall_status(self, metrics: Dict, health: Dict,
                                  performance: Dict) -> str:
        """计算系统整体状态"""
        # 状态优先级：error > critical > warning > healthy
        status_priority = {'healthy': 0, 'warning': 1, 'critical': 2, 'error': 3}

        # 检查各项状态
        statuses = []

        # 指标收集状态
        if metrics.get('status') == 'inactive':
            statuses.append('warning')

        # 健康检查状态
        health_status = health.get('overall_status', 'unknown')
        if health_status in status_priority:
            statuses.append(health_status)

        # 性能监控状态
        if performance.get('memory_tracing_active') is False and self.config['enable_memory_tracing']:
            statuses.append('warning')

        # 计算最高优先级状态
        if not statuses:
            return 'healthy'

        max_priority = max(status_priority.get(s, 0) for s in statuses)
        for status, priority in status_priority.items():
            if priority == max_priority:
                return status

        return 'unknown'

    def update_config(self, new_config: Dict[str, Any]):
        """
        更新监控配置

        Args:
            new_config: 新的配置字典
        """
        # 验证配置
        valid_keys = {
            'metrics_collection_interval',
            'health_check_interval',
            'alert_check_interval',
            'enable_memory_tracing'
        }

        for key, value in new_config.items():
            if key in valid_keys:
                if key.endswith('_interval') and (not isinstance(value, (int, float)) or value <= 0):
                    logger.warning(f"忽略无效的间隔配置: {key}={value}")
                    continue
                elif key == 'enable_memory_tracing' and not isinstance(value, bool):
                    logger.warning(f"忽略无效的内存追踪配置: {key}={value}")
                    continue

                self.config[key] = value
                logger.info(f"更新配置: {key} = {value}")

                # 如果更新了内存追踪设置，需要重新启动
                if key == 'enable_memory_tracing':
                    if value and not self.performance_monitor.tracemalloc_started:
                        self.performance_monitor.start_memory_tracing()
                    elif not value and self.performance_monitor.tracemalloc_started:
                        self.performance_monitor.stop_memory_tracing()

    # 委托方法 - 提供便捷访问
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        return self.metrics_collector.get_system_summary()

    def run_health_checks(self) -> Dict[str, Any]:
        """运行健康检查"""
        return self.health_checker.run_health_checks()

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return self.performance_monitor.get_performance_summary()

    def take_memory_snapshot(self) -> Dict[str, Any]:
        """获取内存快照"""
        return self.performance_monitor.take_memory_snapshot()


def get_enhanced_monitoring() -> EnhancedMonitoringSystem:
    """
    获取增强监控系统实例

    Returns:
        增强监控系统实例
    """
    if not hasattr(get_enhanced_monitoring, '_instance'):
        get_enhanced_monitoring._instance = EnhancedMonitoringSystem()
    return get_enhanced_monitoring._instance


def start_system_monitoring():
    """启动系统监控"""
    monitoring = get_enhanced_monitoring()
    monitoring.start_monitoring()


def stop_system_monitoring():
    """停止系统监控"""
    monitoring = get_enhanced_monitoring()
    monitoring.stop_monitoring()


def get_system_status() -> Dict[str, Any]:
    """
    获取系统状态

    Returns:
        系统状态信息
    """
    monitoring = get_enhanced_monitoring()
    return monitoring.get_system_status()


def increment_performance_counter(counter_name: str, value: int = 1):
    """
    增加性能计数器

    Args:
        counter_name: 计数器名称
        value: 增加值
    """
    monitoring = get_enhanced_monitoring()
    monitoring.metrics_collector.increment_counter(counter_name, value)

# 模块级健康检查函数


def check_health() -> Dict[str, Any]:
    """执行整体健康检查"""
    try:
        logger.info("开始增强监控模块健康检查")

        health_checks = {
            "monitoring_system": check_monitoring_system(),
            "coordinator_functions": check_coordinator_functions()
        }

        overall_healthy = all(check.get("healthy", False) for check in health_checks.values())
        result = {
            "healthy": overall_healthy,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "service": "enhanced_monitoring",
            "checks": health_checks
        }

        if not overall_healthy:
            logger.warning("增强监控模块健康检查发现问题")
            result["issues"] = [name for name, check in health_checks.items()
                                if not check.get("healthy", False)]

        logger.info(f"增强监控模块健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result
    except Exception as e:
        logger.error(f"增强监控模块健康检查失败: {str(e)}", exc_info=True)
        return {
            "healthy": False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "service": "enhanced_monitoring",
            "error": str(e)
        }


def check_monitoring_system() -> Dict[str, Any]:
    """检查监控系统"""
    try:
        # 检查监控协调器类
        coordinator_exists = 'EnhancedMonitoringCoordinator' in globals()

        if not coordinator_exists:
            return {"healthy": False, "error": "EnhancedMonitoringCoordinator class not found"}

        # 检查工厂函数
        factory_exists = 'get_enhanced_monitoring' in globals()

        return {
            "healthy": coordinator_exists and factory_exists,
            "coordinator_exists": coordinator_exists,
            "factory_exists": factory_exists
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def check_coordinator_functions() -> Dict[str, Any]:
    """检查协调器功能"""
    try:
        # 检查协调器相关函数
        functions_exist = all(func in globals() for func in [
            'increment_performance_counter', 'record_health_metric', 'get_monitoring_status'
        ])

        return {
            "healthy": functions_exist,
            "functions_exist": functions_exist
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def health_status() -> Dict[str, Any]:
    """获取健康状态摘要"""
    try:
        health_check = check_health()
        return {
            "status": "healthy" if health_check["healthy"] else "unhealthy",
            "service": "enhanced_monitoring",
            "health_check": health_check,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def health_summary() -> Dict[str, Any]:
    """获取健康摘要报告"""
    try:
        health_check = check_health()
        return {
            "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
            "enhanced_monitoring_module_info": {
                "service_name": "enhanced_monitoring",
                "purpose": "增强监控系统",
                "operational": health_check["healthy"]
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    except Exception as e:
        return {"overall_health": "error", "error": str(e)}


def monitor_enhanced_monitoring() -> Dict[str, Any]:
    """监控增强监控状态"""
    try:
        health_check = check_health()
        monitoring_efficiency = 1.0 if health_check["healthy"] else 0.0
        return {
            "healthy": health_check["healthy"],
            "monitoring_metrics": {
                "service_name": "enhanced_monitoring",
                "monitoring_efficiency": monitoring_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            }
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def validate_enhanced_monitoring() -> Dict[str, Any]:
    """验证增强监控"""
    try:
        validation_results = {
            "monitoring_validation": check_monitoring_system(),
            "coordinator_validation": check_coordinator_functions()
        }
        overall_valid = all(result.get("valid", False) for result in validation_results.values())
        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
