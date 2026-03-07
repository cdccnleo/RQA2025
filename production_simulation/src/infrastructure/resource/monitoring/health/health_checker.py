"""
health_checker 模块

提供 health_checker 相关功能和接口。
"""


# 延迟导入HealthCheck类 (用于类型注解)
import psutil
import socket

from .health_check_executor import HealthCheckExecutor
from .health_check_manager import HealthCheckManager
from .health_check_scheduler import HealthCheckScheduler
from .health_status_reporter import HealthStatusReporter
from ....core.component_registry import InfrastructureComponentRegistry
from ...core.shared_interfaces import StandardLogger, ILogger
from .health_check_manager import HealthCheck
from .health_status import HealthStatus
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
"""
RQA2025 系统健康检查组件 - 重构版本

提供全面的系统健康检查和状态监控功能

重构说明：
- 将原来的单一HealthChecker类拆分为多个职责单一的类
- HealthCheckManager: 管理健康检查项的注册和配置
- HealthCheckExecutor: 执行具体的健康检查
- HealthCheckScheduler: 处理检查调度和周期执行
- HealthStatusReporter: 处理健康状态报告和查询
"""


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    enable_system_checks: bool = True
    enable_service_checks: bool = True
    enable_dependency_checks: bool = True
    check_interval: int = 60  # 检查间隔(秒)
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_warning': 80.0,
        'cpu_critical': 95.0,
        'memory_warning': 85.0,
        'memory_critical': 95.0,
        'disk_warning': 90.0,
        'disk_critical': 95.0
    })


class HealthChecker:
    """
    系统健康检查器 (重构后的外观类)

    协调各个组件提供统一的健康检查接口
    """

    def __init__(self, config: Optional[HealthCheckConfig] = None):
        self.config = config or HealthCheckConfig()

        # 使用组件注册表进行依赖管理 (Phase 6.1组织架构重构)
        self.registry = InfrastructureComponentRegistry()

        # 注册组件工厂函数 (延迟加载)
        self._register_component_factories()

        # 延迟初始化组件
        self._components = {}

        self.logger: ILogger = StandardLogger(f"{self.__class__.__name__}")

        # 设置默认检查
        self._setup_default_checks()

        self.logger.log_info("健康检查器初始化完成 (Phase 6.1重构版本)")

    def _register_component_factories(self):
        """注册组件工厂函数 - 实现延迟加载"""
        # 注册健康检查管理器
        self.registry.register_component('manager',
                                         lambda: self._create_health_check_manager())

        # 注册健康检查执行器
        self.registry.register_component('executor',
                                         lambda: self._create_health_check_executor())

        # 注册健康检查调度器
        self.registry.register_component('scheduler',
                                         lambda: self._create_health_check_scheduler())

        # 注册健康状态报告器
        self.registry.register_component('reporter',
                                         lambda: self._create_health_status_reporter())

    def _create_health_check_manager(self):
        """创建健康检查管理器"""
        try:
            return HealthCheckManager()
        except ImportError:
            self.logger.log_error("HealthCheckManager组件不可用")
            return None

    def _create_health_check_executor(self):
        """创建健康检查执行器"""
        try:
            return HealthCheckExecutor()
        except ImportError:
            self.logger.log_error("HealthCheckExecutor组件不可用")
            return None

    def _create_health_check_scheduler(self):
        """创建健康检查调度器"""
        try:
            executor = self.registry.get_component('executor')
            return HealthCheckScheduler(executor) if executor else None
        except ImportError:
            self.logger.log_error("HealthCheckScheduler组件不可用")
            return None

    def _create_health_status_reporter(self):
        """创建健康状态报告器"""
        try:
            return HealthStatusReporter()
        except ImportError:
            self.logger.log_error("HealthStatusReporter组件不可用")
            return None

    # 属性访问器 - 延迟加载组件
    @property
    def manager(self):
        """延迟加载健康检查管理器"""
        if 'manager' not in self._components:
            self._components['manager'] = self.registry.get_component('manager')
        return self._components['manager']

    @property
    def executor(self):
        """延迟加载健康检查执行器"""
        if 'executor' not in self._components:
            self._components['executor'] = self.registry.get_component('executor')
        return self._components['executor']

    @property
    def scheduler(self):
        """延迟加载健康检查调度器"""
        if 'scheduler' not in self._components:
            self._components['scheduler'] = self.registry.get_component('scheduler')
        return self._components['scheduler']

    @property
    def reporter(self):
        """延迟加载健康状态报告器"""
        if 'reporter' not in self._components:
            self._components['reporter'] = self.registry.get_component('reporter')
        return self._components['reporter']

    def add_health_check(self, check: HealthCheck) -> None:
        """添加健康检查项"""
        self.manager.register_check(check)

    def remove_health_check(self, name: str) -> bool:
        """移除健康检查项"""
        return self.manager.unregister_check(name)

    def start_checking(self) -> None:
        """启动健康检查"""
        self.scheduler.start_scheduler()

    def stop_checking(self) -> None:
        """停止健康检查"""
        self.scheduler.stop_scheduler()

    def run_health_checks(self) -> List[Dict[str, Any]]:
        """运行健康检查"""
        checks = self.manager.get_enabled_checks()
        return self.executor.execute_checks(checks)

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return self.reporter.get_current_status()

    def get_check_results(self, check_name: Optional[str] = None) -> Dict[str, Any]:
        """获取检查结果"""
        return self.reporter.get_current_status(check_name)

    def get_health_report(self, hours: int = 1) -> Dict[str, Any]:
        """获取健康报告"""
        return self.reporter.get_health_report(hours)

    def _setup_default_checks(self) -> None:
        """设置默认健康检查项"""
        # 系统资源检查
        if self.config.enable_system_checks:
            self.add_health_check(HealthCheck(
                name="cpu_usage",
                description="CPU使用率检查",
                check_function=self._check_cpu_usage,
                tags=["system", "cpu"]
            ))

            self.add_health_check(HealthCheck(
                name="memory_usage",
                description="内存使用率检查",
                check_function=self._check_memory_usage,
                tags=["system", "memory"]
            ))

            self.add_health_check(HealthCheck(
                name="disk_usage",
                description="磁盘使用率检查",
                check_function=self._check_disk_usage,
                tags=["system", "disk"]
            ))

            self.add_health_check(HealthCheck(
                name="network_connectivity",
                description="网络连接检查",
                check_function=self._check_network_connectivity,
                tags=["system", "network"]
            ))

    def _check_cpu_usage(self) -> Dict[str, Any]:
        """检查CPU使用率"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            status = HealthStatus.HEALTHY.value

            if cpu_percent >= self.config.alert_thresholds['cpu_critical']:
                status = HealthStatus.CRITICAL.value
            elif cpu_percent >= self.config.alert_thresholds['cpu_warning']:
                status = HealthStatus.WARNING.value

            return {
                'metric': 'cpu_usage',
                'value': cpu_percent,
                'unit': 'percent',
                'status': status,
                'thresholds': {
                    'warning': self.config.alert_thresholds['cpu_warning'],
                    'critical': self.config.alert_thresholds['cpu_critical']
                }
            }
        except Exception as e:
            return {
                'metric': 'cpu_usage',
                'value': None,
                'unit': 'percent',
                'status': HealthStatus.UNKNOWN.value,
                'error': str(e)
            }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """检查内存使用率"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            status = HealthStatus.HEALTHY.value

            if memory_percent >= self.config.alert_thresholds['memory_critical']:
                status = HealthStatus.CRITICAL.value
            elif memory_percent >= self.config.alert_thresholds['memory_warning']:
                status = HealthStatus.WARNING.value

            return {
                'metric': 'memory_usage',
                'value': memory_percent,
                'unit': 'percent',
                'status': status,
                'details': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used
                },
                'thresholds': {
                    'warning': self.config.alert_thresholds['memory_warning'],
                    'critical': self.config.alert_thresholds['memory_critical']
                }
            }
        except Exception as e:
            return {
                'metric': 'memory_usage',
                'value': None,
                'unit': 'percent',
                'status': HealthStatus.UNKNOWN.value,
                'error': str(e)
            }

    def _check_disk_usage(self) -> Dict[str, Any]:
        """检查磁盘使用率"""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            status = HealthStatus.HEALTHY.value

            if disk_percent >= self.config.alert_thresholds['disk_critical']:
                status = HealthStatus.CRITICAL.value
            elif disk_percent >= self.config.alert_thresholds['disk_warning']:
                status = HealthStatus.WARNING.value

            return {
                'metric': 'disk_usage',
                'value': disk_percent,
                'unit': 'percent',
                'status': status,
                'details': {
                    'total': disk.total,
                    'free': disk.free,
                    'used': disk.used
                },
                'thresholds': {
                    'warning': self.config.alert_thresholds['disk_warning'],
                    'critical': self.config.alert_thresholds['disk_critical']
                }
            }
        except Exception as e:
            return {
                'metric': 'disk_usage',
                'value': None,
                'unit': 'percent',
                'status': HealthStatus.UNKNOWN.value,
                'error': str(e)
            }

    def _check_network_connectivity(self) -> Dict[str, Any]:
        """检查网络连接"""
        try:
            # 尝试连接到公共DNS服务器
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return {
                'metric': 'network_connectivity',
                'value': True,
                'status': HealthStatus.HEALTHY.value,
                'message': '网络连接正常'
            }
        except Exception as e:
            return {
                'metric': 'network_connectivity',
                'value': False,
                'status': HealthStatus.CRITICAL.value,
                'error': f'网络连接失败: {str(e)}'
            }


def start_health_checking(config: Optional[HealthCheckConfig] = None) -> HealthChecker:
    """启动健康检查"""
    checker = HealthChecker(config)
    checker.start_checking()
    return checker
