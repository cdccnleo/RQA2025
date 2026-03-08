"""
database_health_monitor 模块

提供 database_health_monitor 相关功能和接口。
"""

import logging

# 配置日志记录器
logger = logging.getLogger(__name__)

# -*- coding: utf-8 -*-
# 统一基础设施接口
import asyncio
import threading
import time

# 条件导入psutil
try:
    import psutil
except ImportError:
    psutil = None

from ..core.interfaces import IUnifiedInfrastructureInterface
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Coroutine

# 导入监控和错误处理组件
try:
    from ...monitoring.application_monitor_core import ApplicationMonitor
except ImportError:
    try:
        from ...monitoring.application_monitor import ApplicationMonitor
    except ImportError:
        # 降级定义
        class ApplicationMonitor:
            def __init__(self):
                pass

try:
    from src.infrastructure.logging.core.error_handler import ErrorHandler
except ImportError:
    try:
        from src.error.handlers.error_handler import ErrorHandler
    except ImportError:
        # 降级定义
        class ErrorHandler:
            def __init__(self):
                pass
"""
基础设施层 - 日志系统组件

database_health_monitor 模块

日志系统相关的文件
提供日志系统相关的功能实现。
"""

# !/usr/bin/env python3
# from infrastructure.error.unified_error_handler import UnifiedErrorHandler as ErrorHandler  # 暂时注释，模块不存在
# from infrastructure.monitoring.application_monitor import ApplicationMonitor  # 暂时注释，模块不存在
"""
数据库健康监控器 - 实现数据库连接状态监控和性能指标收集
优先级：3 - 建立数据库健康监控
"""

# from .unified_data_manager import UnifiedDataManager  # 已删除

# 导入健康状态评估器
try:
    from ..components.health_status_evaluator import HealthStatusEvaluator, ThresholdConfig, MetricValue
except ImportError:
    logger.warning("无法导入健康状态评估器")

# 常量定义 - 清理魔法数字
# 检查间隔和超时配置
DEFAULT_CHECK_INTERVAL = 60  # 秒
ERROR_RETRY_DELAY = 10  # 秒

# 警告阈值配置
WARNING_CONNECTION_COUNT = 80
WARNING_ERROR_RATE = 0.05  # 5%
WARNING_MEMORY_USAGE = 0.8  # 80%
WARNING_CPU_USAGE = 0.7  # 70%
WARNING_DISK_USAGE = 0.85  # 85%

# 严重阈值配置
CRITICAL_CONNECTION_COUNT = 95
CRITICAL_ERROR_RATE = 0.1  # 10%
CRITICAL_MEMORY_USAGE = 0.9  # 90%
CRITICAL_CPU_USAGE = 0.85  # 85%
CRITICAL_DISK_USAGE = 0.95  # 95%

# 历史记录限制
MAX_HISTORY_LENGTH = 100

# 查询时间阈值常量
WARNING_AVG_QUERY_TIME = 2.0  # 2秒
CRITICAL_AVG_QUERY_TIME = 5.0  # 5秒

# 模拟数据配置
SIMULATION_CONNECTION_COUNT = 50
SIMULATION_ACTIVE_CONNECTIONS = 25
SIMULATION_QUERY_COUNT = 1000

SIMULATION_LOW_CONNECTION_COUNT = 30
SIMULATION_LOW_ACTIVE_CONNECTIONS = 15
SIMULATION_LOW_QUERY_COUNT = 500

SIMULATION_HIGH_CONNECTION_COUNT = 100
SIMULATION_HIGH_ACTIVE_CONNECTIONS = 80
SIMULATION_HIGH_QUERY_COUNT = 2000


class HealthStatus(Enum):

    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class DatabaseMetrics:

    """数据库性能指标"""
    connection_count: int
    active_connections: int
    query_count: int
    avg_query_time: float
    error_count: int
    memory_usage: float
    cpu_usage: float
    disk_usage: float
    timestamp: datetime


@dataclass
class HealthCheckResult:

    """健康检查结果"""
    status: HealthStatus
    component: str
    metrics: DatabaseMetrics
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime

    def __getitem__(self, key):
        """支持字典式访问"""
        if key == 'status':
            return self.status.value
        elif key == 'component':
            return self.component
        elif key == 'timestamp':
            return self.timestamp.isoformat()
        elif key == 'issues':
            return self.issues
        elif key == 'recommendations':
            return self.recommendations
        elif key == 'error':
            return 'error' if self.status != HealthStatus.HEALTHY else None
        elif key == 'connection_count':
            return self.metrics.connection_count
        elif key == 'active_connections':
            return self.metrics.active_connections
        elif key == 'query_count':
            return self.metrics.query_count
        elif key == 'avg_query_time':
            return self.metrics.avg_query_time
        elif key == 'error_count':
            return self.metrics.error_count
        elif key == 'memory_usage':
            return self.metrics.memory_usage
        elif key == 'cpu_usage':
            return self.metrics.cpu_usage
        elif key == 'disk_usage':
            return self.metrics.disk_usage
        else:
            raise KeyError(f"HealthCheckResult has no key '{key}'")

    def __contains__(self, key):
        """支持in操作符"""
        return key in ['status', 'component', 'timestamp', 'issues', 'recommendations',
                       'error', 'connection_count', 'active_connections', 'query_count',
                       'avg_query_time', 'error_count', 'memory_usage', 'cpu_usage', 'disk_usage']

    def get(self, key, default=None):
        """支持get方法"""
        try:
            return self[key]
        except KeyError:
            return default


class DatabaseHealthMonitor(IUnifiedInfrastructureInterface):

    """数据库健康监控器"""

    def _initialize_monitor_config(self) -> Dict[str, Any]:
        """初始化监控配置"""
        return {
            'check_interval': DEFAULT_CHECK_INTERVAL,
            'warning_thresholds': {
                'connection_count': WARNING_CONNECTION_COUNT,
                'avg_query_time': WARNING_AVG_QUERY_TIME,
                'error_rate': WARNING_ERROR_RATE,
                'memory_usage': WARNING_MEMORY_USAGE,
                'cpu_usage': WARNING_CPU_USAGE,
                'disk_usage': WARNING_DISK_USAGE
            },
            'critical_thresholds': {
                'connection_count': CRITICAL_CONNECTION_COUNT,
                'avg_query_time': CRITICAL_AVG_QUERY_TIME,
                'error_rate': CRITICAL_ERROR_RATE,
                'memory_usage': CRITICAL_MEMORY_USAGE,
                'cpu_usage': CRITICAL_CPU_USAGE,
                'disk_usage': CRITICAL_DISK_USAGE
            }
        }

    def _initialize_metrics_history(self) -> Dict[str, List[DatabaseMetrics]]:
        """初始化性能指标历史"""
        return {
            'postgresql': [],
            'influxdb': [],
            'redis': []
        }

    def _initialize_health_history(self) -> Dict[str, List[HealthCheckResult]]:
        """初始化健康状态历史"""
        return {
            'postgresql': [],
            'influxdb': [],
            'redis': []
        }

    def _initialize_monitoring_state(self) -> Dict[str, Any]:
        """初始化监控状态"""
        return {
            'monitoring': False,
            'monitor_thread': None
        }

    def __init__(self, data_manager: Any, monitor: Optional[Any] = None):
        """
        初始化数据库健康监控器

        Args:
            data_manager: 统一数据管理器
            monitor: 应用监控器
        """
        # 初始化核心组件
        self.data_manager = data_manager
        if monitor is not None:
            self.monitor = monitor
        else:
            self.monitor = None
        self.error_handler = ErrorHandler()

        # 初始化配置和状态
        self.monitor_config = self._initialize_monitor_config()
        self.metrics_history = self._initialize_metrics_history()
        self.health_history = self._initialize_health_history()

        # 初始化监控状态
        monitor_state = self._initialize_monitoring_state()
        self.monitoring = monitor_state['monitoring']
        self.monitor_thread = monitor_state['monitor_thread']
        
        # 初始化停止事件和锁（用于线程同步）
        import threading
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # 初始化健康状态评估器
        try:
            self._init_health_evaluator()
        except Exception as e:
            logger.warning(f"初始化健康状态评估器失败: {e}")

        logger.info("数据库健康监控器初始化完成")
        logger.debug(f"监控配置: check_interval={self.monitor_config['check_interval']}秒")
        logger.debug(f"警告阈值: 连接数={self.monitor_config['warning_thresholds']['connection_count']}, "
                     f"查询时间={self.monitor_config['warning_thresholds']['avg_query_time']}秒")
        logger.debug(f"关键阈值: 连接数={self.monitor_config['critical_thresholds']['connection_count']}, "
                     f"查询时间={self.monitor_config['critical_thresholds']['avg_query_time']}秒")
    
    def _init_health_evaluator(self):
        """初始化健康状态评估器"""
        try:
            # 从配置中提取阈值配置
            thresholds = {}
            for metric_name in ['error_rate', 'memory_usage', 'cpu_usage', 'disk_usage']:
                thresholds[metric_name] = ThresholdConfig(
                    warning=self.monitor_config['warning_thresholds'].get(metric_name, 0.8),
                    critical=self.monitor_config['critical_thresholds'].get(metric_name, 0.9)
                )
            
            self.health_evaluator = HealthStatusEvaluator(thresholds)
            logger.debug("健康状态评估器初始化成功")
        except Exception as e:
            logger.error(f"初始化健康状态评估器失败: {e}")
            self.health_evaluator = None

    def start_monitoring(self):
        """启动监控"""
        logger.info("尝试启动数据库健康监控")

        if self.monitoring:
            logger.warning("数据库监控已在运行中，无法重复启动")
            return

        try:
            logger.debug("创建监控线程")
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="DatabaseHealthMonitor"
            )
            self.monitor_thread.start()

            logger.info("数据库健康监控启动成功")
            logger.debug(f"监控线程ID: {self.monitor_thread.ident}, 守护线程: {self.monitor_thread.daemon}")
        except Exception as e:
            logger.error(f"启动数据库监控失败: {e}", exc_info=True)
            self.monitoring = False
            self.monitor_thread = None
            raise

    def stop_monitoring(self):
        """停止监控"""
        logger.info("尝试停止数据库健康监控")

        if not self.monitoring:
            logger.info("数据库监控未运行，无需停止")
            return

        try:
            logger.debug("设置监控停止标志")
            self.monitoring = False

            if self.monitor_thread and self.monitor_thread.is_alive():
                logger.debug(f"等待监控线程结束，超时时间: 5秒")
                self.monitor_thread.join(timeout=5.0)
                if self.monitor_thread.is_alive():
                    logger.warning("监控线程在5秒内未正常结束")
                else:
                    logger.debug("监控线程已正常结束")
            else:
                logger.debug("监控线程不存在或已结束")

            logger.info("数据库健康监控停止成功")
        except Exception as e:
            logger.error(f"停止数据库监控时发生错误: {e}", exc_info=True)
            raise

    def _monitor_loop(self):
        """监控主循环"""
        logger.info("数据库健康监控循环开始")
        cycle_count = 0

        while self.monitoring:
            cycle_count += 1
            logger.debug(f"开始第 {cycle_count} 次监控周期")

            try:
                start_time = time.time()

                # 检查各数据库健康状态
                logger.debug("检查PostgreSQL健康状态")
                self._check_postgresql_health()

                logger.debug("检查InfluxDB健康状态")
                self._check_influxdb_health()

                logger.debug("检查Redis健康状态")
                self._check_redis_health()

                cycle_duration = time.time() - start_time
                logger.debug(f"第 {cycle_count} 次监控周期完成，耗时: {cycle_duration:.2f}秒")

                # 等待下次检查
                logger.debug(f"等待 {self.monitor_config['check_interval']} 秒后进行下次检查")
                time.sleep(self.monitor_config['check_interval'])

            except Exception as e:
                logger.error(f"监控循环第 {cycle_count} 次执行出错: {e}", exc_info=True)
                logger.info(f"监控循环错误后等待 {ERROR_RETRY_DELAY} 秒后重试")
                time.sleep(ERROR_RETRY_DELAY)  # 错误后等待

        logger.info(f"数据库健康监控循环结束，共执行 {cycle_count} 次周期")

    def _check_postgresql_health(self):
        """检查PostgreSQL健康状态"""
        try:
            # 获取性能指标
            metrics = self._get_postgresql_metrics()
            # 执行健康检查
            health_result = self._perform_health_check('postgresql', metrics)
            # 记录结果
            self._append_health_history('postgresql', health_result)
            # 监控记录
            self.monitor.record_metric(
                'postgresql_health',
                1 if health_result.status == HealthStatus.HEALTHY else 0,
                {'status': health_result.status.value}
            )

            return health_result
        except Exception as e:
            logger.error(f"PostgreSQL健康检查失败: {e}")
            return None

    def _check_influxdb_health(self):
        """检查InfluxDB健康状态"""
        try:
            # 获取性能指标
            metrics = self._get_influxdb_metrics()

            # 执行健康检查
            health_result = self._perform_health_check('influxdb', metrics)

            # 记录结果
            self._append_health_history('influxdb', health_result)

            # 监控记录
            self.monitor.record_metric(
                'influxdb_health',
                1 if health_result.status == HealthStatus.HEALTHY else 0,
                {'status': health_result.status.value}
            )

        except Exception as e:
            logger.error(f"InfluxDB健康检查失败: {e}")

    def _check_redis_health(self):
        """检查Redis健康状态"""
        try:
            # 获取性能指标
            metrics = self._get_redis_metrics()
            # 执行健康检查
            health_result = self._perform_health_check('redis', metrics)
            # 记录结果
            self._append_health_history('redis', health_result)
            # 监控记录
            self.monitor.record_metric(
                'redis_health',
                1 if health_result.status == HealthStatus.HEALTHY else 0,
                {'status': health_result.status.value}
            )

            return health_result
        except Exception as e:
            logger.error(f"Redis健康检查失败: {e}")
            return None

    def _get_postgresql_metrics(self) -> DatabaseMetrics:
        """获取PostgreSQL性能指标"""
        try:
            # 这里应该从PostgreSQL获取实际指标
            # 目前使用模拟数据
            return DatabaseMetrics(
                connection_count=SIMULATION_CONNECTION_COUNT,
                active_connections=SIMULATION_ACTIVE_CONNECTIONS,
                query_count=SIMULATION_QUERY_COUNT,
                avg_query_time=0.5,
                error_count=5,
                memory_usage=0.6,
                cpu_usage=0.4,
                disk_usage=0.7,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"获取PostgreSQL指标失败: {e}")
            return DatabaseMetrics(
                connection_count=0,
                active_connections=0,
                query_count=0,
                avg_query_time=0.0,
                error_count=0,
                memory_usage=0.0,
                cpu_usage=0.0,
                disk_usage=0.0,
                timestamp=datetime.now()
            )

    def _get_influxdb_metrics(self) -> DatabaseMetrics:
        """获取InfluxDB性能指标"""
        try:
            # 这里应该从InfluxDB获取实际指标
            # 目前使用模拟数据
            return DatabaseMetrics(
                connection_count=SIMULATION_LOW_CONNECTION_COUNT,
                active_connections=SIMULATION_LOW_ACTIVE_CONNECTIONS,
                query_count=SIMULATION_LOW_QUERY_COUNT,
                avg_query_time=0.3,
                error_count=2,
                memory_usage=0.5,
                cpu_usage=0.3,
                disk_usage=0.6,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"获取InfluxDB指标失败: {e}")
            return DatabaseMetrics(
                connection_count=0,
                active_connections=0,
                query_count=0,
                avg_query_time=0.0,
                error_count=0,
                memory_usage=0.0,
                cpu_usage=0.0,
                disk_usage=0.0,
                timestamp=datetime.now()
            )

    def _get_redis_metrics(self) -> DatabaseMetrics:
        """获取Redis性能指标"""
        try:
            # 这里应该从Redis获取实际指标
            # 目前使用模拟数据
            return DatabaseMetrics(
                connection_count=SIMULATION_HIGH_CONNECTION_COUNT,
                active_connections=SIMULATION_HIGH_ACTIVE_CONNECTIONS,
                query_count=SIMULATION_HIGH_QUERY_COUNT,
                avg_query_time=0.1,
                error_count=1,
                memory_usage=0.4,
                cpu_usage=0.2,
                disk_usage=0.3,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"获取Redis指标失败: {e}")
            return DatabaseMetrics(
                connection_count=0,
                active_connections=0,
                query_count=0,
                avg_query_time=0.0,
                error_count=0,
                memory_usage=0.0,
                cpu_usage=0.0,
                disk_usage=0.0,
                timestamp=datetime.now()
            )

    def _check_connection_count(self, metrics: DatabaseMetrics) -> tuple[List[str], List[str], HealthStatus]:
        """检查连接数"""
        issues = []
        recommendations = []
        status = HealthStatus.HEALTHY

        if metrics.connection_count > self.monitor_config['critical_thresholds']['connection_count']:
            issues.append(f"连接数过高: {metrics.connection_count}")
            recommendations.append("考虑增加连接池大小或优化连接管理")
            status = HealthStatus.CRITICAL
        elif metrics.connection_count > self.monitor_config['warning_thresholds']['connection_count']:
            issues.append(f"连接数较高: {metrics.connection_count}")
            recommendations.append("监控连接数趋势")
            status = HealthStatus.WARNING

        return issues, recommendations, status

    def _check_query_time(self, metrics: DatabaseMetrics) -> tuple[List[str], List[str], HealthStatus]:
        """检查查询时间"""
        issues = []
        recommendations = []
        status = HealthStatus.HEALTHY

        if metrics.avg_query_time > self.monitor_config['critical_thresholds']['avg_query_time']:
            issues.append(f"平均查询时间过长: {metrics.avg_query_time:.2f}s")
            recommendations.append("检查慢查询日志，优化索引")
            status = HealthStatus.CRITICAL
        elif metrics.avg_query_time > self.monitor_config['warning_thresholds']['avg_query_time']:
            issues.append(f"平均查询时间较长: {metrics.avg_query_time:.2f}s")
            recommendations.append("考虑查询优化")
            status = HealthStatus.WARNING

        return issues, recommendations, status

    def _check_error_rate(self, metrics: DatabaseMetrics) -> tuple[List[str], List[str], HealthStatus]:
        """检查错误率"""
        issues = []
        recommendations = []
        status = HealthStatus.HEALTHY

        if metrics.query_count > 0:
            error_rate = metrics.error_count / metrics.query_count
            if error_rate > self.monitor_config['critical_thresholds']['error_rate']:
                issues.append(f"错误率过高: {error_rate:.2%}")
                recommendations.append("检查错误日志，排查问题")
                status = HealthStatus.CRITICAL
            elif error_rate > self.monitor_config['warning_thresholds']['error_rate']:
                issues.append(f"错误率较高: {error_rate:.2%}")
                recommendations.append("监控错误趋势")
                status = HealthStatus.WARNING

        return issues, recommendations, status

    def _check_resource_usage(self, metrics: DatabaseMetrics) -> tuple[List[str], List[str], HealthStatus]:
        """检查资源使用（重构版，使用健康状态评估器）"""
        # 如果健康状态评估器可用，使用它来简化逻辑
        if hasattr(self, 'health_evaluator') and self.health_evaluator:
            return self._check_resource_usage_with_evaluator(metrics)
        else:
            # 回退到原有的复杂逻辑
            return self._check_resource_usage_legacy(metrics)
    
    def _check_resource_usage_with_evaluator(self, metrics: DatabaseMetrics) -> tuple[List[str], List[str], HealthStatus]:
        """使用健康状态评估器检查资源使用"""
        # 创建指标列表
        resource_metrics = [
            MetricValue("memory_usage", metrics.memory_usage),
            MetricValue("cpu_usage", metrics.cpu_usage),
            MetricValue("disk_usage", metrics.disk_usage)
        ]
        
        # 使用评估器进行评估
        status, issues, recommendations = self.health_evaluator.evaluate_multiple_metrics(resource_metrics)
        
        # 转换状态枚举（如果需要）
        if hasattr(status, 'value'):
            # 确保状态值是小写的，符合HealthStatus枚举定义
            status_value = status.value.lower()
            try:
                health_status = HealthStatus(status_value)
            except ValueError:
                # 如果状态值不匹配，使用默认的健康状态
                health_status = HealthStatus.HEALTHY
        else:
            health_status = HealthStatus.HEALTHY
        
        return issues, recommendations, health_status
    
    def _check_resource_usage_legacy(self, metrics: DatabaseMetrics) -> tuple[List[str], List[str], HealthStatus]:
        """检查资源使用（原有复杂逻辑，向后兼容）"""
        issues = []
        recommendations = []
        status = HealthStatus.HEALTHY

        # 检查内存使用
        if metrics.memory_usage > self.monitor_config['critical_thresholds']['memory_usage']:
            issues.append(f"内存使用率过高: {metrics.memory_usage:.1%}")
            recommendations.append("考虑增加内存或优化内存使用")
            status = HealthStatus.CRITICAL
        elif metrics.memory_usage > self.monitor_config['warning_thresholds']['memory_usage']:
            issues.append(f"内存使用率较高: {metrics.memory_usage:.1%}")
            recommendations.append("监控内存使用趋势")
            status = HealthStatus.WARNING

        # 检查CPU使用
        if metrics.cpu_usage > self.monitor_config['critical_thresholds']['cpu_usage']:
            issues.append(f"CPU使用率过高: {metrics.cpu_usage:.1%}")
            recommendations.append("考虑增加CPU资源或优化查询")
            status = HealthStatus.CRITICAL
        elif metrics.cpu_usage > self.monitor_config['warning_thresholds']['cpu_usage']:
            issues.append(f"CPU使用率较高: {metrics.cpu_usage:.1%}")
            recommendations.append("监控CPU使用趋势")
            status = HealthStatus.WARNING

        # 检查磁盘使用
        if metrics.disk_usage > self.monitor_config['critical_thresholds']['disk_usage']:
            issues.append(f"磁盘使用率过高: {metrics.disk_usage:.1%}")
            recommendations.append("考虑清理数据或扩容磁盘")
            status = HealthStatus.CRITICAL
        elif metrics.disk_usage > self.monitor_config['warning_thresholds']['disk_usage']:
            issues.append(f"磁盘使用率较高: {metrics.disk_usage:.1%}")
            recommendations.append("监控磁盘使用趋势")
            status = HealthStatus.WARNING

        return issues, recommendations, status

    def _combine_health_status(self, statuses: List[HealthStatus]) -> HealthStatus:
        """合并健康状态"""
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY

    def _perform_health_check(self, component: str, metrics: DatabaseMetrics) -> HealthCheckResult:
        """执行健康检查"""
        # 执行各项检查
        conn_issues, conn_recs, conn_status = self._check_connection_count(metrics)
        time_issues, time_recs, time_status = self._check_query_time(metrics)
        error_issues, error_recs, error_status = self._check_error_rate(metrics)
        resource_issues, resource_recs, resource_status = self._check_resource_usage(metrics)

        # 合并结果
        all_issues = conn_issues + time_issues + error_issues + resource_issues
        all_recommendations = conn_recs + time_recs + error_recs + resource_recs
        overall_status = self._combine_health_status(
            [conn_status, time_status, error_status, resource_status])

        return HealthCheckResult(
            status=overall_status,
            component=component,
            metrics=metrics,
            issues=all_issues,
            recommendations=all_recommendations,
            timestamp=datetime.now()
        )

    def _append_health_history(self, component: str, check_result):
        """追加健康历史并限制长度"""
        self.health_history[component].append(check_result)
        # 限制历史记录长度为100
        if len(self.health_history[component]) > MAX_HISTORY_LENGTH:
            self.health_history[component] = self.health_history[component][-100:]

    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        report = {
            'overall_status': HealthStatus.HEALTHY.value,
            'components': {},
            'summary': {
                'total_checks': 0,
                'healthy_count': 0,
                'warning_count': 0,
                'critical_count': 0
            }
        }

        # 检查各组件状态
        for component, history in self.health_history.items():
            if history:
                latest = history[-1]
                report['components'][component] = {
                    'status': latest.status.value,
                    'last_check': latest.timestamp.isoformat(),
                    'issues': latest.issues,
                    'recommendations': latest.recommendations,
                    'metrics': {
                        'connection_count': latest.metrics.connection_count,
                        'avg_query_time': latest.metrics.avg_query_time,
                        'error_count': latest.metrics.error_count,
                        'memory_usage': latest.metrics.memory_usage,
                        'cpu_usage': latest.metrics.cpu_usage,
                        'disk_usage': latest.metrics.disk_usage
                    }
                }

                # 更新统计
                report['summary']['total_checks'] += 1
                if latest.status == HealthStatus.HEALTHY:
                    report['summary']['healthy_count'] += 1
                elif latest.status == HealthStatus.WARNING:
                    report['summary']['warning_count'] += 1
                elif latest.status == HealthStatus.CRITICAL:
                    report['summary']['critical_count'] += 1

        # 确定整体状态
        if report['summary']['critical_count'] > 0:
            report['overall_status'] = HealthStatus.CRITICAL.value
        elif report['summary']['warning_count'] > 0:
            report['overall_status'] = HealthStatus.WARNING.value

        return report

    def get_component_health(self, component: str) -> Optional[HealthCheckResult]:
        """获取指定组件的健康状态"""
        if component in self.health_history and self.health_history[component]:
            return self.health_history[component][-1]
        return None

    # =========================================================================
    # 异步处理能力扩展
    # =========================================================================

    def _create_database_check_tasks(self) -> List[Coroutine]:
        """创建数据库检查任务"""
        tasks = []
        if self.data_manager.has_postgresql:
            tasks.append(self._check_postgresql_health_async())
        if self.data_manager.has_influxdb:
            tasks.append(self._check_influxdb_health_async())
        return tasks

    def _analyze_health_check_results(self, results: List[Any]) -> Dict[str, int]:
        """分析健康检查结果"""
        healthy_count = 0
        warning_count = 0
        critical_count = 0

        for result in results:
            if isinstance(result, Exception):
                critical_count += 1
                logger.error(f"异步健康检查异常: {result}")
            elif result and hasattr(result, 'status'):
                if result.status == HealthStatus.HEALTHY:
                    healthy_count += 1
                elif result.status == HealthStatus.WARNING:
                    warning_count += 1
                elif result.status == HealthStatus.CRITICAL:
                    critical_count += 1

        return {
            "healthy_count": healthy_count,
            "warning_count": warning_count,
            "critical_count": critical_count
        }

    def _determine_overall_status(self, counts: Dict[str, int]) -> str:
        """确定整体状态"""
        if counts["critical_count"] > 0:
            return HealthStatus.CRITICAL.value
        elif counts["warning_count"] > 0:
            return HealthStatus.WARNING.value
        else:
            return HealthStatus.HEALTHY.value

    def _create_success_response(self, overall_status: str, counts: Dict[str, int],
                                 total_components: int) -> Dict[str, Any]:
        """创建成功响应"""
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "healthy_count": counts["healthy_count"],
            "warning_count": counts["warning_count"],
            "critical_count": counts["critical_count"],
            "total_components": total_components
        }

    def _create_no_components_response(self) -> Dict[str, Any]:
        """创建无组件响应"""
        return {
            "status": HealthStatus.WARNING.value,
            "timestamp": datetime.now().isoformat(),
            "message": "没有配置数据库组件"
        }

    def _create_error_response(self, error: Exception) -> Dict[str, Any]:
        """创建错误响应"""
        logger.error(f"异步数据库健康检查失败: {error}")
        return {
            "status": HealthStatus.CRITICAL.value,
            "timestamp": datetime.now().isoformat(),
            "error": str(error)
        }

    async def check_health_async(self) -> Dict[str, Any]:
        """异步执行数据库健康检查"""
        try:
            # 创建检查任务
            tasks = self._create_database_check_tasks()

            if tasks:
                # 执行并发检查
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 分析结果
                counts = self._analyze_health_check_results(results)
                overall_status = self._determine_overall_status(counts)

                return self._create_success_response(overall_status, counts, len(tasks))
            else:
                return self._create_no_components_response()

        except Exception as e:
            return self._create_error_response(e)

    async def _check_postgresql_health_async(self) -> Optional[HealthCheckResult]:
        """异步检查PostgreSQL健康状态"""
        try:
            # 在线程池中执行同步操作
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._check_postgresql_health)
            return result
        except Exception as e:
            logger.error(f"异步PostgreSQL健康检查失败: {e}")
            return None

    async def _check_influxdb_health_async(self) -> Optional[HealthCheckResult]:
        """异步检查InfluxDB健康状态"""
        try:
            # 在线程池中执行同步操作
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._check_influxdb_health)
            return result
        except Exception as e:
            logger.error(f"异步InfluxDB健康检查失败: {e}")
            return None

    async def start_monitoring_async(self) -> bool:
        """异步启动监控"""
        try:
            if getattr(self, 'monitoring', False):
                logger.warning("监控已在运行中")
                return True

            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop_async, daemon=True)
            self.monitor_thread.start()

            # 调用外部监控器的启动方法
            if hasattr(self, 'monitor') and self.monitor:
                try:
                    self.monitor.start_monitoring()
                except Exception as e:
                    logger.warning(f"调用外部监控器启动失败: {e}")

            logger.info("异步数据库健康监控已启动")
            return True
        except Exception as e:
            logger.error(f"启动异步监控失败: {e}")
            self.monitoring = False
            return False

    async def stop_monitoring_async(self) -> bool:
        """异步停止监控"""
        try:
            # 调用外部监控器的停止方法（无论当前状态如何）
            if hasattr(self, 'monitor') and self.monitor:
                try:
                    self.monitor.stop_monitoring()
                except Exception as e:
                    logger.warning(f"调用外部监控器停止失败: {e}")

            if not getattr(self, 'monitoring', False):
                return True

            self.monitoring = False

            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)

            logger.info("异步数据库健康监控已停止")
            return True
        except Exception as e:
            logger.error(f"停止异步监控失败: {e}")
            return False

    def _monitor_loop_async(self):
        """异步监控循环"""
        while getattr(self, 'monitoring', False):
            try:
                # 创建新的事件循环来运行异步检查
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # 运行异步健康检查
                result = loop.run_until_complete(self.check_health_async())

                # 记录监控指标
                self.monitor.record_metric(
                    'database_health_async',
                    1 if result.get('status') == HealthStatus.HEALTHY.value else 0,
                    {'status': result.get('status', 'unknown')}
                )

                loop.close()

            except Exception as e:
                logger.error(f"异步监控循环异常: {e}")

            time.sleep(DEFAULT_CHECK_INTERVAL)

    # =========================================================================
    # IUnifiedInfrastructureInterface 实现
    # =========================================================================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化数据库健康监控器

        Args:
            config: 配置参数

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("开始初始化DatabaseHealthMonitor")
            logger.debug(f"初始化配置参数: {config.keys() if config else 'None'}")

            # 如果提供了配置，更新现有配置
            if config:
                logger.debug("更新现有配置")
                self.monitor_config.update(config)
                logger.info("配置已更新")

            logger.info("DatabaseHealthMonitor 初始化完成")
            logger.debug(f"监控间隔: {self.monitor_config['check_interval']}秒")
            return True
        except Exception as e:
            logger.error(f"DatabaseHealthMonitor 初始化失败: {e}", exc_info=True)
            return False

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """
        return {
            "component_type": "DatabaseHealthMonitor",
            "version": "1.0.0",
            "description": "数据库健康监控器",
            "capabilities": [
                "database_health_check",
                "performance_monitoring",
                "async_processing",
                "real_time_alerts"
            ],
            "supported_databases": ["postgresql", "influxdb", "redis"],
            "monitoring_active": getattr(self, 'monitoring', False),
            "check_interval": self.monitor_config.get('check_interval', DEFAULT_CHECK_INTERVAL),
            "metrics_history_length": {
                component: len(history)
                for component, history in self.metrics_history.items()
            },
            "health_history_length": {
                component: len(history)
                for component, history in self.health_history.items()
            }
        }

    def is_healthy(self) -> bool:
        """检查组件健康状态

        Returns:
            bool: 组件是否健康
        """
        try:
            # 检查数据管理器是否可用
            if not hasattr(self, 'data_manager') or not self.data_manager:
                return False

            # 检查监控器是否可用
            if not hasattr(self, 'monitor') or not self.monitor:
                return False

            # 检查配置是否有效
            if not hasattr(self, 'monitor_config') or not self.monitor_config:
                return False

            # 检查历史记录数据结构是否完整
            if not hasattr(self, 'metrics_history') or not hasattr(self, 'health_history'):
                return False

            return True
        except Exception as e:
            logger.error(f"DatabaseHealthMonitor 健康检查失败: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标

        Returns:
            Dict[str, Any]: 组件指标数据
        """
        try:
            # 获取最新的健康报告
            health_report = self.get_health_report()

            # 计算监控指标
            total_components = len(self.health_history)
            total_checks = sum(len(history) for history in self.health_history.values())

            # 计算平均响应时间
            response_times = []
            for component_history in self.health_history.values():
                for check_result in component_history:
                    if hasattr(check_result, 'timestamp') and hasattr(check_result, 'issues'):
                        # 估算响应时间（这里简化处理）
                        response_times.append(0.1)  # 默认响应时间

            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            return {
                "component_type": "DatabaseHealthMonitor",
                "timestamp": datetime.now().isoformat(),
                "health_status": health_report.get('overall_status', 'unknown'),
                "monitoring_active": self.monitoring_active,
                "total_components": total_components,
                "total_checks": total_checks,
                "avg_response_time": avg_response_time,
                "memory_usage": psutil.virtual_memory().percent if psutil else 0,
                "cpu_usage": psutil.cpu_percent(interval=0.1) if psutil else 0,
                "components_status": {
                    component: len(history) > 0 and
                    (history[-1].status.value if hasattr(history[-1], 'status') else 'unknown')
                    for component, history in self.health_history.items()
                }
            }
        except Exception as e:
            logger.error(f"获取DatabaseHealthMonitor指标失败: {e}")
            return {
                "component_type": "DatabaseHealthMonitor",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def cleanup(self) -> bool:
        """清理组件资源

        Returns:
            bool: 清理是否成功
        """
        try:
            # 停止监控
            if getattr(self, 'monitoring', False):
                self.stop_monitoring()

            # 停止异步监控
            if hasattr(self, 'monitoring') and getattr(self, 'monitoring', False):
                # 注意：这里需要同步等待异步停止
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.stop_monitoring_async())
                loop.close()

            # 清理历史记录
            for history in self.metrics_history.values():
                history.clear()
            for history in self.health_history.values():
                history.clear()

            # 清理监控线程
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                if self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=2.0)

            logger.info("DatabaseHealthMonitor 资源清理完成")
            return True
        except Exception as e:
            logger.error(f"DatabaseHealthMonitor 资源清理失败: {e}")
            return False
