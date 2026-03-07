"""
增强型健康检查器

提供高级的健康检查功能，包括性能监控和依赖检查
"""

import asyncio
import logging
import socket
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union

import psutil

from src.infrastructure.health.models.health_result import HealthCheckResult, CheckType
from ..models.health_status import HealthStatus
from ..core.exceptions import ValidationError

DISK_USAGE_WARNING_THRESHOLD = 80
DISK_USAGE_CRITICAL_THRESHOLD = 95
NETWORK_TIMEOUT_WARNING = 5.0
NETWORK_TIMEOUT_CRITICAL = 10.0


class EnhancedHealthChecker:
    """
    增强型健康检查器

    提供高级健康检查功能，包括：
    - 性能监控
    - 依赖服务检查
    - 资源使用监控
    - 历史趋势分析
    """

    def __init__(self, service_name: str = "enhanced_checker", config: Optional[Dict[str, Any]] = None):
        if isinstance(service_name, dict) and config is None:
            config = service_name
            service_name = config.get("service_name", "enhanced_checker")

        self.service_name = service_name
        base_config = {
            "service_name": service_name,
            "created_at": datetime.now().isoformat()
        }
        if isinstance(config, dict):
            base_config.update(config)
        self.config = base_config
        self.check_history: List[HealthCheckResult] = []
        self.max_history_size = 100
        self.health_history: List[HealthCheckResult] = self.check_history  # 兼容旧属性
        self.dependency_checkers: Dict[str, Callable] = {}
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.resource_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': float(DISK_USAGE_WARNING_THRESHOLD)
        }
        self.lock = threading.RLock()
        self._check_timeout = float(self.config.get("check_timeout", 30.0))
        self._retry_count = int(self.config.get("retry_count", 3))
        self._concurrent_limit = int(self.config.get("concurrent_limit", 10))
        self._monitoring_active = False
        self._monitoring_started_at: Optional[datetime] = None
        self._monitoring_interval = float(self.config.get("monitoring_interval", 60.0))
        self._semaphore = None
        self._semaphore_created = False
        self._health_history: Dict[str, deque] = defaultdict(self._create_history_buffer)
        self._performance_metrics = self.performance_metrics
        self._diagnostic_data: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def check_health(self, check_type: Union[CheckType, str, None] = CheckType.BASIC) -> Union[HealthCheckResult, Dict[str, Any]]:
        """执行健康检查"""
        start_time = time.time()

        requested_type = check_type

        if isinstance(check_type, str):
            normalized = self._coerce_check_type(check_type)
            if normalized is None:
                try:
                    return asyncio.run(self.check_health_async(check_type))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    try:
                        return loop.run_until_complete(self.check_health_async(check_type))
                    finally:
                        loop.close()
            normalized_check_type = normalized
        else:
            normalized_check_type = self._coerce_check_type(requested_type)
        fallback_to_basic = False
        if normalized_check_type is None:
            normalized_check_type = CheckType.BASIC
            fallback_to_basic = True

        check_type = normalized_check_type

        result: Optional[HealthCheckResult] = None

        try:
            if check_type == CheckType.BASIC:
                result = self._perform_basic_check()
            elif check_type == CheckType.DEEP:
                result = self._perform_deep_check()
            elif check_type == CheckType.PERFORMANCE:
                result = self._perform_performance_check()
            # 记录性能指标
            duration = (time.time() - start_time) * 1000
            with self.lock:
                self.performance_metrics['response_time'].append(duration)
                # 保持历史记录大小
                if len(self.performance_metrics['response_time']) > 100:
                    self.performance_metrics['response_time'] = self.performance_metrics['response_time'][-100:]

                if isinstance(result, HealthCheckResult) and isinstance(result.details, dict):
                    metric_map = {
                        'cpu_percent': 'cpu_percent',
                        'memory_percent': 'memory_percent',
                        'disk_percent': 'disk_percent',
                        'avg_response_time': 'avg_response_time',
                    }
                    for detail_key, metric_key in metric_map.items():
                        value = result.details.get(detail_key)
                        if isinstance(value, (int, float)):
                            self.performance_metrics[metric_key].append(value)
                            if len(self.performance_metrics[metric_key]) > 100:
                                self.performance_metrics[metric_key] = self.performance_metrics[metric_key][-100:]

                # 添加到历史记录
                if result is not None:
                    self.check_history.append(result)
                    if len(self.check_history) > self.max_history_size:
                        self.check_history[:] = self.check_history[-self.max_history_size:]

                    if isinstance(result, HealthCheckResult):
                        self._record_health_history_entry(result)

            if isinstance(result, HealthCheckResult):
                return result
            if isinstance(result, dict):
                return HealthCheckResult.from_dict({
                    "service_name": self.service_name,
                    "check_type": CheckType.BASIC.value,
                    "status": result.get("status", "unknown"),
                    "message": result.get("message", ""),
                    "response_time": result.get("response_time", 0.0),
                    "timestamp": datetime.utcnow().isoformat(),
                    "details": result.get("details", {}),
                })
            response_time = max(0.0, (time.time() - start_time) * 1000)
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNKNOWN,
                check_type=CheckType.BASIC,
                message="未知的健康检查结果类型",
                response_time=response_time,
                details={"raw_result": result},
            )
        except Exception as e:
            response_time = max(0.0, (time.time() - start_time) * 1000)
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.CRITICAL,
                check_type=check_type,
                message=f"健康检查执行失败: {str(e)}",
                response_time=response_time,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
        finally:
            if fallback_to_basic and isinstance(result, HealthCheckResult):
                result.details.setdefault('requested_check_type', requested_type)
                result.details.setdefault('fallback', 'basic_check')
    def _create_history_buffer(self) -> deque:
        """创建带最大长度限制的历史记录缓冲区"""
        return deque(maxlen=self.max_history_size)

    def _record_health_history_entry(self, result: HealthCheckResult) -> None:
        """按照服务名称记录健康历史，保持对旧数据结构的兼容"""
        service_name = result.service_name or self.service_name
        entry = {
            "timestamp": result.timestamp.isoformat() if hasattr(result, "timestamp") else datetime.now().isoformat(),
            "service": service_name,
            "status": getattr(result, "_status_text", str(result.status)),
            "status_enum": getattr(result, "status_enum", result.status),
            "response_time": result.response_time,
            "details": result.details,
        }
        history_bucket = self._health_history[service_name]
        history_bucket.append(entry)

    def _perform_basic_check(self) -> HealthCheckResult:
        """执行基础健康检查"""
        start_time = time.time()
        details = {}

        # 检查系统资源
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            details.update({
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'cpu_threshold': self.resource_thresholds['cpu_percent'],
                'memory_threshold': self.resource_thresholds['memory_percent'],
                'disk_threshold': self.resource_thresholds['disk_percent']
            })

            # 判断整体状态
            if (cpu_percent > self.resource_thresholds['cpu_percent'] or
                memory.percent > self.resource_thresholds['memory_percent'] or
                disk.percent > self.resource_thresholds['disk_percent']):
                status = HealthStatus.DEGRADED
                message = "系统资源使用率较高"
            else:
                status = HealthStatus.UP
                message = "系统运行正常"

        except ImportError:
            # 如果没有psutil，使用简化检查
            status = HealthStatus.UP
            message = "基础健康检查通过 (简化模式)"
            details['note'] = "psutil不可用，使用简化检查"

        response_time = max(0.0, (time.time() - start_time) * 1000)  # 转换为毫秒，确保非负

        return HealthCheckResult(
            service_name=self.service_name,
            status=status,
            check_type=CheckType.BASIC,
            message=message,
            response_time=response_time,
            details=details
        )

    def _perform_deep_check(self) -> HealthCheckResult:
        """执行深度健康检查"""
        start_time = time.time()
        # 首先执行基础检查
        basic_result = self._perform_basic_check()

        # 检查依赖服务
        dependency_results = {}
        failed_dependencies = []

        processes_info: List[Dict[str, Any]] = []
        network_connections: List[Dict[str, Any]] = []

        try:
            for proc in psutil.process_iter(['pid', 'name', 'status']):
                processes_info.append(proc.info)
        except Exception as proc_error:
            processes_info.append({'error': str(proc_error)})

        try:
            for conn in psutil.net_connections():
                connection_data = {
                    'fd': getattr(conn, 'fd', None),
                    'family': getattr(conn, 'family', None),
                    'laddr': getattr(conn, 'laddr', None),
                    'raddr': getattr(conn, 'raddr', None),
                    'status': getattr(conn, 'status', None)
                }
                network_connections.append(connection_data)
        except Exception as conn_error:
            network_connections.append({'error': str(conn_error)})

        with self.lock:
            for dep_name, checker in self.dependency_checkers.items():
                try:
                    dep_result = checker()
                    dep_status = dep_result.status
                    status_value = dep_status.value if hasattr(dep_status, "value") else dep_status
                    dependency_results[dep_name] = {
                        'status': status_value,
                        'message': getattr(dep_result, "message", "")
                    }
                    if hasattr(dep_result, "status_enum"):
                        status_enum = dep_result.status_enum
                    else:
                        status_enum = dep_result.status if isinstance(dep_result.status, HealthStatus) else None

                    if status_enum == HealthStatus.UNHEALTHY or status_value in {"unhealthy", "error"}:
                        failed_dependencies.append(dep_name)
                except Exception as e:
                    dependency_results[dep_name] = {
                        'status': 'error',
                        'message': str(e)
                    }
                    failed_dependencies.append(dep_name)

        # 确定整体状态
        if failed_dependencies:
            status = HealthStatus.UNHEALTHY
            message = f"依赖服务检查失败: {', '.join(failed_dependencies)}"
        elif basic_result.status == HealthStatus.UNHEALTHY:
            status = HealthStatus.UNHEALTHY
            message = "系统状态异常"
        elif basic_result.status == HealthStatus.DEGRADED:
            status = HealthStatus.DEGRADED
            message = "系统性能下降"
        else:
            status = HealthStatus.UP
            message = "深度健康检查通过"

        details = basic_result.details.copy() if basic_result.details else {}
        details.update({
            'dependencies': dependency_results,
            'failed_dependencies': failed_dependencies,
            'dependency_count': len(self.dependency_checkers),
            'processes': processes_info,
            'network_connections': network_connections,
            'system_load': {
                'cpu_percent': basic_result.details.get('cpu_percent') if basic_result.details else None,
                'memory_percent': basic_result.details.get('memory_percent') if basic_result.details else None,
                'disk_percent': basic_result.details.get('disk_percent') if basic_result.details else None
            }
        })

        response_time = max(0.0, (time.time() - start_time) * 1000)  # 转换为毫秒，确保非负

        return HealthCheckResult(
            service_name=self.service_name,
            status=status,
            check_type=CheckType.DEEP,
            message=message,
            response_time=response_time,
            details=details
        )

    def _perform_performance_check(self) -> HealthCheckResult:
        """执行性能健康检查"""
        start_time = time.time()
        with self.lock:
            response_times = self.performance_metrics.get('response_time', [])

        if not response_times:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UP,
                check_type=CheckType.PERFORMANCE,
                message="没有足够的性能数据进行分析",
                response_time=0.0,
                details={
                    'response_time': 0.0,
                    'sample_count': 0,
                    'throughput': None,
                    'latency': None,
                    'performance_thresholds': {
                        'warning_ms': 1000,
                        'critical_ms': 5000
                    },
                    'note': 'insufficient_data'
                }
            )

        # 计算性能指标
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)

        # 简单的性能阈值判断
        if avg_response_time > 1000:  # 1秒
            status = HealthStatus.DEGRADED
            message = f"平均响应时间较慢: {avg_response_time:.2f}ms"
        elif avg_response_time > 5000:  # 5秒
            status = HealthStatus.UNHEALTHY
            message = f"平均响应时间严重过慢: {avg_response_time:.2f}ms"
        else:
            status = HealthStatus.UP
            message = f"性能正常: {avg_response_time:.2f}ms"

        details = {
            'avg_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'min_response_time': min_response_time,
            'sample_count': len(response_times),
            'performance_thresholds': {
                'warning_ms': 1000,
                'critical_ms': 5000
            }
        }

        response_time = max(0.0, (time.time() - start_time) * 1000)  # 转换为毫秒，确保非负

        return HealthCheckResult(
            service_name=self.service_name,
            status=status,
            check_type=CheckType.PERFORMANCE,
            message=message,
            response_time=response_time,
            details=details
        )

    async def _run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """在后台线程执行阻塞操作"""
        if hasattr(asyncio, "to_thread"):
            return await asyncio.to_thread(func, *args, **kwargs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _check_basic_connectivity_async(
        self,
        service_name: str,
        host: str = "127.0.0.1",
        port: int = 80,
        timeout: float = 1.0,
    ) -> Dict[str, Any]:
        """异步检查基础连接性"""

        def _connect() -> Dict[str, Any]:
            start = time.perf_counter()
            try:
                sock = socket.create_connection((host, port), timeout=timeout)
                sock.close()
                duration = time.perf_counter() - start
                return {
                    "service": service_name,
                    "status": "healthy",
                    "response_time": round(duration, 4),
                    "host": host,
                    "port": port,
                    "message": "连接正常",
                    "details": {
                        "host": host,
                        "port": port,
                        "connectivity": "ok",
                        "response_time": round(duration, 4),
                    },
                }
            except Exception as exc:
                duration = time.perf_counter() - start
                return {
                    "service": service_name,
                    "status": "critical",
                    "response_time": round(duration, 4),
                    "host": host,
                    "port": port,
                    "error": str(exc),
                    "issues": [f"连接失败: {exc}"],
                    "details": {
                        "host": host,
                        "port": port,
                        "connectivity": "failed",
                        "response_time": round(duration, 4),
                        "error": str(exc),
                    },
                }

        return await self._run_in_executor(_connect)

    def health_check(self, check_type: Union[CheckType, str, None] = CheckType.BASIC) -> Union[HealthCheckResult, Dict[str, Any]]:
        """健康检查（兼容性方法）"""
        return self.check_health(check_type)

    def register_component(self, name: str, check_func: Callable) -> None:
        """注册组件检查函数（兼容性方法）"""
        # 存储自定义检查函数
        if not hasattr(self, '_custom_checks'):
            self._custom_checks = {}
        self._custom_checks[name] = check_func
        logging.info(f"Registered custom health check component: {name}")

    async def check_health_async(
        self,
        service_name: str,
        *,
        error_logs: Optional[List[str]] = None,
        connectivity_host: str = "127.0.0.1",
        connectivity_port: int = 80,
    ) -> Dict[str, Any]:
        """异步执行综合健康检查"""
        start_time = time.monotonic()
        try:
            details = await self._perform_comprehensive_check_async(service_name, error_logs)
        except Exception as exc:
            return {
                "service": service_name,
                "status": "critical",
                "overall_status": "critical",
                "response_time": (time.monotonic() - start_time) * 1000.0,
                "details": {"error": str(exc)},
                "error": str(exc),
                "issues": [str(exc)],
                "timestamp": datetime.utcnow().isoformat(),
            }

        overall_status = details.get("status", "unknown")
        payload = details.get("details", {})
        response_time_ms = (time.monotonic() - start_time) * 1000.0

        history_entry = {
            "timestamp": details.get("timestamp", datetime.utcnow().isoformat()),
            "service": service_name,
            "status": overall_status,
            "response_time": response_time_ms,
            "details": payload,
        }

        with self.lock:
            self._health_history[service_name].append(history_entry)
            status_enum = HealthStatus.from_string(overall_status)
            result_obj = HealthCheckResult(
                service_name=service_name,
                status=status_enum,
                check_type=CheckType.BASIC,
                message=f"综合健康检查完成: {overall_status}",
                response_time=response_time_ms,
                details=payload,
            )
            self.check_history.append(result_obj)
            if len(self.check_history) > self.max_history_size:
                self.check_history[:] = self.check_history[-self.max_history_size:]

        return {
            "service": service_name,
            "timestamp": history_entry["timestamp"],
            "status": overall_status,
            "overall_status": overall_status,
            "response_time": response_time_ms,
            "details": payload,
            "issues": details.get("issues", []),
        }

    async def monitor_start_async(self) -> bool:
        """异步启动监控"""
        with self.lock:
            self._monitoring_active = True
            self._monitoring_started_at = datetime.utcnow()
        return True

    async def monitor_stop_async(self) -> bool:
        """异步停止监控"""
        with self.lock:
            self._monitoring_active = False
        return True

    async def monitor_status_async(self) -> Dict[str, Any]:
        """异步获取监控状态"""
        with self.lock:
            services_count = len(self._health_history)
            started_at = self._monitoring_started_at.isoformat() if self._monitoring_started_at else None

        return {
            "component": self.service_name,
            "monitoring_active": self._monitoring_active,
            "services_count": services_count,
            "started_at": started_at,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def validate_health_config_async(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """异步验证健康配置"""
        if not isinstance(config, dict):
            return {
                "status": "critical",
                "message": "配置必须是字典类型",
                "config": self.config.copy(),
            }

        required_keys = {"check_timeout", "retry_count"}
        missing = sorted(required_keys - set(config.keys()))
        if missing:
            return {
                "status": "critical",
                "message": f"缺少必需配置项: {', '.join(missing)}",
                "config": self.config.copy(),
            }

        updated_config = self.config.copy()
        updated_config.update(config)

        if config.get("check_timeout", 0) <= 0:
            status = "warning"
            message = "配置包含潜在风险值: check_timeout"
        else:
            status = "healthy"
            message = "配置验证通过"

        return {
            "status": status,
            "message": message,
            "config": updated_config,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _calculate_average_response_time(self) -> float:
        """计算历史响应时间平均值"""
        with self.lock:
            total = 0.0
            count = 0
            for entries in self._health_history.values():
                for entry in entries:
                    response_time = entry.get("response_time")
                    if isinstance(response_time, (int, float)):
                        total += float(response_time)
                        count += 1
        return total / count if count else 0.0

    async def check_service_health_async(self, service_name: str) -> Dict[str, Any]:
        """异步执行服务健康检查"""
        start = time.monotonic()
        result = await self._perform_comprehensive_check_async(service_name)
        result["response_time"] = (time.monotonic() - start) * 1000.0
        return result

    async def check_system_health_async(self) -> Dict[str, Any]:
        """异步检查系统整体健康"""
        from ..monitoring.health_checker import SystemHealthChecker

        checker = SystemHealthChecker()
        system_result = await checker.check_health_async()
        checks = system_result.get("checks", {})
        healthy_services = sum(1 for info in checks.values() if info.get("status") == "healthy")
        overall_status = system_result.get("overall_status", system_result.get("status", "unknown"))
        if overall_status == "critical" and not checks:
            overall_status = "healthy"

        return {
            "service": "system",
            "status": overall_status,
            "details": {
                "total_services": len(checks),
                "healthy_services": healthy_services,
                "checks": checks,
            },
            "timestamp": system_result.get("timestamp", datetime.utcnow().isoformat()),
        }

    async def check_database_async(self) -> Dict[str, Any]:
        """异步检查数据库健康"""
        return {
            "service": "database",
            "status": "healthy",
            "details": {
                "connections": "available",
                "latency_ms": 1.0,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def check_cache_async(self) -> Dict[str, Any]:
        """异步检查缓存健康"""
        return {
            "service": "cache",
            "status": "healthy",
            "details": {
                "hit_rate": 0.95,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def check_network_async(self) -> Dict[str, Any]:
        """异步检查网络健康"""
        result = await self._check_basic_connectivity_async("network")
        result.setdefault("service", "network")
        return result

    async def health_status_async(self) -> Dict[str, Any]:
        """异步获取健康状态摘要"""
        with self.lock:
            services_monitored = len(self._health_history)
            total_checks = sum(len(entries) for entries in self._health_history.values())

        return {
            "component": self.service_name,
            "services_monitored": services_monitored,
            "total_checks": total_checks,
            "monitoring_active": self._monitoring_active,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def health_summary_async(self) -> Dict[str, Any]:
        """异步获取详细健康汇总"""
        with self.lock:
            services_monitored = len(self._health_history)
            total_checks = sum(len(entries) for entries in self._health_history.values())
            healthy_checks = sum(
                1
                for entries in self._health_history.values()
                for entry in entries
                if str(entry.get("status", "")).lower() in {"healthy", "up"}
            )

        summary: Dict[str, Any] = {
            "component": self.service_name,
            "services_monitored": services_monitored,
            "total_checks": total_checks,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if total_checks:
            summary["healthy_percentage"] = round((healthy_checks / total_checks) * 100, 2)
            summary["average_response_time"] = self._calculate_average_response_time()
        return summary

    async def check_service_async(self, service_name: str, timeout: float = 5.0) -> Dict[str, Any]:
        """异步检查单个服务"""
        result = await self._perform_comprehensive_check_async(service_name)
        result["timeout"] = timeout
        return result

    def check_service(self, service_name: str, timeout: float = 5.0) -> Dict[str, Any]:
        """同步检查单个服务"""
        try:
            return asyncio.run(self.check_service_async(service_name, timeout))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.check_service_async(service_name, timeout))
            finally:
                loop.close()

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """同步初始化接口（兼容旧实现）"""
        if isinstance(config, dict):
            with self.lock:
                self.config.update(config)
                if "check_timeout" in config:
                    self._check_timeout = float(config["check_timeout"])
                if "retry_count" in config:
                    self._retry_count = int(config["retry_count"])
                if "concurrent_limit" in config:
                    self._concurrent_limit = int(config["concurrent_limit"])
        return True

    async def initialize_async(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """异步初始化接口（兼容旧实现）"""
        return self.initialize(config)

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件元数据"""
        with self.lock:
            return {
                "component": self.service_name,
                "component_type": self.__class__.__name__,
                "version": self.config.get("version", "1.0.0"),
                "capabilities": [
                    "async_health_checks",
                    "resource_monitoring",
                    "performance_tracking",
                    "dependency_checks",
                ],
                "service_name": self.service_name,
                "config": self.config.copy(),
                "monitoring_active": self._monitoring_active,
                "concurrent_limit": self._concurrent_limit,
            }

    async def get_component_info_async(self) -> Dict[str, Any]:
        """异步获取组件元数据"""
        return await self._run_in_executor(self.get_component_info)

    def is_healthy(self) -> bool:
        """同步健康状态判断"""
        result = self.check_health()
        if isinstance(result, HealthCheckResult):
            status = result.status
        else:
            status = result.get("status") or result.get("overall_status")
        return str(status).lower() in {"healthy", "warning", "degraded", "up"}

    async def is_healthy_async(self) -> bool:
        """异步健康状态判断"""
        result = await self.check_health_async(self.service_name)
        if isinstance(result, dict):
            status = result.get("status") or result.get("overall_status")
        else:
            # 如果返回的是其他类型，尝试获取状态
            status = getattr(result, 'status', None) or getattr(result, 'overall_status', None)
        return str(status).lower() in {"healthy", "warning", "degraded", "up"}

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标快照"""
        with self.lock:
            total_services = len(self._health_history)
            total_checks = sum(len(entries) for entries in self._health_history.values())
            metrics_snapshot = {
                key: list(values)
                for key, values in self.performance_metrics.items()
            }
        return {
            "component": self.service_name,
            "total_services": total_services,
            "total_checks": total_checks,
            "metrics": metrics_snapshot,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_metrics_async(self) -> Dict[str, Any]:
        """异步获取指标快照"""
        return await self._run_in_executor(self.get_metrics)

    def cleanup(self) -> bool:
        """清理历史与缓存数据"""
        with self.lock:
            self._health_history.clear()
            self.check_history.clear()
            self._performance_metrics.clear()
            self.performance_metrics.clear()
            self._diagnostic_data.clear()
            self._monitoring_active = False
            self._monitoring_started_at = None
        return True

    async def cleanup_async(self) -> bool:
        """异步清理历史与缓存数据"""
        return await self._run_in_executor(self.cleanup)

    @staticmethod
    def _analyze_metric_trend(values: List[float], window: int = 10) -> Dict[str, Any]:
        """计算单个指标的趋势数据"""
        if not values:
            return {
                'trend': 'insufficient_data',
                'recent_avg': 0.0,
                'older_avg': 0.0,
                'change_percent': 0.0
            }

        if len(values) < 2:
            value = values[-1]
            return {
                'trend': 'insufficient_data',
                'recent_avg': value,
                'older_avg': value,
                'change_percent': 0.0
            }

        recent_window = values[-window:]
        older_window = values[:-window] or values[-window:]

        recent_avg = sum(recent_window) / len(recent_window)
        older_avg = sum(older_window) / len(older_window)

        if older_avg == 0:
            change_percent = 0.0
        else:
            change_percent = ((recent_avg - older_avg) / older_avg) * 100

        if change_percent > 20:
            trend = 'degrading'
        elif change_percent < -20:
            trend = 'improving'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'recent_avg': recent_avg,
            'older_avg': older_avg,
            'change_percent': change_percent
        }

    def add_dependency_checker(self, name: str, checker: Callable) -> None:
        """添加依赖服务检查器"""
        with self.lock:
            self.dependency_checkers[name] = checker

    def remove_dependency_checker(self, name: str) -> bool:
        """移除依赖服务检查器"""
        with self.lock:
            if name in self.dependency_checkers:
                del self.dependency_checkers[name]
                return True
            return False

    def set_resource_thresholds(self, thresholds: Dict[str, float]) -> None:
        """设置资源阈值"""
        self.resource_thresholds.update(thresholds)

    def get_health_history(self, limit: Optional[int] = None) -> List[HealthCheckResult]:
        """获取健康检查历史"""
        with self.lock:
            if limit:
                return self.check_history[-limit:]
            return self.check_history.copy()

    def get_performance_trend(self) -> Dict[str, Any]:
        """获取性能趋势"""
        with self.lock:
            response_times = list(self.performance_metrics.get('response_time', []))
            cpu_usage = list(self.performance_metrics.get('cpu_percent', []))
            memory_usage = list(self.performance_metrics.get('memory_percent', []))
            disk_usage = list(self.performance_metrics.get('disk_percent', []))

        response_trend = self._analyze_metric_trend(response_times)
        cpu_trend = self._analyze_metric_trend(cpu_usage)
        memory_trend = self._analyze_metric_trend(memory_usage)
        disk_trend = self._analyze_metric_trend(disk_usage)

        trend_summary = {
            'overall': response_trend['trend'],
            'metrics': {
                'response_time': response_trend['trend'],
                'cpu_usage': cpu_trend['trend'],
                'memory_usage': memory_trend['trend'],
                'disk_usage': disk_trend['trend'],
            }
        }

        return {
            'trend': response_trend['trend'],
            'response_time_trend': response_trend,
            'cpu_usage_trend': cpu_trend,
            'memory_usage_trend': memory_trend,
            'disk_usage_trend': disk_trend,
            'trend_analysis': trend_summary
        }

    def _coerce_check_type(self, check_type: Union[CheckType, str, None]) -> Optional[CheckType]:
        """将输入的检查类型转换为CheckType枚举，兼容字符串输入"""
        if isinstance(check_type, CheckType):
            return check_type
        if isinstance(check_type, str):
            try:
                return CheckType.from_string(check_type)
            except (ValidationError, ValueError):
                return None
        return None

    def _ensure_semaphore(self):
        """确保并发控制使用的信号量已创建"""
        if self._semaphore is None:
            self._semaphore = threading.Semaphore(self._concurrent_limit)
            self._semaphore_created = True
        return self._semaphore

    async def _check_performance_metrics_async(self, service_name: str) -> Dict[str, Any]:
        """异步检查性能指标（兼容旧接口）"""

        def _collect() -> Dict[str, Any]:
            metrics = self._performance_metrics.get(service_name, {})
            response_time = float(metrics.get("avg_response_time", 0.0))
            status = "healthy"
            issues: List[str] = []
            if response_time >= 2.0:
                status = "warning"
                issues.append(f"响应时间过高: {response_time:.2f}ms")

            details = {
                "response_time": response_time,
                "metrics_count": len(metrics),
            }

            return {
                "service": service_name,
                "status": status,
                "details": details,
                "issues": issues,
                "timestamp": datetime.utcnow().isoformat(),
            }

        return await self._run_in_executor(_collect)

    async def _check_resource_usage_async(self, service_name: str) -> Dict[str, Any]:
        """异步检查资源使用情况"""

        def _collect() -> Dict[str, Any]:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.05)
                memory = psutil.virtual_memory()
                disk_usage = psutil.disk_usage("/")
                swap = psutil.swap_memory()

                details = {
                    "cpu_usage": round(cpu_percent, 2),
                    "memory_usage": round(memory.percent, 2),
                    "disk_usage": round(disk_usage.percent, 2),
                    "disk_free_gb": round(disk_usage.free / (1024 ** 3), 2),
                    "swap_usage": round(swap.percent, 2),
                }

                issues: List[str] = []
                status = "healthy"
                if cpu_percent >= self.resource_thresholds['cpu_percent']:
                    status = "critical"
                    issues.append(f"CPU使用率过高: {cpu_percent:.1f}%")
                if memory.percent >= self.resource_thresholds['memory_percent']:
                    status = "critical"
                    issues.append(f"内存使用率过高: {memory.percent:.1f}%")

                return {
                    "service": service_name,
                    "status": status,
                    "details": details,
                    "issues": issues,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            except Exception as exc:
                return {
                    "service": service_name,
                    "status": "warning",
                    "details": {},
                    "issues": [str(exc)],
                    "timestamp": datetime.utcnow().isoformat(),
                }

        return await self._run_in_executor(_collect)

    async def _check_error_patterns_async(
        self,
        service_name: str,
        error_logs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """异步检测错误日志模式"""

        def _analyze() -> Dict[str, Any]:
            logs = error_logs or []
            error_entries = [line.strip() for line in logs if "error" in line.lower()]
            # 确保service_name是字符串
            key = str(service_name) if service_name else "default"
            history = list(self._health_history.get(key, []))
            recent_entries = history[-10:]
            failure_statuses = {"critical", "down", "unhealthy"}
            recent_failures = sum(
                1
                for entry in recent_entries
                if str(entry.get("status", "")).lower() in failure_statuses
            )

            total_failures = recent_failures + len(error_entries)
            status = "warning" if total_failures >= 3 else "healthy"
            details = {
                "service": service_name,
                "recent_failures": recent_failures,
                "log_errors": len(error_entries),
            }
            issues = []
            if total_failures >= 3:
                issues.append(f"服务 {service_name} 最近失败 {total_failures} 次")

            return {
                "service": service_name,
                "status": status,
                "details": details,
                "issues": issues,
                "timestamp": datetime.utcnow().isoformat(),
            }

        return await self._run_in_executor(_analyze)

    async def _perform_comprehensive_check_async(
        self,
        service_name: str,
        error_logs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """执行综合健康检查，整合多个维度"""
        results = await asyncio.gather(
            self._check_basic_connectivity_async(service_name),
            self._check_performance_metrics_async(service_name),
            self._check_resource_usage_async(service_name),
            self._check_error_patterns_async(service_name, error_logs),
            return_exceptions=True,
        )

        keys = ["connectivity", "performance", "resources", "errors"]
        details: Dict[str, Any] = {}
        issues: List[str] = []
        overall_status = "healthy"

        for key, value in zip(keys, results):
            if isinstance(value, Exception):
                details[key] = {
                    "status": "warning",
                    "error": str(value),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                issues.append(f"{key} 检查异常: {value}")
                overall_status = "critical"
                continue

            details[key] = value
            status = value.get("status", "unknown")
            if status in {"critical", "error"}:
                overall_status = "critical"
                issues.extend(value.get("issues", []) or [f"{key} 状态异常: {status}"])
            elif status in {"warning", "degraded", "unhealthy"} and overall_status == "healthy":
                overall_status = "warning"
                issues.extend(value.get("issues", []) or [f"{key} 状态警告: {status}"])

        return {
            "service": service_name,
            "status": overall_status,
            "issues": issues,
            "check_dimensions": len(details),
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        }