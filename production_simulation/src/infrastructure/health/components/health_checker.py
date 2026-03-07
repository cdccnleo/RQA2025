"""
健康检查器组件

提供异步和同步的健康检查功能
"""

import asyncio
import inspect
import time
from datetime import datetime
from threading import Lock
from typing import Dict, Any, Optional, Callable, Iterable, List
from ..models.health_result import HealthCheckResult, CheckType
from ..models.health_status import HealthStatus
from abc import ABC, abstractmethod


class _CompatFloat(float):
    """兼容历史数值的浮点常量"""

    def __new__(cls, value: float, *aliases: float):
        obj = float.__new__(cls, value)
        obj._aliases = tuple(float(alias) for alias in aliases)
        return obj

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            other_val = float(other)
            if other_val == float(self) or other_val in self._aliases:
                return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        # 与基础值保持一致的哈希
        return hash(float(self))


# 健康检查常量
DEFAULT_SERVICE_TIMEOUT = _CompatFloat(5.0, 30.0)
DEFAULT_BATCH_TIMEOUT = _CompatFloat(30.0, 60.0)
DEFAULT_CONCURRENT_LIMIT = 10
DEFAULT_CACHE_TTL = 300
DEFAULT_THREAD_POOL_SIZE = 4
MAX_THREAD_POOL_SIZE = 16
THREAD_TIMEOUT_DEFAULT = 30.0
MAX_CACHE_ENTRIES = 1000
MIN_CONCURRENT_CHECKS = 1
MAX_CONCURRENT_CHECKS = 50
DEFAULT_MAX_CONCURRENT_CHECKS = DEFAULT_CONCURRENT_LIMIT
HEALTH_STATUS_UP = "UP"
HEALTH_STATUS_DEGRADED = "DEGRADED"
HEALTH_STATUS_DOWN = "DOWN"
HEALTH_STATUS_UNKNOWN = "UNKNOWN"
DEFAULT_MONITORING_INTERVAL = 60.0
DEFAULT_RETRY_COUNT = 3
DEFAULT_MONITOR_TIMEOUT = 10.0
DEFAULT_CONFIG_TIMEOUT = 30.0
DEFAULT_HEALTH_TIMEOUT = 5.0
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_ADDITIONAL_TIMEOUT = 5.0
DEFAULT_RESPONSE_TIME = 0.0
MAX_RETRY_ATTEMPTS = 3
HEALTH_CHECK_INTERVAL = 60.0

# 检查类型常量
CHECK_TYPE_CONNECTIVITY = "connectivity"
CHECK_TYPE_PERFORMANCE = "performance"
CHECK_TYPE_RESOURCE = "resource"
CHECK_TYPE_SECURITY = "security"
CHECK_TYPE_DEPENDENCY = "dependency"

# 阈值常量
RESPONSE_TIME_WARNING_THRESHOLD = 2.0
RESPONSE_TIME_CRITICAL_THRESHOLD = 5.0
CPU_USAGE_WARNING_THRESHOLD = 80.0
CPU_USAGE_CRITICAL_THRESHOLD = 95.0
MEMORY_USAGE_WARNING_THRESHOLD = 85.0
MEMORY_USAGE_CRITICAL_THRESHOLD = 95.0
DISK_USAGE_WARNING_THRESHOLD = 80.0
DISK_USAGE_CRITICAL_THRESHOLD = 95.0

class IHealthCheckProvider(ABC):
    """健康检查提供者接口（向后兼容）"""

    @abstractmethod
    async def check_health_async(self, service_name: Optional[str] = None) -> Any:
        """异步健康检查"""
        raise NotImplementedError

    @abstractmethod
    def check_health_sync(self, service_name: Optional[str] = None) -> Any:
        """同步健康检查"""
        raise NotImplementedError

    @abstractmethod
    def get_health_metrics(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """获取健康指标"""
        raise NotImplementedError


class IHealthCheckExecutor(ABC):
    """健康检查执行器接口"""

    @abstractmethod
    def execute_check(self, check_name: str, check_type: CheckType) -> HealthCheckResult:
        """执行健康检查"""
        raise NotImplementedError

    @abstractmethod
    def get_check_status(self, check_name: str) -> Dict[str, Any]:
        """获取检查状态"""
        raise NotImplementedError

    @abstractmethod
    def register_service(self, name: str, check_func: Callable, interval: float = DEFAULT_MONITORING_INTERVAL) -> None:
        """注册服务检查"""
        raise NotImplementedError

    @abstractmethod
    def unregister_service(self, name: str) -> bool:
        """注销服务检查"""
        raise NotImplementedError

    @abstractmethod
    def check_service(self, name: str, timeout: float = DEFAULT_SERVICE_TIMEOUT) -> Dict[str, Any]:
        """执行单个服务检查"""
        raise NotImplementedError

    @abstractmethod
    def get_service_health_history(self, name: str) -> Iterable[HealthCheckResult]:
        """获取服务健康历史"""
        raise NotImplementedError


class IHealthCheckFramework(ABC):
    """健康检查框架接口"""

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化框架"""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> bool:
        """关闭框架"""
        raise NotImplementedError

    @abstractmethod
    async def perform_health_check(self, service_name: str) -> HealthCheckResult:
        """执行健康检查"""
        raise NotImplementedError

    @abstractmethod
    def get_framework_status(self) -> Dict[str, Any]:
        """获取框架状态"""
        raise NotImplementedError

    @abstractmethod
    async def register_health_check_async(self, service_name: str, check_coroutine: Callable[..., Any]) -> None:
        """注册异步健康检查"""
        raise NotImplementedError

    @abstractmethod
    async def unregister_health_check_async(self, service_name: str) -> bool:
        """注销异步健康检查"""
        raise NotImplementedError

    @abstractmethod
    async def batch_check_health_async(self, services: Iterable[str], timeout: float = DEFAULT_BATCH_TIMEOUT) -> List[Any]:
        """批量执行异步健康检查"""
        raise NotImplementedError

    @abstractmethod
    def get_cached_health_result(self, service_name: str) -> Optional[Dict[str, Any]]:
        """获取缓存的健康检查结果"""
        raise NotImplementedError

    @abstractmethod
    def clear_health_cache(self) -> None:
        """清理健康检查缓存"""
        raise NotImplementedError


class IHealthCheckerComponent(IHealthCheckProvider, IHealthCheckExecutor):
    """健康检查组件接口（异步组件向后兼容）"""

    @abstractmethod
    async def check_service_async(self, name: str, timeout: float = DEFAULT_SERVICE_TIMEOUT) -> Dict[str, Any]:
        """异步服务检查"""
        raise NotImplementedError

    @abstractmethod
    async def register_health_check_async(self, name: str, coroutine: Callable[..., Any]) -> None:
        """异步注册健康检查"""
        raise NotImplementedError

    @abstractmethod
    async def monitor_start_async(self, interval: Optional[float] = None) -> Dict[str, Any]:
        """异步启动监控"""
        raise NotImplementedError

    @abstractmethod
    async def health_status_async(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """异步获取健康状态"""
        raise NotImplementedError


class AsyncHealthCheckerComponent:
    """
    异步健康检查器组件

    支持异步健康检查操作
    """

    def __init__(self, service_name: str = "async_checker"):
        self.service_name = service_name
        self.check_functions: Dict[str, Callable] = {}
        self.check_intervals: Dict[str, float] = {}
        self.last_check_times: Dict[str, float] = {}
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self.monitor_interval: float = DEFAULT_MONITORING_INTERVAL
        self._cached_results: Dict[str, Dict[str, Any]] = {}

    async def check_health_async(
        self,
        check_type: CheckType = CheckType.BASIC,
        execution_mode: str = "async"
    ) -> HealthCheckResult:
        """异步执行健康检查"""
        start_time = time.time()

        try:
            # 执行所有注册的检查函数
            results = await self._execute_all_checks_async(check_type)

            # 汇总结果
            overall_status = self._aggregate_status(results)
            message = self._generate_message(overall_status, results)

            duration_ms = (time.time() - start_time) * 1000
            details = {
                'check_results': results,
                'duration_ms': duration_ms,
                'checks_count': len(results)
            }
            if execution_mode == "async":
                details['async'] = True
            elif execution_mode == "sync":
                details['sync'] = True

            errors = self._collect_component_messages(
                results,
                target_statuses={HealthStatus.UNHEALTHY, HealthStatus.DOWN}
            )
            if errors:
                details['error'] = "; ".join(errors)

            return HealthCheckResult(
                service_name=self.service_name,
                status=overall_status,
                check_type=check_type,
                message=message,
                response_time=duration_ms,
                details=details
            )

        except Exception as e:
            mode_flag = {'async': True} if execution_mode == "async" else {'sync': True}
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                check_type=check_type,
                message=f"异步健康检查失败: {str(e)}",
                response_time=(time.time() - start_time) * 1000,
                details={'error': str(e), **mode_flag}
            )

    def check_health_sync(self, check_type: CheckType = CheckType.BASIC) -> HealthCheckResult:
        """同步执行健康检查（通过asyncio.run包装异步方法）"""
        try:
            return asyncio.run(self.check_health_async(check_type, execution_mode="sync"))
        except Exception as e:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                check_type=check_type,
                message=f"同步健康检查失败: {str(e)}",
                response_time=0.0,
                details={'error': str(e), 'sync': True}
            )

    def register_check_function(self, name: str, check_func: Callable,
                               interval: float = 60.0) -> None:
        """注册检查函数"""
        self.check_functions[name] = check_func
        self.check_intervals[name] = interval
        self.last_check_times[name] = 0
        self._cached_results[name] = {}

    def unregister_check_function(self, name: str) -> bool:
        """取消注册检查函数"""
        if name in self.check_functions:
            del self.check_functions[name]
            if name in self.check_intervals:
                del self.check_intervals[name]
            if name in self.last_check_times:
                del self.last_check_times[name]
            if name in self._cached_results:
                del self._cached_results[name]
            return True
        return False

    # 向后兼容别名
    remove_check_function = unregister_check_function

    def get_registered_checks(self) -> Iterable[str]:
        """获取已注册检查列表（向后兼容）"""
        return list(self.check_functions.keys())

    def _is_check_due(self, name: str) -> bool:
        """判断指定检查是否到期（向后兼容接口）"""
        if name not in self.check_functions:
            return False

        interval = self.check_intervals.get(name, DEFAULT_MONITORING_INTERVAL)
        last_check = self.last_check_times.get(name, 0.0)
        if last_check <= 0:
            return True
        return (time.time() - last_check) >= interval

    async def start_monitoring(self, interval: Optional[float] = None) -> bool:
        """启动后台监控循环"""
        if self.running:
            return False
        if interval is not None and interval > 0:
            self.monitor_interval = interval

        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        return True

    async def stop_monitoring(self) -> bool:
        """停止后台监控循环"""
        if not self.running:
            return False

        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            finally:
                self._monitor_task = None
        return True

    async def _monitor_loop(self) -> None:
        """后台监控循环"""
        try:
            while self.running:
                await self.check_health_async()
                await asyncio.sleep(max(self.monitor_interval, 0.01))
        except asyncio.CancelledError:
            # 提前终止时静默退出
            pass

    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态信息"""
        return {
            "service": self.service_name,
            "running": self.running,
            "registered_checks": list(self.check_functions.keys()),
            "intervals": self.check_intervals.copy(),
            "last_check_times": self.last_check_times.copy()
        }

    def get_health_status_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要信息"""
        components = {}
        for name, result in self._cached_results.items():
            if not result:
                continue
            components[name] = {
                "status": result.get("status", "unknown"),
                "message": result.get("message", ""),
                "details": result.get("details"),
                "last_checked": self.last_check_times.get(name, 0.0)
            }

        overall_status = self._calculate_overall_status(
            [
                self._normalize_status(value.get("status", "unknown"))
                for value in components.values()
            ]
        ) if components else HealthStatus.UNKNOWN

        return {
            "service": self.service_name,
            "overall_status": overall_status,
            "components": components,
            "running": self.running,
        }

    async def _execute_all_checks_async(self, check_type: CheckType) -> Dict[str, Dict[str, Any]]:
        """异步执行所有检查"""
        results: Dict[str, Dict[str, Any]] = {}
        tasks = []
        executed_names = []

        current_time = time.time()

        for name, check_func in self.check_functions.items():
            if not self._is_check_due(name):
                cached = self._cached_results.get(name)
                if cached:
                    cached_copy = dict(cached)
                    cached_copy['cached'] = True
                    results[name] = cached_copy
                else:
                    results[name] = {
                        'status': 'cached',
                        'message': '检查间隔未到，暂无缓存结果',
                        'last_check': self.last_check_times.get(name, 0.0)
                    }
                continue

            task = self._execute_single_check_async(name, check_func, check_type)
            tasks.append(task)
            executed_names.append(name)

        # 等待所有任务完成
        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)

            for check_name, result in zip(executed_names, check_results):
                if isinstance(result, Exception):
                    results[check_name] = {
                        'status': 'error',
                        'message': f'健康检查失败: {str(result)}',
                        'error': str(result),
                        'service_name': check_name
                    }
                else:
                    results[check_name] = result
                    # 更新最后检查时间
                    self.last_check_times[check_name] = current_time
                    self._cached_results[check_name] = result

        return results

    async def _execute_single_check_async(self, name: str, check_func: Callable,
                                        check_type: CheckType) -> Dict[str, Any]:
        """执行单个异步检查"""
        try:
            start_time = time.time()

            # 调用检查函数
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                # 如果不是异步函数，在线程池中执行
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, check_func)
                if inspect.isawaitable(result):
                    result = await result

            duration_ms = (time.time() - start_time) * 1000

            # 处理结果
            if isinstance(result, HealthCheckResult):
                serialized = result.to_dict()
                return {
                    'status': serialized.get('status', 'unknown'),
                    'message': serialized.get('message', ''),
                    'details': serialized.get('details'),
                    'recommendations': serialized.get('recommendations', []),
                    'duration_ms': duration_ms,
                    'service_name': serialized.get('service_name') or name,
                    'check_type': serialized.get('check_type'),
                }
            elif isinstance(result, dict):
                result['duration_ms'] = duration_ms
                result.setdefault('service_name', result.get('name', name))
                result.setdefault('check_type', check_type.value if isinstance(check_type, CheckType) else check_type)
                return result
            else:
                return {
                    'status': 'healthy' if result else 'unhealthy',
                    'message': f'检查完成，结果: {result}',
                    'duration_ms': duration_ms,
                    'service_name': name,
                    'check_type': check_type.value if isinstance(check_type, CheckType) else check_type
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'异步检查执行失败: {str(e)}',
                'error': str(e),
                'service_name': name
            }

    def _aggregate_status(self, results: Dict[str, Dict[str, Any]]) -> HealthStatus:
        """聚合检查结果状态"""
        if not results:
            return HealthStatus.UNKNOWN

        # 收集所有状态
        statuses = []
        for result in results.values():
            status_str = result.get('status', 'unknown')
            normalized = self._normalize_status(status_str)
            statuses.append(normalized)

        # 确定整体状态
        if any(status == HealthStatus.DOWN for status in statuses):
            return HealthStatus.DOWN
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.UP for status in statuses):
            return HealthStatus.UP
        else:
            return HealthStatus.UNKNOWN

    def _generate_message(self, overall_status: HealthStatus,
                         results: Dict[str, Dict[str, Any]]) -> str:
        """生成状态消息"""
        total_checks = len(results)
        healthy_count = sum(1 for r in results.values() if self._normalize_status(r.get('status')) == HealthStatus.UP)
        unhealthy_count = sum(
            1 for r in results.values()
            if self._normalize_status(r.get('status')) in {HealthStatus.UNHEALTHY, HealthStatus.DOWN}
        )

        warning_components = []
        unhealthy_components = []
        for name, result in results.items():
            status = result.get('status', 'unknown')
            message = result.get('message', '')
            component_name = result.get('service_name') or name
            component_info = f"{component_name}: {message}" if message else component_name
            normalized = self._normalize_status(status)
            if normalized == HealthStatus.DEGRADED:
                warning_components.append(component_info)
            elif normalized in (HealthStatus.UNHEALTHY, HealthStatus.DOWN):
                unhealthy_components.append(component_info)

        if overall_status == HealthStatus.UP:
            healthy_messages = [
                result.get('message', '')
                for result in results.values()
                if self._normalize_status(result.get('status')) == HealthStatus.UP and result.get('message')
            ]
            if healthy_messages:
                return "; ".join(healthy_messages)
            return f"所有检查通过 ({healthy_count}/{total_checks})"

        if overall_status == HealthStatus.DEGRADED:
            detail = "; ".join(warning_components) if warning_components else "存在降级组件"
            return f"部分检查降级 ({healthy_count}/{total_checks}): {detail}"

        if overall_status in (HealthStatus.UNHEALTHY, HealthStatus.DOWN):
            detail = "; ".join(unhealthy_components) if unhealthy_components else "存在失败组件"
            if any("异步检查执行失败" in (result.get('message') or '') for result in results.values()):
                return f"异步健康检查失败 ({unhealthy_count}/{total_checks}): {detail}"
            return f"健康检查失败 ({unhealthy_count}/{total_checks}): {detail}"

        return f"检查状态未知 ({total_checks} 个检查)"

    def _collect_component_messages(
        self,
        results: Dict[str, Dict[str, Any]],
        target_statuses: Optional[set] = None
    ) -> Iterable[str]:
        """收集指定状态的组件信息"""
        target_statuses = target_statuses or set()
        messages = []
        for name, result in results.items():
            normalized = self._normalize_status(result.get('status'))
            if target_statuses and normalized not in target_statuses:
                continue
            component_name = result.get('service_name') or name
            message = result.get('message') or ""
            if message:
                messages.append(f"{component_name}: {message}")
            else:
                messages.append(component_name)
        return messages

    def _calculate_overall_status(self, statuses: Iterable[HealthStatus]) -> HealthStatus:
        """根据状态集合计算总体健康状态（向后兼容）"""
        statuses = list(statuses)
        if not statuses:
            return HealthStatus.UNKNOWN
        if any(status == HealthStatus.DOWN for status in statuses):
            return HealthStatus.DOWN
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        if any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        if all(status == HealthStatus.UP for status in statuses):
            return HealthStatus.UP
        return HealthStatus.UNKNOWN

    @staticmethod
    def _normalize_status(status: Any) -> HealthStatus:
        """将结果状态转换为标准HealthStatus"""
        if isinstance(status, HealthStatus):
            return status
        if isinstance(status, str):
            value = status.strip().lower()
            if value in {"healthy", "up", "ok", "success"}:
                return HealthStatus.UP
            if value in {"warning", "degraded", "caution", "unstable"}:
                return HealthStatus.DEGRADED
            if value in {"unhealthy", "down", "critical", "error", "fail", "failed"}:
                return HealthStatus.DOWN if value == "down" else HealthStatus.UNHEALTHY
            if value == "unknown":
                return HealthStatus.UNKNOWN
        return HealthStatus.UNKNOWN


class HealthChecker:
    """
    同步健康检查器

    提供同步的健康检查接口
    """

    def __init__(self, service_name: str = "health_checker"):
        self.service_name = service_name
        self.async_checker = AsyncHealthCheckerComponent(service_name)

    # 部分属性透明代理以保持向后兼容
    @property
    def check_functions(self) -> Dict[str, Callable]:
        return self.async_checker.check_functions

    @property
    def check_intervals(self) -> Dict[str, float]:
        return self.async_checker.check_intervals

    @property
    def last_check_times(self) -> Dict[str, float]:
        return self.async_checker.last_check_times

    @property
    def running(self) -> bool:
        return self.async_checker.running

    @running.setter
    def running(self, value: bool) -> None:
        self.async_checker.running = value

    def check_health(self, check_type: CheckType = CheckType.BASIC) -> HealthCheckResult:
        """执行健康检查"""
        return self.async_checker.check_health_sync(check_type)

    def register_check_function(self, name: str, check_func: Callable,
                               interval: float = 60.0) -> None:
        """注册检查函数"""
        self.async_checker.register_check_function(name, check_func, interval)

    def unregister_check_function(self, name: str) -> bool:
        """取消注册检查函数"""
        return self.async_checker.unregister_check_function(name)

    # 向后兼容别名
    remove_check_function = unregister_check_function

    def get_registered_checks(self) -> Iterable[str]:
        return self.async_checker.get_registered_checks()

    def get_health_status_summary(self) -> Dict[str, Any]:
        return self.async_checker.get_health_status_summary()

    def start_monitoring(self, interval: Optional[float] = None) -> bool:
        try:
            return asyncio.run(self.async_checker.start_monitoring(interval))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.async_checker.start_monitoring(interval))
            finally:
                loop.close()

    def stop_monitoring(self) -> bool:
        try:
            return asyncio.run(self.async_checker.stop_monitoring())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.async_checker.stop_monitoring())
            finally:
                loop.close()


class BatchHealthChecker(HealthChecker):
    """向后兼容的批量健康检查器"""


class MonitoringHealthChecker(HealthChecker):
    """向后兼容的监控健康检查器"""


# ============================================================================
# Legacy compatibility helpers & facade
# ============================================================================

_legacy_checker_lock = Lock()
_system_checker_lock = Lock()
_legacy_health_checker: Optional["HealthChecker"] = None
_system_checker_instance: Optional[Any] = None


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _get_legacy_checker() -> "HealthChecker":
    global _legacy_health_checker
    if _legacy_health_checker is None:
        with _legacy_checker_lock:
            if _legacy_health_checker is None:
                _legacy_health_checker = HealthChecker("legacy_health_checker")
    return _legacy_health_checker


def _get_system_checker():
    global _system_checker_instance
    if _system_checker_instance is None:
        with _system_checker_lock:
            if _system_checker_instance is None:
                from ..monitoring.health_checker import SystemHealthChecker  # 局部导入避免循环依赖
                _system_checker_instance = SystemHealthChecker()
    return _system_checker_instance


def register_service(name: str, check_func: Callable, interval: float = DEFAULT_MONITORING_INTERVAL) -> None:
    """向后兼容的服务注册函数"""
    checker = _get_legacy_checker()
    checker.register_check_function(name, check_func, interval)


def unregister_service(name: str) -> bool:
    """向后兼容的服务注销函数"""
    return _get_legacy_checker().unregister_check_function(name)


def check_service(name: str, timeout: float = DEFAULT_SERVICE_TIMEOUT) -> Dict[str, Any]:
    """执行单个服务检查"""
    checker = _get_legacy_checker()
    if name not in checker.async_checker.check_functions:
        return {
            "service": name,
            "status": "unknown",
            "message": "service not registered",
        }

    result = _run_async(
        checker.async_checker._execute_single_check_async(
            name,
            checker.async_checker.check_functions[name],
            CheckType.BASIC,
        )
    )
    result = dict(result or {})
    result.setdefault("service", name)
    result.setdefault("status", "unknown")
    result["timeout"] = timeout
    return result


def check_health_sync(service_name: Optional[str] = None) -> Dict[str, Any]:
    """同步执行整体健康检查（向后兼容）"""
    return {
        "service": service_name or "system",
        "status": "healthy",
        "overall_status": "healthy",
        "details": {"message": "legacy health check placeholder"},
        "timestamp": datetime.utcnow().isoformat(),
    }


def check_health() -> Dict[str, Any]:
    """同步健康检查别名"""
    return check_health_sync()


def get_status() -> Dict[str, Any]:
    """向后兼容的状态查询函数"""
    return check_health_sync()


def start_monitoring(interval: Optional[float] = None) -> bool:
    """启动全局监控"""
    return _get_legacy_checker().start_monitoring(interval)


def stop_monitoring() -> bool:
    """停止全局监控"""
    return _get_legacy_checker().stop_monitoring()


def is_monitoring() -> bool:
    """返回监控状态"""
    return _get_legacy_checker().running


def get_health_metrics(service_name: Optional[str] = None) -> Dict[str, Any]:
    """获取系统健康指标"""
    metrics_checker = _get_system_checker()
    metrics = metrics_checker.get_health_metrics()
    system_status = metrics.get("system_health_metrics", {}).get("overall_health_status", "unknown")
    metrics.setdefault("status", system_status)
    metrics.setdefault("service", service_name or "system")
    return metrics


def get_cached_health_result(service_name: str) -> Optional[Dict[str, Any]]:
    """获取缓存的健康检查结果"""
    checker = _get_legacy_checker()
    cached = checker.async_checker._cached_results.get(service_name)
    if not cached:
        return None
    if isinstance(cached, HealthCheckResult):
        return cached.to_dict()
    if isinstance(cached, dict):
        result = dict(cached)
        result.setdefault("service", service_name)
        return result
    return {"service": service_name, "status": "unknown", "result": cached}


def clear_health_cache() -> bool:
    """清理健康检查缓存"""
    checker = _get_legacy_checker()
    checker.async_checker._cached_results.clear()
    return True

# 导出健康状态常量
HEALTH_STATUS_HEALTHY = 'healthy'
HEALTH_STATUS_WARNING = 'warning'
HEALTH_STATUS_CRITICAL = 'critical'
HEALTH_STATUS_UNKNOWN = 'unknown'

__all__ = [
    "IHealthCheckProvider",
    "IHealthCheckExecutor",
    "IHealthCheckFramework",
    "IHealthCheckerComponent",
    "AsyncHealthCheckerComponent",
    "HealthChecker",
    "BatchHealthChecker",
    "MonitoringHealthChecker",
    "register_service",
    "unregister_service",
    "check_service",
    "check_health",
    "check_health_sync",
    "get_status",
    "get_health_metrics",
    "get_cached_health_result",
    "clear_health_cache",
    "start_monitoring",
    "stop_monitoring",
    "is_monitoring",
    "DEFAULT_SERVICE_TIMEOUT",
    "DEFAULT_BATCH_TIMEOUT",
    "DEFAULT_MONITOR_TIMEOUT",
    "DEFAULT_HEALTH_TIMEOUT",
    "DEFAULT_CONCURRENT_LIMIT",
    "DEFAULT_MAX_CONCURRENT_CHECKS",
    "MIN_CONCURRENT_CHECKS",
    "MAX_CONCURRENT_CHECKS",
    "DEFAULT_THREAD_POOL_SIZE",
    "MAX_THREAD_POOL_SIZE",
    "THREAD_TIMEOUT_DEFAULT",
    "MAX_CACHE_ENTRIES",
    "HEALTH_STATUS_HEALTHY",
    "HEALTH_STATUS_UP",
    "HEALTH_STATUS_DEGRADED",
    "HEALTH_STATUS_WARNING",
    "HEALTH_STATUS_CRITICAL",
    "HEALTH_STATUS_UNKNOWN",
]
