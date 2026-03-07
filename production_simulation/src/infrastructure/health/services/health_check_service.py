"""
健康检查服务

提供统一的健康检查服务接口，同时保留老版本测试依赖的属性和方法。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil


class HealthStatus(Enum):
    """健康状态枚举"""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """检查类型枚举"""

    BASIC = "basic"
    DETAILED = "detailed"
    PERFORMANCE = "performance"
    DEPENDENCY = "dependency"


class HealthCheckResult:
    """健康检查结果"""

    def __init__(
        self,
        service_name: str,
        status: HealthStatus,
        check_type: CheckType = CheckType.BASIC,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.service_name = service_name
        self.status = status
        self.check_type = check_type
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "check_type": self.check_type.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class _DependencyRegistry:
    """轻量级依赖检查注册器，向后兼容旧实现"""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, func: Callable[[], Any], config: Optional[Dict[str, Any]] = None) -> None:
        self._registry[name] = {"check": func, "config": config or {}, "registered_at": datetime.now()}

    def unregister(self, name: str) -> None:
        self._registry.pop(name, None)

    def iter_dependencies(self) -> List[Dict[str, Any]]:
        return [
            {"name": name, "check": entry["check"], "config": entry["config"], "registered_at": entry["registered_at"]}
            for name, entry in self._registry.items()
        ]


class HealthCheck:
    """
    健康检查服务

    提供统一的健康检查功能，并补充测试所依赖的元数据接口。
    """

    def __init__(self) -> None:
        self.checkers: Dict[str, Any] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}

        self.router: Dict[str, Any] = {"routes": [], "metadata": {}, "created_at": datetime.now().isoformat()}
        self.dependencies: List[Dict[str, Any]] = []
        self._dependency_checker: Optional[_DependencyRegistry] = _DependencyRegistry()
        self._system_health_checker = self._create_system_health_checker()

        self._initialized: bool = False
        self._check_count: int = 0
        self._last_check_time: Optional[datetime] = None
        self._config: Dict[str, Any] = {}
        self._health_history: List[Dict[str, Any]] = []
        self._metrics: Dict[str, Any] = defaultdict(int)

    # ------------------------------------------------------------------
    # 初始化与配置信息
    # ------------------------------------------------------------------
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化健康检查服务"""
        self._config = dict(config or {})
        self._initialized = True
        self._metrics["initializations"] = self._metrics.get("initializations", 0) + 1
        if "timeout" in self._config:
            self._config["timeout"] = max(0, self._config["timeout"])
        if "retries" in self._config:
            self._config["retries"] = max(0, self._config["retries"])
        return True

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件元信息"""
        return {
            "component_type": "health_check_service",
            "initialized": self._initialized,
            "registered_dependencies": len(self.dependencies),
            "router_routes": len(self.router.get("routes", [])),
            "supports_async": True,
        }

    def is_healthy(self) -> bool:
        """快速健康状态判定，现阶段仅依赖初始化状态"""
        return self._initialized

    def get_metrics(self) -> Dict[str, Any]:
        """返回健康检查指标"""
        return {
            "component_metrics": {
                "check_count": self._check_count,
                "dependency_count": len(self.dependencies),
                "initialized": self._initialized,
            },
            "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
            "configuration": self._config.copy(),
        }

    def cleanup(self) -> bool:
        """清理资源"""
        self.dependencies.clear()
        if self._dependency_checker:
            self._dependency_checker = _DependencyRegistry()
        return True

    # ------------------------------------------------------------------
    # 依赖管理
    # ------------------------------------------------------------------
    def add_dependency_check(
        self,
        name: str,
        check: Callable[[], Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """注册依赖检查"""
        dependency_entry = {
            "name": name,
            "check": check,
            "config": dict(config or {}),
            "registered_at": datetime.now(),
        }
        self.dependencies.append(dependency_entry)

        if self._dependency_checker is not None:
            self._dependency_checker.register(name, check, config)

        return True

    def _get_empty_dependencies_result(self) -> Dict[str, Any]:
        return {"dependency_count": 0, "dependency_results": [], "status": "unknown"}

    def _check_all_dependencies(self, deps: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for entry in deps:
            name = entry.get("name", "unnamed_dependency")
            check_callable = entry.get("check")
            results[name] = self._check_single_dependency({"name": name, "check": check_callable, **entry})
        return results

    def _check_single_dependency(self, dependency: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个依赖检查，保持向后兼容"""
        name = dependency.get("name", "unnamed_dependency")
        check_callable = dependency.get("check")
        metadata = {k: v for k, v in dependency.items() if k not in {"name", "check"}}

        try:
            outcome = check_callable() if callable(check_callable) else {"status": "unknown"}
            if isinstance(outcome, dict):
                status = outcome.get("status", "unknown")
                details = outcome
            else:
                status = "healthy"
                details = {"result": outcome}
            return {"name": name, "status": status, "details": details, "metadata": metadata}
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "name": name,
                "status": "unhealthy",
                "error": str(exc),
                "metadata": metadata,
            }

    def _evaluate_dependencies_health(self, dependency_results: Dict[str, Dict[str, Any]]) -> bool:
        """评估依赖健康状况"""
        if not dependency_results:
            return True
        healthy_statuses = {"healthy", "ok", "success"}
        warning_statuses = {"warning", "degraded"}
        critical_statuses = {"unhealthy", "critical", "error", "failed"}

        has_warning = False
        for result in dependency_results.values():
            status = str(result.get("status", "")).lower()
            if status in critical_statuses:
                return False
            if status not in healthy_statuses:
                if status in warning_statuses:
                    has_warning = True
                else:
                    has_warning = True
        return not has_warning

    # ------------------------------------------------------------------
    # 系统健康与汇总
    # ------------------------------------------------------------------
    def _create_system_health_checker(self) -> Optional[Any]:
        try:
            from src.infrastructure.health.monitoring.health_checker import SystemHealthChecker

            return SystemHealthChecker()
        except Exception:
            return None

    def _get_system_health(self) -> Dict[str, Any]:
        """采集基础系统资源指标"""
        def _safe_call(func: Callable[[], Any], default: Any) -> Any:
            try:
                return func()
            except Exception:
                return default

        cpu_percent = _safe_call(lambda: float(psutil.cpu_percent(interval=0.05)), 0.0)
        memory = _safe_call(psutil.virtual_memory, None)
        disk = _safe_call(lambda: psutil.disk_usage("/"), None)

        memory_percent = getattr(memory, "percent", 0.0) if memory is not None else 0.0
        disk_percent = getattr(disk, "percent", 0.0) if disk is not None else 0.0

        def _status_from_percent(value: float, warning: float, critical: float) -> str:
            if value >= critical:
                return "critical"
            if value >= warning:
                return "warning"
            return "healthy"

        cpu_status = _status_from_percent(cpu_percent, 85.0, 95.0)
        memory_status = _status_from_percent(memory_percent, 85.0, 95.0)
        disk_status = _status_from_percent(disk_percent, 85.0, 95.0)

        statuses = [cpu_status, memory_status, disk_status]
        overall_status = "healthy"
        if "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses:
            overall_status = "warning"

        return {
            "overall_status": overall_status,
            "cpu": {"percent": cpu_percent, "status": cpu_status},
            "memory": {"percent": memory_percent, "status": memory_status},
            "disk": {"percent": disk_percent, "status": disk_status},
        }

    def check_system_health_status(self) -> Dict[str, Any]:
        """对外暴露的系统健康状态"""
        system_metrics = self._get_system_health()
        system_metrics["checked_at"] = datetime.now().isoformat()
        return system_metrics

    def check_router_health(self) -> Dict[str, Any]:
        """校验路由配置"""
        return {
            "routes_count": len(self.router.get("routes", [])),
            "metadata_keys": list(self.router.get("metadata", {}).keys()),
            "status": "healthy" if self.router.get("routes") is not None else "unknown",
        }

    def check_dependencies_health(self) -> Dict[str, Any]:
        """执行依赖健康检查"""
        all_dependencies = self.dependencies[:]
        if self._dependency_checker is not None:
            for entry in self._dependency_checker.iter_dependencies():
                if entry not in all_dependencies:
                    all_dependencies.append(entry)

        if not all_dependencies:
            return self._get_empty_dependencies_result()

        results = self._check_all_dependencies(all_dependencies)
        statuses = [res.get("status", "unknown") for res in results.values()]
        overall_status = "healthy"
        if any(status == "unhealthy" for status in statuses):
            overall_status = "unhealthy"
        elif any(status == "degraded" for status in statuses):
            overall_status = "degraded"
        elif any(status == "warning" for status in statuses):
            overall_status = "warning"

        return {
            "dependency_results": results,
            "dependency_count": len(all_dependencies),
            "status": overall_status,
        }

    def check_initialization_health(self) -> Dict[str, Any]:
        """初始化状况检查"""
        return {
            "initialized": self._initialized,
            "config_applied": bool(self._config),
            "status": "healthy" if self._initialized else "warning",
        }

    def _validate_health_check_initialization(self) -> Dict[str, Any]:
        status = "passed" if self._initialized else "pending"
        return {"validation": "initialization", "status": status}

    def _validate_router_configuration(self) -> Dict[str, Any]:
        has_routes = isinstance(self.router.get("routes"), list)
        return {"validation": "router_configuration", "status": "passed" if has_routes else "warning"}

    def _validate_dependencies_configuration(self) -> Dict[str, Any]:
        return {
            "validation": "dependencies_configuration",
            "status": "passed" if self.dependencies else "warning",
            "dependency_count": len(self.dependencies),
        }

    def _validate_system_monitoring(self) -> Dict[str, Any]:
        return {
            "validation": "system_monitoring",
            "status": "passed" if self._system_health_checker is not None else "warning",
        }

    def validate_health_check_config(self) -> Dict[str, Any]:
        """配置校验"""
        issues: List[str] = []
        timeout = self._config.get("timeout")
        retries = self._config.get("retries")
        if timeout is not None and timeout < 0:
            issues.append("timeout must be non-negative")
        if retries is not None and retries < 0:
            issues.append("retries must be non-negative")
        return {"status": "passed" if not issues else "warning", "issues": issues, "config": self._config.copy()}

    def health_status(self) -> Dict[str, Any]:
        """综合健康状态"""
        initialized_status = self.check_initialization_health()
        system_status = self.check_system_health_status()
        dependency_status = self.check_dependencies_health()
        statuses = [initialized_status.get("status"), system_status.get("overall_status"), dependency_status.get("status")]
        overall = "healthy"
        if "critical" in statuses or "unhealthy" in statuses:
            overall = "critical"
        elif "warning" in statuses or "degraded" in statuses:
            overall = "warning"
        return {
            "overall_status": overall,
            "initialized": initialized_status,
            "system": system_status,
            "dependencies": dependency_status,
        }

    def health_summary(self) -> Dict[str, Any]:
        """健康摘要"""
        summary = self.health_status()
        summary.update(
            {
                "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
                "check_count": self._check_count,
            }
        )
        return summary

    def monitor_health_check_service(self) -> Dict[str, Any]:
        """监控健康检查服务"""
        return {
            "summary": self.health_summary(),
            "metrics": self.get_metrics(),
            "history_length": len(self._health_history),
        }

    # ------------------------------------------------------------------
    # 核心检查执行
    # ------------------------------------------------------------------
    def register_checker(self, name: str, checker: Any) -> None:
        """注册检查器"""
        self.checkers[name] = checker

    def unregister_checker(self, name: str) -> bool:
        """取消注册检查器"""
        if name in self.checkers:
            del self.checkers[name]
            self.last_results.pop(name, None)
            return True
        return False

    def perform_check(self, name: str, check_type: CheckType = CheckType.BASIC) -> HealthCheckResult:
        """执行单项健康检查"""
        if name not in self.checkers:
            return HealthCheckResult(
                service_name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Checker '{name}' not registered",
            )

        checker = self.checkers[name]
        try:
            if hasattr(checker, "check_health"):
                result = checker.check_health(check_type)
                if isinstance(result, HealthCheckResult):
                    self.last_results[name] = result
                    return result
            health_result = HealthCheckResult(
                service_name=name,
                status=HealthStatus.HEALTHY,
                check_type=check_type,
                message="Basic health check passed",
            )
            self.last_results[name] = health_result
            return health_result
        except Exception as exc:  # pragma: no cover
            result = HealthCheckResult(
                service_name=name,
                status=HealthStatus.UNHEALTHY,
                check_type=check_type,
                message=f"Health check failed: {exc}",
            )
            self.last_results[name] = result
            return result

    def perform_all_checks(self, check_type: CheckType = CheckType.BASIC) -> Dict[str, HealthCheckResult]:
        """执行所有注册检查器"""
        return {name: self.perform_check(name, check_type) for name in self.checkers}

    def get_last_result(self, name: str) -> Optional[HealthCheckResult]:
        """获取上次检查结果"""
        return self.last_results.get(name)

    def get_all_last_results(self) -> Dict[str, HealthCheckResult]:
        """获取所有上次检查结果"""
        return self.last_results.copy()

    def get_overall_status(self) -> HealthStatus:
        """获取整体健康状态"""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        statuses = [result.status for result in self.last_results.values()]
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        if any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    def get_registered_checkers(self) -> List[str]:
        """获取已注册的检查器列表"""
        return list(self.checkers.keys())

    def check_health(self) -> Dict[str, Any]:
        """综合健康检查入口"""
        self._check_count += 1
        self._last_check_time = datetime.now()

        result = {
            "timestamp": self._last_check_time.isoformat(),
            "initialized": self._initialized,
            "system": self.check_system_health_status(),
            "dependencies": self.check_dependencies_health(),
            "router": self.check_router_health(),
        }
        result["status"] = self.health_status().get("overall_status", "unknown")
        self._health_history.append(result)
        return result

