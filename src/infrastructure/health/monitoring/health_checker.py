"""
health_checker 模块

提供 health_checker 相关功能和接口。
"""

import logging

import psutil
import time
import asyncio
import time
import threading
# 导入统一健康检查接口

from ..components.health_checker import IHealthCheckExecutor, HealthCheckResult, CheckType
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional
from numbers import Number
"""
系统健康检查器模块

根据Phase 8.2职责边界梳理，该模块为Implementation Layer（实现层），
专注于具体的系统健康检查逻辑实现，不涉及框架管理功能。

职责范围：
- 提供系统级健康检查的具体实现
- 管理系统指标相关的健康检查
- 实现标准化的健康检查接口
- 不涉及并发控制、缓存管理等框架功能
"""

logger = logging.getLogger(__name__)


class SystemHealthChecker(IHealthCheckExecutor):
    """
    系统健康检查器 (Implementation Layer)

    根据Phase 8.2职责边界梳理，重构为系统级健康检查的具体实现组件。
    专注于系统指标相关的健康检查逻辑，不涉及框架管理。

    职责范围：
    - 实现系统级健康检查的具体算法
    - 管理系统指标相关的健康状态评估
    - 提供标准化的健康检查结果
    - 不涉及并发控制、缓存等框架功能

    健康检查类型：
    - CPU使用率检查
    - 内存使用率检查
    - 磁盘空间检查
    - 网络连接检查
    - 进程健康检查
    """

    def __init__(self, metrics_collector=None):
        """
        初始化系统健康检查器

        Args:
            metrics_collector: 系统指标收集器实例，用于获取系统指标数据
        """
        self.metrics_collector = metrics_collector

        # 实现层状态管理
        self._health_checks: Dict[str, Callable] = {}  # 服务名称 -> 检查函数映射
        self._check_history: Dict[str, List[HealthCheckResult]] = {}  # 检查历史记录
        self._max_history_size = 100
        # 向后兼容的属性名称
        self._service_checks = self._health_checks
        self._service_history = self._check_history
        self._health_history = self._check_history

        # 默认系统健康检查
        self._register_system_checks()

        logger.info("系统健康检查器初始化完成")

    # =========================================================================
    # IHealthCheckExecutor 接口实现
    # =========================================================================

    def register_service(self, name: str, check_func: Callable) -> None:
        """注册服务检查函数"""
        if not callable(check_func):
            raise ValueError(f"检查函数 {name} 必须是可调用的")

        self._health_checks[name] = check_func
        self._check_history[name] = []
        logger.info(f"注册系统健康检查服务: {name}")

    def unregister_service(self, name: str) -> None:
        """注销服务检查函数"""
        if name in self._health_checks:
            del self._health_checks[name]
            if name in self._check_history:
                del self._check_history[name]
            logger.info(f"注销系统健康检查服务: {name}")

    def check_service(self, name: str, timeout: float = 5.0) -> HealthCheckResult:
        """检查单个服务"""
        start_time = time.time()

        try:
            # 验证服务是否存在
            if name not in self._health_checks:
                elapsed = max(time.time() - start_time, 1e-6)
                return HealthCheckResult(
                    service_name=name,
                    status="unhealthy",
                    message=f"Service {name} not registered",
                    timestamp=datetime.now(),
                    response_time=elapsed,
                    details={
                        "error": f"服务 {name} 未注册",
                        "error_type": "not_registered"
                    },
                    recommendations=["先注册该服务"]
                )

            # 执行健康检查
            check_func = self._health_checks[name]
            result_container: Dict[str, Any] = {}
            error_container: Dict[str, Exception] = {}

            def _execute_check():
                try:
                    result_container["result"] = check_func()
                except Exception as exec_error:
                    error_container["error"] = exec_error

            worker = threading.Thread(target=_execute_check, daemon=True)
            worker.start()
            worker.join(timeout)

            if worker.is_alive():
                elapsed = max(time.time() - start_time, 1e-6)
                timeout_result = HealthCheckResult(
                    service_name=name,
                    status="unhealthy",
                    message=f"Service {name} check timeout",
                    timestamp=datetime.now(),
                    response_time=elapsed,
                    details={
                        "error": "timeout",
                        "timeout_seconds": timeout
                    },
                    recommendations=["检查服务性能", "考虑延长超时设置"]
                )
                self._record_check_result(name, timeout_result)
                return timeout_result

            if "error" in error_container:
                raise error_container["error"]

            result = result_container.get("result", {})
            elapsed = time.time() - start_time
            response_time = result.get("response_time", elapsed) if isinstance(result, dict) else elapsed

            # 转换结果格式
            health_result = HealthCheckResult(
                service_name=name,
                status=result.get("status", "unknown"),
                message=result.get("message", f"{name} 状态未知"),
                timestamp=datetime.now(),
                response_time=response_time,
                details=self._build_result_details(result),
                recommendations=self._generate_recommendations(result)
            )

            # 记录检查历史
            self._record_check_result(name, health_result)

            return health_result

        except Exception as e:
            logger.error(f"检查服务 {name} 失败: {e}")
            elapsed = max(time.time() - start_time, 1e-6)
            error_result = HealthCheckResult(
                service_name=name,
                status="unhealthy",
                message=str(e),
                timestamp=datetime.now(),
                response_time=elapsed,
                details={
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                recommendations=["检查服务配置", "查看错误日志"]
            )
            self._record_check_result(name, error_result)
            return error_result

    def get_service_health_history(self, name: str) -> List[HealthCheckResult]:
        """获取服务健康检查历史"""
        history = self._check_history.get(name, [])
        if not history:
            return []
        return list(reversed(history))

    def _record_check_result(self, service_name: str, result: HealthCheckResult):
        """记录检查结果到历史"""
        if service_name not in self._check_history:
            self._check_history[service_name] = []

        # 保留最近100条记录
        history = self._check_history[service_name]
        history.append(result)
        if len(history) > 100:
            history.pop(0)

    def _generate_recommendations(self, check_result: Dict[str, Any]) -> List[str]:
        """根据检查结果生成建议"""
        status = check_result.get("status", "unknown")
        recommendations = []

        if status == "critical":
            recommendations.extend([
                "立即检查系统状态",
                "考虑重启相关服务",
                "联系系统管理员"
            ])
        elif status == "warning":
            recommendations.extend([
                "监控系统指标变化",
                "检查资源使用趋势",
                "考虑优化系统配置"
            ])
        elif status == "error":
            recommendations.extend([
                "检查配置参数",
                "验证依赖服务状态",
                "查看详细错误日志"
            ])

        cpu_percent = check_result.get("cpu_percent")
        if isinstance(cpu_percent, (int, float)) and cpu_percent >= 85:
            recommendations.append(f"CPU 利用率过高 ({cpu_percent:.1f}%)，请优化CPU使用或扩容。")

        memory_percent = check_result.get("memory_percent")
        if isinstance(memory_percent, (int, float)) and memory_percent >= 80:
            recommendations.append(f"内存使用率偏高 ({memory_percent:.1f}%)，建议检查内存泄漏或扩容。")

        return recommendations

    @staticmethod
    def _build_result_details(result: Any) -> Dict[str, Any]:
        """构建健康检查详情"""
        if isinstance(result, dict):
            details_data: Dict[str, Any] = {}
            nested_details = result.get("details")
            if isinstance(nested_details, dict):
                details_data.update(nested_details)
            else:
                details_data.update({k: v for k, v in result.items() if k != "details"})

            for key in ("status", "message", "response_time"):
                value = result.get(key)
                if key not in details_data and value is not None:
                    details_data[key] = value
            if "raw_result" not in details_data:
                details_data["raw_result"] = result
            return details_data
        return {"raw_result": result}

    # =========================================================================
    # IHealthCheckProvider 接口实现
    # =========================================================================

    async def check_health_async(self) -> Dict[str, Any]:
        """异步执行健康检查"""
        start_time = time.time()

        results = {name: self.check_service(name) for name in self._health_checks.keys()}
        overall_status = self._calculate_overall_status(results)

        friendly_names = {
            "cpu_usage": "cpu",
            "memory_usage": "memory",
            "disk_usage": "disk",
            "process_health": "process",
        }
        checks: Dict[str, Any] = {}
        for service_name, result in results.items():
            key = friendly_names.get(service_name, service_name)
            result_dict = result.to_dict()
            checks[key] = {
                "status": result_dict["status"],
                "message": result_dict["message"],
                "response_time": result_dict["response_time"],
                "details": result_dict["details"],
                "recommendations": result_dict.get("recommendations", []),
            }

        elapsed = time.time() - start_time
        return {
            "overall_status": overall_status,
            "status": overall_status,
            "response_time": elapsed,
            "timestamp": datetime.now().isoformat(),
            "checks": checks,
            "services": {name: res.__dict__ for name, res in results.items()},
            "summary": {
                "total_services": len(results),
                "healthy_count": sum(1 for r in results.values() if r.status == "healthy"),
                "warning_count": sum(1 for r in results.values() if r.status == "warning"),
                "critical_count": sum(1 for r in results.values() if r.status == "critical"),
                "error_count": sum(1 for r in results.values() if r.status == "error"),
            },
        }

    def check_health_sync(self) -> Dict[str, Any]:
        """同步执行健康检查"""
        try:
            async_result = asyncio.run(self.check_health_async())
            if asyncio.isfuture(async_result):
                return async_result.result()
            if hasattr(async_result, "result") and callable(async_result.result):
                try:
                    return async_result.result()
                except Exception:
                    pass
            return async_result
        except (RuntimeError, TypeError):
            # 如果已在事件循环中或异步接口被替换为同步实现，则回退到同步路径
            return self._check_health_sync_fallback()

    def _check_health_sync_fallback(self) -> Dict[str, Any]:
        start_time = time.time()

        results: Dict[str, HealthCheckResult] = {}
        for service_name in self._health_checks.keys():
            results[service_name] = self.check_service(service_name)

        friendly_names = {
            "cpu_usage": "cpu",
            "memory_usage": "memory",
            "disk_usage": "disk",
            "process_health": "process",
        }
        checks: Dict[str, Any] = {}
        for service_name, result in results.items():
            key = friendly_names.get(service_name, service_name)
            checks[key] = {
                "status": result.status,
                "message": result.message,
                "response_time": result.response_time,
                "details": result.details,
            }

        elapsed = time.time() - start_time
        overall_status = self._calculate_overall_status(results)
        return {
            "overall_status": overall_status,
            "status": overall_status,
            "response_time": elapsed,
            "timestamp": datetime.now().isoformat(),
            "checks": checks,
            "services": {k: v.__dict__ for k, v in results.items()},
            "summary": {
                "total_services": len(results),
                "healthy_count": sum(1 for r in results.values() if r.status == "healthy"),
                "warning_count": sum(1 for r in results.values() if r.status == "warning"),
                "critical_count": sum(1 for r in results.values() if r.status == "critical"),
                "error_count": sum(1 for r in results.values() if r.status == "error"),
            },
        }

    def get_health_metrics(self) -> Dict[str, Any]:
        """获取健康指标"""
        total_checks = sum(len(history) for history in self._check_history.values())
        recent_results = {}
        for service_name, history in self._check_history.items():
            if history:
                recent_results[service_name] = history[-1]

        status_counts: Dict[str, int] = {}
        for history in self._check_history.values():
            for result in history:
                status_key = str(result.status) if not isinstance(result.status, str) else result.status
                status_key = status_key.lower()
                status_counts[status_key] = status_counts.get(status_key, 0) + 1
        if not status_counts:
            status_counts = {"unknown": 0}

        total_results = sum(status_counts.values())
        score_weights = {
            "healthy": 1.0,
            "warning": 0.6,
            "critical": 0.2,
            "error": 0.0,
            "unknown": 0.4,
        }
        overall_health_score = 0.0
        if total_results:
            overall_health_score = sum(
                score_weights.get(status, 0.5) * count
                for status, count in status_counts.items()
            ) / total_results

        return {
            "total_checks": total_checks,
            "services_checked": sorted(recent_results.keys()),
            "overall_health_score": round(overall_health_score, 4),
            "check_distribution": dict(sorted(status_counts.items())),
            "system_health_metrics": {
                "total_registered_services": len(self._health_checks),
                "total_check_history": total_checks,
                "services_with_history": len([h for h in self._check_history.values() if h]),
                "overall_health_status": self._calculate_overall_status(recent_results) if recent_results else "unknown"
            },
            "performance_metrics": {
                "average_response_time": sum(
                    r.response_time for results in self._check_history.values()
                    for r in results[-10:]  # 最近10次检查
                ) / max(total_checks, 1),
                "check_success_rate": len([
                    r for results in self._check_history.values()
                    for r in results if str(r.status).lower() != "error"
                ]) / max(total_checks, 1)
            },
            "timestamp": datetime.now().isoformat()
        }

    def run_health_checks(self) -> Dict[str, Any]:
        """向后兼容的健康检查入口"""
        return self.check_health_sync()

    # =========================================================================
    # IHealthCheckExecutor 抽象方法实现（修复抽象类实例化问题）
    # =========================================================================

    def execute_check(self, check_name: str, check_type: CheckType) -> HealthCheckResult:
        """
        执行健康检查（IHealthCheckExecutor接口要求的抽象方法）
        
        Args:
            check_name: 检查名称
            check_type: 检查类型
            
        Returns:
            健康检查结果
        """
        # 委托给check_service方法
        return self.check_service(check_name)

    def get_check_status(self, check_name: str) -> Dict[str, Any]:
        """
        获取检查状态（IHealthCheckExecutor接口要求的抽象方法）
        
        Args:
            check_name: 检查名称
            
        Returns:
            检查状态字典
        """
        history = self.get_service_health_history(check_name)
        if not history:
            return {
                "check_name": check_name,
                "status": "unknown",
                "message": "无历史记录"
            }
        
        latest = history[0]
        return {
            "check_name": check_name,
            "status": latest.status,
            "timestamp": latest.timestamp.isoformat(),
            "response_time": latest.response_time,
            "details": latest.details
        }

    def _calculate_overall_status(self, results: Dict[str, HealthCheckResult]) -> str:
        """计算整体健康状态"""
        if not results:
            return "unknown"

        statuses = [r.status for r in results.values()]
        if "critical" in statuses or "error" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif all(s == "healthy" for s in statuses):
            return "healthy"
        else:
            return "unknown"

    def _register_system_checks(self):
        """注册默认系统健康检查"""
        self.register_service("cpu_usage", self._check_cpu_usage)
        self.register_service("memory_usage", self._check_memory_usage)
        self.register_service("disk_usage", self._check_disk_usage)
        self.register_service("process_health", self._check_process_health)

    def _check_cpu_usage(self) -> Dict[str, Any]:
        """检查CPU使用率"""
        try:
            if self.metrics_collector:
                latest = self.metrics_collector.get_latest_metrics()
                if latest and 'cpu' in latest:
                    cpu_usage = latest['cpu'].get('usage_percent', 0)
                    return self._build_cpu_status(cpu_usage)

            cpu_usage = psutil.cpu_percent(interval=0.1)
            return self._build_cpu_status(cpu_usage)
        except Exception as e:
            return {
                'status': 'unknown',
                'message': f"无法获取CPU信息: {e}",
                'error': str(e)
            }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """检查内存使用率"""
        try:
            mem_data: Optional[Dict[str, Any]] = None
            if self.metrics_collector:
                latest = self.metrics_collector.get_latest_metrics()
                if latest and 'memory' in latest:
                    mem_data = latest['memory']

            if mem_data:
                mem_percent = mem_data.get('percent', 0.0)
                available_bytes = mem_data.get('available')
                total_bytes = mem_data.get('total')
            else:
                memory_info = psutil.virtual_memory()
                mem_percent = getattr(memory_info, 'percent', 0.0)
                available_bytes = getattr(memory_info, 'available', None)
                total_bytes = getattr(memory_info, 'total', None)

            return self._build_memory_status(mem_percent, available_bytes, total_bytes)
        except Exception as e:
            return {
                'status': 'error',
                'message': f"内存检查失败: {e}"
            }

    def _check_disk_usage(self) -> Dict[str, Any]:
        """检查磁盘使用率"""
        try:
            disk_data: Optional[Dict[str, Any]] = None
            if self.metrics_collector:
                latest = self.metrics_collector.get_latest_metrics()
                if latest and 'disk' in latest:
                    disk_data = latest['disk']

            if disk_data:
                disk_percent = disk_data.get('percent', 0.0)
                free_bytes = disk_data.get('free')
                total_bytes = disk_data.get('total')
            else:
                disk_info = psutil.disk_usage('/')
                disk_percent = getattr(disk_info, 'percent', 0.0)
                free_bytes = getattr(disk_info, 'free', None)
                total_bytes = getattr(disk_info, 'total', None)

            return self._build_disk_status(disk_percent, free_bytes, total_bytes)
        except Exception as e:
            return {
                'status': 'error',
                'message': f"磁盘检查失败: {e}"
            }

    @staticmethod
    def _build_cpu_status(cpu_usage: float) -> Dict[str, Any]:
        status = "healthy" if cpu_usage < 80 else "warning" if cpu_usage < 98 else "critical"
        if status == "healthy":
            message = f"CPU使用率正常: {cpu_usage:.1f}%"
        elif status == "warning":
            message = f"CPU使用率过高: {cpu_usage:.1f}%"
        else:
            message = f"CPU使用率严重过高: {cpu_usage:.1f}%"
        return {
            'status': status,
            'value': cpu_usage,
            'cpu_percent': cpu_usage,
            'threshold': 80,
            'message': message
        }

    @staticmethod
    def _build_memory_status(
        mem_usage: float,
        available_bytes: Optional[float] = None,
        total_bytes: Optional[float] = None
    ) -> Dict[str, Any]:
        status = "healthy" if mem_usage < 85 else "warning" if mem_usage < 95 else "critical"
        if status == "healthy":
            message = f"内存使用率正常: {mem_usage:.1f}%"
        elif status == "warning":
            message = f"内存使用率偏高: {mem_usage:.1f}%"
        else:
            message = f"内存使用率过高: {mem_usage:.1f}%"

        result: Dict[str, Any] = {
            'status': status,
            'value': mem_usage,
            'memory_percent': mem_usage,
            'threshold': 85,
            'message': message
        }

        if isinstance(available_bytes, Number):
            available_gb = round(float(available_bytes) / (1024 ** 3), 2)
            result['available_gb'] = available_gb
            result['available_bytes'] = available_bytes
        else:
            available_gb = None

        if isinstance(total_bytes, Number):
            total_gb = round(float(total_bytes) / (1024 ** 3), 2)
            result['total_gb'] = total_gb
            result['total_bytes'] = total_bytes
            if available_gb is not None:
                used_gb = round(total_gb - available_gb, 2)
                result['used_gb'] = max(used_gb, 0.0)

        return result

    @staticmethod
    def _build_disk_status(
        disk_usage: float,
        free_bytes: Optional[float] = None,
        total_bytes: Optional[float] = None
    ) -> Dict[str, Any]:
        status = "healthy" if disk_usage < 90 else "warning" if disk_usage < 95 else "critical"
        if status == "healthy":
            message = f"磁盘使用率正常: {disk_usage:.1f}%"
        elif status == "warning":
            message = f"磁盘使用率偏高: {disk_usage:.1f}%"
        else:
            message = f"磁盘使用率严重不足: {disk_usage:.1f}%"

        result: Dict[str, Any] = {
            'status': status,
            'value': disk_usage,
            'disk_percent': disk_usage,
            'threshold': 90,
            'message': message
        }

        if isinstance(free_bytes, Number):
            free_gb = round(float(free_bytes) / (1024 ** 3), 2)
            result['free_gb'] = free_gb
            result['free_bytes'] = free_bytes
        else:
            free_gb = None

        if isinstance(total_bytes, Number):
            total_gb = round(float(total_bytes) / (1024 ** 3), 2)
            result['total_gb'] = total_gb
            result['total_bytes'] = total_bytes
            if free_gb is not None:
                used_gb = round(total_gb - free_gb, 2)
                result['used_gb'] = max(used_gb, 0.0)

        return result

    def _check_process_health(self) -> Dict[str, Any]:
        """检查进程健康状态"""
        try:
            current_process = psutil.Process()
            cpu_percent = current_process.cpu_percent(interval=0.1)
            memory_info = current_process.memory_info()

            processes: List[Dict[str, Any]] = []
            critical_processes: List[Dict[str, Any]] = []

            for proc in psutil.process_iter(['pid', 'name', 'status']):
                info = dict(proc.info or {})
                processes.append(info)
                status_text = str(info.get('status', '')).lower()
                if status_text not in {'running', 'sleeping', 'sleep', 'idle', 'disk-sleep'}:
                    critical_processes.append(info)

            total_processes = len(processes)

            status = "healthy"
            messages = []

            if cpu_percent > 50:  # 进程CPU使用率过高
                status = "warning"
                messages.append(f"进程CPU使用率偏高: {cpu_percent:.1f}%")

            if memory_info.rss > 500 * 1024 * 1024:  # 超过500MB
                status = "warning"
                memory_mb = memory_info.rss / (1024 * 1024)
                messages.append(f"进程内存使用偏高: {memory_mb:.1f}MB")

            if critical_processes:
                status = "warning"
                messages.append(f"{len(critical_processes)} 个关键进程状态异常")

            if not messages:
                messages.append("进程运行正常")

            return {
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_rss': memory_info.rss,
                'total_processes': total_processes,
                'critical_processes': critical_processes,
                'healthy_processes': total_processes - len(critical_processes),
                'process_details': processes,
                'timestamp': datetime.now().isoformat(),
                'message': '; '.join(messages)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"进程健康检查失败: {e}"
            }

    # =========================================================================
    # 已废弃的方法 - 由新IHealthCheckExecutor接口替代
    # 保留以向后兼容，但新代码应使用IHealthCheckExecutor接口方法
    # =========================================================================

    """
    以下方法已废弃，请使用新的接口方法：
    - run_health_checks() -> check_health_sync()
    - get_health_status() -> get_health_metrics()
    - register_health_check() -> register_service()
    - remove_health_check() -> unregister_service()
    - get_health_summary() -> get_health_metrics()
    """


# 向后兼容别名
HealthChecker = SystemHealthChecker

# =========================================================================
# 向后兼容的模块级函数
# =========================================================================


def check_health() -> Dict[str, Any]:
    """执行整体健康检查（向后兼容）"""
    try:
        checker = SystemHealthChecker()
        return checker.check_health_sync()
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def get_health_status() -> str:
    """获取健康状态（向后兼容）"""
    try:
        checker = SystemHealthChecker()
        metrics = checker.get_health_metrics()
        return metrics["system_health_metrics"]["overall_health_status"]
    except Exception as e:
        logger.error(f"获取健康状态失败: {e}")
        return "error"
