"""
exceptions 模块

提供 exceptions 相关功能和接口。
"""

import logging

import inspect
import asyncio

from datetime import datetime
from typing import Dict, Any, Optional
"""
健康检查相关异常类

提供完整的异常处理机制，包括详细的错误信息和日志记录。
"""

logger = logging.getLogger(__name__)


class HealthInfrastructureError(Exception):
    """健康基础设施基础异常类"""

    def __init__(self, message: str, error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        初始化异常

        Args:
            message: 错误消息
            error_code: 错误代码
            details: 详细错误信息
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "HEALTH_INFRA_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

        # 记录错误日志
        self._log_error()

    def _log_error(self) -> None:
        """记录错误日志"""
        logger.error(f"{self.__class__.__name__}: {self.message}", extra={
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp
        })

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp
        }


class LoadBalancerError(HealthInfrastructureError):
    """负载均衡器异常"""

    def __init__(self, message: str, service_name: Optional[str] = None,
                 backend_status: Optional[Dict[str, Any]] = None):
        """
        初始化负载均衡器异常

        Args:
            message: 错误消息
            service_name: 服务名称
            backend_status: 后端状态信息
        """
        details = {}
        if service_name:
            details["service_name"] = service_name
        if backend_status:
            details["backend_status"] = backend_status

        super().__init__(message, "LOAD_BALANCER_ERROR", details)


class HealthCheckError(HealthInfrastructureError):
    """健康检查异常"""

    def __init__(self, message: str, component: Optional[str] = None,
                 check_type: Optional[str] = None, last_result: Optional[Dict[str, Any]] = None):
        """
        初始化健康检查异常

        Args:
            message: 错误消息
            component: 组件名称
            check_type: 检查类型
            last_result: 最后检查结果
        """
        details = {}
        if component:
            details["component"] = component
        if check_type:
            details["check_type"] = check_type
        if last_result:
            details["last_result"] = last_result

        super().__init__(message, "HEALTH_CHECK_ERROR", details)


class MonitoringError(HealthInfrastructureError):
    """监控异常"""

    def __init__(self, message: str, monitor_type: Optional[str] = None,
                 metrics: Optional[Dict[str, Any]] = None,
                 alert_triggered: bool = False):
        """
        初始化监控异常

        Args:
            message: 错误消息
            monitor_type: 监控类型
            metrics: 监控指标
            alert_triggered: 是否触发告警
        """
        details = {"alert_triggered": alert_triggered}
        if monitor_type:
            details["monitor_type"] = monitor_type
        if metrics:
            details["metrics"] = metrics

        super().__init__(message, "MONITORING_ERROR", details)


class ConfigurationError(HealthInfrastructureError):
    """配置异常"""

    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_value: Optional[Any] = None):
        """
        初始化配置异常

        Args:
            message: 错误消息
            config_key: 配置键
            config_value: 配置值
        """
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = config_value

        super().__init__(message, "CONFIGURATION_ERROR", details)


class ValidationError(HealthInfrastructureError):
    """验证异常"""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None, validation_rule: Optional[str] = None):
        """
        初始化验证异常

        Args:
            message: 错误消息
            field: 字段名
            value: 字段值
            validation_rule: 验证规则
        """
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        if validation_rule:
            details["validation_rule"] = validation_rule

        super().__init__(message, "VALIDATION_ERROR", details)


class AsyncOperationError(HealthInfrastructureError):
    """异步操作异常"""

    def __init__(self, message: str, operation: Optional[str] = None,
                 timeout: Optional[float] = None, coroutine_info: Optional[Dict[str, Any]] = None):
        """
        初始化异步操作异常

        Args:
            message: 错误消息
            operation: 操作名称
            timeout: 超时时间
            coroutine_info: 协程信息
        """
        details = {}
        if operation:
            details["operation"] = operation
        if timeout is not None:
            details["timeout"] = timeout
        if coroutine_info:
            details["coroutine_info"] = coroutine_info

        super().__init__(message, "ASYNC_OPERATION_ERROR", details)

# 异常处理器工具函数


def handle_health_exception(func_name: str, exc: Exception) -> Dict[str, Any]:
    """
    处理健康检查异常

    Args:
        func_name: 函数名称
        exc: 异常对象

    Returns:
        Dict[str, Any]: 错误响应
    """
    logger.error(f"{func_name} 执行失败: {exc}", exc_info=True)

    if isinstance(exc, HealthInfrastructureError):
        return exc.to_dict()

    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "error_code": "UNKNOWN_ERROR",
        "timestamp": datetime.now().isoformat(),
        "function": func_name
    }


def safe_execute(func, *args, **kwargs) -> tuple[bool, Any]:
    """
    安全执行函数

    Args:
        func: 要执行的函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        tuple[bool, Any]: (是否成功, 结果或错误信息)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"函数 {func.__name__} 执行失败: {e}", exc_info=True)
        return False, handle_health_exception(func.__name__, e)


async def handle_health_exception_async(func_name: str, exc: Exception) -> Dict[str, Any]:
    """异步处理健康检查异常

    Args:
        func_name: 函数名称
        exc: 异常对象

    Returns:
        Dict[str, Any]: 错误响应
    """
    try:
        logger.error(f"异步{func_name} 执行失败: {exc}", exc_info=True)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, handle_health_exception, func_name, exc)
    except Exception as e:
        logger.error(f"异步处理异常失败: {e}")
        return {
            "error_type": "AsyncExceptionHandlingError",
            "message": str(e),
            "error_code": "ASYNC_EXCEPTION_ERROR",
            "timestamp": datetime.now().isoformat(),
            "function": func_name
        }


async def safe_execute_async(func, *args, **kwargs) -> tuple[bool, Any]:
    """异步安全执行函数

    Args:
        func: 要执行的函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        tuple[bool, Any]: (是否成功, 结果或错误信息)
    """
    try:
        logger.debug(f"异步执行函数 {func.__name__}")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, func, *args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"异步函数 {func.__name__} 执行失败: {e}", exc_info=True)
        error_result = await handle_health_exception_async(func.__name__, e)
        return False, error_result

# 模块级健康检查函数


def check_health() -> Dict[str, Any]:
    """执行整体健康检查

    Returns:
        Dict[str, Any]: 健康检查结果
    """
    try:
        logger.info("开始异常处理模块健康检查")

        health_checks = {
            "exception_classes": check_exception_classes(),
            "inheritance_structure": check_inheritance_structure(),
            "utility_functions": check_utility_functions()
        }

        # 综合健康状态
        overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

        result = {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "service": "health_exceptions",
            "checks": health_checks
        }

        if not overall_healthy:
            logger.warning("异常处理模块健康检查发现问题")
            result["issues"] = [
                name for name, check in health_checks.items()
                if not check.get("healthy", False)
            ]

        logger.info(f"异常处理模块健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result

    except Exception as e:
        logger.error(f"异常处理模块健康检查失败: {str(e)}", exc_info=True)
        return {
            "healthy": False,
            "timestamp": datetime.now().isoformat(),
            "service": "health_exceptions",
            "error": str(e)
        }


def check_exception_classes() -> Dict[str, Any]:
    """检查异常类定义

    Returns:
        Dict[str, Any]: 异常类定义健康检查结果
    """
    try:
        # 检查主要的异常类
        exception_classes = [
            'HealthInfrastructureError', 'LoadBalancerError', 'HealthCheckError',
            'MonitoringError', 'ConfigurationError', 'ValidationError', 'AsyncOperationError'
        ]

        classes_exist = all(cls in globals() for cls in exception_classes)

        # 检查异常类的数量
        exception_count = len([name for name in globals() if name.endswith('Error')])

        return {
            "healthy": classes_exist and exception_count >= 7,
            "classes_exist": classes_exist,
            "exception_count": exception_count,
            "expected_classes": exception_classes,
            "missing_classes": [cls for cls in exception_classes if cls not in globals()]
        }
    except Exception as e:
        logger.error(f"异常类定义健康检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_inheritance_structure() -> Dict[str, Any]:
    """检查继承结构

    Returns:
        Dict[str, Any]: 继承结构健康检查结果
    """
    try:
        # 检查继承关系
        inheritance_checks = []

        # HealthInfrastructureError 应该是 Exception 的子类
        if 'HealthInfrastructureError' in globals():
            inheritance_checks.append(issubclass(HealthInfrastructureError, Exception))

        # 其他异常类应该是 HealthInfrastructureError 的子类
        child_exceptions = ['LoadBalancerError', 'HealthCheckError', 'MonitoringError',
                            'ConfigurationError', 'ValidationError', 'AsyncOperationError']

        for exc_name in child_exceptions:
            if exc_name in globals():
                exc_class = globals()[exc_name]
                inheritance_checks.append(issubclass(exc_class, HealthInfrastructureError))
            else:
                inheritance_checks.append(False)

        all_inheritances_correct = all(inheritance_checks)

        return {
            "healthy": all_inheritances_correct,
            "all_inheritances_correct": all_inheritances_correct,
            "inheritance_checks": inheritance_checks,
            "base_exception_inherits_from_exception": inheritance_checks[0] if inheritance_checks else False
        }
    except Exception as e:
        logger.error(f"继承结构健康检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_utility_functions() -> Dict[str, Any]:
    """检查工具函数

    Returns:
        Dict[str, Any]: 工具函数健康检查结果
    """
    try:
        # 检查工具函数存在
        utility_functions = ['handle_health_exception', 'safe_execute']
        functions_exist = all(func in globals() for func in utility_functions)

        # 测试safe_execute函数
        safe_execute_works = False
        if 'safe_execute' in globals():
            try:
                success, result = safe_execute(lambda: "test")
                safe_execute_works = success and result == "test"
            except Exception:
                safe_execute_works = False

        # 测试handle_health_exception函数
        handle_exception_works = False
        if 'handle_health_exception' in globals():
            try:
                test_exc = ValueError("test error")
                error_dict = handle_health_exception("test_func", test_exc)
                handle_exception_works = isinstance(error_dict, dict) and "error_type" in error_dict
            except Exception:
                handle_exception_works = False

        return {
            "healthy": functions_exist and safe_execute_works and handle_exception_works,
            "functions_exist": functions_exist,
            "safe_execute_works": safe_execute_works,
            "handle_exception_works": handle_exception_works,
            "utility_functions": utility_functions
        }
    except Exception as e:
        logger.error(f"工具函数健康检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def health_status() -> Dict[str, Any]:
    """获取健康状态摘要

    Returns:
        Dict[str, Any]: 健康状态摘要
    """
    try:
        health_check = check_health()

        return {
            "status": "healthy" if health_check["healthy"] else "unhealthy",
            "service": "health_exceptions",
            "health_check": health_check,
            "exception_classes_count": len([name for name in globals() if name.endswith('Error')]),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康状态摘要失败: {str(e)}")
        return {"status": "error", "error": str(e)}


def health_summary() -> Dict[str, Any]:
    """获取健康摘要报告

    Returns:
        Dict[str, Any]: 健康摘要报告
    """
    try:
        health_check = check_health()

        # 统计异常类信息
        exception_classes = [name for name in globals() if name.endswith('Error')]
        utility_functions = [name for name in globals() if name in [
            'handle_health_exception', 'safe_execute']]

        return {
            "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
            "health_exceptions_module_info": {
                "service_name": "health_exceptions",
                "purpose": "健康检查异常处理",
                "operational": health_check["healthy"]
            },
            "exception_hierarchy": {
                "exception_classes_defined": len(exception_classes),
                "utility_functions_available": len(utility_functions),
                "inheritance_structure_valid": health_check["checks"]["inheritance_structure"]["healthy"],
                "base_exception_class_working": health_check["checks"]["exception_classes"]["healthy"]
            },
            "functionality_status": {
                "utility_functions_working": health_check["checks"]["utility_functions"]["healthy"],
                "error_handling_complete": health_check["checks"]["utility_functions"]["handle_exception_works"],
                "safe_execution_working": health_check["checks"]["utility_functions"]["safe_execute_works"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康摘要报告失败: {str(e)}")
        return {"overall_health": "error", "error": str(e)}


def monitor_health_exceptions_module() -> Dict[str, Any]:
    """监控健康异常模块状态

    Returns:
        Dict[str, Any]: 模块监控结果
    """
    try:
        health_check = check_health()

        # 计算模块效率指标
        module_efficiency = 1.0 if health_check["healthy"] else 0.0

        return {
            "healthy": health_check["healthy"],
            "module_metrics": {
                "service_name": "health_exceptions",
                "module_efficiency": module_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            },
            "exception_metrics": {
                "exception_classes_available": health_check["checks"]["exception_classes"]["exception_count"],
                "inheritance_structure_valid": health_check["checks"]["inheritance_structure"]["all_inheritances_correct"],
                "utility_functions_working": health_check["checks"]["utility_functions"]["healthy"]
            }
        }
    except Exception as e:
        logger.error(f"健康异常模块监控失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def validate_health_exceptions_config() -> Dict[str, Any]:
    """验证健康异常配置

    Returns:
        Dict[str, Any]: 配置验证结果
    """
    try:
        validation_results = {
            "exception_validation": _validate_exception_classes(),
            "inheritance_validation": _validate_inheritance(),
            "function_validation": _validate_functions()
        }

        overall_valid = all(result.get("valid", False) for result in validation_results.values())

        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"健康异常配置验证失败: {str(e)}")
        return {"valid": False, "error": str(e)}


def _validate_exception_classes() -> Dict[str, Any]:
    """验证异常类定义"""
    try:
        # 检查所有异常类
        required_exceptions = [
            'HealthInfrastructureError', 'LoadBalancerError', 'HealthCheckError',
            'MonitoringError', 'ConfigurationError', 'ValidationError', 'AsyncOperationError'
        ]

        exceptions_exist = all(exc in globals() for exc in required_exceptions)

        # 检查每个异常类是否可以实例化
        instantiation_tests = {}
        for exc_name in required_exceptions:
            if exc_name in globals():
                try:
                    exc_class = globals()[exc_name]
                    # 尝试创建实例（基础异常类需要参数）
                    if exc_name == 'HealthInfrastructureError':
                        instance = exc_class("test message")
                    else:
                        instance = exc_class("test message")
                    instantiation_tests[exc_name] = {"success": True}
                except Exception as e:
                    instantiation_tests[exc_name] = {"success": False, "error": str(e)}
            else:
                instantiation_tests[exc_name] = {"success": False, "error": "Class not found"}

        all_instantiable = all(test["success"] for test in instantiation_tests.values())

        return {
            "valid": exceptions_exist and all_instantiable,
            "exceptions_exist": exceptions_exist,
            "all_instantiable": all_instantiable,
            "instantiation_tests": instantiation_tests,
            "required_exceptions": required_exceptions
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_inheritance() -> Dict[str, Any]:
    """验证继承关系"""
    try:
        # 检查基础异常继承
        base_inherits_exception = False
        if 'HealthInfrastructureError' in globals():
            base_inherits_exception = issubclass(HealthInfrastructureError, Exception)

        # 检查子类继承
        child_exceptions = {
            'LoadBalancerError': 'HealthInfrastructureError',
            'HealthCheckError': 'HealthInfrastructureError',
            'MonitoringError': 'HealthInfrastructureError',
            'ConfigurationError': 'HealthInfrastructureError',
            'ValidationError': 'HealthInfrastructureError',
            'AsyncOperationError': 'HealthInfrastructureError'
        }

        inheritance_tests = {}
        for child, parent in child_exceptions.items():
            if child in globals() and parent in globals():
                child_class = globals()[child]
                parent_class = globals()[parent]
                inheritance_tests[child] = issubclass(child_class, parent_class)
            else:
                inheritance_tests[child] = False

        all_inheritances_correct = base_inherits_exception and all(inheritance_tests.values())

        return {
            "valid": all_inheritances_correct,
            "base_inherits_exception": base_inherits_exception,
            "inheritance_tests": inheritance_tests,
            "all_inheritances_correct": all_inheritances_correct
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_functions() -> Dict[str, Any]:
    """验证工具函数"""
    try:
        # 检查函数存在
        required_functions = ['handle_health_exception', 'safe_execute']
        functions_exist = all(func in globals() for func in required_functions)

        # 检查函数签名
        function_signatures = {}
        if functions_exist:
            try:
                for func_name in required_functions:
                    func = globals()[func_name]
                    sig = inspect.signature(func)
                    function_signatures[func_name] = str(sig)
            except Exception as e:
                function_signatures["error"] = str(e)

        return {
            "valid": functions_exist,
            "functions_exist": functions_exist,
            "function_signatures": function_signatures,
            "required_functions": required_functions
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

# 异步版本的健康检查函数


async def check_health_async() -> Dict[str, Any]:
    """异步执行整体健康检查

    Returns:
        Dict[str, Any]: 健康检查结果
    """
    try:
        logger.info("开始异步健康基础模块健康检查")

        # 并行执行多个检查任务
        tasks = [
            asyncio.create_task(_check_exception_classes_async()),
            asyncio.create_task(_check_inheritance_structure_async()),
            asyncio.create_task(_check_utility_functions_async())
        ]

        # 等待所有检查完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 汇总检查结果
        overall_healthy = all(not isinstance(r, Exception)
                              and r.get("healthy", False) for r in results)
        issues = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                issues.append(f"检查任务{i}失败: {str(result)}")
            elif not result.get("healthy", False):
                issues.extend(result.get("issues", []))

        result = {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "service": "health_exceptions_async",
            "issues": issues,
            "details": {
                "exception_classes": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
                "inheritance_structure": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
                "utility_functions": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])}
            }
        }

        logger.info(f"异步健康基础模块健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result

    except Exception as e:
        logger.error(f"异步健康基础模块健康检查失败: {str(e)}", exc_info=True)
        return {
            "healthy": False,
            "timestamp": datetime.now().isoformat(),
            "service": "health_exceptions_async",
            "error": str(e)
        }


async def health_status_async() -> Dict[str, Any]:
    """异步获取健康状态摘要

    Returns:
        Dict[str, Any]: 健康状态摘要
    """
    try:
        health_check = await check_health_async()
        loop = asyncio.get_event_loop()
        component_count = await loop.run_in_executor(None, lambda: len([name for name in globals() if name.endswith('Error')]))
        function_count = await loop.run_in_executor(None, lambda: len([name for name in globals() if name.startswith(('handle_', 'safe_', 'check_', 'health_', 'monitor_', 'validate_')) and callable(globals()[name])]))

        return {
            "status": "healthy" if health_check["healthy"] else "unhealthy",
            "service": "health_exceptions_async",
            "health_check": health_check,
            "exception_classes_count": component_count,
            "utility_functions_count": function_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"异步获取健康状态摘要失败: {str(e)}")
        return {"status": "error", "error": str(e)}


async def _check_exception_classes_async() -> Dict[str, Any]:
    """异步检查异常类定义"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, check_exception_classes)
        return result
    except Exception as e:
        return {"healthy": False, "error": str(e)}


async def _check_inheritance_structure_async() -> Dict[str, Any]:
    """异步检查继承结构"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, check_inheritance_structure)
        return result
    except Exception as e:
        return {"healthy": False, "error": str(e)}


async def _check_utility_functions_async() -> Dict[str, Any]:
    """异步检查工具函数"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, check_utility_functions)
        return result
    except Exception as e:
        return {"healthy": False, "error": str(e)}
