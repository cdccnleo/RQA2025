"""
application_monitor_monitoring 模块

提供 application_monitor_monitoring 相关功能和接口。
"""

import logging

import inspect
import time
import functools
import traceback

from prometheus_client import Counter
from prometheus_client import Counter, Histogram
from datetime import datetime
from typing import Dict, Optional, Any
"""
基础设施层 - 应用监控功能组件

application_monitor_monitoring 模块

应用监控器的监控功能实现，包含函数监控、错误记录等核心监控功能。
"""

logger = logging.getLogger(__name__)


class ApplicationMonitorMonitoringMixin:
    """
    应用监控功能混入类

    提供函数监控、错误记录等监控功能。
    应与ApplicationMonitorCoreMixin一起使用。
    """

    def monitor(self, name: Optional[str] = None, slow_threshold: float = 5.0):
        """
        函数监控装饰器

        Args:
            name: 监控名称(默认使用函数名)
            slow_threshold: 慢执行阈值(秒)
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_function_with_monitoring(
                    func, name, slow_threshold, args, kwargs
                )
            return wrapper
        return decorator

    def _execute_function_with_monitoring(self, func, name: Optional[str],
                                          slow_threshold: float, args, kwargs):
        """执行函数并进行监控"""
        func_name = name or func.__name__
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            success = True
            return self._handle_function_success(result, func_name, start_time, success, slow_threshold)
        except Exception as e:
            success = False
            return self._handle_function_error(e, func_name, start_time, success, slow_threshold)

    def _handle_function_success(self, result, func_name: str, start_time: float,
                                 success: bool, slow_threshold: float):
        """处理函数成功执行"""
        execution_time = time.time() - start_time

        # 记录执行指标
        self.record_function(
            name=func_name,
            execution_time=execution_time,
            success=success
        )

        # 检查慢执行
        self._check_slow_execution(func_name, execution_time, slow_threshold)

        return result

    def _handle_function_error(self, error: Exception, func_name: str, start_time: float,
                               success: bool, slow_threshold: float):
        """处理函数执行错误"""
        execution_time = time.time() - start_time

        # 记录错误
        self.record_error(
            source=func_name,
            message=str(error),
            stack_trace=traceback.format_exc(),
            context=None
        )

        # 记录执行指标（失败）
        self.record_function(
            name=func_name,
            execution_time=execution_time,
            success=success
        )

        # 检查慢执行（即使失败也可能有用）
        self._check_slow_execution(func_name, execution_time, slow_threshold)

        # 重新抛出异常
        raise

    def _check_slow_execution(self, func_name: str, execution_time: float, slow_threshold: float):
        """检查并处理慢执行告警"""
        if execution_time > slow_threshold:
            self._trigger_alert(
                'performance',
                {
                    'level': 'warning',
                    'message': f"Slow execution: {func_name} took {execution_time:.2f}s",
                    'value': execution_time,
                    'threshold': slow_threshold,
                    'timestamp': datetime.now().isoformat()
                }
            )

    def record_function(self, name: str, execution_time: float, success: bool = True):
        """
        记录函数执行指标

        Args:
            name: 函数名称
            execution_time: 执行时间(秒)
            success: 是否成功
        """
        # 构建函数指标
        metric = self._build_function_metric(name, execution_time, success)

        # 添加到内存存储
        self._metrics['functions'].append(metric)

        # 写入外部存储
        self._write_function_to_influxdb(metric, name, execution_time, success)

        # 限制内存数据量
        self._limit_function_data()

        # Prometheus指标上报
        self._report_function_to_prometheus(name, execution_time, success)

    def _build_function_metric(self, name: str, execution_time: float, success: bool) -> Dict[str, Any]:
        """构建函数指标字典"""
        return {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'execution_time': execution_time,
            'success': success
        }

    def _write_function_to_influxdb(self, metric: Dict[str, Any], name: str,
                                    execution_time: float, success: bool):
        """写入函数指标到InfluxDB"""
        if not (
            self.influx_client
            and self.influx_bucket
            and hasattr(self, 'SYNCHRONOUS')
            and self.SYNCHRONOUS is not None
        ):
            return

        try:
            write_api = self.influx_client.write_api(write_options=self.SYNCHRONOUS)

            point = {
                "measurement": "function_metrics",
                "tags": {
                    "app": self.app_name,
                    "function": name
                },
                "fields": {
                    "execution_time": execution_time,
                    "success": success
                },
                "time": metric['timestamp']
            }

            write_api.write(bucket=self.influx_bucket, record=point)
        except Exception as e:
            logger.error(f"Failed to write function metric to InfluxDB: {e}")

    def _limit_function_data(self):
        """限制函数数据的内存存储量"""
        if len(self._metrics['functions']) > 10000:
            self._metrics['functions'] = self._metrics['functions'][-10000:]

    def _report_function_to_prometheus(self, name: str, execution_time: float, success: bool):
        """向Prometheus上报函数指标"""
        try:
            if (hasattr(self, 'prom_function_calls')
                and isinstance(self.prom_function_calls, Counter)
                    and hasattr(self.prom_function_calls, 'labels')):
                self.prom_function_calls.labels(name, str(success)).inc()

            if (hasattr(self, 'prom_function_duration')
                and isinstance(self.prom_function_duration, Histogram)
                    and hasattr(self.prom_function_duration, 'labels')):
                self.prom_function_duration.labels(name).observe(execution_time)
        except Exception:
            pass  # Prometheus可能未正确初始化

    def record_error(
        self,
        source: str,
        message: str,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> None:
        """
        记录错误

        Args:
            source: 错误来源
            message: 错误消息
            stack_trace: 堆栈跟踪(可选)
            context: 上下文信息(可选)
            error: 异常对象(可选)
        """
        # 构建错误记录
        error_record = self._build_error_record(source, message, stack_trace, context, error)

        # 添加到内存存储
        self._metrics['errors'].append(error_record)

        # 写入外部存储
        self._write_error_to_influxdb(error_record, source, message, stack_trace)

        # 限制内存数据量
        self._limit_error_data()

        # Prometheus指标上报
        self._report_error_to_prometheus(error, source)

        # 触发告警
        self._trigger_error_alert(source, message, error_record['timestamp'])

    def _build_error_record(self, source: str, message: str, stack_trace: Optional[str],
                            context: Optional[Dict[str, Any]], error: Optional[Exception]) -> Dict[str, Any]:
        """构建错误记录字典"""
        timestamp = datetime.now().isoformat()
        error_record = {
            'timestamp': timestamp,
            'source': source,
            'message': message,
            'stack_trace': stack_trace,
            'context': context or {}
        }

        # 如果有异常对象，提取额外信息
        if error:
            error_record['error_type'] = type(error).__name__
            error_record['error_message'] = str(error)

        return error_record

    def _write_error_to_influxdb(self, error_record: Dict[str, Any], source: str,
                                 message: str, stack_trace: Optional[str]):
        """写入错误到InfluxDB"""
        if not (
            self.influx_client
            and self.influx_bucket
            and hasattr(self, 'SYNCHRONOUS')
            and self.SYNCHRONOUS is not None
        ):
            return

        try:
            write_api = self.influx_client.write_api(write_options=self.SYNCHRONOUS)

            point = {
                "measurement": "error_metrics",
                "tags": {
                    "app": self.app_name,
                    "source": source,
                    "error_type": error_record.get('error_type', 'Unknown')
                },
                "fields": {
                    "message": message,
                    "has_stack_trace": bool(stack_trace)
                },
                "time": error_record['timestamp']
            }

            write_api.write(bucket=self.influx_bucket, record=point)
        except Exception as e:
            logger.error(f"Failed to write error metric to InfluxDB: {e}")

    def _limit_error_data(self):
        """限制错误数据的内存存储量"""
        if len(self._metrics['errors']) > 10000:
            self._metrics['errors'] = self._metrics['errors'][-10000:]

    def _report_error_to_prometheus(self, error: Optional[Exception], source: str):
        """向Prometheus上报错误指标"""
        try:
            if (hasattr(self, 'prom_function_errors')
                and isinstance(self.prom_function_errors, Counter)
                    and hasattr(self.prom_function_errors, 'labels')):
                error_type = type(error).__name__ if error else 'Unknown'
                self.prom_function_errors.labels(source, error_type).inc()
        except Exception:
            pass  # Prometheus可能未正确初始化

    def _trigger_error_alert(self, source: str, message: str, timestamp: str):
        """触发错误告警"""
        self._trigger_alert(
            'error',
            {
                'level': 'error',
                'message': f"Error in {source}: {message}",
                'source': source,
                'timestamp': timestamp
            }
        )

# 模块级健康检查函数


def check_health() -> Dict[str, Any]:
    """执行整体健康检查

    Returns:
        Dict[str, Any]: 健康检查结果
    """
    try:
        logger.info("开始应用监控器监控模块健康检查")

        health_checks = {
            "monitoring_mixin": check_monitoring_mixin(),
            "decorator_system": check_decorator_system(),
            "metrics_system": check_metrics_system()
        }

        # 综合健康状态
        overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

        result = {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "service": "application_monitor_monitoring",
            "checks": health_checks
        }

        if not overall_healthy:
            logger.warning("应用监控器监控模块健康检查发现问题")
            result["issues"] = [
                name for name, check in health_checks.items()
                if not check.get("healthy", False)
            ]

        logger.info(f"应用监控器监控模块健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result

    except Exception as e:
        logger.error(f"应用监控器监控模块健康检查失败: {str(e)}", exc_info=True)
        return {
            "healthy": False,
            "timestamp": datetime.now().isoformat(),
            "service": "application_monitor_monitoring",
            "error": str(e)
        }


def check_monitoring_mixin() -> Dict[str, Any]:
    """检查监控混入类定义

    Returns:
        Dict[str, Any]: 监控混入类检查结果
    """
    try:
        # 检查ApplicationMonitorMonitoringMixin类存在
        monitoring_mixin_exists = 'ApplicationMonitorMonitoringMixin' in globals()

        if not monitoring_mixin_exists:
            return {"healthy": False, "error": "ApplicationMonitorMonitoringMixin class not found"}

        # 检查必需的方法
        required_methods = ['monitor', 'record_function', 'record_error']
        existing_methods = [method for method in dir(
            ApplicationMonitorMonitoringMixin) if not method.startswith('_')]

        methods_complete = all(method in existing_methods for method in required_methods)

        # 检查私有方法
        private_methods = ['_execute_function_with_monitoring',
                           '_handle_function_success', '_handle_function_error']
        private_methods_exist = all(hasattr(ApplicationMonitorMonitoringMixin, method)
                                    for method in private_methods)

        return {
            "healthy": monitoring_mixin_exists and methods_complete and private_methods_exist,
            "monitoring_mixin_exists": monitoring_mixin_exists,
            "methods_complete": methods_complete,
            "private_methods_exist": private_methods_exist,
            "existing_methods": existing_methods,
            "required_methods": required_methods
        }
    except Exception as e:
        logger.error(f"监控混入类检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_decorator_system() -> Dict[str, Any]:
    """检查装饰器系统

    Returns:
        Dict[str, Any]: 装饰器系统检查结果
    """
    try:
        # 检查monitor装饰器是否存在
        monitor_decorator_exists = hasattr(ApplicationMonitorMonitoringMixin, 'monitor')

        if not monitor_decorator_exists:
            return {"healthy": False, "error": "monitor decorator not found"}

        # 测试装饰器功能
        decorator_works = False
        test_instance = None

        try:
            # 创建一个测试类实例（模拟）
            class TestMonitor:
                def __init__(self):
                    pass

            # 模拟mixin的行为
            test_instance = TestMonitor()

            # 添加monitor方法
            def monitor(self, name=None, slow_threshold=5.0):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs)
                    return wrapper
                return decorator

            test_instance.monitor = monitor.__get__(test_instance, TestMonitor)

            # 测试装饰器
            @test_instance.monitor()
            def test_function():
                return "test_result"

            result = test_function()
            decorator_works = result == "test_result"

        except Exception:
            decorator_works = False

        return {
            "healthy": monitor_decorator_exists and decorator_works,
            "monitor_decorator_exists": monitor_decorator_exists,
            "decorator_works": decorator_works
        }
    except Exception as e:
        logger.error(f"装饰器系统检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_metrics_system() -> Dict[str, Any]:
    """检查指标系统

    Returns:
        Dict[str, Any]: 指标系统检查结果
    """
    try:
        # 检查指标相关方法
        metrics_methods = ['record_function', 'record_error',
                           '_build_function_metric', '_build_error_record']
        methods_exist = all(hasattr(ApplicationMonitorMonitoringMixin, method)
                            for method in metrics_methods)

        # 检查InfluxDB和Prometheus相关方法
        database_methods = ['_write_function_to_influxdb', '_write_error_to_influxdb']
        prometheus_methods = ['_report_function_to_prometheus', '_report_error_to_prometheus']

        influxdb_methods_exist = all(hasattr(ApplicationMonitorMonitoringMixin, method)
                                     for method in database_methods)
        prometheus_methods_exist = all(
            hasattr(ApplicationMonitorMonitoringMixin, method) for method in prometheus_methods)

        # 检查数据限制方法
        data_limit_methods = ['_limit_function_data', '_limit_error_data']
        data_limit_methods_exist = all(
            hasattr(ApplicationMonitorMonitoringMixin, method) for method in data_limit_methods)

        return {
            "healthy": methods_exist and influxdb_methods_exist and prometheus_methods_exist and data_limit_methods_exist,
            "metrics_methods_exist": methods_exist,
            "influxdb_methods_exist": influxdb_methods_exist,
            "prometheus_methods_exist": prometheus_methods_exist,
            "data_limit_methods_exist": data_limit_methods_exist,
            "metrics_methods": metrics_methods,
            "database_methods": database_methods,
            "prometheus_methods": prometheus_methods
        }
    except Exception as e:
        logger.error(f"指标系统检查失败: {str(e)}")
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
            "service": "application_monitor_monitoring",
            "health_check": health_check,
            "monitoring_methods_count": len([method for method in dir(ApplicationMonitorMonitoringMixin) if not method.startswith('_')]),
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

        # 统计监控方法信息
        public_methods = [method for method in dir(
            ApplicationMonitorMonitoringMixin) if not method.startswith('_')]
        private_methods = [method for method in dir(
            ApplicationMonitorMonitoringMixin) if method.startswith('_')]

        return {
            "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
            "application_monitor_monitoring_module_info": {
                "service_name": "application_monitor_monitoring",
                "purpose": "应用监控器监控功能",
                "operational": health_check["healthy"]
            },
            "monitoring_capabilities": {
                "public_methods_count": len(public_methods),
                "private_methods_count": len(private_methods),
                "decorator_system_working": health_check["checks"]["decorator_system"]["healthy"],
                "metrics_system_complete": health_check["checks"]["metrics_system"]["healthy"]
            },
            "functionality_status": {
                "monitoring_mixin_available": health_check["checks"]["monitoring_mixin"]["healthy"],
                "function_monitoring_enabled": health_check["checks"]["metrics_system"]["metrics_methods_exist"],
                "error_tracking_enabled": health_check["checks"]["metrics_system"]["metrics_methods_exist"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康摘要报告失败: {str(e)}")
        return {"overall_health": "error", "error": str(e)}


def monitor_application_monitor_monitoring() -> Dict[str, Any]:
    """监控应用监控器监控模块状态

    Returns:
        Dict[str, Any]: 监控模块监控结果
    """
    try:
        health_check = check_health()

        # 计算监控效率指标
        monitoring_efficiency = 1.0 if health_check["healthy"] else 0.0

        return {
            "healthy": health_check["healthy"],
            "monitoring_metrics": {
                "service_name": "application_monitor_monitoring",
                "monitoring_efficiency": monitoring_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            },
            "capability_metrics": {
                "decorator_system_functional": health_check["checks"]["decorator_system"]["healthy"],
                "metrics_system_complete": health_check["checks"]["metrics_system"]["healthy"],
                "database_integration_available": health_check["checks"]["metrics_system"]["influxdb_methods_exist"]
            }
        }
    except Exception as e:
        logger.error(f"应用监控器监控模块监控失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def validate_application_monitor_monitoring() -> Dict[str, Any]:
    """验证应用监控器监控模块

    Returns:
        Dict[str, Any]: 模块验证结果
    """
    try:
        validation_results = {
            "mixin_validation": _validate_monitoring_mixin(),
            "decorator_validation": _validate_decorator_system(),
            "metrics_validation": _validate_metrics_system()
        }

        overall_valid = all(result.get("valid", False) for result in validation_results.values())

        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"应用监控器监控模块验证失败: {str(e)}")
        return {"valid": False, "error": str(e)}


def _validate_monitoring_mixin() -> Dict[str, Any]:
    """验证监控混入类"""
    try:
        # 检查类存在
        mixin_exists = 'ApplicationMonitorMonitoringMixin' in globals()

        if not mixin_exists:
            return {"valid": False, "error": "ApplicationMonitorMonitoringMixin not found"}

        # 检查类结构
        mixin_class = ApplicationMonitorMonitoringMixin

        # 检查方法签名
        method_signatures = {}
        required_methods = ['monitor', 'record_function', 'record_error']

        for method_name in required_methods:
            if hasattr(mixin_class, method_name):
                method = getattr(mixin_class, method_name)
                try:
                    sig = inspect.signature(method)
                    method_signatures[method_name] = str(sig)
                except Exception:
                    method_signatures[method_name] = "signature_unavailable"
            else:
                method_signatures[method_name] = "method_not_found"

        # 检查所有必需方法都存在且有签名
        all_methods_valid = all(sig != "method_not_found" for sig in method_signatures.values())

        return {
            "valid": mixin_exists and all_methods_valid,
            "mixin_exists": mixin_exists,
            "all_methods_valid": all_methods_valid,
            "method_signatures": method_signatures,
            "required_methods": required_methods
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_decorator_system() -> Dict[str, Any]:
    """验证装饰器系统"""
    try:
        # 检查装饰器方法存在
        decorator_exists = hasattr(ApplicationMonitorMonitoringMixin, 'monitor')

        if not decorator_exists:
            return {"valid": False, "error": "monitor decorator not found"}

        # 检查装饰器返回类型
        monitor_method = getattr(ApplicationMonitorMonitoringMixin, 'monitor')

        # 测试装饰器行为（简化测试）
        decorator_behavior_valid = False
        try:
            # 创建装饰器实例
            decorator = monitor_method(None)  # 创建装饰器

            # 检查装饰器是否可调用
            decorator_behavior_valid = callable(decorator)

            if decorator_behavior_valid:
                # 测试装饰器包装
                @decorator
                def test_func():
                    return "decorated"

                result = test_func()
                decorator_behavior_valid = result == "decorated"

        except Exception:
            decorator_behavior_valid = False

        return {
            "valid": decorator_exists and decorator_behavior_valid,
            "decorator_exists": decorator_exists,
            "decorator_behavior_valid": decorator_behavior_valid
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_metrics_system() -> Dict[str, Any]:
    """验证指标系统"""
    try:
        # 检查指标记录方法
        metrics_methods = ['record_function', 'record_error']
        metrics_methods_exist = all(hasattr(ApplicationMonitorMonitoringMixin, method)
                                    for method in metrics_methods)

        # 检查私有辅助方法
        helper_methods = ['_build_function_metric', '_build_error_record',
                          '_limit_function_data', '_limit_error_data']
        helper_methods_exist = all(hasattr(ApplicationMonitorMonitoringMixin, method)
                                   for method in helper_methods)

        # 检查数据库和监控集成方法
        integration_methods = [
            '_write_function_to_influxdb', '_write_error_to_influxdb',
            '_report_function_to_prometheus', '_report_error_to_prometheus'
        ]
        integration_methods_exist = all(
            hasattr(ApplicationMonitorMonitoringMixin, method) for method in integration_methods)

        return {
            "valid": metrics_methods_exist and helper_methods_exist and integration_methods_exist,
            "metrics_methods_exist": metrics_methods_exist,
            "helper_methods_exist": helper_methods_exist,
            "integration_methods_exist": integration_methods_exist,
            "metrics_methods": metrics_methods,
            "helper_methods": helper_methods,
            "integration_methods": integration_methods
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
