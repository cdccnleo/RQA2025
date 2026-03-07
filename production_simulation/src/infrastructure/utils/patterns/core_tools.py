"""
基础设施层核心工具模式

提供核心的日志、异常处理、初始化、配置和性能监控工具。
"""

import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, TypeVar

T = TypeVar('T')
logger = logging.getLogger(__name__)

# ==================== 日志模式 ====================


class InfrastructureLogger:
    """基础设施层通用日志工具"""

    @staticmethod
    def log_initialization_success(component_name: str, component_type: str = "组件"):
        """记录初始化成功日志"""
        logger.info(f"{component_name} {component_type}初始化成功")

    @staticmethod
    def log_initialization_failure(component_name: str, error: Exception, component_type: str = "组件"):
        """记录初始化失败日志"""
        logger.error(f"{component_name} {component_type}初始化失败: {error}", exc_info=True)

    @staticmethod
    def log_operation_success(operation: str, details: Optional[str] = None):
        """记录操作成功日志"""
        message = f"{operation} 操作成功"
        if details:
            message += f": {details}"
        logger.info(message)

    @staticmethod
    def log_operation_failure(operation: str, error: Exception, details: Optional[str] = None):
        """记录操作失败日志"""
        message = f"{operation} 操作失败: {error}"
        if details:
            message += f" ({details})"
        logger.error(message, exc_info=True)

    @staticmethod
    def log_configuration_validation(config_name: str, is_valid: bool, errors: Optional[list] = None):
        """记录配置验证结果"""
        if is_valid:
            logger.info(f"配置 '{config_name}' 验证通过")
        else:
            error_msg = f"配置 '{config_name}' 验证失败"
            if errors:
                error_msg += f": {', '.join(errors)}"
            logger.error(error_msg)

    @staticmethod
    def log_cache_operation(operation: str, key: str, hit: Optional[bool] = None, size: Optional[int] = None):
        """记录缓存操作日志"""
        message = f"缓存{operation}: key={key}"
        if hit is not None:
            message += f", hit={'是' if hit else '否'}"
        if size is not None:
            message += f", size={size}"
        logger.debug(message)

    @staticmethod
    def log_performance_metric(operation: str, duration: float, threshold: float = 1.0):
        """记录性能指标"""
        if duration > threshold:
            logger.warning(f"性能警告: {operation} 耗时 {duration:.3f}s (超过阈值 {threshold}s)")
        else:
            logger.debug(f"性能指标: {operation} 耗时 {duration:.3f}s")


# ==================== 异常处理模式 ====================


class InfrastructureExceptionHandler:
    """基础设施层通用异常处理工具"""

    @staticmethod
    def handle_initialization_error(component_name: str, error: Exception,
                                    component_type: str = "组件") -> None:
        """处理初始化异常"""
        InfrastructureLogger.log_initialization_failure(component_name, error, component_type)
        raise error

    @staticmethod
    def handle_operation_error(operation: str, error: Exception,
                               details: Optional[str] = None) -> None:
        """处理操作异常"""
        InfrastructureLogger.log_operation_failure(operation, error, details)
        raise error

    @staticmethod
    def safe_execute(func: Callable[..., T], *args, **kwargs) -> Optional[T]:
        """安全执行函数，捕获异常并记录"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"执行 {func.__name__} 时发生错误: {e}", exc_info=True)
            return None

    @staticmethod
    def handle_validation_error(field_name: str, value: Any, expected_type: str, error: Optional[Exception] = None):
        """处理验证异常"""
        error_msg = f"字段 '{field_name}' 验证失败: 期望 {expected_type}, 实际值 {value}"
        if error:
            error_msg += f" ({error})"
        logger.error(error_msg)
        raise ValueError(error_msg)

    @staticmethod
    def handle_config_error(config_name: str, error: Exception, details: Optional[str] = None):
        """处理配置异常"""
        error_msg = f"配置 '{config_name}' 处理失败: {error}"
        if details:
            error_msg += f" ({details})"
        logger.error(error_msg, exc_info=True)
        raise error

    @staticmethod
    def handle_connection_error(service_name: str, error: Exception, retry_count: int = 0):
        """处理连接异常"""
        error_msg = f"服务 '{service_name}' 连接失败"
        if retry_count > 0:
            error_msg += f" (已重试 {retry_count} 次)"
        logger.error(f"{error_msg}: {error}", exc_info=True)
        raise error


# ==================== 初始化模式 ====================


class InfrastructureInitializer:
    """基础设施层通用初始化工具"""

    @staticmethod
    def initialize_component(component_name: str, init_func: Callable,
                             component_type: str = "组件") -> bool:
        """通用组件初始化模式"""
        try:
            start_time = time.time()
            result = init_func()
            elapsed = time.time() - start_time

            InfrastructureLogger.log_initialization_success(component_name, component_type)
            logger.debug(f"{component_name} {component_type}初始化耗时: {elapsed:.3f}秒")
            return True
        except Exception as e:
            InfrastructureExceptionHandler.handle_initialization_error(
                component_name, e, component_type)
            return False

    @staticmethod
    def initialize_with_fallback(component_name: str, primary_init: Callable,
                                 fallback_init: Optional[Callable] = None,
                                 component_type: str = "组件") -> Any:
        """带降级的初始化模式"""
        try:
            result = primary_init()
            InfrastructureLogger.log_initialization_success(component_name, component_type)
            return result
        except Exception as e:
            logger.warning(f"{component_name} {component_type}主要初始化失败，尝试降级模式: {e}")
            if fallback_init:
                try:
                    result = fallback_init()
                    logger.info(f"{component_name} {component_type}降级初始化成功")
                    return result
                except Exception as fallback_error:
                    logger.error(f"{component_name} {component_type}降级初始化也失败: {fallback_error}")

            InfrastructureExceptionHandler.handle_initialization_error(
                component_name, e, component_type)
            return None


# ==================== 配置模式 ====================


class InfrastructureConfig:
    """基础设施层通用配置工具"""

    @staticmethod
    def get_nested_config(config: Dict[str, Any], keys: list, default: Any = None) -> Any:
        """获取嵌套配置值"""
        try:
            value = config
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return default
            return value if value is not None else default
        except Exception:
            return default

    @staticmethod
    def validate_required_config(config: Dict[str, Any], required_keys: list) -> bool:
        """验证必需的配置项"""
        missing_keys = []
        for key in required_keys:
            if key not in config or config[key] is None:
                missing_keys.append(key)

        if missing_keys:
            logger.error(f"缺少必需的配置项: {', '.join(missing_keys)}")
            return False

        return True

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        merged = base_config.copy()
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = InfrastructureConfig.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged


# ==================== 性能监控模式 ====================


class InfrastructurePerformanceMonitor:
    """基础设施层通用性能监控工具"""

    @staticmethod
    def measure_execution_time(func: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
        """测量函数执行时间"""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        return result, elapsed

    @staticmethod
    def log_performance(operation: str, elapsed: float, threshold: float = 1.0):
        """记录性能信息"""
        if elapsed > threshold:
            logger.warning(f"{operation} 操作耗时过长: {elapsed:.3f}秒")
        else:
            logger.debug(f"{operation} 操作耗时: {elapsed:.3f}秒")


# ==================== 装饰器模式 ====================


def infrastructure_operation(operation_name: str, log_success: bool = True):
    """基础设施操作装饰器"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                result, elapsed = InfrastructurePerformanceMonitor.measure_execution_time(
                    func, *args, **kwargs)

                if log_success:
                    InfrastructureLogger.log_operation_success(operation_name)

                InfrastructurePerformanceMonitor.log_performance(operation_name, elapsed)
                return result

            except Exception as e:
                InfrastructureExceptionHandler.handle_operation_error(operation_name, e)

        return wrapper
    return decorator


def safe_infrastructure_operation(operation_name: str, default_return: Any = None):
    """安全的基础设施操作装饰器"""
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                result, elapsed = InfrastructurePerformanceMonitor.measure_execution_time(
                    func, *args, **kwargs)
                InfrastructureLogger.log_operation_success(operation_name)
                InfrastructurePerformanceMonitor.log_performance(operation_name, elapsed)
                return result
            except Exception as e:
                logger.error(f"{operation_name} 操作失败: {e}", exc_info=True)
                return default_return

        return wrapper
    return decorator


__all__ = [
    # 核心工具类
    'InfrastructureLogger',
    'InfrastructureExceptionHandler',
    'InfrastructureInitializer',
    'InfrastructureConfig',
    'InfrastructurePerformanceMonitor',
    
    # 装饰器
    'infrastructure_operation',
    'safe_infrastructure_operation',
]

