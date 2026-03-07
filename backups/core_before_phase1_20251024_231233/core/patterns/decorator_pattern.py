"""
RQA2025系统装饰器模式实现
Decorator Pattern Implementation for RQA2025 System

实现装饰器模式，为现有功能动态添加行为而不改变接口
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar, Type
from functools import wraps
import time
import logging
import json

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class FunctionDecorator:
    """
    函数装饰器基类

    实现装饰器模式，为函数添加额外的行为
    """

    def __init__(self, func: F):
        self.func = func
        self._decorated_func = None

    def __call__(self, *args, **kwargs):
        """使装饰器实例可以直接调用"""
        if self._decorated_func is None:
            self._decorated_func = self.decorate(self.func)

        return self._decorated_func(*args, **kwargs)

    def decorate(self, func: F) -> F:
        """
        装饰函数

        Args:
            func: 要装饰的函数

        Returns:
            装饰后的函数
        """
        raise NotImplementedError("子类必须实现decorate方法")

    def get_decorator_info(self) -> Dict[str, Any]:
        """
        获取装饰器信息

        Returns:
            装饰器信息
        """
        return {
            'decorator_class': self.__class__.__name__,
            'decorated_function': self.func.__name__ if hasattr(self.func, '__name__') else str(self.func),
            'module': self.__class__.__module__
        }


class CachingDecorator(FunctionDecorator):
    """
    缓存装饰器

    为函数添加缓存功能，避免重复计算
    """

    def __init__(self, func: F, cache_store: Optional[Any] = None, ttl: int = 300):
        super().__init__(func)
        self.cache_store = cache_store
        self.ttl = ttl
        self._cache = {}  # 简单的内存缓存

    def decorate(self, func: F) -> F:
        """装饰函数，添加缓存功能"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = self._generate_cache_key(func, args, kwargs)

            # 尝试从缓存获取
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.debug(f"缓存命中: {func.__name__}")
                return cached_result

            # 执行函数
            result = func(*args, **kwargs)

            # 存储到缓存
            self._store_in_cache(cache_key, result)

            logger.debug(f"缓存存储: {func.__name__}")
            return result

        return wrapper

    def _generate_cache_key(self, func: F, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 序列化参数
        try:
            args_str = json.dumps(args, sort_keys=True, default=str)
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
            return f"{func.__module__}.{func.__name__}:{args_str}:{kwargs_str}"
        except (TypeError, ValueError):
            # 如果无法序列化，使用函数名和参数的字符串表示
            return f"{func.__module__}.{func.__name__}:{str(args)}:{str(kwargs)}"

    def _get_from_cache(self, key: str) -> Any:
        """从缓存获取数据"""
        if self.cache_store:
            # 使用外部缓存存储
            if hasattr(self.cache_store, 'get'):
                return self.cache_store.get(key)
        else:
            # 使用内置缓存
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    return value
                else:
                    # 过期删除
                    del self._cache[key]
        return None

    def _store_in_cache(self, key: str, value: Any) -> None:
        """存储数据到缓存"""
        expiry = time.time() + self.ttl

        if self.cache_store:
            # 使用外部缓存存储
            if hasattr(self.cache_store, 'set'):
                self.cache_store.set(key, value, ttl=self.ttl)
        else:
            # 使用内置缓存
            self._cache[key] = (value, expiry)

    def clear_cache(self) -> None:
        """清空缓存"""
        if self.cache_store and hasattr(self.cache_store, 'clear'):
            self.cache_store.clear()
        else:
            self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if self.cache_store and hasattr(self.cache_store, 'get_stats'):
            return self.cache_store.get_stats()
        else:
            return {
                'cached_items': len(self._cache),
                'ttl': self.ttl
            }


class LoggingDecorator(FunctionDecorator):
    """
    日志装饰器

    为函数添加详细的日志记录
    """

    def __init__(self, func: F, log_level: str = 'INFO', include_args: bool = True,
                 include_result: bool = False, exclude_sensitive: Optional[List[str]] = None):
        super().__init__(func)
        self.log_level = log_level
        self.include_args = include_args
        self.include_result = include_result
        self.exclude_sensitive = exclude_sensitive or ['password', 'token', 'secret', 'key']

    def decorate(self, func: F) -> F:
        """装饰函数，添加日志功能"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # 记录开始执行
            logger.log(getattr(logging, self.log_level),
                       f"开始执行函数: {func.__name__}")

            if self.include_args:
                # 过滤敏感信息
                safe_args = self._filter_sensitive_args(args, kwargs)
                logger.log(getattr(logging, self.log_level),
                           f"函数参数: {safe_args}")

            try:
                # 执行函数
                result = func(*args, **kwargs)

                execution_time = time.time() - start_time

                # 记录成功执行
                logger.log(getattr(logging, self.log_level),
                           f"函数执行成功: {func.__name__}, 耗时: {execution_time:.3f}s")

                if self.include_result:
                    safe_result = self._filter_sensitive_result(result)
                    logger.log(getattr(logging, self.log_level),
                               f"函数结果: {safe_result}")

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                # 记录异常
                logger.error(f"函数执行失败: {func.__name__}, 耗时: {execution_time:.3f}s, 错误: {e}")

                # 重新抛出异常
                raise

        return wrapper

    def _filter_sensitive_args(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """过滤敏感参数"""
        # 这里简化处理，实际应该更复杂
        filtered_kwargs = {}

        # 处理位置参数（这里假设第一个参数不是敏感的）
        if args:
            filtered_kwargs['args_count'] = len(args)

        # 处理关键字参数
        for key, value in kwargs.items():
            if any(sensitive in key.lower() for sensitive in self.exclude_sensitive):
                filtered_kwargs[key] = "***FILTERED***"
            else:
                filtered_kwargs[key] = self._truncate_value(value)

        return filtered_kwargs

    def _filter_sensitive_result(self, result: Any) -> Any:
        """过滤敏感结果"""
        if isinstance(result, dict):
            filtered = {}
            for key, value in result.items():
                if any(sensitive in key.lower() for sensitive in self.exclude_sensitive):
                    filtered[key] = "***FILTERED***"
                else:
                    filtered[key] = self._truncate_value(value)
            return filtered
        else:
            return self._truncate_value(result)

    def _truncate_value(self, value: Any, max_length: int = 100) -> Any:
        """截断过长的值"""
        value_str = str(value)
        if len(value_str) > max_length:
            return value_str[:max_length] + "..."
        return value


class PerformanceMonitoringDecorator(FunctionDecorator):
    """
    性能监控装饰器

    监控函数的执行时间和性能指标
    """

    def __init__(self, func: F, threshold_ms: float = 1000.0,
                 monitor: Optional[Any] = None):
        super().__init__(func)
        self.threshold_ms = threshold_ms
        self.monitor = monitor
        self._execution_times: List[float] = []
        self._call_count = 0

    def decorate(self, func: F) -> F:
        """装饰函数，添加性能监控"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                execution_time = (time.time() - start_time) * 1000  # 转换为毫秒

                # 记录执行时间
                self._record_execution_time(execution_time)

                # 检查是否超过阈值
                if execution_time > self.threshold_ms:
                    logger.warning(f"函数执行时间过长: {func.__name__} - {execution_time:.2f}ms")

                # 发送到监控系统
                if self.monitor:
                    self._send_to_monitor(func.__name__, execution_time, True)

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000

                # 记录失败的执行
                self._record_execution_time(execution_time)

                # 发送错误到监控系统
                if self.monitor:
                    self._send_to_monitor(func.__name__, execution_time, False, str(e))

                raise

        return wrapper

    def _record_execution_time(self, execution_time: float) -> None:
        """记录执行时间"""
        self._execution_times.append(execution_time)
        self._call_count += 1

        # 保持最近1000次调用的记录
        if len(self._execution_times) > 1000:
            self._execution_times.pop(0)

    def _send_to_monitor(self, func_name: str, execution_time: float,
                         success: bool, error: Optional[str] = None) -> None:
        """发送监控数据"""
        if hasattr(self.monitor, 'record_metric'):
            self.monitor.record_metric(
                name=f"function.{func_name}.execution_time",
                value=execution_time,
                tags={'success': success}
            )

            if not success and error:
                self.monitor.record_metric(
                    name=f"function.{func_name}.error",
                    value=1,
                    tags={'error_type': error[:50]}
                )

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if not self._execution_times:
            return {'call_count': 0}

        return {
            'call_count': self._call_count,
            'avg_execution_time': sum(self._execution_times) / len(self._execution_times),
            'max_execution_time': max(self._execution_times),
            'min_execution_time': min(self._execution_times),
            'threshold_ms': self.threshold_ms
        }


class RetryDecorator(FunctionDecorator):
    """
    重试装饰器

    为函数添加自动重试功能
    """

    def __init__(self, func: F, max_retries: int = 3, delay: float = 1.0,
                 backoff: float = 2.0, exceptions: tuple = (Exception,)):
        super().__init__(func)
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions

    def decorate(self, func: F) -> F:
        """装饰函数，添加重试功能"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = self.delay

            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e

                    if attempt < self.max_retries:
                        logger.warning(
                            f"函数执行失败，重试中: {func.__name__} (尝试 {attempt + 1}/{self.max_retries + 1})")
                        time.sleep(current_delay)
                        current_delay *= self.backoff
                    else:
                        logger.error(f"函数执行失败，已达到最大重试次数: {func.__name__}")
                        break

            # 所有重试都失败，抛出最后一次异常
            raise last_exception

        return wrapper


class ValidationDecorator(FunctionDecorator):
    """
    验证装饰器

    为函数添加输入验证功能
    """

    def __init__(self, func: F, validators: Optional[Dict[str, Callable]] = None):
        super().__init__(func)
        self.validators = validators or {}

    def decorate(self, func: F) -> F:
        """装饰函数，添加验证功能"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 验证参数
            self._validate_args(func, args, kwargs)

            # 执行函数
            result = func(*args, **kwargs)

            # 验证结果（如果有结果验证器）
            self._validate_result(func, result)

            return result

        return wrapper

    def _validate_args(self, func: F, args: tuple, kwargs: dict) -> None:
        """验证函数参数"""
        # 这里可以根据函数名或参数名应用不同的验证器
        for validator_name, validator in self.validators.items():
            if validator_name == 'args':
                validator(args)
            elif validator_name == 'kwargs':
                validator(kwargs)
            elif validator_name in kwargs:
                validator(kwargs[validator_name])

    def _validate_result(self, func: F, result: Any) -> None:
        """验证函数结果"""
        if 'result' in self.validators:
            self.validators['result'](result)


# ==================== 装饰器工厂 ====================

class DecoratorFactory:
    """
    装饰器工厂

    创建和管理各种装饰器实例
    """

    def __init__(self):
        self._decorator_types: Dict[str, Type[FunctionDecorator]] = {}

    def register_decorator_type(self, decorator_type: str, decorator_class: Type[FunctionDecorator]) -> None:
        """
        注册装饰器类型

        Args:
            decorator_type: 装饰器类型名称
            decorator_class: 装饰器类
        """
        self._decorator_types[decorator_type] = decorator_class

    def create_decorator(self, decorator_type: str, func: F, **kwargs) -> FunctionDecorator:
        """
        创建装饰器实例

        Args:
            decorator_type: 装饰器类型
            func: 要装饰的函数
            **kwargs: 装饰器参数

        Returns:
            装饰器实例
        """
        if decorator_type not in self._decorator_types:
            raise ValueError(f"未注册的装饰器类型: {decorator_type}")

        decorator_class = self._decorator_types[decorator_type]
        return decorator_class(func, **kwargs)

    def get_supported_decorator_types(self) -> List[str]:
        """
        获取支持的装饰器类型

        Returns:
            装饰器类型列表
        """
        return list(self._decorator_types.keys())


# 全局装饰器工厂实例
global_decorator_factory = DecoratorFactory()

# 注册默认装饰器类型
global_decorator_factory.register_decorator_type('cache', CachingDecorator)
global_decorator_factory.register_decorator_type('logging', LoggingDecorator)
global_decorator_factory.register_decorator_type('performance', PerformanceMonitoringDecorator)
global_decorator_factory.register_decorator_type('retry', RetryDecorator)
global_decorator_factory.register_decorator_type('validation', ValidationDecorator)


# ==================== 便捷装饰器函数 ====================

def cached(ttl: int = 300, cache_store: Optional[Any] = None) -> Callable[[F], F]:
    """
    缓存装饰器

    Args:
        ttl: 缓存过期时间（秒）
        cache_store: 缓存存储对象

    Returns:
        装饰器函数
    """
    def decorator(func: F) -> F:
        caching_decorator = CachingDecorator(func, cache_store, ttl)
        return caching_decorator.decorate(func)
    return decorator


def logged(log_level: str = 'INFO', include_args: bool = True,
           include_result: bool = False) -> Callable[[F], F]:
    """
    日志装饰器

    Args:
        log_level: 日志级别
        include_args: 是否记录参数
        include_result: 是否记录结果

    Returns:
        装饰器函数
    """
    def decorator(func: F) -> F:
        logging_decorator = LoggingDecorator(func, log_level, include_args, include_result)
        return logging_decorator.decorate(func)
    return decorator


def monitored(threshold_ms: float = 1000.0, monitor: Optional[Any] = None) -> Callable[[F], F]:
    """
    性能监控装饰器

    Args:
        threshold_ms: 阈值（毫秒）
        monitor: 监控对象

    Returns:
        装饰器函数
    """
    def decorator(func: F) -> F:
        monitoring_decorator = PerformanceMonitoringDecorator(func, threshold_ms, monitor)
        return monitoring_decorator.decorate(func)
    return decorator


def retried(max_retries: int = 3, delay: float = 1.0,
            backoff: float = 2.0) -> Callable[[F], F]:
    """
    重试装饰器

    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间
        backoff: 退避倍数

    Returns:
        装饰器函数
    """
    def decorator(func: F) -> F:
        retry_decorator = RetryDecorator(func, max_retries, delay, backoff)
        return retry_decorator.decorate(func)
    return decorator


def validated(validators: Optional[Dict[str, Callable]] = None) -> Callable[[F], F]:
    """
    验证装饰器

    Args:
        validators: 验证器字典

    Returns:
        装饰器函数
    """
    def decorator(func: F) -> F:
        validation_decorator = ValidationDecorator(func, validators)
        return validation_decorator.decorate(func)
    return decorator


# ==================== 组合装饰器 ====================

def create_composite_decorator(*decorators: Callable[[F], F]) -> Callable[[F], F]:
    """
    创建组合装饰器

    Args:
        *decorators: 要组合的装饰器

    Returns:
        组合装饰器
    """
    def composite_decorator(func: F) -> F:
        # 从右到左应用装饰器
        result = func
        for decorator in reversed(decorators):
            result = decorator(result)
        return result

    return composite_decorator


# 预定义的组合装饰器
def robust_service(ttl: int = 300, log_level: str = 'INFO',
                   threshold_ms: float = 1000.0) -> Callable[[F], F]:
    """
    健壮的服务装饰器组合

    包括缓存、日志、性能监控功能

    Args:
        ttl: 缓存时间
        log_level: 日志级别
        threshold_ms: 性能阈值

    Returns:
        组合装饰器
    """
    return create_composite_decorator(
        cached(ttl=ttl),
        logged(log_level=log_level),
        monitored(threshold_ms=threshold_ms)
    )


def resilient_operation(max_retries: int = 3, log_level: str = 'WARNING') -> Callable[[F], F]:
    """
    弹性操作装饰器组合

    包括重试和日志功能

    Args:
        max_retries: 最大重试次数
        log_level: 日志级别

    Returns:
        组合装饰器
    """
    return create_composite_decorator(
        retried(max_retries=max_retries),
        logged(log_level=log_level)
    )
