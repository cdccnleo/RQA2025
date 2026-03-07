"""
弹性层异常处理
Resilience Layer Exception Handling

定义系统弹性相关的异常类和错误处理机制
"""


class ResilienceException(Exception):
    """弹性基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class CircuitBreakerException(ResilienceException):
    """熔断器异常"""

    def __init__(self, message: str, service_name: str = None):
        super().__init__(f"熔断器触发 - {service_name}: {message}")
        self.service_name = service_name


class RetryExhaustionException(ResilienceException):
    """重试耗尽异常"""

    def __init__(self, message: str, max_attempts: int = None):
        super().__init__(f"重试耗尽 - 最大尝试次数{max_attempts}: {message}")
        self.max_attempts = max_attempts


class DegradationException(ResilienceException):
    """降级异常"""

    def __init__(self, message: str, degradation_level: int = None):
        super().__init__(f"服务降级 - 级别{degradation_level}: {message}")
        self.degradation_level = degradation_level


class TimeoutException(ResilienceException):
    """超时异常"""

    def __init__(self, message: str, timeout_seconds: int = None):
        super().__init__(f"操作超时 - {timeout_seconds}秒: {message}")
        self.timeout_seconds = timeout_seconds


class ResourceExhaustionException(ResilienceException):
    """资源耗尽异常"""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(f"资源耗尽 - {resource_type}: {message}")
        self.resource_type = resource_type


class LoadBalancerException(ResilienceException):
    """负载均衡器异常"""

    def __init__(self, message: str, balancer_name: str = None):
        super().__init__(f"负载均衡失败 - {balancer_name}: {message}")
        self.balancer_name = balancer_name


class HealthCheckException(ResilienceException):
    """健康检查异常"""

    def __init__(self, message: str, check_target: str = None):
        super().__init__(f"健康检查失败 - {check_target}: {message}")
        self.check_target = check_target


class RecoveryException(ResilienceException):
    """恢复异常"""

    def __init__(self, message: str, recovery_type: str = None):
        super().__init__(f"恢复失败 - {recovery_type}: {message}")
        self.recovery_type = recovery_type


class PoolExhaustionException(ResilienceException):
    """连接池耗尽异常"""

    def __init__(self, message: str, pool_name: str = None):
        super().__init__(f"连接池耗尽 - {pool_name}: {message}")
        self.pool_name = pool_name


def handle_resilience_exception(func):
    """
    装饰器：统一处理弹性异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ResilienceException:
            # 重新抛出弹性异常
            raise
        except Exception as e:
            # 将其他异常包装为弹性异常
            raise ResilienceException(f"意外弹性错误: {str(e)}") from e

    return wrapper


def circuit_breaker_check(failure_count: int, threshold: int, service_name: str):
    """
    检查是否需要触发熔断器

    Args:
        failure_count: 失败计数
        threshold: 阈值
        service_name: 服务名称

    Raises:
        CircuitBreakerException: 熔断器触发
    """
    if failure_count >= threshold:
        raise CircuitBreakerException(f"失败次数超过阈值: {failure_count}/{threshold}", service_name)


def check_retry_limit(current_attempt: int, max_attempts: int, operation_name: str):
    """
    检查重试限制

    Args:
        current_attempt: 当前尝试次数
        max_attempts: 最大尝试次数
        operation_name: 操作名称

    Raises:
        RetryExhaustionException: 重试耗尽
    """
    if current_attempt >= max_attempts:
        raise RetryExhaustionException(f"操作重试耗尽: {operation_name}", max_attempts)


def validate_degradation_level(level: int, max_level: int):
    """
    验证降级级别

    Args:
        level: 当前降级级别
        max_level: 最大降级级别

    Raises:
        DegradationException: 降级级别无效
    """
    if level < 0 or level > max_level:
        raise DegradationException(f"无效的降级级别: {level}，应在0-{max_level}之间", level)


def check_resource_limits(current_usage: float, limit: float, resource_type: str):
    """
    检查资源限制

    Args:
        current_usage: 当前使用量
        limit: 限制值
        resource_type: 资源类型

    Raises:
        ResourceExhaustionException: 资源超限
    """
    if current_usage >= limit:
        raise ResourceExhaustionException(
            f"{resource_type} 使用量超过限制: {current_usage}/{limit}", resource_type)


def validate_timeout_config(timeout: int, max_timeout: int = 300):
    """
    验证超时配置

    Args:
        timeout: 超时时间
        max_timeout: 最大超时时间

    Raises:
        TimeoutException: 超时配置无效
    """
    if timeout <= 0:
        raise TimeoutException("超时时间必须大于0", timeout)
    if timeout > max_timeout:
        raise TimeoutException(f"超时时间不能超过最大值: {timeout} > {max_timeout}", timeout)


def check_health_status(is_healthy: bool, check_type: str, target_name: str):
    """
    检查健康状态

    Args:
        is_healthy: 是否健康
        check_type: 检查类型
        target_name: 目标名称

    Raises:
        HealthCheckException: 健康检查失败
    """
    if not is_healthy:
        raise HealthCheckException(f"{check_type} 健康检查失败", target_name)


def validate_pool_size(current_size: int, max_size: int, pool_name: str):
    """
    验证连接池大小

    Args:
        current_size: 当前大小
        max_size: 最大大小
        pool_name: 池名称

    Raises:
        PoolExhaustionException: 池耗尽
    """
    if current_size >= max_size:
        raise PoolExhaustionException(f"连接池已满: {current_size}/{max_size}", pool_name)
