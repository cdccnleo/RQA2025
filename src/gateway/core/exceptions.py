"""
网关层异常处理
Gateway Layer Exception Handling

定义API网关相关的异常类和错误处理机制
"""

from datetime import datetime


class GatewayException(Exception):
    """网关基础异常类"""

    def __init__(self, message: str, error_code: int = -1, status_code: int = 500):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code


class AuthenticationError(GatewayException):
    """认证错误"""

    def __init__(self, message: str, user_id: str = None):
        super().__init__(f"认证失败 - {user_id}: {message}", status_code=401)
        self.user_id = user_id


class AuthorizationError(GatewayException):
    """授权错误"""

    def __init__(self, message: str, resource: str = None):
        super().__init__(f"授权失败 - {resource}: {message}", status_code=403)
        self.resource = resource


class RateLimitError(GatewayException):
    """速率限制错误"""

    def __init__(self, message: str, client_ip: str = None):
        super().__init__(f"速率限制 - {client_ip}: {message}", status_code=429)
        self.client_ip = client_ip


class RoutingError(GatewayException):
    """路由错误"""

    def __init__(self, message: str, route_path: str = None):
        super().__init__(f"路由失败 - {route_path}: {message}", status_code=404)
        self.route_path = route_path


class UpstreamError(GatewayException):
    """上游服务错误"""

    def __init__(self, message: str, service_name: str = None):
        super().__init__(f"上游服务错误 - {service_name}: {message}", status_code=502)
        self.service_name = service_name


class RequestValidationError(GatewayException):
    """请求验证错误"""

    def __init__(self, message: str, field: str = None):
        super().__init__(f"请求验证失败 - {field}: {message}", status_code=400)
        self.field = field


class CircuitBreakerError(GatewayException):
    """熔断器错误"""

    def __init__(self, message: str, service_name: str = None):
        super().__init__(f"服务熔断 - {service_name}: {message}", status_code=503)
        self.service_name = service_name


class TimeoutError(GatewayException):
    """超时错误"""

    def __init__(self, message: str, timeout_seconds: int = None):
        super().__init__(f"请求超时 - {timeout_seconds}秒: {message}", status_code=504)
        self.timeout_seconds = timeout_seconds


class ResourceExhaustionError(GatewayException):
    """资源耗尽错误"""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(f"资源耗尽 - {resource_type}: {message}", status_code=503)
        self.resource_type = resource_type


class ConfigurationError(GatewayException):
    """配置错误"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(f"配置错误 - {config_key}: {message}", status_code=500)
        self.config_key = config_key


class WebSocketError(GatewayException):
    """WebSocket错误"""

    def __init__(self, message: str, connection_id: str = None):
        super().__init__(f"WebSocket错误 - {connection_id}: {message}", status_code=1011)
        self.connection_id = connection_id


def handle_gateway_exception(func):
    """
    装饰器：统一处理网关异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GatewayException:
            # 重新抛出网关异常
            raise
        except Exception as e:
            # 将其他异常包装为网关异常
            raise GatewayException(f"意外网关错误: {str(e)}") from e

    return wrapper


def validate_request_data(data: dict, required_fields: list):
    """
    验证请求数据

    Args:
        data: 请求数据字典
        required_fields: 必需字段列表

    Raises:
        RequestValidationError: 请求验证失败
    """
    if not data:
        raise RequestValidationError("请求数据不能为空")

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise RequestValidationError(f"缺少必需字段: {missing_fields}")


def validate_api_version(version: str):
    """
    验证API版本

    Args:
        version: API版本字符串

    Raises:
        RequestValidationError: 版本验证失败
    """
    from constants import SUPPORTED_API_VERSIONS

    if version not in SUPPORTED_API_VERSIONS:
        raise RequestValidationError(f"不支持的API版本: {version}")


def validate_request_size(content_length: int, max_size: int):
    """
    验证请求大小

    Args:
        content_length: 请求内容长度
        max_size: 最大允许大小

    Raises:
        RequestValidationError: 请求大小验证失败
    """
    if content_length > max_size:
        raise RequestValidationError(f"请求大小超过限制: {content_length} > {max_size}")


def check_rate_limit(current_requests: int, max_requests: int, time_window: int, client_id: str):
    """
    检查速率限制

    Args:
        current_requests: 当前请求数
        max_requests: 最大请求数
        time_window: 时间窗口(秒)
        client_id: 客户端ID

    Returns:
        是否允许请求

    Raises:
        RateLimitError: 超过速率限制
    """
    if current_requests >= max_requests:
        raise RateLimitError(
            f"超过速率限制: {current_requests}/{max_requests} 请求/{time_window}秒",
            client_id
        )

    return True


def validate_auth_token(token: str, required_scopes: list = None):
    """
    验证认证令牌

    Args:
        token: 认证令牌
        required_scopes: 必需的权限范围

    Raises:
        AuthenticationError: 令牌验证失败
    """
    if not token:
        raise AuthenticationError("认证令牌不能为空")

    # 这里可以添加具体的令牌验证逻辑
    # 例如JWT解码、过期检查、权限验证等

    if required_scopes:
        # 检查令牌是否包含必需的权限范围
        # 这里应该实现具体的权限检查逻辑
        pass


def get_error_response(exception: GatewayException) -> dict:
    """
    获取错误响应

    Args:
        exception: 网关异常

    Returns:
        错误响应字典
    """
    return {
        "error": {
            "code": exception.error_code,
            "message": exception.message,
            "type": exception.__class__.__name__
        },
        "status_code": exception.status_code,
        "timestamp": str(datetime.now())
    }
