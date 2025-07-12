class InfrastructureError(Exception):
    """基础设施层异常基类"""
    pass

class CriticalError(InfrastructureError):
    """严重错误，需要立即处理"""
    pass

class WarningError(InfrastructureError):
    """警告级别错误，可以继续运行但需要注意"""
    pass

class InfoLevelError(InfrastructureError):
    """信息级别错误，仅用于记录"""
    pass

class RetryableError(InfrastructureError):
    """可重试的错误"""
    def __init__(self, message: str, max_retries: int = 3):
        super().__init__(message)
        self.max_retries = max_retries

class RetryError(RetryableError):
    """重试失败异常"""
    def __init__(self, message: str, attempts: int, max_retries: int = 3):
        super().__init__(message, max_retries)
        self.attempts = attempts

class TradingError(InfrastructureError):
    """交易相关错误基类"""
    # 添加枚举属性以满足测试期望
    ORDER_REJECTED = "ORDER_REJECTED"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    INVALID_PRICE = "INVALID_PRICE"
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INVALID_ORDER = "INVALID_ORDER"
    MARKET_CLOSED = "MARKET_CLOSED"
    INSUFFICIENT_LIQUIDITY = "INSUFFICIENT_LIQUIDITY"
    PRICE_SLIPPAGE = "PRICE_SLIPPAGE"
    PARTIAL_FILL = "PARTIAL_FILL"
    CANCEL_FAILED = "CANCEL_FAILED"
    MODIFY_FAILED = "MODIFY_FAILED"
    QUOTE_EXPIRED = "QUOTE_EXPIRED"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    ACCOUNT_LOCKED = "ACCOUNT_LOCKED"
    MAINTENANCE_MODE = "MAINTENANCE_MODE"

class OrderRejectedError(TradingError):
    """订单被拒绝异常"""
    def __init__(self, order_id: str, reason: str):
        self.order_id = order_id
        self.reason = reason
        super().__init__(f"Order {order_id} rejected: {reason}")

class InvalidPriceError(TradingError):
    """无效价格异常"""
    def __init__(self, price: float, valid_range: tuple):
        self.price = price
        self.valid_range = valid_range
        super().__init__(f"Invalid price {price}, valid range: {valid_range}")

class ConfigurationError(InfrastructureError):
    """配置相关错误"""
    pass

class ConfigError(ConfigurationError):
    """配置错误（别名）"""
    pass

class ValidationError(ConfigurationError):
    """验证错误"""
    pass

class TimeoutError(InfrastructureError):
    """超时错误"""
    pass

class PerformanceThresholdExceeded(WarningError):
    """性能指标超过阈值"""
    pass

class ResourceUnavailableError(RetryableError):
    """请求的资源不可用"""
    pass

class CircuitBreakerOpenError(RetryableError):
    """熔断器打开时抛出的异常"""
    def __init__(self, breaker_name: str, retry_after: float = None):
        """
        初始化熔断器异常
        
        Args:
            breaker_name: 熔断器名称
            retry_after: 可重试时间(秒)
        """
        self.breaker_name = breaker_name
        self.retry_after = retry_after
        message = f"Circuit breaker '{breaker_name}' is open"
        if retry_after:
            message += f", retry after {retry_after:.1f} seconds"
        super().__init__(message, max_retries=1)  # 熔断器错误通常只重试一次

class RecoveryFailedError(CriticalError):
    """自动恢复失败时抛出的异常"""
    def __init__(self, recovery_step: str, reason: str):
        """
        初始化恢复失败异常
        
        Args:
            recovery_step: 失败的恢复步骤
            reason: 失败原因
        """
        self.recovery_step = recovery_step
        self.reason = reason
        super().__init__(f"Recovery failed at step '{recovery_step}': {reason}")

class CacheError(InfrastructureError):
    """缓存系统基础异常"""
    pass

class CacheMemoryError(CacheError):
    """缓存内存不足异常"""
    def __init__(self, requested: int, available: int):
        """
        初始化内存不足异常
        
        Args:
            requested: 请求的内存大小(bytes)
            available: 可用的内存大小(bytes)
        """
        self.requested = requested
        self.available = available
        super().__init__(
            f"Cache memory insufficient: requested {requested:,} bytes, "
            f"only {available:,} bytes available"
        )

class CacheConcurrencyError(CacheError):
    """缓存并发访问异常"""
    def __init__(self, key: str, operation: str):
        """
        初始化并发访问异常
        
        Args:
            key: 发生冲突的缓存键
            operation: 操作类型(get/set/delete)
        """
        self.key = key
        self.operation = operation
        super().__init__(
            f"Concurrent cache access detected for key '{key}' during {operation}"
        )

class CacheSerializationError(CacheError):
    """缓存序列化/反序列化异常"""
    def __init__(self, key: str, error: str):
        """
        初始化序列化异常
        
        Args:
            key: 发生错误的缓存键
            error: 错误详情
        """
        self.key = key
        super().__init__(f"Cache serialization failed for key '{key}': {error}")

class DatabaseError(InfrastructureError):
    """数据库相关错误"""
    pass

class DatabaseConnectionError(DatabaseError):
    """数据库连接错误"""
    pass

class DatabaseQueryError(DatabaseError):
    """数据库查询错误"""
    pass

class DatabaseTransactionError(DatabaseError):
    """数据库事务错误"""
    pass

class NetworkError(InfrastructureError):
    """网络相关错误"""
    pass

class ResourceError(InfrastructureError):
    """资源相关错误"""
    pass

class SecurityError(InfrastructureError):
    """安全相关错误"""
    pass
