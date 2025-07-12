class SecurityError(Exception):
    """安全相关异常，如配置签名验证失败、敏感数据访问违规等"""
    def __init__(self, message="Security violation", details=None):
        super().__init__(message)
        self.details = details or {}
        self.code = "SECURITY_VIOLATION"

class ConfigValidationError(Exception):
    """配置验证失败异常"""
    def __init__(self, message="Configuration validation failed", errors=None):
        super().__init__(message)
        self.errors = errors or []
        self.code = "VALIDATION_ERROR"

class ConfigLoadError(Exception):
    """配置加载失败异常"""
    def __init__(self, message="Failed to load configuration", source=None, context=None):
        super().__init__(message)
        self.source = source
        self.context = context or {}
        self.code = "LOAD_ERROR"

class TradingConfigError(Exception):
    """交易相关配置异常"""
    def __init__(self, message="Trading configuration error", config_key=None):
        super().__init__(message)
        self.config_key = config_key
        self.code = "TRADING_CONFIG_ERROR"

class ConfigError(Exception):
    """配置错误基类"""
    def __init__(self, message="Configuration error", details=None):
        super().__init__(message)
        self.details = details or {}
        self.code = "CONFIG_ERROR"
