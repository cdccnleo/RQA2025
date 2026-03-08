"""
核心安全模块

提供全面的安全功能，包括：
- JWT认证与授权
- 日志脱敏与敏感数据过滤
- 安全HTTP头中间件
- 统一安全接口

使用示例:
    from src.core.security import (
        JWTAuth,
        LogSanitizer,
        SecurityHeadersMiddleware,
        sanitize_log_message
    )
"""

# JWT认证
from .jwt_auth import (
    JWTAuth,
    JWTConfig,
    TokenPayload,
    require_auth,
    create_token,
    verify_token,
)

# 日志脱敏
from .log_sanitizer import (
    LogSanitizer,
    SanitizerConfig,
    sanitize_log_message,
    SensitivePattern,
)

# 安全HTTP头
from .security_headers import (
    SecurityHeadersMiddleware,
    SecurityConfig,
    create_security_headers,
)

# 基础安全接口
from .base_security import BaseSecurity
from .unified_security import UnifiedSecurity

__version__ = "1.0.0"

__all__ = [
    # JWT认证
    "JWTAuth",
    "JWTConfig",
    "TokenPayload",
    "require_auth",
    "create_token",
    "verify_token",

    # 日志脱敏
    "LogSanitizer",
    "SanitizerConfig",
    "sanitize_log_message",
    "SensitivePattern",

    # 安全HTTP头
    "SecurityHeadersMiddleware",
    "SecurityConfig",
    "create_security_headers",

    # 基础接口
    "BaseSecurity",
    "UnifiedSecurity",
]
