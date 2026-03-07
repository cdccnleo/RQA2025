#!/usr/bin/env python3
"""
特征层降级服务模块

提供基础设施服务不可用时的降级实现，确保特征层功能稳定运行
"""

from .config_fallback import FallbackConfigManager

__all__ = [
    'FallbackConfigManager',
]

# 可选导入
try:
    from .cache_fallback import FallbackCacheManager
    __all__.append('FallbackCacheManager')
except ImportError:
    pass

try:
    from .monitoring_fallback import FallbackMonitoring
    __all__.append('FallbackMonitoring')
except ImportError:
    pass
