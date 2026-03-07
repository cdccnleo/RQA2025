"""
__init__ 模块

提供 __init__ 相关功能和接口。
"""

from .cache_strategy_manager import CacheStrategyManager
"""
缓存策略实现

包含各种缓存策略的实现：
- 淘汰策略
- 缓存策略
- 策略管理器
"""

try:
    from .cache_strategy_manager import CacheStrategyManager
except ImportError:
    CacheStrategyManager = None

__all__ = [
    'CacheStrategyManager'
]
