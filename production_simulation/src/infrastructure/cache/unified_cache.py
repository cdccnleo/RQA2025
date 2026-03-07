"""
统一缓存模块（别名模块）
提供向后兼容的导入路径

实际实现在 core/cache_manager.py 中
"""

from .core.cache_manager import UnifiedCacheManager

# 别名：UnifiedCache = UnifiedCacheManager（向后兼容）
UnifiedCache = UnifiedCacheManager

__all__ = ['UnifiedCacheManager', 'UnifiedCache']

