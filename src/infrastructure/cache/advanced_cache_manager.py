"""
高级缓存管理器模块（别名模块）
提供向后兼容的导入路径

实际实现在 core/cache_manager.py 中
"""

try:
    from .core.cache_manager import UnifiedCacheManager as AdvancedCacheManager
    from .core.cache_manager import CacheManager
except ImportError:
    # 提供基础实现
    class AdvancedCacheManager:
        pass
    
    CacheManager = AdvancedCacheManager

__all__ = ['AdvancedCacheManager', 'CacheManager']

