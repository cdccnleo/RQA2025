"""
健康适配器模块（别名模块）
提供向后兼容的导入路径

实际实现在 integration/health/health_adapter.py 中
"""

try:
    from .integration.health.health_adapter import HealthAdapter, HealthLayerAdapter
except ImportError:
    try:
        from .health.health_adapter import HealthAdapter, HealthLayerAdapter
    except ImportError:
        # 提供基础实现
        class HealthAdapter:
            pass
        
        class HealthLayerAdapter:
            pass

__all__ = ['HealthAdapter', 'HealthLayerAdapter']

