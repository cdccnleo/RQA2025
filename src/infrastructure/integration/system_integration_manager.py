"""
系统集成管理器模块（别名模块）
提供向后兼容的导入路径
"""

try:
    from .core.system_integration_manager import SystemIntegrationManager
except ImportError:
    # 提供基础实现
    class SystemIntegrationManager:
        pass

__all__ = ['SystemIntegrationManager']

