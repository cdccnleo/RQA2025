
#!/usr/bin/env python3
"""
统一配置管理器 (重构后)

导入完整功能的配置管理器实现
所有功能已拆分到专门的模块中
"""

# 向后兼容的别名
try:
    from .config_manager_complete import UnifiedConfigManager
except ImportError:
    # 如果导入失败，提供占位符
    UnifiedConfigManager = None

ConfigManager = UnifiedConfigManager

__all__ = ['UnifiedConfigManager', 'ConfigManager']




