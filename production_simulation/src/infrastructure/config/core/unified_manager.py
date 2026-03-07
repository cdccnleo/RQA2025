
# 向后兼容的别名

from .config_manager_complete import UnifiedConfigManager
#!/usr/bin/env python3
"""
统一配置管理器 (重构后)

导入完整功能的配置管理器实现
所有功能已拆分到专门的模块中
"""

ConfigManager = UnifiedConfigManager

__all__ = ['UnifiedConfigManager', 'ConfigManager']




