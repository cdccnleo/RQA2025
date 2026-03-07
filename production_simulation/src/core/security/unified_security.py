"""
Unified Security别名模块

安全模块已迁移至基础设施层
提供向后兼容的导入路径
"""

from src.infrastructure.security.core.security import SecurityManager

# 别名
UnifiedSecurity = SecurityManager

__all__ = ['UnifiedSecurity', 'SecurityManager']

