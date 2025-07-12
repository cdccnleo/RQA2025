"""
配置管理模块主入口
版本更新记录：
2024-04-02 v3.6.0 - 重构配置管理模块
    - 统一接口定义
    - 增强版本控制
    - 完善事件系统
"""

from .config_manager import ConfigManager
from .deployment_manager import DeploymentManager
from .factory import ConfigFactory
from .services.lock_manager import LockManager
from .version_service import VersionService
from .security_service import SecurityService

# 主要服务导出
__all__ = [
    'ConfigManager',
    'DeploymentManager',
    'ConfigFactory',
    'LockManager',
    'VersionService',
    'SecurityService'
]
