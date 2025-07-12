"""
配置管理服务实现
版本更新记录：
2024-04-02 v3.6.0 - 服务模块重构
    - 实现服务化架构
    - 增强缓存协调
    - 完善事件处理
2024-04-05 v3.6.1 - 新增锁管理服务
    - 添加LockManager统一锁管理
"""

from .cache_service import CacheService
from .config_loader_service import ConfigLoaderService
from .diff_service import DictDiffService
from .event_service import ConfigEventBus
from .lock_manager import LockManager
from .version_service import VersionService
from .validators import validate_trading_hours

EventService = ConfigEventBus

__all__ = [
    'CacheService',
    'ConfigLoaderService',
    'DictDiffService',
    'ConfigEventBus',
    'LockManager',
    'VersionService',
    'EventService',
    'validate_trading_hours'
]
