"""
事件持久化模块

提供多种持久化模式支持：
- MEMORY: 内存模式，适用于开发和测试
- FILE: 文件模式，适用于单机部署
- DATABASE: 数据库模式，适用于生产环境
"""

from .event_persistence import (
    EventPersistence,
    PersistenceMode,
    EventStatus,
    PersistedEvent
)

# 导入数据库持久化（如果可用）
try:
    from .event_persistence_db import (
        DatabaseEventPersistence,
        DatabaseEventPersistenceConfig
    )
    DATABASE_PERSISTENCE_AVAILABLE = True
except ImportError:
    DATABASE_PERSISTENCE_AVAILABLE = False
    DatabaseEventPersistence = None
    DatabaseEventPersistenceConfig = None

__all__ = [
    'EventPersistence',
    'PersistenceMode',
    'EventStatus',
    'PersistedEvent'
]

# 如果数据库持久化可用，添加到导出列表
if DATABASE_PERSISTENCE_AVAILABLE:
    __all__.extend([
        'DatabaseEventPersistence',
        'DatabaseEventPersistenceConfig'
    ])
