"""
回滚模块

提供自动回滚的触发、执行和记录功能
"""

from .rollback_manager import (
    RollbackManager,
    RollbackTrigger,
    RollbackRecord,
    RollbackStrategy,
    RollbackTriggerType,
    DEFAULT_ROLLBACK_TRIGGERS,
    get_rollback_manager,
    reset_rollback_manager
)

__all__ = [
    "RollbackManager",
    "RollbackTrigger",
    "RollbackRecord",
    "RollbackStrategy",
    "RollbackTriggerType",
    "DEFAULT_ROLLBACK_TRIGGERS",
    "get_rollback_manager",
    "reset_rollback_manager"
]
