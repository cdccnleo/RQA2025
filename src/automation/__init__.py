"""
自动化层 - 工作流自动化和任务调度
RQA2025量化交易系统自动化层
"""

__version__ = "1.0.0"
__all__ = ['get_automation_manager']

from .automation_manager import AutomationManager

_default_manager = None

def get_automation_manager():
    """获取默认管理器实例"""
    global _default_manager
    if _default_manager is None:
        _default_manager = AutomationManager()
    return _default_manager
