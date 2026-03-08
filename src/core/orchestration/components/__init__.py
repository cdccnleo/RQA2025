"""
Orchestration Components模块

导出编排器所需的组件
"""

# 导入EventBus
from ...event_bus.core import EventBus

# 导入其他组件（提供fallback）
try:
    from .state_machine import BusinessProcessStateMachine
except ImportError:
    from ...business_process.state_machine.state_machine import BusinessProcessStateMachine

try:
    from .config_manager import ProcessConfigManager
except ImportError:
    from ...business_process.config.config import ProcessConfigManager

try:
    from .process_monitor import ProcessMonitor
except ImportError:
    from ...business_process.monitor.monitor import ProcessMonitor

try:
    from .instance_pool import ProcessInstancePool
except ImportError:
    # 提供基础实现
    class ProcessInstancePool:
        def __init__(self):
            self.processes = {}

__all__ = [
    'EventBus',
    'BusinessProcessStateMachine',
    'ProcessConfigManager',
    'ProcessMonitor',
    'ProcessInstancePool'
]
