"""
核心服务层 - 业务流程编排模块

此模块提供业务流程编排的核心功能，包括：
- 业务流程编排器 (BusinessProcessOrchestrator)
- 流程实例池管理 (ProcessInstancePool)
- 事件驱动架构 (EventBus)
- 业务流程状态机 (BusinessProcessStateMachine)

作者: RQA2025 Team
版本: 3.0.0
更新时间: 2025-09-30
"""

from .business_process_orchestrator import BusinessProcessOrchestrator, BusinessProcessStateMachine
from .pool.process_instance_pool import ProcessInstancePool

__all__ = [
    'BusinessProcessOrchestrator',
    'ProcessInstancePool',
    'BusinessProcessStateMachine'
]

__version__ = "3.0.0"
__author__ = "RQA2025 Team"
