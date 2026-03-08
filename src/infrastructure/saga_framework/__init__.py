"""
Saga Framework - 分布式事务框架

提供编排式(Orchestration)和协作式(Choreography)两种Saga模式实现，
用于处理RQA2025量化交易系统中的分布式事务场景。

主要功能:
1. Saga编排器 - 用于复杂业务流程的中央协调
2. Saga协调器 - 用于简单业务流程的事件驱动
3. 补偿事务管理器 - 处理事务失败时的回滚
4. 状态管理器 - 持久化Saga执行状态

作者: RQA2025 Architecture Team
版本: 1.0.0
日期: 2026-03-08
"""

from .core.orchestrator import SagaOrchestrator, SagaDefinition, SagaStep
from .core.choreography import ChoreographySaga
from .core.context import SagaContext
from .state.state_models import SagaInstance, SagaStatus
from .state.state_manager import SagaStateManager
from .compensation.compensation_manager import CompensationManager
from .events.events import DomainEvent, SagaEvents

__version__ = "1.0.0"
__all__ = [
    # 核心组件
    "SagaOrchestrator",
    "SagaDefinition",
    "SagaStep",
    "ChoreographySaga",
    "SagaContext",
    # 状态管理
    "SagaInstance",
    "SagaStatus",
    "SagaStateManager",
    # 补偿管理
    "CompensationManager",
    # 事件
    "DomainEvent",
    "SagaEvents",
]
