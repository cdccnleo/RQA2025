"""
RQA2025 业务流程编排模块 (Core Orchestration Layer)

核心业务编排层，提供业务流程管理、状态机、事件驱动编排能力。

🏗️ 架构位置:
    src/core/orchestration (核心业务层)
    
📦 包含组件:
    - BusinessProcessOrchestrator: 业务流程编排器
    - EventBus: 事件总线 (从event_bus导入)
    - StateMachine: 状态机
    - ConfigManager: 配置管理
    - ProcessMonitor: 流程监控
    - InstancePool: 实例池
    - UnifiedScheduler: 统一调度器
    - 各类调度器: AI驱动、分布式、历史数据等

🎯 设计理念:
    - 核心业务逻辑集中在core层
    - 简化导入路径，消除循环依赖
    - 与EventBus、BusinessProcess同层协作

📝 版本历史:
    - v2.0: 从infrastructure层迁移到core层
"""

from .orchestrator_refactored import (
    BusinessProcessOrchestrator,
    ProcessConfig,
    ProcessInstance,
    create_process_config,
    create_process_instance
)
from .configs import OrchestratorConfig

# 导出统一调度器
from .scheduler import (
    UnifiedScheduler,
    get_unified_scheduler,
    TaskManager,
    WorkerManager
)

# 导出业务流程组件
from .business_process import (
    AppStartupListener,
    DataCollectionOrchestrator,
    DataCollectionStateMachine,
    ServiceScheduler,
    ServiceGovernance
)

# 导出事件系统
from .business import EventSystem

# 导出进程池
from .pool import ProcessInstancePool

__all__ = [
    # 业务流程编排
    'BusinessProcessOrchestrator',
    'OrchestratorConfig',
    'ProcessConfig',
    'ProcessInstance',
    'create_process_config',
    'create_process_instance',
    # 统一调度器
    'UnifiedScheduler',
    'get_unified_scheduler',
    'TaskManager',
    'WorkerManager',
    # 业务流程组件
    'AppStartupListener',
    'DataCollectionOrchestrator',
    'DataCollectionStateMachine',
    'ServiceScheduler',
    'ServiceGovernance',
    # 事件系统
    'EventSystem',
    # 进程池
    'ProcessInstancePool'
]
