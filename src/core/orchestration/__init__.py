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

# 导出业务流程组件（使用try/except处理可选导入）
try:
    from .business_process import (
        AppStartupListener,
        get_app_startup_listener
    )
    APP_STARTUP_LISTENER_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"business_process模块导入失败: {e}，跳过相关功能")
    APP_STARTUP_LISTENER_AVAILABLE = False
    AppStartupListener = None
    get_app_startup_listener = None

# 其他业务流程组件（可选）
try:
    from .business_process import DataCollectionServiceScheduler
    SERVICE_SCHEDULER_AVAILABLE = True
except ImportError:
    DataCollectionServiceScheduler = None
    SERVICE_SCHEDULER_AVAILABLE = False

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
    # 业务流程组件（可选）
    'AppStartupListener',
    'get_app_startup_listener',
    'DataCollectionServiceScheduler',
    # 事件系统
    'EventSystem',
    # 进程池
    'ProcessInstancePool'
]
