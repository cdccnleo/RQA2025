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

🎯 设计理念:
    - 核心业务逻辑集中在core层
    - 简化导入路径，消除循环依赖
    - 与EventBus、BusinessProcess同层协作
"""

from .orchestrator_refactored import (
    BusinessProcessOrchestrator,
    ProcessConfig,
    ProcessInstance,
    create_process_config,
    create_process_instance
)
from .configs import OrchestratorConfig

__all__ = [
    'BusinessProcessOrchestrator',
    'OrchestratorConfig',
    'ProcessConfig',
    'ProcessInstance',
    'create_process_config',
    'create_process_instance'
]
