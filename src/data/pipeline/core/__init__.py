"""
管道核心框架模块

提供ML管道的核心功能，包括阶段定义、状态管理、配置管理和管道编排。

主要组件:
    - PipelineStage: 管道阶段基类
    - PipelineState: 管道状态管理
    - PipelineConfig: 管道配置管理
    - MLPipelineController: 管道编排控制器

使用示例:
    >>> from src.data.pipeline.core import (
    ...     PipelineStage, PipelineState, PipelineConfig,
    ...     MLPipelineController, create_default_config
    ... )
    >>> 
    >>> # 创建配置
    >>> config = create_default_config()
    >>> 
    >>> # 创建控制器
    >>> controller = MLPipelineController(config)
    >>> 
    >>> # 注册阶段并执行
    >>> controller.register_stage(MyStage())
    >>> result = controller.execute()
"""

from .pipeline_stage import (
    PipelineStage,
    CompositeStage,
    StageResult,
    StageStatus
)

from .pipeline_state import (
    PipelineState,
    PipelineStatus,
    PipelineStateManager,
    PipelineCheckpoint
)

from .pipeline_config import (
    PipelineConfig,
    StageConfig,
    StageType,
    RollbackConfig,
    RollbackTriggerConfig,
    MonitoringConfig,
    IntegrationConfig,
    create_default_config,
    load_config
)

from .pipeline_controller import (
    MLPipelineController,
    PipelineExecutionResult,
    create_pipeline_controller
)

__all__ = [
    # 阶段相关
    'PipelineStage',
    'CompositeStage',
    'StageResult',
    'StageStatus',
    
    # 状态相关
    'PipelineState',
    'PipelineStatus',
    'PipelineStateManager',
    'PipelineCheckpoint',
    
    # 配置相关
    'PipelineConfig',
    'StageConfig',
    'StageType',
    'RollbackConfig',
    'RollbackTriggerConfig',
    'MonitoringConfig',
    'IntegrationConfig',
    'create_default_config',
    'load_config',
    
    # 控制器相关
    'MLPipelineController',
    'PipelineExecutionResult',
    'create_pipeline_controller'
]
