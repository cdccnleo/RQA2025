#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025 核心服务层统一接口定义
Core Services Layer Unified Interface Definitions

包含ML层和策略层标准化接口契约，以及核心组件统一接口
"""

# 导入ML和策略层接口
from .ml_strategy_interfaces import (
    IMLService, IStrategyService, IStrategyDataPreparation,
    MLFeatures, MLInferenceRequest, MLInferenceResponse,
    StrategyExecutionRequest, StrategyExecutionResponse,
    MLStrategyCollaborationProtocol,
    create_ml_service, create_strategy_service, create_collaboration_protocol,
    InterfaceComplianceValidator
)

# 导入核心组件统一接口
try:
    from ..interfaces import (
        ICoreComponent, IEventBus, IDependencyContainer,
        IBusinessProcessOrchestrator, ILayerInterface,
        CoreComponent, ComponentStatus, ComponentHealth,
        StandardComponent
    )
except ImportError:
    # 如果导入失败，提供默认值
    ICoreComponent = None
    IEventBus = None
    IDependencyContainer = None
    IBusinessProcessOrchestrator = None
    ILayerInterface = None
    CoreComponent = None
    ComponentStatus = None
    ComponentHealth = None
    StandardComponent = None

__all__ = [
    # ML和策略层接口
    'IMLService', 'IStrategyService', 'IStrategyDataPreparation',
    'MLFeatures', 'MLInferenceRequest', 'MLInferenceResponse',
    'StrategyExecutionRequest', 'StrategyExecutionResponse',
    'MLStrategyCollaborationProtocol',
    'create_ml_service', 'create_strategy_service', 'create_collaboration_protocol',
    'InterfaceComplianceValidator',

    # 核心组件统一接口
    'ICoreComponent', 'IEventBus', 'IDependencyContainer',
    'IBusinessProcessOrchestrator', 'ILayerInterface',
    'CoreComponent', 'ComponentStatus', 'ComponentHealth',
    'StandardComponent'
]
