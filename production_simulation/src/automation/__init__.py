#!/usr/bin/env python3
"""
RQA2025 自动化层
Automation Layer

提供智能自动化和规则引擎功能。
"""

from .core.automation_models import (
    AutomationType, ExecutionStatus, TaskPriority,
    AutomationRule, AutomationTask, AutomationWorkflow,
    AutomationMetrics, AIDecisionContext, DeploymentConfig, ScalingDecision
)
from .core.rule_manager import RuleManager
from .core.rule_executor import RuleExecutor
from .core.simple_engine import SimpleAutomationEngine

__version__ = "1.0.0"
__author__ = "RQA2025 Team"

__all__ = [
    # 自动化模型
    'AutomationType', 'ExecutionStatus', 'TaskPriority',
    'AutomationRule', 'AutomationTask', 'AutomationWorkflow',
    'AutomationMetrics', 'AIDecisionContext', 'DeploymentConfig', 'ScalingDecision',

    # 核心组件
    'RuleManager', 'RuleExecutor', 'SimpleAutomationEngine'
]
