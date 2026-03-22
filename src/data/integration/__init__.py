# -*- coding: utf-8 -*-
"""
数据管理层集成模块

提供数据管理层与系统核心组件（统一调度器、事件总线）的集成能力。

函数级注释:
- 所有公开函数均包含详细的中文注释
- 支持PostgreSQL优先策略，连接失败时自动降级

作者: AI系统集成助手
日期: 2026-03-22
版本: 1.0.0
"""

from .scheduler_integration import (
    DataSchedulerIntegration,
    DataTaskType,
    DataTaskConfig,
    DataTaskResult,
    get_data_scheduler_integration,
)

__all__ = [
    "DataSchedulerIntegration",
    "DataTaskType",
    "DataTaskConfig",
    "DataTaskResult",
    "get_data_scheduler_integration",
]
