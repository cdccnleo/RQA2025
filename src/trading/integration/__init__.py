# -*- coding: utf-8 -*-
"""
交易层集成模块

提供交易层与系统核心组件（统一调度器、事件总线）的集成能力。

函数级注释:
- 所有公开函数均包含详细的中文注释
- 支持PostgreSQL优先策略，连接失败时自动降级

作者: AI系统集成助手
日期: 2026-03-22
版本: 1.0.0
"""

from .scheduler_integration import (
    TradingSchedulerIntegration,
    TradingTaskType,
    TradingTaskConfig,
    TradingTaskResult,
    get_trading_scheduler_integration,
)

from .risk_intercept_integration import (
    RiskInterceptIntegration,
    RiskInterceptAction,
    RiskInterceptEvent,
    RiskInterceptResult,
    RiskInterceptHandler,
    get_risk_intercept_integration,
)

__all__ = [
    # 调度器集成
    "TradingSchedulerIntegration",
    "TradingTaskType",
    "TradingTaskConfig",
    "TradingTaskResult",
    "get_trading_scheduler_integration",
    # 风险拦截集成
    "RiskInterceptIntegration",
    "RiskInterceptAction",
    "RiskInterceptEvent",
    "RiskInterceptResult",
    "RiskInterceptHandler",
    "get_risk_intercept_integration",
]
