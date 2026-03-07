#!/usr/bin/env python3
"""
RQA2025 基础设施层规则类型定义

定义配置规则相关的数据类型和枚举。
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class AdaptationStrategy(Enum):
    """适应策略枚举"""
    CONSERVATIVE = "conservative"  # 保守策略，缓慢调整
    AGGRESSIVE = "aggressive"     # 激进策略，快速调整
    BALANCED = "balanced"         # 平衡策略，适中调整


@dataclass
class ConfigurationRule:
    """配置规则"""

    parameter_path: str  # 配置参数路径，如 "monitoring.interval"
    condition: str       # 条件表达式，如 "cpu_usage > 80"
    adjustment_value: Optional[float] = None  # 调整的目标值（可选）
    metric_name: Optional[str] = None         # 关联的性能指标
    action: Optional[str] = None              # 动作，如 "increase_interval"（可选）
    priority: int = 1                         # 优先级
    cooldown_minutes: int = 5                 # 冷却时间（分钟）
    last_applied: Optional[datetime] = None


@dataclass
class AdaptationHistory:
    """适应历史记录"""
    timestamp: datetime
    parameter_path: str
    old_value: any
    new_value: any
    reason: str
    performance_impact: Optional[dict[str, float]] = None
