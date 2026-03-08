"""
Data Quality Module - 数据质量监控模块

提供数据质量监控功能，实时检测数据异常并告警。

主要功能:
1. 数据质量规则引擎 - 定义和执行数据质量规则
2. 实时监控告警 - 监控数据质量指标并触发告警
3. 质量报告生成 - 生成数据质量报告
4. 异常数据处理 - 处理和记录异常数据

作者: RQA2025 Architecture Team
版本: 1.0.0
日期: 2026-03-08
"""

from .core.quality_engine import QualityEngine
from .core.rule_engine import RuleEngine
from .models.quality_models import (
    QualityRule,
    QualityCheck,
    QualityReport,
    QualityAlert,
    AlertLevel
)

__version__ = "1.0.0"
__all__ = [
    # 核心组件
    "QualityEngine",
    "RuleEngine",
    # 数据模型
    "QualityRule",
    "QualityCheck",
    "QualityReport",
    "QualityAlert",
    "AlertLevel",
]
