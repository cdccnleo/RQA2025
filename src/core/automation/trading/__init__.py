# Automation Trading Module
# 自动化交易模块

# This module contains trading automation components
# 此模块包含交易自动化组件

from .dynamic_risk_limits import DynamicRiskLimits
from .emergency_response_system import EmergencyResponseSystem
from .trade_adjustment_engine import TradeAdjustmentEngine

__all__ = [
    'DynamicRiskLimits',
    'EmergencyResponseSystem',
    'TradeAdjustmentEngine'
]
