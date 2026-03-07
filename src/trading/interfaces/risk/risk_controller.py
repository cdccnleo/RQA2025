"""
交易层 - 风险控制接口定义

提供风险控制的统一接口和基础实现。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime


class RiskController(ABC):
    """风险控制器接口"""

    @abstractmethod
    def check_order_risk(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """检查订单风险"""

    @abstractmethod
    def check_portfolio_risk(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """检查投资组合风险"""

    @abstractmethod
    def get_daily_risk_stats(self, date: datetime) -> Dict[str, Any]:
        """获取每日风险统计"""

    @abstractmethod
    def validate_position_limits(self, positions: Dict[str, Any]) -> bool:
        """验证持仓限制"""


class BaseRiskController(RiskController):
    """基础风险控制器实现"""

    def __init__(self):
        self.max_position_size = 1000000  # 最大持仓
        self.max_daily_loss = 50000       # 最大每日损失
        self.risk_checks = []

    def check_order_risk(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """基础订单风险检查"""
        order_value = order.get('quantity', 0) * order.get('price', 0)

        return {
            'approved': order_value <= self.max_position_size,
            'risk_level': 'low' if order_value < 100000 else 'medium',
            'warnings': [] if order_value <= self.max_position_size else ['超大订单']
        }

    def check_portfolio_risk(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """基础投资组合风险检查"""
        total_value = portfolio.get('total_value', 0)
        daily_pnl = portfolio.get('daily_pnl', 0)

        return {
            'risk_level': 'high' if daily_pnl < -self.max_daily_loss else 'low',
            'breach_limits': daily_pnl < -self.max_daily_loss,
            'recommendations': ['减少风险暴露'] if daily_pnl < -self.max_daily_loss else []
        }

    def get_daily_risk_stats(self, date: datetime) -> Dict[str, Any]:
        """获取每日风险统计"""
        return {
            'date': date.strftime('%Y-%m-%d'),
            'total_checks': len(self.risk_checks),
            'violations': 0,
            'auto_rejects': 0,
            'warnings': 0
        }

    def validate_position_limits(self, positions: Dict[str, Any]) -> bool:
        """验证持仓限制"""
        total_exposure = sum(abs(pos.get('value', 0)) for pos in positions.values())
        return total_exposure <= self.max_position_size
