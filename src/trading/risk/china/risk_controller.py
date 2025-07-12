#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""中国证券市场风险控制统一入口

整合各类风险控制功能，提供统一接口
"""

from typing import Dict, Any
from .t1_restriction import T1RestrictionChecker
from .price_limit import PriceLimitChecker
from .star_market import STARMarketRuleChecker
from .circuit_breaker import CircuitBreaker

class ChinaRiskController:
    def __init__(self, config: Dict[str, Any]):
        """初始化风险控制器

        Args:
            config: 配置参数
        """
        self.t1_checker = T1RestrictionChecker(config)
        self.price_checker = PriceLimitChecker(config)
        self.star_checker = STARMarketRuleChecker(config)
        self.circuit_breaker = CircuitBreaker(config)

    def check(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """统一风险检查入口

        Args:
            order: 订单数据

        Returns:
            {
                "passed": bool,  # 是否通过检查
                "reason": str    # 拒绝原因(如未通过)
            }
        """
        # 检查熔断状态
        if not self.circuit_breaker.is_trading_allowed():
            return {"passed": False, "reason": "CIRCUIT_BREAKER"}

        # 检查T+1限制
        if not self.t1_checker.check_order(order):
            return {"passed": False, "reason": "T1_RESTRICTION"}

        # 检查涨跌停限制
        if not self.price_checker.check_order(order):
            return {"passed": False, "reason": "PRICE_LIMIT"}

        # 检查科创板特殊规则
        if not self.star_checker.check_order(order):
            return {"passed": False, "reason": "STAR_MARKET_RULE"}

        # 所有检查通过
        return {"passed": True, "reason": ""}
