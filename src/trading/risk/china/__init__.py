"""中国证券市场风险控制模块

提供完整的A股市场风险控制功能，包括：

核心功能:
- CircuitBreaker: 熔断机制监控
- PositionLimits: 持仓限额管理
- validate_position: 持仓限额验证函数
- PriceLimit: 涨跌停板限制检查
- StarMarketRisk: 科创板/创业板特殊规则
- T1Restriction: T+1交易限制验证

新增特性:
1. 统一面向对象接口
2. 实时风险指标导出
3. 支持多数据源适配
4. 可配置风险阈值

使用示例:
    from trading.risk.china import PriceLimit, T1Restriction, validate_position
    
    # 使用类接口
    price_check = PriceLimit()
    if price_check.is_touching_limit(symbol='600519', price=1000):
        print("触发涨停限制")
        
    # 使用函数接口
    if validate_position(account, '600000', 10000):
        print("持仓验证通过")
"""

from .circuit_breaker import CircuitBreaker
from .position_limits import PositionLimits, validate_position
from .price_limit import PriceLimitChecker
from .star_market import STARMarketRuleChecker
from .t1_restriction import T1RestrictionChecker
from .risk_controller import ChinaRiskController

__all__ = [
    'CircuitBreaker',
    'PositionLimits',
    'validate_position',
    'PriceLimitChecker',
    'STARMarketRuleChecker',
    'T1RestrictionChecker',
    'ChinaRiskController'
]
