"""A股市场基础策略类，处理A股特有规则"""

import backtrader as bt
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)

class ChinaMarketStrategy(bt.Strategy):
    """A股市场基础策略

    处理A股特有规则：
    - T+1交易限制
    - 涨跌停限制
    - 科创板特殊规则
    """

    params = (
        ('is_star_market', False),  # 是否为科创板股票
    )

    def __init__(self):
        super().__init__()
        self.order = None  # 跟踪当前订单

    def next(self):
        """主交易逻辑"""
        if self.order:  # 如果有未完成订单则不操作
            return

        # 在这里添加A股特有规则检查
        if not self._check_t1_restriction():
            return
        if not self._check_price_limit():
            return
        if self.p.is_star_market and not self._check_star_market_rules():
            return

        # 调用子类策略逻辑
        self.next_strategy()

    def next_strategy(self):
        """子类实现的策略逻辑"""
        raise NotImplementedError("子类必须实现next_strategy方法")

    def _check_t1_restriction(self) -> bool:
        """检查T+1限制"""
        from datetime import datetime, timedelta
        from trading.china.adapters import get_trade_date
        
        # 获取当前持仓
        position = self.getposition()
        if not position:
            return True
            
        # 检查最近买入日期
        last_buy_date = getattr(position, 'last_buy_date', None)
        if not last_buy_date:
            return True
            
        current_date = get_trade_date(datetime.now())
        
        # T+1限制：当日买入的股票次日才能卖出
        return current_date > last_buy_date

    def _check_price_limit(self) -> bool:
        """检查涨跌停限制"""
        from trading.risk.china.price_limit import get_price_limits
        
        # 获取当前数据
        data = self.datas[0]
        symbol = data._name
        price = data.close[0]
        
        # 获取该股票的涨跌停价格
        upper, lower = get_price_limits(symbol)
        
        # 检查价格是否在涨跌停范围内
        return lower <= price <= upper

    def _check_star_market_rules(self) -> bool:
        """检查科创板特殊规则"""
        from trading.risk.china.star_market import is_star_market_stock
        from trading.risk.china.star_market import check_star_market_rules
        
        # 如果不是科创板股票则直接通过
        data = self.datas[0]
        if not is_star_market_stock(data._name):
            return True
            
        # 检查科创板特殊规则
        position = self.getposition()
        return check_star_market_rules(data._name, position)

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"买入执行 {order.executed.price}")
            elif order.issell():
                logger.info(f"卖出执行 {order.executed.price}")

        self.order = None
