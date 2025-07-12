import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
import logging
from datetime import datetime, timedelta
from enum import Enum
from src.infrastructure.monitoring import ApplicationMonitor
from src.infrastructure.error import ErrorHandler

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """订单类型枚举"""
    MARKET = 1      # 市价单
    LIMIT = 2       # 限价单
    STOP = 3        # 止损单

class OrderDirection(Enum):
    """订单方向枚举"""
    BUY = 1         # 买入
    SELL = -1       # 卖出

class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = 1     # 待成交
    PARTIAL = 2     # 部分成交
    FILLED = 3     # 完全成交
    CANCELLED = 4   # 已取消
    REJECTED = 5    # 已拒绝

class ChinaMarketAdapter:
    """A股市场适配器"""
    
    ST_PREFIXES = {"ST", "*ST"}
    
    @staticmethod
    def check_trade_restrictions(symbol: str, price: float, last_close: float) -> bool:
        """
        检查A股交易限制
        
        Args:
            symbol: 股票代码
            price: 当前价格
            last_close: 昨日收盘价
            
        Returns:
            bool: True表示可以交易，False表示有交易限制
        """
        # 检查ST/*ST标记
        if any(symbol.startswith(prefix) for prefix in ChinaMarketAdapter.ST_PREFIXES):
            return False
            
        # 检查涨跌停(假设±10%)
        price_limit = last_close * 1.1 if price > last_close else last_close * 0.9
        if abs(price - last_close) >= abs(price_limit - last_close):
            return False
            
        return True
    
    @staticmethod
    def check_t1_restriction(position_date: datetime, current_date: datetime) -> bool:
        """
        检查T+1限制
        
        Args:
            position_date: 持仓日期
            current_date: 当前日期
            
        Returns:
            bool: True表示可以卖出，False表示受T+1限制
        """
        return current_date.date() > position_date.date()
    
    @staticmethod
    def calculate_fees(order: Dict, is_a_stock: bool = True) -> float:
        """
        计算A股交易费用
        
        Args:
            order: 订单信息
            is_a_stock: 是否是A股
            
        Returns:
            float: 总费用
        """
        if not is_a_stock:
            return 0.0
            
        quantity = order["quantity"]
        price = order.get("price", 0)
        amount = quantity * price
        
        # 印花税: 卖出时0.1%
        stamp_tax = amount * 0.001 if order["direction"] == OrderDirection.SELL else 0
        
        # 佣金: 0.025% (最低5元)
        commission = max(amount * 0.00025, 5)
        
        # 过户费: 0.001%
        transfer_fee = amount * 0.00001
        
        return stamp_tax + commission + transfer_fee


class TradingEngine:
    """交易策略引擎"""

    def __init__(self, risk_config: Dict, monitor: Optional[ApplicationMonitor] = None):
        """
        初始化交易引擎

        Args:
            risk_config: 风险控制配置
            monitor: 监控系统实例
        """
        self.risk_config = risk_config
        self.monitor = monitor or ApplicationMonitor("trading_engine")
        self.error_handler = ErrorHandler()
        
        # A股市场配置
        self.is_a_stock = risk_config.get("market_type", "A") == "A"
        self.last_close_prices = {}  # 存储昨日收盘价用于涨跌停检查

        # 持仓状态
        self.positions: Dict[str, float] = {}  # {symbol: quantity}
        self.cash_balance: float = risk_config.get("initial_capital", 1000000.0)

        # 订单记录
        self.order_history: List[Dict] = []

        # 交易统计
        self.trade_stats: Dict = {
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0
        }

    def generate_orders(
        self,
        signals: pd.DataFrame,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        """
        根据信号生成交易订单

        Args:
            signals: 信号DataFrame(包含symbol, signal, strength等列)
            current_prices: 当前价格字典 {symbol: price}

        Returns:
            List[Dict]: 生成的订单列表
        """
        orders = []

        for _, row in signals.iterrows():
            try:
                symbol = row["symbol"]
                signal = row["signal"]  # 1: 买入, -1: 卖出
                strength = row.get("strength", 1.0)  # 信号强度

                # 获取当前价格和昨日收盘价
                current_price = current_prices.get(symbol, 0)
                last_close = self.last_close_prices.get(symbol, current_price)

                # 检查A股交易限制
                if self.is_a_stock and not ChinaMarketAdapter.check_trade_restrictions(
                    symbol=symbol,
                    price=current_price,
                    last_close=last_close
                ):
                    logger.warning(f"Stock {symbol} violates A-share trading restrictions")
                    continue

                # 计算目标仓位
                target_pos = self._calculate_position_size(
                    symbol=symbol,
                    signal=signal,
                    strength=strength,
                    price=current_price
                )

                # 生成订单
                if target_pos != 0:
                    order = self._create_order(
                        symbol=symbol,
                        direction=OrderDirection.BUY if signal > 0 else OrderDirection.SELL,
                        quantity=abs(target_pos),
                        price=current_prices.get(symbol),
                        order_type=OrderType.MARKET
                    )
                    orders.append(order)

            except Exception as e:
                logger.error(f"Failed to generate order for {row['symbol']}: {e}")
                self.error_handler.handle(e)

        return orders

    def _calculate_position_size(
        self,
        symbol: str,
        signal: int,
        strength: float,
        price: float
    ) -> float:
        """
        计算目标仓位变化量

        Args:
            symbol: 标的代码
            signal: 交易信号(1: 买入, -1: 卖出)
            strength: 信号强度(0-1)
            price: 当前价格

        Returns:
            float: 仓位变化量(正数表示买入,负数表示卖出)
        """
        if price <= 0:
            return 0

        # 获取当前仓位
        current_pos = self.positions.get(symbol, 0)

        # 计算目标仓位(基于风险配置)
        position_size = (self.cash_balance * self.risk_config["per_trade_risk"] * strength) / price
        target_pos = position_size * signal

        # 应用头寸限制
        max_pos = self.risk_config.get("max_position", {}).get(symbol, float("inf"))
        target_pos = np.sign(target_pos) * min(abs(target_pos), max_pos)

        # 计算实际可交易量
        if signal > 0:  # 买入
            max_affordable = self.cash_balance / price
            target_pos = min(target_pos, max_affordable)
        else:  # 卖出
            target_pos = max(target_pos, -current_pos)

        return target_pos - current_pos

    def _create_order(
        self,
        symbol: str,
        direction: OrderDirection,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> Dict:
        """
        创建订单字典

        Args:
            symbol: 标的代码
            direction: 交易方向
            quantity: 数量
            price: 价格(限价单需要)
            order_type: 订单类型

        Returns:
            Dict: 订单信息
        """
        order_id = f"order_{len(self.order_history)+1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        order = {
            "order_id": order_id,
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "filled_quantity": 0,
            "price": price,
            "order_type": order_type,
            "status": OrderStatus.PENDING,
            "timestamp": datetime.now().isoformat(),
            "fees": 0.0  # 初始费用设为0
        }

        # 计算A股交易费用
        if self.is_a_stock:
            order["fees"] = ChinaMarketAdapter.calculate_fees(
                order=order,
                is_a_stock=True
            )

        # 记录订单
        self.order_history.append(order.copy())
        self.monitor.record_metric(
            "order_created",
            value=1,
            tags={
                "symbol": symbol,
                "direction": direction.name,
                "type": order_type.name
            }
        )

        return order

    def update_order_status(
        self,
        order_id: str,
        filled_quantity: float,
        avg_price: float,
        status: OrderStatus
    ) -> None:
        """
        更新订单状态

        Args:
            order_id: 订单ID
            filled_quantity: 已成交数量
            avg_price: 成交均价
            status: 新状态
        """
        order = next((o for o in self.order_history if o["order_id"] == order_id), None)
        if not order:
            logger.warning(f"Order {order_id} not found")
            return

        # 更新订单
        order["filled_quantity"] = filled_quantity
        order["avg_price"] = avg_price
        order["status"] = status

        # 如果是成交状态，更新持仓和资金
        if status == OrderStatus.FILLED:
            self._update_position(
                symbol=order["symbol"],
                quantity=filled_quantity * (1 if order["direction"] == OrderDirection.BUY else -1),
                price=avg_price
            )

            # 更新交易统计
            self._update_trade_stats(order)

        self.monitor.record_metric(
            "order_updated",
            value=1,
            tags={
                "symbol": order["symbol"],
                "status": status.name
            }
        )

    def _update_position(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> None:
        """
        更新持仓状态

        Args:
            symbol: 标的代码
            quantity: 数量变化(正:买入,负:卖出)
            price: 成交价格
        """
        # 检查T+1限制(仅对A股卖出操作)
        if (self.is_a_stock and quantity < 0 and 
            not ChinaMarketAdapter.check_t1_restriction(
                position_date=datetime.now() - timedelta(days=1),
                current_date=datetime.now()
            )):
            logger.warning(f"Stock {symbol} violates T+1 restriction")
            return

        # 更新持仓
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity

        # 更新资金(考虑交易费用)
        trade_amount = quantity * price
        if self.is_a_stock:
            # 对于A股，卖出时从金额中扣除费用
            if quantity < 0:
                trade_amount -= self.order_history[-1].get("fees", 0)
            # 买入时费用已包含在订单金额中
        self.cash_balance -= trade_amount

        # 记录持仓变化
        self.monitor.record_metric(
            "position_updated",
            value=quantity,
            tags={
                "symbol": symbol,
                "direction": "buy" if quantity > 0 else "sell"
            }
        )

    def _update_trade_stats(self, order: Dict) -> None:
        """
        更新交易统计

        Args:
            order: 订单信息
        """
        self.trade_stats["total_trades"] += 1

        # 简单判断盈亏(实际应该基于更复杂的逻辑)
        if order["direction"] == OrderDirection.BUY:
            self.trade_stats["win_trades"] += 1
        else:
            self.trade_stats["loss_trades"] += 1

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        计算当前组合价值

        Args:
            current_prices: 当前价格字典 {symbol: price}

        Returns:
            float: 组合总价值
        """
        position_value = sum(
            qty * current_prices.get(sym, 0)
            for sym, qty in self.positions.items()
        )
        return self.cash_balance + position_value

    def get_risk_metrics(self) -> Dict:
        """
        获取风险指标

        Returns:
            Dict: 风险指标字典
        """
        return {
            "max_drawdown": self._calculate_max_drawdown(),
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "win_rate": self.trade_stats["win_trades"] / self.trade_stats["total_trades"] if self.trade_stats["total_trades"] > 0 else 0
        }

    def _calculate_max_drawdown(self) -> float:
        """
        计算最大回撤(简化版)
        """
        # TODO: 实现基于历史交易记录的回撤计算
        return 0.0

    def _calculate_sharpe_ratio(self) -> float:
        """
        计算夏普比率(简化版)
        """
        # TODO: 实现基于收益波动率的夏普比率计算
        return 0.0
