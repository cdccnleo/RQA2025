#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 4: 核心业务功能重建

修复技术债务: 核心业务功能重建
解决业务验收测试中发现的核心业务功能缺失和不完整的问题
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 核心业务数据结构


class TransactionType(Enum):
    """交易类型"""
    BUY = "buy"
    SELL = "sell"
    SHORT_SELL = "short_sell"
    COVER_SHORT = "cover_short"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionType(Enum):
    """持仓类型"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """订单数据结构"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    transaction_type: TransactionType = TransactionType.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = field(init=False)
    avg_fill_price: float = 0.0
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    strategy_id: Optional[str] = None
    account_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.remaining_quantity = abs(self.quantity) - self.filled_quantity

    @property
    def is_active(self) -> bool:
        """检查订单是否活跃"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]

    @property
    def is_completed(self) -> bool:
        """检查订单是否完成"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]

    def update_status(self, new_status: OrderStatus, error_message: Optional[str] = None):
        """更新订单状态"""
        self.status = new_status
        self.updated_time = datetime.now()
        if error_message:
            self.error_message = error_message

    def add_fill(self, fill_quantity: float, fill_price: float):
        """添加成交记录"""
        if fill_quantity <= 0 or fill_price <= 0:
            raise ValueError("成交数量和价格必须为正数")

        if fill_quantity > self.remaining_quantity:
            raise ValueError("成交数量不能超过剩余数量")

        old_filled_value = self.filled_quantity * self.avg_fill_price
        new_filled_value = fill_quantity * fill_price
        total_filled_value = old_filled_value + new_filled_value

        self.filled_quantity += fill_quantity
        self.remaining_quantity = abs(self.quantity) - self.filled_quantity
        self.avg_fill_price = total_filled_value / self.filled_quantity if self.filled_quantity > 0 else 0
        self.updated_time = datetime.now()

        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED


@dataclass
class Position:
    """持仓数据结构"""
    symbol: str = ""
    position_type: PositionType = PositionType.LONG
    quantity: float = 0.0
    average_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    market_value: float = field(init=False)
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.market_value = self.quantity * self.current_price

    def update_price(self, new_price: float):
        """更新价格并重新计算盈亏"""
        self.current_price = new_price
        self.market_value = self.quantity * self.current_price

        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (self.current_price - self.average_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.average_price - self.current_price) * self.quantity

        self.updated_time = datetime.now()

    def add_position(self, quantity: float, price: float):
        """增加持仓"""
        if quantity == 0:
            return

        if self.quantity == 0:
            # 新建持仓
            self.quantity = quantity
            self.average_price = price
            self.position_type = PositionType.LONG if quantity > 0 else PositionType.SHORT
        else:
            # 调整现有持仓
            total_quantity = self.quantity + quantity
            if total_quantity == 0:
                # 平仓
                self.realized_pnl += (price - self.average_price) * abs(quantity)
                self.quantity = 0
                self.average_price = 0
                self.unrealized_pnl = 0
            else:
                # 调整平均价格
                total_value = self.quantity * self.average_price + quantity * price
                self.average_price = total_value / total_quantity
                self.quantity = total_quantity

        self.update_price(self.current_price)

    def close_position(self, quantity: float, price: float) -> float:
        """平仓并返回已实现盈亏"""
        if abs(quantity) > abs(self.quantity):
            raise ValueError("平仓数量不能超过持仓数量")

        if self.position_type == PositionType.LONG:
            realized_pnl = (price - self.average_price) * abs(quantity)
        else:
            realized_pnl = (self.average_price - price) * abs(quantity)

        self.realized_pnl += realized_pnl
        self.quantity -= quantity

        if self.quantity == 0:
            self.average_price = 0
            self.unrealized_pnl = 0

        self.update_price(self.current_price)
        return realized_pnl


@dataclass
class Account:
    """账户数据结构"""
    account_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    balance: float = 0.0
    available_balance: float = 0.0
    margin_used: float = 0.0
    total_value: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: Dict[str, Order] = field(default_factory=dict)
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)

    def update_balance(self, amount: float):
        """更新余额"""
        self.balance += amount
        self.available_balance += amount
        self.total_value = self.balance + sum(pos.market_value for pos in self.positions.values())
        self.updated_time = datetime.now()

    def add_position(self, symbol: str, quantity: float, price: float):
        """增加持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        position = self.positions[symbol]
        position.add_position(quantity, price)
        self.total_value = self.balance + sum(pos.market_value for pos in self.positions.values())
        self.updated_time = datetime.now()

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)

    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        return sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())


class CoreBusinessEngine:
    """核心业务引擎 - Phase 4重建版本"""

    def __init__(self, account_id: str = "demo_account"):
        self.account_id = account_id
        self.account = Account(account_id=account_id, name="Demo Trading Account", balance=100000.0)
        self.account.available_balance = self.account.balance

        # 业务组件
        self.order_manager = None
        self.position_manager = None
        self.risk_manager = None
        self.execution_engine = None

        # 统计信息
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'start_balance': self.account.balance,
            'current_balance': self.account.balance
        }

        # 锁
        self._lock = threading.RLock()

        logger.info(f"核心业务引擎初始化完成 - 账户: {account_id}")

    def initialize(self) -> bool:
        """初始化核心业务引擎"""
        try:
            # 初始化订单管理器
            self.order_manager = OrderManager()
            self.order_manager.initialize()

            # 初始化持仓管理器
            self.position_manager = PositionManager(self.account)
            self.position_manager.initialize()

            # 初始化风险管理器
            self.risk_manager = RiskManager(self.account)
            self.risk_manager.initialize()

            # 初始化执行引擎
            self.execution_engine = ExecutionEngine()
            self.execution_engine.initialize()

            logger.info("核心业务引擎所有组件初始化完成")
            return True

        except Exception as e:
            logger.error(f"核心业务引擎初始化失败: {e}")
            return False

    def submit_order(self, symbol: str, transaction_type: TransactionType,
                     quantity: float, price: Optional[float] = None,
                     order_type: OrderType = OrderType.MARKET,
                     stop_price: Optional[float] = None) -> Tuple[bool, str, Optional[str]]:
        """提交订单"""
        with self._lock:
            try:
                # 创建订单
                order = Order(
                    symbol=symbol,
                    transaction_type=transaction_type,
                    order_type=order_type,
                    quantity=quantity if transaction_type == TransactionType.BUY else -quantity,
                    price=price,
                    stop_price=stop_price,
                    account_id=self.account_id
                )

                # 风险检查
                if not self.risk_manager.check_order_risk(order):
                    return False, "风险检查失败", None

                # 提交订单
                success, message, order_id = self.order_manager.submit_order(order)
                if success:
                    # 添加到账户订单列表
                    self.account.orders[order_id] = order
                    self.stats['total_orders'] += 1

                    # 如果是市价单，立即执行
                    if order_type == OrderType.MARKET:
                        self._execute_market_order(order)

                return success, message, order_id

            except Exception as e:
                logger.error(f"提交订单失败: {e}")
                return False, f"系统错误: {str(e)}", None

    def _execute_market_order(self, order: Order):
        """执行市价订单"""
        try:
            # 模拟市场价格（实际应该从市场数据获取）
            market_price = self._get_market_price(order.symbol)

            # 成交订单
            self.execution_engine.fill_order(order.order_id, order.quantity, market_price)

            # 更新订单状态为已成交
            order.add_fill(abs(order.quantity), market_price)
            order.update_status(OrderStatus.FILLED)

            # 更新持仓
            self.position_manager.update_position_from_order(order, market_price)

            # 更新账户余额
            trade_value = abs(order.quantity) * market_price
            if order.transaction_type == TransactionType.BUY:
                self.account.update_balance(-trade_value)
            else:
                self.account.update_balance(trade_value)

            self.stats['filled_orders'] += 1
            self.stats['total_trades'] += 1
            self.stats['current_balance'] = self.account.balance

            logger.info(
                f"市价订单执行完成: {order.order_id} - {order.symbol} {order.quantity}@{market_price}")

        except Exception as e:
            logger.error(f"执行市价订单失败: {e}")

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """取消订单"""
        with self._lock:
            try:
                success, message = self.order_manager.cancel_order(order_id)
                if success:
                    self.stats['cancelled_orders'] += 1
                return success, message
            except Exception as e:
                logger.error(f"取消订单失败: {e}")
                return False, f"系统错误: {str(e)}"

    def get_account_summary(self) -> Dict[str, Any]:
        """获取账户摘要"""
        with self._lock:
            positions_value = sum(pos.market_value for pos in self.account.positions.values())
            total_pnl = self.account.get_total_pnl()

            return {
                'account_id': self.account.account_id,
                'balance': self.account.balance,
                'available_balance': self.account.available_balance,
                'positions_value': positions_value,
                'total_value': self.account.total_value,
                'total_pnl': total_pnl,
                'positions_count': len(self.account.positions),
                'orders_count': len([o for o in self.account.orders.values() if o.is_active]),
                'stats': self.stats.copy()
            }

    def get_positions(self) -> List[Dict[str, Any]]:
        """获取持仓列表"""
        return [
            {
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'average_price': pos.average_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'total_pnl': pos.unrealized_pnl + pos.realized_pnl
            }
            for pos in self.account.positions.values()
        ]

    def get_orders(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """获取订单列表"""
        orders = self.account.orders.values()
        if active_only:
            orders = [o for o in orders if o.is_active]

        return [
            {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'transaction_type': order.transaction_type.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'status': order.status.value,
                'filled_quantity': order.filled_quantity,
                'remaining_quantity': order.remaining_quantity,
                'avg_fill_price': order.avg_fill_price,
                'created_time': order.created_time.isoformat()
            }
            for order in orders
        ]

    def update_market_prices(self, price_updates: Dict[str, float]):
        """更新市场价格"""
        with self._lock:
            try:
                for symbol, price in price_updates.items():
                    # 更新账户中该股票的持仓价格
                    if symbol in self.account.positions:
                        self.account.positions[symbol].update_price(price)

                    # 检查止损/止盈订单
                    self._check_stop_orders(symbol, price)

                # 更新账户总价值
                self.account.total_value = self.account.balance + sum(
                    pos.market_value for pos in self.account.positions.values()
                )

            except Exception as e:
                logger.error(f"更新市场价格失败: {e}")

    def _check_stop_orders(self, symbol: str, current_price: float):
        """检查止损订单"""
        try:
            for order in self.account.orders.values():
                if (order.symbol == symbol and
                    order.is_active and
                        order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]):

                    should_trigger = False
                    if order.transaction_type == TransactionType.SELL and order.stop_price:
                        # 卖出止损：价格跌破止损价
                        if current_price <= order.stop_price:
                            should_trigger = True
                    elif order.transaction_type == TransactionType.BUY and order.stop_price:
                        # 买入止损：价格突破止损价
                        if current_price >= order.stop_price:
                            should_trigger = True

                    if should_trigger:
                        logger.info(f"触发止损订单: {order.order_id} - {symbol}@{current_price}")
                        self._execute_market_order(order)

        except Exception as e:
            logger.error(f"检查止损订单失败: {e}")

    def _get_market_price(self, symbol: str) -> float:
        """获取市场价格（模拟）"""
        # 简单的模拟价格生成
        import random
        base_prices = {
            'AAPL': 150.0, 'GOOGL': 2500.0, 'MSFT': 300.0,
            'TSLA': 200.0, 'AMZN': 3000.0, 'NVDA': 400.0
        }
        base_price = base_prices.get(symbol, 100.0)
        # 添加一些随机波动
        return base_price * (1 + random.uniform(-0.02, 0.02))


class OrderManager:
    """订单管理器"""

    def __init__(self):
        self.orders = {}
        self._lock = threading.RLock()

    def initialize(self) -> bool:
        logger.info("订单管理器初始化完成")
        return True

    def submit_order(self, order: Order) -> Tuple[bool, str, Optional[str]]:
        """提交订单"""
        with self._lock:
            try:
                # 基本验证
                if not order.symbol or order.quantity == 0:
                    return False, "无效的订单参数", None

                self.orders[order.order_id] = order
                return True, "订单提交成功", order.order_id

            except Exception as e:
                return False, f"订单提交失败: {str(e)}", None

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """取消订单"""
        with self._lock:
            try:
                if order_id not in self.orders:
                    return False, "订单不存在"

                order = self.orders[order_id]
                if not order.is_active:
                    return False, "订单无法取消"

                order.update_status(OrderStatus.CANCELLED)
                return True, "订单取消成功"

            except Exception as e:
                return False, f"订单取消失败: {str(e)}"


class PositionManager:
    """持仓管理器"""

    def __init__(self, account: Account):
        self.account = account
        self._lock = threading.RLock()

    def initialize(self) -> bool:
        logger.info("持仓管理器初始化完成")
        return True

    def update_position_from_order(self, order: Order, fill_price: float):
        """根据订单更新持仓"""
        with self._lock:
            try:
                symbol = order.symbol
                quantity = order.filled_quantity
                price = fill_price

                # 直接更新账户的持仓
                self.account.add_position(symbol, quantity, price)

                logger.debug(f"持仓更新: {symbol} {quantity}@{price}")

            except Exception as e:
                logger.error(f"更新持仓失败: {e}")


class RiskManager:
    """风险管理器"""

    def __init__(self, account: Account):
        self.account = account
        self.risk_limits = {
            'max_position_size': 50000,  # 单股票最大持仓
            'max_total_exposure': 200000,  # 总风险暴露
            'max_single_stock_ratio': 0.2,  # 单股票占总资产比例
            'max_total_loss': 10000  # 最大总亏损
        }

    def initialize(self) -> bool:
        logger.info("风险管理器初始化完成")
        return True

    def check_order_risk(self, order: Order) -> bool:
        """检查订单风险"""
        try:
            # 检查余额充足
            if order.transaction_type == TransactionType.BUY:
                required_amount = abs(order.quantity) * (order.price or 100)  # 估算金额
                if required_amount > self.account.available_balance:
                    return False

            # 检查持仓限制
            if order.symbol in self.account.positions:
                position = self.account.positions[order.symbol]
                new_quantity = position.quantity + order.quantity
                if abs(new_quantity) > self.risk_limits['max_position_size']:
                    return False

            # 检查总风险暴露
            total_exposure = sum(abs(pos.market_value) for pos in self.account.positions.values())
            if total_exposure > self.risk_limits['max_total_exposure']:
                return False

            return True

        except Exception as e:
            logger.error(f"风险检查失败: {e}")
            return False


class ExecutionEngine:
    """执行引擎"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    def initialize(self) -> bool:
        logger.info("执行引擎初始化完成")
        return True

    def fill_order(self, order_id: str, quantity: float, price: float):
        """成交订单"""
        try:
            # 这里应该是实际的订单路由和成交逻辑
            # 现在只是模拟成交
            logger.info(f"订单成交: {order_id} - {quantity}@{price}")
        except Exception as e:
            logger.error(f"订单成交失败: {e}")


def test_core_business_engine():
    """测试核心业务引擎"""
    logger.info("测试核心业务功能重建...")

    # 初始化引擎
    engine = CoreBusinessEngine()
    if not engine.initialize():
        logger.error("引擎初始化失败")
        return

    # 1. 提交买入订单
    logger.info("\n1. 提交买入订单")
    success, message, order_id1 = engine.submit_order(
        symbol="AAPL",
        transaction_type=TransactionType.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    if success:
        logger.info(f"买入订单提交成功: {message} (ID: {order_id1})")
    else:
        logger.error(f"买入订单提交失败: {message}")

    # 2. 提交卖出订单
    logger.info("\n2. 提交卖出订单")
    success, message, order_id2 = engine.submit_order(
        symbol="AAPL",
        transaction_type=TransactionType.SELL,
        quantity=50,
        order_type=OrderType.LIMIT,
        price=160.0
    )
    if success:
        logger.info(f"卖出订单提交成功: {message} (ID: {order_id2})")
    else:
        logger.error(f"卖出订单提交失败: {message}")

    # 3. 查看账户摘要
    logger.info("\n3. 查看账户摘要")
    summary = engine.get_account_summary()
    logger.info("账户摘要:")
    for key, value in summary.items():
        if key != 'stats':
            logger.info(f"  {key}: {value}")

    # 4. 查看持仓
    logger.info("\n4. 查看持仓")
    positions = engine.get_positions()
    logger.info(f"当前持仓数量: {len(positions)}")
    for pos in positions:
        logger.info(
            f"  {pos['symbol']}: {pos['quantity']}股 @ {pos['average_price']:.2f} (市值: {pos['market_value']:.2f})")

    # 5. 查看订单
    logger.info("\n5. 查看订单")
    orders = engine.get_orders()
    logger.info(f"活跃订单数量: {len(orders)}")
    for order in orders:
        logger.info(
            f"  {order['order_id']}: {order['transaction_type']} {order['quantity']} {order['symbol']} ({order['status']})")

    # 6. 更新市场价格
    logger.info("\n6. 更新市场价格")
    price_updates = {"AAPL": 155.0, "GOOGL": 2550.0}
    engine.update_market_prices(price_updates)
    logger.info("市场价格更新完成")

    # 7. 查看更新后的账户摘要
    logger.info("\n7. 查看更新后的账户摘要")
    summary = engine.get_account_summary()
    logger.info("更新后的账户摘要:")
    logger.info(f"  余额: {summary['balance']:.2f}")
    logger.info(f"  总价值: {summary['total_value']:.2f}")
    logger.info(f"  总盈亏: {summary['total_pnl']:.2f}")

    logger.info("\n✅ 核心业务功能重建测试完成")


if __name__ == "__main__":
    test_core_business_engine()
