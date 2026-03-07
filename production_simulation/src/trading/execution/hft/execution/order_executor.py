# -*- coding: utf-8 -*-
"""
高频交易订单执行器模块
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """订单类"""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None

    # 订单状态
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = field(init=False)
    average_fill_price: Optional[float] = None

    # 时间戳
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None

    # 附加信息
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        self.remaining_quantity = self.quantity

    def update_fill(self, fill_quantity: float, fill_price: float):
        """更新成交信息"""
        if fill_quantity > self.remaining_quantity:
            raise ValueError(f"成交数量 {fill_quantity} 超过剩余数量 {self.remaining_quantity}")

        self.filled_quantity += fill_quantity
        self.remaining_quantity -= fill_quantity
        self.last_update_time = datetime.now()

        # 更新平均成交价
        if self.average_fill_price is None:
            self.average_fill_price = fill_price
        else:
            total_value = self.average_fill_price * \
                (self.filled_quantity - fill_quantity) + fill_price * fill_quantity
            self.average_fill_price = total_value / self.filled_quantity

        # 更新状态
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL

    def cancel(self):
        """取消订单"""
        if self.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            self.status = OrderStatus.CANCELLED
            self.last_update_time = datetime.now()

    def is_completed(self) -> bool:
        """检查订单是否完成"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]

    def is_active(self) -> bool:
        """检查订单是否活跃"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]


class OrderExecutor:
    """订单执行器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化订单执行器

        Args:
            config: 执行器配置
        """
        self.config = config or {}
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

        # 性能统计
        self.total_orders = 0
        self.filled_orders = 0
        self.cancelled_orders = 0

        logger.info("订单执行器初始化完成")

    def submit_order(self, order: Order) -> bool:
        """提交订单

        Args:
            order: 订单对象

        Returns:
            是否提交成功
        """
        try:
            # 验证订单
            if not self._validate_order(order):
                logger.error(f"订单验证失败: {order.order_id}")
                order.status = OrderStatus.REJECTED
                return False

            # 设置提交时间
            order.submitted_time = datetime.now()
            order.status = OrderStatus.SUBMITTED

            # 添加到活跃订单
            self.active_orders[order.order_id] = order
            self.total_orders += 1

            logger.info(
                f"订单已提交: {order.order_id}, {order.symbol}, {order.side.value}, {order.quantity}")

            # 模拟执行过程
            self._simulate_execution(order)

            return True

        except Exception as e:
            logger.error(f"提交订单失败: {order.order_id}, 错误: {str(e)}")
            order.status = OrderStatus.REJECTED
            return False

    def cancel_order(self, order_id: str) -> bool:
        """取消订单

        Args:
            order_id: 订单ID

        Returns:
            是否取消成功
        """
        order = self.active_orders.get(order_id)
        if order is None:
            logger.warning(f"订单不存在: {order_id}")
            return False

        if order.is_completed():
            logger.warning(f"订单已完成，无法取消: {order_id}")
            return False

        order.cancel()
        self.cancelled_orders += 1

        # 从活跃订单中移除
        del self.active_orders[order_id]

        logger.info(f"订单已取消: {order_id}")
        return True

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单对象或None
        """
        return self.active_orders.get(order_id)

    def get_order_history(self, symbol: Optional[str] = None,
                          status: Optional[OrderStatus] = None) -> List[Order]:
        """获取订单历史

        Args:
            symbol: 股票代码过滤
            status: 状态过滤

        Returns:
            订单列表
        """
        orders = self.order_history.copy()

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        if status:
            orders = [o for o in orders if o.status == status]

        return orders

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        total_value = sum(
            (order.filled_quantity * order.average_fill_price)
            for order in self.order_history
            if order.average_fill_price is not None
        )

        return {
            "total_orders": self.total_orders,
            "active_orders": len(self.active_orders),
            "filled_orders": self.filled_orders,
            "cancelled_orders": self.cancelled_orders,
            "fill_rate": self.filled_orders / self.total_orders if self.total_orders > 0 else 0,
            "total_value": total_value
        }

    def _validate_order(self, order: Order) -> bool:
        """验证订单

        Args:
            order: 订单对象

        Returns:
            是否有效
        """
        # 基本验证
        if order.quantity <= 0:
            logger.error(f"订单数量无效: {order.quantity}")
            return False

        if order.order_type == OrderType.LIMIT and order.price is None:
            logger.error(f"限价单必须指定价格")
            return False

        # 这里可以添加更多验证逻辑
        # 如资金检查、持仓检查等

        return True

    def _simulate_execution(self, order: Order):
        """模拟订单执行

        Args:
            order: 订单对象
        """
        # 模拟执行延迟
        time.sleep(0.01)  # 10ms延迟

        # 简单的模拟成交逻辑
        if order.order_type == OrderType.MARKET:
            # 市价单立即成交
            fill_price = 100.0  # 模拟价格
            order.update_fill(order.quantity, fill_price)
            self.filled_orders += 1

        elif order.order_type == OrderType.LIMIT:
            # 限价单根据价格判断是否成交
            market_price = 100.0  # 模拟市场价格
            if (order.side == OrderSide.BUY and order.price >= market_price) or \
               (order.side == OrderSide.SELL and order.price <= market_price):
                order.update_fill(order.quantity, order.price)
                self.filled_orders += 1

        # 将完成的订单移到历史记录
        if order.is_completed():
            self.order_history.append(order)
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
