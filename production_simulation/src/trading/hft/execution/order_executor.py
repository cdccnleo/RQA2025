"""订单执行器模块"""

from typing import Dict, Any, Optional, List
from enum import Enum
from abc import ABC, abstractmethod
import time


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
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Order:

    """订单类"""

    def __init__(self, symbol: str, side: OrderSide, order_type: OrderType,


                 quantity: float, price: Optional[float] = None,
                 order_id: Optional[str] = None):
        """初始化订单

        Args:
            symbol: 交易标的
            side: 订单方向
            order_type: 订单类型
            quantity: 数量
            price: 价格（市价单可为None）
            order_id: 订单ID
        """
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.order_id = order_id or f"order_{int(time.time() * 1000)}"
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.avg_price = 0.0
        self.submit_time = None
        self.fill_time = None

    def submit(self) -> None:
        """提交订单"""
        self.status = OrderStatus.SUBMITTED
        self.submit_time = time.time()

    def fill(self, quantity: float, price: float) -> None:
        """成交订单

        Args:
            quantity: 成交数量
            price: 成交价格
        """
        if quantity > self.quantity - self.filled_quantity:
            raise ValueError("成交数量超过剩余数量")

        self.filled_quantity += quantity
        self.avg_price = (self.avg_price * (self.filled_quantity - quantity)
                          + price * quantity) / self.filled_quantity

        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.fill_time = time.time()
        else:
            self.status = OrderStatus.PARTIAL_FILLED

    def cancel(self) -> None:
        """取消订单"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError("订单状态不允许取消")
        self.status = OrderStatus.CANCELLED


class OrderExecutor(ABC):

    """订单执行器基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化订单执行器

        Args:
            config: 执行器配置
        """
        self.config = config or {}
        self.orders: Dict[str, Order] = {}

    @abstractmethod
    def submit_order(self, order: Order) -> bool:
        """提交订单

        Args:
            order: 订单对象

        Returns:
            是否提交成功
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单

        Args:
            order_id: 订单ID

        Returns:
            是否取消成功
        """

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单状态
        """

    def get_orders(self) -> List[Order]:
        """获取所有订单

        Returns:
            订单列表
        """
        return list(self.orders.values())


class ChinaOrderExecutor(OrderExecutor):

    """中国股市订单执行器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化中国股市订单执行器

        Args:
            config: 执行器配置，包含：
                - market_open_time: 开市时间
                - market_close_time: 闭市时间
                - max_order_size: 最大订单数量
        """
        super().__init__(config)
        self.market_open_time = config.get('market_open_time', '09:30')
        self.market_close_time = config.get('market_close_time', '15:00')
        self.max_order_size = config.get('max_order_size', 1000000)

    def submit_order(self, order: Order) -> bool:
        """提交订单

        Args:
            order: 订单对象

        Returns:
            是否提交成功
        """
        # 检查订单数量限制
        if order.quantity > self.max_order_size:
            return False

        # 检查市价单价格
        if order.order_type == OrderType.MARKET and order.price is not None:
            return False

        # 检查限价单价格
        if order.order_type == OrderType.LIMIT and order.price is None:
            return False

        order.submit()
        self.orders[order.order_id] = order
        return True

    def cancel_order(self, order_id: str) -> bool:
        """取消订单

        Args:
            order_id: 订单ID

        Returns:
            是否取消成功
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        try:
            order.cancel()
            return True
        except ValueError:
            return False

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单状态
        """
        order = self.orders.get(order_id)
        return order.status if order else None


class OrderManager:

    """订单管理器"""

    def __init__(self, executor: OrderExecutor):
        """初始化订单管理器

        Args:
            executor: 订单执行器
        """
        self.executor = executor
        self.order_history: List[Order] = []

    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,


                    quantity: float, price: Optional[float] = None) -> Optional[str]:
        """下单

        Args:
            symbol: 交易标的
            side: 订单方向
            order_type: 订单类型
            quantity: 数量
            price: 价格

        Returns:
            订单ID，失败返回None
        """
        order = Order(symbol, side, order_type, quantity, price)

        if self.executor.submit_order(order):
            return order.order_id
        return None

    def cancel_order(self, order_id: str) -> bool:
        """取消订单

        Args:
            order_id: 订单ID

        Returns:
            是否取消成功
        """
        return self.executor.cancel_order(order_id)

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单状态
        """
        return self.executor.get_order_status(order_id)

    def get_active_orders(self) -> List[Order]:
        """获取活跃订单

        Returns:
            活跃订单列表
        """
        return [order for order in self.executor.get_orders()
                if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]]
