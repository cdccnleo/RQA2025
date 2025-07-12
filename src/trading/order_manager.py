import datetime
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import pandas as pd
import numpy as np
from queue import PriorityQueue

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """订单类型枚举"""
    MARKET = auto()      # 市价单
    LIMIT = auto()       # 限价单
    STOP = auto()        # 止损单
    STOP_LIMIT = auto()  # 止损限价单
    TWAP = auto()        # 时间加权单
    VWAP = auto()        # 成交量加权单

class OrderStatus(Enum):
    """订单状态枚举"""
    NEW = auto()            # 新订单
    PENDING_NEW = auto()    # 待确认
    PARTIALLY_FILLED = auto()  # 部分成交
    FILLED = auto()         # 完全成交
    CANCELLED = auto()      # 已取消
    REJECTED = auto()       # 已拒绝
    EXPIRED = auto()        # 已过期

@dataclass
class Order:
    """订单数据结构"""
    order_id: str                   # 订单ID
    symbol: str                     # 交易标的
    order_type: OrderType           # 订单类型
    quantity: float                 # 订单数量(正数为买,负数为卖)
    price: Optional[float] = None   # 限价/止损价(可选)
    stop_price: Optional[float] = None  # 止损触发价(可选)
    time_in_force: str = "DAY"      # 有效时间(GTC/IOC/FOK/DAY)
    status: OrderStatus = OrderStatus.NEW  # 订单状态
    created_time: datetime.datetime = datetime.datetime.now()  # 创建时间
    filled_quantity: float = 0.0   # 已成交数量
    avg_fill_price: float = 0.0     # 平均成交价
    last_updated: datetime.datetime = datetime.datetime.now()  # 最后更新时间
    parent_id: Optional[str] = None  # 母订单ID(用于拆分订单)
    strategy_id: Optional[str] = None  # 策略ID
    metadata: Dict = None           # 附加元数据

class OrderManager:
    """订单管理系统"""

    def __init__(self, max_queue_size: int = 10000):
        """
        初始化订单管理器

        Args:
            max_queue_size: 最大订单队列长度
        """
        self.active_orders: Dict[str, Order] = {}  # 活动订单字典
        self.order_queue = PriorityQueue(maxsize=max_queue_size)  # 订单优先级队列
        self.order_history: List[Order] = []  # 订单历史记录
        self.next_order_id = 1  # 订单ID计数器

    def generate_order_id(self) -> str:
        """生成唯一订单ID"""
        order_id = f"ORD{self.next_order_id:08d}"
        self.next_order_id += 1
        return order_id

    def create_order(self,
                    symbol: str,
                    quantity: float,
                    order_type: OrderType,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: str = "DAY",
                    strategy_id: Optional[str] = None,
                    metadata: Dict = None) -> Order:
        """
        创建新订单

        Args:
            symbol: 交易标的
            quantity: 数量(正为买,负为卖)
            order_type: 订单类型
            price: 限价(可选)
            stop_price: 止损价(可选)
            time_in_force: 有效时间
            strategy_id: 策略ID(可选)
            metadata: 附加元数据(可选)

        Returns:
            创建的订单对象
        """
        # 参数校验
        if quantity == 0:
            raise ValueError("Order quantity cannot be zero")
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            raise ValueError("Price must be specified for limit orders")
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            raise ValueError("Stop price must be specified for stop orders")

        # 创建订单对象
        order = Order(
            order_id=self.generate_order_id(),
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            status=OrderStatus.NEW,
            strategy_id=strategy_id,
            metadata=metadata or {}
        )

        # 添加到活动订单字典
        self.active_orders[order.order_id] = order
        logger.info(f"Created new order: {order.order_id}")

        return order

    def submit_order(self, order: Order) -> bool:
        """
        提交订单到执行队列

        Args:
            order: 要提交的订单

        Returns:
            是否提交成功
        """
        if order.order_id not in self.active_orders:
            logger.error(f"Order {order.order_id} not found in active orders")
            return False

        try:
            # 设置状态为待确认
            order.status = OrderStatus.PENDING_NEW
            order.last_updated = datetime.datetime.now()

            # 根据订单类型确定优先级
            if order.order_type in [OrderType.MARKET, OrderType.STOP]:
                priority = 1  # 高优先级
            else:
                priority = 2  # 普通优先级

            # 添加到优先级队列
            self.order_queue.put((priority, order))
            logger.info(f"Submitted order {order.order_id} to execution queue")
            return True
        except Exception as e:
            logger.error(f"Failed to submit order {order.order_id}: {str(e)}")
            order.status = OrderStatus.REJECTED
            order.last_updated = datetime.datetime.now()
            return False

    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单

        Args:
            order_id: 要取消的订单ID

        Returns:
            是否取消成功
        """
        if order_id not in self.active_orders:
            logger.error(f"Order {order_id} not found in active orders")
            return False

        order = self.active_orders[order_id]

        # 检查订单状态是否可以取消
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            logger.warning(f"Order {order_id} is already {order.status.name}, cannot cancel")
            return False

        # 更新订单状态
        order.status = OrderStatus.CANCELLED
        order.last_updated = datetime.datetime.now()
        logger.info(f"Cancelled order {order_id}")

        return True

    def update_order_status(self,
                          order_id: str,
                          status: OrderStatus,
                          filled_qty: Optional[float] = None,
                          fill_price: Optional[float] = None) -> bool:
        """
        更新订单状态

        Args:
            order_id: 订单ID
            status: 新状态
            filled_qty: 已成交数量(可选)
            fill_price: 成交价格(可选)

        Returns:
            是否更新成功
        """
        if order_id not in self.active_orders:
            logger.error(f"Order {order_id} not found in active orders")
            return False

        order = self.active_orders[order_id]

        # 更新状态
        order.status = status
        order.last_updated = datetime.datetime.now()

        # 更新成交信息
        if filled_qty is not None:
            order.filled_quantity = filled_qty
        if fill_price is not None and filled_qty is not None:
            if order.filled_quantity != 0:
                order.avg_fill_price = (
                    (order.avg_fill_price * (order.filled_quantity - filled_qty) +
                     fill_price * filled_qty) / order.filled_quantity
                )

        # 如果订单完成，移到历史记录
        if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self.order_history.append(order)
            del self.active_orders[order_id]

        logger.info(f"Updated order {order_id} to status {status.name}")
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单信息"""
        return self.active_orders.get(order_id)

    def get_active_orders(self,
                        symbol: Optional[str] = None,
                        strategy_id: Optional[str] = None) -> List[Order]:
        """获取活动订单列表"""
        orders = list(self.active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if strategy_id:
            orders = [o for o in orders if o.strategy_id == strategy_id]
        return orders

    def get_order_history(self,
                        start_time: Optional[datetime.datetime] = None,
                        end_time: Optional[datetime.datetime] = None) -> List[Order]:
        """获取历史订单"""
        if start_time is None and end_time is None:
            return self.order_history

        filtered = []
        for order in self.order_history:
            if start_time and order.created_time < start_time:
                continue
            if end_time and order.created_time > end_time:
                continue
            filtered.append(order)
        return filtered

class TWAPExecution:
    """TWAP执行算法"""

    def __init__(self, order: Order, slices: int = 10):
        """
        初始化TWAP执行

        Args:
            order: 母订单
            slices: 拆分数量
        """
        if order.order_type != OrderType.TWAP:
            raise ValueError("Only TWAP orders can use TWAP execution")

        self.parent_order = order
        self.slices = slices
        self.slice_orders = []
        self.next_slice = 0
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(minutes=30)  # 默认30分钟执行期

    def generate_slice_orders(self, order_manager: OrderManager) -> List[Order]:
        """生成切片订单"""
        if self.slice_orders:
            return self.slice_orders

        slice_qty = self.parent_order.quantity / self.slices
        interval = (self.end_time - self.start_time) / self.slices

        for i in range(self.slices):
            slice_time = self.start_time + i * interval
            slice_order = order_manager.create_order(
                symbol=self.parent_order.symbol,
                quantity=slice_qty,
                order_type=OrderType.MARKET,
                time_in_force="IOC",  # 立即成交否则取消
                parent_id=self.parent_order.order_id,
                strategy_id=self.parent_order.strategy_id,
                metadata={
                    "twap_slice": i+1,
                    "parent_order": self.parent_order.order_id
                }
            )
            self.slice_orders.append((slice_time, slice_order))

        return [o for _, o in self.slice_orders]

    def get_next_slice(self, current_time: datetime.datetime) -> Optional[Order]:
        """获取下一个应执行的切片订单"""
        if self.next_slice >= len(self.slice_orders):
            return None

        slice_time, slice_order = self.slice_orders[self.next_slice]
        if current_time >= slice_time:
            self.next_slice += 1
            return slice_order
        return None

class ExecutionEngine:
    """订单执行引擎"""

    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        self.twap_executions = {}  # TWAP执行跟踪器
        self.last_execution_time = datetime.datetime.now()

    def process_order_queue(self):
        """处理订单队列"""
        processed = 0
        current_time = datetime.datetime.now()

        while not self.order_manager.order_queue.empty():
            priority, order = self.order_manager.order_queue.get()

            # 处理TWAP订单
            if order.order_type == OrderType.TWAP:
                if order.order_id not in self.twap_executions:
                    self.twap_executions[order.order_id] = TWAPExecution(order)
                    self.twap_executions[order.order_id].generate_slice_orders(self.order_manager)

                twap_exec = self.twap_executions[order.order_id]
                slice_order = twap_exec.get_next_slice(current_time)
                if slice_order:
                    self._execute_order(slice_order)

            # 处理普通订单
            else:
                self._execute_order(order)

            processed += 1
            if processed >= 100:  # 每次最多处理100个订单
                break

        self.last_execution_time = current_time

    def _execute_order(self, order: Order):
        """执行单个订单(模拟)"""
        # 模拟执行 - 实际实现需要连接交易所API
        if order.order_type == OrderType.MARKET:
            # 假设全部成交
            self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.FILLED,
                order.quantity,
                self._get_market_price(order.symbol)
            )
        elif order.order_type == OrderType.LIMIT:
            # 模拟部分成交
            filled_qty = min(order.quantity, np.random.uniform(0, order.quantity))
            if filled_qty > 0:
                status = OrderStatus.PARTIALLY_FILLED if filled_qty < order.quantity else OrderStatus.FILLED
                self.order_manager.update_order_status(
                    order.order_id,
                    status,
                    filled_qty,
                    order.price
                )

    def _get_market_price(self, symbol: str) -> float:
        """获取当前市场价格(模拟)"""
        # 实际实现需要从行情系统获取
        return np.random.uniform(10, 100)
