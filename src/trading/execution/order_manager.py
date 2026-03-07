# -*- coding: utf-8 -*-
"""
订单管理系统 - Phase 4修复版本
完全重构的订单管理功能，支持完整的订单生命周期管理
"""

import datetime as dt
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import uuid
import logging
import queue

# 从trading.core.constants导入（注意：...core.constants指向trading.core.constants，不是src.core.constants）
# 必须在类定义之前导入，因为类定义的默认参数需要用到这些常量
try:
    from ...core.constants import (
        ORDER_CACHE_SIZE,
        POSITION_CACHE_SIZE,
        CACHE_TTL_SECONDS,
        DEFAULT_ORDER_TIMEOUT,
        MAX_ORDERS_PER_SECOND,
        DEFAULT_BATCH_SIZE,
        MAX_BATCH_SIZE,
        MAX_POSITION_SIZE
    )
except ImportError:
    # 如果导入失败，使用默认值（确保常量在模块级别定义）
    ORDER_CACHE_SIZE = 10000
    POSITION_CACHE_SIZE = 1000
    CACHE_TTL_SECONDS = 3600
    DEFAULT_ORDER_TIMEOUT = 300
    MAX_ORDERS_PER_SECOND = 100
    DEFAULT_BATCH_SIZE = 100
    MAX_BATCH_SIZE = 1000
    MAX_POSITION_SIZE = 1000000

try:
    from ...core.exceptions import *
except ImportError:
    pass

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格
    CANCEL = "cancel"


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    PARTIALLY_FILLED = "partially_filled"  # 兼容性别名
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """订单数据结构 - 修复版本"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = field(init=False)
    avg_fill_price: float = 0.0
    created_time: dt.datetime = field(default_factory=dt.datetime.now)
    updated_time: dt.datetime = field(default_factory=dt.datetime.now)
    strategy_id: Optional[str] = None
    account_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        self.remaining_quantity = abs(self.quantity) - self.filled_quantity
        if self.metadata is None:
            self.metadata = {}

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
        self.updated_time = dt.datetime.now()
        if error_message:
            self.error_message = error_message

    def add_fill(self, fill_quantity: float, fill_price: float):
        """添加成交记录"""
        if fill_quantity <= 0 or fill_price <= 0:
            raise ValueError("成交数量和价格必须为正数")

        if fill_quantity > self.remaining_quantity:
            raise ValueError("成交数量不能超过剩余数量")

        # 更新成交信息
        old_filled_value = self.filled_quantity * self.avg_fill_price
        new_filled_value = fill_quantity * fill_price
        total_filled_value = old_filled_value + new_filled_value

        self.filled_quantity += fill_quantity
        self.remaining_quantity = abs(self.quantity) - self.filled_quantity
        self.avg_fill_price = total_filled_value / self.filled_quantity if self.filled_quantity > 0 else 0
        self.updated_time = dt.datetime.now()

        # 更新状态
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL

    def __hash__(self):
        """计算哈希值，用于集合和字典"""
        return hash(self.order_id)

    def __eq__(self, other):
        """检查相等性"""
        if not isinstance(other, Order):
            return False
        return self.order_id == other.order_id


@dataclass
class OrderValidationResult:
    """订单验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class OrderValidator:
    """订单验证器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_order(self, order: Order) -> OrderValidationResult:
        """验证订单"""
        errors = []
        warnings = []

        # 基本字段验证
        if not order.symbol or not order.symbol.strip():
            errors.append("交易标的不能为空")

        if order.quantity <= 0:
            errors.append("订单数量必须大于0")

        if order.side not in [OrderSide.BUY, OrderSide.SELL]:
            errors.append("无效的订单方向")

        # 价格验证
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                errors.append(f"{order.order_type.value}订单必须指定有效价格")

        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
            if order.stop_price is None or order.stop_price <= 0:
                errors.append(f"{order.order_type.value}订单必须指定有效止损价格")

        # 业务规则验证
        if order.order_type == OrderType.STOP_LIMIT:
            if order.price is not None and order.stop_price is not None and order.price <= order.stop_price:
                warnings.append("止损限价单的价格应高于止损价格")

        # 数量合理性检查
        if order.quantity > MAX_POSITION_SIZE:  # 单笔订单上限检查
            warnings.append("订单数量较大，请确认是否有足够资金")

        return OrderValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class OrderManager:
    """订单管理系统 - Phase 4修复版本"""

    def __init__(self, max_orders: int = ORDER_CACHE_SIZE):
        self.max_orders = max_orders
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.order_queue = queue.Queue(maxsize=max_orders)
        self.validator = OrderValidator()

        # 超时和重试配置
        self.order_timeout_seconds = DEFAULT_ORDER_TIMEOUT
        self.retry_attempts = 3

        # 统计信息
        self.stats = {
            'total_submitted': 0,
            'total_filled': 0,
            'total_cancelled': 0,
            'total_rejected': 0
        }

        self.logger = logging.getLogger(__name__)
        self.logger.info("订单管理器初始化完成")

    def create_order(self, symbol: str, order_type: OrderType, quantity: float,
                    price: Optional[float] = None, direction: Optional[str] = None,
                    **kwargs) -> Order:
        """
        创建订单

        Args:
            symbol: 交易标的
            order_type: 订单类型
            quantity: 数量（正数买入，负数卖出）
            price: 价格（限价单需要）
            direction: 方向（可选，会根据quantity自动判断）
            **kwargs: 其他参数

        Returns:
            Order对象

        Raises:
            ValueError: 当参数无效时抛出
        """
        # 参数验证
        if not symbol or not isinstance(symbol, str):
            raise ValueError("无效的交易标的")

        if not isinstance(quantity, (int, float)) or quantity == 0:
            raise ValueError("无效的数量")

        # 基于字符串值进行比较，以兼容不同模块的枚举
        order_type_str = order_type.value if hasattr(order_type, 'value') else str(order_type)
        if order_type_str == "limit" and (price is None or price <= 0):
            raise ValueError("限价订单必须指定有效的价格")

        if order_type_str == "stop" and (price is None or price <= 0):
            raise ValueError("止损订单必须指定有效的价格")

        # 确定方向
        if direction is None:
            direction = "buy" if quantity > 0 else "sell"

        # 转换为OrderSide枚举
        from .order_manager import OrderSide
        side = OrderSide.BUY if direction == "buy" else OrderSide.SELL

        # 创建订单对象
        order = Order(
            symbol=symbol,
            order_type=order_type,
            quantity=abs(quantity),
            price=price,
            side=side,
            **kwargs
        )

        return order

    def submit_order(self, order: Order) -> Tuple[bool, str, Optional[str]]:
        """
        提交订单

        Returns:
            (成功标志, 消息, 订单ID)
        """
        try:
            # 验证订单
            validation = self.validator.validate_order(order)
            if not validation.is_valid:
                error_msg = "; ".join(validation.errors)
                self.logger.warning(f"订单验证失败: {error_msg}")
                self.stats['total_rejected'] += 1
                return False, error_msg, None

            # 检查队列容量
            if len(self.active_orders) >= self.max_orders:
                return False, "订单队列已满", None

            # 生成订单ID（如果没有提供）
            if not order.order_id:
                order.order_id = str(uuid.uuid4())

            # 检查订单ID是否重复
            if order.order_id in self.active_orders or order.order_id in self.completed_orders:
                return False, "订单ID重复", None

            # 设置订单状态
            order.status = OrderStatus.SUBMITTED
            order.updated_time = dt.datetime.now()

            # 添加到活跃订单
            self.active_orders[order.order_id] = order
            self.stats['total_submitted'] += 1

            # 添加到队列
            try:
                self.order_queue.put(order, timeout=1)
            except queue.Full:
                # 从活跃订单中移除
                del self.active_orders[order.order_id]
                return False, "订单队列已满", None

            # 记录警告信息
            if validation.warnings:
                warning_msg = "; ".join(validation.warnings)
                self.logger.info(f"订单提交警告: {warning_msg}")

            self.logger.info(
                f"订单提交成功: {order.order_id} - {order.side.value} {order.quantity} {order.symbol}")
            return True, "订单提交成功", order.order_id

        except Exception as e:
            self.logger.error(f"订单提交异常: {e}")
            return False, f"系统错误: {str(e)}", None

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        取消订单

        Returns:
            (成功标志, 消息)
        """
        try:
            if order_id not in self.active_orders:
                return False, "订单不存在或已完成"

            order = self.active_orders[order_id]

            if not order.is_active:
                return False, f"订单状态为{order.status.value}，无法取消"

            # 更新订单状态
            order.update_status(OrderStatus.CANCELLED)
            order.updated_time = dt.datetime.now()

            # 移动到完成订单
            self.completed_orders[order_id] = order
            del self.active_orders[order_id]

            self.stats['total_cancelled'] += 1

            self.logger.info(f"订单取消成功: {order_id}")
            return True, "订单取消成功"

        except Exception as e:
            self.logger.error(f"订单取消异常: {e}")
            return False, f"系统错误: {str(e)}"

    def update_order_status(self, order_id: str, status: OrderStatus,
                            filled_quantity: float = 0, fill_price: float = 0,
                            error_message: Optional[str] = None) -> bool:
        """更新订单状态"""
        try:
            if order_id not in self.active_orders:
                self.logger.warning(f"订单不存在: {order_id}")
                return False

            order = self.active_orders[order_id]

            # 处理成交
            if filled_quantity > 0:
                order.add_fill(filled_quantity, fill_price)

            # 更新状态
            order.update_status(status, error_message)

            # 如果订单完成，移动到完成订单列表
            if order.is_completed:
                self.completed_orders[order_id] = order
                del self.active_orders[order_id]

                if status == OrderStatus.FILLED:
                    self.stats['total_filled'] += 1
                elif status == OrderStatus.CANCELLED:
                    self.stats['total_cancelled'] += 1
                elif status == OrderStatus.REJECTED:
                    self.stats['total_rejected'] += 1

            self.logger.info(f"订单状态更新: {order_id} -> {status.value}")
            return True

        except Exception as e:
            self.logger.error(f"订单状态更新异常: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.active_orders.get(order_id) or self.completed_orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取活跃订单"""
        orders = list(self.active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_completed_orders(self, symbol: Optional[str] = None, limit: int = DEFAULT_BATCH_SIZE) -> List[Order]:
        """获取完成订单"""
        orders = list(self.completed_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders[-limit:]  # 返回最新的limit个订单

    def cancel_all_orders(self, symbol: Optional[str] = None) -> Tuple[int, int]:
        """
        取消所有订单

        Returns:
            (成功取消数量, 失败数量)
        """
        orders_to_cancel = self.get_active_orders(symbol)
        success_count = 0
        fail_count = 0

        for order in orders_to_cancel:
            success, _ = self.cancel_order(order.order_id)
            if success:
                success_count += 1
            else:
                fail_count += 1

        self.logger.info(f"批量取消订单完成: {success_count}成功, {fail_count}失败")
        return success_count, fail_count

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'queue_size': self.order_queue.qsize(),
            'max_orders': self.max_orders,
            'queue_utilization': len(self.active_orders) / self.max_orders
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_orders = self.stats['total_submitted']
        success_rate = self.stats['total_filled'] / total_orders if total_orders > 0 else 0

        return {
            'total_submitted': self.stats['total_submitted'],
            'total_filled': self.stats['total_filled'],
            'total_cancelled': self.stats['total_cancelled'],
            'total_rejected': self.stats['total_rejected'],
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'success_rate': success_rate,
            'queue_utilization': len(self.active_orders) / self.max_orders
        }

    def cleanup_expired_orders(self, max_age_hours: int = 24) -> int:
        """清理过期订单"""
        cutoff_time = dt.datetime.now() - timedelta(hours=max_age_hours)
        expired_orders = []

        for order_id, order in list(self.active_orders.items()):
            if order.created_time < cutoff_time and order.status == OrderStatus.PENDING:
                order.update_status(OrderStatus.EXPIRED, "订单过期")
                self.completed_orders[order_id] = order
                del self.active_orders[order_id]
                expired_orders.append(order_id)

        if expired_orders:
            self.logger.info(f"清理了{len(expired_orders)}个过期订单")

        return len(expired_orders)
