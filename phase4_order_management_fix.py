#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 订单管理系统重构脚本
修复订单管理功能，实现完整的订单生命周期管理
"""

import uuid
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import queue


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    CANCEL = "cancel"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """订单类"""
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
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
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
        self.updated_time = datetime.now()
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
        self.updated_time = datetime.now()

        # 更新状态
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL


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
            if order.price <= order.stop_price:
                warnings.append("止损限价单的价格应高于止损价格")

        # 数量合理性检查
        if order.quantity > 1000000:  # 假设单笔订单上限
            warnings.append("订单数量较大，请确认是否有足够资金")

        return OrderValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class OrderManager:
    """重构后的订单管理器"""

    def __init__(self, max_orders: int = 10000):
        self.max_orders = max_orders
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.order_queue = queue.Queue(maxsize=max_orders)
        self.validator = OrderValidator()

        # 统计信息
        self.stats = {
            'total_submitted': 0,
            'total_filled': 0,
            'total_cancelled': 0,
            'total_rejected': 0
        }

        self.logger = logging.getLogger(__name__)
        self.logger.info("订单管理器初始化完成")

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
            order.updated_time = datetime.now()

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
            order.updated_time = datetime.now()

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

    def get_completed_orders(self, symbol: Optional[str] = None, limit: int = 100) -> List[Order]:
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
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
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


class OrderRouter:
    """订单路由器"""

    def __init__(self, order_manager: OrderManager):
        self.order_manager = order_manager
        self.routing_rules: Dict[str, List[str]] = {}  # symbol -> brokers
        self.broker_status: Dict[str, bool] = {}  # broker -> is_available
        self.logger = logging.getLogger(__name__)

    def add_routing_rule(self, symbol: str, brokers: List[str]):
        """添加路由规则"""
        self.routing_rules[symbol] = brokers

    def update_broker_status(self, broker: str, is_available: bool):
        """更新经纪商状态"""
        self.broker_status[broker] = is_available

    def route_order(self, order: Order) -> Optional[str]:
        """路由订单到最佳经纪商"""
        symbol = order.symbol

        # 获取可用经纪商
        available_brokers = self.routing_rules.get(symbol, [])
        available_brokers = [b for b in available_brokers if self.broker_status.get(b, False)]

        if not available_brokers:
            self.logger.warning(f"没有可用的经纪商处理{symbol}订单")
            return None

        # 简单的路由策略：轮询
        # 在实际系统中，这里会考虑费用、速度、可靠性等因素
        broker = available_brokers[0]  # 简化：选择第一个可用经纪商

        self.logger.info(f"订单{order.order_id}路由到经纪商: {broker}")
        return broker


def test_order_management():
    """测试订单管理功能"""
    print("测试订单管理系统...")

    # 创建订单管理器
    manager = OrderManager(max_orders=1000)

    # 创建订单路由器
    router = OrderRouter(manager)
    router.add_routing_rule("AAPL", ["broker1", "broker2"])
    router.update_broker_status("broker1", True)
    router.update_broker_status("broker2", True)

    # 测试订单创建和提交
    print("\n1. 测试订单提交")

    # 创建买入订单
    buy_order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        price=150.0,
        strategy_id="test_strategy"
    )

    success, message, order_id = manager.submit_order(buy_order)
    print(f"买入订单提交: {'成功' if success else '失败'} - {message}")
    print(f"订单ID: {order_id}")

    # 创建卖出订单
    sell_order = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=50,
        strategy_id="test_strategy"
    )

    success, message, sell_order_id = manager.submit_order(sell_order)
    print(f"卖出订单提交: {'成功' if success else '失败'} - {message}")
    print(f"订单ID: {sell_order_id}")

    # 测试订单查询
    print("\n2. 测试订单查询")

    if order_id:
        order = manager.get_order(order_id)
        if order:
            print(f"订单状态: {order.status.value}")
            print(f"剩余数量: {order.remaining_quantity}")

    # 测试订单成交
    print("\n3. 测试订单成交")

    if order_id:
        manager.update_order_status(order_id, OrderStatus.PARTIAL,
                                    filled_quantity=50, fill_price=150.0)
        order = manager.get_order(order_id)
        if order:
            print(f"部分成交后状态: {order.status.value}")
            print(f"已成交数量: {order.filled_quantity}")
            print(f"平均成交价: {order.avg_fill_price}")

        # 完成剩余部分
        manager.update_order_status(order_id, OrderStatus.FILLED,
                                    filled_quantity=50, fill_price=151.0)
        order = manager.get_order(order_id)
        if order:
            print(f"完全成交后状态: {order.status.value}")
            print(f"最终平均价: {order.avg_fill_price}")

    # 测试订单取消
    print("\n4. 测试订单取消")

    if sell_order_id:
        success, message = manager.cancel_order(sell_order_id)
        print(f"订单取消: {'成功' if success else '失败'} - {message}")

        order = manager.get_order(sell_order_id)
        if order:
            print(f"取消后状态: {order.status.value}")

    # 测试统计信息
    print("\n5. 测试统计信息")

    stats = manager.get_statistics()
    print("订单统计:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if 'rate' in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # 测试队列状态
    print("\n6. 测试队列状态")

    queue_status = manager.get_queue_status()
    print("队列状态:")
    for key, value in queue_status.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1%}" if 'utilization' in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n✅ 订单管理系统测试完成")


if __name__ == "__main__":
    test_order_management()
