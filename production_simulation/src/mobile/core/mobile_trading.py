#!/usr/bin/env python3
"""
RQA2025 移动端交易功能
提供移动设备友好的完整交易体验
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from flask import Flask, render_template, request, jsonify
import threading
import time
import os
import uuid


logger = logging.getLogger(__name__)


class OrderType(Enum):

    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):

    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):

    """订单状态"""
    PENDING = "pending"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionType(Enum):

    """持仓类型"""
    LONG = "long"
    SHORT = "short"


@dataclass
class MobileOrder:

    """移动端订单"""
    order_id: str
    user_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    time_in_force: str = "GTC"
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    updated_at: datetime = None
    filled_quantity: float = 0
    remaining_quantity: float = 0
    average_fill_price: Optional[float] = None
    fees: float = 0
    strategy: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class MobilePosition:

    """移动端持仓"""
    position_id: str
    user_id: str
    symbol: str
    position_type: PositionType
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    created_at: datetime
    updated_at: datetime


@dataclass
class WatchlistItem:

    """自选股项目"""
    symbol: str
    name: str
    current_price: float
    change_percent: float
    volume: float
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    added_at: datetime = None


@dataclass
class MobileUser:

    """移动端用户"""
    user_id: str
    username: str
    email: str
    balance: float
    buying_power: float
    total_value: float
    daily_pnl: float
    total_pnl: float
    risk_score: int
    created_at: datetime
    last_login: datetime


class MobileTradingService:

    """移动端交易服务"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 数据存储（内存实现，实际应该使用数据库）
        self.users: Dict[str, MobileUser] = {}
        self.orders: Dict[str, MobileOrder] = {}
        self.positions: Dict[str, List[MobilePosition]] = {}
        self.watchlists: Dict[str, List[WatchlistItem]] = {}
        self.market_data: Dict[str, Dict[str, Any]] = {}

        # 交易配置
        self.min_order_size = self.config.get('min_order_size', 1)
        self.max_order_size = self.config.get('max_order_size', 1000000)
        self.commission_rate = self.config.get('commission_rate', 0.001)  # 0.1%

        # 市场数据更新
        self.market_update_interval = self.config.get('market_update_interval', 5)
        self.market_data_thread = None
        self.is_market_running = False

        logger.info("移动端交易服务初始化完成")

    def start_market_data_service(self):
        """启动市场数据服务"""
        if self.is_market_running:
            return

        self.is_market_running = True
        self.market_data_thread = threading.Thread(target=self._market_data_loop)
        self.market_data_thread.daemon = True
        self.market_data_thread.start()

        logger.info("市场数据服务已启动")

    def stop_market_data_service(self):
        """停止市场数据服务"""
        self.is_market_running = False
        if self.market_data_thread and self.market_data_thread.is_alive():
            self.market_data_thread.join(timeout=5)

        logger.info("市场数据服务已停止")

    def create_user(self, username: str, email: str, initial_balance: float = 10000) -> str:
        """创建用户"""
        user_id = str(uuid.uuid4())

        user = MobileUser(
            user_id=user_id,
            username=username,
            email=email,
            balance=initial_balance,
            buying_power=initial_balance,
            total_value=initial_balance,
            daily_pnl=0,
            total_pnl=0,
            risk_score=5,  # 中等风险偏好
            created_at=datetime.now(),
            last_login=datetime.now()
        )

        self.users[user_id] = user
        self.positions[user_id] = []
        self.watchlists[user_id] = []

        logger.info(f"创建用户: {username} ({user_id})")
        return user_id

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """用户认证"""
        # 简化的认证实现
        for user_id, user in self.users.items():
            if user.username == username:
                # 在实际实现中，这里应该验证密码哈希
                user.last_login = datetime.now()
                return user_id

        return None

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户资料"""
        if user_id not in self.users:
            return None

        user = self.users[user_id]
        positions = self.positions.get(user_id, [])
        orders = [order for order in self.orders.values() if order.user_id == user_id]

        # 计算总资产
        total_portfolio_value = sum(pos.market_value for pos in positions)
        total_value = user.balance + total_portfolio_value

        # 计算今日盈亏
        daily_pnl = sum(pos.unrealized_pnl for pos in positions)

        return {
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'balance': user.balance,
            'buying_power': user.buying_power,
            'total_value': total_value,
            'daily_pnl': daily_pnl,
            'total_pnl': user.total_pnl + daily_pnl,
            'positions_count': len(positions),
            'orders_count': len(orders),
            'risk_score': user.risk_score,
            'last_login': user.last_login.isoformat()
        }

    def place_order(self, user_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """下单"""
        if user_id not in self.users:
            return {'success': False, 'message': '用户不存在'}

        try:
            # 验证订单参数
            symbol = order_data.get('symbol', '').upper()
            side = OrderSide(order_data.get('side', 'buy'))
            order_type = OrderType(order_data.get('order_type', 'market'))
            quantity = float(order_data.get('quantity', 0))
            price = order_data.get('price')
            stop_price = order_data.get('stop_price')
            strategy = order_data.get('strategy')
            notes = order_data.get('notes')

            # 基本验证
            if not symbol or quantity <= 0:
                return {'success': False, 'message': '无效的订单参数'}

            if quantity < self.min_order_size or quantity > self.max_order_size:
                return {'success': False, 'message': f'订单数量必须在 {self.min_order_size} 到 {self.max_order_size} 之间'}

            # 验证价格参数
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if not price or price <= 0:
                    return {'success': False, 'message': '限价单必须指定有效价格'}

            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
                if not stop_price or stop_price <= 0:
                    return {'success': False, 'message': '止损单必须指定止损价格'}

            # 检查购买力
            user = self.users[user_id]
            market_price = self._get_market_price(symbol)

            if not market_price:
                return {'success': False, 'message': f'无法获取 {symbol} 的市场价格'}

            order_value = quantity * (price or market_price)
            estimated_fees = order_value * self.commission_rate

            if side == OrderSide.BUY and order_value + estimated_fees > user.buying_power:
                return {'success': False, 'message': '购买力不足'}

            # 检查持仓（卖出时）
            if side == OrderSide.SELL:
                current_position = self._get_position_quantity(user_id, symbol)
                if current_position < quantity:
                    return {'success': False, 'message': '持仓不足'}

            # 创建订单
            order_id = str(uuid.uuid4())

            order = MobileOrder(
                order_id=order_id,
                user_id=user_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force="GTC",
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                remaining_quantity=quantity,
                strategy=strategy,
                notes=notes
            )

            self.orders[order_id] = order

            # 尝试执行订单
            execution_result = self._execute_order(order)

            if execution_result['success']:
                order.status = OrderStatus.FILLED
                order.filled_quantity = quantity
                order.remaining_quantity = 0
                order.average_fill_price = execution_result['fill_price']
                order.fees = execution_result['fees']
                order.updated_at = datetime.now()

                # 更新用户账户
                self._update_user_account(user_id, order, execution_result)

                # 更新持仓
                self._update_position(user_id, order, execution_result)

            else:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()

            logger.info(
                f"用户 {user_id} 下单: {symbol} {side.value} {quantity}@{order.average_fill_price or price or market_price}")

            return {
                'success': execution_result['success'],
                'order_id': order_id,
                'message': execution_result.get('message', '订单已提交'),
                'fill_price': order.average_fill_price,
                'fees': order.fees
            }

        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {'success': False, 'message': f'下单失败: {str(e)}'}

    def cancel_order(self, user_id: str, order_id: str) -> Dict[str, Any]:
        """取消订单"""
        if order_id not in self.orders:
            return {'success': False, 'message': '订单不存在'}

        order = self.orders[order_id]

        if order.user_id != user_id:
            return {'success': False, 'message': '无权取消此订单'}

        if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED]:
            return {'success': False, 'message': '订单无法取消'}

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()

        logger.info(f"用户 {user_id} 取消订单: {order_id}")

        return {'success': True, 'message': '订单已取消'}

    def get_orders(self, user_id: str, status: Optional[OrderStatus] = None) -> List[Dict[str, Any]]:
        """获取用户订单"""
        user_orders = [order for order in self.orders.values() if order.user_id == user_id]

        if status:
            user_orders = [order for order in user_orders if order.status == status]

        # 按创建时间倒序排列
        user_orders.sort(key=lambda x: x.created_at, reverse=True)

        return [self._order_to_dict(order) for order in user_orders]

    def get_positions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户持仓"""
        user_positions = self.positions.get(user_id, [])

        # 更新当前价格和盈亏
        for position in user_positions:
            current_price = self._get_market_price(position.symbol)
            if current_price:
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (
                    current_price - position.average_cost) * position.quantity
                position.updated_at = datetime.now()

        return [self._position_to_dict(pos) for pos in user_positions]

    def manage_watchlist(self, user_id: str, action: str, symbol: str) -> Dict[str, Any]:
        """管理自选股"""
        if user_id not in self.users:
            return {'success': False, 'message': '用户不存在'}

        watchlist = self.watchlists.get(user_id, [])

        if action == 'add':
            # 检查是否已存在
            if any(item.symbol == symbol for item in watchlist):
                return {'success': False, 'message': '股票已在自选股中'}

            # 添加到自选股
            market_data = self.market_data.get(symbol, {})
            watchlist_item = WatchlistItem(
                symbol=symbol,
                name=market_data.get('name', symbol),
                current_price=market_data.get('price', 0),
                change_percent=market_data.get('change_percent', 0),
                volume=market_data.get('volume', 0),
                market_cap=market_data.get('market_cap'),
                sector=market_data.get('sector'),
                added_at=datetime.now()
            )

            watchlist.append(watchlist_item)
            self.watchlists[user_id] = watchlist

            return {'success': True, 'message': f'{symbol} 已添加到自选股'}

        elif action == 'remove':
            # 从自选股移除
            original_count = len(watchlist)
            watchlist = [item for item in watchlist if item.symbol != symbol]

            if len(watchlist) < original_count:
                self.watchlists[user_id] = watchlist
                return {'success': True, 'message': f'{symbol} 已从自选股移除'}
            else:
                return {'success': False, 'message': '股票不在自选股中'}

        elif action == 'get':
            # 获取自选股
            return {
                'success': True,
                'watchlist': [self._watchlist_to_dict(item) for item in watchlist]
            }

        return {'success': False, 'message': '无效的操作'}

    def get_portfolio_summary(self, user_id: str) -> Dict[str, Any]:
        """获取投资组合摘要"""
        if user_id not in self.users:
            return {'success': False, 'message': '用户不存在'}

        user = self.users[user_id]
        positions = self.positions.get(user_id, [])

        # 计算投资组合指标
        total_value = user.balance
        total_cost = 0
        total_unrealized_pnl = 0

        for position in positions:
            current_price = self._get_market_price(position.symbol) or position.current_price
            position.current_price = current_price
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.average_cost) * position.quantity

            total_value += position.market_value
            total_cost += position.average_cost * position.quantity
            total_unrealized_pnl += position.unrealized_pnl

        # 计算收益指标
        total_pnl = user.total_pnl + total_unrealized_pnl
        portfolio_return = total_pnl / \
            (user.balance + total_cost) if (user.balance + total_cost) > 0 else 0

        # 资产配置
        asset_allocation = {}
        for position in positions:
            allocation = (position.market_value / total_value) * 100 if total_value > 0 else 0
            asset_allocation[position.symbol] = {
                'value': position.market_value,
                'allocation': allocation,
                'quantity': position.quantity,
                'avg_cost': position.average_cost,
                'current_price': position.current_price,
                'pnl': position.unrealized_pnl
            }

        return {
            'success': True,
            'portfolio': {
                'total_value': total_value,
                'cash_balance': user.balance,
                'positions_value': total_value - user.balance,
                'total_pnl': total_pnl,
                'daily_pnl': total_unrealized_pnl,
                'portfolio_return': portfolio_return,
                'positions_count': len(positions),
                'asset_allocation': asset_allocation
            }
        }

    def _execute_order(self, order: MobileOrder) -> Dict[str, Any]:
        """执行订单"""
        try:
            # 获取市场价格
            market_price = self._get_market_price(order.symbol)

            if not market_price:
                return {'success': False, 'message': '无法获取市场价格'}

            # 计算执行价格
            if order.order_type == OrderType.MARKET:
                execution_price = market_price
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and order.price >= market_price:
                    execution_price = order.price
                elif order.side == OrderSide.SELL and order.price <= market_price:
                    execution_price = order.price
                else:
                    return {'success': False, 'message': '限价单未满足执行条件'}
            else:
                # 其他订单类型的简化为市价执行
                execution_price = market_price

            # 计算费用
            order_value = order.quantity * execution_price
            fees = order_value * self.commission_rate

            return {
                'success': True,
                'fill_price': execution_price,
                'fees': fees,
                'message': '订单执行成功'
            }

        except Exception as e:
            logger.error(f"订单执行失败: {e}")
            return {'success': False, 'message': f'执行失败: {str(e)}'}

    def _update_user_account(self, user_id: str, order: MobileOrder, execution_result: Dict[str, Any]):
        """更新用户账户"""
        user = self.users[user_id]

        order_value = order.quantity * execution_result['fill_price']
        fees = execution_result['fees']

        if order.side == OrderSide.BUY:
            user.balance -= (order_value + fees)
            user.buying_power = user.balance  # 简化的购买力计算
        else:  # SELL
            user.balance += (order_value - fees)

        logger.debug(f"用户 {user_id} 账户更新: 余额={user.balance:.2f}")

    def _update_position(self, user_id: str, order: MobileOrder, execution_result: Dict[str, Any]):
        """更新持仓"""
        positions = self.positions.get(user_id, [])
        fill_price = execution_result['fill_price']

        if order.side == OrderSide.BUY:
            # 检查是否已有该股票的持仓
            existing_position = None
            for pos in positions:
                if pos.symbol == order.symbol:
                    existing_position = pos
                    break

            if existing_position:
                # 更新现有持仓
                total_cost = (existing_position.quantity * existing_position.average_cost
                              + order.quantity * fill_price)
                existing_position.quantity += order.quantity
                existing_position.average_cost = total_cost / existing_position.quantity
                existing_position.updated_at = datetime.now()
            else:
                # 创建新持仓
                position = MobilePosition(
                    position_id=str(uuid.uuid4()),
                    user_id=user_id,
                    symbol=order.symbol,
                    position_type=PositionType.LONG,
                    quantity=order.quantity,
                    average_cost=fill_price,
                    current_price=fill_price,
                    market_value=order.quantity * fill_price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                positions.append(position)

        else:  # SELL
            # 查找现有持仓
            for pos in positions:
                if pos.symbol == order.symbol:
                    if pos.quantity >= order.quantity:
                        # 计算已实现盈亏
                        realized_pnl = (fill_price - pos.average_cost) * order.quantity
                        pos.realized_pnl += realized_pnl
                        pos.quantity -= order.quantity
                        pos.updated_at = datetime.now()

                        # 如果持仓为0，从列表中移除
                        if pos.quantity <= 0:
                            positions.remove(pos)

                        break

        self.positions[user_id] = positions

    def _get_position_quantity(self, user_id: str, symbol: str) -> float:
        """获取持仓数量"""
        positions = self.positions.get(user_id, [])
        for pos in positions:
            if pos.symbol == symbol:
                return pos.quantity
        return 0

    def _get_market_price(self, symbol: str) -> Optional[float]:
        """获取市场价格"""
        market_info = self.market_data.get(symbol, {})
        return market_info.get('price')

    def _market_data_loop(self):
        """市场数据循环"""
        while self.is_market_running:
            try:
                # 更新市场数据
                self._update_market_data()
                time.sleep(self.market_update_interval)

            except Exception as e:
                logger.error(f"市场数据更新异常: {e}")
                time.sleep(5)

    def _update_market_data(self):
        """更新市场数据"""
        # 简化的市场数据更新
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'BTC', 'ETH']

        for symbol in symbols:
            # 生成模拟价格数据
            base_price = 100 + (hash(symbol) % 900)  # 基础价格
            price_change = np.secrets.normal(0, 0.02)  # 正态分布的价格变化
            current_price = base_price * (1 + price_change)

            self.market_data[symbol] = {
                'symbol': symbol,
                'name': f'{symbol} Corp' if symbol not in ['BTC', 'ETH'] else symbol,
                'price': current_price,
                'change_percent': price_change * 100,
                'volume': np.secrets.randint(1000000, 10000000),
                'market_cap': base_price * 1000000000 if symbol not in ['BTC', 'ETH'] else None,
                'sector': 'Technology' if symbol not in ['BTC', 'ETH'] else 'Cryptocurrency',
                'updated_at': datetime.now()
            }

    def _order_to_dict(self, order: MobileOrder) -> Dict[str, Any]:
        """订单转换为字典"""
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'stop_price': order.stop_price,
            'status': order.status.value,
            'created_at': order.created_at.isoformat(),
            'updated_at': order.updated_at.isoformat(),
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'average_fill_price': order.average_fill_price,
            'fees': order.fees,
            'strategy': order.strategy,
            'notes': order.notes
        }

    def _position_to_dict(self, position: MobilePosition) -> Dict[str, Any]:
        """持仓转换为字典"""
        return {
            'position_id': position.position_id,
            'symbol': position.symbol,
            'position_type': position.position_type.value,
            'quantity': position.quantity,
            'average_cost': position.average_cost,
            'current_price': position.current_price,
            'market_value': position.market_value,
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'total_pnl': position.unrealized_pnl + position.realized_pnl,
            'created_at': position.created_at.isoformat(),
            'updated_at': position.updated_at.isoformat()
        }

    def _watchlist_to_dict(self, item: WatchlistItem) -> Dict[str, Any]:
        """自选股项目转换为字典"""
        return {
            'symbol': item.symbol,
            'name': item.name,
            'current_price': item.current_price,
            'change_percent': item.change_percent,
            'volume': item.volume,
            'market_cap': item.market_cap,
            'sector': item.sector,
            'added_at': item.added_at.isoformat() if item.added_at else None
        }

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """用户认证"""
        # 简化的认证实现
        for user_id, user in self.users.items():
            if user.username == username:
                # 在实际实现中，这里应该验证密码哈希
                user.last_login = datetime.now()
                return user_id

        return None

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户资料"""
        if user_id not in self.users:
            return None

        user = self.users[user_id]
        positions = self.positions.get(user_id, [])
        orders = [order for order in self.orders.values() if order.user_id == user_id]

        # 计算总资产
        total_portfolio_value = sum(pos.market_value for pos in positions)
        total_value = user.balance + total_portfolio_value

        # 计算今日盈亏
        daily_pnl = sum(pos.unrealized_pnl for pos in positions)

        return {
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'balance': user.balance,
            'buying_power': user.buying_power,
            'total_value': total_value,
            'daily_pnl': daily_pnl,
            'total_pnl': user.total_pnl + daily_pnl,
            'positions_count': len(positions),
            'orders_count': len(orders),
            'risk_score': user.risk_score,
            'last_login': user.last_login.isoformat()
        }

    def place_order(self, user_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """下单"""
        if user_id not in self.users:
            return {'success': False, 'message': '用户不存在'}

        try:
            # 验证订单参数
            symbol = order_data.get('symbol', '').upper()
            side = OrderSide(order_data.get('side', 'buy'))
            order_type = OrderType(order_data.get('order_type', 'market'))
            quantity = float(order_data.get('quantity', 0))
            price = order_data.get('price')
            stop_price = order_data.get('stop_price')
            strategy = order_data.get('strategy')
            notes = order_data.get('notes')

            # 基本验证
            if not symbol or quantity <= 0:
                return {'success': False, 'message': '无效的订单参数'}

            if quantity < self.min_order_size or quantity > self.max_order_size:
                return {'success': False, 'message': f'订单数量必须在 {self.min_order_size} 到 {self.max_order_size} 之间'}

            # 验证价格参数
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if not price or price <= 0:
                    return {'success': False, 'message': '限价单必须指定有效价格'}

            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
                if not stop_price or stop_price <= 0:
                    return {'success': False, 'message': '止损单必须指定止损价格'}

            # 检查购买力
            user = self.users[user_id]
            market_price = self._get_market_price(symbol)

            if not market_price:
                return {'success': False, 'message': f'无法获取 {symbol} 的市场价格'}

            order_value = quantity * (price or market_price)
            estimated_fees = order_value * self.commission_rate

            if side == OrderSide.BUY and order_value + estimated_fees > user.buying_power:
                return {'success': False, 'message': '购买力不足'}

            # 检查持仓（卖出时）
            if side == OrderSide.SELL:
                current_position = self._get_position_quantity(user_id, symbol)
                if current_position < quantity:
                    return {'success': False, 'message': '持仓不足'}

            # 创建订单
            order_id = str(uuid.uuid4())

            order = MobileOrder(
                order_id=order_id,
                user_id=user_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force="GTC",
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                remaining_quantity=quantity,
                strategy=strategy,
                notes=notes
            )

            self.orders[order_id] = order

            # 尝试执行订单
            execution_result = self._execute_order(order)

            if execution_result['success']:
                order.status = OrderStatus.FILLED
                order.filled_quantity = quantity
                order.remaining_quantity = 0
                order.average_fill_price = execution_result['fill_price']
                order.fees = execution_result['fees']
                order.updated_at = datetime.now()

                # 更新用户账户
                self._update_user_account(user_id, order, execution_result)

                # 更新持仓
                self._update_position(user_id, order, execution_result)

            else:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()

            logger.info(
                f"用户 {user_id} 下单: {symbol} {side.value} {quantity}@{order.average_fill_price or price or market_price}")

            return {
                'success': execution_result['success'],
                'order_id': order_id,
                'message': execution_result.get('message', '订单已提交'),
                'fill_price': order.average_fill_price,
                'fees': order.fees
            }

        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {'success': False, 'message': f'下单失败: {str(e)}'}

    def cancel_order(self, user_id: str, order_id: str) -> Dict[str, Any]:
        """取消订单"""
        if order_id not in self.orders:
            return {'success': False, 'message': '订单不存在'}

        order = self.orders[order_id]

        if order.user_id != user_id:
            return {'success': False, 'message': '无权取消此订单'}

        if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED]:
            return {'success': False, 'message': '订单无法取消'}

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()

        logger.info(f"用户 {user_id} 取消订单: {order_id}")

        return {'success': True, 'message': '订单已取消'}

    def get_orders(self, user_id: str, status: Optional[OrderStatus] = None) -> List[Dict[str, Any]]:
        """获取用户订单"""
        user_orders = [order for order in self.orders.values() if order.user_id == user_id]

        if status:
            user_orders = [order for order in user_orders if order.status == status]

        # 按创建时间倒序排列
        user_orders.sort(key=lambda x: x.created_at, reverse=True)

        return [self._order_to_dict(order) for order in user_orders]

    def get_positions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户持仓"""
        user_positions = self.positions.get(user_id, [])

        # 更新当前价格和盈亏
        for position in user_positions:
            current_price = self._get_market_price(position.symbol)
            if current_price:
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (
                    current_price - position.average_cost) * position.quantity
                position.updated_at = datetime.now()

        return [self._position_to_dict(pos) for pos in user_positions]

    def manage_watchlist(self, user_id: str, action: str, symbol: str) -> Dict[str, Any]:
        """管理自选股"""
        if user_id not in self.users:
            return {'success': False, 'message': '用户不存在'}

        watchlist = self.watchlists.get(user_id, [])

        if action == 'add':
            # 检查是否已存在
            if any(item.symbol == symbol for item in watchlist):
                return {'success': False, 'message': '股票已在自选股中'}

            # 添加到自选股
            market_data = self.market_data.get(symbol, {})
            watchlist_item = WatchlistItem(
                symbol=symbol,
                name=market_data.get('name', symbol),
                current_price=market_data.get('price', 0),
                change_percent=market_data.get('change_percent', 0),
                volume=market_data.get('volume', 0),
                market_cap=market_data.get('market_cap'),
                sector=market_data.get('sector'),
                added_at=datetime.now()
            )

            watchlist.append(watchlist_item)
            self.watchlists[user_id] = watchlist

            return {'success': True, 'message': f'{symbol} 已添加到自选股'}

        elif action == 'remove':
            # 从自选股移除
            original_count = len(watchlist)
            watchlist = [item for item in watchlist if item.symbol != symbol]

            if len(watchlist) < original_count:
                self.watchlists[user_id] = watchlist
                return {'success': True, 'message': f'{symbol} 已从自选股移除'}
            else:
                return {'success': False, 'message': '股票不在自选股中'}

        elif action == 'get':
            # 获取自选股
            return {
                'success': True,
                'watchlist': [self._watchlist_to_dict(item) for item in watchlist]
            }

        return {'success': False, 'message': '无效的操作'}

    def get_portfolio_summary(self, user_id: str) -> Dict[str, Any]:
        """获取投资组合摘要"""
        if user_id not in self.users:
            return {'success': False, 'message': '用户不存在'}

        user = self.users[user_id]
        positions = self.positions.get(user_id, [])

        # 计算投资组合指标
        total_value = user.balance
        total_cost = 0
        total_unrealized_pnl = 0

        for position in positions:
            current_price = self._get_market_price(position.symbol) or position.current_price
            position.current_price = current_price
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.average_cost) * position.quantity

            total_value += position.market_value
            total_cost += position.average_cost * position.quantity
            total_unrealized_pnl += position.unrealized_pnl

        # 计算收益指标
        total_pnl = user.total_pnl + total_unrealized_pnl
        portfolio_return = total_pnl / \
            (user.balance + total_cost) if (user.balance + total_cost) > 0 else 0

        # 资产配置
        asset_allocation = {}
        for position in positions:
            allocation = (position.market_value / total_value) * 100 if total_value > 0 else 0
            asset_allocation[position.symbol] = {
                'value': position.market_value,
                'allocation': allocation,
                'quantity': position.quantity,
                'avg_cost': position.average_cost,
                'current_price': position.current_price,
                'pnl': position.unrealized_pnl
            }

        return {
            'success': True,
            'portfolio': {
                'total_value': total_value,
                'cash_balance': user.balance,
                'positions_value': total_value - user.balance,
                'total_pnl': total_pnl,
                'daily_pnl': total_unrealized_pnl,
                'portfolio_return': portfolio_return,
                'positions_count': len(positions),
                'asset_allocation': asset_allocation
            }
        }

# 移动端交易演示


def demo_mobile_trading():
    """移动端交易演示"""
    service = MobileTradingService()
    service.start_market_data_service()

    # 创建用户
    user_id = service.create_user("demo_user", "demo@example.com", 10000)

    # 模拟交易
    orders = [
        {'symbol': 'AAPL', 'side': 'buy', 'order_type': 'market', 'quantity': 10},
        {'symbol': 'GOOGL', 'side': 'buy', 'order_type': 'market', 'quantity': 5},
        {'symbol': 'AAPL', 'side': 'sell', 'order_type': 'market', 'quantity': 5},
    ]

    for order_data in orders:
        result = service.place_order(user_id, order_data)
        print(f"订单结果: {result}")

    # 获取投资组合摘要
    portfolio = service.get_portfolio_summary(user_id)
    print(f"投资组合: {portfolio}")

    # 停止服务
    service.stop_market_data_service()


if __name__ == "__main__":
    demo_mobile_trading()

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """用户认证"""
        # 简化的认证实现
        for user_id, user in self.users.items():
            if user.username == username:
                # 在实际实现中，这里应该验证密码哈希
                user.last_login = datetime.now()
                return user_id

        return None

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户资料"""
        if user_id not in self.users:
            return None

        user = self.users[user_id]
        positions = self.positions.get(user_id, [])
        orders = [order for order in self.orders.values() if order.user_id == user_id]

        # 计算总资产
        total_portfolio_value = sum(pos.market_value for pos in positions)
        total_value = user.balance + total_portfolio_value

        # 计算今日盈亏
        daily_pnl = sum(pos.unrealized_pnl for pos in positions)

        return {
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'balance': user.balance,
            'buying_power': user.buying_power,
            'total_value': total_value,
            'daily_pnl': daily_pnl,
            'total_pnl': user.total_pnl + daily_pnl,
            'positions_count': len(positions),
            'orders_count': len(orders),
            'risk_score': user.risk_score,
            'last_login': user.last_login.isoformat()
        }

    def place_order(self, user_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """下单"""
        if user_id not in self.users:
            return {'success': False, 'message': '用户不存在'}

        try:
            # 验证订单参数
            symbol = order_data.get('symbol', '').upper()
            side = OrderSide(order_data.get('side', 'buy'))
            order_type = OrderType(order_data.get('order_type', 'market'))
            quantity = float(order_data.get('quantity', 0))
            price = order_data.get('price')
            stop_price = order_data.get('stop_price')
            strategy = order_data.get('strategy')
            notes = order_data.get('notes')

            # 基本验证
            if not symbol or quantity <= 0:
                return {'success': False, 'message': '无效的订单参数'}

            if quantity < self.min_order_size or quantity > self.max_order_size:
                return {'success': False, 'message': f'订单数量必须在 {self.min_order_size} 到 {self.max_order_size} 之间'}

            # 验证价格参数
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if not price or price <= 0:
                    return {'success': False, 'message': '限价单必须指定有效价格'}

            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
                if not stop_price or stop_price <= 0:
                    return {'success': False, 'message': '止损单必须指定止损价格'}

            # 检查购买力
            user = self.users[user_id]
            market_price = self._get_market_price(symbol)

            if not market_price:
                return {'success': False, 'message': f'无法获取 {symbol} 的市场价格'}

            order_value = quantity * (price or market_price)
            estimated_fees = order_value * self.commission_rate

            if side == OrderSide.BUY and order_value + estimated_fees > user.buying_power:
                return {'success': False, 'message': '购买力不足'}

            # 检查持仓（卖出时）
            if side == OrderSide.SELL:
                current_position = self._get_position_quantity(user_id, symbol)
                if current_position < quantity:
                    return {'success': False, 'message': '持仓不足'}

            # 创建订单
            order_id = str(uuid.uuid4())

            order = MobileOrder(
                order_id=order_id,
                user_id=user_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force="GTC",
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                remaining_quantity=quantity,
                strategy=strategy,
                notes=notes
            )

            self.orders[order_id] = order

            # 尝试执行订单
            execution_result = self._execute_order(order)

            if execution_result['success']:
                order.status = OrderStatus.FILLED
                order.filled_quantity = quantity
                order.remaining_quantity = 0
                order.average_fill_price = execution_result['fill_price']
                order.fees = execution_result['fees']
                order.updated_at = datetime.now()

                # 更新用户账户
                self._update_user_account(user_id, order, execution_result)

                # 更新持仓
                self._update_position(user_id, order, execution_result)

            else:
                order.status = OrderStatus.REJECTED
                order.updated_at = datetime.now()

            logger.info(
                f"用户 {user_id} 下单: {symbol} {side.value} {quantity}@{order.average_fill_price or price or market_price}")

            return {
                'success': execution_result['success'],
                'order_id': order_id,
                'message': execution_result.get('message', '订单已提交'),
                'fill_price': order.average_fill_price,
                'fees': order.fees
            }

        except Exception as e:
            logger.error(f"下单失败: {e}")
            return {'success': False, 'message': f'下单失败: {str(e)}'}

    def cancel_order(self, user_id: str, order_id: str) -> Dict[str, Any]:
        """取消订单"""
        if order_id not in self.orders:
            return {'success': False, 'message': '订单不存在'}

        order = self.orders[order_id]

        if order.user_id != user_id:
            return {'success': False, 'message': '无权取消此订单'}

        if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIAL_FILLED]:
            return {'success': False, 'message': '订单无法取消'}

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()

        logger.info(f"用户 {user_id} 取消订单: {order_id}")

        return {'success': True, 'message': '订单已取消'}

    def get_orders(self, user_id: str, status: Optional[OrderStatus] = None) -> List[Dict[str, Any]]:
        """获取用户订单"""
        user_orders = [order for order in self.orders.values() if order.user_id == user_id]

        if status:
            user_orders = [order for order in user_orders if order.status == status]

        # 按创建时间倒序排列
        user_orders.sort(key=lambda x: x.created_at, reverse=True)

        return [self._order_to_dict(order) for order in user_orders]

    def get_positions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户持仓"""
        user_positions = self.positions.get(user_id, [])

        # 更新当前价格和盈亏
        for position in user_positions:
            current_price = self._get_market_price(position.symbol)
            if current_price:
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (
                    current_price - position.average_cost) * position.quantity
                position.updated_at = datetime.now()

        return [self._position_to_dict(pos) for pos in user_positions]

    def manage_watchlist(self, user_id: str, action: str, symbol: str) -> Dict[str, Any]:
        """管理自选股"""
        if user_id not in self.users:
            return {'success': False, 'message': '用户不存在'}

        watchlist = self.watchlists.get(user_id, [])

        if action == 'add':
            # 检查是否已存在
            if any(item.symbol == symbol for item in watchlist):
                return {'success': False, 'message': '股票已在自选股中'}

            # 添加到自选股
            market_data = self.market_data.get(symbol, {})
            watchlist_item = WatchlistItem(
                symbol=symbol,
                name=market_data.get('name', symbol),
                current_price=market_data.get('price', 0),
                change_percent=market_data.get('change_percent', 0),
                volume=market_data.get('volume', 0),
                market_cap=market_data.get('market_cap'),
                sector=market_data.get('sector'),
                added_at=datetime.now()
            )

            watchlist.append(watchlist_item)
            self.watchlists[user_id] = watchlist

            return {'success': True, 'message': f'{symbol} 已添加到自选股'}

        elif action == 'remove':
            # 从自选股移除
            original_count = len(watchlist)
            watchlist = [item for item in watchlist if item.symbol != symbol]

            if len(watchlist) < original_count:
                self.watchlists[user_id] = watchlist
                return {'success': True, 'message': f'{symbol} 已从自选股移除'}
            else:
                return {'success': False, 'message': '股票不在自选股中'}

        elif action == 'get':
            # 获取自选股
            return {
                'success': True,
                'watchlist': [self._watchlist_to_dict(item) for item in watchlist]
            }

        return {'success': False, 'message': '无效的操作'}

    def get_portfolio_summary(self, user_id: str) -> Dict[str, Any]:
        """获取投资组合摘要"""
        if user_id not in self.users:
            return {'success': False, 'message': '用户不存在'}

        user = self.users[user_id]
        positions = self.positions.get(user_id, [])

        # 计算投资组合指标
        total_value = user.balance
        total_cost = 0
        total_unrealized_pnl = 0

        for position in positions:
            current_price = self._get_market_price(position.symbol) or position.current_price
            position.current_price = current_price
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.average_cost) * position.quantity

            total_value += position.market_value
            total_cost += position.average_cost * position.quantity
            total_unrealized_pnl += position.unrealized_pnl

        # 计算收益指标
        total_pnl = user.total_pnl + total_unrealized_pnl
        portfolio_return = total_pnl / \
            (user.balance + total_cost) if (user.balance + total_cost) > 0 else 0

        # 资产配置
        asset_allocation = {}
        for position in positions:
            allocation = (position.market_value / total_value) * 100 if total_value > 0 else 0
            asset_allocation[position.symbol] = {
                'value': position.market_value,
                'allocation': allocation,
                'quantity': position.quantity,
                'avg_cost': position.average_cost,
                'current_price': position.current_price,
                'pnl': position.unrealized_pnl
            }

        return {
            'success': True,
            'portfolio': {
                'total_value': total_value,
                'cash_balance': user.balance,
                'positions_value': total_value - user.balance,
                'total_pnl': total_pnl,
                'daily_pnl': total_unrealized_pnl,
                'portfolio_return': portfolio_return,
                'positions_count': len(positions),
                'asset_allocation': asset_allocation
            }
        }

    def _execute_order(self, order: MobileOrder) -> Dict[str, Any]:
        """执行订单"""
        try:
            # 获取市场价格
            market_price = self._get_market_price(order.symbol)

            if not market_price:
                return {'success': False, 'message': '无法获取市场价格'}

            # 计算执行价格
            if order.order_type == OrderType.MARKET:
                execution_price = market_price
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and order.price >= market_price:
                    execution_price = order.price
                elif order.side == OrderSide.SELL and order.price <= market_price:
                    execution_price = order.price
                else:
                    return {'success': False, 'message': '限价单未满足执行条件'}
            else:
                # 其他订单类型的简化为市价执行
                execution_price = market_price

            # 计算费用
            order_value = order.quantity * execution_price
            fees = order_value * self.commission_rate

            return {
                'success': True,
                'fill_price': execution_price,
                'fees': fees,
                'message': '订单执行成功'
            }

        except Exception as e:
            logger.error(f"订单执行失败: {e}")
            return {'success': False, 'message': f'执行失败: {str(e)}'}

    def _update_user_account(self, user_id: str, order: MobileOrder, execution_result: Dict[str, Any]):
        """更新用户账户"""
        user = self.users[user_id]

        order_value = order.quantity * execution_result['fill_price']
        fees = execution_result['fees']

        if order.side == OrderSide.BUY:
            user.balance -= (order_value + fees)
            user.buying_power = user.balance  # 简化的购买力计算
        else:  # SELL
            user.balance += (order_value - fees)

        logger.debug(f"用户 {user_id} 账户更新: 余额={user.balance:.2f}")

    def _update_position(self, user_id: str, order: MobileOrder, execution_result: Dict[str, Any]):
        """更新持仓"""
        positions = self.positions.get(user_id, [])
        fill_price = execution_result['fill_price']

        if order.side == OrderSide.BUY:
            # 检查是否已有该股票的持仓
            existing_position = None
            for pos in positions:
                if pos.symbol == order.symbol:
                    existing_position = pos
                    break

            if existing_position:
                # 更新现有持仓
                total_cost = (existing_position.quantity * existing_position.average_cost
                              + order.quantity * fill_price)
                existing_position.quantity += order.quantity
                existing_position.average_cost = total_cost / existing_position.quantity
                existing_position.updated_at = datetime.now()
            else:
                # 创建新持仓
                position = MobilePosition(
                    position_id=str(uuid.uuid4()),
                    user_id=user_id,
                    symbol=order.symbol,
                    position_type=PositionType.LONG,
                    quantity=order.quantity,
                    average_cost=fill_price,
                    current_price=fill_price,
                    market_value=order.quantity * fill_price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                positions.append(position)

        else:  # SELL
            # 查找现有持仓
            for pos in positions:
                if pos.symbol == order.symbol:
                    if pos.quantity >= order.quantity:
                        # 计算已实现盈亏
                        realized_pnl = (fill_price - pos.average_cost) * order.quantity
                        pos.realized_pnl += realized_pnl
                        pos.quantity -= order.quantity
                        pos.updated_at = datetime.now()

                        # 如果持仓为0，从列表中移除
                        if pos.quantity <= 0:
                            positions.remove(pos)

                        break

        self.positions[user_id] = positions

    def _get_position_quantity(self, user_id: str, symbol: str) -> float:
        """获取持仓数量"""
        positions = self.positions.get(user_id, [])
        for pos in positions:
            if pos.symbol == symbol:
                return pos.quantity
        return 0

    def _get_market_price(self, symbol: str) -> Optional[float]:
        """获取市场价格"""
        market_info = self.market_data.get(symbol, {})
        return market_info.get('price')

    def _market_data_loop(self):
        """市场数据循环"""
        while self.is_market_running:
            try:
                # 更新市场数据
                self._update_market_data()
                time.sleep(self.market_update_interval)

            except Exception as e:
                logger.error(f"市场数据更新异常: {e}")
                time.sleep(5)

    def _update_market_data(self):
        """更新市场数据"""
        # 简化的市场数据更新
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'BTC', 'ETH']

        for symbol in symbols:
            # 生成模拟价格数据
            base_price = 100 + (hash(symbol) % 900)  # 基础价格
            price_change = np.secrets.normal(0, 0.02)  # 正态分布的价格变化
            current_price = base_price * (1 + price_change)

            self.market_data[symbol] = {
                'symbol': symbol,
                'name': f'{symbol} Corp' if symbol not in ['BTC', 'ETH'] else symbol,
                'price': current_price,
                'change_percent': price_change * 100,
                'volume': np.secrets.randint(1000000, 10000000),
                'market_cap': base_price * 1000000000 if symbol not in ['BTC', 'ETH'] else None,
                'sector': 'Technology' if symbol not in ['BTC', 'ETH'] else 'Cryptocurrency',
                'updated_at': datetime.now()
            }

    def _order_to_dict(self, order: MobileOrder) -> Dict[str, Any]:
        """订单转换为字典"""
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'stop_price': order.stop_price,
            'status': order.status.value,
            'created_at': order.created_at.isoformat(),
            'updated_at': order.updated_at.isoformat(),
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'average_fill_price': order.average_fill_price,
            'fees': order.fees,
            'strategy': order.strategy,
            'notes': order.notes
        }

    def _position_to_dict(self, position: MobilePosition) -> Dict[str, Any]:
        """持仓转换为字典"""
        return {
            'position_id': position.position_id,
            'symbol': position.symbol,
            'position_type': position.position_type.value,
            'quantity': position.quantity,
            'average_cost': position.average_cost,
            'current_price': position.current_price,
            'market_value': position.market_value,
            'unrealized_pnl': position.unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'total_pnl': position.unrealized_pnl + position.realized_pnl,
            'created_at': position.created_at.isoformat(),
            'updated_at': position.updated_at.isoformat()
        }

    def _watchlist_to_dict(self, item: WatchlistItem) -> Dict[str, Any]:
        """自选股项目转换为字典"""
        return {
            'symbol': item.symbol,
            'name': item.name,
            'current_price': item.current_price,
            'change_percent': item.change_percent,
            'volume': item.volume,
            'market_cap': item.market_cap,
            'sector': item.sector,
            'added_at': item.added_at.isoformat() if item.added_at else None
        }

