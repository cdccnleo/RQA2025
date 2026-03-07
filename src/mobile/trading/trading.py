import logging
from typing import Dict, List, Any, Optional
from mobile.mobile_trading import *


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
class MobileOrder:

    """移动端订单"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    price: Optional[float]
    timestamp: datetime
    status: str

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

        def trade(symbol):
            """交易页面"""
            return render_template('trade.html', symbol=symbol)

        def orders():
            """订单页面"""
            return render_template('orders.html')

        def api_place_order():
            """下单API"""
            data = request.get_json()

            if not data:
                return jsonify({'success': False, 'message': 'No data provided'})

            required_fields = ['symbol', 'side', 'quantity', 'order_type']
            for field in required_fields:
                if field not in data:
                    return jsonify({'success': False, 'message': f'Missing {field}'})

            symbol = data['symbol'].upper()
            side = data['side'].lower()
            quantity = float(data['quantity'])
            order_type = data['order_type'].lower()
            price = data.get('price')

            # 验证订单
            if side not in ['buy', 'sell']:
                return jsonify({'success': False, 'message': 'Invalid side'})

            if order_type not in ['market', 'limit']:
                return jsonify({'success': False, 'message': 'Invalid order type'})

            if side == 'buy':
                cost = quantity * (price or 100)  # 简化的价格计算
                if cost > self.user_balance:
                    return jsonify({'success': False, 'message': 'Insufficient balance'})

                self.user_balance -= cost

                # 更新投资组合
                if symbol in self.portfolio:
                    item = self.portfolio[symbol]
                    total_shares = item.shares + quantity
                    total_cost = (item.shares * item.avg_cost) + cost
                    new_avg_cost = total_cost / total_shares

                    item.shares = total_shares
                    item.avg_cost = new_avg_cost
                else:
                    self.portfolio[symbol] = PortfolioItem(
                        symbol=symbol,
                        name=f"{symbol} Corp",
                        shares=quantity,
                        avg_cost=price or 100,
                        current_price=price or 100,
                        market_value=quantity * (price or 100),
                        gain_loss=0,
                        gain_loss_percent=0
                    )

            elif side == 'sell':
                if symbol not in self.portfolio or self.portfolio[symbol].shares < quantity:
                    return jsonify({'success': False, 'message': 'Insufficient shares'})

                item = self.portfolio[symbol]
                sell_value = quantity * (price or item.current_price)
                self.user_balance += sell_value

                # 计算收益
                cost_basis = quantity * item.avg_cost
                gain_loss = sell_value - cost_basis

                # 更新持仓
                item.shares -= quantity
                item.gain_loss += gain_loss
                item.gain_loss_percent = item.gain_loss / \
                    (item.shares * item.avg_cost) * 100 if item.shares > 0 else 0

                if item.shares <= 0:
                    del self.portfolio[symbol]

            # 创建订单记录
            order = MobileOrder(
                order_id=f"ORD_{datetime.now().strftime('%Y % m % d % H % M % S % f')}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                timestamp=datetime.now(),
                status='filled'
            )

            self.orders.append(order)

            logger.info(f"订单执行: {order.order_id} - {symbol} {side} {quantity}@{price}")

            return jsonify({
                'success': True,
                'order_id': order.order_id,
                'message': 'Order placed successfully'
            })

        def api_orders_list():
            """订单列表API"""
            orders = []
            for order in self.orders[-20:]:  # 返回最近20个订单
                orders.append({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': order.quantity,
                    'order_type': order.order_type,
                    'price': order.price,
                    'timestamp': order.timestamp.isoformat(),
                    'status': order.status
