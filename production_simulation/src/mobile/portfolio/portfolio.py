import logging
from typing import Dict, List, Any, Optional
from mobile.mobile_trading import *


class PositionType(Enum):

    """持仓类型"""
    LONG = "long"
    SHORT = "short"


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
class PortfolioItem:

    """投资组合项目"""
    symbol: str
    name: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    gain_loss: float
    gain_loss_percent: float

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

        def portfolio():
            """投资组合页面"""
            return render_template('portfolio.html')

        def api_portfolio_summary():
            """投资组合摘要API"""
            total_value = self.user_balance
            total_gain_loss = 0

            for item in self.portfolio.values():
                total_value += item.market_value
                total_gain_loss += item.gain_loss

            return jsonify({
                'total_value': total_value,
                'available_balance': self.user_balance,
                'total_gain_loss': total_gain_loss,
                'total_gain_loss_percent': total_gain_loss / (total_value - total_gain_loss) * 100 if (total_value - total_gain_loss) > 0 else 0,
                'items': len(self.portfolio)
            })

        def api_portfolio_items():
            """投资组合项目API"""
            items = []
            for symbol, item in self.portfolio.items():
                items.append({
                    'symbol': symbol,
                    'name': item.name,
                    'shares': item.shares,
                    'avg_cost': item.avg_cost,
                    'current_price': item.current_price,
                    'market_value': item.market_value,
                    'gain_loss': item.gain_loss,
                    'gain_loss_percent': item.gain_loss_percent
                })
