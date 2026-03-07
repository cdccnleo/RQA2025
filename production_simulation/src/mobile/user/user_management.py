from typing import Dict, Any, Optional
from mobile.mobile_trading import *


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

    def create_user(
        self, username: str, email: str, initial_balance: float = 10000
    ) -> str:
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
            last_login=datetime.now(),
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
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "balance": user.balance,
            "buying_power": user.buying_power,
            "total_value": total_value,
            "daily_pnl": daily_pnl,
            "total_pnl": user.total_pnl + daily_pnl,
            "positions_count": len(positions),
            "orders_count": len(orders),
            "risk_score": user.risk_score,
            "last_login": user.last_login.isoformat(),
        }

    def _update_user_account(
        self, user_id: str, order: MobileOrder, execution_result: Dict[str, Any]
    ):
        """更新用户账户"""
        user = self.users[user_id]

        order_value = order.quantity * execution_result["fill_price"]
        fees = execution_result["fees"]

        if order.side == OrderSide.BUY:
            user.balance -= order_value + fees
            user.buying_power = user.balance  # 简化的购买力计算
        else:  # SELL
            user.balance += order_value - fees

        logger.debug(f"用户 {user_id} 账户更新: 余额={user.balance:.2f}")

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
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "balance": user.balance,
            "buying_power": user.buying_power,
            "total_value": total_value,
            "daily_pnl": daily_pnl,
            "total_pnl": user.total_pnl + daily_pnl,
            "positions_count": len(positions),
            "orders_count": len(orders),
            "risk_score": user.risk_score,
            "last_login": user.last_login.isoformat(),
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
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "balance": user.balance,
            "buying_power": user.buying_power,
            "total_value": total_value,
            "daily_pnl": daily_pnl,
            "total_pnl": user.total_pnl + daily_pnl,
            "positions_count": len(positions),
            "orders_count": len(orders),
            "risk_score": user.risk_score,
            "last_login": user.last_login.isoformat(),
        }

    def _update_user_account(
        self, user_id: str, order: MobileOrder, execution_result: Dict[str, Any]
    ):
        """更新用户账户"""
        user = self.users[user_id]

        order_value = order.quantity * execution_result["fill_price"]
        fees = execution_result["fees"]

        if order.side == OrderSide.BUY:
            user.balance -= order_value + fees
            user.buying_power = user.balance  # 简化的购买力计算
        else:  # SELL
            user.balance += order_value - fees

        logger.debug(f"用户 {user_id} 账户更新: 余额={user.balance:.2f}")


@dataclass
