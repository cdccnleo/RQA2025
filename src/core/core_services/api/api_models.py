"""
API数据模型

包含API请求和响应的Pydantic模型定义。
"""

from typing import Dict, Any
from pydantic import BaseModel, Field, validator


# 请求模型
class UserCreateRequest(BaseModel):
    """用户创建请求"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
    password: str = Field(..., min_length=6, max_length=100)
    initial_balance: float = Field(default=10000.0, ge=0)


class UserLoginRequest(BaseModel):
    """用户登录请求"""
    username: str
    password: str


class OrderCreateRequest(BaseModel):
    """订单创建请求"""
    symbol: str = Field(..., min_length=1, max_length=10)
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)
    order_type: str = Field(..., pattern=r'^(market|limit)$')
    side: str = Field(..., pattern=r'^(buy|sell)$')

    @validator('symbol')
    def validate_symbol(cls, v):
        # 简单的A股代码验证
        if not v.replace('.', '').replace('SZ', '').replace('SH', '').isdigit():
            raise ValueError('无效的股票代码')
        return v.upper()


# 响应模型
class TokenResponse(BaseModel):
    """令牌响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class OrderResponse(BaseModel):
    """订单响应"""
    order_id: str
    status: str
    details: Dict[str, Any]


class PositionResponse(BaseModel):
    """持仓响应"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float


class APIResponse(BaseModel):
    """通用API响应"""
    success: bool
    message: str
    data: Any = None
    timestamp: str = Field(default_factory=lambda: str(time.time()))

