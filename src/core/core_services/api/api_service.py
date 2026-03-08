#!/usr/bin/env python3
"""
RQA2025核心API服务

提供完整的REST API接口，包括用户管理、交易、持仓查询等功能。
基于FastAPI实现，支持异步操作和自动文档生成。
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import jwt
import bcrypt
from functools import wraps
import secrets

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field, validator

# 创建全局速率限制器
limiter = Limiter(key_func=get_remote_address, config_filename="nonexistent.env")

from src.core.core_services.core.database_service import get_database_service, DatabaseService
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
from src.trading.order_manager import OrderManager
from src.trading.execution_engine import ExecutionEngine
# Strategy service import (optional)
try:
    from src.strategy.core.strategy_service import UnifiedStrategyService as StrategyService
    STRATEGY_AVAILABLE = True
except ImportError:
    StrategyService = None
    STRATEGY_AVAILABLE = False

logger = get_logger(__name__)


# Pydantic模型定义
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
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class AuthService:
    """认证服务"""

    def __init__(self):
        config_manager = UnifiedConfigManager()
        # 修复：使用get而不是get_config
        self.secret_key = config_manager.get("auth.secret_key", secrets.token_hex(32))
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30

    def hash_password(self, password: str) -> str:
        """使用bcrypt哈希密码"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """创建访问令牌"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
        """获取当前用户"""
        token = credentials.credentials
        payload = self.verify_token(token)

        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效或过期的令牌",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="令牌中缺少用户信息"
            )

        # 从数据库获取完整的用户信息（包括角色）
        # 注意：这里应该在异步上下文中调用，或者使用同步方法
        user_info = None
        if api_service and hasattr(api_service, 'db_service') and hasattr(api_service.db_service, 'get_user_sync'):
            user_info = api_service.db_service.get_user_sync(user_id)

        return {
            "user_id": user_id,
            "username": payload.get("username"),
            "role": user_info.get("role", "user") if user_info else "user",  # 默认角色
            "permissions": user_info.get("permissions", []) if user_info else []
        }

        # 如果无法获取数据库信息，返回基本信息
        return {"user_id": user_id, "username": payload.get("username"), "role": "user"}

    def check_permission(self, user: Dict[str, Any], required_role: str = None,
                         required_permissions: List[str] = None) -> bool:
        """检查用户权限"""
        if not user:
            return False

        # 检查角色权限
        user_role = user.get("role", "user")
        if required_role:
            role_hierarchy = {
                "admin": 3,
                "manager": 2,
                "user": 1,
                "guest": 0
            }

            user_level = role_hierarchy.get(user_role, 0)
            required_level = role_hierarchy.get(required_role, 999)

            if user_level < required_level:
                return False

        # 检查具体权限
        if required_permissions:
            user_permissions = set(user.get("permissions", []))
            required_perms = set(required_permissions)

            if not required_perms.issubset(user_permissions):
                return False

        return True

    def require_role(self, role: str):
        """角色要求装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 从kwargs中获取current_user参数
                current_user = kwargs.get('current_user')
                if not current_user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="用户未认证"
                    )

                if not self.check_permission(current_user, required_role=role):
                    logger.warning(f"SECURITY: Access denied for user {current_user.get('user_id')} "
                                   f"role {current_user.get('role')} accessing {func.__name__} "
                                   f"(requires role: {role})")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"需要 {role} 角色权限"
                    )

                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def require_permission(self, permissions: List[str]):
        """权限要求装饰器"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                current_user = kwargs.get('current_user')
                if not current_user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="用户未认证"
                    )

                if not self.check_permission(current_user, required_permissions=permissions):
                    logger.warning(f"SECURITY: Access denied for user {current_user.get('user_id')} "
                                   f"permissions {current_user.get('permissions', [])} accessing {func.__name__} "
                                   f"(requires permissions: {permissions})")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"缺少必要权限: {permissions}"
                    )

                return await func(*args, **kwargs)
            return wrapper
        return decorator


class TradingAPIService:
    """交易API服务"""

    def __init__(self, db_service: DatabaseService = None):
        self.db_service = db_service
        self.auth_service = AuthService()
        # 初始化速率限制器
        self.limiter = Limiter(key_func=get_remote_address)

        # 初始化交易组件（如果存在的话）
        try:
            self.order_manager = OrderManager()
        except Exception as e:
            logger.warning(f"订单管理器初始化失败: {e}")
            self.order_manager = None

        try:
            self.execution_engine = ExecutionEngine()
        except Exception as e:
            logger.warning(f"执行引擎初始化失败: {e}")
            self.execution_engine = None

        try:
            if STRATEGY_AVAILABLE:
                self.strategy_service = StrategyService()
            else:
                self.strategy_service = None
        except Exception as e:
            logger.warning(f"策略服务初始化失败: {e}")
            self.strategy_service = None

    async def initialize(self):
        """初始化API服务"""
        try:
            # 如果没有数据库服务，尝试获取
            if not self.db_service:
                from src.core.core_services.core.database_service import get_database_service
                self.db_service = await get_database_service()
            logger.info("API服务初始化成功")
        except Exception as e:
            logger.error(f"API服务初始化失败: {e}")
            raise

    async def shutdown(self):
        """关闭API服务"""
        try:
            if self.db_service:
                await self.db_service.close()
            logger.info("API服务关闭成功")
        except Exception as e:
            logger.warning(f"API服务关闭失败: {e}")

    async def create_user(self, user_data: UserCreateRequest) -> Dict[str, Any]:
        """创建用户"""
        try:
            await self._validate_user_uniqueness(user_data)
            result = await self._create_user_record(user_data)
            return await self._generate_user_response(result, user_data)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"创建用户失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

    async def _validate_user_uniqueness(self, user_data: UserCreateRequest):
        """验证用户唯一性"""
        existing_users = await self.db_service.db_pool.execute_query(
            "SELECT id FROM users WHERE username = $1 OR email = $2",
            (user_data.username, user_data.email)
        )

        if existing_users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名或邮箱已存在"
            )

    async def _create_user_record(self, user_data: UserCreateRequest) -> Dict[str, Any]:
        """创建用户记录"""
        result = await self.db_service.create_user(
            user_data.username,
            user_data.email,
            user_data.password,
            user_data.initial_balance
        )

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "用户创建失败")
            )

        return result

    async def _generate_user_response(self, result: Dict[str, Any], user_data: UserCreateRequest) -> Dict[str, Any]:
        """生成用户响应"""
        token_data = {
            "user_id": result["user_id"],
            "username": user_data.username
        }
        access_token = self.auth_service.create_access_token(token_data)

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.auth_service.access_token_expire_minutes * 60,
            "user": result["user"]
        }

    async def authenticate_user(self, login_data: UserLoginRequest) -> Dict[str, Any]:
        """用户认证"""
        try:
            user = await self.db_service.authenticate_user(
                login_data.username,
                login_data.password
            )

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="用户名或密码错误"
                )

            # 生成访问令牌
            token_data = {
                "user_id": user["id"],
                "username": user["username"]
            }
            access_token = self.auth_service.create_access_token(token_data)

            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": self.auth_service.access_token_expire_minutes * 60,
                "user": user
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"用户认证失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

    async def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """获取用户资料"""
        try:
            user = await self.db_service.get_user(user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="用户不存在"
                )

            # 获取用户统计信息
            orders_count = await self.db_service.db_pool.execute_query("""
                SELECT COUNT(*) as count FROM orders WHERE user_id = $1
            """, (user_id,))

            positions_count = await self.db_service.db_pool.execute_query("""
                SELECT COUNT(*) as count FROM positions WHERE user_id = $1
            """, (user_id,))

            user["stats"] = {
                "total_orders": orders_count[0]["count"] if orders_count else 0,
                "total_positions": positions_count[0]["count"] if positions_count else 0
            }

            return user

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取用户资料失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

    async def create_order(self, user_id: int, order_data: OrderCreateRequest) -> Dict[str, Any]:
        """创建订单"""
        try:
            # 检查用户余额（如果是买入订单）
            if order_data.side == "buy":
                user = await self.db_service.get_user(user_id)
                required_amount = order_data.quantity * order_data.price

                if user["balance"] < required_amount:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"余额不足，需要 {required_amount}，当前余额 {user['balance']}"
                    )

            # 创建订单
            result = await self.db_service.create_order(user_id, order_data.dict())

            if result["success"]:
                # 如果有真实的交易引擎，执行订单
                if self.execution_engine:
                    try:
                        # 这里可以调用真实的执行引擎
                        execution_result = await self.execution_engine.execute_order(result["order"])
                        if execution_result.get("success"):
                            # 更新持仓
                            await self.db_service.update_position(
                                user_id,
                                order_data.symbol,
                                order_data.quantity if order_data.side == "buy" else -order_data.quantity,
                                order_data.price
                            )

                            # 更新用户余额
                            if order_data.side == "buy":
                                new_balance = user["balance"] - required_amount
                            else:
                                new_balance = user["balance"] + \
                                    (order_data.quantity * order_data.price)

                            await self.db_service.db_pool.execute_command(
                                "UPDATE users SET balance = $1 WHERE id = $2",
                                (new_balance, user_id)
                            )

                    except Exception as e:
                        logger.warning(f"订单执行失败，使用模拟模式: {e}")

                return result
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.get("error", "订单创建失败")
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"创建订单失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

    async def get_user_orders(self, user_id: int, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """获取用户订单"""
        try:
            orders = await self.db_service.get_user_orders(user_id, status, limit)
            return orders

        except Exception as e:
            logger.error(f"获取用户订单失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

    async def get_user_positions(self, user_id: int) -> List[Dict[str, Any]]:
        """获取用户持仓"""
        try:
            positions = await self.db_service.get_user_positions(user_id)

            # 计算市值和盈亏（模拟数据）
            for position in positions:
                # 模拟当前价格（实际应该从市场数据获取）
                current_price = position["avg_price"] * (1 + (time.time() % 10 - 5) / 100)  # 模拟价格波动
                position["current_price"] = round(current_price, 2)
                position["market_value"] = position["quantity"] * current_price
                position["unrealized_pnl"] = (
                    current_price - position["avg_price"]) * position["quantity"]

            return positions

        except Exception as e:
            logger.error(f"获取用户持仓失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取市场数据

        根据环境配置返回真实或模拟数据：
        - 生产环境：从数据源管理器获取实时数据
        - 开发/测试环境：可配置使用模拟数据

        Args:
            symbol: 股票代码

        Returns:
            Dict[str, Any]: 市场数据

        Raises:
            HTTPException: 获取数据失败时抛出
        """
        try:
            # 检查环境配置
            environment = self.config_manager.get("environment", "development")
            use_mock = self.config_manager.get("market_data.use_mock", False)

            if environment == "production" and not use_mock:
                # 生产环境：使用真实数据源
                return await self._get_real_market_data(symbol)
            else:
                # 开发/测试环境：使用模拟数据
                return await self._get_mock_market_data(symbol)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

    async def _get_real_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        从真实数据源获取市场数据

        Args:
            symbol: 股票代码

        Returns:
            Dict[str, Any]: 市场数据

        Raises:
            HTTPException: 数据源不可用或获取失败
        """
        try:
            # 获取数据源管理器
            from ....data.sources.intelligent_source_manager import get_data_source_manager
            data_source_manager = get_data_source_manager()

            # 获取实时数据
            data = await data_source_manager.get_realtime_data(symbol)

            if not data:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="数据源暂时不可用"
                )

            return {
                "symbol": symbol,
                "price": data.get("price", 0.0),
                "change": data.get("change_percent", 0.0),
                "volume": data.get("volume", 0),
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                "source": data.get("source", "unknown"),
                "high": data.get("high"),
                "low": data.get("low"),
                "open": data.get("open"),
                "prev_close": data.get("prev_close")
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"从数据源获取市场数据失败: {e}")
            # 检查是否有降级方案
            if self.config_manager.get("market_data.fallback_to_mock", False):
                logger.warning("数据源获取失败，降级到模拟数据")
                return await self._get_mock_market_data(symbol)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="数据源服务不可用"
            )

    async def _get_mock_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取模拟市场数据（仅用于开发测试）

        Args:
            symbol: 股票代码

        Returns:
            Dict[str, Any]: 模拟市场数据
        """
        import time
        import hashlib

        # 基于股票代码生成稳定的基准价格
        hash_value = int(hashlib.md5(symbol.encode()).hexdigest(), 16)
        base_price = 10.0 + (hash_value % 990)  # 10-1000元

        # 生成波动价格
        time_seed = time.time() % 3600  # 每小时重置
        volatility = 0.02  # 2%波动
        price_change = (time_seed / 3600 - 0.5) * 2 * volatility
        current_price = base_price * (1 + price_change)

        # 生成成交量
        volume_base = hash_value % 1000000
        volume = int(volume_base * (0.5 + time_seed / 3600))

        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "change": round(price_change * 100, 2),
            "volume": volume,
            "timestamp": datetime.now().isoformat(),
            "source": "mock",
            "high": round(current_price * 1.02, 2),
            "low": round(current_price * 0.98, 2),
            "open": round(base_price, 2),
            "prev_close": round(base_price, 2),
            "warning": "This is mock data for development/testing only"
        }


def create_trading_api_app() -> FastAPI:
    """创建交易API应用

    工厂方法，创建完整的交易API应用实例，包含：
    - FastAPI应用配置
    - 安全中间件配置
    - 路由注册
    - 生命周期管理
    """
    app = FastAPI(
        title="RQA2025量化交易API",
        description="A股量化交易系统REST API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # 配置安全中间件
    _configure_security_middleware(app)

    # 配置HTTPS重定向（生产环境）
    _configure_https_redirect(app)

    # 配置路由
    _configure_routes(app)

    return app


def _configure_security_middleware(app: FastAPI):
    """配置安全中间件"""
    config_manager = UnifiedConfigManager()
    is_production = config_manager.get("environment", "development") == "production"

    if is_production:
        # 生产环境：只允许特定的域名
        allowed_origins = [
            "https://app.rqa2025.com",
            "https://admin.rqa2025.com",
            "https://api.rqa2025.com"
        ]
        allow_credentials = True
    else:
        # 开发环境：允许所有域名（用于测试）
        allowed_origins = ["*"]
        allow_credentials = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=86400,  # 24小时缓存预检请求
    )


def _configure_https_redirect(app: FastAPI):
    """配置HTTPS重定向中间件"""
    config_manager = UnifiedConfigManager()
    is_production = config_manager.get("environment", "development") == "production"

    if is_production:
        @app.middleware("http")
        async def https_redirect_middleware(request, call_next):
            return await _handle_https_redirect(request, call_next)

        async def _handle_https_redirect(request, call_next):
            # 检查是否是HTTP请求
            if request.headers.get("x-forwarded-proto", "http") != "https":
                # 构建HTTPS URL
                host = request.headers.get("host", request.url.hostname)
                if host:
                    https_url = f"https://{host}{request.url.path}"
                    if request.url.query:
                        https_url += f"?{request.url.query}"

                    from starlette.responses import RedirectResponse
                    return RedirectResponse(https_url, status_code=301)

            response = await call_next(request)
            # 添加安全头
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

            return response


def _configure_routes(app: FastAPI):
    """配置API路由"""
    # 初始化服务
    db_service = None
    api_service = None

    # 配置生命周期事件
    _configure_lifecycle_events(app, db_service, api_service)

    # 配置健康检查路由
    _configure_health_routes(app, db_service)

    # 配置认证路由
    _configure_auth_routes(app, api_service)

    # 配置用户路由
    _configure_user_routes(app, api_service)

    # 配置其他路由
    _configure_other_routes(app, api_service)


def _configure_lifecycle_events(app: FastAPI, db_service, api_service):
    """配置生命周期事件"""
    @app.on_event("startup")
    async def startup_event():
        """应用启动事件"""
        nonlocal db_service, api_service
        try:
            db_service = await get_database_service()
            api_service = TradingAPIService(db_service)
            logger.info("交易API服务启动成功")
        except Exception as e:
            logger.error(f"交易API服务启动失败: {e}")
            raise

    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭事件"""
        nonlocal db_service
        if db_service:
            await db_service.close()
            logger.info("交易API服务关闭")


def _configure_health_routes(app: FastAPI, db_service):
    """配置健康检查路由"""
    @app.get("/health", response_model=APIResponse)
    async def health_check():
        """健康检查"""
        try:
            if db_service:
                health_data = await db_service.health_check()
                return APIResponse(
                    success=True,
                    message="服务正常",
                    data=health_data
                )
            else:
                return APIResponse(
                    success=False,
                    message="服务未初始化",
                    data={"status": "uninitialized"}
                )
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return APIResponse(
                success=False,
                message="健康检查失败",
                data={"error": str(e)}
            )


def _configure_auth_routes(app: FastAPI, api_service):
    """配置认证路由"""
    @app.post("/auth/register", response_model=APIResponse)
    async def register_user(user_data: UserCreateRequest):
        """用户注册"""
        if not api_service:
            raise HTTPException(status_code=503, detail="服务不可用")

        result = await api_service.create_user(user_data)
        return APIResponse(
            success=True,
            message="用户注册成功",
            data=result
        )

    # 用户登录 (带速率限制)
    @app.post("/auth/login", response_model=APIResponse)
    @limiter.limit("5/minute")  # 每分钟最多5次登录尝试
    async def login_user(request: Request, login_data: UserLoginRequest):
        """用户登录"""
        if not api_service:
            raise HTTPException(status_code=503, detail="服务不可用")

        result = await api_service.authenticate_user(login_data)
        return APIResponse(
            success=True,
            message="登录成功",
            data=result
        )


def _configure_user_routes(app: FastAPI, api_service):
    """配置用户路由"""
    @app.get("/users/me", response_model=APIResponse)
    async def get_current_user(current_user: Dict = Depends(api_service.auth_service.get_current_user if api_service else None)):
        """获取当前用户信息"""
        if not api_service:
            raise HTTPException(status_code=503, detail="服务不可用")

        user_profile = await api_service.get_user_profile(current_user["user_id"])
        return APIResponse(
            success=True,
            message="获取用户信息成功",
            data=user_profile
        )


def _configure_other_routes(app: FastAPI, api_service):
    """配置其他路由"""
    @app.post("/orders", response_model=APIResponse)
    async def create_order(
        order_data: OrderCreateRequest,
        current_user: Dict = Depends(
            api_service.auth_service.get_current_user if api_service else None)
    ):
        """创建交易订单"""
        if not api_service:
            raise HTTPException(status_code=503, detail="服务不可用")

        # 检查用户是否有交易权限
        if not api_service.auth_service.check_permission(current_user, required_role="user"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="需要用户权限才能创建订单"
            )

        result = await api_service.create_order(current_user["user_id"], order_data)
        return APIResponse(
            success=True,
            message="订单创建成功",
            data=result
        )

    @app.get("/orders", response_model=APIResponse)
    async def get_orders(
        status: Optional[str] = None,
        limit: int = 50,
        current_user: Dict = Depends(
            api_service.auth_service.get_current_user if api_service else None)
    ):
        """获取用户订单"""
        if not api_service:
            raise HTTPException(status_code=503, detail="服务不可用")

        orders = await api_service.get_user_orders(current_user["user_id"], status=status, limit=limit)
        return APIResponse(
            success=True,
            message="获取订单成功",
            data=orders
        )

    # 添加基本面数据API端点
    @app.get("/fundamental/{symbol}", response_model=APIResponse)
    async def get_fundamental_data(
        symbol: str,
        current_user: Dict = Depends(
            api_service.auth_service.get_current_user if api_service else None)
    ):
        """获取股票基本面数据"""
        if not api_service:
            raise HTTPException(status_code=503, detail="服务不可用")

        try:
            from src.infrastructure.integration.data_source_manager import get_data_source_manager
            data_source_manager = get_data_source_manager()
            
            # 获取股票基本信息
            stock_info = await data_source_manager.get_stock_info(symbol)
            
            if stock_info:
                return APIResponse(
                    success=True,
                    message="获取基本面数据成功",
                    data=stock_info
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="无法获取基本面数据"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取基本面数据失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

    # 添加数据源健康检查API端点
    @app.get("/data-sources/health", response_model=APIResponse)
    async def get_data_sources_health(
        current_user: Dict = Depends(
            api_service.auth_service.get_current_user if api_service else None)
    ):
        """获取数据源健康状态"""
        if not api_service:
            raise HTTPException(status_code=503, detail="服务不可用")

        try:
            from src.infrastructure.integration.data_source_manager import get_data_source_manager
            data_source_manager = get_data_source_manager()
            
            # 获取数据源统计信息
            stats = data_source_manager.get_data_source_stats()
            cache_stats = data_source_manager.get_cache_stats()
            
            return APIResponse(
                success=True,
                message="获取数据源健康状态成功",
                data={
                    "data_sources": stats,
                    "cache": cache_stats
                }
            )
            
        except Exception as e:
            logger.error(f"获取数据源健康状态失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

    # 添加数据质量监控API端点
    @app.get("/data-quality/{symbol}", response_model=APIResponse)
    async def get_data_quality(
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        current_user: Dict = Depends(
            api_service.auth_service.get_current_user if api_service else None)
    ):
        """获取数据质量信息"""
        if not api_service:
            raise HTTPException(status_code=503, detail="服务不可用")

        try:
            from src.infrastructure.integration.data_source_manager import get_data_source_manager
            data_source_manager = get_data_source_manager()
            
            # 获取股票数据用于质量评估
            end_date = end_date or datetime.now().strftime("%Y%m%d")
            start_date = start_date or (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            
            stock_data = await data_source_manager.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_type="daily"
            )
            
            if stock_data:
                # 简单的数据质量评估
                quality_score = len(stock_data) / 30  # 假设30天的完整数据
                quality_score = min(quality_score, 1.0)
                
                return APIResponse(
                    success=True,
                    message="获取数据质量信息成功",
                    data={
                        "symbol": symbol,
                        "data_points": len(stock_data),
                        "quality_score": round(quality_score, 2),
                        "date_range": {
                            "start": start_date,
                            "end": end_date
                        }
                    }
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="无法获取数据质量信息"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取数据质量信息失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

# 应用启动配置

# 创建全局交易API应用实例
trading_api_app = create_trading_api_app()
