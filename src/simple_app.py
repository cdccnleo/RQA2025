#!/usr/bin/env python3
"""
RQA2025简化版应用 - 用于性能测试

这个版本避免了复杂的异步初始化，直接提供基本的API端点。
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="RQA2025量化交易平台 - 简化版",
    description="A股量化交易系统 - 性能测试版本",
    version="1.0.0-test"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模拟数据存储
mock_portfolio = {
    "total_value": 100000.0,
    "cash": 50000.0,
    "positions": {
        "AAPL": {"quantity": 100, "avg_price": 150.0, "current_price": 155.0},
        "GOOGL": {"quantity": 50, "avg_price": 2500.0, "current_price": 2550.0}
    }
}

mock_market_data = {
    "AAPL": {"price": 155.0, "volume": 1000000, "change": 3.33},
    "GOOGL": {"price": 2550.0, "volume": 500000, "change": 2.00},
    "MSFT": {"price": 305.0, "volume": 800000, "change": -1.50}
}


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RQA2025量化交易系统 - 性能测试版本",
        "version": "1.0.0-test",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "1.0.0-test",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "ok",
            "database": "mock",
            "cache": "mock",
            "trading": "mock"
        }
    }


@app.get("/api/health")
async def api_health():
    """API健康检查"""
    return {
        "status": "ok",
        "message": "RQA2025 API服务正常",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/market/data")
async def get_market_data(symbol: Optional[str] = None):
    """获取市场数据"""
    if symbol:
        if symbol in mock_market_data:
            return {
                "symbol": symbol,
                **mock_market_data[symbol],
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    return {
        "data": mock_market_data,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/portfolio/balance")
async def get_portfolio_balance():
    """获取投资组合余额"""
    return {
        "portfolio": mock_portfolio,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/portfolio/history")
async def get_portfolio_history(period: str = "7d"):
    """获取投资组合历史"""
    # 生成模拟历史数据
    import random
    history = []
    base_value = 100000.0

    for i in range(30 if period == "30d" else 7):
        change = random.uniform(-0.02, 0.02)  # -2% 到 +2%的变化
        value = base_value * (1 + change)
        history.append({
            "date": f"2025-09-{30-i:02d}",
            "value": round(value, 2),
            "change_percent": round(change * 100, 2)
        })

    return {
        "period": period,
        "history": history,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/trading/orders")
async def get_trading_orders(status: Optional[str] = None, limit: int = 10):
    """获取交易订单"""
    # 生成模拟订单数据
    orders = [
        {
            "id": f"order_{i}",
            "symbol": ["AAPL", "GOOGL", "MSFT"][i % 3],
            "side": "buy" if i % 2 == 0 else "sell",
            "quantity": (i + 1) * 10,
            "price": 100 + i * 10,
            "status": ["pending", "filled", "cancelled"][i % 3],
            "created_at": datetime.now().isoformat()
        }
        for i in range(limit)
    ]

    if status:
        orders = [o for o in orders if o["status"] == status]

    return {
        "orders": orders,
        "total": len(orders),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/trading/order")
async def create_trading_order(order: Dict[str, Any]):
    """创建交易订单"""
    # 验证订单数据
    required_fields = ["symbol", "quantity", "order_type", "side"]
    for field in required_fields:
        if field not in order:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    # 生成模拟订单ID
    order_id = f"order_{int(datetime.now().timestamp() * 1000)}"

    # 模拟订单处理
    response_order = {
        "id": order_id,
        **order,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "message": "Order created successfully (mock)"
    }

    return response_order


@app.get("/api/user/profile")
async def get_user_profile():
    """获取用户信息"""
    return {
        "user_id": "test_user_001",
        "username": "test_user",
        "email": "test@example.com",
        "balance": 50000.0,
        "portfolio_value": 100000.0,
        "created_at": "2025-01-01T00:00:00Z",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/auth/login")
async def login(credentials: Dict[str, str]):
    """用户登录 (模拟)"""
    username = credentials.get("username", "")
    password = credentials.get("password", "")

    # 简单的模拟验证
    if username and password:
        return {
            "access_token": f"mock_token_{username}",
            "token_type": "bearer",
            "user_id": f"user_{username}",
            "expires_in": 3600,
            "message": "Login successful (mock)"
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/api/user/register")
async def register_user(user_data: Dict[str, Any]):
    """用户注册 (模拟)"""
    required_fields = ["username", "email", "password"]
    for field in required_fields:
        if field not in user_data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    # 模拟用户创建
    return {
        "user_id": f"user_{user_data['username']}",
        "username": user_data["username"],
        "email": user_data["email"],
        "balance": user_data.get("initial_balance", 10000.0),
        "created_at": datetime.now().isoformat(),
        "message": "User registered successfully (mock)"
    }


@app.get("/api/monitoring/metrics")
async def get_monitoring_metrics():
    """获取监控指标"""
    return {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "disk_usage": 23.1,
        "active_connections": 12,
        "response_time_avg": 245,  # ms
        "error_rate": 0.02,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/system/info")
async def get_system_info():
    """获取系统信息"""
    return {
        "version": "1.0.0-test",
        "environment": "testing",
        "database_status": "mock",
        "cache_status": "mock",
        "services": ["api", "trading", "monitoring"],
        "uptime": "00:15:30",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("启动RQA2025简化版应用 (性能测试模式)...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
