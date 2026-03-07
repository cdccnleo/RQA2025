"""
基础路由模块
包含系统状态、健康检查等基础API端点
"""

import os
import time
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "rqa2025-app",
        "environment": os.getenv("RQA_ENV", "unknown"),
        "timestamp": time.time()
    }


@router.get("/api/v1/status")
async def system_status():
    """系统状态"""
    return {
        "system": "RQA2025",
        "status": "operational",
        "components": {
            "strategy_service": "healthy",
            "trading_service": "healthy",
            "risk_service": "healthy",
            "data_service": "healthy"
        },
        "uptime": time.time(),
        "version": "1.0.0"
    }


@router.get("/api/v1/strategy/status")
async def strategy_status():
    """策略服务状态 - 使用真实数据"""
    try:
        # 尝试从策略执行服务获取真实的活跃策略数量
        from .strategy_execution_service import get_strategy_execution_status
        execution_status = await get_strategy_execution_status()
        
        active_strategies = execution_status.get("running_count", 0)
        total_strategies = execution_status.get("total_count", 0)
        
        return {
            "service": "strategy",
            "status": "healthy",
            "strategies_count": total_strategies,
            "active_strategies": active_strategies,
            "running_count": active_strategies,
            "paused_count": execution_status.get("paused_count", 0),
            "stopped_count": execution_status.get("stopped_count", 0),
            "last_update": time.time()
        }
    except Exception as e:
        # 如果获取失败，尝试从策略构思列表获取
        try:
            from .strategy_routes import load_strategy_conceptions
            strategies = load_strategy_conceptions()
            
            # 统计活跃策略（状态为running或active的策略）
            active_count = 0
            for strategy in strategies:
                status = strategy.get("status", strategy.get("lifecycle_stage", "created"))
                if status in ["running", "active", "deployed", "executing"]:
                    active_count += 1
            
            return {
                "service": "strategy",
                "status": "healthy",
                "strategies_count": len(strategies),
                "active_strategies": active_count,
                "running_count": active_count,
                "paused_count": 0,
                "stopped_count": len(strategies) - active_count,
                "last_update": time.time()
            }
        except Exception as e2:
            # 如果都失败，返回错误状态，不使用模拟数据
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"获取策略状态失败: {e}, {e2}")
            return {
                "service": "strategy",
                "status": "error",
                "strategies_count": 0,
                "active_strategies": 0,
                "running_count": 0,
                "paused_count": 0,
                "stopped_count": 0,
                "error": "无法获取策略状态",
                "last_update": time.time()
            }


@router.get("/api/v1/trading/status")
async def trading_status():
    """交易服务状态"""
    return {
        "service": "trading",
        "status": "healthy",
        "active_orders": 0,
        "executed_trades": 0,
        "last_update": time.time()
    }


@router.get("/api/v1/risk/status")
async def risk_status():
    """风险控制服务状态"""
    return {
        "service": "risk",
        "status": "healthy",
        "risk_alerts": 0,
        "compliance_checks": 0,
        "last_update": time.time()
    }


@router.get("/test")
async def test_endpoint():
    """测试端点"""
    return {"message": "test endpoint works", "timestamp": time.time()}
