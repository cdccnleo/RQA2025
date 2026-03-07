"""
交易信号API路由
提供实时信号、信号统计、信号分布等API接口
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random

# 导入服务层
from .trading_signal_service import (
    get_realtime_signals,
    get_signal_stats,
    get_signal_distribution
)

router = APIRouter()

# ==================== 实时信号API ====================

@router.get("/trading/signals/realtime")
async def get_realtime_signals_endpoint() -> Dict[str, Any]:
    """获取实时交易信号 - 使用真实信号生成器数据，不使用模拟数据"""
    try:
        signals = get_realtime_signals()
        # 量化交易系统要求：不使用模拟数据，即使为空也返回真实结果
        return {
            "signals": signals,
            "note": "量化交易系统要求使用真实信号数据。如果列表为空，表示当前没有生成的交易信号。"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取实时信号失败: {str(e)}")


# ==================== 信号统计API ====================

@router.get("/trading/signals/stats")
async def get_signal_stats_endpoint() -> Dict[str, Any]:
    """获取信号统计"""
    try:
        stats = get_signal_stats()
        return {
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取信号统计失败: {str(e)}")


# ==================== 信号分布API ====================

@router.get("/trading/signals/distribution")
async def get_signal_distribution_endpoint() -> Dict[str, Any]:
    """获取信号分布"""
    try:
        distribution = get_signal_distribution()
        return distribution
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取信号分布失败: {str(e)}")

