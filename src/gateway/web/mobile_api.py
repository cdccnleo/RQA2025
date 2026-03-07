"""
移动端专用API
为移动端应用提供优化的API接口
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mobile", tags=["移动端"])


class MobileSignalResponse(BaseModel):
    """移动端信号响应"""
    id: str
    symbol: str
    type: str
    strength: str
    timestamp: float
    price: Optional[float] = None


class MobileMarketDataResponse(BaseModel):
    """移动端市场数据响应"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: float


class MobilePortfolioResponse(BaseModel):
    """移动端投资组合响应"""
    total_value: float
    total_return: float
    daily_return: float
    positions: List[Dict[str, Any]]


# 字段过滤函数
def filter_fields(data: Dict, fields: str) -> Dict:
    """过滤数据字段"""
    if not fields:
        return data
    
    field_list = [f.strip() for f in fields.split(",")]
    return {k: v for k, v in data.items() if k in field_list}


@router.get("/signals", response_model=List[MobileSignalResponse])
async def get_mobile_signals(
    limit: int = Query(20, description="返回数量限制", ge=1, le=50),
    fields: str = Query("id,symbol,type,strength,timestamp", description="返回字段"),
    symbol: Optional[str] = Query(None, description="股票代码过滤")
):
    """
    获取移动端信号列表
    
    为移动端优化的信号列表接口，返回精简的数据
    """
    try:
        from .trading_signal_service import get_realtime_signals
        
        # 获取信号（get_realtime_signals 是同步函数）
        signals = get_realtime_signals()
        
        # 过滤股票代码
        if symbol:
            signals = [s for s in signals if s.get('symbol') == symbol]
        
        # 限制数量
        signals = signals[:limit]
        
        # 过滤字段并转换格式
        mobile_signals = []
        for signal in signals:
            mobile_signal = filter_fields(signal, fields)
            
            # 确保必要字段存在
            mobile_signal.setdefault('id', signal.get('id', ''))
            mobile_signal.setdefault('symbol', signal.get('symbol', 'UNKNOWN'))
            mobile_signal.setdefault('type', signal.get('type', 'unknown'))
            mobile_signal.setdefault('strength', signal.get('strength', 'medium'))
            mobile_signal.setdefault('timestamp', signal.get('timestamp', datetime.now().timestamp()))
            mobile_signal.setdefault('price', signal.get('price'))
            
            mobile_signals.append(MobileSignalResponse(**mobile_signal))
        
        logger.info(f"移动端信号列表: 返回 {len(mobile_signals)} 条信号")
        return mobile_signals
        
    except Exception as e:
        logger.error(f"获取移动端信号列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/market-data", response_model=List[MobileMarketDataResponse])
async def get_mobile_market_data(
    symbols: str = Query(..., description="股票代码列表，逗号分隔"),
    fields: str = Query("symbol,price,change,change_percent,volume,timestamp", description="返回字段")
):
    """
    获取移动端市场数据
    
    为移动端优化的市场数据接口，返回实时行情
    """
    try:
        from .market_data_service import get_market_data_service
        
        service = get_market_data_service()
        
        # 解析股票代码
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        
        if not symbol_list:
            raise HTTPException(status_code=400, detail="股票代码不能为空")
        
        if len(symbol_list) > 20:
            raise HTTPException(status_code=400, detail="最多支持20个股票代码")
        
        # 获取市场数据
        market_data_list = []
        for symbol in symbol_list:
            try:
                # 获取最新数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)
                
                df = service.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    limit=1
                )
                
                if not df.empty:
                    latest = df.iloc[-1]
                    
                    # 计算涨跌
                    prev_close = latest.get('close', 0)
                    current_price = latest.get('close', 0)
                    
                    # 如果有前收盘价，计算涨跌
                    if len(df) > 1:
                        prev_close = df.iloc[-2].get('close', current_price)
                    
                    change = current_price - prev_close
                    change_percent = (change / prev_close * 100) if prev_close else 0
                    
                    data = {
                        'symbol': symbol,
                        'price': round(current_price, 2),
                        'change': round(change, 2),
                        'change_percent': round(change_percent, 2),
                        'volume': int(latest.get('volume', 0)),
                        'timestamp': datetime.now().timestamp()
                    }
                    
                    # 过滤字段
                    filtered_data = filter_fields(data, fields)
                    market_data_list.append(MobileMarketDataResponse(**filtered_data))
                else:
                    # 无数据时返回占位符
                    market_data_list.append(MobileMarketDataResponse(
                        symbol=symbol,
                        price=0.0,
                        change=0.0,
                        change_percent=0.0,
                        volume=0,
                        timestamp=datetime.now().timestamp()
                    ))
                    
            except Exception as e:
                logger.warning(f"获取股票 {symbol} 市场数据失败: {e}")
                market_data_list.append(MobileMarketDataResponse(
                    symbol=symbol,
                    price=0.0,
                    change=0.0,
                    change_percent=0.0,
                    volume=0,
                    timestamp=datetime.now().timestamp()
                ))
        
        logger.info(f"移动端市场数据: 返回 {len(market_data_list)} 条数据")
        return market_data_list
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取移动端市场数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/portfolio", response_model=MobilePortfolioResponse)
async def get_mobile_portfolio(
    user_id: str = Query(..., description="用户ID")
):
    """
    获取移动端投资组合
    
    为移动端优化的投资组合接口
    """
    try:
        # 这里应该从实际的投资组合服务获取数据
        # 目前返回模拟数据
        
        # 模拟投资组合数据
        portfolio = {
            "total_value": 100000.0,
            "total_return": 5000.0,
            "daily_return": 200.0,
            "positions": [
                {
                    "symbol": "002837",
                    "name": "英维克",
                    "quantity": 100,
                    "avg_price": 25.5,
                    "current_price": 28.2,
                    "market_value": 2820.0,
                    "return": 270.0,
                    "return_percent": 10.59
                },
                {
                    "symbol": "688702",
                    "name": "盛科通信",
                    "quantity": 50,
                    "avg_price": 42.0,
                    "current_price": 45.5,
                    "market_value": 2275.0,
                    "return": 175.0,
                    "return_percent": 8.33
                }
            ]
        }
        
        logger.info(f"移动端投资组合: 用户 {user_id}")
        return MobilePortfolioResponse(**portfolio)
        
    except Exception as e:
        logger.error(f"获取移动端投资组合失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/dashboard")
async def get_mobile_dashboard(
    user_id: str = Query(..., description="用户ID")
):
    """
    获取移动端仪表盘数据
    
    聚合展示关键数据
    """
    try:
        # 获取市场概览
        from .market_data_service import get_market_data_service
        service = get_market_data_service()
        
        available_symbols = service.get_available_symbols()
        default_symbol = service.get_default_symbol()
        
        # 获取最新信号
        from .trading_signal_service import get_realtime_signals
        signals = await get_realtime_signals()
        
        # 获取投资组合
        portfolio = await get_mobile_portfolio(user_id)
        
        # 构建仪表盘数据
        dashboard = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "market_overview": {
                "available_symbols": available_symbols,
                "default_symbol": default_symbol,
                "market_status": "open"  # 可以从市场状态服务获取
            },
            "latest_signals": [
                {
                    "id": s.get('id', ''),
                    "symbol": s.get('symbol', ''),
                    "type": s.get('type', ''),
                    "strength": s.get('strength', '')
                }
                for s in signals[:5]  # 只显示前5个信号
            ],
            "portfolio_summary": {
                "total_value": portfolio.total_value,
                "total_return": portfolio.total_return,
                "daily_return": portfolio.daily_return,
                "position_count": len(portfolio.positions)
            },
            "quick_actions": [
                {"action": "view_signals", "label": "查看信号", "icon": "signal"},
                {"action": "view_portfolio", "label": "投资组合", "icon": "portfolio"},
                {"action": "view_market", "label": "市场行情", "icon": "chart"},
                {"action": "settings", "label": "设置", "icon": "settings"}
            ]
        }
        
        return dashboard
        
    except Exception as e:
        logger.error(f"获取移动端仪表盘失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/push-token")
async def register_push_token(
    user_id: str = Query(..., description="用户ID"),
    token: str = Query(..., description="推送令牌"),
    platform: str = Query(..., description="平台（ios/android）")
):
    """
    注册移动端推送令牌
    
    用于接收推送通知
    """
    try:
        # 这里应该将令牌保存到数据库
        # 目前仅记录日志
        
        logger.info(f"注册推送令牌: 用户 {user_id}, 平台 {platform}")
        
        return {
            "success": True,
            "message": "推送令牌注册成功",
            "user_id": user_id,
            "platform": platform
        }
        
    except Exception as e:
        logger.error(f"注册推送令牌失败: {e}")
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")


@router.get("/notifications")
async def get_mobile_notifications(
    user_id: str = Query(..., description="用户ID"),
    limit: int = Query(20, description="返回数量", ge=1, le=100)
):
    """
    获取移动端通知
    
    返回用户的推送通知历史
    """
    try:
        # 模拟通知数据
        notifications = [
            {
                "id": "notif_001",
                "type": "signal",
                "title": "新的交易信号",
                "message": "002837 英维克 出现买入信号",
                "timestamp": datetime.now().timestamp() - 3600,
                "read": False
            },
            {
                "id": "notif_002",
                "type": "alert",
                "title": "价格提醒",
                "message": "688702 盛科通信 价格达到目标价",
                "timestamp": datetime.now().timestamp() - 7200,
                "read": True
            }
        ]
        
        return {
            "user_id": user_id,
            "notifications": notifications[:limit],
            "unread_count": sum(1 for n in notifications if not n['read'])
        }
        
    except Exception as e:
        logger.error(f"获取移动端通知失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")
