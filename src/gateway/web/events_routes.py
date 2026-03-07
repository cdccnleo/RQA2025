"""
事件监控API路由
提供系统事件查询接口
符合架构设计：使用EventBus进行事件通信，使用ServiceContainer进行依赖管理
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime, timedelta

# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

router = APIRouter()

# 事件存储（内存存储，用于演示）
# 在实际生产环境中，应该使用持久化存储（如数据库、文件系统等）
_event_history: List[Dict[str, Any]] = []
_max_history_size = 1000  # 最大存储事件数


def _add_event(event_type: str, message: str, source: str, level: str = "info"):
    """
    添加事件到历史记录
    在实际生产环境中，应该集成EventBus或使用持久化存储
    """
    global _event_history
    
    event = {
        "id": f"event_{int(time.time() * 1000)}",
        "type": event_type,
        "message": message,
        "source": source,
        "timestamp": datetime.now().isoformat(),
        "level": level  # info, warning, error
    }
    
    _event_history.append(event)
    
    # 限制历史记录大小
    if len(_event_history) > _max_history_size:
        _event_history = _event_history[-_max_history_size:]
    
    logger.debug(f"事件已添加: {event_type} - {message}")
    return event


@router.get("/api/v1/system/events")
async def get_system_events(
    limit: Optional[int] = 50,
    level: Optional[str] = None,
    source: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取系统事件列表
    
    参数:
    - limit: 返回事件数量限制（默认50）
    - level: 事件级别过滤（info, warning, error）
    - source: 事件来源过滤
    
    返回:
    {
        "events": [...],
        "total": 100,
        "timestamp": 1234567890
    }
    """
    try:
        # 发布事件查询事件
        try:
            from src.core.event_bus.core import EventBus
            from src.core.event_bus.types import EventType
            
            event_bus = EventBus()
            if not event_bus._initialized:
                event_bus.initialize()
            
            event_bus.publish(
                EventType.SYSTEM_STATUS_CHECKED,
                {"source": "events_routes", "action": "get_system_events"},
                source="events_routes"
            )
        except Exception as e:
            logger.debug(f"发布事件失败: {e}")
        
        # 获取事件列表
        events = _event_history.copy()
        
        # 过滤
        if level:
            events = [e for e in events if e.get("level") == level]
        
        if source:
            events = [e for e in events if e.get("source") == source]
        
        # 排序（最新的在前）
        events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # 限制数量
        if limit:
            events = events[:limit]
        
        return {
            "events": events,
            "total": len(_event_history),
            "filtered_total": len(events),
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"获取系统事件失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统事件失败: {str(e)}")


@router.get("/api/v1/system/events/{event_id}")
async def get_event_detail(event_id: str) -> Dict[str, Any]:
    """
    获取事件详情
    
    参数:
    - event_id: 事件ID
    
    返回:
    事件详情对象
    """
    try:
        # 查找事件
        event = next((e for e in _event_history if e.get("id") == event_id), None)
        
        if not event:
            raise HTTPException(status_code=404, detail=f"事件不存在: {event_id}")
        
        return event
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取事件详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取事件详情失败: {str(e)}")


# 初始化一些示例事件（用于演示）
def _initialize_sample_events():
    """初始化示例事件（仅用于演示）"""
    current_time = datetime.now()
    
    # 添加一些示例事件
    sample_events = [
        {
            "type": "system",
            "message": "系统启动完成",
            "source": "system",
            "level": "info",
            "timestamp": (current_time - timedelta(minutes=60)).isoformat()
        },
        {
            "type": "component",
            "message": "组件检查完成",
            "source": "health_check",
            "level": "info",
            "timestamp": (current_time - timedelta(minutes=30)).isoformat()
        },
        {
            "type": "data",
            "message": "数据源连接成功",
            "source": "data_layer",
            "level": "info",
            "timestamp": (current_time - timedelta(minutes=15)).isoformat()
        },
        {
            "type": "trading",
            "message": "交易执行完成",
            "source": "trading_layer",
            "level": "info",
            "timestamp": (current_time - timedelta(minutes=5)).isoformat()
        },
        {
            "type": "risk",
            "message": "风险评估完成",
            "source": "risk_layer",
            "level": "info",
            "timestamp": (current_time - timedelta(minutes=2)).isoformat()
        }
    ]
    
    for event_data in sample_events:
        _add_event(
            event_data["type"],
            event_data["message"],
            event_data["source"],
            event_data["level"]
        )


# 初始化示例事件
_initialize_sample_events()

