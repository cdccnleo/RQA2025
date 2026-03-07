"""
风险告警服务
提供详细的告警信息查询接口
符合架构设计：使用EventBus进行事件通信，使用ServiceContainer进行依赖管理
"""

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

# 告警存储（内存存储，用于演示）
# 在实际生产环境中，应该使用持久化存储（如数据库、文件系统等）
_alert_history: List[Dict[str, Any]] = []
_max_alert_history_size = 1000  # 最大存储告警数


def _add_alert(alert_type: str, message: str, level: str = "warning", source: str = "risk_control", details: Optional[Dict[str, Any]] = None):
    """
    添加告警到历史记录
    在实际生产环境中，应该集成EventBus或使用持久化存储
    """
    global _alert_history
    
    alert = {
        "id": f"alert_{int(time.time() * 1000)}",
        "type": alert_type,  # risk_limit, compliance_violation, system_error, etc.
        "message": message,
        "level": level,  # info, warning, error, critical
        "source": source,
        "details": details or {},
        "timestamp": datetime.now().isoformat(),
        "status": "active",  # active, resolved, acknowledged
        "resolved_at": None
    }
    
    _alert_history.append(alert)
    
    # 限制历史记录大小
    if len(_alert_history) > _max_alert_history_size:
        _alert_history = _alert_history[-_max_alert_history_size:]
    
    logger.debug(f"告警已添加: {alert_type} - {message}")
    return alert


async def get_risk_alerts(
    limit: Optional[int] = 50,
    level: Optional[str] = None,
    status: Optional[str] = None,
    source: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取风险告警列表
    
    参数:
    - limit: 返回告警数量限制（默认50）
    - level: 告警级别过滤（info, warning, error, critical）
    - status: 告警状态过滤（active, resolved, acknowledged）
    - source: 告警来源过滤
    
    返回:
    {
        "alerts": [...],
        "total": 100,
        "active_count": 10,
        "resolved_count": 90,
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
                {"source": "risk_alerts_service", "action": "get_risk_alerts"},
                source="risk_alerts_service"
            )
        except Exception as e:
            logger.debug(f"发布事件失败: {e}")
        
        # 获取告警列表
        alerts = _alert_history.copy()
        
        # 过滤
        if level:
            alerts = [a for a in alerts if a.get("level") == level]
        
        if status:
            alerts = [a for a in alerts if a.get("status") == status]
        
        if source:
            alerts = [a for a in alerts if a.get("source") == source]
        
        # 排序（最新的在前）
        alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # 限制数量
        if limit:
            alerts = alerts[:limit]
        
        # 统计
        active_count = sum(1 for a in _alert_history if a.get("status") == "active")
        resolved_count = sum(1 for a in _alert_history if a.get("status") == "resolved")
        acknowledged_count = sum(1 for a in _alert_history if a.get("status") == "acknowledged")
        
        return {
            "alerts": alerts,
            "total": len(_alert_history),
            "active_count": active_count,
            "resolved_count": resolved_count,
            "acknowledged_count": acknowledged_count,
            "filtered_total": len(alerts),
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"获取风险告警列表失败: {e}")
        raise


async def get_alert_detail(alert_id: str) -> Dict[str, Any]:
    """
    获取告警详情
    
    参数:
    - alert_id: 告警ID
    
    返回:
    告警详情对象
    """
    try:
        # 查找告警
        alert = next((a for a in _alert_history if a.get("id") == alert_id), None)
        
        if not alert:
            return {
                "error": f"告警不存在: {alert_id}",
                "timestamp": int(time.time())
            }
        
        return alert
    except Exception as e:
        logger.error(f"获取告警详情失败: {e}")
        return {
            "error": f"获取告警详情失败: {str(e)}",
            "timestamp": int(time.time())
        }


# 初始化一些示例告警（用于演示）
def _initialize_sample_alerts():
    """初始化示例告警（仅用于演示）"""
    current_time = datetime.now()
    
    # 添加一些示例告警
    sample_alerts = [
        {
            "type": "risk_limit",
            "message": "持仓风险超过预设阈值",
            "level": "warning",
            "source": "risk_control",
            "details": {"threshold": 0.8, "current": 0.85},
            "timestamp": (current_time - timedelta(minutes=30)).isoformat()
        },
        {
            "type": "compliance_violation",
            "message": "检测到潜在的合规违规",
            "level": "error",
            "source": "compliance_check",
            "details": {"rule": "position_limit", "violation": "exceeded"},
            "timestamp": (current_time - timedelta(minutes=15)).isoformat()
        }
    ]
    
    for alert_data in sample_alerts:
        _add_alert(
            alert_data["type"],
            alert_data["message"],
            alert_data["level"],
            alert_data["source"],
            alert_data["details"]
        )


# 初始化示例告警
_initialize_sample_alerts()

