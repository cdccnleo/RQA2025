"""
订单路由API路由
提供路由决策、路由统计、路由性能等API接口

量化交易系统合规要求：
- QTS-015: 权限控制 - 所有API需要权限验证
- QTS-016: 操作日志 - 记录所有操作
- QTS-017: 数据脱敏 - 敏感数据需要脱敏处理
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random

# 导入服务层
from .order_routing_service import (
    get_routing_decisions,
    get_routing_stats,
    get_routing_performance,
    get_routing_decision_detail,
    get_filtered_routing_decisions
)

# 导入权限控制 - 使用FastAPI兼容的简化版
from .simple_auth import Permission, require_permission, require_any_permission, audit_log, AuditCategory

# 导入数据脱敏
from .data_masking import DataMasker, MaskingRule

router = APIRouter()

# 初始化数据脱敏器
_masker = DataMasker()


def _mask_id(value: str, prefix: str = 'ID') -> str:
    """
    脱敏ID值 - 保留前缀，中间部分用星号代替
    
    Args:
        value: 原始ID值
        prefix: ID前缀标识
        
    Returns:
        脱敏后的ID值，如 ORD***789
    """
    if not value or len(value) < 6:
        return f"{prefix}***"
    
    # 保留前缀和最后3位，中间用***代替
    return f"{prefix}***{value[-3:]}"


def _mask_decision_data(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对路由决策数据进行脱敏处理
    
    Args:
        decisions: 原始路由决策列表
        
    Returns:
        脱敏后的决策列表
    """
    masked_decisions = []
    for decision in decisions:
        masked = decision.copy()
        # 脱敏订单ID
        if 'order_id' in masked and masked['order_id']:
            masked['order_id'] = _mask_id(masked['order_id'], prefix='ORD')
        # 脱敏决策ID
        if 'decision_id' in masked and masked['decision_id']:
            masked['decision_id'] = _mask_id(masked['decision_id'], prefix='DEC')
        masked_decisions.append(masked)
    return masked_decisions


# ==================== 路由决策API ====================

@router.get("/trading/routing/decisions")
@require_permission(Permission.TRADING_VIEW)
@audit_log(action="get_routing_decisions", category=AuditCategory.TRADING)
async def get_routing_decisions_endpoint(
    request: Request,
    strategy_id: Optional[str] = None,
    status: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    获取路由决策列表 - 使用真实路由系统数据，不使用模拟数据
    
    Args:
        request: FastAPI请求对象
        strategy_id: 策略ID筛选
        status: 状态筛选
        start_time: 开始时间筛选
        end_time: 结束时间筛选
        limit: 返回记录数量限制
        
    Returns:
        包含脱敏后路由决策列表的字典
    """
    try:
        # 如果有筛选条件，使用筛选查询
        if any([strategy_id, status, start_time, end_time]):
            decisions = get_filtered_routing_decisions(
                strategy_id=strategy_id,
                status=status,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
        else:
            decisions = get_routing_decisions()
        
        # 数据脱敏处理
        masked_decisions = _mask_decision_data(decisions)
        
        # 量化交易系统要求：不使用模拟数据，即使为空也返回真实结果
        return {
            "decisions": masked_decisions,
            "total_count": len(decisions),
            "note": "量化交易系统要求使用真实路由数据。如果列表为空，表示当前没有路由决策记录。",
            "filters_applied": {
                "strategy_id": strategy_id,
                "status": status,
                "start_time": start_time,
                "end_time": end_time
            } if any([strategy_id, status, start_time, end_time]) else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取路由决策失败: {str(e)}")


@router.get("/trading/routing/decisions/{decision_id}")
@require_permission(Permission.TRADING_VIEW)
@audit_log(action="get_routing_decision_detail", category=AuditCategory.TRADING)
async def get_routing_decision_detail_endpoint(
    request: Request,
    decision_id: str
) -> Dict[str, Any]:
    """
    获取路由决策详情 - 用于详情弹窗展示
    
    Args:
        request: FastAPI请求对象
        decision_id: 决策ID
        
    Returns:
        包含脱敏后决策详情的字典
    """
    try:
        detail = get_routing_decision_detail(decision_id)
        
        if not detail:
            raise HTTPException(status_code=404, detail=f"决策ID {decision_id} 不存在")
        
        # 数据脱敏处理
        masked_detail = detail.copy()
        if 'order_id' in masked_detail and masked_detail['order_id']:
            masked_detail['order_id'] = _masker.mask_id(masked_detail['order_id'], prefix='ORD')
        if 'decision_id' in masked_detail and masked_detail['decision_id']:
            masked_detail['decision_id'] = _masker.mask_id(masked_detail['decision_id'], prefix='DEC')
        
        return {
            "decision": masked_detail,
            "query_time": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取路由决策详情失败: {str(e)}")


# ==================== 路由统计API ====================

@router.get("/trading/routing/stats")
@require_permission(Permission.TRADING_VIEW)
@audit_log(action="get_routing_stats", category=AuditCategory.TRADING)
async def get_routing_stats_endpoint(request: Request) -> Dict[str, Any]:
    """
    获取路由统计
    
    Args:
        request: FastAPI请求对象
        
    Returns:
        路由统计数据字典
    """
    try:
        stats = get_routing_stats()
        return {
            "stats": stats,
            "query_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取路由统计失败: {str(e)}")


# ==================== 路由性能API ====================

@router.get("/trading/routing/performance")
@require_permission(Permission.TRADING_VIEW)
@audit_log(action="get_routing_performance", category=AuditCategory.TRADING)
async def get_routing_performance_endpoint(request: Request) -> Dict[str, Any]:
    """
    获取路由性能
    
    Args:
        request: FastAPI请求对象
        
    Returns:
        路由性能数据字典
    """
    try:
        performance = get_routing_performance()
        return {
            **performance,
            "query_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取路由性能失败: {str(e)}")


# ==================== 路由失败告警API ====================

@router.get("/trading/routing/alerts")
@require_any_permission([Permission.TRADING_VIEW, Permission.ALERT_VIEW])
@audit_log(action="get_routing_alerts", category=AuditCategory.ALERT)
async def get_routing_alerts_endpoint(
    request: Request,
    acknowledged: Optional[bool] = None,
    source: Optional[str] = None
) -> Dict[str, Any]:
    """
    获取路由相关告警
    
    Args:
        request: FastAPI请求对象
        acknowledged: 是否已确认筛选
        source: 告警来源筛选
        
    Returns:
        告警列表字典
    """
    try:
        from .alert_center import get_alert_center, AlertStatus
        
        alert_center = get_alert_center()
        
        # 根据 acknowledged 参数确定状态筛选
        status = None
        if acknowledged is not None:
            status = AlertStatus.ACKNOWLEDGED if acknowledged else AlertStatus.PENDING
        
        # 使用 source 参数筛选路由相关告警
        alerts = alert_center.get_all_alerts(
            status=status,
            source=source or "routing"
        )
        
        # 将 Alert 对象转换为字典
        alert_dicts = []
        for alert in alerts:
            alert_dicts.append({
                "alert_id": alert.alert_id,
                "strategy_id": alert.strategy_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity),
                "status": alert.status.value if hasattr(alert.status, 'value') else str(alert.status),
                "source": alert.source,
                "created_at": alert.created_at,
                "acknowledged_by": getattr(alert, 'acknowledged_by', None),
                "acknowledged_at": getattr(alert, 'acknowledged_at', None)
            })
        
        return {
            "alerts": alert_dicts,
            "total_count": len(alert_dicts),
            "query_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取路由告警失败: {str(e)}")


@router.post("/trading/routing/alerts/{alert_id}/acknowledge")
@require_permission(Permission.ALERT_ACKNOWLEDGE)
@audit_log(action="acknowledge_routing_alert", category=AuditCategory.ALERT)
async def acknowledge_routing_alert_endpoint(
    request: Request,
    alert_id: str
) -> Dict[str, Any]:
    """
    确认路由告警
    
    Args:
        request: FastAPI请求对象
        alert_id: 告警ID
        
    Returns:
        确认结果字典
    """
    try:
        from .alert_center import get_alert_center
        
        alert_center = get_alert_center()
        
        # 获取当前用户ID - 从请求头或默认system
        user_id = request.headers.get('X-User-ID', 'system')
        
        success = alert_center.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=user_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"告警ID {alert_id} 不存在或无法确认")
        
        return {
            "success": True,
            "message": f"告警 {alert_id} 已确认",
            "acknowledge_time": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"确认告警失败: {str(e)}")
