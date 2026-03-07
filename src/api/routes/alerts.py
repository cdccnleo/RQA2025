"""
告警管理API路由

提供告警查询、确认、解决等功能的REST API接口
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ...pipeline.monitoring.alert_manager import AlertManager, AlertSeverity, AlertStatus
from ...pipeline.monitoring.rollback_manager import RollbackManager


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])

# 全局告警管理器实例
_alert_manager: Optional[AlertManager] = None
_rollback_manager: Optional[RollbackManager] = None


def get_alert_manager() -> AlertManager:
    """获取或创建告警管理器实例"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def get_rollback_manager() -> RollbackManager:
    """获取或创建回滚管理器实例"""
    global _rollback_manager
    if _rollback_manager is None:
        _rollback_manager = RollbackManager(
            model_id="default_model",
            model_path="models/model_latest.joblib"
        )
    return _rollback_manager


# ============ 请求/响应模型 ============

class AlertResponse(BaseModel):
    """告警响应"""
    id: str
    title: str
    message: str
    severity: str
    status: str
    source: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class AlertListResponse(BaseModel):
    """告警列表响应"""
    alerts: List[AlertResponse]
    total: int
    by_severity: Dict[str, int]
    by_status: Dict[str, int]
    page: int
    page_size: int


class AcknowledgeAlertRequest(BaseModel):
    """确认告警请求"""
    acknowledged_by: str = Field(..., description="确认人")


class AcknowledgeAlertResponse(BaseModel):
    """确认告警响应"""
    success: bool
    message: str
    alert_id: str


class ResolveAlertResponse(BaseModel):
    """解决告警响应"""
    success: bool
    message: str
    alert_id: str


class SuppressAlertRequest(BaseModel):
    """抑制告警请求"""
    duration_minutes: int = Field(default=60, ge=5, le=1440, description="抑制持续时间（分钟）")


class SuppressAlertResponse(BaseModel):
    """抑制告警响应"""
    success: bool
    message: str
    alert_id: str
    suppressed_until: datetime


class AlertStatisticsResponse(BaseModel):
    """告警统计响应"""
    total_rules: int
    enabled_rules: int
    active_alerts: int
    total_alerts_history: int
    alerts_by_severity: Dict[str, int]
    suppressed_metrics: List[str]


class RollbackDecisionResponse(BaseModel):
    """回滚决策响应"""
    should_rollback: bool
    trigger: Optional[str] = None
    confidence: float
    reasons: List[str]
    recommended_action: str
    timestamp: datetime


class ExecuteRollbackRequest(BaseModel):
    """执行回滚请求"""
    force: bool = Field(default=False, description="是否强制回滚")
    target_version: Optional[str] = Field(default=None, description="目标版本")


class ExecuteRollbackResponse(BaseModel):
    """执行回滚响应"""
    success: bool
    status: str
    previous_version: Optional[str] = None
    current_version: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    message: str


class RollbackStatusResponse(BaseModel):
    """回滚状态响应"""
    model_id: str
    is_rollback_in_progress: bool
    thresholds: Dict[str, float]
    baseline_metrics: Dict[str, float]
    rollback_count: int
    backup_available: bool


# ============ API端点 ============

@router.get("", response_model=AlertListResponse)
async def get_alerts(
    severity: Optional[str] = Query(None, description="严重程度过滤（逗号分隔）"),
    status: Optional[str] = Query(None, description="状态过滤（逗号分隔）"),
    source: Optional[str] = Query(None, description="来源过滤"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量")
):
    """
    获取告警列表
    
    Args:
        severity: 严重程度过滤（如：critical,error）
        status: 状态过滤（如：active,acknowledged）
        source: 来源过滤
        start_time: 开始时间
        end_time: 结束时间
        page: 页码
        page_size: 每页数量
        
    Returns:
        告警列表和统计信息
    """
    try:
        alert_manager = get_alert_manager()
        
        # 解析过滤条件
        severity_filter = None
        if severity:
            severity_filter = [AlertSeverity(s) for s in severity.split(",") if s]
        
        status_filter = None
        if status:
            status_filter = [AlertStatus(s) for s in status.split(",") if s]
        
        # 获取告警
        if status_filter and AlertStatus("active") in status_filter:
            alerts_data = alert_manager.get_active_alerts(
                severity=severity_filter[0] if severity_filter and len(severity_filter) == 1 else None,
                source=source
            )
        else:
            alerts_data = alert_manager.get_alert_history(
                start_time=start_time,
                end_time=end_time
            )
        
        # 转换为响应格式
        alerts = []
        for alert in alerts_data:
            duration = None
            if alert.timestamp:
                end = alert.resolved_at or datetime.now()
                duration = (end - alert.timestamp).total_seconds()
            
            alerts.append(AlertResponse(
                id=alert.alert_id,
                title=alert.title,
                message=alert.message,
                severity=alert.severity.value,
                status=alert.status.value,
                source=alert.source,
                metric_name=alert.metric_name,
                metric_value=alert.metric_value,
                threshold=alert.threshold,
                timestamp=alert.timestamp,
                acknowledged_by=alert.acknowledged_by,
                acknowledged_at=alert.acknowledged_at,
                resolved_at=alert.resolved_at,
                duration_seconds=duration
            ))
        
        # 分页
        total = len(alerts)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_alerts = alerts[start_idx:end_idx]
        
        # 统计
        by_severity = {s.value: 0 for s in AlertSeverity}
        by_status = {s.value: 0 for s in AlertStatus}
        
        for alert in alerts_data:
            by_severity[alert.severity.value] += 1
            by_status[alert.status.value] += 1
        
        return AlertListResponse(
            alerts=paginated_alerts,
            total=total,
            by_severity=by_severity,
            by_status=by_status,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"获取告警列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=AlertStatisticsResponse)
async def get_alert_statistics():
    """
    获取告警统计信息
    
    Returns:
        告警统计
    """
    try:
        alert_manager = get_alert_manager()
        stats = alert_manager.get_statistics()
        
        return AlertStatisticsResponse(
            total_rules=stats.get("total_rules", 0),
            enabled_rules=stats.get("enabled_rules", 0),
            active_alerts=stats.get("active_alerts", 0),
            total_alerts_history=stats.get("total_alerts_history", 0),
            alerts_by_severity=stats.get("alerts_by_severity", {}),
            suppressed_metrics=stats.get("suppressed_metrics", [])
        )
        
    except Exception as e:
        logger.error(f"获取告警统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert_detail(alert_id: str):
    """
    获取告警详情
    
    Args:
        alert_id: 告警ID
        
    Returns:
        告警详情
    """
    try:
        alert_manager = get_alert_manager()
        
        # 查找告警
        alert = None
        for a in alert_manager._alerts.values():
            if a.alert_id == alert_id:
                alert = a
                break
        
        if alert is None:
            raise HTTPException(status_code=404, detail=f"告警 {alert_id} 不存在")
        
        # 计算持续时间
        duration = None
        if alert.timestamp:
            end = alert.resolved_at or datetime.now()
            duration = (end - alert.timestamp).total_seconds()
        
        return AlertResponse(
            id=alert.alert_id,
            title=alert.title,
            message=alert.message,
            severity=alert.severity.value,
            status=alert.status.value,
            source=alert.source,
            metric_name=alert.metric_name,
            metric_value=alert.metric_value,
            threshold=alert.threshold,
            timestamp=alert.timestamp,
            acknowledged_by=alert.acknowledged_by,
            acknowledged_at=alert.acknowledged_at,
            resolved_at=alert.resolved_at,
            duration_seconds=duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取告警详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/acknowledge", response_model=AcknowledgeAlertResponse)
async def acknowledge_alert(alert_id: str, request: AcknowledgeAlertRequest):
    """
    确认告警
    
    Args:
        alert_id: 告警ID
        request: 确认请求
        
    Returns:
        确认结果
    """
    try:
        alert_manager = get_alert_manager()
        
        success = alert_manager.acknowledge_alert(alert_id, request.acknowledged_by)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"告警 {alert_id} 不存在")
        
        return AcknowledgeAlertResponse(
            success=True,
            message="告警已确认",
            alert_id=alert_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"确认告警失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/resolve", response_model=ResolveAlertResponse)
async def resolve_alert(alert_id: str):
    """
    解决告警
    
    Args:
        alert_id: 告警ID
        
    Returns:
        解决结果
    """
    try:
        alert_manager = get_alert_manager()
        
        success = alert_manager.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"告警 {alert_id} 不存在")
        
        return ResolveAlertResponse(
            success=True,
            message="告警已解决",
            alert_id=alert_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"解决告警失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{alert_id}/suppress", response_model=SuppressAlertResponse)
async def suppress_alert(alert_id: str, request: SuppressAlertRequest):
    """
    抑制告警
    
    Args:
        alert_id: 告警ID
        request: 抑制请求
        
    Returns:
        抑制结果
    """
    try:
        alert_manager = get_alert_manager()
        
        # 查找告警获取metric_name
        alert = None
        for a in alert_manager._alerts.values():
            if a.alert_id == alert_id:
                alert = a
                break
        
        if alert is None:
            raise HTTPException(status_code=404, detail=f"告警 {alert_id} 不存在")
        
        if alert.metric_name:
            alert_manager.suppress_metric(alert.metric_name, request.duration_minutes)
        
        suppressed_until = datetime.now() + timedelta(minutes=request.duration_minutes)
        
        return SuppressAlertResponse(
            success=True,
            message=f"告警已抑制 {request.duration_minutes} 分钟",
            alert_id=alert_id,
            suppressed_until=suppressed_until
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"抑制告警失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 回滚管理API ============

@router.get("/rollback/decision", response_model=RollbackDecisionResponse)
async def get_rollback_decision():
    """
    获取回滚决策建议
    
    Returns:
        回滚决策
    """
    try:
        rollback_manager = get_rollback_manager()
        decision = rollback_manager.evaluate_rollback_need()
        
        return RollbackDecisionResponse(
            should_rollback=decision.should_rollback,
            trigger=decision.trigger.value if decision.trigger else None,
            confidence=decision.confidence,
            reasons=decision.reasons,
            recommended_action=decision.recommended_action,
            timestamp=decision.timestamp
        )
        
    except Exception as e:
        logger.error(f"获取回滚决策失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback/execute", response_model=ExecuteRollbackResponse)
async def execute_rollback(request: ExecuteRollbackRequest):
    """
    执行回滚
    
    Args:
        request: 回滚请求
        
    Returns:
        回滚结果
    """
    try:
        rollback_manager = get_rollback_manager()
        
        result = rollback_manager.execute_rollback(
            force=request.force,
            target_version=request.target_version
        )
        
        return ExecuteRollbackResponse(
            success=result.success,
            status=result.status.value,
            previous_version=result.previous_version,
            current_version=result.current_version,
            start_time=result.start_time or datetime.now(),
            end_time=result.end_time,
            message="回滚执行成功" if result.success else (result.error_message or "回滚执行失败")
        )
        
    except Exception as e:
        logger.error(f"执行回滚失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rollback/status", response_model=RollbackStatusResponse)
async def get_rollback_status():
    """
    获取回滚状态
    
    Returns:
        回滚状态
    """
    try:
        rollback_manager = get_rollback_manager()
        status = rollback_manager.get_status()
        
        return RollbackStatusResponse(
            model_id=status.get("model_id", ""),
            is_rollback_in_progress=status.get("is_rollback_in_progress", False),
            thresholds=status.get("thresholds", {}),
            baseline_metrics=status.get("baseline_metrics", {}),
            rollback_count=status.get("rollback_count", 0),
            backup_available=status.get("backup_available", False)
        )
        
    except Exception as e:
        logger.error(f"获取回滚状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rollback/history")
async def get_rollback_history():
    """
    获取回滚历史
    
    Returns:
        回滚历史列表
    """
    try:
        rollback_manager = get_rollback_manager()
        history = rollback_manager.get_rollback_history()
        
        return {
            "history": history,
            "total": len(history)
        }
        
    except Exception as e:
        logger.error(f"获取回滚历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ WebSocket端点 ============

@router.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """
    告警实时推送WebSocket
    
    实时推送新告警和告警状态更新
    """
    await websocket.accept()
    
    try:
        alert_manager = get_alert_manager()
        last_alert_count = len(alert_manager._alerts)
        
        while True:
            # 检查是否有新告警
            current_count = len(alert_manager._alerts)
            
            if current_count > last_alert_count:
                # 获取最新告警
                new_alerts = list(alert_manager._alerts.values())[last_alert_count:]
                
                for alert in new_alerts:
                    await websocket.send_json({
                        "type": "alert",
                        "timestamp": datetime.now().isoformat(),
                        "data": {
                            "alert_id": alert.alert_id,
                            "title": alert.title,
                            "message": alert.message,
                            "severity": alert.severity.value,
                            "status": alert.status.value,
                            "metric_name": alert.metric_name,
                            "metric_value": alert.metric_value,
                            "threshold": alert.threshold
                        }
                    })
                
                last_alert_count = current_count
            
            # 检查活跃告警状态变化
            active_alerts = alert_manager.get_active_alerts()
            for alert in active_alerts:
                await websocket.send_json({
                    "type": "alert_status_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "alert_id": alert.alert_id,
                        "status": alert.status.value,
                        "acknowledged_by": alert.acknowledged_by
                    }
                })
            
            # 等待5秒后再次检查
            import asyncio
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        logger.info("告警WebSocket连接断开")
    except Exception as e:
        logger.error(f"告警WebSocket错误: {e}")
        await websocket.close()


# ============ 辅助函数 ============

@router.on_event("startup")
async def init_default_alerts():
    """初始化默认告警规则"""
    try:
        alert_manager = get_alert_manager()
        
        from ...pipeline.monitoring.alert_manager import create_default_alert_rules
        
        for rule in create_default_alert_rules():
            alert_manager.register_rule(rule)
        
        logger.info(f"已注册 {len(create_default_alert_rules())} 个默认告警规则")
        
    except Exception as e:
        logger.error(f"初始化告警规则失败: {e}")
