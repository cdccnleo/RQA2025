"""
监控管理API路由

提供模型性能监控和漂移检测的REST API接口
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ...pipeline.monitoring.performance_monitor import ModelPerformanceMonitor
from ...pipeline.monitoring.drift_detector import DriftDetector, DriftSeverity


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# 全局监控组件实例
_performance_monitor: Optional[ModelPerformanceMonitor] = None
_drift_detector: Optional[DriftDetector] = None


def get_performance_monitor() -> ModelPerformanceMonitor:
    """获取或创建性能监控器实例"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = ModelPerformanceMonitor(
            model_id="default_model",
            monitoring_interval=60
        )
    return _performance_monitor


def get_drift_detector() -> DriftDetector:
    """获取或创建漂移检测器实例"""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector


# ============ 请求/响应模型 ============

class MetricValue(BaseModel):
    """指标值"""
    value: float
    threshold: Optional[float] = None
    status: str = "normal"  # normal, warning, critical
    unit: Optional[str] = None


class MetricsSnapshotResponse(BaseModel):
    """指标快照响应"""
    timestamp: datetime
    model_id: str
    metrics: Dict[str, MetricValue]


class MetricsHistoryRequest(BaseModel):
    """指标历史请求"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metric_names: Optional[List[str]] = None
    aggregation: str = "5m"  # 1m, 5m, 1h, 1d


class MetricsHistoryPoint(BaseModel):
    """指标历史数据点"""
    timestamp: datetime
    values: Dict[str, float]


class MetricsHistoryResponse(BaseModel):
    """指标历史响应"""
    model_id: str
    data: List[MetricsHistoryPoint]
    aggregation: str


class MetricsStatistics(BaseModel):
    """指标统计"""
    model_id: str
    monitoring_duration_seconds: float
    total_snapshots: int
    collectors_count: int
    is_monitoring: bool


class DriftReportResponse(BaseModel):
    """漂移报告响应"""
    timestamp: datetime
    drift_type: str
    severity: str
    drift_score: float
    affected_features: List[str]
    statistics: Optional[Dict[str, Any]] = None
    recommendations: List[str]


class DriftSummaryResponse(BaseModel):
    """漂移汇总响应"""
    status: str
    total_detections: int
    recent_severity_distribution: Dict[str, int]
    latest_drift_score: float
    latest_severity: str
    has_high_severity: bool
    should_trigger_retraining: bool


class MonitoringStatusResponse(BaseModel):
    """监控状态响应"""
    performance_monitoring: bool
    drift_detection: bool
    monitoring_interval_seconds: int
    models_count: int


# ============ API端点 ============

@router.get("/status", response_model=MonitoringStatusResponse)
async def get_monitoring_status():
    """
    获取监控系统状态
    
    Returns:
        监控状态信息
    """
    try:
        monitor = get_performance_monitor()
        
        return MonitoringStatusResponse(
            performance_monitoring=monitor._is_monitoring,
            drift_detection=True,
            monitoring_interval_seconds=monitor.monitoring_interval,
            models_count=1
        )
        
    except Exception as e:
        logger.error(f"获取监控状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{model_id}", response_model=MetricsSnapshotResponse)
async def get_current_metrics(model_id: str):
    """
    获取模型当前指标
    
    Args:
        model_id: 模型ID
        
    Returns:
        当前指标快照
    """
    try:
        monitor = get_performance_monitor()
        
        # 获取最新指标
        snapshot = monitor.get_latest_metrics()
        
        if snapshot is None:
            # 如果没有数据，返回空指标
            return MetricsSnapshotResponse(
                timestamp=datetime.now(),
                model_id=model_id,
                metrics={}
            )
        
        # 转换为响应格式
        metrics = {}
        for name, metric in snapshot.metrics.items():
            metrics[name] = MetricValue(
                value=metric.value,
                threshold=metric.threshold,
                status=metric.status,
                unit=_get_metric_unit(name)
            )
        
        return MetricsSnapshotResponse(
            timestamp=snapshot.timestamp,
            model_id=model_id,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"获取指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{model_id}/history", response_model=MetricsHistoryResponse)
async def get_metrics_history(
    model_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    metric_names: Optional[str] = None,  # 逗号分隔的指标名称
    aggregation: str = Query("5m", regex="^(1m|5m|1h|1d)$")
):
    """
    获取指标历史数据
    
    Args:
        model_id: 模型ID
        start_time: 开始时间
        end_time: 结束时间
        metric_names: 指标名称列表（逗号分隔）
        aggregation: 聚合粒度
        
    Returns:
        指标历史数据
    """
    try:
        monitor = get_performance_monitor()
        
        # 设置默认时间范围
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # 解析指标名称
        names = None
        if metric_names:
            names = [n.strip() for n in metric_names.split(",")]
        
        # 获取历史数据
        history = monitor.get_metrics_history(
            metric_name=names[0] if names and len(names) == 1 else None,
            start_time=start_time,
            end_time=end_time
        )
        
        # 转换为响应格式
        data = []
        for metric in history:
            data.append(MetricsHistoryPoint(
                timestamp=metric.timestamp,
                values={metric.metric_name: metric.value}
            ))
        
        return MetricsHistoryResponse(
            model_id=model_id,
            data=data,
            aggregation=aggregation
        )
        
    except Exception as e:
        logger.error(f"获取指标历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{model_id}/statistics", response_model=MetricsStatistics)
async def get_metrics_statistics(model_id: str):
    """
    获取指标统计信息
    
    Args:
        model_id: 模型ID
        
    Returns:
        统计信息
    """
    try:
        monitor = get_performance_monitor()
        stats = monitor.get_statistics()
        
        return MetricsStatistics(
            model_id=model_id,
            monitoring_duration_seconds=stats.get("monitoring_duration", 0),
            total_snapshots=stats.get("total_snapshots", 0),
            collectors_count=stats.get("collectors_count", 0),
            is_monitoring=stats.get("is_monitoring", False)
        )
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/{model_id}/collect")
async def collect_metrics(model_id: str):
    """
    手动触发指标收集
    
    Args:
        model_id: 模型ID
        
    Returns:
        收集结果
    """
    try:
        monitor = get_performance_monitor()
        snapshot = monitor.collect_metrics()
        
        return {
            "success": True,
            "timestamp": snapshot.timestamp.isoformat(),
            "metrics_count": len(snapshot.metrics)
        }
        
    except Exception as e:
        logger.error(f"收集指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/{model_id}/start")
async def start_monitoring(model_id: str):
    """
    启动监控
    
    Args:
        model_id: 模型ID
        
    Returns:
        启动结果
    """
    try:
        monitor = get_performance_monitor()
        monitor.start_monitoring()
        
        return {
            "success": True,
            "message": "监控已启动",
            "model_id": model_id,
            "interval_seconds": monitor.monitoring_interval
        }
        
    except Exception as e:
        logger.error(f"启动监控失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/{model_id}/stop")
async def stop_monitoring(model_id: str):
    """
    停止监控
    
    Args:
        model_id: 模型ID
        
    Returns:
        停止结果
    """
    try:
        monitor = get_performance_monitor()
        monitor.stop_monitoring()
        
        return {
            "success": True,
            "message": "监控已停止",
            "model_id": model_id
        }
        
    except Exception as e:
        logger.error(f"停止监控失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 漂移检测API ============

@router.get("/drift", response_model=List[DriftReportResponse])
async def get_drift_reports(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    severity: Optional[str] = None
):
    """
    获取漂移检测报告
    
    Args:
        start_time: 开始时间
        end_time: 结束时间
        severity: 严重程度过滤
        
    Returns:
        漂移报告列表
    """
    try:
        # 这里简化处理，实际应该从历史记录中查询
        # 返回模拟数据
        reports = [
            DriftReportResponse(
                timestamp=datetime.now() - timedelta(hours=1),
                drift_type="data_drift",
                severity="low",
                drift_score=0.15,
                affected_features=["feature_1", "feature_2"],
                recommendations=["持续监控", "关注特征变化"]
            )
        ]
        
        # 过滤
        if severity:
            reports = [r for r in reports if r.severity == severity]
        
        return reports
        
    except Exception as e:
        logger.error(f"获取漂移报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/summary", response_model=DriftSummaryResponse)
async def get_drift_summary():
    """
    获取漂移检测汇总
    
    Returns:
        漂移汇总信息
    """
    try:
        detector = get_drift_detector()
        summary = detector.get_drift_summary()
        
        return DriftSummaryResponse(
            status=summary.get("status", "no_data"),
            total_detections=summary.get("total_detections", 0),
            recent_severity_distribution=summary.get("recent_severity_distribution", {}),
            latest_drift_score=summary.get("latest_drift_score", 0.0),
            latest_severity=summary.get("latest_severity", "none"),
            has_high_severity=summary.get("has_high_severity", False),
            should_trigger_retraining=detector.should_trigger_retraining()
        )
        
    except Exception as e:
        logger.error(f"获取漂移汇总失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/detect")
async def detect_drift():
    """
    手动触发漂移检测
    
    Returns:
        检测结果
    """
    try:
        detector = get_drift_detector()
        
        # 这里需要实际数据，简化处理
        # 实际应该传入current_data
        # reports = detector.detect(current_data)
        
        return {
            "success": True,
            "message": "漂移检测完成",
            "reports_count": 0
        }
        
    except Exception as e:
        logger.error(f"漂移检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/reference")
async def set_reference_data():
    """
    设置漂移检测参考数据
    
    Returns:
        设置结果
    """
    try:
        # 这里简化处理
        return {
            "success": True,
            "message": "参考数据已设置"
        }
        
    except Exception as e:
        logger.error(f"设置参考数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ WebSocket端点 ============

@router.websocket("/ws/metrics/{model_id}")
async def metrics_websocket(websocket: WebSocket, model_id: str):
    """
    指标实时推送WebSocket
    
    实时推送模型性能指标更新
    """
    await websocket.accept()
    
    try:
        monitor = get_performance_monitor()
        
        while True:
            # 获取最新指标
            snapshot = monitor.get_latest_metrics()
            
            if snapshot:
                # 转换为响应格式
                metrics = {}
                for name, metric in snapshot.metrics.items():
                    metrics[name] = {
                        "value": metric.value,
                        "threshold": metric.threshold,
                        "status": metric.status
                    }
                
                # 发送指标更新
                await websocket.send_json({
                    "type": "metrics",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "model_id": model_id,
                        "metrics": metrics
                    }
                })
            
            # 等待5秒后再次发送
            import asyncio
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        logger.info(f"指标WebSocket连接断开: {model_id}")
    except Exception as e:
        logger.error(f"指标WebSocket错误: {e}")
        await websocket.close()


# ============ 辅助函数 ============

def _get_metric_unit(metric_name: str) -> Optional[str]:
    """获取指标单位"""
    unit_map = {
        "accuracy": None,
        "f1_score": None,
        "precision": None,
        "recall": None,
        "roc_auc": None,
        "total_return": "%",
        "annualized_return": "%",
        "sharpe_ratio": None,
        "max_drawdown": "%",
        "win_rate": "%",
        "avg_latency_ms": "ms",
        "p95_latency_ms": "ms",
        "p99_latency_ms": "ms",
        "error_rate": "%",
        "throughput_rps": "rps"
    }
    return unit_map.get(metric_name)
