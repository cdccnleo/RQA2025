"""
调度器路由模块

提供统一调度器的RESTful API接口
"""

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum
import time


# ========== Pydantic Models ==========

class JobType(str, Enum):
    """任务类型枚举"""
    # 数据层任务
    DATA_COLLECTION = "data_collection"
    DATA_CLEANING = "data_cleaning"
    DATA_VALIDATION = "data_validation"

    # 特征层任务
    FEATURE_EXTRACTION = "feature_extraction"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_VALIDATION = "feature_validation"

    # 模型层任务
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_INFERENCE = "model_inference"
    MODEL_DEPLOYMENT = "model_deployment"

    # 策略层任务
    STRATEGY_BACKTEST = "strategy_backtest"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    STRATEGY_VALIDATION = "strategy_validation"

    # 信号层任务
    SIGNAL_GENERATION = "signal_generation"
    SIGNAL_FILTERING = "signal_filtering"
    SIGNAL_AGGREGATION = "signal_aggregation"

    # 交易执行层任务
    ORDER_PREPARATION = "order_preparation"
    ORDER_VALIDATION = "order_validation"
    ORDER_EXECUTION = "order_execution"
    ORDER_CONFIRMATION = "order_confirmation"

    # 风险控制层任务
    RISK_CALCULATION = "risk_calculation"
    RISK_MONITORING = "risk_monitoring"
    RISK_ALERTING = "risk_alerting"
    POSITION_LIMIT_CHECK = "position_limit_check"

    # 组合管理层任务
    PORTFOLIO_CONSTRUCTION = "portfolio_construction"
    PORTFOLIO_REBALANCING = "portfolio_rebalancing"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"

    # 传统任务类型（向后兼容）
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"


class SubmitTaskRequest(BaseModel):
    """提交任务请求模型"""
    task_type: JobType = Field(..., description="任务类型")
    payload: Dict[str, Any] = Field(default_factory=dict, description="任务数据")
    priority: int = Field(default=5, ge=1, le=10, description="优先级（1-10，数字越小优先级越高）")
    timeout_seconds: Optional[int] = Field(default=None, ge=1, description="任务超时时间（秒）")
    max_retries: int = Field(default=0, ge=0, le=10, description="最大重试次数")
    retry_delay_seconds: int = Field(default=0, ge=0, description="重试延迟（秒）")


class CreateJobRequest(BaseModel):
    """创建定时任务请求模型"""
    name: str = Field(..., description="任务名称")
    job_type: JobType = Field(..., description="任务类型")
    trigger_type: str = Field(..., description="触发器类型（interval、cron、date、once）")
    trigger_config: Dict[str, Any] = Field(default_factory=dict, description="触发器配置")
    config: Dict[str, Any] = Field(default_factory=dict, description="任务配置")
    timeout_seconds: Optional[int] = Field(default=None, ge=1, description="任务超时时间（秒）")
    max_retries: int = Field(default=0, ge=0, le=10, description="最大重试次数")


# 创建路由
router = APIRouter(
    prefix="/api/v1/data/scheduler",
    tags=["scheduler"],
    responses={404: {"description": "Not found"}}
)


def get_scheduler():
    """获取调度器实例的辅助函数"""
    from src.infrastructure.orchestration.scheduler import get_unified_scheduler
    return get_unified_scheduler()


# ========== Dashboard ==========

@router.get("/dashboard")
async def get_scheduler_dashboard():
    """
    获取调度器仪表板数据
    
    返回调度器的整体状态、工作进程、任务统计等信息
    """
    try:
        scheduler = get_scheduler()
        status = scheduler.get_status()
        stats = scheduler.get_statistics()
        
        return {
            "scheduler": {
                "is_running": scheduler.is_running(),
                "status": "running" if scheduler.is_running() else "stopped",
                "uptime_seconds": status.get("uptime_seconds", 0),
                "started_at": status.get("started_at"),
                "config": status.get("config", {})
            },
            "workers": status.get("workers", {}),
            "tasks": {
                "total": stats.get("total", 0),
                "running": stats.get("running", 0),
                "completed": stats.get("completed", 0),
                "failed": stats.get("failed", 0),
                "pending": stats.get("pending", 0),
                "paused": stats.get("paused", 0)
            },
            "statistics": {
                "success_rate": stats.get("success_rate", 0),
                "avg_execution_time": stats.get("avg_execution_time", 0)
            },
            "last_update": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取调度器仪表板数据失败: {str(e)}")


# ========== Auto Collection ==========

@router.get("/auto-collection/status")
async def get_auto_collection_status():
    """
    获取自动采集状态
    
    返回自动数据采集的当前状态和配置
    """
    try:
        scheduler = get_scheduler()
        status = scheduler.get_status()
        
        return {
            "enabled": scheduler.is_running(),
            "is_running": scheduler.is_running(),
            "status": "running" if scheduler.is_running() else "stopped",
            "config": status.get("config", {}),
            "workers": status.get("workers", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取自动采集状态失败: {str(e)}")


@router.post("/auto-collection/start")
async def start_auto_collection():
    """
    启动自动采集
    
    启动调度器开始自动数据采集
    """
    try:
        scheduler = get_scheduler()
        
        if scheduler.is_running():
            return {
                "success": True,
                "message": "自动采集已在运行中",
                "status": "running"
            }
        
        success = await scheduler.start()
        
        return {
            "success": success,
            "message": "自动采集已启动" if success else "启动失败",
            "status": "running" if success else "stopped"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动自动采集失败: {str(e)}")


@router.post("/auto-collection/stop")
async def stop_auto_collection():
    """
    停止自动采集
    
    停止调度器，暂停自动数据采集
    """
    try:
        scheduler = get_scheduler()
        
        if not scheduler.is_running():
            return {
                "success": True,
                "message": "自动采集已停止",
                "status": "stopped"
            }
        
        success = await scheduler.stop()
        
        return {
            "success": success,
            "message": "自动采集已停止" if success else "停止失败",
            "status": "stopped" if success else "running"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止自动采集失败: {str(e)}")


# ========== Tasks ==========

@router.get("/tasks/completed")
async def get_completed_tasks(
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量")
):
    """
    获取已完成任务列表
    
    返回已完成的任务列表，支持分页
    """
    try:
        scheduler = get_scheduler()
        tasks = scheduler.get_completed_tasks(limit=limit, offset=offset)
        stats = scheduler.get_statistics()
        
        return {
            "tasks": tasks,
            "total": stats.get("completed", 0),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取已完成任务列表失败: {str(e)}")


@router.get("/tasks/running")
async def get_running_tasks():
    """
    获取运行中任务列表
    
    返回当前正在执行的任务列表
    """
    try:
        scheduler = get_scheduler()
        tasks = scheduler.get_running_tasks()
        
        return {
            "tasks": tasks,
            "total": len(tasks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取运行中任务列表失败: {str(e)}")


@router.get("/tasks/{task_id}")
async def get_task_detail(task_id: str):
    """
    获取任务详情
    
    根据任务ID获取详细信息
    """
    try:
        scheduler = get_scheduler()
        task = scheduler.get_task_detail(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return task
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务详情失败: {str(e)}")


@router.post("/tasks/{task_id}/pause")
async def pause_task(task_id: str):
    """
    暂停任务
    
    暂停指定任务的执行
    """
    try:
        scheduler = get_scheduler()
        success = await scheduler.pause_task(task_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="任务不存在或无法暂停")
        
        return {
            "success": success,
            "task_id": task_id,
            "status": "paused"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"暂停任务失败: {str(e)}")


@router.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str):
    """
    恢复任务
    
    恢复已暂停的任务
    """
    try:
        scheduler = get_scheduler()
        success = await scheduler.resume_task(task_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="任务不存在或无法恢复")
        
        return {
            "success": success,
            "task_id": task_id,
            "status": "running"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢复任务失败: {str(e)}")


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """
    取消任务

    取消指定任务的执行
    """
    try:
        scheduler = get_scheduler()
        success = await scheduler.cancel_task(task_id)

        if not success:
            raise HTTPException(status_code=400, detail="任务不存在或无法取消")

        return {
            "success": success,
            "task_id": task_id,
            "status": "cancelled"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@router.post("/tasks/{task_id}/retry")
async def retry_task(task_id: str):
    """
    重试失败的任务

    手动重试指定失败的任务，创建新任务执行
    """
    try:
        scheduler = get_scheduler()
        new_task_id = await scheduler.retry_task(task_id)

        if not new_task_id:
            raise HTTPException(status_code=400, detail="任务不存在、未达到重试条件或已达到最大重试次数")

        return {
            "success": True,
            "original_task_id": task_id,
            "new_task_id": new_task_id,
            "status": "retrying"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重试任务失败: {str(e)}")


@router.post("/tasks/submit")
async def submit_task(request: SubmitTaskRequest):
    """
    提交新任务

    提交一次性任务到调度器执行，支持超时和重试配置
    """
    try:
        scheduler = get_scheduler()
        task_id = await scheduler.submit_task(
            task_type=request.task_type.value,
            payload=request.payload,
            priority=request.priority,
            timeout_seconds=request.timeout_seconds,
            max_retries=request.max_retries,
            retry_delay_seconds=request.retry_delay_seconds
        )

        return {
            "success": True,
            "task_id": task_id,
            "task_type": request.task_type.value,
            "status": "submitted",
            "timeout_seconds": request.timeout_seconds,
            "max_retries": request.max_retries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交任务失败: {str(e)}")


@router.get("/tasks/types")
async def get_task_types():
    """
    获取支持的任务类型列表

    返回所有可用的任务类型及其描述
    """
    return {
        "task_types": [
            # 数据层任务
            {"value": "data_collection", "label": "数据采集", "category": "数据层"},
            {"value": "data_cleaning", "label": "数据清洗", "category": "数据层"},
            {"value": "data_validation", "label": "数据验证", "category": "数据层"},

            # 特征层任务
            {"value": "feature_extraction", "label": "特征提取", "category": "特征层"},
            {"value": "feature_engineering", "label": "特征工程", "category": "特征层"},
            {"value": "feature_validation", "label": "特征验证", "category": "特征层"},

            # 模型层任务
            {"value": "model_training", "label": "模型训练", "category": "模型层"},
            {"value": "model_validation", "label": "模型验证", "category": "模型层"},
            {"value": "model_inference", "label": "模型推理", "category": "模型层"},
            {"value": "model_deployment", "label": "模型部署", "category": "模型层"},

            # 策略层任务
            {"value": "strategy_backtest", "label": "策略回测", "category": "策略层"},
            {"value": "strategy_optimization", "label": "策略优化", "category": "策略层"},
            {"value": "strategy_validation", "label": "策略验证", "category": "策略层"},

            # 信号层任务
            {"value": "signal_generation", "label": "信号生成", "category": "信号层"},
            {"value": "signal_filtering", "label": "信号过滤", "category": "信号层"},
            {"value": "signal_aggregation", "label": "信号聚合", "category": "信号层"},

            # 交易执行层任务
            {"value": "order_preparation", "label": "订单准备", "category": "交易执行层"},
            {"value": "order_validation", "label": "订单验证", "category": "交易执行层"},
            {"value": "order_execution", "label": "订单执行", "category": "交易执行层"},
            {"value": "order_confirmation", "label": "订单确认", "category": "交易执行层"},

            # 风险控制层任务
            {"value": "risk_calculation", "label": "风险计算", "category": "风险控制层"},
            {"value": "risk_monitoring", "label": "风险监控", "category": "风险控制层"},
            {"value": "risk_alerting", "label": "风险告警", "category": "风险控制层"},
            {"value": "position_limit_check", "label": "仓位限制检查", "category": "风险控制层"},

            # 组合管理层任务
            {"value": "portfolio_construction", "label": "组合构建", "category": "组合管理层"},
            {"value": "portfolio_rebalancing", "label": "组合再平衡", "category": "组合管理层"},
            {"value": "portfolio_analysis", "label": "组合分析", "category": "组合管理层"},

            # 兼容旧版
            {"value": "backtest", "label": "回测（兼容）", "category": "兼容"},
            {"value": "optimization", "label": "优化（兼容）", "category": "兼容"}
        ]
    }


# ========== Config ==========

@router.get("/config")
async def get_scheduler_config():
    """
    获取调度器配置

    返回当前调度器的配置信息
    """
    try:
        scheduler = get_scheduler()
        config = scheduler.get_config()

        # 添加告警配置
        alert_config = scheduler.get_alert_config()
        if alert_config:
            config['alert'] = alert_config

        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取调度器配置失败: {str(e)}")


@router.post("/config")
async def update_scheduler_config(config: Dict[str, Any]):
    """
    更新调度器配置

    更新调度器的配置参数
    """
    try:
        scheduler = get_scheduler()
        success = await scheduler.update_config(config)

        if not success:
            raise HTTPException(status_code=400, detail="配置更新失败")

        return {
            "success": success,
            "config": scheduler.get_config()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新调度器配置失败: {str(e)}")


# ========== Alert Configuration ==========

@router.get("/config/alert")
async def get_alert_config():
    """
    获取告警配置

    返回当前告警配置信息
    """
    try:
        scheduler = get_scheduler()
        config = scheduler.get_alert_config()

        if config is None:
            return {
                "enabled": False,
                "message": "告警功能未启用"
            }

        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取告警配置失败: {str(e)}")


@router.post("/config/alert")
async def update_alert_config(config: Dict[str, Any]):
    """
    更新告警配置

    更新告警配置参数
    """
    try:
        scheduler = get_scheduler()
        success = scheduler.update_alert_config(config)

        if not success:
            raise HTTPException(status_code=400, detail="告警配置更新失败，告警功能可能未启用")

        return {
            "success": True,
            "config": scheduler.get_alert_config()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新告警配置失败: {str(e)}")


@router.post("/config/alert/test")
async def test_alert():
    """
    测试告警功能

    发送测试告警消息
    """
    try:
        scheduler = get_scheduler()

        if not scheduler._alert_manager:
            raise HTTPException(status_code=400, detail="告警功能未启用")

        # 发送测试告警
        await scheduler._alert_manager.info(
            title="告警测试",
            content="这是一条测试告警消息，用于验证告警功能是否正常工作。",
            metadata={"test": True, "timestamp": datetime.now().isoformat()}
        )

        return {
            "success": True,
            "message": "测试告警已发送"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"测试告警失败: {str(e)}")


# ========== Analytics ==========

@router.get("/analytics/trends")
async def get_scheduler_trends(
    days: int = Query(7, ge=1, le=30, description="天数")
):
    """
    获取调度器趋势分析
    
    返回指定天数内的任务执行趋势
    """
    try:
        scheduler = get_scheduler()
        trends = scheduler.get_trends(days=days)
        
        return {
            "days": days,
            "trends": trends
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取调度器趋势分析失败: {str(e)}")


@router.get("/analytics/performance")
async def get_scheduler_performance():
    """
    获取调度器性能分析
    
    返回调度器的性能指标和统计数据
    """
    try:
        scheduler = get_scheduler()
        return scheduler.get_performance_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取调度器性能分析失败: {str(e)}")


@router.get("/analytics/sources")
async def get_scheduler_sources_analytics():
    """
    获取数据源分析
    
    返回各数据源的任务执行情况分析
    """
    try:
        scheduler = get_scheduler()
        sources = scheduler.get_sources_analytics()
        
        return {
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据源分析失败: {str(e)}")


# ========== Legacy API Compatibility ==========

@router.get("/status")
async def scheduler_status():
    """
    调度器状态（兼容旧API）
    
    返回调度器的运行状态
    """
from datetime import datetime
    try:
        scheduler = get_scheduler()
        status = scheduler.get_status()
        stats = scheduler.get_statistics()
        
        return {
            "service": "unified_scheduler",
            "status": "running" if scheduler.is_running() else "stopped",
            "is_running": scheduler.is_running(),
            "uptime_seconds": status.get("uptime_seconds", 0),
            "started_at": status.get("started_at"),
            "scheduler_status": {
                "workers": status.get("workers", {}),
                "tasks": stats,
                "config": status.get("config", {})
            },
            "last_update": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取调度器状态失败: {str(e)}")
