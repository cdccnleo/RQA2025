#!/usr/bin/env python3
"""
历史数据采集监控API

提供历史数据采集任务的监控、调度和管理API接口
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.core.monitoring.historical_data_monitor import (
    get_historical_data_monitor,
    HistoricalTaskStatus,
    HistoricalTaskPriority
)
from src.infrastructure.orchestration.historical_data_scheduler import (
    get_historical_data_scheduler,
    SchedulerConfig
)

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1/monitoring/historical-collection", tags=["historical-collection-monitoring"])

# 获取监控器和调度器实例
from src.gateway.api.historical_collection_websocket import get_historical_collection_websocket_manager
websocket_manager = get_historical_collection_websocket_manager()
monitor = get_historical_data_monitor(websocket_callback=websocket_manager)
scheduler = get_historical_data_scheduler()


# 请求/响应模型
class TaskCreateRequest(BaseModel):
    """创建任务请求"""
    symbol: str = Field(..., description="股票代码")
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)")
    data_types: List[str] = Field(default=["price"], description="数据类型")
    priority: str = Field(default="normal", description="优先级 (low/normal/high/urgent)")


class WorkerRegistrationRequest(BaseModel):
    """工作进程注册请求"""
    worker_id: str = Field(..., description="工作进程ID")
    host: str = Field(..., description="主机地址")
    port: int = Field(..., description="端口号")
    capabilities: List[str] = Field(default=["historical_data"], description="能力列表")
    max_concurrent: int = Field(default=2, description="最大并发数")


class SimpleWorkerRegistrationRequest(BaseModel):
    """简化工作进程注册请求"""
    worker_id: str = Field(..., description="工作进程ID")
    max_concurrent: int = Field(default=2, description="最大并发数")


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    symbol: str
    status: str
    progress: float
    records_collected: int
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    error_message: Optional[str]
    worker_id: Optional[str]
    priority: str
    retry_count: int
    metadata: Dict[str, Any]


class MonitoringStatusResponse(BaseModel):
    """监控状态响应"""
    scheduler_status: str
    scheduler_uptime: float
    active_workers: int
    active_tasks: List[Dict[str, Any]]
    queued_tasks_count: int
    recent_completed: List[Dict[str, Any]]
    stats: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class WorkerInfo(BaseModel):
    """工作进程信息"""
    worker_id: str
    active_tasks: List[str]
    max_concurrent: int
    is_active: bool
    last_heartbeat: float
    performance_stats: Dict[str, Any]


@router.get("/status", response_model=MonitoringStatusResponse)
async def get_monitoring_status():
    """
    获取历史数据采集监控状态

    Returns:
        MonitoringStatusResponse: 监控状态信息
    """
    try:
        data = monitor.get_monitoring_data()

        # 添加工作进程信息
        workers = []
        for worker_id, worker in monitor.workers.items():
            workers.append({
                'worker_id': worker.worker_id,
                'active_tasks': worker.active_tasks,
                'max_concurrent': worker.max_concurrent,
                'is_active': worker.is_active,
                'last_heartbeat': worker.last_heartbeat,
                'performance_stats': worker.performance_stats
            })

        data['workers'] = workers

        return MonitoringStatusResponse(**data)

    except Exception as e:
        logger.error(f"获取监控状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取监控状态失败: {str(e)}")


@router.post("/scheduler/stop")
async def stop_scheduler():
    """
    停止任务调度器

    Returns:
        dict: 操作结果
    """
    try:
        success = await scheduler.stop()
        if not success:
            raise HTTPException(status_code=400, detail="调度器停止失败")

        return {"success": True, "message": "调度器停止成功"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止调度器失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止调度器失败: {str(e)}")


@router.post("/tasks/create")
async def create_task(request: TaskCreateRequest):
    """
    创建历史数据采集任务

    Args:
        request: 任务创建请求

    Returns:
        dict: 任务创建结果
    """
    try:
        # 转换优先级
        priority_map = {
            "low": HistoricalTaskPriority.LOW,
            "normal": HistoricalTaskPriority.NORMAL,
            "high": HistoricalTaskPriority.HIGH,
            "urgent": HistoricalTaskPriority.URGENT
        }

        priority = priority_map.get(request.priority.lower(), HistoricalTaskPriority.NORMAL)

        # 创建任务
        task_id = monitor.create_task(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            data_types=request.data_types,
            priority=priority
        )

        return {
            "success": True,
            "task_id": task_id,
            "message": f"任务创建成功: {task_id}"
        }

    except Exception as e:
        logger.error(f"创建任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    获取任务状态

    Args:
        task_id: 任务ID

    Returns:
        TaskStatusResponse: 任务状态信息
    """
    try:
        status = monitor.get_task_status(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail="任务不存在")

        return TaskStatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


@router.get("/tasks", response_model=List[TaskStatusResponse])
async def get_all_tasks(
    status: Optional[str] = None,
    limit: int = 100
):
    """
    获取所有任务

    Args:
        status: 状态过滤器
        limit: 最大返回数量

    Returns:
        List[TaskStatusResponse]: 任务列表
    """
    try:
        # 转换状态过滤器
        status_filter = None
        if status:
            status_map = {
                "pending": HistoricalTaskStatus.PENDING,
                "running": HistoricalTaskStatus.RUNNING,
                "paused": HistoricalTaskStatus.PAUSED,
                "completed": HistoricalTaskStatus.COMPLETED,
                "failed": HistoricalTaskStatus.FAILED,
                "cancelled": HistoricalTaskStatus.CANCELLED
            }
            status_filter = status_map.get(status.lower())

        tasks = monitor.get_all_tasks(status_filter=status_filter, limit=limit)

        return [TaskStatusResponse(**task) for task in tasks]

    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """
    取消任务

    Args:
        task_id: 任务ID

    Returns:
        dict: 操作结果
    """
    try:
        success = await monitor.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=400, detail="任务无法取消")

        return {"success": True, "message": f"任务已取消: {task_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


@router.post("/tasks/{task_id}/pause")
async def pause_task(task_id: str):
    """
    暂停任务

    Args:
        task_id: 任务ID

    Returns:
        dict: 操作结果
    """
    try:
        success = await monitor.pause_task(task_id)
        if not success:
            raise HTTPException(status_code=400, detail="任务无法暂停")

        return {"success": True, "message": f"任务已暂停: {task_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"暂停任务失败: {str(e)}")


@router.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str):
    """
    恢复任务

    Args:
        task_id: 任务ID

    Returns:
        dict: 操作结果
    """
    try:
        success = await monitor.resume_task(task_id)
        if not success:
            raise HTTPException(status_code=400, detail="任务无法恢复")

        return {"success": True, "message": f"任务已恢复: {task_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"恢复任务失败: {str(e)}")


@router.post("/tasks/pause-all")
async def pause_all_tasks():
    """
    暂停所有任务

    Returns:
        dict: 操作结果
    """
    try:
        paused_count = 0
        for task_id, task in monitor.tasks.items():
            if task.status == HistoricalTaskStatus.RUNNING:
                success = await monitor.pause_task(task_id)
                if success:
                    paused_count += 1

        return {
            "success": True,
            "message": f"已暂停 {paused_count} 个任务"
        }

    except Exception as e:
        logger.error(f"暂停所有任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"暂停所有任务失败: {str(e)}")


@router.post("/tasks/resume-all")
async def resume_all_tasks():
    """
    恢复所有任务

    Returns:
        dict: 操作结果
    """
    try:
        resumed_count = 0
        for task_id, task in monitor.tasks.items():
            if task.status == HistoricalTaskStatus.PAUSED:
                success = await monitor.resume_task(task_id)
                if success:
                    resumed_count += 1

        return {
            "success": True,
            "message": f"已恢复 {resumed_count} 个任务"
        }

    except Exception as e:
        logger.error(f"恢复所有任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"恢复所有任务失败: {str(e)}")


@router.post("/tasks/clear-completed")
async def clear_completed_tasks():
    """
    清理已完成的任务

    Returns:
        dict: 操作结果
    """
    try:
        # 获取所有已完成和失败的任务
        completed_tasks = [
            task_id for task_id, task in monitor.tasks.items()
            if task.status in [HistoricalTaskStatus.COMPLETED, HistoricalTaskStatus.FAILED, HistoricalTaskStatus.CANCELLED]
        ]

        # 从监控器中移除这些任务
        for task_id in completed_tasks:
            if task_id in monitor.tasks:
                del monitor.tasks[task_id]
            if task_id in monitor.completed_tasks:
                monitor.completed_tasks.remove(task_id)
            if task_id in monitor.failed_tasks:
                monitor.failed_tasks.remove(task_id)

        return {
            "success": True,
            "message": f"已清理 {len(completed_tasks)} 个已完成任务"
        }

    except Exception as e:
        logger.error(f"清理已完成任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理已完成任务失败: {str(e)}")


@router.post("/workers/register")
async def register_worker(request: SimpleWorkerRegistrationRequest):
    """
    注册工作进程

    Args:
        request: 工作进程注册请求

    Returns:
        dict: 操作结果
    """
    try:
        success = monitor.register_worker(request.worker_id, request.max_concurrent)
        if not success:
            raise HTTPException(status_code=400, detail="工作进程已存在")

        return {"success": True, "message": f"工作进程注册成功: {request.worker_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注册工作进程失败: {e}")
        raise HTTPException(status_code=500, detail=f"注册工作进程失败: {str(e)}")


@router.post("/workers/{worker_id}/heartbeat")
async def update_worker_heartbeat(worker_id: str):
    """
    更新工作进程心跳

    Args:
        worker_id: 工作进程ID

    Returns:
        dict: 操作结果
    """
    try:
        monitor.update_worker_heartbeat(worker_id)
        return {"success": True, "message": f"心跳更新成功: {worker_id}"}

    except Exception as e:
        logger.error(f"更新心跳失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新心跳失败: {str(e)}")


@router.post("/workers/{worker_id}/unregister")
async def unregister_worker(worker_id: str):
    """
    注销工作进程

    Args:
        worker_id: 工作进程ID

    Returns:
        dict: 操作结果
    """
    try:
        success = monitor.unregister_worker(worker_id)
        if not success:
            raise HTTPException(status_code=404, detail="工作进程不存在")

        return {"success": True, "message": f"工作进程注销成功: {worker_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注销工作进程失败: {e}")
        raise HTTPException(status_code=500, detail=f"注销工作进程失败: {str(e)}")


@router.post("/tasks/{task_id}/progress")
async def update_task_progress(
    task_id: str,
    progress: float,
    records_collected: int = 0,
    error_message: Optional[str] = None
):
    """
    更新任务进度

    Args:
        task_id: 任务ID
        progress: 进度百分比 (0.0-1.0)
        records_collected: 已采集记录数
        error_message: 错误消息

    Returns:
        dict: 操作结果
    """
    try:
        monitor.update_task_progress(task_id, progress, records_collected, error_message)
        return {"success": True, "message": f"任务进度更新成功: {task_id}"}

    except Exception as e:
        logger.error(f"更新任务进度失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新任务进度失败: {str(e)}")


@router.post("/tasks/{task_id}/complete")
async def complete_task(
    task_id: str,
    records_collected: int = 0,
    error_message: Optional[str] = None
):
    """
    完成任务

    Args:
        task_id: 任务ID
        records_collected: 采集到的记录数
        error_message: 错误消息（如果失败）

    Returns:
        dict: 操作结果
    """
    try:
        monitor.complete_task(task_id, records_collected, error_message)

        status = "失败" if error_message else "成功"
        return {"success": True, "message": f"任务{status}: {task_id}"}

    except Exception as e:
        logger.error(f"完成任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"完成任务失败: {str(e)}")


@router.post("/scheduler/start")
async def start_scheduler():
    """
    启动任务调度器

    Returns:
        dict: 操作结果
    """
    try:
        success = await scheduler.start()
        if not success:
            raise HTTPException(status_code=400, detail="调度器启动失败")

        # 确保工作进程可用
        worker_available = await scheduler.ensure_worker_available()

        return {
            "success": True,
            "message": "调度器启动成功",
            "worker_available": worker_available,
            "active_workers": len([w for w in scheduler.worker_nodes.values() if w.is_active])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动调度器失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动调度器失败: {str(e)}")


@router.get("/scheduler/status")
async def get_scheduler_status():
    """
    获取调度器状态

    Returns:
        dict: 调度器状态信息
    """
    try:
        status = scheduler.get_scheduler_status()
        return status

    except Exception as e:
        logger.error(f"获取调度器状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取调度器状态失败: {str(e)}")


@router.post("/scheduler/trigger-immediate", summary="触发立即历史数据采集")
async def trigger_immediate_collection(force: bool = False):
    """
    触发立即历史数据采集

    - **force**: 是否强制执行，跳过时间窗口检查
    """
    try:
        result = await scheduler.trigger_immediate_collection(force=force)

        if not result["success"]:
            # 如果是时间窗口限制，返回400错误
            if "next_window" in result:
                raise HTTPException(status_code=400, detail=result["message"])
            # 其他错误返回500
            raise HTTPException(status_code=500, detail=result["message"])

        logger.info(f"立即采集触发成功: 创建了 {result['tasks_created']} 个任务, 强制模式: {force}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"触发立即采集失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发失败: {str(e)}")


@router.post("/scheduler/workers/register")
async def register_scheduler_worker(request: WorkerRegistrationRequest):
    """
    注册调度器工作节点

    Args:
        request: 工作节点注册请求

    Returns:
        dict: 操作结果
    """
    try:
        success = scheduler.register_worker(
            request.worker_id,
            request.host,
            request.port,
            request.capabilities,
            request.max_concurrent
        )
        if not success:
            raise HTTPException(status_code=400, detail="工作节点注册失败")

        return {"success": True, "message": f"工作节点 {request.worker_id} 注册成功"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注册工作节点失败: {e}")
        raise HTTPException(status_code=500, detail=f"注册工作节点失败: {str(e)}")


@router.post("/scheduler/workers/register-test")
async def register_scheduler_worker_test():
    """测试工作进程注册API"""
    return {"success": True, "message": "测试API工作正常"}


@router.post("/scheduler/workers/{worker_id}/unregister")
async def unregister_scheduler_worker(worker_id: str):
    """
    注销调度器工作节点

    Args:
        worker_id: 工作节点ID

    Returns:
        dict: 操作结果
    """
    try:
        success = scheduler.unregister_worker(worker_id)
        if not success:
            raise HTTPException(status_code=404, detail="工作节点不存在")

        return {"success": True, "message": f"工作节点 {worker_id} 注销成功"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"注销工作节点失败: {e}")
        raise HTTPException(status_code=500, detail=f"注销工作节点失败: {str(e)}")


@router.post("/scheduler/workers/{worker_id}/heartbeat")
async def update_scheduler_worker_heartbeat(worker_id: str):
    """
    更新调度器工作节点心跳

    Args:
        worker_id: 工作节点ID

    Returns:
        dict: 操作结果
    """
    try:
        # 更新工作进程的心跳时间
        if worker_id in scheduler.worker_nodes:
            scheduler.worker_nodes[worker_id].last_heartbeat = time.time()
            return {"success": True, "message": f"心跳更新成功: {worker_id}"}
        else:
            raise HTTPException(status_code=404, detail="工作节点不存在")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新工作节点心跳失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新心跳失败: {str(e)}")


@router.post("/scheduler/tasks/schedule")
async def schedule_task(request: TaskCreateRequest):
    """
    通过调度器调度任务

    Args:
        request: 任务创建请求

    Returns:
        dict: 任务调度结果
    """
    try:
        # 转换优先级
        priority_map = {
            "low": HistoricalTaskPriority.LOW,
            "normal": HistoricalTaskPriority.NORMAL,
            "high": HistoricalTaskPriority.HIGH,
            "urgent": HistoricalTaskPriority.URGENT
        }

        priority = priority_map.get(request.priority.lower(), HistoricalTaskPriority.NORMAL)

        # 通过调度器调度任务
        task_id = scheduler.schedule_task(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            data_types=request.data_types,
            priority=priority
        )

        return {
            "success": True,
            "task_id": task_id,
            "message": f"任务已调度: {task_id}"
        }

    except Exception as e:
        logger.error(f"调度任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"调度任务失败: {str(e)}")


@router.post("/scheduler/tasks/{task_id}/progress")
async def update_scheduler_task_progress(
    task_id: str,
    progress: float,
    records_collected: int = 0,
    error_message: Optional[str] = None
):
    """
    更新调度器任务进度

    Args:
        task_id: 任务ID
        progress: 进度百分比 (0.0-1.0)
        records_collected: 已采集记录数
        error_message: 错误消息

    Returns:
        dict: 操作结果
    """
    try:
        scheduler.update_task_progress(task_id, progress, records_collected, error_message)
        return {"success": True, "message": f"任务进度已更新: {task_id}"}

    except Exception as e:
        logger.error(f"更新任务进度失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新任务进度失败: {str(e)}")


@router.post("/scheduler/tasks/{task_id}/complete")
async def complete_scheduler_task(
    task_id: str,
    records_collected: int = 0,
    error_message: Optional[str] = None
):
    """
    完成调度器任务

    Args:
        task_id: 任务ID
        records_collected: 采集到的记录数
        error_message: 错误消息（如果失败）

    Returns:
        dict: 操作结果
    """
    try:
        scheduler.complete_task(task_id, records_collected, error_message)

        status = "失败" if error_message else "成功"
        return {"success": True, "message": f"任务{status}: {task_id}"}

    except Exception as e:
        logger.error(f"完成任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"完成任务失败: {str(e)}")


@router.get("/scheduler/workers")
async def get_scheduler_workers():
    """
    获取所有调度器工作节点

    Returns:
        dict: 工作节点列表
    """
    try:
        workers = {}
        for worker_id, worker in scheduler.worker_nodes.items():
            workers[worker_id] = {
                'worker_id': worker.worker_id,
                'host': worker.host,
                'port': worker.port,
                'capabilities': worker.capabilities,
                'max_concurrent': worker.max_concurrent,
                'active_tasks': worker.active_tasks,
                'is_active': worker.is_active,
                'last_heartbeat': worker.last_heartbeat,
                'available_slots': worker.available_slots,
                'utilization_rate': worker.utilization_rate,
                'performance_stats': worker.performance_stats
            }

        return {
            'workers': workers,
            'total_workers': len(workers),
            'active_workers': len([w for w in workers.values() if w['is_active']])
        }

    except Exception as e:
        logger.error(f"获取工作节点失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取工作节点失败: {str(e)}")


@router.get("/scheduler/tasks/queue")
async def get_scheduler_queue():
    """
    获取调度器任务队列状态

    Returns:
        dict: 队列状态
    """
    try:
        # 注意：这里无法直接获取队列内容，需要额外的跟踪
        return {
            'queue_size': scheduler.task_queue.qsize() if hasattr(scheduler.task_queue, 'qsize') else 0,
            'pending_tasks': len(scheduler.pending_tasks),
            'running_tasks': len(scheduler.running_tasks),
            'pending_task_details': list(scheduler.pending_tasks.keys()),
            'running_task_details': list(scheduler.running_tasks.keys())
        }

    except Exception as e:
        logger.error(f"获取队列状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取队列状态失败: {str(e)}")


@router.get("/report")
async def get_task_report():
    """
    获取任务报告

    Returns:
        dict: 任务执行报告
    """
    try:
        # 获取所有任务数据
        all_tasks = monitor.get_all_tasks(limit=1000)
        monitoring_data = monitor.get_monitoring_data()
        scheduler_status = scheduler.get_scheduler_status()

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_tasks": len(all_tasks),
                "completed_tasks": len([t for t in all_tasks if t["status"] == "completed"]),
                "failed_tasks": len([t for t in all_tasks if t["status"] == "failed"]),
                "running_tasks": len([t for t in all_tasks if t["status"] == "running"]),
                "pending_tasks": len([t for t in all_tasks if t["status"] == "pending"])
            },
            "performance": monitoring_data.get("performance_metrics", {}),
            "scheduler_info": scheduler_status,
            "system_info": {
                "monitor_status": monitoring_data.get("scheduler_status"),
                "monitor_uptime": monitoring_data.get("scheduler_uptime"),
                "workers_registered": scheduler_status.get("workers", {}).get("total", 0),
                "workers_active": scheduler_status.get("workers", {}).get("active", 0)
            },
            "tasks": all_tasks
        }

        return report

    except Exception as e:
        logger.error(f"生成任务报告失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成任务报告失败: {str(e)}")


# 配置管理相关模型
class TimeWindowConfig(BaseModel):
    """时间窗口配置"""
    start_hour: int = Field(..., ge=0, le=23, description="开始小时 (0-23)")
    end_hour: int = Field(..., ge=0, le=23, description="结束小时 (0-23)")
    enable_weekend: bool = Field(default=False, description="是否在周末采集")
    timezone: str = Field(default="Asia/Shanghai", description="时区")


class CollectionRuleConfig(BaseModel):
    """采集规则配置"""
    name: str = Field(..., description="规则名称")
    symbols: List[str] = Field(default_factory=list, description="股票代码列表")
    data_types: List[str] = Field(default_factory=lambda: ['price', 'volume'], description="数据类型")
    priority: str = Field(default="normal", description="优先级 (low/normal/high/urgent)")
    max_history_days: int = Field(default=365, gt=0, description="最大历史天数")
    min_interval_days: int = Field(default=7, ge=0, description="最小采集间隔（天）")
    enabled: bool = Field(default=True, description="是否启用")


class GlobalConfigUpdate(BaseModel):
    """全局配置更新"""
    enabled: Optional[bool] = Field(None, description="是否启用定期采集")
    max_daily_tasks: Optional[int] = Field(None, gt=0, description="每日最大任务数")
    batch_size: Optional[int] = Field(None, gt=0, description="批次大小")
    check_interval_minutes: Optional[int] = Field(None, gt=0, description="检查间隔（分钟）")
    time_window: Optional[TimeWindowConfig] = Field(None, description="时间窗口配置")
    collection_period_type: Optional[str] = Field(None, description="历史数据采集期类型 (quarterly/annual)")
    daily_period_days: Optional[int] = Field(None, ge=0, description="日常采集周期（天），历史轨右边界=今日-该值-1，与日常轨不重叠")
    max_history_days: Optional[int] = Field(None, gt=0, description="历史最大回溯天数，如3650表示10年")


# 导入配置管理器
from src.infrastructure.orchestration.historical_collection_config import get_historical_collection_config_manager
config_manager = get_historical_collection_config_manager()


@router.get("/config/schedule", summary="获取采集调度配置")
async def get_collection_schedule_config():
    """获取当前的采集调度配置信息"""
    try:
        schedule_info = config_manager.get_collection_schedule_info()
        return {
            "status": "success",
            "data": schedule_info
        }
    except Exception as e:
        logger.error(f"获取采集调度配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")


@router.get("/config/rules", summary="获取采集规则列表")
async def get_collection_rules():
    """获取所有采集规则"""
    try:
        rules = config_manager.get_active_rules()
        return {
            "status": "success",
            "data": {
                "total_rules": len(config_manager.config.rules),
                "active_rules": len(rules),
                "rules": [
                    {
                        "name": rule.name,
                        "symbols": rule.symbols,
                        "data_types": rule.data_types,
                        "priority": rule.priority,
                        "max_history_days": rule.max_history_days,
                        "min_interval_days": rule.min_interval_days,
                        "enabled": rule.enabled,
                        "last_collection": rule.last_collection,
                        "needs_collection": rule.needs_collection()
                    }
                    for rule in config_manager.config.rules
                ]
            }
        }
    except Exception as e:
        logger.error(f"获取采集规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取规则失败: {str(e)}")


@router.post("/config/rules", summary="添加采集规则")
async def add_collection_rule(rule: CollectionRuleConfig):
    """添加新的采集规则"""
    try:
        from src.infrastructure.orchestration.historical_collection_config import CollectionRule

        new_rule = CollectionRule(
            name=rule.name,
            symbols=rule.symbols,
            data_types=rule.data_types,
            priority=rule.priority,
            max_history_days=rule.max_history_days,
            min_interval_days=rule.min_interval_days,
            enabled=rule.enabled
        )

        config_manager.add_rule(new_rule)
        config_manager.save_config()

        logger.info(f"添加采集规则: {rule.name}")
        return {
            "status": "success",
            "message": f"采集规则 '{rule.name}' 添加成功",
            "data": {
                "rule_name": rule.name,
                "symbols_count": len(rule.symbols)
            }
        }
    except Exception as e:
        logger.error(f"添加采集规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加规则失败: {str(e)}")


@router.put("/config/rules/{rule_name}", summary="更新采集规则")
async def update_collection_rule(rule_name: str, rule: CollectionRuleConfig):
    """更新指定的采集规则"""
    try:
        # 查找现有规则
        existing_rule = None
        for r in config_manager.config.rules:
            if r.name == rule_name:
                existing_rule = r
                break

        if not existing_rule:
            raise HTTPException(status_code=404, detail=f"采集规则 '{rule_name}' 不存在")

        # 更新规则属性
        existing_rule.symbols = rule.symbols
        existing_rule.data_types = rule.data_types
        existing_rule.priority = rule.priority
        existing_rule.max_history_days = rule.max_history_days
        existing_rule.min_interval_days = rule.min_interval_days
        existing_rule.enabled = rule.enabled

        config_manager.save_config()

        logger.info(f"更新采集规则: {rule_name}")
        return {
            "status": "success",
            "message": f"采集规则 '{rule_name}' 更新成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新采集规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新规则失败: {str(e)}")


@router.delete("/config/rules/{rule_name}", summary="删除采集规则")
async def delete_collection_rule(rule_name: str):
    """删除指定的采集规则"""
    try:
        success = config_manager.remove_rule(rule_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"采集规则 '{rule_name}' 不存在")

        config_manager.save_config()

        logger.info(f"删除采集规则: {rule_name}")
        return {
            "status": "success",
            "message": f"采集规则 '{rule_name}' 删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除采集规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除规则失败: {str(e)}")


@router.put("/config/global", summary="更新全局配置")
async def update_global_config(config: GlobalConfigUpdate):
    """更新全局采集配置"""
    try:
        updated_fields = []

        # 更新全局配置
        if config.enabled is not None:
            config_manager.config.enabled = config.enabled
            updated_fields.append(f"enabled={config.enabled}")

        if config.max_daily_tasks is not None:
            config_manager.config.max_daily_tasks = config.max_daily_tasks
            updated_fields.append(f"max_daily_tasks={config.max_daily_tasks}")

        if config.batch_size is not None:
            config_manager.config.batch_size = config.batch_size
            updated_fields.append(f"batch_size={config.batch_size}")

        if config.check_interval_minutes is not None:
            config_manager.config.check_interval_minutes = config.check_interval_minutes
            updated_fields.append(f"check_interval_minutes={config.check_interval_minutes}")

        # 更新时间窗口
        if config.time_window is not None:
            config_manager.config.time_window.start_hour = config.time_window.start_hour
            config_manager.config.time_window.end_hour = config.time_window.end_hour
            config_manager.config.time_window.enable_weekend = config.time_window.enable_weekend
            config_manager.config.time_window.timezone = config.time_window.timezone
            updated_fields.append("time_window")

        # 更新采集期类型
        if config.collection_period_type is not None:
            if config_manager.set_collection_period_type(config.collection_period_type):
                updated_fields.append(f"collection_period_type={config.collection_period_type}")
            else:
                raise HTTPException(status_code=400, detail=f"无效的采集期类型: {config.collection_period_type}")

        if config.daily_period_days is not None:
            config_manager.config.daily_period_days = config.daily_period_days
            updated_fields.append(f"daily_period_days={config.daily_period_days}")

        if config.max_history_days is not None:
            config_manager.config.max_history_days = config.max_history_days
            updated_fields.append(f"max_history_days={config.max_history_days}")

        if updated_fields:
            config_manager.save_config()
            logger.info(f"更新全局配置: {', '.join(updated_fields)}")

            return {
                "status": "success",
                "message": "全局配置更新成功",
                "updated_fields": updated_fields
            }
        else:
            return {
                "status": "success",
                "message": "没有配置需要更新"
            }

    except Exception as e:
        logger.error(f"更新全局配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


@router.post("/config/reload", summary="重新加载配置")
async def reload_collection_config():
    """重新加载采集配置（从文件）"""
    try:
        # 重新创建配置管理器实例来重新加载
        global config_manager
        config_manager = get_historical_collection_config_manager()

        logger.info("采集配置重新加载完成")
        return {
            "status": "success",
            "message": "采集配置重新加载完成",
            "data": config_manager.get_collection_schedule_info()
        }
    except Exception as e:
        logger.error(f"重新加载配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"重新加载配置失败: {str(e)}")


@router.post("/config/refresh-symbols", summary="刷新股票代码列表")
async def refresh_collection_symbols():
    """从数据源配置刷新股票代码列表"""
    try:
        # 获取最新的股票代码
        symbols = config_manager.get_symbols_to_collect(refresh_from_data_sources=True)

        return {
            "status": "success",
            "message": f"股票代码列表刷新完成，共 {len(symbols)} 个股票代码",
            "data": {
                "total_symbols": len(symbols),
                "symbols_sample": symbols[:10],  # 只返回前10个作为示例
                "has_more": len(symbols) > 10
            }
        }
    except Exception as e:
        logger.error(f"刷新股票代码列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"刷新失败: {str(e)}")


@router.get("/config/symbols", summary="获取当前股票代码列表")
async def get_collection_symbols():
    """获取当前配置的股票代码列表"""
    try:
        symbols = config_manager.get_symbols_to_collect()

        return {
            "status": "success",
            "data": {
                "total_symbols": len(symbols),
                "symbols": symbols
            }
        }
    except Exception as e:
        logger.error(f"获取股票代码列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/config/status", summary="获取配置状态")
async def get_config_status():
    """获取配置文件的当前状态"""
    try:
        import os
        from pathlib import Path

        config_file = Path("config/historical_collection_config.json")
        file_exists = config_file.exists()

        status_info = {
            "config_file_exists": file_exists,
            "config_file_path": str(config_file.absolute()) if file_exists else None,
            "file_size": config_file.stat().st_size if file_exists else 0,
            "last_modified": config_file.stat().st_mtime if file_exists else None,
            "current_config": config_manager.get_collection_schedule_info()
        }

        return {
            "status": "success",
            "data": status_info
        }
    except Exception as e:
        logger.error(f"获取配置状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")