"""
独立的健康检查路由
避免在主API文件中定义复杂的健康检查逻辑
包含系统资源监控和数据采集状态检查
"""

import time
import os
import psutil
from fastapi import APIRouter

router = APIRouter()

def get_system_resources():
    """获取系统资源使用情况"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3)
        }
    except Exception as e:
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "error": str(e)
        }

def check_data_collection_status():
    """检查数据采集状态"""
    try:
        # 检查数据采集调度器状态
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler
        scheduler = get_data_collection_scheduler()

        if scheduler and scheduler.is_running():
            active_tasks = len(scheduler.active_tasks)
            pending_tasks = len(scheduler.pending_sources)

            # 如果活跃任务过多，系统可能负载过高
            if active_tasks > 5:
                return {
                    "scheduler_running": True,
                    "active_tasks": active_tasks,
                    "pending_tasks": pending_tasks,
                    "status": "high_load",
                    "warning": f"活跃采集任务过多: {active_tasks}"
                }
            elif active_tasks > 2:
                return {
                    "scheduler_running": True,
                    "active_tasks": active_tasks,
                    "pending_tasks": pending_tasks,
                    "status": "moderate_load"
                }
            else:
                return {
                    "scheduler_running": True,
                    "active_tasks": active_tasks,
                    "pending_tasks": pending_tasks,
                    "status": "normal"
                }
        else:
            return {
                "scheduler_running": False,
                "active_tasks": 0,
                "pending_tasks": 0,
                "status": "stopped"
            }

    except Exception as e:
        return {
            "scheduler_running": False,
            "error": str(e),
            "status": "error"
        }

@router.get("/health")
def health_check():
    """智能健康检查 - 考虑系统负载和数据采集状态"""
    system_resources = get_system_resources()
    collection_status = check_data_collection_status()

    # 健康检查逻辑
    is_healthy = True
    warnings = []
    details = {
        "status": "healthy",
        "service": "rqa2025-app",
        "environment": os.getenv("RQA_ENV", "unknown"),
        "timestamp": time.time(),
        "version": "1.0.0",
        "system_resources": system_resources,
        "data_collection": collection_status
    }

    # 检查CPU使用率
    if system_resources.get("cpu_percent", 0) > 90:
        is_healthy = False
        warnings.append(f"CPU使用率过高: {system_resources['cpu_percent']:.1f}%")
    elif system_resources.get("cpu_percent", 0) > 70:
        warnings.append(f"CPU使用率较高: {system_resources['cpu_percent']:.1f}%")

    # 检查内存使用率
    if system_resources.get("memory_percent", 0) > 90:
        is_healthy = False
        warnings.append(f"内存使用率过高: {system_resources['memory_percent']:.1f}%")
    elif system_resources.get("memory_percent", 0) > 80:
        warnings.append(f"内存使用率较高: {system_resources['memory_percent']:.1f}%")

    # 检查数据采集状态
    if collection_status.get("status") == "high_load":
        warnings.append(collection_status.get("warning", "数据采集负载过高"))
    elif collection_status.get("status") == "error":
        warnings.append("数据采集服务异常")

    # 根据健康状态调整响应
    if not is_healthy:
        details["status"] = "unhealthy"
        details["warnings"] = warnings
        # 对于严重问题，返回503服务不可用
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail={
            "status": "unhealthy",
            "warnings": warnings,
            "system_resources": system_resources,
            "data_collection": collection_status,
            "timestamp": time.time()
        })

    if warnings:
        details["status"] = "warning"
        details["warnings"] = warnings

    return details

@router.get("/health/simple")
def simple_health_check():
    """简化健康检查"""
    return {"status": "ok", "timestamp": time.time()}

@router.get("/ping")
def ping():
    """Ping端点"""
    return {"pong": True, "timestamp": time.time()}