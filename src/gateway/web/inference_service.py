"""
推理服务
提供模型推理任务提交到统一调度器的接口
符合架构设计：所有任务通过统一调度器调度
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def submit_inference_task(
    model_id: str,
    input_data: Dict[str, Any],
    inference_type: str = "prediction",
    priority: str = "high",
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    提交模型推理任务到统一调度器（符合架构设计）
    
    Args:
        model_id: 模型ID
        input_data: 输入数据
        inference_type: 推理类型 (prediction, classification, regression, etc.)
        priority: 优先级 (low, normal, high, critical)
        metadata: 额外的元数据
        
    Returns:
        任务信息，包含 task_id 和 scheduler_task_id
    """
    try:
        # 生成任务ID
        task_id = f"inference_{int(datetime.now().timestamp())}"
        
        # 构建任务配置
        config = {
            "model_id": model_id,
            "input_data": input_data,
            "inference_type": inference_type,
            "created_at": datetime.now().isoformat()
        }
        
        # 映射优先级字符串到 TaskPriority
        priority_map = {
            "low": "LOW",
            "normal": "NORMAL",
            "high": "HIGH",
            "critical": "CRITICAL"
        }
        task_priority = priority_map.get(priority.lower(), "HIGH")
        
        # 使用统一调度器提交任务
        from src.core.orchestration.scheduler import (
            get_unified_scheduler, TaskType, TaskPriority
        )
        from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
        
        scheduler = get_unified_scheduler()
        registry = get_unified_worker_registry()
        
        # 确保调度器已启动
        if not scheduler._running:
            logger.info("🔄 统一调度器未运行，正在启动...")
            scheduler.start()
            logger.info("✅ 统一调度器已启动")
        
        # 检查推理工作节点数量
        inference_workers = registry.get_workers_by_type(WorkerType.INFERENCE_WORKER)
        logger.info(f"👷 当前推理工作节点数量: {len(inference_workers)}")
        
        # 提交推理任务
        logger.info(f"📤 正在提交推理任务到统一调度器: {task_id}, 模型: {model_id}")
        scheduler_task_id = scheduler.submit_task(
            task_type=TaskType.MODEL_INFERENCE,
            data=config,
            priority=getattr(TaskPriority, task_priority),
            metadata={
                "task_id": task_id,
                "source": "inference_service",
                **(metadata or {})
            }
        )
        
        logger.info(f"✅ 推理任务已提交到统一调度器: {task_id} (调度器ID: {scheduler_task_id})")
        
        return {
            "success": True,
            "task_id": task_id,
            "scheduler_task_id": scheduler_task_id,
            "config": config,
            "status": "submitted",
            "message": f"推理任务已提交: 模型 {model_id}"
        }
        
    except Exception as e:
        logger.error(f"❌ 提交推理任务失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "提交推理任务失败"
        }


def submit_batch_inference_task(
    model_id: str,
    batch_input_data: List[Dict[str, Any]],
    inference_type: str = "prediction",
    priority: str = "normal",
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    提交批量模型推理任务到统一调度器
    
    Args:
        model_id: 模型ID
        batch_input_data: 批量输入数据列表
        inference_type: 推理类型
        priority: 优先级
        metadata: 额外的元数据
        
    Returns:
        任务信息
    """
    try:
        # 生成任务ID
        task_id = f"batch_inference_{int(datetime.now().timestamp())}"
        
        # 构建任务配置
        config = {
            "model_id": model_id,
            "batch_input_data": batch_input_data,
            "inference_type": inference_type,
            "batch_size": len(batch_input_data),
            "created_at": datetime.now().isoformat()
        }
        
        # 映射优先级
        priority_map = {
            "low": "LOW",
            "normal": "NORMAL",
            "high": "HIGH",
            "critical": "CRITICAL"
        }
        task_priority = priority_map.get(priority.lower(), "NORMAL")
        
        # 使用统一调度器提交任务
        from src.core.orchestration.scheduler import (
            get_unified_scheduler, TaskType, TaskPriority
        )
        from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
        
        scheduler = get_unified_scheduler()
        registry = get_unified_worker_registry()
        
        # 确保调度器已启动
        if not scheduler._running:
            logger.info("🔄 统一调度器未运行，正在启动...")
            scheduler.start()
            logger.info("✅ 统一调度器已启动")
        
        # 检查推理工作节点数量
        inference_workers = registry.get_workers_by_type(WorkerType.INFERENCE_WORKER)
        logger.info(f"👷 当前推理工作节点数量: {len(inference_workers)}")
        
        # 提交批量推理任务
        logger.info(f"📤 正在提交批量推理任务到统一调度器: {task_id}, 模型: {model_id}, 批次大小: {len(batch_input_data)}")
        scheduler_task_id = scheduler.submit_task(
            task_type=TaskType.MODEL_INFERENCE,
            data=config,
            priority=getattr(TaskPriority, task_priority),
            metadata={
                "task_id": task_id,
                "source": "inference_service",
                "batch_inference": True,
                **(metadata or {})
            }
        )
        
        logger.info(f"✅ 批量推理任务已提交到统一调度器: {task_id} (调度器ID: {scheduler_task_id})")
        
        return {
            "success": True,
            "task_id": task_id,
            "scheduler_task_id": scheduler_task_id,
            "config": config,
            "status": "submitted",
            "message": f"批量推理任务已提交: 模型 {model_id}, {len(batch_input_data)} 条数据"
        }
        
    except Exception as e:
        logger.error(f"❌ 提交批量推理任务失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "提交批量推理任务失败"
        }


def get_inference_status(task_id: str) -> Dict[str, Any]:
    """
    获取推理任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务状态信息
    """
    try:
        from src.core.orchestration.scheduler import get_unified_scheduler
        
        scheduler = get_unified_scheduler()
        
        # 获取调度器统计信息
        stats = scheduler.get_scheduler_stats()
        
        return {
            "success": True,
            "task_id": task_id,
            "scheduler_stats": stats,
            "message": "任务状态已获取"
        }
        
    except Exception as e:
        logger.error(f"❌ 获取推理任务状态失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "获取任务状态失败"
        }


def get_inference_scheduler_status() -> Dict[str, Any]:
    """
    获取推理调度器状态
    
    Returns:
        调度器状态信息
    """
    try:
        from src.core.orchestration.scheduler import get_unified_scheduler
        from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
        
        scheduler = get_unified_scheduler()
        registry = get_unified_worker_registry()
        
        stats = scheduler.get_scheduler_stats()
        inference_workers = registry.get_workers_by_type(WorkerType.INFERENCE_WORKER)
        
        return {
            "success": True,
            "is_running": scheduler._running,
            "scheduler_stats": stats,
            "inference_workers_count": len(inference_workers),
            "scheduler_type": "unified_scheduler",
            "note": "使用统一调度器管理推理任务"
        }
        
    except Exception as e:
        logger.error(f"❌ 获取推理调度器状态失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "获取调度器状态失败"
        }
