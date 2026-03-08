"""
数据采集服务
提供数据采集任务提交到统一调度器的接口
符合架构设计：所有任务通过统一调度器调度
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def submit_data_collection_task(
    symbols: List[str],
    start_date: str,
    end_date: str,
    data_types: List[str] = None,
    priority: str = "normal",
    data_source: Optional[str] = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    提交数据采集任务到统一调度器（符合架构设计）
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        data_types: 数据类型列表，默认 ["stock"]
        priority: 优先级 (low, normal, high, critical)
        data_source: 数据源 (可选)
        metadata: 额外的元数据
        
    Returns:
        任务信息，包含 task_id 和 scheduler_task_id
    """
    try:
        # 生成任务ID
        task_id = f"data_collection_{int(datetime.now().timestamp())}"
        
        # 构建任务配置
        config = {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "data_types": data_types or ["stock"],
            "data_source": data_source,
            "created_at": datetime.now().isoformat()
        }
        
        # 映射优先级字符串到 TaskPriority
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
        
        # 检查数据采集器数量
        data_collectors = registry.get_workers_by_type(WorkerType.DATA_COLLECTOR)
        logger.info(f"👷 当前数据采集器数量: {len(data_collectors)}")
        
        # 提交数据采集任务
        logger.info(f"📤 正在提交数据采集任务到统一调度器: {task_id}")
        scheduler_task_id = scheduler.submit_task(
            task_type=TaskType.DATA_COLLECTION,
            data=config,
            priority=getattr(TaskPriority, task_priority),
            metadata={
                "task_id": task_id,
                "source": "data_collection_service",
                **(metadata or {})
            }
        )
        
        logger.info(f"✅ 数据采集任务已提交到统一调度器: {task_id} (调度器ID: {scheduler_task_id})")
        
        return {
            "success": True,
            "task_id": task_id,
            "scheduler_task_id": scheduler_task_id,
            "config": config,
            "status": "submitted",
            "message": f"数据采集任务已提交: {len(symbols)} 只股票"
        }
        
    except Exception as e:
        logger.error(f"❌ 提交数据采集任务失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "提交数据采集任务失败"
        }


def get_data_collection_status(task_id: str) -> Dict[str, Any]:
    """
    获取数据采集任务状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务状态信息
    """
    try:
        from src.core.orchestration.scheduler import get_unified_scheduler
        
        scheduler = get_unified_scheduler()
        
        # 查找任务状态
        # 注意：这里需要通过 scheduler_task_id 查找，但我们需要先找到映射
        # 简化处理：返回调度器统计信息
        stats = scheduler.get_scheduler_stats()
        
        return {
            "success": True,
            "task_id": task_id,
            "scheduler_stats": stats,
            "message": "任务状态已获取"
        }
        
    except Exception as e:
        logger.error(f"❌ 获取数据采集任务状态失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "获取任务状态失败"
        }


def get_data_collection_scheduler_status() -> Dict[str, Any]:
    """
    获取数据采集调度器状态
    
    Returns:
        调度器状态信息
    """
    try:
        from src.core.orchestration.scheduler import get_unified_scheduler
        from src.infrastructure.distributed.registry import get_unified_worker_registry, WorkerType
        
        scheduler = get_unified_scheduler()
        registry = get_unified_worker_registry()
        
        stats = scheduler.get_scheduler_stats()
        data_collectors = registry.get_workers_by_type(WorkerType.DATA_COLLECTOR)
        
        return {
            "success": True,
            "is_running": scheduler._running,
            "scheduler_stats": stats,
            "data_collectors_count": len(data_collectors),
            "scheduler_type": "unified_scheduler",
            "note": "使用统一调度器管理数据采集任务"
        }
        
    except Exception as e:
        logger.error(f"❌ 获取数据采集调度器状态失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "获取调度器状态失败"
        }
