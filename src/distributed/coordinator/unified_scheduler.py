"""
⚠️ 废弃文件 ⚠️

此调度器实现已迁移到 src.core.orchestration.scheduler.unified_scheduler

保留此文件仅作为历史参考，将在未来版本中删除。

新的统一调度器位置：
- src/core/orchestration/scheduler/unified_scheduler.py
- src/core/orchestration/scheduler/task_manager.py
- src/core/orchestration/scheduler/worker_manager.py

迁移说明：
- 所有调度功能已迁移到统一调度器
- 统一调度器现在位于核心编排层
- 支持数据采集、特征工程、模型训练等多种任务类型
"""

"""
unified_scheduler.py（已废弃）

统一任务调度器模块

支持多类型任务调度，根据任务类型自动路由到对应工作节点类型。
符合分布式协调器架构设计。

⚠️ 注意：此文件已废弃，请使用新的统一调度器

作者: RQA2025 Team
日期: 2026-02-16
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Any
from queue import PriorityQueue
import threading
import logging
import uuid
import time
from datetime import datetime

from src.distributed.registry import (
    get_unified_worker_registry,
    WorkerType,
    WorkerStatus
)

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """任务类型枚举"""
    FEATURE_EXTRACTION = "feature_extraction"  # 特征提取
    MODEL_TRAINING = "model_training"          # 模型训练
    DATA_COLLECTION = "data_collection"        # 数据采集
    MODEL_INFERENCE = "model_inference"        # 模型推理
    CUSTOM = "custom"                          # 自定义任务


class TaskPriority(Enum):
    """任务优先级枚举"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class Task:
    """任务数据类"""
    def __init__(self, task_id: str, task_type: TaskType, data: Dict,
                 priority: TaskPriority = TaskPriority.NORMAL,
                 metadata: Dict = None):
        self.task_id = task_id
        self.task_type = task_type
        self.data = data
        self.priority = priority
        self.metadata = metadata or {}
        self.status = "pending"
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.worker_id = None
        self.result = None
        self.error = None


class UnifiedScheduler:
    """
    统一任务调度器
    
    支持多类型任务调度，根据任务类型自动路由到对应工作节点类型。
    符合分布式协调器架构设计。
    """
    
    # 任务类型到工作节点类型的映射
    TASK_TYPE_TO_WORKER_TYPE = {
        TaskType.FEATURE_EXTRACTION: WorkerType.FEATURE_WORKER,
        TaskType.MODEL_TRAINING: WorkerType.TRAINING_EXECUTOR,
        TaskType.DATA_COLLECTION: WorkerType.DATA_COLLECTOR,
        TaskType.MODEL_INFERENCE: WorkerType.INFERENCE_WORKER,
    }
    
    def __init__(self):
        """初始化统一调度器"""
        self._registry = get_unified_worker_registry()
        self._task_queues: Dict[TaskType, PriorityQueue] = {
            task_type: PriorityQueue() for task_type in TaskType
        }
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.RLock()
        self._running = False
        self._scheduler_thread = None
        
        # 统计信息
        self._stats = {
            "total_tasks": 0,
            "pending_tasks": 0,
            "running_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "by_type": {task_type.value: {
                "total": 0,
                "pending": 0,
                "running": 0,
                "completed": 0,
                "failed": 0
            } for task_type in TaskType}
        }
        
        logger.info("✅ UnifiedScheduler 初始化完成")
    
    def start(self) -> None:
        """启动调度器"""
        with self._lock:
            if self._running:
                logger.debug("调度器已在运行中")
                return
            
            self._running = True
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self._scheduler_thread.daemon = True
            self._scheduler_thread.start()
            
            logger.info("🚀 统一调度器已启动")
    
    def stop(self) -> None:
        """停止调度器"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=5)
            
            logger.info("🛑 统一调度器已停止")
    
    def submit_task(self, task_type: TaskType, data: Dict,
                    priority: TaskPriority = TaskPriority.NORMAL,
                    metadata: Dict = None) -> str:
        """
        提交任务到调度器
        
        Args:
            task_type: 任务类型
            data: 任务数据
            priority: 任务优先级
            metadata: 任务元数据
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        
        task = Task(task_id, task_type, data, priority, metadata)
        
        with self._lock:
            # 添加到对应类型的队列
            self._task_queues[task_type].put((-priority.value, task_id))
            self._tasks[task_id] = task
            
            # 更新统计
            self._stats["total_tasks"] += 1
            self._stats["pending_tasks"] += 1
            self._stats["by_type"][task_type.value]["total"] += 1
            self._stats["by_type"][task_type.value]["pending"] += 1
            
            logger.info(f"📋 提交任务: {task_id}, 类型: {task_type.value}, "
                       f"优先级: {priority.value}, 队列大小: {self._task_queues[task_type].qsize()}")
        
        return task_id
    
    def get_task(self, worker_id: str, worker_type: WorkerType) -> Optional[Task]:
        """
        工作节点获取任务
        
        Args:
            worker_id: 工作节点ID
            worker_type: 工作节点类型
            
        Returns:
            任务对象，如果没有可用任务则返回None
        """
        with self._lock:
            # 找到该工作节点类型可以处理的任务类型
            supported_task_types = [
                task_type for task_type, wt in self.TASK_TYPE_TO_WORKER_TYPE.items()
                if wt == worker_type
            ]
            
            # 按优先级尝试获取任务
            for task_type in supported_task_types:
                if not self._task_queues[task_type].empty():
                    _, task_id = self._task_queues[task_type].get()
                    task = self._tasks.get(task_id)
                    
                    if task and task.status == "pending":
                        task.status = "running"
                        task.started_at = datetime.now()
                        task.worker_id = worker_id
                        
                        # 更新统计
                        self._stats["pending_tasks"] -= 1
                        self._stats["running_tasks"] += 1
                        self._stats["by_type"][task_type.value]["pending"] -= 1
                        self._stats["by_type"][task_type.value]["running"] += 1
                        
                        logger.info(f"🎯 分配任务: {task_id} -> 工作节点: {worker_id}, "
                                   f"类型: {task_type.value}")
                        return task
            
            return None
    
    def complete_task(self, task_id: str, result: Any = None, error: str = None) -> None:
        """
        完成任务
        
        Args:
            task_id: 任务ID
            result: 任务结果
            error: 错误信息
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"⚠️ 完成任务失败: 任务 {task_id} 不存在")
                return
            
            task.completed_at = datetime.now()
            task.result = result
            task.error = error
            
            if error:
                task.status = "failed"
                self._stats["failed_tasks"] += 1
                self._stats["by_type"][task.task_type.value]["failed"] += 1
                logger.error(f"❌ 任务失败: {task_id}, 错误: {error}")
            else:
                task.status = "completed"
                self._stats["completed_tasks"] += 1
                self._stats["by_type"][task.task_type.value]["completed"] += 1
                logger.info(f"✅ 任务完成: {task_id}")
            
            # 更新运行中统计
            if self._stats["running_tasks"] > 0:
                self._stats["running_tasks"] -= 1
            if self._stats["by_type"][task.task_type.value]["running"] > 0:
                self._stats["by_type"][task.task_type.value]["running"] -= 1
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        with self._lock:
            return {
                "is_running": self._running,
                "total_tasks": self._stats["total_tasks"],
                "pending_tasks": self._stats["pending_tasks"],
                "running_tasks": self._stats["running_tasks"],
                "completed_tasks": self._stats["completed_tasks"],
                "failed_tasks": self._stats["failed_tasks"],
                "by_type": self._stats["by_type"],
                "queue_sizes": {
                    task_type.value: queue.qsize()
                    for task_type, queue in self._task_queues.items()
                },
                "active_workers": len(self._registry.get_available_workers())
            }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            
            return {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "status": task.status,
                "priority": task.priority.value,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "worker_id": task.worker_id,
                "error": task.error
            }
    
    def _scheduler_loop(self) -> None:
        """调度器主循环"""
        logger.info("🔄 调度器主循环已启动")
        
        while self._running:
            try:
                # 自动路由任务到可用工作节点
                self._auto_route_tasks()
                
                # 休眠一段时间
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ 调度器循环错误: {e}")
                time.sleep(5)
        
        logger.info("🛑 调度器主循环已停止")
    
    def _auto_route_tasks(self) -> None:
        """自动路由任务到工作节点"""
        # 获取所有可用工作节点ID
        available_worker_ids = self._registry.get_available_workers()
        
        for worker_id in available_worker_ids:
            # 获取工作节点信息
            worker = self._registry.get_worker(worker_id)
            if not worker:
                continue
            
            # 找到该工作节点可以处理的任务类型
            supported_task_types = [
                task_type for task_type, wt in self.TASK_TYPE_TO_WORKER_TYPE.items()
                if wt == worker.worker_type
            ]
            
            # 尝试获取任务
            for task_type in supported_task_types:
                if not self._task_queues[task_type].empty():
                    # 这里可以添加更复杂的路由逻辑
                    # 例如：负载均衡、优先级排序等
                    break


# 全局调度器实例
_unified_scheduler: Optional[UnifiedScheduler] = None


def get_unified_scheduler() -> UnifiedScheduler:
    """获取全局统一调度器实例"""
    global _unified_scheduler
    if _unified_scheduler is None:
        _unified_scheduler = UnifiedScheduler()
    return _unified_scheduler
