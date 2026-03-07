"""
任务管理器

负责任务调度、执行和状态管理。

从coordinator_core.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import threading
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from .models import DistributedTask, TaskStatus, TaskPriority, NodeInfo
from .scheduling_engine import SchedulingEngine
from .queue_engine import QueueEngine
from .priority_engine import PriorityEngine
from .load_balancer import LoadBalancer

logger = logging.getLogger(__name__)


class TaskManager:
    """
    任务管理器
    
    负责:
    1. 任务提交和取消
    2. 任务调度和分配
    3. 任务执行和监控
    4. 任务状态管理
    """
    
    def __init__(self):
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue: List[str] = []
        self.completed_tasks: List[str] = []
        
        # 组件
        self.scheduling_engine = SchedulingEngine()
        self.queue_engine = QueueEngine()
        self.priority_engine = PriorityEngine()
        self.load_balancer = LoadBalancer()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        self._lock = threading.RLock()
        
        logger.info("任务管理器初始化完成")
    
    def submit_task(self, task_type: str, data: Dict[str, Any],
                    priority: TaskPriority = TaskPriority.NORMAL,
                    timeout_seconds: int = 3600) -> str:
        """提交任务"""
        with self._lock:
            try:
                task_id = str(uuid.uuid4())
                
                task = DistributedTask(
                    task_id=task_id,
                    task_type=task_type,
                    priority=priority,
                    timeout_seconds=timeout_seconds,
                    data=data
                )
                
                self.tasks[task_id] = task
                self.task_queue.append(task_id)
                
                # 使用队列引擎添加任务
                success = self.queue_engine.enqueue_task(task)
                if not success:
                    logger.warning(f"任务 {task_id} 添加到队列失败")
                    return ""
                
                logger.info(f"任务 {task_id} ({task_type}) 已提交")
                return task_id
                
            except Exception as e:
                logger.error(f"任务提交失败: {e}")
                return ""
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            try:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    
                    if task.status == TaskStatus.RUNNING:
                        # 标记为取消
                        task.status = TaskStatus.CANCELLED
                    
                    task.completed_time = datetime.now()
                    
                    logger.info(f"任务 {task_id} 已取消")
                    return True
                    
                return False
                
            except Exception as e:
                logger.error(f"任务取消失败: {e}")
                return False
    
    def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """获取任务"""
        with self._lock:
            return self.tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'status': task.status.value,
                    'priority': task.priority.value,
                    'assigned_node': task.assigned_node,
                    'created_time': task.created_time.isoformat(),
                    'started_time': task.started_time.isoformat() if task.started_time else None,
                    'completed_time': task.completed_time.isoformat() if task.completed_time else None,
                    'result': task.result,
                    'error_message': task.error_message
                }
            return None
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    return task.result
            return None
    
    def schedule_task(self, task_id: str, available_nodes: Dict[str, NodeInfo]) -> bool:
        """调度任务到节点"""
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            # 使用调度引擎选择节点
            node_id = self.scheduling_engine.schedule_task(
                task, available_nodes, self.task_queue
            )
            
            if not node_id:
                logger.warning(f"任务 {task_id} 无法找到合适的节点")
                return False
            
            # 分配任务到节点
            task.assigned_node = node_id
            task.status = TaskStatus.RUNNING
            task.started_time = datetime.now()
            
            # 更新节点的活跃任务
            if node_id in available_nodes:
                available_nodes[node_id].active_tasks.add(task_id)
            
            logger.info(f"任务 {task_id} 已调度到节点 {node_id}")
            return True
    
    def complete_task(self, task_id: str, result: Any = None, error: str = None) -> bool:
        """完成任务"""
        with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            if error:
                task.status = TaskStatus.FAILED
                task.error_message = error
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
            
            task.completed_time = datetime.now()
            
            # 从队列移除
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            
            # 添加到完成列表
            self.completed_tasks.append(task_id)
            
            logger.info(f"任务 {task_id} 已完成，状态: {task.status.value}")
            return True
    
    def get_pending_tasks(self) -> List[DistributedTask]:
        """获取待处理任务"""
        with self._lock:
            return [
                self.tasks[task_id]
                for task_id in self.task_queue
                if self.tasks[task_id].status == TaskStatus.PENDING
            ]
    
    def get_running_tasks(self) -> List[DistributedTask]:
        """获取运行中任务"""
        with self._lock:
            return [
                task for task in self.tasks.values()
                if task.status == TaskStatus.RUNNING
            ]
    
    def get_task_stats(self) -> Dict[str, Any]:
        """获取任务统计"""
        with self._lock:
            total_tasks = len(self.tasks)
            running_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
            completed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
            
            return {
                'total_tasks': total_tasks,
                'running_tasks': running_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'pending_tasks': len(self.task_queue)
            }


__all__ = ['TaskManager']

