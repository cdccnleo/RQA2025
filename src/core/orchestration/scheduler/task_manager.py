"""
任务管理器

负责任务的创建、状态管理、查询和历史记录
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque

from .base import Task, TaskStatus, generate_task_id


class TaskManager:
    """
    任务管理器
    
    管理任务的生命周期，包括创建、状态更新、查询和历史记录
    """
    
    def __init__(self, max_history: int = 1000):
        """
        初始化任务管理器
        
        Args:
            max_history: 最大历史记录数
        """
        self._tasks: Dict[str, Task] = {}
        self._task_history: deque = deque(maxlen=max_history)
        self._lock = asyncio.Lock()
        self._max_history = max_history
    
    async def create_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        timeout_seconds: Optional[int] = None,
        max_retries: int = 0,
        retry_delay_seconds: int = 0
    ) -> str:
        """
        创建新任务

        Args:
            task_type: 任务类型
            payload: 任务数据
            priority: 优先级（1-10，数字越小优先级越高）
            timeout_seconds: 任务超时时间（秒）
            max_retries: 最大重试次数
            retry_delay_seconds: 重试延迟（秒）

        Returns:
            str: 任务ID
        """
        async with self._lock:
            task_id = generate_task_id()
            created_at = datetime.now()

            # 计算截止时间
            deadline = None
            if timeout_seconds:
                deadline = created_at + timedelta(seconds=timeout_seconds)

            task = Task(
                id=task_id,
                type=task_type,
                status=TaskStatus.PENDING,
                priority=priority,
                created_at=created_at,
                payload=payload,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_count=0,
                retry_delay_seconds=retry_delay_seconds,
                deadline=deadline
            )
            self._tasks[task_id] = task
            return task_id
    
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Any = None,
        error: str = None,
        worker_id: str = None
    ) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            result: 执行结果
            error: 错误信息
            worker_id: 执行工作进程ID
        
        Returns:
            bool: 更新是否成功
        """
        async with self._lock:
            if task_id not in self._tasks:
                return False
            
            task = self._tasks[task_id]
            task.status = status
            
            if worker_id:
                task.worker_id = worker_id
            
            if status == TaskStatus.RUNNING:
                task.started_at = datetime.now()
            
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                task.completed_at = datetime.now()
                task.result = result
                task.error = error
                
                # 移动到历史记录
                self._task_history.append(task)
                del self._tasks[task_id]
            
            return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        获取任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            Optional[Task]: 任务对象，不存在则返回None
        """
        # 先在活跃任务中查找
        if task_id in self._tasks:
            return self._tasks[task_id]
        
        # 再在历史记录中查找
        for task in self._task_history:
            if task.id == task_id:
                return task
        
        return None
    
    def get_task_dict(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务字典
        
        Args:
            task_id: 任务ID
        
        Returns:
            Optional[Dict]: 任务字典，不存在则返回None
        """
        task = self.get_task(task_id)
        return task.to_dict() if task else None
    
    def get_running_tasks(self) -> List[Task]:
        """
        获取运行中任务
        
        Returns:
            List[Task]: 运行中任务列表
        """
        return [
            task for task in self._tasks.values()
            if task.status == TaskStatus.RUNNING
        ]
    
    def get_running_tasks_dict(self) -> List[Dict[str, Any]]:
        """
        获取运行中任务字典列表
        
        Returns:
            List[Dict]: 运行中任务字典列表
        """
        return [task.to_dict() for task in self.get_running_tasks()]
    
    def get_pending_tasks(self) -> List[Task]:
        """
        获取等待中任务
        
        Returns:
            List[Task]: 等待中任务列表
        """
        return [
            task for task in self._tasks.values()
            if task.status == TaskStatus.PENDING
        ]
    
    def get_pending_tasks_dict(self) -> List[Dict[str, Any]]:
        """
        获取等待中任务字典列表
        
        Returns:
            List[Dict]: 等待中任务字典列表
        """
        return [task.to_dict() for task in self.get_pending_tasks()]
    
    def get_paused_tasks(self) -> List[Task]:
        """
        获取已暂停任务
        
        Returns:
            List[Task]: 已暂停任务列表
        """
        return [
            task for task in self._tasks.values()
            if task.status == TaskStatus.PAUSED
        ]
    
    def get_completed_tasks(
        self,
        limit: int = 20,
        offset: int = 0
    ) -> List[Task]:
        """
        获取已完成任务
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
        
        Returns:
            List[Task]: 已完成任务列表
        """
        sorted_history = sorted(
            self._task_history,
            key=lambda t: t.completed_at or datetime.min,
            reverse=True
        )
        return sorted_history[offset:offset + limit]
    
    def get_completed_tasks_dict(
        self,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        获取已完成任务字典列表
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
        
        Returns:
            List[Dict]: 已完成任务字典列表
        """
        tasks = self.get_completed_tasks(limit=limit, offset=offset)
        return [task.to_dict() for task in tasks]
    
    def get_all_active_tasks(self) -> List[Task]:
        """
        获取所有活跃任务（非已完成）
        
        Returns:
            List[Task]: 活跃任务列表
        """
        return list(self._tasks.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        total_active = len(self._tasks)
        total_history = len(self._task_history)
        total = total_active + total_history
        
        running = len(self.get_running_tasks())
        pending = len(self.get_pending_tasks())
        paused = len(self.get_paused_tasks())
        
        completed = len([
            t for t in self._task_history 
            if t.status == TaskStatus.COMPLETED
        ])
        failed = len([
            t for t in self._task_history 
            if t.status == TaskStatus.FAILED
        ])
        cancelled = len([
            t for t in self._task_history 
            if t.status == TaskStatus.CANCELLED
        ])
        
        # 计算成功率
        finished = completed + failed
        success_rate = completed / finished if finished > 0 else 0
        
        # 计算平均执行时间
        execution_times = []
        for task in self._task_history:
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                execution_times.append(duration)
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "total": total,
            "active": total_active,
            "history": total_history,
            "running": running,
            "pending": pending,
            "paused": paused,
            "completed": completed,
            "failed": failed,
            "cancelled": cancelled,
            "success_rate": round(success_rate, 4),
            "avg_execution_time": round(avg_execution_time, 2)
        }
    
    def get_task_count_by_status(self) -> Dict[str, int]:
        """
        按状态获取任务数量
        
        Returns:
            Dict: 各状态任务数量
        """
        stats = self.get_statistics()
        return {
            "pending": stats["pending"],
            "running": stats["running"],
            "paused": stats["paused"],
            "completed": stats["completed"],
            "failed": stats["failed"],
            "cancelled": stats["cancelled"]
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            bool: 取消是否成功
        """
        return await self.update_task_status(
            task_id, 
            TaskStatus.CANCELLED,
            error="Cancelled by user"
        )
    
    async def pause_task(self, task_id: str) -> bool:
        """
        暂停任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            bool: 暂停是否成功
        """
        async with self._lock:
            if task_id not in self._tasks:
                return False
            
            task = self._tasks[task_id]
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PAUSED
                return True
            return False
    
    async def resume_task(self, task_id: str) -> bool:
        """
        恢复任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 恢复是否成功
        """
        async with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]
            if task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.PENDING
                return True
            return False

    async def retry_task(self, task_id: str) -> Optional[str]:
        """
        重试失败的任务

        创建一个新任务作为原任务的重试，保留原任务的payload和配置

        Args:
            task_id: 原任务ID

        Returns:
            Optional[str]: 新任务ID，如果无法重试则返回None
        """
        async with self._lock:
            # 查找原任务（可能在活跃任务或历史记录中）
            original_task = None
            if task_id in self._tasks:
                original_task = self._tasks[task_id]
            else:
                for task in self._task_history:
                    if task.id == task_id:
                        original_task = task
                        break

            if not original_task:
                return None

            # 检查是否可以重试
            if original_task.retry_count >= original_task.max_retries:
                return None

            if original_task.status != TaskStatus.FAILED:
                return None

            # 创建新任务作为重试
            new_task_id = generate_task_id()
            created_at = datetime.now()

            # 计算新的截止时间
            deadline = None
            if original_task.timeout_seconds:
                deadline = created_at + timedelta(seconds=original_task.timeout_seconds)

            # 延迟后重试
            if original_task.retry_delay_seconds > 0:
                await asyncio.sleep(original_task.retry_delay_seconds)

            new_task = Task(
                id=new_task_id,
                type=original_task.type,
                status=TaskStatus.PENDING,
                priority=original_task.priority,
                created_at=created_at,
                payload=original_task.payload.copy(),
                timeout_seconds=original_task.timeout_seconds,
                max_retries=original_task.max_retries,
                retry_count=original_task.retry_count + 1,
                retry_delay_seconds=original_task.retry_delay_seconds,
                deadline=deadline
            )

            # 在payload中记录重试信息
            new_task.payload['_retry_info'] = {
                'original_task_id': task_id,
                'retry_count': new_task.retry_count,
                'max_retries': new_task.max_retries
            }

            self._tasks[new_task_id] = new_task
            return new_task_id

    def get_timeout_tasks(self) -> List[Task]:
        """
        获取已超时的任务

        Returns:
            List[Task]: 已超时的任务列表
        """
        timeout_tasks = []
        for task in self._tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING] and task.is_timeout():
                timeout_tasks.append(task)
        return timeout_tasks

    async def mark_task_timeout(self, task_id: str) -> bool:
        """
        将任务标记为超时

        Args:
            task_id: 任务ID

        Returns:
            bool: 操作是否成功
        """
        return await self.update_task_status(
            task_id,
            TaskStatus.FAILED,
            error="Task timeout"
        )

    def get_tasks_needing_retry(self) -> List[Task]:
        """
        获取需要重试的任务列表

        Returns:
            List[Task]: 需要重试的任务列表
        """
        retry_tasks = []
        for task in self._task_history:
            if task.should_retry():
                retry_tasks.append(task)
        return retry_tasks

    def clear_history(self, before: Optional[datetime] = None):
        """
        清理历史记录
        
        Args:
            before: 清理此时间之前的历史，None则清理所有
        """
        if before is None:
            self._task_history.clear()
        else:
            self._task_history = deque(
                [t for t in self._task_history if t.completed_at and t.completed_at > before],
                maxlen=self._max_history
            )
