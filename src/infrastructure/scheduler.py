"""
统一调度器模块

负责定时任务调度、依赖任务管理和失败重试
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
import time
import logging
import uuid


class TaskStatus(Enum):
    """任务状态"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    CANCELLED = auto()


@dataclass
class ScheduledTask:
    """计划任务"""
    task_id: str
    name: str
    func: Callable
    args: tuple
    kwargs: Dict[str, Any]
    schedule_type: str  # cron, interval, once
    schedule_config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 3
    retry_delay: int = 60
    timeout: int = 3600
    enabled: bool = True
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "schedule_type": self.schedule_type,
            "schedule_config": self.schedule_config,
            "dependencies": self.dependencies,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "enabled": self.enabled,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "status": self.status.name
        }


@dataclass
class TaskExecution:
    """任务执行记录"""
    execution_id: str
    task_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TaskStatus = TaskStatus.RUNNING
    result: Any = None
    error: Optional[str] = None
    retry_attempt: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.name,
            "error": self.error,
            "retry_attempt": self.retry_attempt
        }


class UnifiedScheduler:
    """
    统一调度器
    
    功能：
    - 定时任务调度（Cron表达式支持）
    - 依赖任务管理
    - 并发控制
    - 失败重试
    """
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger("infrastructure.scheduler")
        self._tasks: Dict[str, ScheduledTask] = {}
        self._executions: Dict[str, TaskExecution] = {}
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._worker_threads: List[threading.Thread] = []
        self._max_workers = max_workers
        self._task_queue: List[str] = []  # 待执行的任务ID队列
        self._lock = threading.Lock()
        self._execution_history: List[TaskExecution] = []
    
    def add_task(self, task: ScheduledTask) -> None:
        """添加任务"""
        with self._lock:
            self._tasks[task.task_id] = task
            
            # 计算下次执行时间
            if task.schedule_type == "interval":
                interval_seconds = task.schedule_config.get("seconds", 3600)
                task.next_run = datetime.now() + timedelta(seconds=interval_seconds)
            elif task.schedule_type == "once":
                task.next_run = task.schedule_config.get("run_at", datetime.now())
            elif task.schedule_type == "cron":
                # 简化的cron支持，实际应使用croniter库
                task.next_run = datetime.now() + timedelta(minutes=1)
        
        self.logger.info(f"添加任务: {task.name} ({task.task_id})")
    
    def remove_task(self, task_id: str) -> bool:
        """移除任务"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
        return False
    
    def enable_task(self, task_id: str) -> bool:
        """启用任务"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].enabled = True
                return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """禁用任务"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].enabled = False
                return True
        return False
    
    def start(self) -> None:
        """启动调度器"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        # 启动工作线程
        for i in range(self._max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._worker_threads.append(worker)
        
        self.logger.info(f"调度器已启动，工作线程数: {self._max_workers}")
    
    def stop(self) -> None:
        """停止调度器"""
        self._running = False
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        for worker in self._worker_threads:
            worker.join(timeout=2)
        
        self.logger.info("调度器已停止")
    
    def _scheduler_loop(self) -> None:
        """调度器主循环"""
        while self._running:
            try:
                now = datetime.now()
                
                with self._lock:
                    for task in self._tasks.values():
                        if not task.enabled:
                            continue
                        
                        if task.status == TaskStatus.RUNNING:
                            continue
                        
                        if task.next_run and now >= task.next_run:
                            # 检查依赖
                            if self._check_dependencies(task):
                                self._task_queue.append(task.task_id)
                                task.status = TaskStatus.PENDING
                                
                                # 更新下次执行时间
                                if task.schedule_type == "interval":
                                    interval_seconds = task.schedule_config.get("seconds", 3600)
                                    task.next_run = now + timedelta(seconds=interval_seconds)
                                elif task.schedule_type == "cron":
                                    task.next_run = now + timedelta(minutes=1)
                                else:
                                    task.next_run = None  # 一次性任务
                            else:
                                self.logger.warning(f"任务 {task.name} 依赖未满足，跳过")
                
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"调度器循环异常: {e}")
                time.sleep(1)
    
    def _worker_loop(self) -> None:
        """工作线程循环"""
        while self._running:
            try:
                task_id = None
                
                with self._lock:
                    if self._task_queue:
                        task_id = self._task_queue.pop(0)
                
                if task_id:
                    self._execute_task(task_id)
                else:
                    time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"工作线程异常: {e}")
                time.sleep(1)
    
    def _check_dependencies(self, task: ScheduledTask) -> bool:
        """检查任务依赖是否满足"""
        for dep_id in task.dependencies:
            if dep_id not in self._tasks:
                return False
            
            dep_task = self._tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _execute_task(self, task_id: str) -> None:
        """执行任务"""
        task = self._tasks.get(task_id)
        if not task:
            return
        
        execution_id = str(uuid.uuid4())
        execution = TaskExecution(
            execution_id=execution_id,
            task_id=task_id,
            start_time=datetime.now()
        )
        
        with self._lock:
            self._executions[execution_id] = execution
            task.status = TaskStatus.RUNNING
            task.last_run = datetime.now()
        
        self.logger.info(f"执行任务: {task.name} ({execution_id})")
        
        # 执行重试逻辑
        for attempt in range(task.retry_count + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"任务 {task.name} 第 {attempt} 次重试")
                    execution.status = TaskStatus.RETRYING
                    execution.retry_attempt = attempt
                    time.sleep(task.retry_delay)
                
                # 执行任务
                result = task.func(*task.args, **task.kwargs)
                
                execution.status = TaskStatus.COMPLETED
                execution.result = result
                execution.end_time = datetime.now()
                
                with self._lock:
                    task.status = TaskStatus.COMPLETED
                
                self.logger.info(f"任务 {task.name} 执行成功")
                break
                
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"任务 {task.name} 执行失败: {error_msg}")
                execution.error = error_msg
                
                if attempt == task.retry_count:
                    execution.status = TaskStatus.FAILED
                    execution.end_time = datetime.now()
                    
                    with self._lock:
                        task.status = TaskStatus.FAILED
        
        # 保存执行历史
        with self._lock:
            self._execution_history.append(execution)
            if execution_id in self._executions:
                del self._executions[execution_id]
    
    def run_task_now(self, task_id: str) -> Optional[str]:
        """立即执行任务"""
        with self._lock:
            if task_id in self._tasks:
                self._task_queue.append(task_id)
                return task_id
        return None
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.status if task else None
    
    def get_execution_history(self, task_id: Optional[str] = None) -> List[TaskExecution]:
        """获取执行历史"""
        with self._lock:
            history = self._execution_history.copy()
        
        if task_id:
            history = [e for e in history if e.task_id == task_id]
        
        return history
    
    def export_tasks(self) -> List[Dict[str, Any]]:
        """导出所有任务"""
        with self._lock:
            return [task.to_dict() for task in self._tasks.values()]


# 全局调度器实例
_global_scheduler: Optional[UnifiedScheduler] = None


def get_scheduler(max_workers: int = 4) -> UnifiedScheduler:
    """获取全局调度器"""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = UnifiedScheduler(max_workers)
    return _global_scheduler


def reset_scheduler() -> None:
    """重置全局调度器"""
    global _global_scheduler
    _global_scheduler = None
