"""
统一调度器模块

提供统一的任务调度管理功能，支持定时触发（Cron）和事件触发（Event）两种模式，
集成MLPipelineController实现管道自动化执行
"""

import asyncio
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .schedule_job import (
    ScheduleJob,
    JobTrigger,
    JobStatus,
    TriggerType,
    JobExecutionHistory,
    CRON_PRESETS
)


class SchedulerException(Exception):
    """
    调度器异常类
    
    调度器操作过程中发生的错误
    
    Attributes:
        message: 错误信息
        job_id: 相关任务ID
        cause: 原始异常
    """
    
    def __init__(
        self,
        message: str,
        job_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.job_id = job_id
        self.cause = cause
    
    def __str__(self) -> str:
        parts = [f"[SchedulerException] {self.message}"]
        if self.job_id:
            parts.append(f"Job: {self.job_id}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)


class UnifiedScheduler:
    """
    统一调度器
    
    管理调度任务的创建、执行、暂停、恢复和删除，支持：
    - 定时触发（Cron表达式、间隔触发）
    - 事件触发（基于事件名称和过滤条件）
    - 管道执行集成
    - 任务状态持久化
    - 并发执行控制
    
    Attributes:
        max_workers: 最大并发工作线程数
        timezone: 默认时区
        jobs: 任务字典
        running: 是否运行中
        logger: 日志记录器
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        timezone: str = "Asia/Shanghai",
        pipeline_controller: Optional[Any] = None
    ):
        """
        初始化统一调度器
        
        Args:
            max_workers: 最大并发工作线程数
            timezone: 默认时区
            pipeline_controller: ML管道控制器实例
        """
        self.max_workers = max_workers
        self.timezone = timezone
        self._pipeline_controller = pipeline_controller
        
        self._jobs: Dict[str, ScheduleJob] = {}
        self._running_jobs: Dict[str, str] = {}  # job_id -> execution_id
        self._event_handlers: Dict[str, List[str]] = {}  # event_name -> [job_ids]
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
        self._running = False
        self._shutdown = False
        
        self.logger = logging.getLogger("pipeline.scheduler.UnifiedScheduler")
        
        # 调度线程
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self) -> None:
        """
        启动调度器
        
        启动后台调度线程，开始监听和执行任务
        """
        if self._running:
            self.logger.warning("调度器已经在运行中")
            return
        
        self._running = True
        self._shutdown = False
        self._stop_event.clear()
        
        # 启动调度线程
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="UnifiedScheduler",
            daemon=True
        )
        self._scheduler_thread.start()
        
        self.logger.info(f"调度器已启动，时区: {self.timezone}，最大工作线程: {self.max_workers}")
    
    def stop(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        停止调度器
        
        Args:
            wait: 是否等待正在执行的任务完成
            timeout: 等待超时时间（秒）
        """
        if not self._running:
            return
        
        self.logger.info("正在停止调度器...")
        self._running = False
        self._shutdown = True
        self._stop_event.set()
        
        # 停止调度线程
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)
        
        # 关闭线程池
        if wait:
            self._executor.shutdown(wait=True)
        else:
            self._executor.shutdown(wait=False, cancel_futures=True)
        
        self.logger.info("调度器已停止")
    
    def _scheduler_loop(self) -> None:
        """
        调度器主循环
        
        定期检查并执行到期的定时任务
        """
        self.logger.debug("调度循环已启动")
        
        while self._running and not self._shutdown:
            try:
                now = datetime.now()
                
                with self._lock:
                    jobs_to_run = []
                    
                    for job in self._jobs.values():
                        # 检查是否可以执行
                        if not job.can_execute():
                            continue
                        
                        # 检查定时触发
                        if job.trigger.trigger_type in [TriggerType.CRON, TriggerType.INTERVAL]:
                            if job.next_run_time and job.next_run_time <= now:
                                jobs_to_run.append(job)
                        
                        # 检查一次性任务
                        elif job.trigger.trigger_type == TriggerType.ONCE:
                            if job.next_run_time and job.next_run_time <= now:
                                jobs_to_run.append(job)
                            elif job.next_run_time is None and job.execution_count == 0:
                                # 立即执行
                                jobs_to_run.append(job)
                
                # 执行任务
                for job in jobs_to_run:
                    self._execute_job(job)
                
                # 更新下次执行时间
                self._update_next_run_times()
                
                # 等待下一次检查
                self._stop_event.wait(timeout=1.0)
                
            except Exception as e:
                self.logger.error(f"调度循环异常: {e}")
                self._stop_event.wait(timeout=1.0)
        
        self.logger.debug("调度循环已结束")
    
    def _update_next_run_times(self) -> None:
        """
        更新所有任务的下次执行时间
        """
        now = datetime.now()
        
        with self._lock:
            for job in self._jobs.values():
                if not job.can_execute():
                    continue
                
                if job.trigger.trigger_type == TriggerType.CRON and job.cron_expression:
                    # 计算下次Cron执行时间
                    next_time = self._get_next_cron_time(
                        job.trigger.cron_expression,
                        job.last_run_time or now
                    )
                    job.update_next_run_time(next_time)
                
                elif job.trigger.trigger_type == TriggerType.INTERVAL and job.trigger.interval_seconds:
                    # 计算下次间隔执行时间
                    base_time = job.last_run_time or now
                    next_time = base_time + timedelta(seconds=job.trigger.interval_seconds)
                    job.update_next_run_time(next_time)
    
    def _get_next_cron_time(self, cron_expr: str, base_time: datetime) -> datetime:
        """
        计算下次Cron执行时间
        
        简化版Cron解析，支持基本格式：
        - * * * * * (分 时 日 月 周)
        
        Args:
            cron_expr: Cron表达式
            base_time: 基准时间
            
        Returns:
            下次执行时间
        """
        try:
            parts = cron_expr.split()
            if len(parts) != 5:
                # 无效表达式，默认1小时后
                return base_time + timedelta(hours=1)
            
            minute, hour, day, month, weekday = parts
            
            # 从基准时间开始查找
            candidate = base_time + timedelta(minutes=1)
            max_search = base_time + timedelta(days=366)
            
            while candidate < max_search:
                # 检查分钟
                if minute != "*" and candidate.minute != int(minute):
                    candidate += timedelta(minutes=1)
                    continue
                
                # 检查小时
                if hour != "*" and candidate.hour != int(hour):
                    candidate += timedelta(hours=1)
                    candidate = candidate.replace(minute=0)
                    continue
                
                # 检查日期
                if day != "*" and candidate.day != int(day):
                    candidate += timedelta(days=1)
                    candidate = candidate.replace(hour=0, minute=0)
                    continue
                
                # 检查月份
                if month != "*" and candidate.month != int(month):
                    # 简单处理：进入下月
                    if candidate.month == 12:
                        candidate = candidate.replace(year=candidate.year + 1, month=1)
                    else:
                        candidate = candidate.replace(month=candidate.month + 1)
                    candidate = candidate.replace(day=1, hour=0, minute=0)
                    continue
                
                # 检查星期（0=周日，6=周六）
                if weekday != "*":
                    if "-" in weekday:
                        # 范围，如 1-5
                        start, end = map(int, weekday.split("-"))
                        if not (start <= candidate.weekday() <= end):
                            candidate += timedelta(days=1)
                            candidate = candidate.replace(hour=0, minute=0)
                            continue
                    elif "," in weekday:
                        # 列表，如 1,3,5
                        days = [int(d) for d in weekday.split(",")]
                        if candidate.weekday() not in days:
                            candidate += timedelta(days=1)
                            candidate = candidate.replace(hour=0, minute=0)
                            continue
                    else:
                        # 单个值
                        if candidate.weekday() != int(weekday):
                            candidate += timedelta(days=1)
                            candidate = candidate.replace(hour=0, minute=0)
                            continue
                
                # 找到匹配的时间
                return candidate
            
            # 未找到，返回默认值
            return base_time + timedelta(hours=1)
            
        except Exception as e:
            self.logger.warning(f"Cron解析失败 '{cron_expr}': {e}")
            return base_time + timedelta(hours=1)
    
    def create_job(
        self,
        name: str,
        trigger: Union[JobTrigger, str],
        pipeline_config: Optional[Dict[str, Any]] = None,
        description: str = "",
        **kwargs
    ) -> ScheduleJob:
        """
        创建调度任务
        
        Args:
            name: 任务名称
            trigger: 触发器配置或Cron预设名称
            pipeline_config: 管道配置
            description: 任务描述
            **kwargs: 其他任务参数
            
        Returns:
            创建的ScheduleJob实例
            
        Raises:
            SchedulerException: 创建失败
        """
        try:
            # 处理字符串触发器（Cron预设）
            if isinstance(trigger, str):
                if trigger in CRON_PRESETS:
                    trigger = JobTrigger.cron(CRON_PRESETS[trigger])
                else:
                    # 尝试作为Cron表达式
                    trigger = JobTrigger.cron(trigger)
            
            job = ScheduleJob(
                job_id=str(uuid.uuid4()),
                name=name,
                description=description,
                trigger=trigger,
                pipeline_config=pipeline_config,
                **kwargs
            )
            
            # 计算初始下次执行时间
            if trigger.trigger_type == TriggerType.CRON and trigger.cron_expression:
                job.next_run_time = self._get_next_cron_time(
                    trigger.cron_expression,
                    datetime.now()
                )
            elif trigger.trigger_type == TriggerType.INTERVAL and trigger.interval_seconds:
                job.next_run_time = datetime.now() + timedelta(seconds=trigger.interval_seconds)
            elif trigger.trigger_type == TriggerType.ONCE:
                job.next_run_time = trigger.run_date or datetime.now()
            
            with self._lock:
                self._jobs[job.job_id] = job
                
                # 注册事件处理器
                if trigger.trigger_type == TriggerType.EVENT and trigger.event_name:
                    if trigger.event_name not in self._event_handlers:
                        self._event_handlers[trigger.event_name] = []
                    self._event_handlers[trigger.event_name].append(job.job_id)
            
            self.logger.info(f"创建任务成功: {name} (ID: {job.job_id})")
            return job
            
        except Exception as e:
            raise SchedulerException(
                message=f"创建任务失败: {e}",
                cause=e
            )
    
    def get_job(self, job_id: str) -> Optional[ScheduleJob]:
        """
        获取任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            ScheduleJob实例，不存在返回None
        """
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> List[ScheduleJob]:
        """
        获取所有任务
        
        Returns:
            任务列表
        """
        with self._lock:
            return list(self._jobs.values())
    
    def get_jobs_by_status(self, status: JobStatus) -> List[ScheduleJob]:
        """
        按状态获取任务
        
        Args:
            status: 任务状态
            
        Returns:
            符合条件的任务列表
        """
        with self._lock:
            return [job for job in self._jobs.values() if job.status == status]
    
    def pause_job(self, job_id: str) -> bool:
        """
        暂停任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功暂停
            
        Raises:
            SchedulerException: 任务不存在
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise SchedulerException(
                    message=f"任务不存在: {job_id}",
                    job_id=job_id
                )
            
            job.pause()
            self.logger.info(f"任务已暂停: {job.name} (ID: {job_id})")
            return True
    
    def resume_job(self, job_id: str) -> bool:
        """
        恢复任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功恢复
            
        Raises:
            SchedulerException: 任务不存在
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise SchedulerException(
                    message=f"任务不存在: {job_id}",
                    job_id=job_id
                )
            
            job.resume()
            
            # 重新计算下次执行时间
            if job.trigger.trigger_type == TriggerType.CRON and job.trigger.cron_expression:
                job.next_run_time = self._get_next_cron_time(
                    job.trigger.cron_expression,
                    datetime.now()
                )
            elif job.trigger.trigger_type == TriggerType.INTERVAL and job.trigger.interval_seconds:
                job.next_run_time = datetime.now() + timedelta(seconds=job.trigger.interval_seconds)
            
            self.logger.info(f"任务已恢复: {job.name} (ID: {job_id})")
            return True
    
    def delete_job(self, job_id: str) -> bool:
        """
        删除任务
        
        Args:
            job_id: 任务ID
            
        Returns:
            是否成功删除
            
        Raises:
            SchedulerException: 任务不存在或正在执行
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise SchedulerException(
                    message=f"任务不存在: {job_id}",
                    job_id=job_id
                )
            
            # 检查是否正在执行
            if job_id in self._running_jobs:
                raise SchedulerException(
                    message=f"任务正在执行中，无法删除: {job_id}",
                    job_id=job_id
                )
            
            # 从事件处理器中移除
            if job.trigger.trigger_type == TriggerType.EVENT and job.trigger.event_name:
                if job.trigger.event_name in self._event_handlers:
                    handlers = self._event_handlers[job.trigger.event_name]
                    if job_id in handlers:
                        handlers.remove(job_id)
            
            del self._jobs[job_id]
            self.logger.info(f"任务已删除: {job.name} (ID: {job_id})")
            return True
    
    def trigger_job_now(self, job_id: str) -> str:
        """
        立即触发任务执行
        
        Args:
            job_id: 任务ID
            
        Returns:
            执行实例ID
            
        Raises:
            SchedulerException: 任务不存在或无法执行
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise SchedulerException(
                    message=f"任务不存在: {job_id}",
                    job_id=job_id
                )
            
            if not job.can_execute():
                raise SchedulerException(
                    message=f"任务当前无法执行: {job_id}, 状态: {job.status.name}",
                    job_id=job_id
                )
        
        return self._execute_job(job)
    
    def emit_event(
        self,
        event_name: str,
        event_data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        触发事件
        
        触发所有监听该事件的任务
        
        Args:
            event_name: 事件名称
            event_data: 事件数据
            
        Returns:
            触发的执行实例ID列表
        """
        execution_ids = []
        
        with self._lock:
            job_ids = self._event_handlers.get(event_name, [])
            
            for job_id in job_ids:
                job = self._jobs.get(job_id)
                if not job or not job.can_execute():
                    continue
                
                # 检查事件过滤条件
                if job.trigger.event_filter:
                    match = True
                    for key, value in job.trigger.event_filter.items():
                        if event_data and event_data.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                # 执行任务
                try:
                    execution_id = self._execute_job(job, event_data)
                    execution_ids.append(execution_id)
                except Exception as e:
                    self.logger.error(f"事件触发任务失败 {job_id}: {e}")
        
        if execution_ids:
            self.logger.info(f"事件 '{event_name}' 触发了 {len(execution_ids)} 个任务")
        
        return execution_ids
    
    def _execute_job(
        self,
        job: ScheduleJob,
        event_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        执行任务
        
        Args:
            job: 任务实例
            event_context: 事件上下文（事件触发时使用）
            
        Returns:
            执行实例ID
        """
        execution_id = str(uuid.uuid4())
        
        with self._lock:
            self._running_jobs[job.job_id] = execution_id
        
        # 记录开始执行
        job.record_execution(execution_id, JobStatus.RUNNING)
        
        self.logger.info(f"开始执行任务: {job.name} (Execution: {execution_id})")
        
        # 在线程池中执行
        self._executor.submit(
            self._run_job_wrapper,
            job,
            execution_id,
            event_context
        )
        
        return execution_id
    
    def _run_job_wrapper(
        self,
        job: ScheduleJob,
        execution_id: str,
        event_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        任务执行包装器
        
        处理任务执行和异常捕获
        
        Args:
            job: 任务实例
            execution_id: 执行实例ID
            event_context: 事件上下文
        """
        try:
            # 构建执行上下文
            context = {
                "job_id": job.job_id,
                "execution_id": execution_id,
                "event_context": event_context,
                "pipeline_config": job.pipeline_config
            }
            
            # 执行管道
            result = self._execute_pipeline(job, context)
            
            # 记录成功
            job.record_execution(execution_id, JobStatus.COMPLETED, result=result)
            self.logger.info(f"任务执行成功: {job.name} (Execution: {execution_id})")
            
        except Exception as e:
            # 记录失败
            error_msg = str(e)
            job.record_execution(execution_id, JobStatus.FAILED, error_message=error_msg)
            self.logger.error(f"任务执行失败: {job.name} (Execution: {execution_id}): {e}")
            
        finally:
            # 清理运行状态
            with self._lock:
                if job.job_id in self._running_jobs:
                    del self._running_jobs[job.job_id]
    
    def _execute_pipeline(
        self,
        job: ScheduleJob,
        context: Dict[str, Any]
    ) -> Any:
        """
        执行ML管道
        
        Args:
            job: 任务实例
            context: 执行上下文
            
        Returns:
            管道执行结果
            
        Raises:
            SchedulerException: 管道执行失败
        """
        try:
            # 如果有配置管道控制器，使用它执行
            if self._pipeline_controller and job.pipeline_config:
                self.logger.debug(f"使用MLPipelineController执行管道: {job.name}")
                
                # 导入控制器
                from ..controller import MLPipelineController
                from ..config import PipelineConfig
                
                # 创建管道配置
                pipeline_config = PipelineConfig.from_dict(job.pipeline_config)
                
                # 执行管道
                result = self._pipeline_controller.execute(
                    initial_context=context,
                    pipeline_id=context.get("execution_id")
                )
                
                return result.to_dict() if hasattr(result, 'to_dict') else result
            
            # 否则执行自定义回调
            elif job.metadata.get("callback"):
                callback = job.metadata["callback"]
                if callable(callback):
                    return callback(context)
                else:
                    raise SchedulerException("回调函数不可调用")
            
            # 默认返回上下文
            else:
                self.logger.warning(f"任务 {job.name} 没有配置管道或回调")
                return context
                
        except Exception as e:
            raise SchedulerException(
                message=f"管道执行失败: {e}",
                job_id=job.job_id,
                cause=e
            )
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        获取任务状态
        
        Args:
            job_id: 任务ID
            
        Returns:
            任务状态，不存在返回None
        """
        job = self.get_job(job_id)
        return job.status if job else None
    
    def get_execution_status(self, job_id: str, execution_id: str) -> Optional[JobExecutionHistory]:
        """
        获取执行状态
        
        Args:
            job_id: 任务ID
            execution_id: 执行实例ID
            
        Returns:
            执行历史记录，不存在返回None
        """
        job = self.get_job(job_id)
        if not job:
            return None
        
        for history in job.execution_history:
            if history.execution_id == execution_id:
                return history
        
        return None
    
    def save_state(self, file_path: Union[str, Path]) -> None:
        """
        保存调度器状态到文件
        
        Args:
            file_path: 保存路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            state = {
                "version": "1.0.0",
                "saved_at": datetime.now().isoformat(),
                "timezone": self.timezone,
                "jobs": [job.to_dict() for job in self._jobs.values()]
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"调度器状态已保存: {file_path}")
    
    def load_state(self, file_path: Union[str, Path]) -> None:
        """
        从文件加载调度器状态
        
        Args:
            file_path: 状态文件路径
            
        Raises:
            SchedulerException: 加载失败
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise SchedulerException(message=f"状态文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            with self._lock:
                self._jobs.clear()
                self._event_handlers.clear()
                
                for job_data in state.get("jobs", []):
                    job = ScheduleJob.from_dict(job_data)
                    self._jobs[job.job_id] = job
                    
                    # 恢复事件处理器
                    if job.trigger.trigger_type == TriggerType.EVENT and job.trigger.event_name:
                        if job.trigger.event_name not in self._event_handlers:
                            self._event_handlers[job.trigger.event_name] = []
                        self._event_handlers[job.trigger.event_name].append(job.job_id)
            
            self.logger.info(f"调度器状态已加载: {file_path}，共 {len(self._jobs)} 个任务")
            
        except Exception as e:
            raise SchedulerException(
                message=f"加载状态失败: {e}",
                cause=e
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取调度器统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total_jobs = len(self._jobs)
            running_jobs = len(self._running_jobs)
            
            status_counts = {}
            for status in JobStatus:
                status_counts[status.name] = sum(
                    1 for job in self._jobs.values() if job.status == status
                )
            
            total_executions = sum(
                job.execution_count for job in self._jobs.values()
            )
            
            return {
                "total_jobs": total_jobs,
                "running_jobs": running_jobs,
                "status_counts": status_counts,
                "total_executions": total_executions,
                "is_running": self._running,
                "max_workers": self.max_workers,
                "timezone": self.timezone
            }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop(wait=True)
        return False
