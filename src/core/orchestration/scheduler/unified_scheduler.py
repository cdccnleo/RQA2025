"""
统一调度器

系统的核心调度器，整合数据采集、特征工程、模型训练等所有调度任务
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)

from .base import (
    BaseScheduler, Task, TaskStatus, Job, JobType, TriggerType,
    generate_task_id, generate_job_id
)
from .task_manager import TaskManager
from .worker_manager import WorkerManager

# 导入持久化和告警模块
try:
    from .persistence import SchedulerPersistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    SchedulerPersistence = None

try:
    from .alerting import AlertManager, AlertConfig, get_alert_manager
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False
    AlertManager = None
    AlertConfig = None
    get_alert_manager = None

# 导入事件总线集成模块
try:
    from .integration.event_bus_integration import (
        EventBusIntegration, EventDrivenTaskTrigger,
        SchedulerEventType
    )
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False
    EventBusIntegration = None
    EventDrivenTaskTrigger = None
    SchedulerEventType = None


class UnifiedScheduler(BaseScheduler):
    """
    统一调度器
    
    系统唯一的调度器实现，支持：
    - 一次性任务提交
    - 定时任务调度（Interval、Cron、Date、Once）
    - 多工作进程并发执行
    - 任务状态管理和历史记录
    - 完整的统计和分析功能
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        max_workers: int = 4,
        max_task_history: int = 1000,
        timezone: str = "Asia/Shanghai",
        enable_persistence: bool = False,
        enable_alerting: bool = False,
        alert_config: Optional[Dict[str, Any]] = None,
        enable_event_bus: bool = False,
        event_bus_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化统一调度器

        Args:
            max_workers: 最大工作进程数
            max_task_history: 最大任务历史记录数
            timezone: 时区
            enable_persistence: 是否启用数据库持久化
            enable_alerting: 是否启用告警
            alert_config: 告警配置
            enable_event_bus: 是否启用事件总线
            event_bus_config: 事件总线配置
        """
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._max_workers = max_workers
        self._timezone = timezone
        self._running = False
        self._started_at: Optional[datetime] = None

        # 初始化组件
        self._task_manager = TaskManager(max_history=max_task_history)
        self._worker_manager = WorkerManager(max_workers=max_workers)

        # 定时任务管理
        self._jobs: Dict[str, Job] = {}
        self._job_lock = asyncio.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None

        # 配置
        self._config = {
            "max_workers": max_workers,
            "max_task_history": max_task_history,
            "timezone": timezone,
            "check_interval": 1,  # 调度检查间隔（秒）
            "enable_persistence": enable_persistence,
            "enable_alerting": enable_alerting,
            "enable_event_bus": enable_event_bus
        }

        # 初始化持久化
        self._persistence = None
        if enable_persistence and PERSISTENCE_AVAILABLE:
            try:
                self._persistence = SchedulerPersistence()
                self._persistence.initialize_database()
                print("✅ 数据库持久化已启用")
            except Exception as e:
                print(f"⚠️ 数据库持久化初始化失败: {e}")

        # 初始化告警
        self._alert_manager = None
        if enable_alerting and ALERTING_AVAILABLE and get_alert_manager:
            try:
                config = None
                if alert_config:
                    config = AlertConfig(**alert_config)
                self._alert_manager = get_alert_manager(config)
                print("✅ 告警功能已启用")
            except Exception as e:
                print(f"⚠️ 告警功能初始化失败: {e}")

        # 初始化事件总线
        self._event_bus_integration = None
        self._event_driven_trigger = None
        if enable_event_bus and EVENT_BUS_AVAILABLE:
            try:
                self._event_bus_integration = EventBusIntegration(self)
                self._event_driven_trigger = EventDrivenTaskTrigger(self)
                print("✅ 事件总线集成已启用")
            except Exception as e:
                print(f"⚠️ 事件总线集成初始化失败: {e}")
    
    async def start(self) -> bool:
        """
        启动调度器
        
        Returns:
            bool: 启动是否成功
        """
        if self._running:
            return True
        
        try:
            # 启动工作进程管理器
            await self._worker_manager.start()
            
            # 启动调度循环
            self._running = True
            self._started_at = datetime.now()
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            print(f"✅ 统一调度器已启动（工作进程: {self._max_workers}）")
            return True
        
        except Exception as e:
            print(f"❌ 统一调度器启动失败: {e}")
            self._running = False
            return False
    
    async def stop(self) -> bool:
        """
        停止调度器
        
        Returns:
            bool: 停止是否成功
        """
        if not self._running:
            return True
        
        try:
            self._running = False
            
            # 停止调度循环
            if self._scheduler_task:
                self._scheduler_task.cancel()
                try:
                    await self._scheduler_task
                except asyncio.CancelledError:
                    pass
            
            # 停止工作进程管理器
            await self._worker_manager.stop()
            
            print("✅ 统一调度器已停止")
            return True
        
        except Exception as e:
            print(f"❌ 统一调度器停止失败: {e}")
            return False
    
    def is_running(self) -> bool:
        """
        检查调度器是否运行中
        
        Returns:
            bool: 是否运行中
        """
        return self._running
    
    def get_task_stats(self) -> Dict[str, Any]:
        """
        获取任务统计信息
        
        Returns:
            Dict: 任务统计信息
        """
        try:
            # 直接从TaskManager获取统计信息（同步方式）
            # 获取活跃任务
            total_active = len(self._task_manager._tasks)
            total_history = len(self._task_manager._task_history)
            total = total_active + total_history
            
            # 统计各状态任务数
            running = 0
            pending = 0
            paused = 0
            
            for task in self._task_manager._tasks.values():
                if task.status.name == "RUNNING":
                    running += 1
                elif task.status.name == "PENDING":
                    pending += 1
                elif task.status.name == "PAUSED":
                    paused += 1
            
            # 统计历史任务
            completed = 0
            failed = 0
            cancelled = 0
            
            for task in self._task_manager._task_history:
                if task.status.name == "COMPLETED":
                    completed += 1
                elif task.status.name == "FAILED":
                    failed += 1
                elif task.status.name == "CANCELLED":
                    cancelled += 1
            
            # 计算成功率
            finished = completed + failed
            success_rate = completed / finished if finished > 0 else 0
            
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
                "success_rate": success_rate,
                "avg_execution_time": 0  # 简化处理
            }
        except Exception as e:
            logger.error(f"获取任务统计失败: {e}")
            return {
                "total": 0, "active": 0, "history": 0,
                "running": 0, "pending": 0, "paused": 0,
                "completed": 0, "failed": 0, "cancelled": 0,
                "success_rate": 0, "avg_execution_time": 0
            }
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """
        获取工作节点统计信息
        
        Returns:
            Dict: 工作节点统计信息
        """
        try:
            # WorkerManager使用的是get_statistics()方法
            if hasattr(self._worker_manager, 'get_statistics'):
                stats = self._worker_manager.get_statistics()
                # 转换为统一的字段名
                return {
                    "total": stats.get('total', 0),
                    "active": stats.get('active', 0),
                    "active_workers": stats.get('active', 0),  # 兼容字段
                    "idle": stats.get('idle', 0),
                    "stopped": stats.get('stopped', 0),
                    "total_tasks_executed": stats.get('total_tasks_executed', 0),
                    "queue_size": stats.get('queue_size', 0)
                }
            else:
                # 降级：直接计算
                total = len(self._worker_manager._workers)
                active = len([w for w in self._worker_manager._workers.values() if w.status == "busy"])
                idle = len([w for w in self._worker_manager._workers.values() if w.status == "idle"])
                return {
                    "total": total,
                    "active": active,
                    "active_workers": active,
                    "idle": idle,
                    "stopped": 0,
                    "total_tasks_executed": 0,
                    "queue_size": self._worker_manager._task_queue.qsize() if hasattr(self._worker_manager, '_task_queue') else 0
                }
        except Exception as e:
            logger.error(f"获取工作节点统计失败: {e}")
            return {
                "total": 0, "active": 0, "active_workers": 0, "idle": 0, "stopped": 0,
                "total_tasks_executed": 0, "queue_size": 0
            }
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        # 启动超时监控任务
        timeout_monitor_task = asyncio.create_task(self._timeout_monitor_loop())
        # 启动自动重试任务
        retry_monitor_task = asyncio.create_task(self._retry_monitor_loop())

        try:
            while self._running:
                try:
                    # 检查并执行定时任务
                    await self._check_scheduled_jobs()

                    # 等待下一次检查
                    await asyncio.sleep(self._config["check_interval"])

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"调度循环错误: {e}")
                    await asyncio.sleep(1)
        finally:
            # 取消监控任务
            timeout_monitor_task.cancel()
            retry_monitor_task.cancel()
            try:
                await timeout_monitor_task
                await retry_monitor_task
            except asyncio.CancelledError:
                pass

    async def _timeout_monitor_loop(self):
        """
        超时监控循环

        定期检查所有运行中任务是否超时，将超时任务标记为失败
        """
        while self._running:
            try:
                # 获取所有已超时的任务
                timeout_tasks = self._task_manager.get_timeout_tasks()

                for task in timeout_tasks:
                    print(f"⏱️ 任务超时: {task.id} (类型: {task.type})")
                    await self._task_manager.mark_task_timeout(task.id)

                    # 检查是否需要自动重试
                    if task.should_retry():
                        print(f"🔄 任务将自动重试: {task.id}")
                        new_task_id = await self._task_manager.retry_task(task.id)
                        if new_task_id:
                            # 提交重试任务到工作队列
                            task_data = {
                                "id": new_task_id,
                                "type": task.type,
                                "payload": task.payload
                            }
                            self._worker_manager.submit_task(task_data)
                            await self._task_manager.update_task_status(
                                task_id=new_task_id,
                                status=TaskStatus.RUNNING
                            )

                # 每5秒检查一次超时
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"超时监控错误: {e}")
                await asyncio.sleep(5)

    async def _retry_monitor_loop(self):
        """
        自动重试监控循环

        定期检查历史记录中需要重试的任务
        """
        while self._running:
            try:
                # 获取需要重试的任务
                retry_tasks = self._task_manager.get_tasks_needing_retry()

                for task in retry_tasks:
                    print(f"🔄 自动重试任务: {task.id} (第 {task.retry_count + 1}/{task.max_retries} 次)")
                    new_task_id = await self._task_manager.retry_task(task.id)
                    if new_task_id:
                        # 提交重试任务到工作队列
                        task_data = {
                            "id": new_task_id,
                            "type": task.type,
                            "payload": task.payload
                        }
                        self._worker_manager.submit_task(task_data)
                        await self._task_manager.update_task_status(
                            task_id=new_task_id,
                            status=TaskStatus.RUNNING
                        )

                # 每10秒检查一次需要重试的任务
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"重试监控错误: {e}")
                await asyncio.sleep(10)
    
    async def _check_scheduled_jobs(self):
        """检查并执行定时任务"""
        now = datetime.now()
        
        async with self._job_lock:
            for job in self._jobs.values():
                if not job.enabled:
                    continue
                
                # 检查是否到达执行时间
                if job.next_run and now >= job.next_run:
                    # 提交任务
                    await self.submit_task(
                        task_type=job.job_type.value,
                        payload=job.config,
                        priority=5
                    )
                    
                    # 更新任务执行记录
                    job.last_run = now
                    job.run_count += 1
                    
                    # 计算下次执行时间
                    job.next_run = self._calculate_next_run(job)
                    
                    # 如果是一次性任务，禁用
                    if job.trigger_type == TriggerType.ONCE:
                        job.enabled = False
    
    def _calculate_next_run(self, job: Job) -> Optional[datetime]:
        """
        计算下次执行时间
        
        Args:
            job: 定时任务
        
        Returns:
            Optional[datetime]: 下次执行时间
        """
        if job.trigger_type == TriggerType.ONCE:
            return None
        
        elif job.trigger_type == TriggerType.INTERVAL:
            seconds = job.trigger_config.get("seconds", 60)
            return datetime.now() + timedelta(seconds=seconds)
        
        elif job.trigger_type == TriggerType.DATE:
            # 指定日期只执行一次
            return None
        
        elif job.trigger_type == TriggerType.CRON:
            # Cron表达式计算（简化实现）
            # 实际项目中可以使用 croniter 库
            seconds = job.trigger_config.get("seconds", 3600)
            return datetime.now() + timedelta(seconds=seconds)
        
        return None
    
    # ========== 任务管理 ==========
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        timeout_seconds: Optional[int] = None,
        max_retries: int = 0,
        retry_delay_seconds: int = 0
    ) -> str:
        """
        提交一次性任务

        Args:
            task_type: 任务类型
            payload: 任务数据
            priority: 优先级（1-10）
            timeout_seconds: 任务超时时间（秒）
            max_retries: 最大重试次数
            retry_delay_seconds: 重试延迟（秒）

        Returns:
            str: 任务ID
        """
        # 创建任务
        task_id = await self._task_manager.create_task(
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds
        )

        # 获取创建的任务对象
        task = self._task_manager.get_task(task_id)

        # 发布任务创建事件
        if self._event_bus_integration and task:
            await self._event_bus_integration.publish_task_created(task)

        # 提交到工作队列
        task_data = {
            "id": task_id,
            "type": task_type,
            "payload": payload
        }

        # 注册任务完成/失败回调
        self._worker_manager.register_task_callback(
            task_id,
            self._on_task_completed_or_failed
        )

        self._worker_manager.submit_task(task_data)

        # 任务状态保持为PENDING，等待工作节点分配执行
        # 工作节点开始执行时会通过回调更新状态为RUNNING
        logger.debug(f"任务 {task_id} 已提交到工作队列，等待分配执行")

        # 发布任务开始事件
        if self._event_bus_integration and task:
            await self._event_bus_integration.publish_task_started(task)

        return task_id
    
    async def create_job(
        self,
        name: str,
        job_type: str,
        trigger_type: str,
        trigger_config: Dict[str, Any],
        config: Dict[str, Any]
    ) -> str:
        """
        创建定时任务
        
        Args:
            name: 任务名称
            job_type: 任务类型
            trigger_type: 触发器类型（interval、cron、date、once）
            trigger_config: 触发器配置
            config: 任务配置
        
        Returns:
            str: 任务ID
        """
        async with self._job_lock:
            job_id = generate_job_id()
            
            job = Job(
                id=job_id,
                name=name,
                job_type=JobType(job_type),
                trigger_type=TriggerType(trigger_type),
                trigger_config=trigger_config,
                handler=None,  # 任务处理器在注册时设置
                config=config,
                enabled=True
            )
            
            # 计算首次执行时间
            job.next_run = self._calculate_next_run(job)
            
            self._jobs[job_id] = job
            
            return job_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            bool: 取消是否成功
        """
        return await self._task_manager.cancel_task(task_id)
    
    async def pause_task(self, task_id: str) -> bool:
        """
        暂停任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            bool: 暂停是否成功
        """
        return await self._task_manager.pause_task(task_id)
    
    async def resume_task(self, task_id: str) -> bool:
        """
        恢复任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 恢复是否成功
        """
        return await self._task_manager.resume_task(task_id)

    async def retry_task(self, task_id: str) -> Optional[str]:
        """
        手动重试失败的任务

        Args:
            task_id: 原失败任务ID

        Returns:
            Optional[str]: 新任务ID，如果无法重试则返回None
        """
        new_task_id = await self._task_manager.retry_task(task_id)
        if new_task_id:
            # 获取新任务信息
            task = self._task_manager.get_task(new_task_id)
            if task:
                # 提交到工作队列
                task_data = {
                    "id": new_task_id,
                    "type": task.type,
                    "payload": task.payload
                }
                self._worker_manager.submit_task(task_data)
                await self._task_manager.update_task_status(
                    task_id=new_task_id,
                    status=TaskStatus.RUNNING
                )
        return new_task_id
    
    # ========== 查询接口 ==========
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取调度器状态
        
        Returns:
            Dict: 状态信息
        """
        uptime_seconds = 0
        if self._started_at:
            uptime_seconds = (datetime.now() - self._started_at).total_seconds()
        
        return {
            "is_running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "uptime_seconds": int(uptime_seconds),
            "config": self._config,
            "workers": self._worker_manager.get_statistics(),
            "jobs": {
                "total": len(self._jobs),
                "enabled": len([j for j in self._jobs.values() if j.enabled])
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return self._task_manager.get_statistics()
    
    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """
        获取运行中任务
        
        Returns:
            List[Dict]: 运行中任务列表
        """
        return self._task_manager.get_running_tasks_dict()
    
    def get_completed_tasks(
        self,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        获取已完成任务
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
        
        Returns:
            List[Dict]: 已完成任务列表
        """
        return self._task_manager.get_completed_tasks_dict(limit=limit, offset=offset)
    
    def get_task_detail(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务详情
        
        Args:
            task_id: 任务ID
        
        Returns:
            Optional[Dict]: 任务详情
        """
        return self._task_manager.get_task_dict(task_id)
    
    # ========== 配置管理 ==========
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取配置
        
        Returns:
            Dict: 配置信息
        """
        return self._config.copy()
    
    async def update_config(self, config: Dict[str, Any]) -> bool:
        """
        更新配置
        
        Args:
            config: 新配置
        
        Returns:
            bool: 更新是否成功
        """
        try:
            self._config.update(config)
            return True
        except Exception as e:
            print(f"更新配置失败: {e}")
            return False
    
    # ========== 分析统计 ==========
    
    def get_trends(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        获取趋势分析
        
        Args:
            days: 天数
        
        Returns:
            List[Dict]: 趋势数据
        """
        # 简化实现，实际项目中可以从数据库查询历史数据
        trends = []
        now = datetime.now()
        
        for i in range(days):
            date = now - timedelta(days=i)
            trends.append({
                "date": date.strftime("%Y-%m-%d"),
                "total_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "success_rate": 0
            })
        
        return list(reversed(trends))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            Dict: 性能指标
        """
        stats = self._task_manager.get_statistics()
        worker_stats = self._worker_manager.get_statistics()
        
        return {
            "task_statistics": stats,
            "worker_statistics": worker_stats,
            "throughput": stats.get("total", 0) / (stats.get("avg_execution_time", 1) or 1),
            "efficiency": stats.get("success_rate", 0)
        }
    
    def get_sources_analytics(self) -> List[Dict[str, Any]]:
        """
        获取数据源分析
        
        Returns:
            List[Dict]: 数据源分析
        """
        # 简化实现
        return [
            {
                "source": "alpha_vantage",
                "total_tasks": 0,
                "success_rate": 0,
                "avg_execution_time": 0
            },
            {
                "source": "yahoo_finance",
                "total_tasks": 0,
                "success_rate": 0,
                "avg_execution_time": 0
            }
        ]
    
    # ========== 任务处理器注册 ==========

    def register_task_handler(self, task_type: str, handler: Callable):
        """
        注册任务处理器

        Args:
            task_type: 任务类型
            handler: 处理函数
        """
        self._worker_manager.register_task_handler(task_type, handler)

    # ========== 持久化集成 ==========

    async def _persist_task(self, task: Task):
        """持久化任务到数据库"""
        if self._persistence:
            try:
                self._persistence.tasks.save_task(task)
            except Exception as e:
                logger.error(f"持久化任务失败: {e}")

    async def _persist_job(self, job: Job):
        """持久化定时任务到数据库"""
        if self._persistence:
            try:
                self._persistence.jobs.save_job(job)
            except Exception as e:
                logger.error(f"持久化定时任务失败: {e}")

    def restore_jobs_from_db(self) -> int:
        """
        从数据库恢复定时任务

        Returns:
            int: 恢复的任务数量
        """
        if not self._persistence:
            return 0

        try:
            jobs = self._persistence.restore_jobs_from_db()
            for job in jobs:
                self._jobs[job.id] = job
            return len(jobs)
        except Exception as e:
            logger.error(f"恢复定时任务失败: {e}")
            return 0

    # ========== 告警集成 ==========

    async def _alert_task_failed(self, task: Task, error: str):
        """任务失败告警"""
        if self._alert_manager:
            try:
                await self._alert_manager.task_failed(
                    task_id=task.id,
                    task_type=task.type,
                    error=error,
                    retry_count=task.retry_count
                )
            except Exception as e:
                logger.error(f"发送任务失败告警失败: {e}")

    async def _alert_task_timeout(self, task: Task):
        """任务超时告警"""
        if self._alert_manager:
            try:
                await self._alert_manager.task_timeout(
                    task_id=task.id,
                    task_type=task.type,
                    timeout_seconds=task.timeout_seconds or 0
                )
            except Exception as e:
                logger.error(f"发送任务超时告警失败: {e}")

    async def _alert_retry_exhausted(self, task: Task):
        """重试次数耗尽告警"""
        if self._alert_manager:
            try:
                await self._alert_manager.task_retry_exhausted(
                    task_id=task.id,
                    task_type=task.type,
                    max_retries=task.max_retries
                )
            except Exception as e:
                logger.error(f"发送重试耗尽告警失败: {e}")

    async def _alert_scheduler_error(self, error: str, context: Optional[Dict[str, Any]] = None):
        """调度器错误告警"""
        if self._alert_manager:
            try:
                await self._alert_manager.scheduler_error(error, context)
            except Exception as e:
                logger.error(f"发送调度器错误告警失败: {e}")

    # ========== 告警配置管理 ==========

    def update_alert_config(self, config: Dict[str, Any]) -> bool:
        """
        更新告警配置

        Args:
            config: 告警配置字典

        Returns:
            bool: 更新是否成功
        """
        if not self._alert_manager:
            return False

        try:
            from .alerting import AlertConfig
            alert_config = AlertConfig(**config)
            self._alert_manager.update_config(alert_config)
            return True
        except Exception as e:
            logger.error(f"更新告警配置失败: {e}")
            return False

    def get_alert_config(self) -> Optional[Dict[str, Any]]:
        """
        获取当前告警配置

        Returns:
            Optional[Dict[str, Any]]: 告警配置
        """
        if not self._alert_manager:
            return None

        try:
            config = self._alert_manager._config
            return {
                'enabled': config.enabled,
                'channels': [c.value for c in config.channels],
                'level_threshold': config.level_threshold.value,
                'rate_limit_seconds': config.rate_limit_seconds
            }
        except Exception as e:
            logger.error(f"获取告警配置失败: {e}")
            return None

    # ========== 任务回调处理 ==========

    def _on_task_completed_or_failed(
        self,
        task_id: str,
        status: str,
        result: Any,
        error: Optional[str]
    ):
        """
        任务完成或失败的回调处理

        Args:
            task_id: 任务ID
            status: 任务状态 (completed/failed)
            result: 执行结果
            error: 错误信息
        """
        # 使用 asyncio.create_task 在事件循环中执行异步操作
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    self._handle_task_completion(task_id, status, result, error)
                )
            else:
                loop.run_until_complete(
                    self._handle_task_completion(task_id, status, result, error)
                )
        except RuntimeError:
            # 没有事件循环，创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self._handle_task_completion(task_id, status, result, error)
            )
            loop.close()

    async def _handle_task_completion(
        self,
        task_id: str,
        status: str,
        result: Any,
        error: Optional[str]
    ):
        """
        处理任务完成或失败

        Args:
            task_id: 任务ID
            status: 任务状态
            result: 执行结果
            error: 错误信息
        """
        from .base import TaskStatus

        # 获取任务对象
        task = self._task_manager.get_task(task_id)
        if not task:
            return

        if status == "completed":
            # 更新任务状态为完成
            await self._task_manager.update_task_status(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result
            )
            # 发布任务完成事件
            await self._publish_task_completed(task, result)

        elif status == "failed":
            # 更新任务状态为失败
            await self._task_manager.update_task_status(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=error
            )
            # 发布任务失败事件
            await self._publish_task_failed(task, error or "Unknown error")

            # 检查是否需要自动重试
            if task.should_retry():
                print(f"🔄 任务失败，将自动重试: {task_id}")
                new_task_id = await self._task_manager.retry_task(task_id)
                if new_task_id:
                    new_task = self._task_manager.get_task(new_task_id)
                    if new_task:
                        # 注册新任务的回调
                        self._worker_manager.register_task_callback(
                            new_task_id,
                            self._on_task_completed_or_failed
                        )
                        # 提交重试任务到工作队列
                        task_data = {
                            "id": new_task_id,
                            "type": new_task.type,
                            "payload": new_task.payload
                        }
                        self._worker_manager.submit_task(task_data)
                        await self._task_manager.update_task_status(
                            task_id=new_task_id,
                            status=TaskStatus.RUNNING
                        )
                        # 发布任务重试事件
                        await self._publish_task_retried(task, new_task)

    # ========== 事件总线集成 ==========

    async def _publish_task_completed(self, task: Task, result: Any):
        """发布任务完成事件"""
        if self._event_bus_integration:
            try:
                await self._event_bus_integration.publish_task_completed(task, result)
            except Exception as e:
                logger.error(f"发布任务完成事件失败: {e}")

    async def _publish_task_failed(self, task: Task, error: str):
        """发布任务失败事件"""
        if self._event_bus_integration:
            try:
                await self._event_bus_integration.publish_task_failed(task, error)
            except Exception as e:
                logger.error(f"发布任务失败事件失败: {e}")

    async def _publish_task_cancelled(self, task: Task):
        """发布任务取消事件"""
        if self._event_bus_integration:
            try:
                await self._event_bus_integration.publish_task_cancelled(task)
            except Exception as e:
                logger.error(f"发布任务取消事件失败: {e}")

    async def _publish_task_timeout(self, task: Task):
        """发布任务超时事件"""
        if self._event_bus_integration:
            try:
                await self._event_bus_integration.publish_task_timeout(task)
            except Exception as e:
                logger.error(f"发布任务超时事件失败: {e}")

    async def _publish_task_retried(self, original_task: Task, new_task: Task):
        """发布任务重试事件"""
        if self._event_bus_integration:
            try:
                await self._event_bus_integration.publish_task_retried(original_task, new_task)
            except Exception as e:
                logger.error(f"发布任务重试事件失败: {e}")

    def register_event_trigger(
        self,
        event_type: str,
        task_type: str,
        payload_template: Optional[Dict[str, Any]] = None,
        priority: int = 5,
        timeout_seconds: Optional[int] = None
    ) -> bool:
        """
        注册事件触发器

        当指定类型的事件发生时，自动创建并执行相应任务

        Args:
            event_type: 事件类型
            task_type: 任务类型
            payload_template: 任务数据模板
            priority: 任务优先级
            timeout_seconds: 任务超时时间

        Returns:
            bool: 注册是否成功
        """
        if not self._event_driven_trigger:
            logger.warning("事件驱动触发器未启用")
            return False

        try:
            self._event_driven_trigger.register_event_trigger(
                event_type=event_type,
                task_type=task_type,
                payload_template=payload_template,
                priority=priority,
                timeout_seconds=timeout_seconds
            )
            logger.info(f"已注册事件触发器: {event_type} -> {task_type}")
            return True
        except Exception as e:
            logger.error(f"注册事件触发器失败: {e}")
            return False

    def unregister_event_trigger(self, event_type: str) -> bool:
        """
        注销事件触发器

        Args:
            event_type: 事件类型

        Returns:
            bool: 注销是否成功
        """
        if not self._event_driven_trigger:
            return False

        try:
            self._event_driven_trigger.unregister_event_trigger(event_type)
            return True
        except Exception as e:
            logger.error(f"注销事件触发器失败: {e}")
            return False

    def get_event_triggers(self) -> List[Dict[str, Any]]:
        """
        获取所有已注册的事件触发器

        Returns:
            List[Dict[str, Any]]: 事件触发器列表
        """
        if not self._event_driven_trigger:
            return []

        try:
            return self._event_driven_trigger.get_registered_triggers()
        except Exception as e:
            logger.error(f"获取事件触发器列表失败: {e}")
            return []

    def subscribe_to_external_event(self, event_type: str, handler: Callable) -> bool:
        """
        订阅外部事件

        Args:
            event_type: 事件类型
            handler: 事件处理函数

        Returns:
            bool: 订阅是否成功
        """
        if not self._event_bus_integration:
            logger.warning("事件总线集成未启用")
            return False

        try:
            self._event_bus_integration.subscribe_to_external_events(event_type, handler)
            return True
        except Exception as e:
            logger.error(f"订阅外部事件失败: {e}")
            return False


# 全局调度器实例
_scheduler_instance: Optional[UnifiedScheduler] = None


def get_unified_scheduler(
    max_workers: int = 4,
    max_task_history: int = 1000,
    timezone: str = "Asia/Shanghai",
    enable_persistence: bool = False,
    enable_alerting: bool = False,
    alert_config: Optional[Dict[str, Any]] = None,
    enable_event_bus: bool = False,
    event_bus_config: Optional[Dict[str, Any]] = None
) -> UnifiedScheduler:
    """
    获取统一调度器实例（单例）

    Args:
        max_workers: 最大工作进程数
        max_task_history: 最大任务历史记录数
        timezone: 时区
        enable_persistence: 是否启用数据库持久化
        enable_alerting: 是否启用告警
        alert_config: 告警配置
        enable_event_bus: 是否启用事件总线
        event_bus_config: 事件总线配置

    Returns:
        UnifiedScheduler: 统一调度器实例
    """
    global _scheduler_instance

    if _scheduler_instance is None:
        _scheduler_instance = UnifiedScheduler(
            max_workers=max_workers,
            max_task_history=max_task_history,
            timezone=timezone,
            enable_persistence=enable_persistence,
            enable_alerting=enable_alerting,
            alert_config=alert_config,
            enable_event_bus=enable_event_bus,
            event_bus_config=event_bus_config
        )

    return _scheduler_instance
