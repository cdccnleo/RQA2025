"""
调度器持久化仓库

负责任务和定时任务的数据库CRUD操作
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

from .models import (
    TaskModel, JobModel, TaskHistoryModel, SchedulerMetricsModel,
    get_session_factory, init_database
)
from ..base import Task, TaskStatus, Job, JobType, TriggerType

logger = logging.getLogger(__name__)


class TaskRepository:
    """任务仓库"""

    def __init__(self, session_factory=None):
        """
        初始化任务仓库

        Args:
            session_factory: SQLAlchemy会话工厂
        """
        self._session_factory = session_factory
        if self._session_factory is None:
            self._session_factory = get_session_factory()

    @contextmanager
    def _session_scope(self):
        """提供事务范围的会话上下文管理器"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def save_task(self, task: Task) -> bool:
        """
        保存任务到数据库

        Args:
            task: 任务对象

        Returns:
            bool: 保存是否成功
        """
        try:
            with self._session_scope() as session:
                # 检查是否已存在
                existing = session.query(TaskModel).filter_by(id=task.id).first()

                if existing:
                    # 更新现有任务
                    existing.type = task.type
                    existing.status = task.status.value
                    existing.priority = task.priority
                    existing.started_at = task.started_at
                    existing.completed_at = task.completed_at
                    existing.timeout_seconds = task.timeout_seconds
                    existing.max_retries = task.max_retries
                    existing.retry_count = task.retry_count
                    existing.retry_delay_seconds = task.retry_delay_seconds
                    existing.deadline = task.deadline
                    existing.worker_id = task.worker_id
                    existing.error = task.error
                    existing.result = task.result
                    existing.payload = task.payload
                    existing.retry_info = task.payload.get('_retry_info')
                else:
                    # 创建新任务
                    task_model = TaskModel(
                        id=task.id,
                        type=task.type,
                        status=task.status.value,
                        priority=task.priority,
                        created_at=task.created_at,
                        started_at=task.started_at,
                        completed_at=task.completed_at,
                        timeout_seconds=task.timeout_seconds,
                        max_retries=task.max_retries,
                        retry_count=task.retry_count,
                        retry_delay_seconds=task.retry_delay_seconds,
                        deadline=task.deadline,
                        worker_id=task.worker_id,
                        error=task.error,
                        result=task.result,
                        payload=task.payload,
                        retry_info=task.payload.get('_retry_info')
                    )
                    session.add(task_model)

                return True
        except Exception as e:
            logger.error(f"保存任务失败: {e}")
            return False

    def get_task(self, task_id: str) -> Optional[TaskModel]:
        """
        根据ID获取任务

        Args:
            task_id: 任务ID

        Returns:
            Optional[TaskModel]: 任务模型
        """
        try:
            with self._session_scope() as session:
                return session.query(TaskModel).filter_by(id=task_id).first()
        except Exception as e:
            logger.error(f"获取任务失败: {e}")
            return None

    def get_tasks_by_status(self, status: TaskStatus, limit: int = 100) -> List[TaskModel]:
        """
        根据状态获取任务列表

        Args:
            status: 任务状态
            limit: 返回数量限制

        Returns:
            List[TaskModel]: 任务模型列表
        """
        try:
            with self._session_scope() as session:
                return session.query(TaskModel)\
                    .filter_by(status=status.value)\
                    .order_by(desc(TaskModel.created_at))\
                    .limit(limit)\
                    .all()
        except Exception as e:
            logger.error(f"获取任务列表失败: {e}")
            return []

    def get_pending_tasks(self, limit: int = 100) -> List[TaskModel]:
        """获取待处理任务"""
        return self.get_tasks_by_status(TaskStatus.PENDING, limit)

    def get_running_tasks(self, limit: int = 100) -> List[TaskModel]:
        """获取运行中任务"""
        return self.get_tasks_by_status(TaskStatus.RUNNING, limit)

    def get_timeout_tasks(self) -> List[TaskModel]:
        """
        获取已超时的任务

        Returns:
            List[TaskModel]: 超时任务列表
        """
        try:
            with self._session_scope() as session:
                now = datetime.utcnow()
                return session.query(TaskModel)\
                    .filter(
                        and_(
                            TaskModel.status.in_(['pending', 'running']),
                            TaskModel.deadline < now
                        )
                    )\
                    .all()
        except Exception as e:
            logger.error(f"获取超时任务失败: {e}")
            return []

    def update_task_status(self, task_id: str, status: TaskStatus,
                          result: Any = None, error: str = None) -> bool:
        """
        更新任务状态

        Args:
            task_id: 任务ID
            status: 新状态
            result: 执行结果
            error: 错误信息

        Returns:
            bool: 更新是否成功
        """
        try:
            with self._session_scope() as session:
                task = session.query(TaskModel).filter_by(id=task_id).first()
                if not task:
                    return False

                task.status = status.value

                if status == TaskStatus.RUNNING and not task.started_at:
                    task.started_at = datetime.utcnow()

                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task.completed_at = datetime.utcnow()
                    task.result = result
                    task.error = error

                    # 移动到历史记录
                    self._save_to_history(session, task)

                    # 删除活跃任务记录
                    session.delete(task)

                return True
        except Exception as e:
            logger.error(f"更新任务状态失败: {e}")
            return False

    def _save_to_history(self, session: Session, task: TaskModel):
        """保存任务到历史记录"""
        try:
            execution_time = None
            if task.started_at and task.completed_at:
                execution_time = (task.completed_at - task.started_at).total_seconds()

            history = TaskHistoryModel(
                id=f"hist-{task.id}",
                task_id=task.id,
                type=task.type,
                status=task.status,
                priority=task.priority,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                worker_id=task.worker_id,
                error=task.error,
                result=task.result,
                payload=task.payload,
                execution_time=execution_time
            )
            session.add(history)
        except Exception as e:
            logger.error(f"保存历史记录失败: {e}")

    def delete_task(self, task_id: str) -> bool:
        """
        删除任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 删除是否成功
        """
        try:
            with self._session_scope() as session:
                task = session.query(TaskModel).filter_by(id=task_id).first()
                if task:
                    session.delete(task)
                    return True
                return False
        except Exception as e:
            logger.error(f"删除任务失败: {e}")
            return False

    def get_task_statistics(self) -> Dict[str, Any]:
        """
        获取任务统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            with self._session_scope() as session:
                total = session.query(TaskModel).count()
                pending = session.query(TaskModel).filter_by(status='pending').count()
                running = session.query(TaskModel).filter_by(status='running').count()
                failed = session.query(TaskModel).filter_by(status='failed').count()
                completed = session.query(TaskModel).filter_by(status='completed').count()

                return {
                    'total': total,
                    'pending': pending,
                    'running': running,
                    'failed': failed,
                    'completed': completed
                }
        except Exception as e:
            logger.error(f"获取任务统计失败: {e}")
            return {'total': 0, 'pending': 0, 'running': 0, 'failed': 0, 'completed': 0}


class JobRepository:
    """定时任务仓库"""

    def __init__(self, session_factory=None):
        """
        初始化定时任务仓库

        Args:
            session_factory: SQLAlchemy会话工厂
        """
        self._session_factory = session_factory
        if self._session_factory is None:
            self._session_factory = get_session_factory()

    @contextmanager
    def _session_scope(self):
        """提供事务范围的会话上下文管理器"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def save_job(self, job: Job) -> bool:
        """
        保存定时任务到数据库

        Args:
            job: 定时任务对象

        Returns:
            bool: 保存是否成功
        """
        try:
            with self._session_scope() as session:
                existing = session.query(JobModel).filter_by(id=job.id).first()

                if existing:
                    existing.name = job.name
                    existing.job_type = job.job_type.value
                    existing.trigger_type = job.trigger_type.value
                    existing.trigger_config = job.trigger_config
                    existing.config = job.config
                    existing.enabled = job.enabled
                    existing.last_run = job.last_run
                    existing.next_run = job.next_run
                    existing.run_count = job.run_count
                else:
                    job_model = JobModel(
                        id=job.id,
                        name=job.name,
                        job_type=job.job_type.value,
                        trigger_type=job.trigger_type.value,
                        trigger_config=job.trigger_config,
                        config=job.config,
                        enabled=job.enabled,
                        created_at=job.created_at,
                        last_run=job.last_run,
                        next_run=job.next_run,
                        run_count=job.run_count
                    )
                    session.add(job_model)

                return True
        except Exception as e:
            logger.error(f"保存定时任务失败: {e}")
            return False

    def get_job(self, job_id: str) -> Optional[JobModel]:
        """
        根据ID获取定时任务

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobModel]: 定时任务模型
        """
        try:
            with self._session_scope() as session:
                return session.query(JobModel).filter_by(id=job_id).first()
        except Exception as e:
            logger.error(f"获取定时任务失败: {e}")
            return None

    def get_all_jobs(self) -> List[JobModel]:
        """
        获取所有定时任务

        Returns:
            List[JobModel]: 定时任务列表
        """
        try:
            with self._session_scope() as session:
                return session.query(JobModel).all()
        except Exception as e:
            logger.error(f"获取定时任务列表失败: {e}")
            return []

    def get_enabled_jobs(self) -> List[JobModel]:
        """
        获取启用的定时任务

        Returns:
            List[JobModel]: 启用的定时任务列表
        """
        try:
            with self._session_scope() as session:
                return session.query(JobModel).filter_by(enabled=True).all()
        except Exception as e:
            logger.error(f"获取启用的定时任务失败: {e}")
            return []

    def get_due_jobs(self) -> List[JobModel]:
        """
        获取到期的定时任务

        Returns:
            List[JobModel]: 到期的定时任务列表
        """
        try:
            with self._session_scope() as session:
                now = datetime.utcnow()
                return session.query(JobModel)\
                    .filter(
                        and_(
                            JobModel.enabled == True,
                            JobModel.next_run <= now
                        )
                    )\
                    .all()
        except Exception as e:
            logger.error(f"获取到期任务失败: {e}")
            return []

    def update_job_run(self, job_id: str, success: bool = True) -> bool:
        """
        更新任务执行记录

        Args:
            job_id: 任务ID
            success: 是否执行成功

        Returns:
            bool: 更新是否成功
        """
        try:
            with self._session_scope() as session:
                job = session.query(JobModel).filter_by(id=job_id).first()
                if not job:
                    return False

                job.last_run = datetime.utcnow()
                job.run_count += 1

                if success:
                    job.success_count += 1
                else:
                    job.fail_count += 1

                return True
        except Exception as e:
            logger.error(f"更新任务执行记录失败: {e}")
            return False

    def delete_job(self, job_id: str) -> bool:
        """
        删除定时任务

        Args:
            job_id: 任务ID

        Returns:
            bool: 删除是否成功
        """
        try:
            with self._session_scope() as session:
                job = session.query(JobModel).filter_by(id=job_id).first()
                if job:
                    session.delete(job)
                    return True
                return False
        except Exception as e:
            logger.error(f"删除定时任务失败: {e}")
            return False


class SchedulerMetricsRepository:
    """调度器指标仓库"""

    def __init__(self, session_factory=None):
        """
        初始化指标仓库

        Args:
            session_factory: SQLAlchemy会话工厂
        """
        self._session_factory = session_factory
        if self._session_factory is None:
            self._session_factory = get_session_factory()

    @contextmanager
    def _session_scope(self):
        """提供事务范围的会话上下文管理器"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def save_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        保存指标数据

        Args:
            metrics: 指标数据字典

        Returns:
            bool: 保存是否成功
        """
        try:
            with self._session_scope() as session:
                metrics_model = SchedulerMetricsModel(
                    total_tasks=metrics.get('total_tasks', 0),
                    pending_tasks=metrics.get('pending_tasks', 0),
                    running_tasks=metrics.get('running_tasks', 0),
                    completed_tasks=metrics.get('completed_tasks', 0),
                    failed_tasks=metrics.get('failed_tasks', 0),
                    success_rate=metrics.get('success_rate', 0.0),
                    avg_execution_time=metrics.get('avg_execution_time', 0.0),
                    active_workers=metrics.get('active_workers', 0),
                    busy_workers=metrics.get('busy_workers', 0)
                )
                session.add(metrics_model)
                return True
        except Exception as e:
            logger.error(f"保存指标数据失败: {e}")
            return False

    def get_latest_metrics(self, limit: int = 1) -> List[SchedulerMetricsModel]:
        """
        获取最新的指标数据

        Args:
            limit: 返回数量

        Returns:
            List[SchedulerMetricsModel]: 指标数据列表
        """
        try:
            with self._session_scope() as session:
                return session.query(SchedulerMetricsModel)\
                    .order_by(desc(SchedulerMetricsModel.timestamp))\
                    .limit(limit)\
                    .all()
        except Exception as e:
            logger.error(f"获取指标数据失败: {e}")
            return []

    def get_metrics_history(self, hours: int = 24) -> List[SchedulerMetricsModel]:
        """
        获取指标历史数据

        Args:
            hours: 查询小时数

        Returns:
            List[SchedulerMetricsModel]: 指标历史数据
        """
        try:
            with self._session_scope() as session:
                from datetime import timedelta
                since = datetime.utcnow() - timedelta(hours=hours)
                return session.query(SchedulerMetricsModel)\
                    .filter(SchedulerMetricsModel.timestamp >= since)\
                    .order_by(SchedulerMetricsModel.timestamp)\
                    .all()
        except Exception as e:
            logger.error(f"获取指标历史失败: {e}")
            return []


class SchedulerPersistence:
    """调度器持久化管理器"""

    def __init__(self, session_factory=None):
        """
        初始化持久化管理器

        Args:
            session_factory: SQLAlchemy会话工厂
        """
        self._session_factory = session_factory
        if self._session_factory is None:
            self._session_factory = get_session_factory()

        self.tasks = TaskRepository(self._session_factory)
        self.jobs = JobRepository(self._session_factory)
        self.metrics = SchedulerMetricsRepository(self._session_factory)

    def initialize_database(self):
        """初始化数据库表"""
        try:
            init_database()
            logger.info("数据库表初始化成功")
            return True
        except Exception as e:
            logger.error(f"数据库表初始化失败: {e}")
            return False

    def restore_jobs_from_db(self) -> List[Job]:
        """
        从数据库恢复定时任务

        Returns:
            List[Job]: 定时任务列表
        """
        try:
            job_models = self.jobs.get_all_jobs()
            jobs = []

            for model in job_models:
                try:
                    job = Job(
                        id=model.id,
                        name=model.name,
                        job_type=JobType(model.job_type),
                        trigger_type=TriggerType(model.trigger_type),
                        trigger_config=model.trigger_config,
                        handler=None,  # 处理器需要重新注册
                        config=model.config,
                        enabled=model.enabled,
                        created_at=model.created_at,
                        last_run=model.last_run,
                        next_run=model.next_run,
                        run_count=model.run_count
                    )
                    jobs.append(job)
                except Exception as e:
                    logger.error(f"恢复任务 {model.id} 失败: {e}")

            logger.info(f"从数据库恢复了 {len(jobs)} 个定时任务")
            return jobs
        except Exception as e:
            logger.error(f"恢复定时任务失败: {e}")
            return []
