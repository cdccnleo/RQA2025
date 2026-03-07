"""
调度器数据库模型

定义任务和定时任务的数据库表结构
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, String, Integer, DateTime, Text, Boolean,
    Float, JSON, Index, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import os

Base = declarative_base()


class TaskModel(Base):
    """任务数据库模型"""
    __tablename__ = 'scheduler_tasks'

    # 主键
    id = Column(String(32), primary_key=True, index=True)

    # 任务基本信息
    type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True, default='pending')
    priority = Column(Integer, default=5)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # 超时和重试配置
    timeout_seconds = Column(Integer, nullable=True)
    max_retries = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)
    retry_delay_seconds = Column(Integer, default=0)
    deadline = Column(DateTime, nullable=True)

    # 执行信息
    worker_id = Column(String(32), nullable=True, index=True)
    error = Column(Text, nullable=True)
    result = Column(JSON, nullable=True)

    # 任务数据
    payload = Column(JSON, default=dict)

    # 重试信息
    retry_info = Column(JSON, nullable=True)

    # 索引
    __table_args__ = (
        Index('idx_task_status_created', 'status', 'created_at'),
        Index('idx_task_type_status', 'type', 'status'),
        Index('idx_task_worker', 'worker_id', 'status'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'type': self.type,
            'status': self.status,
            'priority': self.priority,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'timeout_seconds': self.timeout_seconds,
            'max_retries': self.max_retries,
            'retry_count': self.retry_count,
            'retry_delay_seconds': self.retry_delay_seconds,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'worker_id': self.worker_id,
            'error': self.error,
            'result': self.result,
            'payload': self.payload,
            'retry_info': self.retry_info
        }


class JobModel(Base):
    """定时任务数据库模型"""
    __tablename__ = 'scheduler_jobs'

    # 主键
    id = Column(String(32), primary_key=True, index=True)

    # 任务基本信息
    name = Column(String(100), nullable=False)
    job_type = Column(String(50), nullable=False, index=True)
    trigger_type = Column(String(20), nullable=False)
    trigger_config = Column(JSON, default=dict)

    # 任务配置
    config = Column(JSON, default=dict)
    enabled = Column(Boolean, default=True, index=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    last_run = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=True, index=True)

    # 执行统计
    run_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    fail_count = Column(Integer, default=0)

    # 超时和重试配置
    timeout_seconds = Column(Integer, nullable=True)
    max_retries = Column(Integer, default=0)

    # 索引
    __table_args__ = (
        Index('idx_job_enabled_next_run', 'enabled', 'next_run'),
        Index('idx_job_type_enabled', 'job_type', 'enabled'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'job_type': self.job_type,
            'trigger_type': self.trigger_type,
            'trigger_config': self.trigger_config,
            'config': self.config,
            'enabled': self.enabled,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'run_count': self.run_count,
            'success_count': self.success_count,
            'fail_count': self.fail_count,
            'timeout_seconds': self.timeout_seconds,
            'max_retries': self.max_retries
        }


class TaskHistoryModel(Base):
    """任务历史记录数据库模型"""
    __tablename__ = 'scheduler_task_history'

    # 主键
    id = Column(String(32), primary_key=True)
    task_id = Column(String(32), nullable=False, index=True)

    # 任务基本信息
    type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)
    priority = Column(Integer)

    # 时间戳
    created_at = Column(DateTime, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True, index=True)

    # 执行信息
    worker_id = Column(String(32), nullable=True)
    error = Column(Text, nullable=True)
    result = Column(JSON, nullable=True)

    # 任务数据
    payload = Column(JSON, default=dict)

    # 执行时长（秒）
    execution_time = Column(Float, nullable=True)

    # 索引
    __table_args__ = (
        Index('idx_history_type_completed', 'type', 'completed_at'),
        Index('idx_history_status_completed', 'status', 'completed_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'task_id': self.task_id,
            'type': self.type,
            'status': self.status,
            'priority': self.priority,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'worker_id': self.worker_id,
            'error': self.error,
            'result': self.result,
            'payload': self.payload,
            'execution_time': self.execution_time
        }


class SchedulerMetricsModel(Base):
    """调度器指标数据库模型"""
    __tablename__ = 'scheduler_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # 任务统计
    total_tasks = Column(Integer, default=0)
    pending_tasks = Column(Integer, default=0)
    running_tasks = Column(Integer, default=0)
    completed_tasks = Column(Integer, default=0)
    failed_tasks = Column(Integer, default=0)

    # 成功率
    success_rate = Column(Float, default=0.0)

    # 平均执行时间
    avg_execution_time = Column(Float, default=0.0)

    # 工作进程统计
    active_workers = Column(Integer, default=0)
    busy_workers = Column(Integer, default=0)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'total_tasks': self.total_tasks,
            'pending_tasks': self.pending_tasks,
            'running_tasks': self.running_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.success_rate,
            'avg_execution_time': self.avg_execution_time,
            'active_workers': self.active_workers,
            'busy_workers': self.busy_workers
        }


def get_database_url() -> str:
    """
    获取数据库连接URL

    从环境变量读取配置，支持容器化环境
    """
    # 检测容器化环境
    is_container = any([
        os.path.exists('/.dockerenv'),
        os.environ.get('KUBERNETES_SERVICE_HOST'),
        os.environ.get('DOCKER_CONTAINER'),
        os.environ.get('CONTAINER_ENV')
    ])

    # 数据库配置
    db_host = os.getenv('RQA_DB_HOST',
                      os.getenv('DB_HOST',
                                os.getenv('POSTGRES_HOST',
                                          'postgres' if is_container else 'localhost')))
    db_port = os.getenv('RQA_DB_PORT',
                      os.getenv('DB_PORT',
                                os.getenv('POSTGRES_PORT', '5432')))
    db_name = os.getenv('RQA_DB_NAME',
                      os.getenv('DB_NAME',
                                os.getenv('POSTGRES_DB', 'rqa2025_prod')))
    db_user = os.getenv('RQA_DB_USER',
                      os.getenv('DB_USER',
                                os.getenv('POSTGRES_USER', 'rqa2025_admin')))
    db_password = os.getenv('RQA_DB_PASSWORD',
                          os.getenv('DB_PASSWORD',
                                    os.getenv('POSTGRES_PASSWORD', 'SecurePass123!')))

    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def create_database_engine():
    """创建数据库引擎"""
    database_url = get_database_url()

    return create_engine(
        database_url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        echo=False
    )


def init_database():
    """初始化数据库表"""
    engine = create_database_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine=None):
    """获取会话工厂"""
    if engine is None:
        engine = create_database_engine()
    return sessionmaker(bind=engine)
