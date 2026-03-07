"""
任务历史记录管理器

提供任务历史记录的存储、查询和管理功能。
支持内存存储（开发/测试）和PostgreSQL存储（生产）。
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskHistoryRecord:
    """任务历史记录"""
    task_id: str
    source_id: str
    source_name: str
    status: str
    collection_type: str  # immediate, scheduled
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    records_count: int = 0
    data_size_mb: float = 0.0
    duration_ms: int = 0
    error_message: Optional[str] = None
    logs: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.metadata is None:
            self.metadata = {}


class TaskHistoryManager:
    """
    任务历史记录管理器

    管理数据采集任务的完整生命周期记录。
    """

    def __init__(self):
        """初始化任务历史管理器"""
        # 内存存储（开发/测试使用）
        self._task_history: Dict[str, TaskHistoryRecord] = {}
        self._source_task_map: Dict[str, List[str]] = {}  # source_id -> task_ids

        # 状态变更回调函数列表
        self._status_change_callbacks: List[callable] = []

        logger.info("✅ 任务历史记录管理器初始化完成")

    def register_status_change_callback(self, callback: callable):
        """
        注册状态变更回调函数

        Args:
            callback: 回调函数，参数为 (task_id, old_status, new_status)
        """
        self._status_change_callbacks.append(callback)
        logger.debug(f"注册状态变更回调，当前回调数: {len(self._status_change_callbacks)}")

    def _notify_status_change(self, task_id: str, old_status: str, new_status: str):
        """
        通知状态变更

        Args:
            task_id: 任务ID
            old_status: 旧状态
            new_status: 新状态
        """
        for callback in self._status_change_callbacks:
            try:
                callback(task_id, old_status, new_status)
            except Exception as e:
                logger.error(f"状态变更回调执行失败: {e}")

    def create_task_record(
        self,
        task_id: str,
        source_id: str,
        source_name: str,
        collection_type: str = "scheduled",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskHistoryRecord:
        """
        创建任务记录

        Args:
            task_id: 任务ID
            source_id: 数据源ID
            source_name: 数据源名称
            collection_type: 采集类型（immediate/scheduled）
            metadata: 额外元数据

        Returns:
            TaskHistoryRecord: 创建的任务记录
        """
        record = TaskHistoryRecord(
            task_id=task_id,
            source_id=source_id,
            source_name=source_name,
            status=TaskStatus.PENDING.value,
            collection_type=collection_type,
            submitted_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        # 保存到内存
        self._task_history[task_id] = record

        # 更新数据源-任务映射
        if source_id not in self._source_task_map:
            self._source_task_map[source_id] = []
        self._source_task_map[source_id].append(task_id)

        logger.info(f"✅ 任务记录已创建: {task_id} (数据源: {source_id})")
        return record

    def update_task_started(self, task_id: str) -> bool:
        """
        更新任务开始状态

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否更新成功
        """
        if task_id not in self._task_history:
            logger.warning(f"任务记录不存在: {task_id}")
            return False

        record = self._task_history[task_id]
        old_status = record.status
        record.status = TaskStatus.RUNNING.value
        record.started_at = datetime.now().isoformat()

        # 通知状态变更
        self._notify_status_change(task_id, old_status, record.status)

        logger.info(f"🚀 任务开始执行: {task_id}")
        return True

    def update_task_completed(
        self,
        task_id: str,
        records_count: int,
        data_size_mb: float = 0.0,
        logs: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        更新任务完成状态

        Args:
            task_id: 任务ID
            records_count: 采集记录数
            data_size_mb: 数据大小(MB)
            logs: 执行日志

        Returns:
            bool: 是否更新成功
        """
        if task_id not in self._task_history:
            logger.warning(f"任务记录不存在: {task_id}")
            return False

        record = self._task_history[task_id]
        old_status = record.status
        record.status = TaskStatus.COMPLETED.value
        record.completed_at = datetime.now().isoformat()
        record.records_count = records_count
        record.data_size_mb = data_size_mb

        # 计算耗时
        if record.started_at:
            started = datetime.fromisoformat(record.started_at)
            completed = datetime.fromisoformat(record.completed_at)
            record.duration_ms = int((completed - started).total_seconds() * 1000)

        # 添加日志
        if logs:
            record.logs.extend(logs)

        # 通知状态变更
        self._notify_status_change(task_id, old_status, record.status)

        logger.info(f"✅ 任务完成: {task_id}, 记录数: {records_count}, 耗时: {record.duration_ms}ms")
        return True

    def update_task_failed(
        self,
        task_id: str,
        error_message: str,
        logs: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        更新任务失败状态

        Args:
            task_id: 任务ID
            error_message: 错误信息
            logs: 执行日志

        Returns:
            bool: 是否更新成功
        """
        if task_id not in self._task_history:
            logger.warning(f"任务记录不存在: {task_id}")
            return False

        record = self._task_history[task_id]
        old_status = record.status
        record.status = TaskStatus.FAILED.value
        record.completed_at = datetime.now().isoformat()
        record.error_message = error_message

        # 计算耗时
        if record.started_at:
            started = datetime.fromisoformat(record.started_at)
            completed = datetime.fromisoformat(record.completed_at)
            record.duration_ms = int((completed - started).total_seconds() * 1000)

        # 添加日志
        if logs:
            record.logs.extend(logs)

        # 通知状态变更
        self._notify_status_change(task_id, old_status, record.status)

        logger.info(f"❌ 任务失败: {task_id}, 错误: {error_message}")
        return True

    def add_task_log(
        self,
        task_id: str,
        level: str,
        message: str
    ) -> bool:
        """
        添加任务日志

        Args:
            task_id: 任务ID
            level: 日志级别 (INFO/WARN/ERROR)
            message: 日志消息

        Returns:
            bool: 是否添加成功
        """
        if task_id not in self._task_history:
            logger.warning(f"任务记录不存在: {task_id}")
            return False

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }

        self._task_history[task_id].logs.append(log_entry)
        return True

    def get_task_record(self, task_id: str) -> Optional[TaskHistoryRecord]:
        """
        获取任务记录

        Args:
            task_id: 任务ID

        Returns:
            Optional[TaskHistoryRecord]: 任务记录
        """
        return self._task_history.get(task_id)

    def get_completed_tasks(
        self,
        limit: int = 20,
        offset: int = 0,
        source_id: Optional[str] = None
    ) -> List[TaskHistoryRecord]:
        """
        获取已完成的任务列表

        Args:
            limit: 返回数量限制
            offset: 偏移量
            source_id: 数据源ID过滤

        Returns:
            List[TaskHistoryRecord]: 任务记录列表
        """
        # 过滤已完成或失败的任务
        tasks = [
            record for record in self._task_history.values()
            if record.status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]
        ]

        # 按数据源过滤
        if source_id:
            tasks = [t for t in tasks if t.source_id == source_id]

        # 按完成时间倒序排序
        tasks.sort(
            key=lambda x: x.completed_at or x.submitted_at,
            reverse=True
        )

        # 分页
        return tasks[offset:offset + limit]

    def get_tasks_by_source(self, source_id: str) -> List[TaskHistoryRecord]:
        """
        获取指定数据源的所有任务

        Args:
            source_id: 数据源ID

        Returns:
            List[TaskHistoryRecord]: 任务记录列表
        """
        task_ids = self._source_task_map.get(source_id, [])
        return [self._task_history[tid] for tid in task_ids if tid in self._task_history]

    def get_task_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        获取任务统计信息

        Args:
            days: 统计天数

        Returns:
            Dict[str, Any]: 统计信息
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_tasks = [
            record for record in self._task_history.values()
            if record.completed_at and datetime.fromisoformat(record.completed_at) >= cutoff_date
        ]

        completed = [t for t in recent_tasks if t.status == TaskStatus.COMPLETED.value]
        failed = [t for t in recent_tasks if t.status == TaskStatus.FAILED.value]

        total_records = sum(t.records_count for t in completed)
        avg_duration = sum(t.duration_ms for t in completed) / len(completed) if completed else 0

        return {
            "total_tasks": len(recent_tasks),
            "completed_tasks": len(completed),
            "failed_tasks": len(failed),
            "success_rate": len(completed) / len(recent_tasks) * 100 if recent_tasks else 0,
            "total_records": total_records,
            "avg_duration_ms": int(avg_duration),
            "period_days": days
        }

    def cleanup_old_records(self, days: int = 30) -> int:
        """
        清理旧的任务记录

        Args:
            days: 保留天数

        Returns:
            int: 清理的记录数
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        to_remove = []

        for task_id, record in self._task_history.items():
            if record.completed_at:
                completed = datetime.fromisoformat(record.completed_at)
                if completed < cutoff_date:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._task_history[task_id]
            # 从数据源映射中移除
            for source_id, task_ids in self._source_task_map.items():
                if task_id in task_ids:
                    task_ids.remove(task_id)

        logger.info(f"🧹 清理了 {len(to_remove)} 条旧任务记录")
        return len(to_remove)


# 全局任务历史管理器实例
_task_history_manager: Optional[TaskHistoryManager] = None


def get_task_history_manager() -> TaskHistoryManager:
    """获取任务历史管理器实例"""
    global _task_history_manager
    if _task_history_manager is None:
        _task_history_manager = TaskHistoryManager()
    return _task_history_manager
