"""
补全进度追踪和恢复机制
实现补全任务的持久化状态管理和断点续传功能
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict

from src.infrastructure.logging.core.unified_logger import get_unified_logger
from .data_complement_scheduler import ComplementTask
from .batch_complement_processor import ComplementBatch, BatchStatus

logger = get_unified_logger(__name__)


@dataclass
class ComplementProgressSnapshot:
    """补全进度快照"""
    task_id: str
    source_id: str
    total_batches: int
    completed_batches: int
    failed_batches: int
    running_batches: int
    total_estimated_records: int
    total_actual_records: int
    start_time: datetime
    last_update_time: datetime
    status: str  # 'running', 'paused', 'completed', 'failed'
    progress_percentage: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    error_messages: List[str] = field(default_factory=list)
    batch_progress: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RecoveryPoint:
    """恢复点"""
    task_id: str
    batch_id: str
    last_processed_date: datetime
    records_processed: int
    checkpoint_data: Dict[str, Any]
    created_at: datetime


class ComplementProgressTracker:
    """
    补全进度追踪器

    实现功能：
    1. 实时进度追踪和持久化
    2. 断点续传和恢复机制
    3. 性能监控和统计分析
    4. 错误记录和故障排查
    """

    def __init__(self, storage_dir: str = "data/complement_progress"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 进度快照存储
        self.snapshots_file = self.storage_dir / "progress_snapshots.json"
        self.recovery_file = self.storage_dir / "recovery_points.json"

        # 内存缓存
        self.active_snapshots: Dict[str, ComplementProgressSnapshot] = {}
        self.recovery_points: Dict[str, RecoveryPoint] = {}

        # 配置
        self.save_interval = 30  # 每30秒保存一次
        self.max_snapshots_per_task = 10  # 每个任务最多保留10个快照
        self.recovery_retention_days = 7  # 恢复点保留7天

        # 加载现有数据
        self._load_data()

        logger.info(f"补全进度追踪器初始化完成，存储目录: {self.storage_dir}")

    def start_task_tracking(self, task: ComplementTask) -> str:
        """
        开始任务进度追踪

        Args:
            task: 补全任务

        Returns:
            快照ID
        """
        snapshot = ComplementProgressSnapshot(
            task_id=task.task_id,
            source_id=task.source_id,
            total_batches=0,  # 稍后更新
            completed_batches=0,
            failed_batches=0,
            running_batches=0,
            total_estimated_records=task.estimated_records,
            total_actual_records=0,
            start_time=task.started_at or datetime.now(),
            last_update_time=datetime.now(),
            status='running',
            progress_percentage=0.0
        )

        self.active_snapshots[task.task_id] = snapshot
        self._save_snapshot(snapshot)

        logger.info(f"开始任务进度追踪: {task.task_id}")
        return task.task_id

    def update_batch_progress(self, task_id: str, batch: ComplementBatch):
        """
        更新批次进度

        Args:
            task_id: 任务ID
            batch: 补全批次
        """
        if task_id not in self.active_snapshots:
            logger.warning(f"任务快照不存在: {task_id}")
            return

        snapshot = self.active_snapshots[task_id]

        # 更新批次进度信息
        batch_info = {
            'batch_index': batch.batch_index,
            'status': batch.status.value,
            'estimated_records': batch.estimated_records,
            'actual_records': batch.actual_records,
            'start_date': batch.start_date.isoformat(),
            'end_date': batch.end_date.isoformat(),
            'started_at': batch.started_at.isoformat() if batch.started_at else None,
            'completed_at': batch.completed_at.isoformat() if batch.completed_at else None,
            'error_message': batch.error_message
        }

        snapshot.batch_progress[batch.batch_id] = batch_info

        # 更新汇总统计
        self._update_snapshot_stats(snapshot)

        # 保存快照
        snapshot.last_update_time = datetime.now()
        self._save_snapshot(snapshot)

    def create_recovery_point(self, task_id: str, batch_id: str,
                            last_processed_date: datetime, records_processed: int,
                            checkpoint_data: Optional[Dict[str, Any]] = None):
        """
        创建恢复点

        Args:
            task_id: 任务ID
            batch_id: 批次ID
            last_processed_date: 最后处理日期
            records_processed: 已处理记录数
            checkpoint_data: 检查点数据
        """
        recovery_point = RecoveryPoint(
            task_id=task_id,
            batch_id=batch_id,
            last_processed_date=last_processed_date,
            records_processed=records_processed,
            checkpoint_data=checkpoint_data or {},
            created_at=datetime.now()
        )

        key = f"{task_id}_{batch_id}"
        self.recovery_points[key] = recovery_point
        self._save_recovery_points()

        logger.debug(f"创建恢复点: {key}")

    def get_recovery_point(self, task_id: str, batch_id: str) -> Optional[RecoveryPoint]:
        """
        获取恢复点

        Args:
            task_id: 任务ID
            batch_id: 批次ID

        Returns:
            恢复点信息
        """
        key = f"{task_id}_{batch_id}"
        return self.recovery_points.get(key)

    def complete_task_tracking(self, task_id: str, success: bool = True,
                             error_messages: Optional[List[str]] = None):
        """
        完成任务进度追踪

        Args:
            task_id: 任务ID
            success: 是否成功
            error_messages: 错误信息列表
        """
        if task_id not in self.active_snapshots:
            logger.warning(f"任务快照不存在: {task_id}")
            return

        snapshot = self.active_snapshots[task_id]
        snapshot.status = 'completed' if success else 'failed'
        snapshot.last_update_time = datetime.now()

        if error_messages:
            snapshot.error_messages.extend(error_messages)

        # 计算最终进度
        self._update_snapshot_stats(snapshot)

        # 保存最终快照
        self._save_snapshot(snapshot)

        # 从活跃快照中移除
        del self.active_snapshots[task_id]

        status_msg = "成功" if success else f"失败: {error_messages}"
        logger.info(f"完成任务进度追踪: {task_id} - {status_msg}")

    def get_task_progress(self, task_id: str) -> Optional[ComplementProgressSnapshot]:
        """
        获取任务进度

        Args:
            task_id: 任务ID

        Returns:
            进度快照
        """
        return self.active_snapshots.get(task_id)

    def get_incomplete_tasks(self) -> List[ComplementProgressSnapshot]:
        """
        获取未完成的任务

        Returns:
            未完成任务列表
        """
        incomplete_tasks = []

        # 检查活跃快照
        for snapshot in self.active_snapshots.values():
            if snapshot.status in ['running', 'paused']:
                incomplete_tasks.append(snapshot)

        # 检查持久化存储中是否有中断的任务
        # 这里可以扩展为从文件加载

        return incomplete_tasks

    def resume_task(self, task_id: str) -> Optional[ComplementProgressSnapshot]:
        """
        恢复任务执行

        Args:
            task_id: 任务ID

        Returns:
            任务快照（用于恢复执行）
        """
        snapshot = self.get_task_progress(task_id)
        if snapshot:
            snapshot.status = 'running'
            snapshot.last_update_time = datetime.now()
            self._save_snapshot(snapshot)

            logger.info(f"恢复任务执行: {task_id}")
            return snapshot

        logger.warning(f"无法恢复任务，找不到快照: {task_id}")
        return None

    def pause_task(self, task_id: str):
        """
        暂停任务

        Args:
            task_id: 任务ID
        """
        if task_id in self.active_snapshots:
            snapshot = self.active_snapshots[task_id]
            snapshot.status = 'paused'
            snapshot.last_update_time = datetime.now()
            self._save_snapshot(snapshot)

            logger.info(f"暂停任务: {task_id}")

    def get_task_statistics(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务统计信息

        Args:
            task_id: 任务ID

        Returns:
            统计信息
        """
        snapshot = self.get_task_progress(task_id)
        if not snapshot:
            return {}

        # 计算执行时间
        execution_time = (snapshot.last_update_time - snapshot.start_time).total_seconds()

        # 计算预计完成时间
        if snapshot.progress_percentage > 0:
            total_estimated_time = execution_time / (snapshot.progress_percentage / 100)
            remaining_time = total_estimated_time - execution_time
            estimated_completion = datetime.now() + timedelta(seconds=remaining_time)
        else:
            estimated_completion = None

        stats = {
            'task_id': task_id,
            'status': snapshot.status,
            'progress_percentage': snapshot.progress_percentage,
            'execution_time_seconds': execution_time,
            'estimated_completion_time': estimated_completion.isoformat() if estimated_completion else None,
            'total_batches': snapshot.total_batches,
            'completed_batches': snapshot.completed_batches,
            'failed_batches': snapshot.failed_batches,
            'running_batches': snapshot.running_batches,
            'total_estimated_records': snapshot.total_estimated_records,
            'total_actual_records': snapshot.total_actual_records,
            'error_count': len(snapshot.error_messages),
            'recent_errors': snapshot.error_messages[-5:] if snapshot.error_messages else []
        }

        return stats

    def get_system_statistics(self) -> Dict[str, Any]:
        """
        获取系统级统计信息

        Returns:
            系统统计信息
        """
        stats = {
            'active_tasks': len(self.active_snapshots),
            'running_tasks': len([s for s in self.active_snapshots.values() if s.status == 'running']),
            'paused_tasks': len([s for s in self.active_snapshots.values() if s.status == 'paused']),
            'total_recovery_points': len(self.recovery_points),
            'total_snapshots_stored': self._count_stored_snapshots()
        }

        # 计算平均进度
        if self.active_snapshots:
            avg_progress = sum(s.progress_percentage for s in self.active_snapshots.values()) / len(self.active_snapshots)
            stats['average_progress'] = avg_progress

        return stats

    def _update_snapshot_stats(self, snapshot: ComplementProgressSnapshot):
        """更新快照统计信息"""
        batch_progress = snapshot.batch_progress

        # 重新计算批次统计
        completed = 0
        failed = 0
        running = 0
        total_actual = 0

        for batch_info in batch_progress.values():
            status = batch_info['status']
            if status == 'completed':
                completed += 1
            elif status == 'failed':
                failed += 1
            elif status == 'running':
                running += 1

            total_actual += batch_info.get('actual_records', 0)

        snapshot.completed_batches = completed
        snapshot.failed_batches = failed
        snapshot.running_batches = running
        snapshot.total_batches = len(batch_progress)
        snapshot.total_actual_records = total_actual

        # 计算总体进度
        if snapshot.total_batches > 0:
            completed_ratio = (completed + failed) / snapshot.total_batches
            snapshot.progress_percentage = completed_ratio * 100
        else:
            snapshot.progress_percentage = 0.0

        # 更新预计完成时间
        if snapshot.progress_percentage > 0 and snapshot.progress_percentage < 100:
            elapsed_time = (datetime.now() - snapshot.start_time).total_seconds()
            total_estimated_time = elapsed_time / (snapshot.progress_percentage / 100)
            remaining_time = total_estimated_time - elapsed_time
            snapshot.estimated_completion_time = datetime.now() + timedelta(seconds=remaining_time)

    def _save_snapshot(self, snapshot: ComplementProgressSnapshot):
        """保存快照到文件"""
        try:
            # 确保快照目录存在
            task_dir = self.storage_dir / snapshot.task_id
            task_dir.mkdir(exist_ok=True)

            # 快照文件名包含时间戳
            timestamp = snapshot.last_update_time.strftime("%Y%m%d_%H%M%S")
            snapshot_file = task_dir / f"snapshot_{timestamp}.json"

            # 转换为字典并保存
            snapshot_data = asdict(snapshot)
            # 转换datetime对象
            for key, value in snapshot_data.items():
                if isinstance(value, datetime):
                    snapshot_data[key] = value.isoformat()

            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2, ensure_ascii=False)

            # 清理旧快照
            self._cleanup_old_snapshots(task_dir)

        except Exception as e:
            logger.error(f"保存快照失败: {e}")

    def _cleanup_old_snapshots(self, task_dir: Path):
        """清理旧的快照文件"""
        try:
            snapshot_files = list(task_dir.glob("snapshot_*.json"))
            if len(snapshot_files) > self.max_snapshots_per_task:
                # 按修改时间排序，保留最新的
                snapshot_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                files_to_delete = snapshot_files[self.max_snapshots_per_task:]

                for file_path in files_to_delete:
                    file_path.unlink()

        except Exception as e:
            logger.warning(f"清理旧快照失败: {e}")

    def _save_recovery_points(self):
        """保存恢复点到文件"""
        try:
            recovery_data = {}
            for key, recovery_point in self.recovery_points.items():
                recovery_data[key] = asdict(recovery_point)
                # 转换datetime对象
                for field_key, value in recovery_data[key].items():
                    if isinstance(value, datetime):
                        recovery_data[key][field_key] = value.isoformat()

            with open(self.recovery_file, 'w', encoding='utf-8') as f:
                json.dump(recovery_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"保存恢复点失败: {e}")

    def _load_data(self):
        """加载持久化数据"""
        try:
            # 加载恢复点
            if self.recovery_file.exists():
                with open(self.recovery_file, 'r', encoding='utf-8') as f:
                    recovery_data = json.load(f)

                for key, data in recovery_data.items():
                    # 转换datetime字符串
                    for field_key, value in data.items():
                        if field_key.endswith('_time') or field_key.endswith('_at'):
                            try:
                                data[field_key] = datetime.fromisoformat(value)
                            except (ValueError, TypeError):
                                pass

                    recovery_point = RecoveryPoint(**data)
                    self.recovery_points[key] = recovery_point

            # 加载最近的活跃快照
            if self.snapshots_file.exists():
                # 这里可以扩展为加载最近的快照
                pass

            logger.info(f"加载了 {len(self.recovery_points)} 个恢复点")

        except Exception as e:
            logger.warning(f"加载持久化数据失败: {e}")

    def _count_stored_snapshots(self) -> int:
        """统计存储的快照数量"""
        try:
            total_snapshots = 0
            for task_dir in self.storage_dir.iterdir():
                if task_dir.is_dir():
                    snapshot_files = list(task_dir.glob("snapshot_*.json"))
                    total_snapshots += len(snapshot_files)
            return total_snapshots
        except Exception:
            return 0

    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        清理旧的进度数据

        Args:
            days_to_keep: 保留天数
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)

            # 清理旧的恢复点
            old_recovery_keys = []
            for key, recovery_point in self.recovery_points.items():
                if recovery_point.created_at < cutoff_time:
                    old_recovery_keys.append(key)

            for key in old_recovery_keys:
                del self.recovery_points[key]

            # 清理旧的快照目录
            for task_dir in self.storage_dir.iterdir():
                if task_dir.is_dir():
                    # 检查目录是否太旧
                    try:
                        dir_mtime = datetime.fromtimestamp(task_dir.stat().st_mtime)
                        if dir_mtime < cutoff_time:
                            import shutil
                            shutil.rmtree(task_dir)
                    except Exception as e:
                        logger.warning(f"清理任务目录失败 {task_dir}: {e}")

            if old_recovery_keys:
                self._save_recovery_points()

            logger.info(f"清理了 {len(old_recovery_keys)} 个旧恢复点和相关快照")

        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")


# 全局实例
_progress_tracker = None


def get_complement_progress_tracker() -> ComplementProgressTracker:
    """获取补全进度追踪器实例"""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ComplementProgressTracker()
    return _progress_tracker