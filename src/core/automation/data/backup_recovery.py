"""
Data Backup and Recovery Automation Module
数据备份和恢复自动化模块

This module provides automated data backup and recovery capabilities for quantitative trading
此模块为量化交易提供自动化数据备份和恢复能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import threading
import time
import os
import shutil
import gzip
import json
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)


class BackupType(Enum):

    """Backup types"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):

    """Backup status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


class StorageType(Enum):

    """Storage types for backups"""
    LOCAL = "local"
    CLOUD = "cloud"
    NAS = "nas"
    TAPE = "tape"


@dataclass

class BackupJob:

    """
    Backup job data class
    备份作业数据类
    """
    job_id: str
    backup_type: str
    source_path: str
    destination_path: str
    storage_type: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


    def to_dict(self) -> Dict[str, Any]:

        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


class RecoveryJob:

    """
    Recovery job data class
    恢复作业数据类
    """
    job_id: str
    backup_job_id: str
    source_path: str
    destination_path: str
    recovery_type: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    files_recovered: int = 0
    execution_time: float = 0.0
    error_message: Optional[str] = None


    def to_dict(self) -> Dict[str, Any]:

        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


class BackupManager:

    """
    Backup Manager Class
    备份管理器类

    Manages automated data backup operations
    管理自动化数据备份操作
    """


    def __init__(self, manager_name: str = "default_backup_manager"):

        """
        Initialize backup manager
        初始化备份管理器

        Args:
            manager_name: Name of the backup manager
                        备份管理器名称
        """
        self.manager_name = manager_name
        self.backup_jobs: Dict[str, BackupJob] = {}
        self.active_jobs: Dict[str, threading.Thread] = {}

        # Configuration
        self.max_concurrent_jobs = 3
        self.compression_enabled = True
        self.compression_level = 6  # gzip level
        self.backup_retention_days = 30
        self.verify_backups = True

        # Statistics
        self.stats = {
            'total_backups': 0,
            'successful_backups': 0,
            'failed_backups': 0,
            'total_backup_size': 0,
            'total_compressed_size': 0,
            'compression_ratio': 0.0
        }

        # Storage configurations
        self.storage_configs = {}

        logger.info(f"Backup manager {manager_name} initialized")


    def create_backup_job(self,


                          job_id: str,
                          backup_type: BackupType,
                          source_path: str,
                          destination_path: str,
                          storage_type: StorageType,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a backup job
        创建备份作业

        Args:
            job_id: Unique job identifier
                   唯一作业标识符
            backup_type: Type of backup
                        备份类型
            source_path: Source path to backup
                        要备份的源路径
            destination_path: Destination path for backup
                             备份目标路径
            storage_type: Type of storage
                         存储类型
            metadata: Additional metadata
                     其他元数据

        Returns:
            str: Created job ID
                 创建的作业ID
        """
        job = BackupJob(
            job_id=job_id,
            backup_type=backup_type.value,
            source_path=source_path,
            destination_path=destination_path,
            storage_type=storage_type.value,
            status=BackupStatus.PENDING.value,
            created_at=datetime.now(),
            metadata=metadata or {}
        )

        self.backup_jobs[job_id] = job
        logger.info(f"Created backup job: {job_id}")
        return job_id


    def execute_backup(self, job_id: str, async_execution: bool = True) -> Dict[str, Any]:

        """
        Execute a backup job
        执行备份作业

        Args:
            job_id: Job identifier
                   作业标识符
            async_execution: Whether to execute asynchronously
                           是否异步执行

        Returns:
            dict: Execution result
                  执行结果
        """
        if job_id not in self.backup_jobs:
            return {'success': False, 'error': f'Backup job {job_id} not found'}

        job = self.backup_jobs[job_id]

        # Check concurrent job limit
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return {
                'success': False,
                'error': 'Maximum concurrent backup jobs reached'
            }

        if async_execution:
            # Start async execution
            execution_thread = threading.Thread(
                target=self._execute_backup_sync,
                args=(job_id,),
                daemon=True
            )
            self.active_jobs[job_id] = execution_thread
            execution_thread.start()

            return {
                'success': True,
                'execution_mode': 'async',
                'job_id': job_id
            }
        else:
            # Execute synchronously
            return self._execute_backup_sync(job_id)


    def _execute_backup_sync(self, job_id: str) -> Dict[str, Any]:

        """
        Execute backup job synchronously
        同步执行备份作业

        Args:
            job_id: Job identifier
                   作业标识符

        Returns:
            dict: Execution result
                  执行结果
        """
        job = self.backup_jobs[job_id]
        job.status = BackupStatus.RUNNING.value
        job.started_at = datetime.now()

        result = {
            'job_id': job_id,
            'success': False,
            'start_time': job.started_at,
            'execution_time': 0.0
        }

        start_time = time.time()

        try:
            # Execute backup based on type
            if job.backup_type == BackupType.FULL.value:
                backup_result = self._execute_full_backup(job)
            elif job.backup_type == BackupType.INCREMENTAL.value:
                backup_result = self._execute_incremental_backup(job)
            elif job.backup_type == BackupType.DIFFERENTIAL.value:
                backup_result = self._execute_differential_backup(job)
            else:
                raise ValueError(f"Unknown backup type: {job.backup_type}")

            # Update job with results
            job.size_bytes = backup_result.get('original_size', 0)
            job.compressed_size_bytes = backup_result.get('compressed_size', 0)
            job.completed_at = datetime.now()
            job.execution_time = time.time() - start_time
            job.status = BackupStatus.COMPLETED.value

            result.update({
                'success': True,
                'end_time': job.completed_at,
                'execution_time': job.execution_time,
                'backup_details': backup_result
            })

            # Update statistics
            self._update_backup_stats(job, True)

            # Verify backup if enabled
            if self.verify_backups:
                verification_result = self._verify_backup(job)
                result['verification'] = verification_result

                if verification_result['verified']:
                    job.status = BackupStatus.VERIFIED.value

            logger.info(f"Backup job {job_id} completed successfully")

        except Exception as e:
            execution_time = time.time() - start_time
            job.execution_time = execution_time
            job.completed_at = datetime.now()
            job.status = BackupStatus.FAILED.value
            job.error_message = str(e)

            result.update({
                'success': False,
                'end_time': job.completed_at,
                'execution_time': execution_time,
                'error': str(e)
            })

            # Update statistics
            self._update_backup_stats(job, False)

            logger.error(f"Backup job {job_id} failed: {str(e)}")

        # Clean up
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]

        return result


    def _execute_full_backup(self, job: BackupJob) -> Dict[str, Any]:

        """
        Execute full backup
        执行完整备份

        Args:
            job: Backup job
                备份作业

        Returns:
            dict: Backup result
                  备份结果
        """
        source_path = Path(job.source_path)
        destination_path = Path(job.destination_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        # Create destination directory
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate original size
        original_size = self._calculate_directory_size(source_path)

        # Create backup
        if self.compression_enabled:
            backup_file = destination_path.with_suffix('.tar.gz')
            self._create_compressed_backup(source_path, backup_file)
            compressed_size = backup_file.stat().st_size
        else:
            backup_file = destination_path
            self._create_uncompressed_backup(source_path, backup_file)
            compressed_size = original_size

        return {
            'backup_type': 'full',
            'source_path': str(source_path),
            'destination_path': str(backup_file),
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / max(original_size, 1),
            'files_backed_up': self._count_files(source_path)
        }


    def _execute_incremental_backup(self, job: BackupJob) -> Dict[str, Any]:

        """
        Execute incremental backup
        执行增量备份

        Args:
            job: Backup job
                备份作业

        Returns:
            dict: Backup result
                  备份结果
        """
        # For incremental backup, we need to track last backup time
        last_backup_time = job.metadata.get('last_backup_time')

        if last_backup_time:
            last_backup = datetime.fromisoformat(last_backup_time)
        else:
            # First incremental backup, use full backup logic
            return self._execute_full_backup(job)

        source_path = Path(job.source_path)
        destination_path = Path(job.destination_path)

        # Find files modified since last backup
        modified_files = self._find_modified_files(source_path, last_backup)

        if not modified_files:
            return {
                'backup_type': 'incremental',
                'message': 'No files modified since last backup',
                'files_backed_up': 0,
                'original_size': 0,
                'compressed_size': 0
            }

        # Create incremental backup
        if self.compression_enabled:
            backup_file = destination_path.with_suffix('.tar.gz')
            self._create_incremental_backup(modified_files, source_path, backup_file)
            compressed_size = backup_file.stat().st_size
        else:
            backup_file = destination_path
            self._create_uncompressed_incremental_backup(modified_files, source_path, backup_file)
            compressed_size = sum(f.stat().st_size for f in modified_files)

        original_size = sum(f.stat().st_size for f in modified_files)

        return {
            'backup_type': 'incremental',
            'source_path': str(source_path),
            'destination_path': str(backup_file),
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / max(original_size, 1),
            'files_backed_up': len(modified_files),
            'last_backup_time': last_backup.isoformat()
        }


    def _execute_differential_backup(self, job: BackupJob) -> Dict[str, Any]:

        """
        Execute differential backup
        执行差异备份

        Args:
            job: Backup job
                备份作业

        Returns:
            dict: Backup result
                  备份结果
        """
        # Differential backup logic (simplified)
        # In practice, this would track against the last full backup
        return self._execute_full_backup(job)


    def _create_compressed_backup(self, source_path: Path, backup_file: Path) -> None:

        """
        Create compressed backup
        创建压缩备份

        Args:
            source_path: Source directory path
                        源目录路径
            backup_file: Backup file path
                        备份文件路径
        """
        import tarfile

        with tarfile.open(backup_file, 'w:gz', compresslevel=self.compression_level) as tar:
            tar.add(source_path, arcname=source_path.name)


    def _create_uncompressed_backup(self, source_path: Path, destination_path: Path) -> None:

        """
        Create uncompressed backup
        创建未压缩备份

        Args:
            source_path: Source directory path
                        源目录路径
            destination_path: Destination directory path
                             目标目录路径
        """
        if destination_path.exists():
            shutil.rmtree(destination_path)

        shutil.copytree(source_path, destination_path)


    def _create_incremental_backup(self, files: List[Path], source_path: Path, backup_file: Path) -> None:

        """
        Create incremental backup
        创建增量备份

        Args:
            files: List of files to backup
                  要备份的文件列表
            source_path: Base source path
                        基础源路径
            backup_file: Backup file path
                        备份文件路径
        """
        import tarfile

        with tarfile.open(backup_file, 'w:gz', compresslevel=self.compression_level) as tar:
            for file_path in files:
                relative_path = file_path.relative_to(source_path.parent)
                tar.add(file_path, arcname=str(relative_path))


    def _create_uncompressed_incremental_backup(self, files: List[Path], source_path: Path, destination_path: Path) -> None:

        """
        Create uncompressed incremental backup
        创建未压缩增量备份

        Args:
            files: List of files to backup
                  要备份的文件列表
            source_path: Base source path
                        基础源路径
            destination_path: Destination directory path
                             目标目录路径
        """
        destination_path.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            relative_path = file_path.relative_to(source_path.parent)
            dest_file = destination_path / relative_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_file)


    def _calculate_directory_size(self, path: Path) -> int:

        """
        Calculate total size of directory
        计算目录总大小

        Args:
            path: Directory path
                 目录路径

        Returns:
            int: Total size in bytes
                 总大小（字节）
        """
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size


    def _count_files(self, path: Path) -> int:

        """
        Count files in directory
        计算目录中的文件数

        Args:
            path: Directory path
                 目录路径

        Returns:
            int: Number of files
                 文件数量
        """
        return sum(1 for _ in path.rglob('*') if _.is_file())


    def _find_modified_files(self, path: Path, since: datetime) -> List[Path]:

        """
        Find files modified since given time
        查找自给定时间以来修改的文件

        Args:
            path: Directory path
                 目录路径
            since: Time threshold
                  时间阈值

        Returns:
            list: List of modified files
                  修改的文件列表
        """
        modified_files = []
        for file_path in path.rglob('*'):
            if file_path.is_file():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime > since:
                    modified_files.append(file_path)
        return modified_files


    def _verify_backup(self, job: BackupJob) -> Dict[str, Any]:

        """
        Verify backup integrity
        验证备份完整性

        Args:
            job: Backup job
                备份作业

        Returns:
            dict: Verification result
                  验证结果
        """
        # Simplified verification - in practice, this would do more thorough checks
        backup_path = Path(job.destination_path)
        if job.backup_type == BackupType.FULL.value and self.compression_enabled:
            backup_path = backup_path.with_suffix('.tar.gz')

        if backup_path.exists():
            return {
                'verified': True,
                'backup_size': backup_path.stat().st_size,
                'backup_exists': True
            }
        else:
            return {
                'verified': False,
                'error': 'Backup file does not exist',
                'backup_exists': False
            }


    def _update_backup_stats(self, job: BackupJob, success: bool) -> None:

        """
        Update backup statistics
        更新备份统计信息

        Args:
            job: Backup job
                备份作业
            success: Whether backup was successful
                    备份是否成功
        """
        self.stats['total_backups'] += 1

        if success:
            self.stats['successful_backups'] += 1
        else:
            self.stats['failed_backups'] += 1

        self.stats['total_backup_size'] += job.size_bytes
        self.stats['total_compressed_size'] += job.compressed_size_bytes

        # Update compression ratio
        if self.stats['total_backup_size'] > 0:
            self.stats['compression_ratio'] = (
                self.stats['total_compressed_size'] / self.stats['total_backup_size']
            )


    def get_backup_status(self, job_id: str) -> Optional[Dict[str, Any]]:

        """
        Get backup job status
        获取备份作业状态

        Args:
            job_id: Job identifier
                   作业标识符

        Returns:
            dict: Job status or None if not found
                  作业状态，如果未找到则返回None
        """
        if job_id in self.backup_jobs:
            return self.backup_jobs[job_id].to_dict()
        return None


    def list_backup_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:

        """
        List backup jobs with optional status filter
        列出备份作业，可选状态过滤

        Args:
            status_filter: Status to filter by (optional)
                          要过滤的状态（可选）

        Returns:
            list: List of backup jobs
                  备份作业列表
        """
        jobs = []
        for job in self.backup_jobs.values():
            if status_filter is None or job.status == status_filter:
                jobs.append(job.to_dict())
        return jobs


    def get_manager_stats(self) -> Dict[str, Any]:

        """
        Get backup manager statistics
        获取备份管理器统计信息

        Returns:
            dict: Manager statistics
                  管理器统计信息
        """
        return {
            'manager_name': self.manager_name,
            'total_jobs': len(self.backup_jobs),
            'active_jobs': len(self.active_jobs),
            'stats': self.stats
        }


    def cleanup_old_backups(self, retention_days: Optional[int] = None) -> Dict[str, Any]:

        """
        Clean up old backup files
        清理旧备份文件

        Args:
            retention_days: Number of days to retain backups
                           保留备份的天数

        Returns:
            dict: Cleanup result
                  清理结果
        """
        if retention_days is None:
            retention_days = self.backup_retention_days

        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cleaned_count = 0
        cleaned_size = 0

        for job in self.backup_jobs.values():
            if job.completed_at and job.completed_at < cutoff_date:
                try:
                    backup_path = Path(job.destination_path)
                    if job.backup_type == BackupType.FULL.value and self.compression_enabled:
                        backup_path = backup_path.with_suffix('.tar.gz')

                    if backup_path.exists():
                        size = backup_path.stat().st_size
                        backup_path.unlink()
                        cleaned_size += size
                        cleaned_count += 1

                except Exception as e:
                    logger.error(f"Failed to cleanup backup {job.job_id}: {str(e)}")

        return {
            'backups_cleaned': cleaned_count,
            'space_reclaimed': cleaned_size,
            'retention_days': retention_days
        }


# Global backup manager instance
# 全局备份管理器实例
backup_manager = BackupManager()

__all__ = [
    'BackupType',
    'BackupStatus',
    'StorageType',
    'BackupJob',
    'RecoveryJob',
    'BackupManager',
    'backup_manager'
]
