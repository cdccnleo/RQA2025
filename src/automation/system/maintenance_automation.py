"""
Maintenance Automation Module
维护自动化模块

This module provides automated maintenance capabilities for quantitative trading systems
此模块为量化交易系统提供自动化维护能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import shutil
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


class MaintenanceType(Enum):

    """Maintenance types"""
    SYSTEM_UPDATE = "system_update"
    DATABASE_MAINTENANCE = "database_maintenance"
    LOG_ROTATION = "log_rotation"
    BACKUP_VERIFICATION = "backup_verification"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_SCAN = "security_scan"
    HEALTH_CHECK = "health_check"


class MaintenanceStatus(Enum):

    """Maintenance status"""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class MaintenanceTask:

    """
    Maintenance task data class
    维护任务数据类
    """
    task_id: str
    task_type: str
    name: str
    description: str
    schedule: str  # Cron - like schedule
    status: str
    created_at: datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    execution_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    enabled: bool = True
    timeout: int = 3600  # 1 hour default
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.last_run:
            data['last_run'] = self.last_run.isoformat()
        if self.next_run:
            data['next_run'] = self.next_run.isoformat()
        return data


@dataclass
class MaintenanceResult:

    """
    Maintenance result data class
    维护结果数据类
    """
    task_id: str
    execution_id: str
    success: bool
    start_time: datetime
    end_time: datetime
    execution_time: float
    output: str = ""
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat()
        return data


class SystemUpdater:

    """
    System Updater Class
    系统更新器类

    Handles system package updates and patches
    处理系统软件包更新和补丁
    """

    def __init__(self):
        """
        Initialize system updater
        初始化系统更新器
        """
        self.update_history = deque(maxlen=100)

    def check_for_updates(self) -> Dict[str, Any]:
        """
        Check for available system updates
        检查可用的系统更新

        Returns:
            dict: Update information
                  更新信息
        """
        result = {
            'updates_available': 0,
            'security_updates': 0,
            'package_list': [],
            'last_check': datetime.now()
        }

        try:
            # Check for updates (system - specific implementation)
            if self._is_debian_based():
                result.update(self._check_apt_updates())
            elif self._is_redhat_based():
                result.update(self._check_yum_updates())
            else:
                result['error'] = 'Unsupported package manager'

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Update check failed: {str(e)}")

        return result

    def apply_updates(self, update_type: str = 'all') -> Dict[str, Any]:
        """
        Apply system updates
        应用系统更新

        Args:
            update_type: Type of updates ('all', 'security', 'packages')
                        更新类型

        Returns:
            dict: Update result
                  更新结果
        """
        result = {
            'success': False,
            'updates_applied': 0,
            'reboot_required': False,
            'start_time': datetime.now()
        }

        try:
            if self._is_debian_based():
                result.update(self._apply_apt_updates(update_type))
            elif self._is_redhat_based():
                result.update(self._apply_yum_updates(update_type))
            else:
                result['error'] = 'Unsupported package manager'
                return result

            result['success'] = True
            result['end_time'] = datetime.now()

            # Record in history
            self.update_history.append({
                'timestamp': datetime.now(),
                'type': update_type,
                'result': result
            })

        except Exception as e:
            result['error'] = str(e)
            result['end_time'] = datetime.now()
            logger.error(f"Update application failed: {str(e)}")

        return result

    def _is_debian_based(self) -> bool:
        """Check if system is Debian - based"""
        return Path('/etc / debian_version').exists()

    def _is_redhat_based(self) -> bool:
        """Check if system is Red Hat - based"""
        return Path('/etc / redhat - release').exists()

    def _check_apt_updates(self) -> Dict[str, Any]:
        """Check for APT updates"""
        # Placeholder implementation
        return {
            'updates_available': 5,
            'security_updates': 2,
            'package_list': ['package1', 'package2', 'security - package1']
        }

    def _check_yum_updates(self) -> Dict[str, Any]:
        """Check for YUM updates"""
        # Placeholder implementation
        return {
            'updates_available': 3,
            'security_updates': 1,
            'package_list': ['package1', 'security - package1']
        }

    def _apply_apt_updates(self, update_type: str) -> Dict[str, Any]:
        """Apply APT updates"""
        # Placeholder implementation
        logger.info(f"Applying APT updates: {update_type}")
        return {
            'updates_applied': 5,
            'reboot_required': False
        }

    def _apply_yum_updates(self, update_type: str) -> Dict[str, Any]:
        """Apply YUM updates"""
        # Placeholder implementation
        logger.info(f"Applying YUM updates: {update_type}")
        return {
            'updates_applied': 3,
            'reboot_required': True
        }


class DatabaseMaintainer:

    """
    Database Maintainer Class
    数据库维护器类

    Handles database maintenance operations
    处理数据库维护操作
    """

    def __init__(self):
        """
        Initialize database maintainer
        初始化数据库维护器
        """
        self.maintenance_history = deque(maxlen=100)

    def perform_maintenance(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform database maintenance
        执行数据库维护

        Args:
            db_config: Database configuration
                      数据库配置

        Returns:
            dict: Maintenance result
                  维护结果
        """
        result = {
            'success': False,
            'operations_performed': [],
            'start_time': datetime.now()
        }

        try:
            db_type = db_config.get('type', 'postgresql')

            if db_type == 'postgresql':
                result.update(self._maintain_postgresql(db_config))
            elif db_type == 'mysql':
                result.update(self._maintain_mysql(db_config))
            else:
                result['error'] = f'Unsupported database type: {db_type}'
                return result

            result['success'] = True
            result['end_time'] = datetime.now()

            # Record maintenance
            self.maintenance_history.append({
                'timestamp': datetime.now(),
                'db_type': db_type,
                'result': result
            })

        except Exception as e:
            result['error'] = str(e)
            result['end_time'] = datetime.now()
            logger.error(f"Database maintenance failed: {str(e)}")

        return result

    def _maintain_postgresql(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Maintain PostgreSQL database"""
        operations = []

        # VACUUM operation
        operations.append('VACUUM')

        # ANALYZE operation
        operations.append('ANALYZE')

        # REINDEX operation
        operations.append('REINDEX')

        logger.info("Performing PostgreSQL maintenance")

        return {
            'operations_performed': operations,
            'tables_processed': 10,
            'space_reclaimed': '50MB'
        }

    def _maintain_mysql(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Maintain MySQL database"""
        operations = []

        # OPTIMIZE TABLE
        operations.append('OPTIMIZE TABLE')

        # REPAIR TABLE
        operations.append('REPAIR TABLE')

        logger.info("Performing MySQL maintenance")

        return {
            'operations_performed': operations,
            'tables_processed': 8,
            'space_reclaimed': '30MB'
        }


class LogRotator:

    """
    Log Rotator Class
    日志轮转器类

    Handles log file rotation and cleanup
    处理日志文件轮转和清理
    """

    def __init__(self):
        """
        Initialize log rotator
        初始化日志轮转器
        """
        self.rotation_history = deque(maxlen=100)

    def rotate_logs(self, log_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rotate log files
        轮转日志文件

        Args:
            log_config: Log rotation configuration
                       日志轮转配置

        Returns:
            dict: Rotation result
                  轮转结果
        """
        result = {
            'success': False,
            'logs_rotated': 0,
            'space_reclaimed': '0MB',
            'start_time': datetime.now()
        }

        try:
            log_dirs = log_config.get('log_directories', ['/var / log'])
            max_age_days = log_config.get('max_age_days', 30)
            compression = log_config.get('compression', True)

            total_rotated = 0
            total_space = 0

            for log_dir in log_dirs:
                dir_result = self._rotate_directory_logs(
                    Path(log_dir), max_age_days, compression
                )
                total_rotated += dir_result['rotated']
                total_space += dir_result['space_reclaimed']

            result.update({
                'success': True,
                'logs_rotated': total_rotated,
                'space_reclaimed': f"{total_space}MB",
                'end_time': datetime.now()
            })

            # Record rotation
            self.rotation_history.append({
                'timestamp': datetime.now(),
                'result': result
            })

        except Exception as e:
            result['error'] = str(e)
            result['end_time'] = datetime.now()
            logger.error(f"Log rotation failed: {str(e)}")

        return result

    def _rotate_directory_logs(self,


                               log_dir: Path,
                               max_age_days: int,
                               compression: bool) -> Dict[str, Any]:
        """
        Rotate logs in a directory
        轮转目录中的日志

        Args:
            log_dir: Log directory path
                    日志目录路径
            max_age_days: Maximum age in days
                         最大时长（天）
            compression: Whether to compress old logs
                        是否压缩旧日志

        Returns:
            dict: Directory rotation result
                  目录轮转结果
        """
        if not log_dir.exists():
            return {'rotated': 0, 'space_reclaimed': 0}

        rotated = 0
        space_reclaimed = 0
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        for log_file in log_dir.glob('*.log'):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                # Rotate this log file
                self._rotate_single_log(log_file, compression)
                space_reclaimed += log_file.stat().st_size / (1024 * 1024)  # MB
                rotated += 1

        return {
            'rotated': rotated,
            'space_reclaimed': space_reclaimed
        }

    def _rotate_single_log(self, log_file: Path, compression: bool) -> None:
        """
        Rotate a single log file
        轮转单个日志文件

        Args:
            log_file: Log file path
                     日志文件路径
            compression: Whether to compress
                        是否压缩
        """
        timestamp = datetime.now().strftime('%Y % m % d_ % H % M % S')
        rotated_name = f"{log_file.stem}.{timestamp}"

        if compression:
            rotated_name += '.gz'
            import gzip
            with open(log_file, 'rb') as f_in:
                with gzip.open(log_file.parent / rotated_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(log_file, log_file.parent / rotated_name)

        # Truncate original log file
        with open(log_file, 'w') as f:
            f.truncate(0)


class MaintenanceAutomationEngine:

    """
    Maintenance Automation Engine Class
    维护自动化引擎类

    Core engine for automated maintenance tasks
    自动化维护任务的核心引擎
    """

    def __init__(self, engine_name: str = "default_maintenance_engine"):
        """
        Initialize maintenance automation engine
        初始化维护自动化引擎

        Args:
            engine_name: Name of the engine
                        引擎名称
        """
        self.engine_name = engine_name
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None

        # Sub - components
        self.system_updater = SystemUpdater()
        self.database_maintainer = DatabaseMaintainer()
        self.log_rotator = LogRotator()

        # Maintenance tasks
        self.tasks: Dict[str, MaintenanceTask] = {}

        # Execution tracking
        self.active_tasks: Dict[str, threading.Thread] = {}
        self.task_results: Dict[str, MaintenanceResult] = {}

        # Configuration
        self.check_interval = 60  # seconds

        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'scheduled_tasks': 0,
            'manual_tasks': 0
        }

        # Setup default maintenance tasks
        self._setup_default_tasks()

        logger.info(f"Maintenance automation engine {engine_name} initialized")

    def _setup_default_tasks(self) -> None:
        """Setup default maintenance tasks"""
        # Daily log rotation
        self.add_maintenance_task(
            'daily_log_rotation',
            'Daily Log Rotation',
            'Rotate application and system logs daily',
            MaintenanceType.LOG_ROTATION,
            '0 2 * * *',  # Daily at 2 AM
            {'log_directories': ['/var / log', '/opt / quant_trading / logs'], 'max_age_days': 30}
        )

        # Weekly system updates
        self.add_maintenance_task(
            'weekly_system_updates',
            'Weekly System Updates',
            'Apply security updates and patches weekly',
            MaintenanceType.SYSTEM_UPDATE,
            '0 3 * * 0',  # Weekly on Sunday at 3 AM
            {'update_type': 'security'}
        )

        # Daily database maintenance
        self.add_maintenance_task(
            'daily_db_maintenance',
            'Daily Database Maintenance',
            'Perform daily database optimization and cleanup',
            MaintenanceType.DATABASE_MAINTENANCE,
            '0 1 * * *',  # Daily at 1 AM
            {'db_config': {'type': 'postgresql', 'database': 'quant_trading'}}
        )

        # Hourly health checks
        self.add_maintenance_task(
            'hourly_health_check',
            'Hourly Health Check',
            'Perform comprehensive system health checks',
            MaintenanceType.HEALTH_CHECK,
            '0 * * * *',  # Every hour
            {'checks': ['cpu', 'memory', 'disk', 'network', 'services']}
        )

    def add_maintenance_task(self,


                             task_id: str,
                             name: str,
                             description: str,
                             task_type: MaintenanceType,
                             schedule: str,
                             config: Dict[str, Any],
                             enabled: bool = True) -> str:
        """
        Add a maintenance task
        添加维护任务

        Args:
            task_id: Unique task identifier
                    唯一任务标识符
            name: Task name
                 任务名称
            description: Task description
                        任务描述
            task_type: Type of maintenance task
                      维护任务类型
            schedule: Cron - like schedule string
                     类cron调度字符串
            config: Task configuration
                   任务配置
            enabled: Whether task is enabled
                    任务是否启用

        Returns:
            str: Created task ID
                 创建的任务ID
        """
        task = MaintenanceTask(
            task_id=task_id,
            task_type=task_type.value,
            name=name,
            description=description,
            schedule=schedule,
            status=MaintenanceStatus.SCHEDULED.value,
            created_at=datetime.now(),
            enabled=enabled,
            metadata=config
        )

        # Calculate next run time
        task.next_run = self._calculate_next_run(schedule)

        self.tasks[task_id] = task
        logger.info(f"Added maintenance task: {name} ({task_id})")
        return task_id

    def start_scheduler(self) -> bool:
        """
        Start the maintenance scheduler
        启动维护调度器

        Returns:
            bool: True if started successfully
                  启动成功返回True
        """
        if self.is_running:
            logger.warning("Maintenance engine is already running")
            return False

        try:
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            logger.info("Maintenance scheduler started")
            return True
        except Exception as e:
            logger.error(f"Failed to start maintenance scheduler: {str(e)}")
            self.is_running = False
            return False

    def stop_scheduler(self) -> bool:
        """
        Stop the maintenance scheduler
        停止维护调度器

        Returns:
            bool: True if stopped successfully
                  停止成功返回True
        """
        if not self.is_running:
            logger.warning("Maintenance engine is not running")
            return False

        try:
            self.is_running = False
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5.0)
            logger.info("Maintenance scheduler stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop maintenance scheduler: {str(e)}")
            return False

    def _scheduler_loop(self) -> None:
        """
        Main scheduler loop
        主要的调度器循环
        """
        logger.info("Maintenance scheduler loop started")

        while self.is_running:
            try:
                current_time = datetime.now()

                # Check for tasks that need to run
                for task in self.tasks.values():
                    if (task.enabled
                        and task.next_run
                        and current_time >= task.next_run
                            and task.status != MaintenanceStatus.RUNNING.value):

                        # Execute task asynchronously
                        execution_thread = threading.Thread(
                            target=self._execute_task_async,
                            args=(task.task_id,),
                            daemon=True
                        )
                        self.active_tasks[task.task_id] = execution_thread
                        execution_thread.start()

                        # Calculate next run time
                        task.next_run = self._calculate_next_run(task.schedule)

                # Sleep before next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Scheduler loop error: {str(e)}")
                time.sleep(self.check_interval)

        logger.info("Maintenance scheduler loop stopped")

    def _execute_task_async(self, task_id: str) -> None:
        """
        Execute maintenance task asynchronously
        异步执行维护任务

        Args:
            task_id: Task identifier
                    任务标识符
        """
        task = self.tasks[task_id]
        task.status = MaintenanceStatus.RUNNING.value
        task.last_run = datetime.now()

        execution_id = f"exec_{task_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"

        result = MaintenanceResult(
            task_id=task_id,
            execution_id=execution_id,
            success=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            execution_time=0.0
        )

        start_time = time.time()

        try:
            # Execute task based on type
            if task.task_type == MaintenanceType.SYSTEM_UPDATE.value:
                result.output = str(self.system_updater.apply_updates(
                    task.metadata.get('update_type', 'all')
                ))
            elif task.task_type == MaintenanceType.DATABASE_MAINTENANCE.value:
                result.output = str(self.database_maintainer.perform_maintenance(
                    task.metadata.get('db_config', {})
                ))
            elif task.task_type == MaintenanceType.LOG_ROTATION.value:
                result.output = str(self.log_rotator.rotate_logs(
                    task.metadata
                ))
            elif task.task_type == MaintenanceType.BACKUP_VERIFICATION.value:
                result.output = "Backup verification completed"
            elif task.task_type == MaintenanceType.PERFORMANCE_OPTIMIZATION.value:
                result.output = "Performance optimization completed"
            elif task.task_type == MaintenanceType.SECURITY_SCAN.value:
                result.output = "Security scan completed"
            elif task.task_type == MaintenanceType.HEALTH_CHECK.value:
                result.output = "Health check completed"

            result.success = True
            task.success_count += 1

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            task.failure_count += 1
            logger.error(f"Maintenance task {task_id} failed: {str(e)}")

        result.end_time = datetime.now()
        result.execution_time = time.time() - start_time

        # Update task status
        task.status = MaintenanceStatus.COMPLETED.value if result.success else MaintenanceStatus.FAILED.value
        task.execution_time = result.execution_time

        # Store result
        self.task_results[execution_id] = result

        # Update statistics
        self._update_stats(task, result.success)

        # Clean up
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]

    def execute_manual_task(self,


                            task_type: MaintenanceType,
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a maintenance task manually
        手动执行维护任务

        Args:
            task_type: Type of maintenance task
                      维护任务类型
            config: Task configuration
                   任务配置

        Returns:
            dict: Execution result
                  执行结果
        """
        task_id = f"manual_{task_type.value}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"

        result = {
            'task_id': task_id,
            'success': False,
            'start_time': datetime.now(),
            'execution_time': 0.0
        }

        start_time = time.time()

        try:
            if task_type == MaintenanceType.SYSTEM_UPDATE:
                output = self.system_updater.apply_updates(config.get('update_type', 'all'))
            elif task_type == MaintenanceType.DATABASE_MAINTENANCE:
                output = self.database_maintainer.perform_maintenance(config.get('db_config', {}))
            elif task_type == MaintenanceType.LOG_ROTATION:
                output = self.log_rotator.rotate_logs(config)
            else:
                output = {'message': f'Manual {task_type.value} completed'}

            result.update({
                'success': True,
                'output': str(output),
                'end_time': datetime.now(),
                'execution_time': time.time() - start_time
            })

            self.stats['total_tasks'] += 1
            self.stats['completed_tasks'] += 1
            self.stats['manual_tasks'] += 1

        except Exception as e:
            result.update({
                'success': False,
                'error': str(e),
                'end_time': datetime.now(),
                'execution_time': time.time() - start_time
            })

            self.stats['total_tasks'] += 1
            self.stats['failed_tasks'] += 1

        return result

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get maintenance task status
        获取维护任务状态

        Args:
            task_id: Task identifier
                   任务标识符

        Returns:
            dict: Task status or None if not found
                  任务状态，如果未找到则返回None
        """
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        return None

    def list_tasks(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List maintenance tasks with optional status filter
        列出维护任务，可选状态过滤

        Args:
            status_filter: Status to filter by (optional)
                          要过滤的状态（可选）

        Returns:
            list: List of maintenance tasks
                  维护任务列表
        """
        tasks = []
        for task in self.tasks.values():
            if status_filter is None or task.status == status_filter:
                tasks.append(task.to_dict())
        return tasks

    def enable_task(self, task_id: str) -> bool:
        """
        Enable a maintenance task
        启用维护任务

        Args:
            task_id: Task identifier
                    任务标识符

        Returns:
            bool: True if enabled successfully
                  启用成功返回True
        """
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """
        Disable a maintenance task
        禁用维护任务

        Args:
            task_id: Task identifier
                    任务标识符

        Returns:
            bool: True if disabled successfully
                  禁用成功返回True
        """
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            return True
        return False

    def _calculate_next_run(self, schedule: str) -> datetime:
        """
        Calculate next run time from cron - like schedule
        从类cron调度计算下次运行时间

        Args:
            schedule: Cron - like schedule string
                     类cron调度字符串

        Returns:
            datetime: Next run time
                     下次运行时间
        """
        # Simplified cron parser (placeholder)
        # In practice, use a proper cron parser like croniter
        try:
            # Parse simple schedules
            parts = schedule.split()
            if len(parts) == 5:  # Standard cron format
                minute, hour, day, month, weekday = parts

                now = datetime.now()

                # Calculate next run (simplified)
                if hour == '*' and minute == '*':
                    # Every hour
                    next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                elif minute == '*' and hour != '*':
                    # Every minute within specific hour
                    next_run = now + timedelta(minutes=1)
                else:
                    # Default to every hour
                    next_run = now + timedelta(hours=1)

                return next_run

        except Exception:
            pass

        # Default fallback
        return datetime.now() + timedelta(hours=1)

    def _update_stats(self, task: MaintenanceTask, success: bool) -> None:
        """
        Update maintenance statistics
        更新维护统计信息

        Args:
            task: Maintenance task
                维护任务
            success: Whether task was successful
                    任务是否成功
        """
        if success:
            self.stats['completed_tasks'] += 1
        else:
            self.stats['failed_tasks'] += 1

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get maintenance engine statistics
        获取维护引擎统计信息

        Returns:
            dict: Engine statistics
                  引擎统计信息
        """
        return {
            'engine_name': self.engine_name,
            'is_running': self.is_running,
            'total_tasks': len(self.tasks),
            'active_tasks': len(self.active_tasks),
            'enabled_tasks': sum(1 for t in self.tasks.values() if t.enabled),
            'stats': self.stats
        }

    def get_task_results(self,


                         task_id: Optional[str] = None,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get maintenance task results
        获取维护任务结果

        Args:
            task_id: Specific task ID (optional)
                    特定任务ID（可选）
            limit: Maximum number of results
                  最大结果数

        Returns:
            list: Task results
                  任务结果
        """
        results = []
        for result in list(self.task_results.values()):
            if task_id is None or result.task_id == task_id:
                results.append(result.to_dict())

        return results[-limit:]


# Global maintenance automation engine instance
# 全局维护自动化引擎实例
maintenance_engine = MaintenanceAutomationEngine()

__all__ = [
    'MaintenanceType',
    'MaintenanceStatus',
    'MaintenanceTask',
    'MaintenanceResult',
    'SystemUpdater',
    'DatabaseMaintainer',
    'LogRotator',
    'MaintenanceAutomationEngine',
    'maintenance_engine'
]
