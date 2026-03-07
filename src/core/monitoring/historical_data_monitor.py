#!/usr/bin/env python3
"""
历史数据采集监控和调度服务

提供专门的历史数据采集任务调度、监控和状态管理功能。
支持任务队列管理、并发控制、进度跟踪和告警通知。
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.core.monitoring.data_collection_monitor import AlertLevel, AlertType, Alert

logger = logging.getLogger(__name__)


class HistoricalTaskStatus(Enum):
    """历史数据采集任务状态"""
    PENDING = "pending"         # 等待执行
    RUNNING = "running"         # 正在执行
    PAUSED = "paused"          # 已暂停
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"          # 执行失败
    CANCELLED = "cancelled"     # 已取消


class HistoricalTaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class HistoricalCollectionTask:
    """历史数据采集任务"""
    task_id: str
    symbol: str
    start_date: str
    end_date: str
    data_types: List[str]
    priority: HistoricalTaskPriority = HistoricalTaskPriority.NORMAL
    status: HistoricalTaskStatus = HistoricalTaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    records_collected: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration: Optional[float] = None
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionWorker:
    """采集工作进程"""
    worker_id: str
    active_tasks: List[str] = field(default_factory=list)
    max_concurrent: int = 2
    is_active: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    performance_stats: Dict[str, Any] = field(default_factory=dict)


class HistoricalDataMonitor:
    """
    历史数据采集监控和调度器

    提供以下功能：
    - 任务队列管理和调度
    - 并发控制和负载均衡
    - 实时进度监控和状态跟踪
    - 智能重试和错误恢复
    - 性能指标收集和告警
    - 任务控制接口
    """

    def __init__(self, alert_callback: Optional[Callable[[Alert], None]] = None,
                 websocket_callback: Optional[Callable] = None):
        """
        初始化历史数据采集监控器

        Args:
            alert_callback: 告警回调函数
            websocket_callback: WebSocket广播回调函数
        """
        self.alert_callback = alert_callback
        self.websocket_callback = websocket_callback

        # 任务管理
        self.tasks: Dict[str, HistoricalCollectionTask] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.pending_tasks: List[tuple] = []  # 用于兼容不同队列实现
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self.alerts: List[Alert] = []

        # 工作进程管理
        self.workers: Dict[str, CollectionWorker] = {}
        self.worker_heartbeat_timeout = 90  # 90秒心跳超时，与调度器保持一致

        # 调度控制
        self.max_concurrent_tasks = 5
        self.active_tasks = 0
        self.is_scheduler_active = False
        self.scheduler_task: Optional[asyncio.Task] = None

        # 监控指标
        self.monitoring_enabled = True
        self.stats = {
            'total_tasks_created': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_records_collected': 0,
            'avg_collection_time': 0.0,
            'success_rate': 0.0,
            'active_workers': 0,
            'scheduler_start_time': None,
            'last_activity_time': None
        }

        # 配置
        self.task_timeout = 3600  # 1小时超时
        self.progress_update_interval = 5  # 5秒更新一次进度
        self.cleanup_interval = 3600  # 1小时清理一次过期任务

        logger.info("历史数据采集监控器已初始化")

    async def start_scheduler(self):
        """启动任务调度器"""
        if self.is_scheduler_active:
            logger.warning("调度器已在运行")
            return

        self.is_scheduler_active = True
        self.stats['scheduler_start_time'] = time.time()
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())

        logger.info("历史数据采集调度器已启动")

    async def stop_scheduler(self):
        """停止任务调度器"""
        if not self.is_scheduler_active:
            return

        self.is_scheduler_active = False

        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("历史数据采集调度器已停止")

    def create_task(self, symbol: str, start_date: str, end_date: str,
                   data_types: List[str], priority: HistoricalTaskPriority = HistoricalTaskPriority.NORMAL,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        创建历史数据采集任务

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            data_types: 数据类型列表
            priority: 任务优先级
            metadata: 元数据

        Returns:
            任务ID
        """
        task_id = f"hist_{symbol}_{int(time.time())}_{hash(f'{symbol}_{start_date}_{end_date}') % 1000}"

        task = HistoricalCollectionTask(
            task_id=task_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            priority=priority,
            metadata=metadata or {}
        )

        self.tasks[task_id] = task
        self.stats['total_tasks_created'] += 1
        self.stats['last_activity_time'] = time.time()

        # 添加到队列 - 使用直接put而不是asyncio.create_task，避免事件循环问题
        priority_value = -priority.value  # 负值使高优先级排在前面
        # 直接添加到队列，不使用asyncio.create_task
        if hasattr(self.task_queue, 'put_nowait'):
            self.task_queue.put_nowait((priority_value, task_id))
        else:
            # 兼容不同队列实现
            self.pending_tasks.append((priority_value, task_id))

        logger.info(f"创建历史采集任务: {task_id} (标的: {symbol}, 优先级: {priority.name})")

        return task_id

    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功取消
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        if task.status in [HistoricalTaskStatus.COMPLETED, HistoricalTaskStatus.FAILED]:
            return False

        task.status = HistoricalTaskStatus.CANCELLED
        task.completed_at = time.time()

        if task.worker_id and task.worker_id in self.workers:
            worker = self.workers[task.worker_id]
            if task_id in worker.active_tasks:
                worker.active_tasks.remove(task_id)

        logger.info(f"任务已取消: {task_id}")
        return True

    async def pause_task(self, task_id: str) -> bool:
        """
        暂停任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功暂停
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        if task.status != HistoricalTaskStatus.RUNNING:
            return False

        task.status = HistoricalTaskStatus.PAUSED

        if task.worker_id and task.worker_id in self.workers:
            worker = self.workers[task.worker_id]
            if task_id in worker.active_tasks:
                worker.active_tasks.remove(task_id)

        logger.info(f"任务已暂停: {task_id}")
        return True

    async def resume_task(self, task_id: str) -> bool:
        """
        恢复任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功恢复
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        if task.status != HistoricalTaskStatus.PAUSED:
            return False

        task.status = HistoricalTaskStatus.PENDING

        # 重新添加到队列 - 使用直接put而不是asyncio.create_task，避免事件循环问题
        priority_value = -task.priority.value
        # 直接添加到pending_tasks，由调度器循环处理
        self.pending_tasks.append((priority_value, task_id))

        logger.info(f"任务已恢复: {task_id}")
        return True

    def register_worker(self, worker_id: str, max_concurrent: int = 2) -> bool:
        """
        注册采集工作进程

        Args:
            worker_id: 工作进程ID
            max_concurrent: 最大并发任务数

        Returns:
            是否成功注册
        """
        if worker_id in self.workers:
            logger.warning(f"工作进程已存在: {worker_id}")
            return False

        worker = CollectionWorker(
            worker_id=worker_id,
            max_concurrent=max_concurrent
        )

        self.workers[worker_id] = worker
        self.stats['active_workers'] = len([w for w in self.workers.values() if w.is_active])

        logger.info(f"注册工作进程: {worker_id} (最大并发: {max_concurrent})")
        return True

    def unregister_worker(self, worker_id: str) -> bool:
        """
        注销采集工作进程

        Args:
            worker_id: 工作进程ID

        Returns:
            是否成功注销
        """
        if worker_id not in self.workers:
            return False

        worker = self.workers[worker_id]

        # 重新排队该工作进程的任务
        for task_id in worker.active_tasks:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == HistoricalTaskStatus.RUNNING:
                    task.status = HistoricalTaskStatus.PENDING
                    task.worker_id = None

                    # 重新添加到队列 - 使用直接put而不是asyncio.create_task，避免事件循环问题
                    priority_value = -task.priority.value
                    # 直接添加到pending_tasks，由调度器循环处理
                    self.pending_tasks.append((priority_value, task_id))

        del self.workers[worker_id]
        self.stats['active_workers'] = len([w for w in self.workers.values() if w.is_active])

        logger.info(f"注销工作进程: {worker_id}")
        return True

    def update_worker_heartbeat(self, worker_id: str):
        """更新工作进程心跳"""
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = time.time()

    def update_task_progress(self, task_id: str, progress: float,
                           records_collected: int = 0, error_message: Optional[str] = None):
        """
        更新任务进度

        Args:
            task_id: 任务ID
            progress: 进度百分比 (0.0-1.0)
            records_collected: 已采集记录数
            error_message: 错误消息
        """
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task.progress = progress
        if records_collected > 0:
            task.records_collected = records_collected
        if error_message:
            task.error_message = error_message

        self.stats['last_activity_time'] = time.time()

        # WebSocket广播任务进度更新
        if self.websocket_callback:
            try:
                # 直接调用同步方法或使用其他方式处理，避免事件循环问题
                # 这里只记录日志，不使用asyncio.create_task
                logger.debug(f"任务进度更新: {task_id} -> {progress:.2f}, 采集记录数: {records_collected}")
            except Exception as e:
                logger.error(f"处理任务进度更新失败: {e}")

    def update_data_quality(self, task_id: str, quality_score: float):
        """
        更新任务数据质量评分

        Args:
            task_id: 任务ID
            quality_score: 质量评分（0.0-1.0）
        """
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        # 这里可以存储质量评分，用于监控和报告
        # 暂时只记录日志
        logger.info(f"任务 {task_id} 数据质量评分: {quality_score:.2f}")

        # 检查质量告警
        if quality_score < 0.7:  # 质量阈值
            self._trigger_alert(
                AlertType.DATA_QUALITY_LOW,
                AlertLevel.WARNING,
                f"历史数据采集质量评分过低: {quality_score:.2f}",
                task_id,
                {'quality_score': quality_score, 'task_symbol': task.symbol}
            )

    async def _scheduler_loop(self):
        """调度器主循环"""
        logger.info("历史数据采集调度器循环启动")

        while self.is_scheduler_active:
            try:
                # 清理超期任务和不活跃的工作进程
                await self._cleanup_tasks()

                # 分配任务给工作进程
                await self._assign_tasks()

                # 检查超时任务
                await self._check_timeouts()

                await asyncio.sleep(1)  # 1秒调度间隔

            except Exception as e:
                logger.error(f"调度器循环异常: {e}")
                await asyncio.sleep(5)

        logger.info("历史数据采集调度器循环结束")

    async def _assign_tasks(self):
        """分配任务给工作进程"""
        # 查找可用的工作进程
        available_workers = [
            worker for worker in self.workers.values()
            if worker.is_active and
            len(worker.active_tasks) < worker.max_concurrent and
            time.time() - worker.last_heartbeat < self.worker_heartbeat_timeout
        ]

        if not available_workers:
            return

        # 1. 先将pending_tasks中的任务添加到task_queue
        while self.pending_tasks:
            priority_value, task_id = self.pending_tasks.pop(0)
            await self.task_queue.put((priority_value, task_id))

        # 2. 分配任务
        assigned_count = 0
        while not self.task_queue.empty() and available_workers and assigned_count < 5:
            try:
                priority_value, task_id = self.task_queue.get_nowait()

                if task_id not in self.tasks:
                    continue

                task = self.tasks[task_id]
                if task.status != HistoricalTaskStatus.PENDING:
                    continue

                # 选择负载最轻的工作进程
                available_workers.sort(key=lambda w: len(w.active_tasks))
                worker = available_workers[0]

                # 分配任务
                task.status = HistoricalTaskStatus.RUNNING
                task.started_at = time.time()
                task.worker_id = worker.worker_id
                worker.active_tasks.append(task_id)

                # 通知工作进程（这里需要具体的通知机制）
                await self._notify_worker_task_assigned(worker.worker_id, task)

                assigned_count += 1
                logger.debug(f"分配任务 {task_id} 给工作进程 {worker.worker_id}")

            except asyncio.QueueEmpty:
                break

    async def _notify_worker_task_assigned(self, worker_id: str, task: HistoricalCollectionTask):
        """通知工作进程任务分配"""
        # 这里应该实现具体的工作进程通知机制
        # 可能是通过消息队列、WebSocket或共享存储
        logger.debug(f"通知工作进程 {worker_id} 执行任务 {task.task_id}")

    async def _cleanup_tasks(self):
        """清理过期任务和不活跃的工作进程"""
        current_time = time.time()

        # 清理不活跃的工作进程
        inactive_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if current_time - worker.last_heartbeat > self.worker_heartbeat_timeout
        ]

        for worker_id in inactive_workers:
            logger.warning(f"工作进程心跳超时: {worker_id}")
            self.unregister_worker(worker_id)

        # 清理过期任务（可选）
        # 这里可以清理过期的已完成任务

    async def _check_timeouts(self):
        """检查超时任务"""
        current_time = time.time()

        for task in self.tasks.values():
            if task.status == HistoricalTaskStatus.RUNNING and task.started_at:
                if current_time - task.started_at > self.task_timeout:
                    logger.warning(f"任务超时: {task.task_id}")

                    task.status = HistoricalTaskStatus.FAILED
                    task.error_message = f"任务执行超时 ({self.task_timeout}秒)"
                    task.completed_at = current_time

                    # 从工作进程中移除
                    if task.worker_id and task.worker_id in self.workers:
                        worker = self.workers[task.worker_id]
                        if task.task_id in worker.active_tasks:
                            worker.active_tasks.remove(task.task_id)

                    self.failed_tasks.append(task.task_id)
                    self.stats['total_tasks_failed'] += 1

                    # 触发告警
                    await self._trigger_alert(
                        AlertType.DATA_COLLECTION_FAILED,
                        AlertLevel.WARNING,
                        f"历史数据采集任务超时: {task.symbol}",
                        f"historical_{task.symbol}",
                        {
                            'task_id': task.task_id,
                            'timeout_seconds': self.task_timeout,
                            'symbol': task.symbol
                        }
                    )

    async def _trigger_alert(self, alert_type: AlertType, level: AlertLevel,
                           message: str, source_id: str, details: Dict[str, Any]):
        """触发告警"""
        if self.alert_callback:
            alert = Alert(
                alert_id=f"{alert_type.value}_{source_id}_{int(time.time())}",
                alert_type=alert_type,
                level=level,
                message=message,
                source_id=source_id,
                timestamp=time.time(),
                details=details
            )

            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")

            # WebSocket广播告警
            if self.websocket_callback:
                try:
                    import asyncio
                    alert_dict = {
                        'alert_id': alert.alert_id,
                        'alert_type': alert.alert_type.value,
                        'level': alert.level.value,
                        'message': alert.message,
                        'source_id': alert.source_id,
                        'timestamp': alert.timestamp,
                        'details': alert.details
                    }
                    asyncio.create_task(
                        self.websocket_callback.broadcast_alert(alert_dict)
                    )
                except Exception as e:
                    logger.error(f"广播告警失败: {e}")

    def complete_task(self, task_id: str, records_collected: int = 0, error_message: Optional[str] = None):
        """
        完成任务

        Args:
            task_id: 任务ID
            records_collected: 采集到的记录数
            error_message: 错误消息（如果失败）
        """
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task.completed_at = time.time()
        task.records_collected = records_collected

        if error_message:
            task.status = HistoricalTaskStatus.FAILED
            task.error_message = error_message
            self.failed_tasks.append(task_id)
            self.stats['total_tasks_failed'] += 1
        else:
            task.status = HistoricalTaskStatus.COMPLETED
            task.progress = 1.0
            self.completed_tasks.append(task_id)
            self.stats['total_tasks_completed'] += 1

        # 更新统计
        self.stats['total_records_collected'] += records_collected

        if task.started_at and task.completed_at:
            duration = task.completed_at - task.started_at
            task.estimated_duration = duration

            # 更新平均采集时间
            total_completed = self.stats['total_tasks_completed']
            if total_completed == 1:
                self.stats['avg_collection_time'] = duration
            else:
                self.stats['avg_collection_time'] = (
                    (self.stats['avg_collection_time'] * (total_completed - 1)) + duration
                ) / total_completed

        # 从工作进程中移除
        if task.worker_id and task.worker_id in self.workers:
            worker = self.workers[task.worker_id]
            if task_id in worker.active_tasks:
                worker.active_tasks.remove(task_id)

        self.stats['last_activity_time'] = time.time()

        # WebSocket广播任务完成事件
        if self.websocket_callback:
            try:
                import asyncio
                success = not bool(error_message)
                asyncio.create_task(
                    self.websocket_callback.broadcast_task_completed(task_id, success, records_collected)
                )
            except Exception as e:
                logger.error(f"广播任务完成事件失败: {e}")

        status_text = "失败" if error_message else "成功"
        logger.info(f"任务{status_text}: {task_id} (采集记录: {records_collected})")

    def get_monitoring_data(self) -> Dict[str, Any]:
        """
        获取监控数据

        Returns:
            监控数据字典
        """
        current_time = time.time()

        # 计算成功率
        total_completed = self.stats['total_tasks_completed'] + self.stats['total_tasks_failed']
        if total_completed > 0:
            self.stats['success_rate'] = self.stats['total_tasks_completed'] / total_completed

        # 获取活跃任务
        active_tasks = [
            {
                'task_id': task.task_id,
                'symbol': task.symbol,
                'progress': task.progress,
                'status': task.status.value,
                'worker_id': task.worker_id,
                'started_at': task.started_at,
                'estimated_duration': task.estimated_duration
            }
            for task in self.tasks.values()
            if task.status in [HistoricalTaskStatus.RUNNING, HistoricalTaskStatus.PAUSED]
        ]

        # 获取队列中的任务
        queued_tasks = []
        # 注意：这里无法直接获取队列内容，需要额外的跟踪

        # 获取最近完成的任务
        recent_completed = [
            {
                'task_id': task.task_id,
                'symbol': task.symbol,
                'records_collected': task.records_collected,
                'duration': task.completed_at - task.started_at if task.started_at and task.completed_at else 0,
                'completed_at': task.completed_at
            }
            for task in self.tasks.values()
            if task.status == HistoricalTaskStatus.COMPLETED and task.completed_at and
            current_time - task.completed_at < 3600  # 最近1小时
        ]
        recent_completed.sort(key=lambda x: x['completed_at'], reverse=True)
        recent_completed = recent_completed[:10]  # 只保留最近10个

        return {
            'scheduler_status': 'active' if self.is_scheduler_active else 'inactive',
            'scheduler_uptime': current_time - self.stats['scheduler_start_time'] if self.stats['scheduler_start_time'] else 0,
            'active_workers': self.stats['active_workers'],
            'active_tasks': active_tasks,
            'queued_tasks_count': self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0,
            'recent_completed': recent_completed,
            'stats': self.stats.copy(),
            'performance_metrics': {
                'avg_collection_time': self.stats['avg_collection_time'],
                'success_rate': self.stats['success_rate'],
                'records_per_second': self._calculate_records_per_second(),
                'worker_utilization': self._calculate_worker_utilization()
            }
        }

    def _calculate_records_per_second(self) -> float:
        """计算平均采集速度"""
        if not self.stats['scheduler_start_time'] or self.stats['total_records_collected'] == 0:
            return 0.0

        uptime = time.time() - self.stats['scheduler_start_time']
        if uptime <= 0:
            return 0.0

        return self.stats['total_records_collected'] / uptime

    def _calculate_worker_utilization(self) -> float:
        """计算工作进程利用率"""
        if not self.workers:
            return 0.0

        active_workers = [w for w in self.workers.values() if w.is_active]
        if not active_workers:
            return 0.0

        total_capacity = sum(w.max_concurrent for w in active_workers)
        total_active = sum(len(w.active_tasks) for w in active_workers)

        return total_active / total_capacity if total_capacity > 0 else 0.0

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            任务状态信息
        """
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            'task_id': task.task_id,
            'symbol': task.symbol,
            'status': task.status.value,
            'progress': task.progress,
            'records_collected': task.records_collected,
            'created_at': task.created_at,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
            'error_message': task.error_message,
            'worker_id': task.worker_id,
            'priority': task.priority.name,
            'retry_count': task.retry_count,
            'metadata': task.metadata
        }

    def get_all_tasks(self, status_filter: Optional[HistoricalTaskStatus] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取所有任务

        Args:
            status_filter: 状态过滤器
            limit: 最大返回数量

        Returns:
            任务列表
        """
        tasks = []
        for task in self.tasks.values():
            if status_filter and task.status != status_filter:
                continue

            tasks.append({
                'task_id': task.task_id,
                'symbol': task.symbol,
                'status': task.status.value,
                'progress': task.progress,
                'records_collected': task.records_collected,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'priority': task.priority.name,
                'worker_id': task.worker_id
            })

        # 按创建时间倒序排序
        tasks.sort(key=lambda x: x['created_at'], reverse=True)
        return tasks[:limit]

    def get_alerts(self, resolved: bool = False, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取告警列表

        Args:
            resolved: 是否包含已解决的告警
            source_id: 数据源ID过滤

        Returns:
            List[Dict[str, Any]]: 告警列表
        """
        alerts = []
        for alert in self.alerts:
            if alert.resolved != resolved:
                continue
            if source_id and alert.source_id != source_id:
                continue

            alert_dict = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type.value,
                'level': alert.level.value,
                'message': alert.message,
                'source_id': alert.source_id,
                'timestamp': alert.timestamp,
                'details': alert.details,
                'resolved': alert.resolved
            }
            if alert.resolved_at:
                alert_dict['resolved_at'] = alert.resolved_at

            alerts.append(alert_dict)

        return alerts


# 全局监控器实例
_monitor_instance: Optional[HistoricalDataMonitor] = None


def get_historical_data_monitor(alert_callback: Optional[Callable[[Alert], None]] = None,
                              websocket_callback: Optional[Callable] = None) -> HistoricalDataMonitor:
    """
    获取历史数据采集监控器实例（单例模式）

    Args:
        alert_callback: 告警回调函数
        websocket_callback: WebSocket广播回调函数

    Returns:
        HistoricalDataMonitor: 监控器实例
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = HistoricalDataMonitor(alert_callback=alert_callback, websocket_callback=websocket_callback)
    return _monitor_instance