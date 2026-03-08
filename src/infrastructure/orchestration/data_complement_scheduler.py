"""
历史数据补全调度器
实现季度/半年周期的历史数据补全策略
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class ComplementMode(Enum):
    """补全模式"""
    NONE = "none"                    # 不需要补全
    QUARTERLY = "quarterly"          # 季度补全（90天）
    MONTHLY = "monthly"             # 月度补全（30天）
    WEEKLY = "weekly"               # 每周补全（7天）
    SEMI_ANNUAL = "semi_annual"     # 半年补全（180天）
    FULL_HISTORY = "full_history"   # 全历史补全（10年+）
    STRATEGY_BACKTEST = "strategy_backtest"  # 策略回测专用补全


class ComplementPriority(Enum):
    """补全优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplementTask:
    """补全任务"""
    task_id: str
    source_id: str
    data_type: str
    mode: ComplementMode
    priority: ComplementPriority
    start_date: datetime
    end_date: datetime
    estimated_records: int = 0
    actual_records: int = 0
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: float = 0.0
    last_complement_date: Optional[datetime] = None


@dataclass
class ComplementSchedule:
    """补全调度配置"""
    source_id: str
    data_type: str
    mode: ComplementMode
    priority: ComplementPriority
    schedule_interval_days: int  # 调度检查间隔（天）
    complement_window_days: int  # 补全时间窗口（天）
    min_gap_days: int  # 最小补全间隔（天）
    enabled: bool = True
    last_schedule_check: Optional[datetime] = None
    last_complement_date: Optional[datetime] = None


class DataComplementScheduler:
    """
    历史数据补全调度器

    实现智能的历史数据补全策略：
    1. 根据数据优先级确定补全周期（季度/半年）
    2. 智能检测补全时机和范围
    3. 分批次补全避免系统过载
    4. 进度追踪和断点续传
    """

    def __init__(self):
        self.schedules: Dict[str, ComplementSchedule] = {}
        self.active_tasks: Dict[str, ComplementTask] = {}
        self.completed_tasks: List[ComplementTask] = []

        # 初始化默认补全调度配置
        self._initialize_default_schedules()

        logger.info("历史数据补全调度器初始化完成")

    def _initialize_default_schedules(self):
        """初始化默认补全调度配置"""
        # 基于数据优先级管理器的配置
        from src.infrastructure.orchestration.data_priority_manager import get_data_priority_manager
        priority_manager = get_data_priority_manager()

        # 为不同优先级设置补全策略
        complement_configs = {
            'core_stocks': {
                'mode': ComplementMode.MONTHLY,
                'priority': ComplementPriority.CRITICAL,
                'schedule_interval_days': 1,  # 每日检查
                'complement_window_days': 30,  # 月度补全
                'min_gap_days': 25  # 最少25天间隔
            },
            'major_indices': {
                'mode': ComplementMode.WEEKLY,
                'priority': ComplementPriority.HIGH,
                'schedule_interval_days': 1,
                'complement_window_days': 7,
                'min_gap_days': 5
            },
            'all_stocks': {
                'mode': ComplementMode.QUARTERLY,
                'priority': ComplementPriority.MEDIUM,
                'schedule_interval_days': 7,  # 每周检查
                'complement_window_days': 90,  # 季度补全
                'min_gap_days': 80
            },
            'macro_data': {
                'mode': ComplementMode.SEMI_ANNUAL,
                'priority': ComplementPriority.LOW,
                'schedule_interval_days': 30,  # 每月检查
                'complement_window_days': 180,  # 半年补全
                'min_gap_days': 150
            },
            'strategy_backtest': {
                'mode': ComplementMode.STRATEGY_BACKTEST,
                'priority': ComplementPriority.HIGH,
                'schedule_interval_days': 365,  # 每年检查一次
                'complement_window_days': 3650,  # 10年数据
                'min_gap_days': 330,  # 最少330天间隔
                'batch_size_days': 365,  # 按年分批
                'max_concurrent_batches': 2  # 限制并发
            }
        }

        # 注册默认配置 - 为策略回测创建专门的调度配置
        for priority_key, config in complement_configs.items():
            try:
                # 为策略回测创建专门的调度配置
                if priority_key == 'strategy_backtest':
                    schedule = ComplementSchedule(
                        source_id='strategy_backtest_data',
                        data_type='stock',
                        mode=config['mode'],
                        priority=config['priority'],
                        schedule_interval_days=config['schedule_interval_days'],
                        complement_window_days=config['complement_window_days'],
                        min_gap_days=config['min_gap_days'],
                        enabled=True
                    )
                    self.schedules['strategy_backtest_data'] = schedule
                    logger.info(f"注册策略回测补全调度配置: {schedule.source_id}")

            except Exception as e:
                logger.warning(f"初始化补全调度配置失败 {priority_key}: {e}")

    def register_complement_schedule(self, source_id: str, data_type: str,
                                   mode: ComplementMode, priority: ComplementPriority,
                                   schedule_config: Optional[Dict[str, Any]] = None):
        """
        注册补全调度配置

        Args:
            source_id: 数据源ID
            data_type: 数据类型
            mode: 补全模式
            priority: 补全优先级
            schedule_config: 调度配置
        """
        config = schedule_config or self._get_default_schedule_config(mode, priority)

        schedule = ComplementSchedule(
            source_id=source_id,
            data_type=data_type,
            mode=mode,
            priority=priority,
            schedule_interval_days=config['schedule_interval_days'],
            complement_window_days=config['complement_window_days'],
            min_gap_days=config['min_gap_days'],
            enabled=True
        )

        self.schedules[source_id] = schedule
        logger.info(f"注册补全调度配置: {source_id}, 模式: {mode.value}, 优先级: {priority.value}")

    def _get_default_schedule_config(self, mode: ComplementMode,
                                   priority: ComplementPriority) -> Dict[str, Any]:
        """获取默认调度配置"""
        # 基于模式和优先级的默认配置
        mode_configs = {
            ComplementMode.MONTHLY: {
                'schedule_interval_days': 1,
                'complement_window_days': 30,
                'min_gap_days': 25
            },
            ComplementMode.WEEKLY: {
                'schedule_interval_days': 1,
                'complement_window_days': 7,
                'min_gap_days': 5
            },
            ComplementMode.QUARTERLY: {
                'schedule_interval_days': 7,
                'complement_window_days': 90,
                'min_gap_days': 80
            },
            ComplementMode.SEMI_ANNUAL: {
                'schedule_interval_days': 30,
                'complement_window_days': 180,
                'min_gap_days': 150
            }
        }

        return mode_configs.get(mode, {
            'schedule_interval_days': 7,
            'complement_window_days': 90,
            'min_gap_days': 80
        })

    def check_complement_needed(self, source_id: str) -> Tuple[bool, Optional[ComplementTask]]:
        """
        检查是否需要补全数据

        Args:
            source_id: 数据源ID

        Returns:
            (是否需要补全, 补全任务)
        """
        if source_id not in self.schedules:
            logger.debug(f"数据源 {source_id} 未注册补全调度配置")
            return False, None

        schedule = self.schedules[source_id]
        current_time = datetime.now()

        # 检查是否到调度检查时间
        if schedule.last_schedule_check:
            time_since_check = (current_time - schedule.last_schedule_check).days
            if time_since_check < schedule.schedule_interval_days:
                return False, None

        # 更新检查时间
        schedule.last_schedule_check = current_time

        # 检查是否满足补全条件
        if self._should_trigger_complement(schedule):
            task = self._create_complement_task(schedule)
            return True, task

        return False, None

    def _should_trigger_complement(self, schedule: ComplementSchedule) -> bool:
        """判断是否应该触发补全"""
        current_time = datetime.now()

        # 如果从未补全过，立即触发
        if schedule.last_complement_date is None:
            return True

        # 检查距离上次补全的时间间隔
        days_since_last = (current_time - schedule.last_complement_date).days

        # 根据补全模式检查是否达到触发条件
        mode_triggers = {
            ComplementMode.MONTHLY: days_since_last >= 30,
            ComplementMode.WEEKLY: days_since_last >= 7,
            ComplementMode.QUARTERLY: days_since_last >= 90,
            ComplementMode.SEMI_ANNUAL: days_since_last >= 180,
            ComplementMode.FULL_HISTORY: days_since_last >= 365,  # 每年检查一次
            ComplementMode.STRATEGY_BACKTEST: days_since_last >= 330,  # 每年检查一次（330天缓冲）
        }

        should_trigger = mode_triggers.get(schedule.mode, False)

        if should_trigger:
            logger.info(f"数据源 {schedule.source_id} 触发补全条件: "
                       f"模式={schedule.mode.value}, "
                       f"距离上次补全={days_since_last}天")

        return should_trigger

    def _create_complement_task(self, schedule: ComplementSchedule) -> ComplementTask:
        """创建补全任务"""
        current_time = datetime.now()

        # 计算补全时间范围
        if schedule.mode == ComplementMode.STRATEGY_BACKTEST:
            # 策略回测模式：补全完整的10年历史数据
            if schedule.last_complement_date:
                # 从上次补全时间开始，补全到当前时间
                start_date = schedule.last_complement_date
            else:
                # 首次补全，补全最近10年的完整数据
                start_date = current_time - timedelta(days=schedule.complement_window_days)
            end_date = current_time
        else:
            # 普通补全模式
            if schedule.last_complement_date:
                # 从上次补全时间开始
                start_date = schedule.last_complement_date
            else:
                # 首次补全，补全最近的窗口期
                start_date = current_time - timedelta(days=schedule.complement_window_days)
            end_date = current_time

        # 估算记录数（基于历史数据和时间范围）
        estimated_records = self._estimate_complement_records(schedule, start_date, end_date)

        task = ComplementTask(
            task_id=f"complement_{schedule.source_id}_{int(current_time.timestamp())}",
            source_id=schedule.source_id,
            data_type=schedule.data_type,
            mode=schedule.mode,
            priority=schedule.priority,
            start_date=start_date,
            end_date=end_date,
            estimated_records=estimated_records
        )

        return task

    def _estimate_complement_records(self, schedule: ComplementSchedule,
                                   start_date: datetime, end_date: datetime) -> int:
        """估算补全记录数"""
        try:
            # 计算时间范围（天数）
            days_range = (end_date - start_date).days

            # 根据数据类型估算日均记录数
            daily_estimates = {
                'stock': 1,      # 股票：每日1条记录
                'index': 1,      # 指数：每日1条记录
                'macro': 0.1,    # 宏观：每日约0.1条记录
                'news': 10,      # 新闻：每日约10条记录
            }

            daily_rate = daily_estimates.get(schedule.data_type, 1)
            estimated_total = int(days_range * daily_rate)

            # 考虑补全效率（通常只有部分日期有数据）
            efficiency_factor = 0.7  # 70%的工作日有数据
            estimated_total = int(estimated_total * efficiency_factor)

            return max(estimated_total, 1)  # 至少1条记录

        except Exception as e:
            logger.warning(f"估算补全记录数失败: {e}")
            return 100  # 默认估算值

    def start_complement_task(self, task: ComplementTask):
        """启动补全任务"""
        task.status = "running"
        task.started_at = datetime.now()

        self.active_tasks[task.task_id] = task

        # 更新调度配置的最后补全时间
        if task.source_id in self.schedules:
            self.schedules[task.source_id].last_complement_date = task.started_at

        logger.info(f"启动补全任务: {task.task_id} for {task.source_id}")

    def update_complement_progress(self, task_id: str, progress_percentage: float,
                                 current_records: int):
        """更新补全进度"""
        if task_id not in self.active_tasks:
            logger.warning(f"补全任务不存在: {task_id}")
            return

        task = self.active_tasks[task_id]
        task.progress_percentage = progress_percentage
        task.actual_records = current_records

        logger.debug(f"补全进度更新: {task_id} - {progress_percentage:.1f}%, {current_records}条记录")

    def complete_complement_task(self, task_id: str, success: bool = True,
                               error_message: Optional[str] = None):
        """完成补全任务"""
        if task_id not in self.active_tasks:
            logger.warning(f"补全任务不存在: {task_id}")
            return

        task = self.active_tasks[task_id]
        task.completed_at = datetime.now()
        task.status = "completed" if success else "failed"
        task.error_message = error_message

        # 计算实际耗时
        if task.started_at:
            duration = task.completed_at - task.started_at
            task.actual_records = getattr(task, 'actual_records', 0)

        # 从活跃任务中移除
        del self.active_tasks[task_id]

        # 添加到完成任务列表
        self.completed_tasks.append(task)

        # 只保留最近1000个完成任务
        if len(self.completed_tasks) > 1000:
            self.completed_tasks = self.completed_tasks[-1000:]

        status_msg = "成功" if success else f"失败: {error_message}"
        logger.info(f"补全任务完成: {task_id} - {status_msg}")

    def get_pending_complement_tasks(self) -> List[ComplementTask]:
        """获取待处理的补全任务"""
        pending_tasks = []

        for schedule in self.schedules.values():
            if not schedule.enabled:
                continue

            needed, task = self.check_complement_needed(schedule.source_id)
            if needed and task:
                pending_tasks.append(task)

        # 按优先级排序
        priority_order = {
            ComplementPriority.CRITICAL: 0,
            ComplementPriority.HIGH: 1,
            ComplementPriority.MEDIUM: 2,
            ComplementPriority.LOW: 3
        }

        pending_tasks.sort(key=lambda t: priority_order.get(t.priority, 99))

        return pending_tasks

    def get_active_complement_tasks(self) -> List[ComplementTask]:
        """获取正在执行的补全任务"""
        return list(self.active_tasks.values())

    def get_complement_task(self, task_id: str) -> Optional[ComplementTask]:
        """获取补全任务"""
        return self.active_tasks.get(task_id)

    def cancel_complement_task(self, task_id: str):
        """取消补全任务"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = "cancelled"
            task.completed_at = datetime.now()

            del self.active_tasks[task_id]
            self.completed_tasks.append(task)

            logger.info(f"补全任务已取消: {task_id}")

    def get_complement_statistics(self) -> Dict[str, Any]:
        """获取补全统计信息"""
        stats = {
            'total_schedules': len(self.schedules),
            'active_tasks': len(self.active_tasks),
            'completed_tasks_today': 0,
            'failed_tasks_today': 0,
            'total_complemented_records': 0
        }

        today = date.today()
        for task in self.completed_tasks:
            if task.completed_at and task.completed_at.date() == today:
                if task.status == "completed":
                    stats['completed_tasks_today'] += 1
                    stats['total_complemented_records'] += task.actual_records
                elif task.status == "failed":
                    stats['failed_tasks_today'] += 1

        return stats

    def cleanup_old_tasks(self, days_to_keep: int = 30):
        """清理旧的补全任务记录"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        old_tasks = [
            task for task in self.completed_tasks
            if task.completed_at and task.completed_at < cutoff_date
        ]

        for task in old_tasks:
            self.completed_tasks.remove(task)

        if old_tasks:
            logger.info(f"清理了 {len(old_tasks)} 个旧补全任务记录")


# 全局实例
_complement_scheduler = None


def get_data_complement_scheduler() -> DataComplementScheduler:
    """获取历史数据补全调度器实例"""
    global _complement_scheduler
    if _complement_scheduler is None:
        _complement_scheduler = DataComplementScheduler()
    return _complement_scheduler