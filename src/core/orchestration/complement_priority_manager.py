"""
补全任务优先级管理器
实现补全任务的优先级排序、调度和管理
"""

import heapq
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from src.infrastructure.logging.core.unified_logger import get_unified_logger
from .data_complement_scheduler import ComplementTask, ComplementPriority

logger = get_unified_logger(__name__)


class PriorityScore(Enum):
    """优先级得分枚举"""
    CRITICAL = 1000
    HIGH = 750
    MEDIUM = 500
    LOW = 250


@dataclass(order=True)
class PrioritizedTask:
    """带优先级的任务"""
    priority_score: int  # 用于排序的主要字段
    task: ComplementTask = field(compare=False)
    enqueue_time: datetime = field(default_factory=datetime.now, compare=False)
    retry_count: int = field(default=0, compare=False)
    dependencies: Set[str] = field(default_factory=set, compare=False)

    def __post_init__(self):
        # 确保priority_score为负数（heapq是最小堆）
        if self.priority_score > 0:
            self.priority_score = -self.priority_score


class ComplementPriorityManager:
    """
    补全任务优先级管理器

    实现功能：
    1. 任务优先级计算和动态调整
    2. 优先级队列管理和调度
    3. 依赖关系处理
    4. 资源分配优化
    5. 紧急任务插队机制
    """

    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size

        # 优先级队列（最小堆，优先级得分越小越优先）
        self.priority_queue: List[PrioritizedTask] = []

        # 任务状态跟踪
        self.active_tasks: Dict[str, PrioritizedTask] = {}
        self.completed_tasks: Dict[str, datetime] = {}
        self.failed_tasks: Dict[str, int] = {}  # 任务ID -> 失败次数

        # 依赖关系图
        self.task_dependencies: Dict[str, Set[str]] = {}  # 任务ID -> 依赖任务集合
        self.task_dependents: Dict[str, Set[str]] = {}   # 任务ID -> 被依赖任务集合

        # 资源限制
        self.max_concurrent_per_priority = {
            ComplementPriority.CRITICAL: 3,
            ComplementPriority.HIGH: 2,
            ComplementPriority.MEDIUM: 1,
            ComplementPriority.LOW: 1
        }

        # 当前活跃任务统计
        self.active_count_per_priority = {
            ComplementPriority.CRITICAL: 0,
            ComplementPriority.HIGH: 0,
            ComplementPriority.MEDIUM: 0,
            ComplementPriority.LOW: 0
        }

        logger.info("补全任务优先级管理器初始化完成")

    def enqueue_task(self, task: ComplementTask, priority_override: Optional[ComplementPriority] = None) -> bool:
        """
        将任务加入优先级队列

        Args:
            task: 补全任务
            priority_override: 优先级覆盖（可选）

        Returns:
            是否成功入队
        """
        try:
            # 检查队列大小限制
            if len(self.priority_queue) >= self.max_queue_size:
                logger.warning(f"优先级队列已满，无法添加任务: {task.task_id}")
                return False

            # 检查任务是否已在队列中
            if task.task_id in self.active_tasks or task.task_id in self.completed_tasks:
                logger.debug(f"任务已在处理中或已完成: {task.task_id}")
                return True

            # 计算优先级得分
            priority = priority_override or task.priority
            priority_score = self._calculate_priority_score(task, priority)

            # 创建优先级任务
            prioritized_task = PrioritizedTask(
                priority_score=priority_score,
                task=task,
                dependencies=set()
            )

            # 添加到优先级队列
            heapq.heappush(self.priority_queue, prioritized_task)

            logger.info(f"任务加入优先级队列: {task.task_id}, 优先级: {priority.value}, 得分: {priority_score}")
            return True

        except Exception as e:
            logger.error(f"添加任务到优先级队列失败: {e}")
            return False

    def dequeue_task(self) -> Optional[ComplementTask]:
        """
        从优先级队列中取出最高优先级的任务

        Returns:
            最高优先级任务
        """
        try:
            # 检查是否有任务
            if not self.priority_queue:
                return None

            # 获取最高优先级任务
            prioritized_task = heapq.heappop(self.priority_queue)

            # 检查资源限制
            if not self._check_resource_limits(prioritized_task):
                # 如果资源不足，放回队列
                heapq.heappush(self.priority_queue, prioritized_task)
                return None

            # 检查依赖关系
            if not self._check_dependencies(prioritized_task):
                # 如果依赖未满足，放回队列
                heapq.heappush(self.priority_queue, prioritized_task)
                return None

            # 标记为活跃任务
            task_id = prioritized_task.task.task_id
            self.active_tasks[task_id] = prioritized_task

            # 更新资源计数
            self.active_count_per_priority[prioritized_task.task.priority] += 1

            logger.info(f"从优先级队列取出任务: {task_id}")
            return prioritized_task.task

        except Exception as e:
            logger.error(f"从优先级队列取出任务失败: {e}")
            return None

    def complete_task(self, task_id: str, success: bool = True):
        """
        标记任务完成

        Args:
            task_id: 任务ID
            success: 是否成功
        """
        if task_id in self.active_tasks:
            prioritized_task = self.active_tasks[task_id]

            # 更新资源计数
            self.active_count_per_priority[prioritized_task.task.priority] -= 1

            # 移除活跃任务
            del self.active_tasks[task_id]

            # 添加到完成/失败记录
            if success:
                self.completed_tasks[task_id] = datetime.now()
                # 清理失败计数
                if task_id in self.failed_tasks:
                    del self.failed_tasks[task_id]
            else:
                # 增加失败计数
                self.failed_tasks[task_id] = self.failed_tasks.get(task_id, 0) + 1

            # 处理依赖关系
            self._resolve_dependencies(task_id)

            status_msg = "成功" if success else "失败"
            logger.info(f"任务完成: {task_id} - {status_msg}")

    def add_task_dependency(self, task_id: str, dependency_task_id: str):
        """
        添加任务依赖关系

        Args:
            task_id: 任务ID
            dependency_task_id: 依赖的任务ID
        """
        # 添加正向依赖
        if task_id not in self.task_dependencies:
            self.task_dependencies[task_id] = set()
        self.task_dependencies[task_id].add(dependency_task_id)

        # 添加反向依赖
        if dependency_task_id not in self.task_dependents:
            self.task_dependents[dependency_task_id] = set()
        self.task_dependents[dependency_task_id].add(task_id)

        logger.debug(f"添加任务依赖: {task_id} -> {dependency_task_id}")

    def get_next_available_tasks(self, max_count: int = 5) -> List[ComplementTask]:
        """
        获取下一个可用的任务列表

        Args:
            max_count: 最大任务数量

        Returns:
            可执行的任务列表
        """
        available_tasks = []

        # 创建临时队列副本用于检查
        temp_queue = self.priority_queue.copy()
        temp_active = self.active_tasks.copy()

        while temp_queue and len(available_tasks) < max_count:
            prioritized_task = heapq.heappop(temp_queue)

            task_id = prioritized_task.task.task_id

            # 跳过已在活跃列表中的任务
            if task_id in temp_active:
                continue

            # 检查资源限制
            if not self._check_resource_limits(prioritized_task):
                continue

            # 检查依赖关系
            if not self._check_dependencies(prioritized_task):
                continue

            available_tasks.append(prioritized_task.task)

        return available_tasks

    def update_task_priority(self, task_id: str, new_priority: ComplementPriority):
        """
        更新任务优先级

        Args:
            task_id: 任务ID
            new_priority: 新优先级
        """
        # 查找任务在队列中的位置
        for i, prioritized_task in enumerate(self.priority_queue):
            if prioritized_task.task.task_id == task_id:
                # 更新优先级
                prioritized_task.task.priority = new_priority
                prioritized_task.priority_score = -self._calculate_priority_score(
                    prioritized_task.task, new_priority
                )

                # 重新构建堆
                heapq.heapify(self.priority_queue)

                logger.info(f"更新任务优先级: {task_id} -> {new_priority.value}")
                return

        # 检查活跃任务
        if task_id in self.active_tasks:
            self.active_tasks[task_id].task.priority = new_priority
            logger.info(f"更新活跃任务优先级: {task_id} -> {new_priority.value}")

    def _calculate_priority_score(self, task: ComplementTask, priority: ComplementPriority) -> int:
        """
        计算任务优先级得分

        得分计算规则：
        1. 基础优先级得分
        2. 时间因子（等待时间越长，优先级越高）
        3. 失败惩罚（失败次数越多，优先级越低）
        4. 数据重要性加成
        """
        # 基础优先级得分
        base_scores = {
            ComplementPriority.CRITICAL: PriorityScore.CRITICAL.value,
            ComplementPriority.HIGH: PriorityScore.HIGH.value,
            ComplementPriority.MEDIUM: PriorityScore.MEDIUM.value,
            ComplementPriority.LOW: PriorityScore.LOW.value
        }

        base_score = base_scores.get(priority, PriorityScore.MEDIUM.value)

        # 时间因子（每小时等待增加10分）
        if task.created_at:
            wait_hours = (datetime.now() - task.created_at).total_seconds() / 3600
            time_bonus = min(int(wait_hours * 10), 200)  # 最多加200分
        else:
            time_bonus = 0

        # 失败惩罚（每次失败减少50分）
        failure_penalty = self.failed_tasks.get(task.task_id, 0) * 50

        # 数据重要性加成
        importance_bonus = self._calculate_importance_bonus(task)

        # 计算最终得分
        final_score = base_score + time_bonus - failure_penalty + importance_bonus

        return max(final_score, 50)  # 最低50分

    def _calculate_importance_bonus(self, task: ComplementTask) -> int:
        """计算数据重要性加成"""
        bonus = 0

        # 根据数据源重要性
        source_id = task.source_id.lower()
        if 'core' in source_id or 'critical' in source_id:
            bonus += 100
        elif 'index' in source_id or 'benchmark' in source_id:
            bonus += 50

        # 根据补全模式
        if task.mode.name == 'QUARTERLY':
            bonus += 30  # 季度补全较重要
        elif task.mode.name == 'SEMI_ANNUAL':
            bonus += 20  # 半年补全重要性较低

        # 根据预计记录数（记录数越多越重要）
        if task.estimated_records > 10000:
            bonus += 50
        elif task.estimated_records > 1000:
            bonus += 25

        return bonus

    def _check_resource_limits(self, prioritized_task: PrioritizedTask) -> bool:
        """
        检查资源限制

        Args:
            prioritized_task: 优先级任务

        Returns:
            是否可以执行
        """
        priority = prioritized_task.task.priority
        current_active = self.active_count_per_priority[priority]
        max_allowed = self.max_concurrent_per_priority[priority]

        return current_active < max_allowed

    def _check_dependencies(self, prioritized_task: PrioritizedTask) -> bool:
        """
        检查任务依赖关系

        Args:
            prioritized_task: 优先级任务

        Returns:
            依赖是否满足
        """
        task_id = prioritized_task.task.task_id
        dependencies = self.task_dependencies.get(task_id, set())

        # 检查所有依赖任务是否已完成
        for dep_task_id in dependencies:
            if dep_task_id not in self.completed_tasks:
                return False

        return True

    def _resolve_dependencies(self, completed_task_id: str):
        """
        处理任务完成后的依赖关系

        Args:
            completed_task_id: 已完成的任务ID
        """
        # 通知所有依赖此任务的其他任务
        dependents = self.task_dependents.get(completed_task_id, set())

        # 可以在这里触发依赖任务的重新评估
        # 目前只记录日志
        if dependents:
            logger.debug(f"任务 {completed_task_id} 完成，{len(dependents)} 个依赖任务可以重新评估")

    def get_queue_statistics(self) -> Dict[str, Any]:
        """
        获取队列统计信息

        Returns:
            统计信息字典
        """
        stats = {
            'queue_size': len(self.priority_queue),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'active_by_priority': self.active_count_per_priority.copy(),
            'queue_by_priority': {}
        }

        # 统计队列中各优先级的任务数
        priority_counts = {}
        for prioritized_task in self.priority_queue:
            priority = prioritized_task.task.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        stats['queue_by_priority'] = priority_counts

        # 计算平均等待时间
        if self.priority_queue:
            total_wait_time = 0
            for prioritized_task in self.priority_queue:
                wait_time = (datetime.now() - prioritized_task.enqueue_time).total_seconds()
                total_wait_time += wait_time

            stats['average_wait_time_seconds'] = total_wait_time / len(self.priority_queue)
        else:
            stats['average_wait_time_seconds'] = 0

        return stats

    def emergency_enqueue(self, task: ComplementTask, emergency_priority: ComplementPriority = ComplementPriority.CRITICAL):
        """
        紧急任务插队

        Args:
            task: 紧急任务
            emergency_priority: 紧急优先级
        """
        # 设置最高优先级得分
        emergency_score = -2000  # 确保排在最前面

        prioritized_task = PrioritizedTask(
            priority_score=emergency_score,
            task=task
        )

        # 添加到队列前面（通过负数得分实现）
        heapq.heappush(self.priority_queue, prioritized_task)

        logger.warning(f"紧急任务插队: {task.task_id} 优先级: {emergency_priority.value}")

    def clear_completed_tasks(self, days_to_keep: int = 7):
        """
        清理已完成的任务记录

        Args:
            days_to_keep: 保留天数
        """
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)

        # 清理已完成的任务
        old_completed = [
            task_id for task_id, completion_time in self.completed_tasks.items()
            if completion_time < cutoff_time
        ]

        for task_id in old_completed:
            del self.completed_tasks[task_id]

        # 清理失败计数（失败太多次的任务）
        old_failed = [
            task_id for task_id, failure_count in self.failed_tasks.items()
            if failure_count > 10  # 失败超过10次的任务
        ]

        for task_id in old_failed:
            del self.failed_tasks[task_id]

        if old_completed or old_failed:
            logger.info(f"清理了 {len(old_completed)} 个旧完成任务和 {len(old_failed)} 个失败任务")


# 全局实例
_priority_manager_instance = None


def get_complement_priority_manager() -> ComplementPriorityManager:
    """获取补全任务优先级管理器实例"""
    global _priority_manager_instance
    if _priority_manager_instance is None:
        _priority_manager_instance = ComplementPriorityManager()
    return _priority_manager_instance