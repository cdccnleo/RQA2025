"""
优先级引擎模块

负责任务优先级计算、优先级老化管理和抢占决策。

Author: RQA2025 Development Team
Date: 2026-02-13
"""

import logging
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """任务优先级"""
    CRITICAL = 4    # 关键任务 - 必须立即执行
    HIGH = 3        # 高优先级 - 尽快执行
    NORMAL = 2      # 普通优先级 - 正常调度
    LOW = 1         # 低优先级 - 资源空闲时执行
    BACKGROUND = 0  # 后台任务 - 最低优先级


@dataclass
class PriorityConfig:
    """优先级配置"""
    # 老化配置
    aging_enabled: bool = True
    aging_interval_seconds: int = 300  # 每5分钟老化一次
    aging_increment: int = 1           # 每次老化提升1级
    max_aging_levels: int = 2          # 最多老化2级
    
    # 抢占配置
    preemption_enabled: bool = True
    min_preemption_interval: int = 10  # 最小抢占间隔（秒）
    
    # 优先级权重
    priority_weights: Dict[TaskPriority, float] = field(default_factory=lambda: {
        TaskPriority.CRITICAL: 100.0,
        TaskPriority.HIGH: 50.0,
        TaskPriority.NORMAL: 10.0,
        TaskPriority.LOW: 5.0,
        TaskPriority.BACKGROUND: 1.0
    })


@dataclass
class PriorityScore:
    """优先级分数"""
    base_priority: int
    aging_boost: int
    resource_score: float
    deadline_score: float
    final_score: float


class PriorityEngine:
    """
    优先级引擎
    
    提供以下功能：
    1. 任务优先级计算
    2. 优先级老化管理
    3. 抢占决策
    4. 资源优先级评分
    
    Attributes:
        config: 优先级配置
        task_priorities: 任务优先级缓存
        last_aging_time: 上次老化时间
    """
    
    def __init__(self, config: Optional[PriorityConfig] = None):
        self.config = config or PriorityConfig()
        self.task_priorities: Dict[str, TaskPriority] = {}
        self.task_submit_times: Dict[str, float] = {}
        self.task_aging_levels: Dict[str, int] = {}
        self.last_aging_time: float = time.time()
        self.last_preemption_time: float = 0
        
        logger.info("PriorityEngine initialized")
    
    def calculate_priority_score(
        self,
        task_id: str,
        base_priority: TaskPriority,
        resource_requirements: Optional[Dict] = None,
        deadline: Optional[datetime] = None
    ) -> PriorityScore:
        """
        计算任务优先级分数
        
        Args:
            task_id: 任务ID
            base_priority: 基础优先级
            resource_requirements: 资源需求
            deadline: 截止时间
            
        Returns:
            PriorityScore: 优先级分数
        """
        # 基础优先级分数
        base_score = self.config.priority_weights.get(base_priority, 1.0)
        
        # 老化提升
        aging_boost = self.task_aging_levels.get(task_id, 0)
        aging_score = aging_boost * 10  # 每级老化加10分
        
        # 资源分数（资源需求越高，分数略低）
        resource_score = 0.0
        if resource_requirements:
            cpu_required = resource_requirements.get('cpu_cores', 1)
            memory_required = resource_requirements.get('memory_gb', 1)
            # 资源需求大的任务分数略低，避免资源垄断
            resource_score = -1.0 * (cpu_required + memory_required / 4)
        
        # 截止时间分数（越紧急分数越高）
        deadline_score = 0.0
        if deadline:
            time_to_deadline = (deadline - datetime.now()).total_seconds()
            if time_to_deadline < 0:
                # 已过期，最高优先级
                deadline_score = 1000.0
            elif time_to_deadline < 300:  # 5分钟内
                deadline_score = 500.0
            elif time_to_deadline < 1800:  # 30分钟内
                deadline_score = 200.0
            elif time_to_deadline < 3600:  # 1小时内
                deadline_score = 50.0
        
        # 最终分数
        final_score = base_score + aging_score + resource_score + deadline_score
        
        return PriorityScore(
            base_priority=base_priority.value,
            aging_boost=aging_boost,
            resource_score=resource_score,
            deadline_score=deadline_score,
            final_score=final_score
        )
    
    def check_aging_tasks(self, tasks: List) -> List[str]:
        """
        检查并老化任务
        
        Args:
            tasks: 任务列表
            
        Returns:
            List[str]: 被老化的任务ID列表
        """
        if not self.config.aging_enabled:
            return []
        
        current_time = time.time()
        
        # 检查是否到达老化间隔
        if current_time - self.last_aging_time < self.config.aging_interval_seconds:
            return []
        
        aged_tasks = []
        
        for task in tasks:
            task_id = task.task_id if hasattr(task, 'task_id') else str(task)
            priority = task.priority if hasattr(task, 'priority') else TaskPriority.NORMAL
            
            # 只有低优先级任务需要老化
            if priority in [TaskPriority.LOW, TaskPriority.BACKGROUND]:
                current_aging = self.task_aging_levels.get(task_id, 0)
                
                if current_aging < self.config.max_aging_levels:
                    self.task_aging_levels[task_id] = current_aging + self.config.aging_increment
                    aged_tasks.append(task_id)
                    logger.info(f"Task {task_id} aged to level {current_aging + 1}")
        
        self.last_aging_time = current_time
        
        if aged_tasks:
            logger.info(f"Aged {len(aged_tasks)} tasks")
        
        return aged_tasks
    
    def get_preemption_candidates(
        self,
        new_task,
        running_tasks: Dict[str, any]
    ) -> List[str]:
        """
        获取可被抢占的任务列表
        
        Args:
            new_task: 新任务
            running_tasks: 运行中的任务字典 {task_id: task}
            
        Returns:
            List[str]: 可被抢占的任务ID列表
        """
        if not self.config.preemption_enabled:
            return []
        
        # 检查抢占间隔
        current_time = time.time()
        if current_time - self.last_preemption_time < self.config.min_preemption_interval:
            return []
        
        new_task_priority = new_task.priority if hasattr(new_task, 'priority') else TaskPriority.NORMAL
        
        candidates = []
        
        for task_id, task in running_tasks.items():
            running_priority = task.priority if hasattr(task, 'priority') else TaskPriority.NORMAL
            
            # 新任务优先级必须高于运行中任务
            if new_task_priority.value > running_priority.value:
                # 计算优先级差值
                priority_diff = new_task_priority.value - running_priority.value
                
                # 只有优先级差值足够大时才抢占（避免频繁抢占）
                if priority_diff >= 2:  # 至少差2个级别
                    candidates.append(task_id)
        
        return candidates
    
    def can_preempt(
        self,
        new_task,
        running_task
    ) -> bool:
        """
        判断新任务是否可以抢占运行中的任务
        
        Args:
            new_task: 新任务
            running_task: 运行中的任务
            
        Returns:
            bool: 是否可以抢占
        """
        if not self.config.preemption_enabled:
            return False
        
        new_priority = new_task.priority if hasattr(new_task, 'priority') else TaskPriority.NORMAL
        running_priority = running_task.priority if hasattr(running_task, 'priority') else TaskPriority.NORMAL
        
        # 新任务优先级必须显著高于运行中任务
        priority_diff = new_priority.value - running_priority.value
        
        # 至少差2个级别，且满足最小间隔
        if priority_diff >= 2:
            current_time = time.time()
            if current_time - self.last_preemption_time >= self.config.min_preemption_interval:
                return True
        
        return False
    
    def record_preemption(self):
        """记录抢占事件"""
        self.last_preemption_time = time.time()
    
    def reset_aging(self, task_id: str):
        """
        重置任务老化
        
        Args:
            task_id: 任务ID
        """
        if task_id in self.task_aging_levels:
            del self.task_aging_levels[task_id]
            logger.debug(f"Reset aging for task {task_id}")
    
    def get_task_effective_priority(self, task_id: str, base_priority: TaskPriority) -> TaskPriority:
        """
        获取任务的有效优先级（考虑老化）
        
        Args:
            task_id: 任务ID
            base_priority: 基础优先级
            
        Returns:
            TaskPriority: 有效优先级
        """
        aging_level = self.task_aging_levels.get(task_id, 0)
        
        if aging_level == 0:
            return base_priority
        
        # 根据老化级别提升优先级
        priority_levels = [
            TaskPriority.BACKGROUND,
            TaskPriority.LOW,
            TaskPriority.NORMAL,
            TaskPriority.HIGH,
            TaskPriority.CRITICAL
        ]
        
        current_index = priority_levels.index(base_priority)
        new_index = min(current_index + aging_level, len(priority_levels) - 1)
        
        return priority_levels[new_index]
    
    def sort_tasks_by_priority(
        self,
        tasks: List,
        resource_requirements: Optional[Dict[str, Dict]] = None,
        deadlines: Optional[Dict[str, datetime]] = None
    ) -> List:
        """
        按优先级排序任务
        
        Args:
            tasks: 任务列表
            resource_requirements: 资源需求字典
            deadlines: 截止时间字典
            
        Returns:
            List: 排序后的任务列表
        """
        task_scores = []
        
        for task in tasks:
            task_id = task.task_id if hasattr(task, 'task_id') else str(task)
            priority = task.priority if hasattr(task, 'priority') else TaskPriority.NORMAL
            
            resource_req = resource_requirements.get(task_id) if resource_requirements else None
            deadline = deadlines.get(task_id) if deadlines else None
            
            score = self.calculate_priority_score(task_id, priority, resource_req, deadline)
            task_scores.append((task, score.final_score))
        
        # 按分数降序排序
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [task for task, _ in task_scores]
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            'total_tracked_tasks': len(self.task_priorities),
            'aged_tasks': len(self.task_aging_levels),
            'last_aging_time': self.last_aging_time,
            'last_preemption_time': self.last_preemption_time,
            'config': {
                'aging_enabled': self.config.aging_enabled,
                'preemption_enabled': self.config.preemption_enabled
            }
        }


# 便捷函数
def create_default_priority_engine() -> PriorityEngine:
    """创建默认优先级引擎"""
    return PriorityEngine()


def create_aggressive_priority_engine() -> PriorityEngine:
    """创建激进优先级引擎（更频繁的抢占）"""
    config = PriorityConfig(
        aging_enabled=True,
        aging_interval_seconds=60,  # 每分钟老化
        aging_increment=1,
        max_aging_levels=3,
        preemption_enabled=True,
        min_preemption_interval=5   # 5秒间隔
    )
    return PriorityEngine(config)


def create_conservative_priority_engine() -> PriorityEngine:
    """创建保守优先级引擎（更少的抢占）"""
    config = PriorityConfig(
        aging_enabled=True,
        aging_interval_seconds=600,  # 每10分钟老化
        aging_increment=1,
        max_aging_levels=1,
        preemption_enabled=True,
        min_preemption_interval=60   # 60秒间隔
    )
    return PriorityEngine(config)


__all__ = [
    'PriorityEngine',
    'PriorityConfig',
    'PriorityScore',
    'TaskPriority',
    'create_default_priority_engine',
    'create_aggressive_priority_engine',
    'create_conservative_priority_engine'
]