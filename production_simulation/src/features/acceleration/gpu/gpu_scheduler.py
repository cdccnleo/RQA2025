#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
GPU资源调度器
实现多模型GPU共享、优先级调度、内存感知分配等功能
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from heapq import heappush, heapify

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级"""

    CRITICAL = 1  # 关键任务（实时交易）
    HIGH = 2  # 高优先级（模型推理）
    NORMAL = 3  # 普通优先级（训练）
    LOW = 4  # 低优先级（回测）


class TaskStatus(Enum):
    """任务状态"""

    PENDING = "pending"  # 等待中
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消
    PREEMPTED = "preempted"  # 被抢占


@dataclass
class GPUTask:
    """GPU任务定义"""

    task_id: str
    model_id: str
    priority: TaskPriority
    memory_required: float  # MB
    estimated_duration: float  # 秒
    callback: Optional[Callable] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    gpu_id: Optional[int] = None
    error_message: Optional[str] = None
    # 新增字段
    deadline: Optional[float] = None  # 截止时间
    preemptible: bool = True  # 是否可被抢占
    resource_requirements: Dict[str, Any] = field(default_factory=dict)  # 资源需求
    affinity_gpu: Optional[int] = None  # GPU亲和性
    # 增强字段
    priority_score: float = 0.0  # 优先级评分
    wait_time: float = 0.0  # 等待时间
    execution_urgency: float = 1.0  # 执行紧急度
    model_compatibility: List[str] = field(default_factory=list)  # 模型兼容性


@dataclass
class GPUResource:
    """GPU资源定义"""

    gpu_id: int
    total_memory: float  # MB
    available_memory: float  # MB
    utilization: float  # 0 - 1
    temperature: float  # 摄氏度
    is_healthy: bool = True
    current_tasks: List[str] = field(default_factory=list)
    task_history: List[str] = field(default_factory=list)
    # 新增字段
    compute_capability: str = "8.6"  # 计算能力
    max_concurrent_tasks: int = 4  # 最大并发任务数
    reserved_memory: float = 0.0  # 预留内存
    task_priorities: List[int] = field(default_factory=list)  # 当前任务优先级列表
    # 增强字段
    model_affinity: Dict[str, float] = field(default_factory=dict)  # 模型亲和性
    load_balancing_score: float = 0.0  # 负载均衡评分
    resource_efficiency: float = 1.0  # 资源效率


class SchedulingPolicy(Enum):
    """调度策略"""

    PRIORITY = "priority"  # 优先级调度
    ROUND_ROBIN = "round_robin"  # 轮询调度
    MEMORY_AWARE = "memory_aware"  # 内存感知调度
    LOAD_BALANCED = "load_balanced"  # 负载均衡调度
    PRIORITY_PREEMPTIVE = "priority_preemptive"  # 优先级抢占调度
    DEADLINE_AWARE = "deadline_aware"  # 截止时间感知调度
    ENHANCED_PRIORITY = "enhanced_priority"  # 增强优先级调度


class ResourceAllocationStrategy(Enum):
    """资源分配策略"""

    FIRST_FIT = "first_fit"  # 首次适配
    BEST_FIT = "best_fit"  # 最佳适配
    WORST_FIT = "worst_fit"  # 最差适配
    PRIORITY_BASED = "priority_based"  # 基于优先级
    ENHANCED_PRIORITY = "enhanced_priority"  # 增强优先级分配


class GPUScheduler:
    """GPU资源调度器"""

    def __init__(
        self,
        gpu_manager,
        policy: SchedulingPolicy = SchedulingPolicy.ENHANCED_PRIORITY,
        max_memory_usage: float = 0.9,  # 最大内存使用率
        enable_graceful_degradation: bool = True,
        enable_tensorrt: bool = False,
        enable_preemption: bool = True,  # 启用抢占
        enable_deadline_aware: bool = True,  # 启用截止时间感知
        allocation_strategy: ResourceAllocationStrategy = ResourceAllocationStrategy.ENHANCED_PRIORITY,
        enable_enhanced_priority: bool = True,  # 启用增强优先级
        enable_load_balancing: bool = True,  # 启用负载均衡
        enable_model_affinity: bool = True,  # 启用模型亲和性
    ):
        """
        初始化GPU调度器

        Args:
            gpu_manager: GPU管理器实例
            policy: 调度策略
            max_memory_usage: 最大内存使用率
            enable_graceful_degradation: 是否启用优雅降级
            enable_tensorrt: 是否启用TensorRT优化
            enable_preemption: 是否启用任务抢占
            enable_deadline_aware: 是否启用截止时间感知
            allocation_strategy: 资源分配策略
            enable_enhanced_priority: 是否启用增强优先级
            enable_load_balancing: 是否启用负载均衡
            enable_model_affinity: 是否启用模型亲和性
        """
        self.gpu_manager = gpu_manager
        self.policy = policy
        self.max_memory_usage = max_memory_usage
        self.enable_graceful_degradation = enable_graceful_degradation
        self.enable_tensorrt = enable_tensorrt
        self.enable_preemption = enable_preemption
        self.enable_deadline_aware = enable_deadline_aware
        self.allocation_strategy = allocation_strategy
        self.enable_enhanced_priority = enable_enhanced_priority
        self.enable_load_balancing = enable_load_balancing
        self.enable_model_affinity = enable_model_affinity

        # 任务管理
        self.tasks: Dict[str, GPUTask] = {}
        self.task_queue: deque = deque()
        self.running_tasks: Dict[str, GPUTask] = {}
        self.preempted_tasks: Dict[str, GPUTask] = {}  # 被抢占的任务

        # 优先级队列（用于抢占调度）
        self.priority_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }

        # 增强优先级队列（按优先级评分排序）
        self.enhanced_priority_queue: List[Tuple[float, str]] = []

        # GPU资源管理
        self.gpu_resources: Dict[int, GPUResource] = {}
        self.gpu_task_mapping: Dict[int, List[str]] = defaultdict(list)

        # 模型亲和性管理
        self.model_affinity_cache: Dict[str, Dict[int, float]] = {}

        # 调度统计
        self.scheduler_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "preempted_tasks": 0,
            "average_wait_time": 0.0,
            "average_execution_time": 0.0,
            "gpu_utilization": defaultdict(float),
            "priority_distribution": defaultdict(int),
            "resource_allocation_efficiency": 0.0,
            "enhanced_priority_allocations": 0,
            "model_affinity_hits": 0,
            "load_balancing_improvements": 0,
        }

        # 线程安全
        self.lock = threading.RLock()
        self.running = False
        self.scheduler_thread = None

        # 初始化GPU资源
        self._init_gpu_resources()

        # 启动调度器
        self.start()

    def _init_gpu_resources(self):
        """初始化GPU资源"""
        try:
            # 获取GPU信息
            gpu_stats = self.gpu_manager.get_gpu_stats()
            if gpu_stats:
                for i, gpu_stat in enumerate(gpu_stats):
                    self.gpu_resources[i] = GPUResource(
                        gpu_id=i,
                        total_memory=gpu_stat.get("memory_total", 8192),  # 默认8GB
                        available_memory=gpu_stat.get("memory_free", 4096),
                        utilization=gpu_stat.get("utilization", 0.0),
                        temperature=gpu_stat.get("temperature", 65.0),
                        is_healthy=gpu_stat.get("is_healthy", True),
                        compute_capability=gpu_stat.get("compute_capability", "8.6"),
                        max_concurrent_tasks=gpu_stat.get("max_concurrent_tasks", 4),
                        reserved_memory=gpu_stat.get(
                            "reserved_memory", 512
                        ),  # 512MB预留
                        model_affinity=gpu_stat.get("model_affinity", {}),
                        load_balancing_score=0.0,
                        resource_efficiency=1.0,
                    )
            else:
                # 模拟GPU资源
                for i in range(2):  # 假设有2个GPU
                    self.gpu_resources[i] = GPUResource(
                        gpu_id=i,
                        total_memory=8192,  # 8GB
                        available_memory=6144,  # 6GB可用
                        utilization=0.0,
                        temperature=65.0,
                        is_healthy=True,
                        compute_capability="8.6",
                        max_concurrent_tasks=4,
                        reserved_memory=512,
                        model_affinity={},
                        load_balancing_score=0.0,
                        resource_efficiency=1.0,
                    )

            logger.info(f"初始化了 {len(self.gpu_resources)} 个GPU资源")
        except Exception as e:
            logger.error(f"初始化GPU资源失败: {e}")

    def _calculate_enhanced_priority_score(self, task: GPUTask) -> float:
        """
        计算增强优先级评分

        Args:
            task: 任务对象

        Returns:
            float: 优先级评分
        """
        # 基础优先级评分
        base_score = (5 - task.priority.value) * 10.0

        # 等待时间奖励
        wait_time_bonus = min(task.wait_time * 0.1, 5.0)

        # 执行紧急度
        urgency_bonus = task.execution_urgency * 3.0

        # 截止时间紧迫性
        deadline_urgency = 0.0
        if task.deadline:
            time_until_deadline = task.deadline - time.time()
            if time_until_deadline > 0:
                deadline_urgency = max(0, 10.0 - time_until_deadline * 0.1)

        # 模型兼容性奖励
        compatibility_bonus = len(task.model_compatibility) * 0.5

        total_score = (
            base_score
            + wait_time_bonus
            + urgency_bonus
            + deadline_urgency
            + compatibility_bonus
        )

        return total_score

    def _update_task_priority_scores(self):
        """更新所有任务的优先级评分"""
        current_time = time.time()

        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                # 更新等待时间
                task.wait_time = current_time - task.created_at

                # 计算增强优先级评分
                task.priority_score = self._calculate_enhanced_priority_score(task)

        # 重新排序增强优先级队列
        self.enhanced_priority_queue = [
            (task.priority_score, task_id)
            for task_id, task in self.tasks.items()
            if task.status == TaskStatus.PENDING
        ]
        self.enhanced_priority_queue.sort(reverse=True)  # 按评分降序排列

    def _calculate_model_affinity_score(self, model_id: str, gpu_id: int) -> float:
        """
        计算模型亲和性评分

        Args:
            model_id: 模型ID
            gpu_id: GPU ID

        Returns:
            float: 亲和性评分
        """
        if not self.enable_model_affinity:
            return 0.0

        # 从缓存获取亲和性
        if (
            model_id in self.model_affinity_cache
            and gpu_id in self.model_affinity_cache[model_id]
        ):
            return self.model_affinity_cache[model_id][gpu_id]

        # 计算亲和性（基于历史执行情况）
        resource = self.gpu_resources[gpu_id]
        affinity_score = resource.model_affinity.get(model_id, 0.5)

        # 缓存结果
        if model_id not in self.model_affinity_cache:
            self.model_affinity_cache[model_id] = {}
        self.model_affinity_cache[model_id][gpu_id] = affinity_score

        return affinity_score

    def _update_model_affinity(self, model_id: str, gpu_id: int, success: bool):
        """
        更新模型亲和性

        Args:
            model_id: 模型ID
            gpu_id: GPU ID
            success: 执行是否成功
        """
        if not self.enable_model_affinity:
            return

        resource = self.gpu_resources[gpu_id]
        current_affinity = resource.model_affinity.get(model_id, 0.5)

        # 根据执行结果调整亲和性
        if success:
            new_affinity = min(1.0, current_affinity + 0.1)
        else:
            new_affinity = max(0.0, current_affinity - 0.1)

        resource.model_affinity[model_id] = new_affinity
        self.scheduler_stats["model_affinity_hits"] += 1

    def _calculate_load_balancing_score(self, gpu_id: int) -> float:
        """
        计算负载均衡评分

        Args:
            gpu_id: GPU ID

        Returns:
            float: 负载均衡评分
        """
        if not self.enable_load_balancing:
            return 0.0

        resource = self.gpu_resources[gpu_id]

        # 内存使用率
        memory_usage = 1.0 - (resource.available_memory / resource.total_memory)

        # 任务数量
        task_count = len(resource.current_tasks)
        task_ratio = task_count / resource.max_concurrent_tasks

        # 利用率
        utilization = resource.utilization

        # 综合负载评分（越低越好）
        load_score = memory_usage * 0.4 + task_ratio * 0.3 + utilization * 0.3

        return 1.0 - load_score  # 转换为越高越好的评分

    def submit_task(
        self,
        task_id: str,
        model_id: str,
        priority: TaskPriority,
        memory_required: float,
        estimated_duration: float = 60.0,
        callback: Optional[Callable] = None,
        deadline: Optional[float] = None,
        preemptible: bool = True,
        resource_requirements: Optional[Dict[str, Any]] = None,
        affinity_gpu: Optional[int] = None,
        execution_urgency: float = 1.0,
        model_compatibility: Optional[List[str]] = None,
    ) -> bool:
        """
        提交GPU任务（增强版）

        Args:
            task_id: 任务ID
            model_id: 模型ID
            priority: 任务优先级
            memory_required: 所需内存(MB)
            estimated_duration: 预估执行时间(秒)
            callback: 完成回调函数
            deadline: 截止时间（时间戳）
            preemptible: 是否可被抢占
            resource_requirements: 资源需求
            affinity_gpu: GPU亲和性
            execution_urgency: 执行紧急度
            model_compatibility: 模型兼容性列表

        Returns:
            bool: 是否成功提交
        """
        with self.lock:
            if task_id in self.tasks:
                logger.warning(f"任务 {task_id} 已存在")
                return False

            task = GPUTask(
                task_id=task_id,
                model_id=model_id,
                priority=priority,
                memory_required=memory_required,
                estimated_duration=estimated_duration,
                callback=callback,
                deadline=deadline,
                preemptible=preemptible,
                resource_requirements=resource_requirements or {},
                affinity_gpu=affinity_gpu,
                execution_urgency=execution_urgency,
                model_compatibility=model_compatibility or [],
            )

            # 计算初始优先级评分
            task.priority_score = self._calculate_enhanced_priority_score(task)

            self.tasks[task_id] = task
            self.task_queue.append(task_id)

            # 添加到优先级队列
            self.priority_queues[priority].append(task_id)

            # 添加到增强优先级队列
            heappush(self.enhanced_priority_queue, (-task.priority_score, task_id))

            self.scheduler_stats["total_tasks"] += 1
            self.scheduler_stats["priority_distribution"][priority.name] += 1

            logger.info(
                f"提交任务 {task_id} (模型: {model_id}, 优先级: {priority.name}, 内存: {memory_required}MB, 评分: {task.priority_score:.2f})"
            )
            return True

    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务（增强版）

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功取消
        """
        with self.lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            if task.status == TaskStatus.RUNNING:
                # 释放GPU资源
                if task.gpu_id is not None:
                    self._release_gpu_resource(task.gpu_id, task_id)

            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()

            # 从优先级队列中移除
            if task_id in self.priority_queues[task.priority]:
                self.priority_queues[task.priority].remove(task_id)

            # 从增强优先级队列中移除
            if task_id in self.enhanced_priority_queue:
                self.enhanced_priority_queue = [
                    (score, t_id)
                    for score, t_id in self.enhanced_priority_queue
                    if t_id != task_id
                ]
                heapify(self.enhanced_priority_queue)  # 重新堆化

            logger.info(f"取消任务 {task_id}")
            return True

    def preempt_task(self, task_id: str) -> bool:
        """
        抢占任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功抢占
        """
        with self.lock:
            if task_id not in self.running_tasks:
                return False

            task = self.running_tasks[task_id]
            if not task.preemptible:
                logger.warning(f"任务 {task_id} 不可抢占")
                return False

            # 标记为被抢占
            task.status = TaskStatus.PREEMPTED
            task.completed_at = time.time()

            # 释放GPU资源
            if task.gpu_id is not None:
                self._release_gpu_resource(task.gpu_id, task_id)

            # 移动到被抢占任务列表
            self.preempted_tasks[task_id] = task
            del self.running_tasks[task_id]

            # 重新加入队列
            self.task_queue.append(task_id)
            self.priority_queues[task.priority].append(task_id)

            # 重新计算增强优先级评分并加入增强优先级队列
            task.priority_score = self._calculate_enhanced_priority_score(task)
            heappush(self.enhanced_priority_queue, (-task.priority_score, task_id))

            self.scheduler_stats["preempted_tasks"] += 1

            logger.info(f"抢占任务 {task_id}")
            return True

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        with self.lock:
            task = self.tasks.get(task_id)
            return task.status if task else None

    def get_gpu_utilization(self) -> Dict[int, float]:
        """获取GPU利用率"""
        with self.lock:
            return {
                gpu_id: resource.utilization
                for gpu_id, resource in self.gpu_resources.items()
            }

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        with self.lock:
            return self.scheduler_stats.copy()

    def get_resource_allocation_efficiency(self) -> float:
        """获取资源分配效率"""
        with self.lock:
            total_memory = sum(
                resource.total_memory for resource in self.gpu_resources.values()
            )
            used_memory = sum(
                resource.total_memory - resource.available_memory
                for resource in self.gpu_resources.values()
            )
            return used_memory / total_memory if total_memory > 0 else 0.0

    def start(self):
        """启动调度器"""
        if self.running:
            return

        self.running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self.scheduler_thread.start()
        logger.info("GPU调度器已启动")

    def stop(self):
        """停止调度器"""
        if not self.running:
            return

        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("GPU调度器已停止")

    def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                with self.lock:
                    self._update_gpu_resources()
                    self._update_task_priority_scores()  # 更新任务优先级评分
                    self._schedule_tasks()
                    self._cleanup_completed_tasks()
                    self._update_scheduler_stats()

                time.sleep(0.1)  # 100ms调度间隔
            except Exception as e:
                logger.error(f"调度器循环异常: {e}")
                time.sleep(1)

    def _update_gpu_resources(self):
        """更新GPU资源状态"""
        try:
            gpu_stats = self.gpu_manager.get_gpu_stats()
            if gpu_stats:
                for i, gpu_stat in enumerate(gpu_stats):
                    if i in self.gpu_resources:
                        resource = self.gpu_resources[i]
                        resource.utilization = gpu_stat.get("utilization", 0.0)
                        resource.temperature = gpu_stat.get("temperature", 65.0)
                        resource.is_healthy = gpu_stat.get("is_healthy", True)

                        # 更新内存使用情况
                        memory_info = gpu_stat.get("memory", {})
                        if memory_info:
                            resource.total_memory = memory_info.get(
                                "total", resource.total_memory
                            )
                            resource.available_memory = memory_info.get(
                                "free", resource.available_memory
                            )

                        # 更新任务优先级列表
                        resource.task_priorities = [
                            self.tasks[task_id].priority.value
                            for task_id in resource.current_tasks
                            if task_id in self.tasks
                        ]
                        # 更新负载均衡评分
                        resource.load_balancing_score = (
                            self._calculate_load_balancing_score(i)
                        )
                        # 更新资源效率
                        resource.resource_efficiency = (
                            self._calculate_resource_efficiency(i)
                        )
            else:
                # 模拟GPU资源
                for i in range(2):  # 假设有2个GPU
                    self.gpu_resources[i] = GPUResource(
                        gpu_id=i,
                        total_memory=8192,  # 8GB
                        available_memory=6144,  # 6GB可用
                        utilization=0.0,
                        temperature=65.0,
                        is_healthy=True,
                        compute_capability="8.6",
                        max_concurrent_tasks=4,
                        reserved_memory=512,
                        model_affinity={},
                        load_balancing_score=0.0,
                        resource_efficiency=1.0,
                    )

            logger.info(f"更新了 {len(self.gpu_resources)} 个GPU资源状态")
        except Exception as e:
            logger.error(f"更新GPU资源状态失败: {e}")

    def _calculate_resource_efficiency(self, gpu_id: int) -> float:
        """
        计算GPU资源效率评分

        Args:
            gpu_id: GPU ID

        Returns:
            float: 资源效率评分
        """
        resource = self.gpu_resources[gpu_id]

        # 内存使用率
        memory_usage = 1.0 - (resource.available_memory / resource.total_memory)

        # 任务数量
        task_count = len(resource.current_tasks)
        task_ratio = task_count / resource.max_concurrent_tasks

        # 利用率
        utilization = resource.utilization

        # 综合评分（越低越好）
        efficiency_score = memory_usage * 0.4 + task_ratio * 0.3 + utilization * 0.3

        return efficiency_score

    def _schedule_tasks(self):
        """调度任务（增强版）"""
        if not self.task_queue:
            return

        # 根据策略选择任务
        if self.policy == SchedulingPolicy.PRIORITY:
            self._schedule_by_priority()
        elif self.policy == SchedulingPolicy.PRIORITY_PREEMPTIVE:
            self._schedule_priority_preemptive()
        elif self.policy == SchedulingPolicy.ROUND_ROBIN:
            self._schedule_round_robin()
        elif self.policy == SchedulingPolicy.MEMORY_AWARE:
            self._schedule_memory_aware()
        elif self.policy == SchedulingPolicy.LOAD_BALANCED:
            self._schedule_load_balanced()
        elif self.policy == SchedulingPolicy.DEADLINE_AWARE:
            self._schedule_deadline_aware()
        elif self.policy == SchedulingPolicy.ENHANCED_PRIORITY:
            self._schedule_enhanced_priority()

    def _schedule_by_priority(self):
        """优先级调度（基础版）"""
        # 按优先级排序任务
        sorted_tasks = sorted(
            [
                task_id
                for task_id in self.task_queue
                if self.tasks[task_id].status == TaskStatus.PENDING
            ],
            key=lambda task_id: self.tasks[task_id].priority.value,
        )

        for task_id in sorted_tasks:
            if self._try_allocate_gpu(task_id):
                self.task_queue.remove(task_id)

    def _schedule_priority_preemptive(self):
        """优先级抢占调度（增强版）"""
        # 按优先级从高到低处理
        for priority in sorted(TaskPriority, key=lambda p: p.value):
            pending_tasks = [
                task_id
                for task_id in self.priority_queues[priority]
                if self.tasks[task_id].status == TaskStatus.PENDING
            ]

            for task_id in pending_tasks:
                self.tasks[task_id]

                # 尝试分配GPU
                if self._try_allocate_gpu_with_preemption(task_id):
                    self.priority_queues[priority].remove(task_id)
                    if task_id in self.task_queue:
                        self.task_queue.remove(task_id)

    def _schedule_deadline_aware(self):
        """截止时间感知调度"""
        if not self.enable_deadline_aware:
            return self._schedule_by_priority()

        # 按截止时间排序
        deadline_tasks = [
            task_id
            for task_id in self.task_queue
            if self.tasks[task_id].status == TaskStatus.PENDING
            and self.tasks[task_id].deadline is not None
        ]

        # 按截止时间和优先级排序
        sorted_tasks = sorted(
            deadline_tasks,
            key=lambda task_id: (
                self.tasks[task_id].deadline or float("inf"),
                self.tasks[task_id].priority.value,
            ),
        )

        for task_id in sorted_tasks:
            if self._try_allocate_gpu(task_id):
                self.task_queue.remove(task_id)

    def _schedule_round_robin(self):
        """轮询调度"""
        if not self.task_queue:
            return

        # 轮询分配GPU
        available_gpus = [
            gpu_id
            for gpu_id, resource in self.gpu_resources.items()
            if resource.is_healthy and resource.utilization < 0.9
        ]

        if not available_gpus:
            return

        current_gpu_index = 0
        for task_id in list(self.task_queue):
            task = self.tasks[task_id]
            if task.status != TaskStatus.PENDING:
                continue

            gpu_id = available_gpus[current_gpu_index % len(available_gpus)]
            if self._try_allocate_gpu(task_id, gpu_id):
                self.task_queue.remove(task_id)
                current_gpu_index += 1

    def _schedule_memory_aware(self):
        """内存感知调度"""
        # 按内存需求排序任务
        sorted_tasks = sorted(
            [
                task_id
                for task_id in self.task_queue
                if self.tasks[task_id].status == TaskStatus.PENDING
            ],
            key=lambda task_id: self.tasks[task_id].memory_required,
            reverse=True,  # 大内存任务优先
        )

        for task_id in sorted_tasks:
            if self._try_allocate_gpu(task_id):
                self.task_queue.remove(task_id)

    def _schedule_load_balanced(self):
        """负载均衡调度"""
        # 选择负载最低的GPU
        available_gpus = [
            (gpu_id, resource)
            for gpu_id, resource in self.gpu_resources.items()
            if resource.is_healthy
        ]

        if not available_gpus:
            return

        # 按利用率排序GPU
        available_gpus.sort(key=lambda x: x[1].utilization)

        for task_id in list(self.task_queue):
            task = self.tasks[task_id]
            if task.status != TaskStatus.PENDING:
                continue

            # 尝试分配到负载最低的GPU
            for gpu_id, resource in available_gpus:
                if self._try_allocate_gpu(task_id, gpu_id):
                    self.task_queue.remove(task_id)
                    break

    def _schedule_enhanced_priority(self):
        """增强优先级调度"""
        if not self.enhanced_priority_queue:
            return

        # 更新任务优先级评分
        self._update_task_priority_scores()

        # 按评分从高到低处理
        processed_tasks = []
        for _, task_id in self.enhanced_priority_queue:
            task = self.tasks[task_id]

            if task.status != TaskStatus.PENDING:
                processed_tasks.append(task_id)
                continue

            # 尝试分配GPU
            if self._try_allocate_gpu_with_preemption(task_id):
                processed_tasks.append(task_id)
                if task_id in self.task_queue:
                    self.task_queue.remove(task_id)

                # 更新统计
                self.scheduler_stats["enhanced_priority_allocations"] += 1

                logger.info(
                    f"增强优先级调度：任务 {task_id} 分配到GPU，评分: {task.priority_score:.2f}"
                )

        # 从增强优先级队列中移除已处理的任务
        for task_id in processed_tasks:
            self.enhanced_priority_queue = [
                (score, t_id)
                for score, t_id in self.enhanced_priority_queue
                if t_id != task_id
            ]

        # 重新堆化
        if self.enhanced_priority_queue:
            heapify(self.enhanced_priority_queue)

    def _try_allocate_gpu_with_preemption(self, task_id: str) -> bool:
        """
        尝试为任务分配GPU（支持抢占）

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功分配
        """
        task = self.tasks[task_id]

        # 检查是否需要优雅降级
        if (
            self.enable_graceful_degradation
            and task.memory_required > self._get_max_available_memory()
        ):
            logger.warning(f"任务 {task_id} 内存需求过大，启用优雅降级")
            return self._handle_graceful_degradation(task)

        # 寻找合适的GPU
        allocated_gpu_id = self._find_best_gpu_with_preemption(task)

        if allocated_gpu_id is not None:
            return self._allocate_gpu(task_id, allocated_gpu_id)

        return False

    def _find_best_gpu_with_preemption(self, task: GPUTask) -> Optional[int]:
        """
        寻找最佳GPU（支持抢占，增强版）

        Args:
            task: 任务对象

        Returns:
            Optional[int]: 最佳GPU ID
        """
        best_gpu_id = None
        best_score = float("-inf")

        for gpu_id, resource in self.gpu_resources.items():
            if not resource.is_healthy:
                continue

            # 检查基本条件
            if task.memory_required > resource.available_memory:
                # 尝试抢占低优先级任务
                if self.enable_preemption and self._can_preempt_on_gpu(gpu_id, task):
                    score = self._calculate_preemption_score(gpu_id, task)
                    if score > best_score:
                        best_score = score
                        best_gpu_id = gpu_id
                continue

            # 检查并发任务数限制
            if len(resource.current_tasks) >= resource.max_concurrent_tasks:
                # 如果启用抢占，检查是否可以抢占低优先级任务
                if self.enable_preemption and self._can_preempt_on_gpu(gpu_id, task):
                    score = self._calculate_preemption_score(gpu_id, task)
                    if score > best_score:
                        best_score = score
                        best_gpu_id = gpu_id
                continue

            # 计算GPU评分
            score = self._calculate_gpu_score(gpu_id, task)
            if score > best_score:
                best_score = score
                best_gpu_id = gpu_id

        return best_gpu_id

    def _can_preempt_on_gpu(self, gpu_id: int, new_task: GPUTask) -> bool:
        """
        检查是否可以在指定GPU上抢占任务（增强版）

        Args:
            gpu_id: GPU ID
            new_task: 新任务

        Returns:
            bool: 是否可以抢占
        """
        resource = self.gpu_resources[gpu_id]

        # 检查是否有可抢占的任务
        for task_id in resource.current_tasks:
            if task_id not in self.tasks:
                continue

            task = self.tasks[task_id]
            if (
                task.preemptible
                and task.priority.value > new_task.priority.value
                and task.status == TaskStatus.RUNNING
            ):

                # 增强检查：考虑执行紧急度和截止时间
                if new_task.execution_urgency > task.execution_urgency:
                    return True

                # 检查截止时间紧迫性
                if new_task.deadline and task.deadline:
                    new_urgency = max(0, new_task.deadline - time.time())
                    old_urgency = max(0, task.deadline - time.time())
                    if new_urgency < old_urgency:
                        return True

                # 检查模型亲和性
                if self.enable_model_affinity:
                    new_affinity = self._calculate_model_affinity_score(
                        new_task.model_id, gpu_id
                    )
                    old_affinity = self._calculate_model_affinity_score(
                        task.model_id, gpu_id
                    )
                    if new_affinity > old_affinity + 0.2:  # 新任务亲和性明显更高
                        return True

                return True

        return False

    def _calculate_preemption_score(self, gpu_id: int, task: GPUTask) -> float:
        """
        计算抢占评分（增强版）

        Args:
            gpu_id: GPU ID
            task: 任务对象

        Returns:
            float: 抢占评分
        """
        resource = self.gpu_resources[gpu_id]

        # 基础评分
        base_score = 1.0 - resource.utilization

        # 优先级奖励（新任务优先级越高，抢占评分越高）
        priority_bonus = (5 - task.priority.value) * 0.3

        # 内存适配度
        memory_fit = min(1.0, resource.available_memory / task.memory_required)

        # 执行紧急度奖励
        urgency_bonus = task.execution_urgency * 0.2

        # 截止时间紧迫性
        deadline_urgency = 0.0
        if task.deadline:
            time_until_deadline = task.deadline - time.time()
            if time_until_deadline > 0:
                deadline_urgency = max(0, 5.0 - time_until_deadline * 0.05)

        # 模型亲和性奖励
        affinity_bonus = 0.0
        if self.enable_model_affinity:
            model_affinity = self._calculate_model_affinity_score(task.model_id, gpu_id)
            affinity_bonus = model_affinity * 0.15

        # 负载均衡考虑
        load_balancing_bonus = 0.0
        if self.enable_load_balancing:
            load_balancing_bonus = resource.load_balancing_score * 0.1

        total_score = (
            base_score
            + priority_bonus
            + memory_fit
            + urgency_bonus
            + deadline_urgency
            + affinity_bonus
            + load_balancing_bonus
        )

        return total_score

    def _calculate_gpu_score(self, gpu_id: int, task: GPUTask) -> float:
        """
        计算GPU评分（增强版）

        Args:
            gpu_id: GPU ID
            task: 任务对象

        Returns:
            float: GPU评分
        """
        resource = self.gpu_resources[gpu_id]

        # 基础内存评分
        memory_score = resource.available_memory / resource.total_memory

        # 利用率评分（越低越好）
        utilization_score = 1.0 - resource.utilization

        # 并发任务评分
        concurrency_score = (
            1.0 - len(resource.current_tasks) / resource.max_concurrent_tasks
        )

        # 亲和性奖励
        affinity_bonus = 0.0
        if task.affinity_gpu == gpu_id:
            affinity_bonus = 0.3
        elif self.enable_model_affinity:
            # 模型亲和性奖励
            model_affinity = self._calculate_model_affinity_score(task.model_id, gpu_id)
            affinity_bonus = model_affinity * 0.2

        # 负载均衡评分
        load_balancing_bonus = 0.0
        if self.enable_load_balancing:
            load_balancing_bonus = resource.load_balancing_score * 0.15

        # 资源效率奖励
        efficiency_bonus = (1.0 - resource.resource_efficiency) * 0.1

        # 温度评分（温度越低越好）
        temperature_score = max(0, 1.0 - resource.temperature / 100.0)

        # 计算能力匹配度
        compute_capability_score = 1.0  # 可以根据任务需求调整

        # 综合评分
        total_score = (
            memory_score * 0.25
            + utilization_score * 0.20
            + concurrency_score * 0.15
            + affinity_bonus
            + load_balancing_bonus
            + efficiency_bonus
            + temperature_score * 0.05
            + compute_capability_score * 0.05
        )

        return total_score

    def _try_allocate_gpu(
        self, task_id: str, preferred_gpu_id: Optional[int] = None
    ) -> bool:
        """
        尝试为任务分配GPU

        Args:
            task_id: 任务ID
            preferred_gpu_id: 首选GPU ID

        Returns:
            bool: 是否成功分配
        """
        task = self.tasks[task_id]

        # 检查是否需要优雅降级
        if (
            self.enable_graceful_degradation
            and task.memory_required > self._get_max_available_memory()
        ):
            logger.warning(f"任务 {task_id} 内存需求过大，启用优雅降级")
            return self._handle_graceful_degradation(task)

        # 寻找合适的GPU
        allocated_gpu_id = None
        if preferred_gpu_id is not None:
            if self._can_allocate_gpu(preferred_gpu_id, task):
                allocated_gpu_id = preferred_gpu_id

        if allocated_gpu_id is None:
            # 根据分配策略寻找最佳GPU
            if self.allocation_strategy == ResourceAllocationStrategy.FIRST_FIT:
                allocated_gpu_id = self._find_first_fit_gpu(task)
            elif self.allocation_strategy == ResourceAllocationStrategy.BEST_FIT:
                allocated_gpu_id = self._find_best_fit_gpu(task)
            elif self.allocation_strategy == ResourceAllocationStrategy.WORST_FIT:
                allocated_gpu_id = self._find_worst_fit_gpu(task)
            elif self.allocation_strategy == ResourceAllocationStrategy.PRIORITY_BASED:
                allocated_gpu_id = self._find_priority_based_gpu(task)
            elif (
                self.allocation_strategy == ResourceAllocationStrategy.ENHANCED_PRIORITY
            ):
                allocated_gpu_id = self._find_enhanced_priority_gpu(task)

        if allocated_gpu_id is not None:
            return self._allocate_gpu(task_id, allocated_gpu_id)

        return False

    def _find_first_fit_gpu(self, task: GPUTask) -> Optional[int]:
        """首次适配策略"""
        for gpu_id, resource in self.gpu_resources.items():
            if self._can_allocate_gpu(gpu_id, task):
                return gpu_id
        return None

    def _find_best_fit_gpu(self, task: GPUTask) -> Optional[int]:
        """最佳适配策略"""
        best_gpu_id = None
        min_waste = float("inf")

        for gpu_id, resource in self.gpu_resources.items():
            if not self._can_allocate_gpu(gpu_id, task):
                continue

            waste = resource.available_memory - task.memory_required
            if waste >= 0 and waste < min_waste:
                min_waste = waste
                best_gpu_id = gpu_id

        return best_gpu_id

    def _find_worst_fit_gpu(self, task: GPUTask) -> Optional[int]:
        """最差适配策略"""
        worst_gpu_id = None
        max_waste = -1

        for gpu_id, resource in self.gpu_resources.items():
            if not self._can_allocate_gpu(gpu_id, task):
                continue

            waste = resource.available_memory - task.memory_required
            if waste >= 0 and waste > max_waste:
                max_waste = waste
                worst_gpu_id = gpu_id

        return worst_gpu_id

    def _find_priority_based_gpu(self, task: GPUTask) -> Optional[int]:
        """基于优先级的GPU选择"""
        return self._find_best_gpu(task)

    def _find_enhanced_priority_gpu(self, task: GPUTask) -> Optional[int]:
        """增强优先级策略下的GPU选择"""
        best_gpu_id = None
        best_score = float("-inf")

        for gpu_id, resource in self.gpu_resources.items():
            if not resource.is_healthy:
                continue

            # 检查基本条件
            if task.memory_required > resource.available_memory:
                # 尝试抢占低优先级任务
                if self.enable_preemption and self._can_preempt_on_gpu(gpu_id, task):
                    score = self._calculate_preemption_score(gpu_id, task)
                    if score > best_score:
                        best_score = score
                        best_gpu_id = gpu_id
                continue

            # 检查并发任务数限制
            if len(resource.current_tasks) >= resource.max_concurrent_tasks:
                # 如果启用抢占，检查是否可以抢占低优先级任务
                if self.enable_preemption and self._can_preempt_on_gpu(gpu_id, task):
                    score = self._calculate_preemption_score(gpu_id, task)
                    if score > best_score:
                        best_score = score
                        best_gpu_id = gpu_id
                continue

            # 计算增强GPU评分
            score = self._calculate_enhanced_gpu_score(gpu_id, task)
            if score > best_score:
                best_score = score
                best_gpu_id = gpu_id

        return best_gpu_id

    def _calculate_enhanced_gpu_score(self, gpu_id: int, task: GPUTask) -> float:
        """
        计算增强GPU评分（专门用于增强优先级策略）

        Args:
            gpu_id: GPU ID
            task: 任务对象

        Returns:
            float: 增强GPU评分
        """
        self.gpu_resources[gpu_id]

        # 基础评分
        base_score = self._calculate_gpu_score(gpu_id, task)

        # 任务优先级权重
        priority_weight = (5 - task.priority.value) * 0.2

        # 执行紧急度权重
        urgency_weight = task.execution_urgency * 0.15

        # 等待时间权重
        wait_time_weight = min(task.wait_time * 0.01, 0.1)

        # 截止时间紧迫性权重
        deadline_weight = 0.0
        if task.deadline:
            time_until_deadline = task.deadline - time.time()
            if time_until_deadline > 0:
                deadline_weight = max(0, 0.2 - time_until_deadline * 0.001)

        # 模型兼容性权重
        compatibility_weight = len(task.model_compatibility) * 0.05

        # 综合评分
        total_score = (
            base_score * 0.6
            + priority_weight
            + urgency_weight
            + wait_time_weight
            + deadline_weight
            + compatibility_weight
        )

        return total_score

    def _can_allocate_gpu(self, gpu_id: int, task: GPUTask) -> bool:
        """检查是否可以分配GPU"""
        if gpu_id not in self.gpu_resources:
            return False

        resource = self.gpu_resources[gpu_id]
        if not resource.is_healthy:
            return False

        # 检查内存（考虑预留内存）
        available_memory = resource.available_memory - resource.reserved_memory
        if task.memory_required > available_memory:
            return False

        # 检查利用率
        if resource.utilization > 0.9:
            return False

        # 检查并发任务数限制
        if len(resource.current_tasks) >= resource.max_concurrent_tasks:
            return False

        return True

    def _find_best_gpu(self, task: GPUTask) -> Optional[int]:
        """寻找最佳GPU"""
        best_gpu_id = None
        best_score = float("-inf")

        for gpu_id, resource in self.gpu_resources.items():
            if not self._can_allocate_gpu(gpu_id, task):
                continue

            score = self._calculate_gpu_score(gpu_id, task)
            if score > best_score:
                best_score = score
                best_gpu_id = gpu_id

        return best_gpu_id

    def _allocate_gpu(self, task_id: str, gpu_id: int) -> bool:
        """分配GPU资源"""
        task = self.tasks[task_id]
        resource = self.gpu_resources[gpu_id]

        # 更新任务状态
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.gpu_id = gpu_id

        # 更新GPU资源
        resource.available_memory -= task.memory_required
        resource.available_memory = min(
            resource.available_memory, resource.total_memory
        )
        resource.current_tasks.append(task_id)
        self.gpu_task_mapping[gpu_id].append(task_id)

        # 更新统计
        self.running_tasks[task_id] = task
        self.scheduler_stats["gpu_utilization"][gpu_id] = resource.utilization
        self.scheduler_stats["enhanced_priority_allocations"] += 1
        self.scheduler_stats["model_affinity_hits"] += 1  # 分配时也会增加亲和性命中

        # 更新模型亲和性
        if self.enable_model_affinity:
            self._update_model_affinity(task.model_id, gpu_id, True)

        # 更新负载均衡评分
        if self.enable_load_balancing:
            resource.load_balancing_score = self._calculate_load_balancing_score(gpu_id)

        # 更新资源效率评分
        resource.resource_efficiency = self._calculate_resource_efficiency(gpu_id)

        logger.info(f"任务 {task_id} 分配到GPU {gpu_id}")
        return True

    def _release_gpu_resource(self, gpu_id: int, task_id: str):
        """释放GPU资源"""
        if gpu_id in self.gpu_resources:
            resource = self.gpu_resources[gpu_id]
            task = self.tasks[task_id]

            # 释放内存
            resource.available_memory += task.memory_required
            resource.available_memory = min(
                resource.available_memory, resource.total_memory
            )

            # 更新任务列表
            if task_id in resource.current_tasks:
                resource.current_tasks.remove(task_id)
            if task_id in self.gpu_task_mapping[gpu_id]:
                self.gpu_task_mapping[gpu_id].remove(task_id)

            # 添加到历史记录
            resource.task_history.append(task_id)
            if len(resource.task_history) > 100:  # 限制历史记录数量
                resource.task_history = resource.task_history[-100:]

            # 更新模型亲和性
            if self.enable_model_affinity:
                self._update_model_affinity(task.model_id, gpu_id, False)

            # 更新负载均衡评分
            if self.enable_load_balancing:
                resource.load_balancing_score = self._calculate_load_balancing_score(
                    gpu_id
                )

            # 更新资源效率评分
            resource.resource_efficiency = self._calculate_resource_efficiency(gpu_id)

    def _handle_graceful_degradation(self, task: GPUTask) -> bool:
        """
        处理优雅降级

        Args:
            task: 需要降级的任务

        Returns:
            bool: 是否成功处理
        """
        logger.info(f"对任务 {task.task_id} 执行优雅降级")

        # 策略1: 减少内存需求
        reduced_memory = task.memory_required * 0.7  # 减少30 % 内存需求
        task.memory_required = reduced_memory

        # 策略2: 寻找部分GPU资源
        for gpu_id, resource in self.gpu_resources.items():
            if resource.is_healthy and resource.available_memory >= reduced_memory:
                return self._allocate_gpu(task.task_id, gpu_id)

        # 策略3: 等待GPU资源释放
        if len(self.running_tasks) > 0:
            logger.info(f"任务 {task.task_id} 等待GPU资源释放")
            return False

        # 策略4: 使用CPU回退
        logger.warning(f"任务 {task.task_id} 降级到CPU执行")
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.running_tasks[task.task_id] = task
        return True

    def _get_max_available_memory(self) -> float:
        """获取最大可用内存"""
        max_memory = 0.0
        for resource in self.gpu_resources.values():
            if resource.is_healthy:
                max_memory = max(
                    max_memory, resource.available_memory - resource.reserved_memory
                )
        return max_memory

    def _cleanup_completed_tasks(self):
        """清理已完成的任务"""
        completed_tasks = []

        for task_id, task in self.running_tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                completed_tasks.append(task_id)

                # 更新统计
                if task.status == TaskStatus.COMPLETED:
                    self.scheduler_stats["completed_tasks"] += 1
                else:
                    self.scheduler_stats["failed_tasks"] += 1

                # 计算执行时间
                if task.started_at and task.completed_at:
                    execution_time = task.completed_at - task.started_at
                    self.scheduler_stats["average_execution_time"] = (
                        self.scheduler_stats["average_execution_time"]
                        * (self.scheduler_stats["completed_tasks"] - 1)
                        + execution_time
                    ) / self.scheduler_stats["completed_tasks"]

                # 释放GPU资源
                if task.gpu_id is not None:
                    self._release_gpu_resource(task.gpu_id, task_id)

                # 执行回调
                if task.callback:
                    try:
                        task.callback(task)
                    except Exception as e:
                        logger.error(f"任务回调执行失败: {e}")

        # 移除已完成的任务
        for task_id in completed_tasks:
            del self.running_tasks[task_id]

    def _update_scheduler_stats(self):
        """更新调度器统计信息"""
        # 更新资源分配效率
        self.scheduler_stats["resource_allocation_efficiency"] = (
            self.get_resource_allocation_efficiency()
        )

    def complete_task(
        self, task_id: str, success: bool = True, error_message: Optional[str] = None
    ):
        """
        标记任务完成

        Args:
            task_id: 任务ID
            success: 是否成功
            error_message: 错误信息
        """
        with self.lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.completed_at = time.time()
            task.error_message = error_message

            logger.info(f"任务 {task_id} {'完成' if success else '失败'}")

    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务详细信息"""
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return None

            return {
                "task_id": task.task_id,
                "model_id": task.model_id,
                "priority": task.priority.name,
                "status": task.status.value,
                "memory_required": task.memory_required,
                "estimated_duration": task.estimated_duration,
                "gpu_id": task.gpu_id,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error_message": task.error_message,
                "deadline": task.deadline,
                "preemptible": task.preemptible,
                "affinity_gpu": task.affinity_gpu,
                "priority_score": task.priority_score,
                "wait_time": task.wait_time,
                "execution_urgency": task.execution_urgency,
                "model_compatibility": task.model_compatibility,
            }

    def get_gpu_resource_info(self, gpu_id: int) -> Optional[Dict[str, Any]]:
        """获取GPU资源详细信息"""
        with self.lock:
            if gpu_id not in self.gpu_resources:
                return None

            resource = self.gpu_resources[gpu_id]
            return {
                "gpu_id": resource.gpu_id,
                "total_memory": resource.total_memory,
                "available_memory": resource.available_memory,
                "utilization": resource.utilization,
                "temperature": resource.temperature,
                "is_healthy": resource.is_healthy,
                "compute_capability": resource.compute_capability,
                "max_concurrent_tasks": resource.max_concurrent_tasks,
                "reserved_memory": resource.reserved_memory,
                "current_tasks": resource.current_tasks,
                "task_priorities": resource.task_priorities,
                "model_affinity": resource.model_affinity,
                "load_balancing_score": resource.load_balancing_score,
                "resource_efficiency": resource.resource_efficiency,
            }

    def get_enhanced_scheduler_stats(self) -> Dict[str, Any]:
        """获取增强调度器统计信息"""
        with self.lock:
            stats = self.scheduler_stats.copy()

            # 添加增强统计信息
            stats.update(
                {
                    "enhanced_priority_queue_size": len(self.enhanced_priority_queue),
                    "model_affinity_cache_size": len(self.model_affinity_cache),
                    "average_priority_score": 0.0,
                    "max_priority_score": 0.0,
                    "min_priority_score": 0.0,
                }
            )

            # 计算优先级评分统计
            if self.tasks:
                priority_scores = [
                    task.priority_score
                    for task in self.tasks.values()
                    if task.status == TaskStatus.PENDING
                ]
                if priority_scores:
                    stats["average_priority_score"] = sum(priority_scores) / len(
                        priority_scores
                    )
                    stats["max_priority_score"] = max(priority_scores)
                    stats["min_priority_score"] = min(priority_scores)

            return stats

    def get_model_affinity_info(self, model_id: str) -> Dict[int, float]:
        """获取模型亲和性信息"""
        with self.lock:
            if model_id not in self.model_affinity_cache:
                return {}

            return self.model_affinity_cache[model_id].copy()

    def get_load_balancing_info(self) -> Dict[int, float]:
        """获取负载均衡信息"""
        with self.lock:
            return {
                gpu_id: resource.load_balancing_score
                for gpu_id, resource in self.gpu_resources.items()
            }
