#!/usr/bin/env python3
"""
RQA2025 流程实例池管理组件

提供企业级的业务流程实例池管理功能，支持实例创建、分配、回收和监控。
优化系统资源利用率，提高业务流程执行效率。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import uuid
import time

from ...base import ComponentStatus, ComponentHealth
from ...patterns.standard_interface_template import StandardComponent

logger = logging.getLogger(__name__)


class ProcessInstanceStatus(Enum):
    """流程实例状态"""
    CREATED = "created"         # 已创建
    ACTIVE = "active"          # 活动中
    SUSPENDED = "suspended"    # 已暂停
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"         # 已失败
    TERMINATED = "terminated"  # 已终止
    EXPIRED = "expired"        # 已过期


class PoolStrategy(Enum):
    """池化策略"""
    FIXED_SIZE = "fixed_size"           # 固定大小
    DYNAMIC_SIZE = "dynamic_size"       # 动态大小
    ADAPTIVE_SIZE = "adaptive_size"     # 自适应大小


@dataclass
class ProcessInstance:
    """流程实例"""
    instance_id: str
    process_type: str
    status: ProcessInstanceStatus = ProcessInstanceStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_active_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: int = 3600  # 1小时超时
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 1  # 1-10, 10最高
    context_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """是否处于活动状态"""
        return self.status == ProcessInstanceStatus.ACTIVE

    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.status in [ProcessInstanceStatus.COMPLETED, ProcessInstanceStatus.FAILED]:
            return False

        timeout = timedelta(seconds=self.timeout_seconds)
        return datetime.now() - self.last_active_at > timeout

    @property
    def duration(self) -> Optional[float]:
        """执行时长（秒）"""
        if not self.started_at:
            return None

        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()


@dataclass
class PoolStats:
    """池统计信息"""
    total_instances: int = 0
    active_instances: int = 0
    idle_instances: int = 0
    completed_instances: int = 0
    failed_instances: int = 0
    expired_instances: int = 0
    avg_creation_time: float = 0.0
    avg_completion_time: float = 0.0
    pool_hit_rate: float = 0.0
    resource_utilization: float = 0.0


class ProcessInstanceFactory(ABC):
    """流程实例工厂基类"""

    @abstractmethod
    def create_instance(self, process_type: str, context_data: Optional[Dict[str, Any]] = None) -> ProcessInstance:
        """创建流程实例"""

    @abstractmethod
    def destroy_instance(self, instance: ProcessInstance) -> bool:
        """销毁流程实例"""

    @abstractmethod
    def validate_instance(self, instance: ProcessInstance) -> bool:
        """验证实例有效性"""


class DefaultProcessInstanceFactory(ProcessInstanceFactory):
    """默认流程实例工厂"""

    def create_instance(self, process_type: str, context_data: Optional[Dict[str, Any]] = None) -> ProcessInstance:
        """创建流程实例"""
        instance_id = str(uuid.uuid4())

        instance = ProcessInstance(
            instance_id=instance_id,
            process_type=process_type,
            context_data=context_data or {},
            metadata={
                'factory': 'DefaultProcessInstanceFactory',
                'created_by': 'pool_manager'
            }
        )

        logger.debug(f"创建流程实例: {instance_id} (类型: {process_type})")
        return instance

    def destroy_instance(self, instance: ProcessInstance) -> bool:
        """销毁流程实例"""
        try:
            # 清理实例资源
            instance.context_data.clear()
            instance.metadata.clear()
            instance.status = ProcessInstanceStatus.TERMINATED

            logger.debug(f"销毁流程实例: {instance.instance_id}")
            return True

        except Exception as e:
            logger.error(f"销毁流程实例失败 {instance.instance_id}: {e}")
            return False

    def validate_instance(self, instance: ProcessInstance) -> bool:
        """验证实例有效性"""
        # 检查基本属性
        if not instance.instance_id or not instance.process_type:
            return False

        # 检查状态一致性
        if instance.status == ProcessInstanceStatus.EXPIRED and not instance.is_expired:
            return False

        # 检查时间戳合理性
        now = datetime.now()
        if instance.created_at > now:
            return False

        if instance.started_at and instance.started_at < instance.created_at:
            return False

        return True


class ProcessInstancePool(StandardComponent):
    """流程实例池管理器"""

    def __init__(self, pool_name: str, strategy: PoolStrategy = PoolStrategy.DYNAMIC_SIZE,
                 config: Optional[Dict[str, Any]] = None):
        """初始化流程实例池

        Args:
            pool_name: 池名称
            strategy: 池化策略
            config: 配置参数
        """
        super().__init__(pool_name, "2.0.0", f"{pool_name}流程实例池")

        self.pool_name = pool_name
        self.strategy = strategy
        self.config = config or {}

        # 池配置
        self.min_size = self.config.get('min_size', 10)
        self.max_size = self.config.get('max_size', 100)
        self.idle_timeout = self.config.get('idle_timeout', 300)  # 5分钟
        self.creation_timeout = self.config.get('creation_timeout', 30)  # 30秒

        # 实例管理
        self._instances: Dict[str, ProcessInstance] = {}
        self._idle_instances: List[ProcessInstance] = []
        self._active_instances: Dict[str, ProcessInstance] = {}
        self._factory: Optional[ProcessInstanceFactory] = None

        # 统计信息
        self._stats = PoolStats()
        self._creation_times: List[float] = []
        self._completion_times: List[float] = []

        # 监控和清理
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False

        # 事件回调
        self._instance_created_callbacks: List[Callable] = []
        self._instance_destroyed_callbacks: List[Callable] = []

    def set_factory(self, factory: ProcessInstanceFactory):
        """设置实例工厂"""
        with self._lock:
            self._factory = factory

    def initialize(self) -> bool:
        """初始化实例池"""
        try:
            self.set_status(ComponentStatus.INITIALIZING)

            # 设置默认工厂
            if not self._factory:
                self._factory = DefaultProcessInstanceFactory()

            # 预热实例池
            self._warmup_pool()

            # 启动清理线程
            self._running = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name=f"{self.pool_name}_cleanup",
                daemon=True
            )
            self._cleanup_thread.start()

            self.set_status(ComponentStatus.INITIALIZED)
            self.set_health(ComponentHealth.HEALTHY)

            logger.info(f"流程实例池 {self.pool_name} 初始化完成")
            return True

        except Exception as e:
            self.set_status(ComponentStatus.ERROR)
            self.set_health(ComponentHealth.UNHEALTHY)
            logger.error(f"流程实例池 {self.pool_name} 初始化失败: {e}")
            return False

    def shutdown(self) -> bool:
        """关闭实例池"""
        try:
            self.set_status(ComponentStatus.STOPPING)
            self._running = False

            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5)

            # 清理所有实例
            self._cleanup_all_instances()

            self.set_status(ComponentStatus.STOPPED)
            logger.info(f"流程实例池 {self.pool_name} 已关闭")
            return True

        except Exception as e:
            logger.error(f"流程实例池 {self.pool_name} 关闭失败: {e}")
            return False

    def acquire_instance(self, process_type: str,
                         context_data: Optional[Dict[str, Any]] = None,
                         priority: int = 1,
                         timeout: Optional[float] = None) -> Optional[ProcessInstance]:
        """获取流程实例"""
        start_time = time.time()
        timeout = timeout or self.creation_timeout

        try:
            with self._lock:
                # 尝试从空闲池获取
                instance = self._acquire_from_idle_pool(process_type)
                if instance:
                    self._activate_instance(instance, context_data)
                    return instance

                # 检查是否可以创建新实例
                if len(self._instances) >= self.max_size:
                    logger.warning(f"实例池已达到最大容量 {self.max_size}")
                    return None

                # 创建新实例
                instance = self._create_new_instance(process_type, context_data, priority)
                if instance:
                    self._activate_instance(instance, context_data)
                    return instance

                return None

        except Exception as e:
            logger.error(f"获取流程实例失败 (类型: {process_type}): {e}")
            return None

    def release_instance(self, instance_id: str) -> bool:
        """释放流程实例"""
        try:
            with self._lock:
                if instance_id not in self._active_instances:
                    logger.warning(f"实例 {instance_id} 不在活动池中")
                    return False

                instance = self._active_instances.pop(instance_id)

                # 更新实例状态
                instance.status = ProcessInstanceStatus.CREATED
                instance.last_active_at = datetime.now()

                # 检查是否可以回收到空闲池
                if self._should_recycle_to_idle(instance):
                    self._idle_instances.append(instance)
                    self._update_stats()
                    logger.debug(f"实例 {instance_id} 已回收到空闲池")
                else:
                    # 销毁实例
                    self._destroy_instance(instance)

                return True

        except Exception as e:
            logger.error(f"释放流程实例失败 {instance_id}: {e}")
            return False

    def destroy_instance(self, instance_id: str) -> bool:
        """销毁流程实例"""
        try:
            with self._lock:
                instance = None

                # 从不同池中查找实例
                if instance_id in self._active_instances:
                    instance = self._active_instances.pop(instance_id)
                elif instance_id in self._instances:
                    instance = self._instances[instance_id]
                    # 从空闲池移除
                    self._idle_instances = [
                        i for i in self._idle_instances if i.instance_id != instance_id]

                if instance:
                    return self._destroy_instance(instance)

                logger.warning(f"实例 {instance_id} 不存在")
                return False

        except Exception as e:
            logger.error(f"销毁流程实例失败 {instance_id}: {e}")
            return False

    def get_instance(self, instance_id: str) -> Optional[ProcessInstance]:
        """获取实例信息"""
        with self._lock:
            return self._instances.get(instance_id)

    def get_active_instances(self) -> List[ProcessInstance]:
        """获取所有活动实例"""
        with self._lock:
            return list(self._active_instances.values())

    def get_idle_instances(self) -> List[ProcessInstance]:
        """获取所有空闲实例"""
        with self._lock:
            return self._idle_instances.copy()

    def get_stats(self) -> PoolStats:
        """获取池统计信息"""
        with self._lock:
            self._update_stats()
            return PoolStats(
                total_instances=len(self._instances),
                active_instances=len(self._active_instances),
                idle_instances=len(self._idle_instances),
                completed_instances=self._stats.completed_instances,
                failed_instances=self._stats.failed_instances,
                expired_instances=self._stats.expired_instances,
                avg_creation_time=sum(self._creation_times) /
                len(self._creation_times) if self._creation_times else 0.0,
                avg_completion_time=sum(self._completion_times) /
                len(self._completion_times) if self._completion_times else 0.0,
                pool_hit_rate=self._calculate_hit_rate(),
                resource_utilization=self._calculate_utilization()
            )

    def force_cleanup(self) -> Dict[str, int]:
        """强制清理过期实例"""
        try:
            with self._lock:
                cleaned = {'expired': 0, 'idle': 0, 'failed': 0}

                # 清理过期实例
                expired_instances = []
                for instance in list(self._instances.values()):
                    if instance.is_expired:
                        expired_instances.append(instance)
                        cleaned['expired'] += 1

                for instance in expired_instances:
                    self._destroy_instance(instance)

                # 清理超时的空闲实例
                idle_timeout = timedelta(seconds=self.idle_timeout)
                now = datetime.now()

                expired_idle = []
                for instance in self._idle_instances:
                    if now - instance.last_active_at > idle_timeout:
                        expired_idle.append(instance)
                        cleaned['idle'] += 1

                for instance in expired_idle:
                    self._idle_instances.remove(instance)
                    self._destroy_instance(instance)

                # 清理失败实例
                failed_instances = [i for i in self._instances.values(
                ) if i.status == ProcessInstanceStatus.FAILED]
                for instance in failed_instances:
                    self._destroy_instance(instance)
                    cleaned['failed'] += 1

                self._update_stats()
                logger.info(f"强制清理完成: {cleaned}")
                return cleaned

        except Exception as e:
            logger.error(f"强制清理失败: {e}")
            return {}

    def add_instance_created_callback(self, callback: Callable):
        """添加实例创建回调"""
        self._instance_created_callbacks.append(callback)

    def add_instance_destroyed_callback(self, callback: Callable):
        """添加实例销毁回调"""
        self._instance_destroyed_callbacks.append(callback)

    def _warmup_pool(self):
        """预热实例池"""
        try:
            warmup_count = min(self.min_size, self.max_size)
            logger.info(f"预热实例池，创建 {warmup_count} 个实例")

            for i in range(warmup_count):
                instance = self._factory.create_instance("warmup") if self._factory else None
                if instance:
                    self._instances[instance.instance_id] = instance
                    self._idle_instances.append(instance)

            self._update_stats()

        except Exception as e:
            logger.error(f"实例池预热失败: {e}")

    def _acquire_from_idle_pool(self, process_type: str) -> Optional[ProcessInstance]:
        """从空闲池获取实例"""
        # 优先获取相同类型的实例
        for instance in self._idle_instances:
            if instance.process_type == process_type and not instance.is_expired:
                self._idle_instances.remove(instance)
                return instance

        # 如果没有相同类型的，获取任何可用的实例
        for instance in self._idle_instances:
            if not instance.is_expired:
                self._idle_instances.remove(instance)
                return instance

        return None

    def _create_new_instance(self, process_type: str, context_data: Optional[Dict[str, Any]],
                             priority: int) -> Optional[ProcessInstance]:
        """创建新实例"""
        try:
            if not self._factory:
                return None

            start_time = time.time()
            instance = self._factory.create_instance(process_type, context_data)
            creation_time = time.time() - start_time

            if instance:
                instance.priority = priority
                self._instances[instance.instance_id] = instance
                self._creation_times.append(creation_time)

                # 保持创建时间历史记录
                if len(self._creation_times) > 100:
                    self._creation_times.pop(0)

                self._notify_instance_created(instance)
                logger.debug(f"创建新实例 {instance.instance_id} 耗时: {creation_time:.3f}s")

            return instance

        except Exception as e:
            logger.error(f"创建新实例失败 (类型: {process_type}): {e}")
            return None

    def _activate_instance(self, instance: ProcessInstance, context_data: Optional[Dict[str, Any]]):
        """激活实例"""
        instance.status = ProcessInstanceStatus.ACTIVE
        instance.started_at = datetime.now()
        instance.last_active_at = datetime.now()
        instance.context_data.update(context_data or {})

        self._active_instances[instance.instance_id] = instance
        self._update_stats()

    def _destroy_instance(self, instance: ProcessInstance) -> bool:
        """销毁实例"""
        try:
            # 从所有池中移除
            self._instances.pop(instance.instance_id, None)
            if instance.instance_id in self._active_instances:
                del self._active_instances[instance.instance_id]

            try:
                self._idle_instances.remove(instance)
            except ValueError:
                pass

            # 通知销毁回调
            self._notify_instance_destroyed(instance)

            # 使用工厂销毁
            if self._factory:
                success = self._factory.destroy_instance(instance)
            else:
                success = True

            # 更新统计
            if instance.status == ProcessInstanceStatus.COMPLETED:
                self._stats.completed_instances += 1
                if instance.duration:
                    self._completion_times.append(instance.duration)
                    if len(self._completion_times) > 100:
                        self._completion_times.pop(0)
            elif instance.status == ProcessInstanceStatus.FAILED:
                self._stats.failed_instances += 1
            elif instance.status == ProcessInstanceStatus.EXPIRED:
                self._stats.expired_instances += 1

            self._update_stats()
            return success

        except Exception as e:
            logger.error(f"销毁实例失败 {instance.instance_id}: {e}")
            return False

    def _should_recycle_to_idle(self, instance: ProcessInstance) -> bool:
        """判断是否应该回收到空闲池"""
        # 检查实例是否仍然有效
        if not self._factory or not self._factory.validate_instance(instance):
            return False

        # 检查空闲池大小
        if len(self._idle_instances) >= self.min_size:
            return False

        # 检查实例状态
        if instance.status not in [ProcessInstanceStatus.CREATED, ProcessInstanceStatus.COMPLETED]:
            return False

        return True

    def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                time.sleep(60)  # 每分钟清理一次
                self.force_cleanup()
            except Exception as e:
                logger.error(f"清理循环异常: {e}")
                time.sleep(5)

    def _cleanup_all_instances(self):
        """清理所有实例"""
        try:
            with self._lock:
                all_instances = list(self._instances.values())
                for instance in all_instances:
                    self._destroy_instance(instance)

                self._idle_instances.clear()
                self._active_instances.clear()
                self._instances.clear()

        except Exception as e:
            logger.error(f"清理所有实例失败: {e}")

    def _update_stats(self):
        """更新统计信息"""
        # 这里可以添加更复杂的统计计算

    def _calculate_hit_rate(self) -> float:
        """计算池命中率"""
        # 简化的命中率计算
        total_instances = len(self._instances)
        if total_instances == 0:
            return 0.0

        active_instances = len(self._active_instances)
        return (total_instances - active_instances) / total_instances

    def _calculate_utilization(self) -> float:
        """计算资源利用率"""
        total_instances = len(self._instances)
        if total_instances == 0:
            return 0.0

        active_instances = len(self._active_instances)
        return active_instances / total_instances

    def _notify_instance_created(self, instance: ProcessInstance):
        """通知实例创建"""
        for callback in self._instance_created_callbacks:
            try:
                callback("created", instance)
            except Exception as e:
                logger.error(f"实例创建回调执行失败: {e}")

    def _notify_instance_destroyed(self, instance: ProcessInstance):
        """通知实例销毁"""
        for callback in self._instance_destroyed_callbacks:
            try:
                callback("destroyed", instance)
            except Exception as e:
                logger.error(f"实例销毁回调执行失败: {e}")

    def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查（StandardComponent要求）"""
        try:
            stats = self.get_stats()

            health_status = {
                'component_name': self.service_name,
                'status': 'healthy',
                'total_instances': stats.total_instances,
                'active_instances': stats.active_instances,
                'idle_instances': stats.idle_instances,
                'pool_utilization': stats.resource_utilization,
                'pool_hit_rate': stats.pool_hit_rate,
                'expired_instances': stats.expired_instances,
                'failed_instances': stats.failed_instances,
                'last_check': datetime.now().isoformat()
            }

            # 检查池健康状态
            if stats.resource_utilization > 0.9:  # 利用率过高
                health_status['status'] = 'warning'
            elif stats.expired_instances > stats.total_instances * 0.1:  # 过期实例太多
                health_status['status'] = 'warning'

            return health_status

        except Exception as e:
            logger.error(f"实例池健康检查失败: {e}")
            return {
                'component_name': self.service_name,
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
