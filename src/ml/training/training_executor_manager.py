"""
training_executor_manager.py

训练执行器生命周期管理器模块

提供训练执行器的完整生命周期管理：
- 注册和注销
- 心跳维护
- 状态监控
- 故障恢复
- 资源管理

符合架构设计：统一的工作节点生命周期管理

作者: RQA2025 Team
日期: 2026-02-15
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

# 配置日志
logger = logging.getLogger(__name__)

# 导入统一注册表（从分布式协调器层）
from src.infrastructure.distributed.registry import (
    get_unified_worker_registry,
    WorkerType,
    WorkerStatus
)


@dataclass
class TrainingExecutorConfig:
    """训练执行器配置"""
    executor_id: str
    max_concurrent_tasks: int = 1
    heartbeat_interval: int = 10  # 秒
    heartbeat_timeout: int = 30   # 秒
    auto_recovery: bool = True
    recovery_max_retries: int = 3
    gpu_enabled: bool = False
    gpu_devices: List[int] = field(default_factory=list)
    memory_limit_gb: float = 8.0
    cpu_cores: int = 4
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainingExecutorLifecycleManager:
    """
    训练执行器生命周期管理器
    
    管理训练执行器的完整生命周期，包括注册、心跳、状态监控和故障恢复。
    
    Attributes:
        config: 训练执行器配置
        _registry: 统一工作节点注册表
        _heartbeat_task: 心跳任务
        _running: 是否运行中
        _shutdown: 是否关闭
        
    Example:
        >>> config = TrainingExecutorConfig(
        ...     executor_id="training_executor_1",
        ...     max_concurrent_tasks=2,
        ...     gpu_enabled=True
        ... )
        >>> 
        >>> manager = TrainingExecutorLifecycleManager(config)
        >>> await manager.start()
        >>> 
        >>> # 执行训练任务
        >>> await manager.execute_training_task(task_config)
        >>> 
        >>> await manager.stop()
    """
    
    def __init__(self, config: TrainingExecutorConfig):
        """
        初始化训练执行器生命周期管理器
        
        Args:
            config: 训练执行器配置
        """
        self.config = config
        self._registry = get_unified_worker_registry()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
        self._shutdown = False
        self._lock = threading.RLock()
        
        # 任务管理
        self._current_tasks: Dict[str, Dict[str, Any]] = {}
        self._task_callbacks: Dict[str, Callable] = {}
        
        # 统计信息
        self._stats = {
            "registered_at": None,
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "last_heartbeat": None,
            "uptime_seconds": 0
        }
        
        logger.info(f"TrainingExecutorLifecycleManager 初始化完成: {config.executor_id}")
    
    async def start(self) -> bool:
        """
        启动训练执行器
        
        Returns:
            是否启动成功
        """
        if self._running:
            logger.warning(f"训练执行器 {self.config.executor_id} 已在运行")
            return True
        
        try:
            # 注册到统一注册表
            capabilities = {
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "gpu_enabled": self.config.gpu_enabled,
                "gpu_devices": self.config.gpu_devices,
                "memory_limit_gb": self.config.memory_limit_gb,
                "cpu_cores": self.config.cpu_cores,
                "supports_distributed": True,
                "supports_async": True
            }
            
            success = self._registry.register_worker(
                worker_id=self.config.executor_id,
                worker_type=WorkerType.TRAINING_EXECUTOR,
                capabilities=capabilities,
                metadata=self.config.metadata
            )
            
            if not success:
                logger.error(f"训练执行器 {self.config.executor_id} 注册失败")
                return False
            
            self._stats["registered_at"] = datetime.now()
            self._running = True
            
            # 启动心跳任务
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info(f"训练执行器 {self.config.executor_id} 启动成功")
            return True
            
        except Exception as e:
            logger.error(f"训练执行器 {self.config.executor_id} 启动失败: {e}")
            return False
    
    async def stop(self) -> bool:
        """
        停止训练执行器
        
        Returns:
            是否停止成功
        """
        if not self._running:
            return True
        
        self._shutdown = True
        self._running = False
        
        try:
            # 取消心跳任务
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # 更新状态为停止中
            self._registry.update_status(
                self.config.executor_id,
                WorkerStatus.STOPPING
            )
            
            # 等待当前任务完成
            if self._current_tasks:
                logger.info(f"等待 {len(self._current_tasks)} 个任务完成...")
                await asyncio.sleep(2)
            
            # 从注册表注销
            self._registry.unregister_worker(self.config.executor_id)
            
            logger.info(f"训练执行器 {self.config.executor_id} 已停止")
            return True
            
        except Exception as e:
            logger.error(f"训练执行器 {self.config.executor_id} 停止时出错: {e}")
            return False
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while not self._shutdown:
            try:
                # 发送心跳
                self._registry.update_heartbeat(self.config.executor_id)
                self._stats["last_heartbeat"] = datetime.now()
                
                # 更新运行时间
                if self._stats["registered_at"]:
                    self._stats["uptime_seconds"] = (
                        datetime.now() - self._stats["registered_at"]
                    ).seconds
                
                # 更新负载信息
                await self._update_load_info()
                
            except Exception as e:
                logger.error(f"训练执行器 {self.config.executor_id} 心跳发送失败: {e}")
            
            # 等待下一次心跳
            await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _update_load_info(self):
        """更新负载信息"""
        try:
            worker = self._registry.get_worker(self.config.executor_id)
            if worker:
                # 计算当前负载
                current_load = len(self._current_tasks) / self.config.max_concurrent_tasks
                worker.current_load = current_load
                
                # 更新元数据
                worker.metadata.update({
                    "current_tasks": list(self._current_tasks.keys()),
                    "uptime_seconds": self._stats["uptime_seconds"],
                    "last_heartbeat_time": datetime.now().isoformat()
                })
        except Exception as e:
            logger.warning(f"更新负载信息失败: {e}")
    
    async def execute_training_task(
        self,
        task_id: str,
        task_config: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> bool:
        """
        执行训练任务
        
        Args:
            task_id: 任务ID
            task_config: 任务配置
            callback: 完成回调
            
        Returns:
            是否开始执行
        """
        # 检查是否还有可用槽位
        if len(self._current_tasks) >= self.config.max_concurrent_tasks:
            logger.warning(f"训练执行器 {self.config.executor_id} 已达到最大并发任务数")
            return False
        
        # 分配任务
        success = self._registry.assign_task(self.config.executor_id, task_id)
        if not success:
            logger.error(f"为训练执行器 {self.config.executor_id} 分配任务失败")
            return False
        
        # 记录任务
        with self._lock:
            self._current_tasks[task_id] = {
                "config": task_config,
                "started_at": datetime.now(),
                "status": "running"
            }
            if callback:
                self._task_callbacks[task_id] = callback
        
        self._stats["total_tasks"] += 1
        
        logger.info(f"训练执行器 {self.config.executor_id} 开始执行任务: {task_id}")
        
        # 启动任务执行
        asyncio.create_task(self._run_training_task(task_id, task_config))
        
        return True
    
    async def _run_training_task(self, task_id: str, task_config: Dict[str, Any]):
        """运行训练任务"""
        try:
            # 更新状态为忙碌
            self._registry.update_status(
                self.config.executor_id,
                WorkerStatus.BUSY
            )
            
            # 模拟训练过程（实际应调用训练代码）
            logger.info(f"训练任务 {task_id} 正在执行...")
            
            # TODO: 这里应该调用实际的训练代码
            # from src.ml.training import execute_training
            # result = await execute_training(task_config)
            
            # 模拟训练时间
            await asyncio.sleep(5)
            
            # 任务完成
            processing_time = 5.0  # 实际应从任务执行时间计算
            self._registry.complete_task(self.config.executor_id, processing_time)
            
            self._stats["completed_tasks"] += 1
            
            # 调用回调
            if task_id in self._task_callbacks:
                callback = self._task_callbacks[task_id]
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(task_id, {"status": "completed"})
                    else:
                        callback(task_id, {"status": "completed"})
                except Exception as e:
                    logger.error(f"任务回调执行失败: {e}")
            
            logger.info(f"训练任务 {task_id} 完成")
            
        except Exception as e:
            logger.error(f"训练任务 {task_id} 执行失败: {e}")
            
            # 任务失败
            self._registry.fail_task(self.config.executor_id)
            self._stats["failed_tasks"] += 1
            
            # 调用回调（带错误信息）
            if task_id in self._task_callbacks:
                callback = self._task_callbacks[task_id]
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(task_id, {"status": "failed", "error": str(e)})
                    else:
                        callback(task_id, {"status": "failed", "error": str(e)})
                except Exception as callback_error:
                    logger.error(f"任务错误回调执行失败: {callback_error}")
        
        finally:
            # 清理任务记录
            with self._lock:
                self._current_tasks.pop(task_id, None)
                self._task_callbacks.pop(task_id, None)
            
            # 如果没有更多任务，更新状态为空闲
            if not self._current_tasks:
                self._registry.update_status(
                    self.config.executor_id,
                    WorkerStatus.IDLE
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            **self._stats,
            "executor_id": self.config.executor_id,
            "current_tasks_count": len(self._current_tasks),
            "current_tasks": list(self._current_tasks.keys()),
            "is_running": self._running,
            "is_shutdown": self._shutdown
        }
    
    def is_healthy(self) -> bool:
        """
        检查健康状态
        
        Returns:
            是否健康
        """
        if not self._running:
            return False
        
        # 检查最后一次心跳时间
        if self._stats["last_heartbeat"]:
            elapsed = (datetime.now() - self._stats["last_heartbeat"]).seconds
            if elapsed > self.config.heartbeat_timeout:
                return False
        
        return True


class TrainingExecutorPool:
    """
    训练执行器池
    
    管理多个训练执行器的池化资源。
    
    Attributes:
        _executors: 执行器字典
        _pool_config: 池配置
        
    Example:
        >>> pool = TrainingExecutorPool()
        >>> 
        >>> # 添加执行器
        >>> await pool.add_executor(config1)
        >>> await pool.add_executor(config2)
        >>> 
        >>> # 获取可用执行器
        >>> executor = pool.get_available_executor()
        >>> 
        >>> # 提交训练任务
        >>> await pool.submit_training_task(task_config)
    """
    
    def __init__(self):
        """初始化训练执行器池"""
        self._executors: Dict[str, TrainingExecutorLifecycleManager] = {}
        self._lock = threading.RLock()
        
        logger.info("TrainingExecutorPool 初始化完成")
    
    async def add_executor(self, config: TrainingExecutorConfig) -> bool:
        """
        添加训练执行器
        
        Args:
            config: 训练执行器配置
            
        Returns:
            是否添加成功
        """
        with self._lock:
            if config.executor_id in self._executors:
                logger.warning(f"训练执行器 {config.executor_id} 已存在")
                return False
            
            manager = TrainingExecutorLifecycleManager(config)
            success = await manager.start()
            
            if success:
                self._executors[config.executor_id] = manager
                logger.info(f"训练执行器 {config.executor_id} 已添加到池")
                return True
            else:
                logger.error(f"训练执行器 {config.executor_id} 启动失败")
                return False
    
    async def remove_executor(self, executor_id: str) -> bool:
        """
        移除训练执行器
        
        Args:
            executor_id: 执行器ID
            
        Returns:
            是否移除成功
        """
        with self._lock:
            if executor_id not in self._executors:
                return False
            
            manager = self._executors[executor_id]
            await manager.stop()
            
            del self._executors[executor_id]
            logger.info(f"训练执行器 {executor_id} 已从池移除")
            return True
    
    def get_available_executor(self) -> Optional[TrainingExecutorLifecycleManager]:
        """
        获取可用的训练执行器
        
        Returns:
            可用的训练执行器管理器，如果没有则返回None
        """
        with self._lock:
            for executor_id, manager in self._executors.items():
                if manager.is_healthy():
                    worker = manager._registry.get_worker(executor_id)
                    if worker and worker.status == WorkerStatus.IDLE:
                        return manager
            
            return None
    
    async def submit_training_task(
        self,
        task_id: str,
        task_config: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> Optional[str]:
        """
        提交训练任务
        
        Args:
            task_id: 任务ID
            task_config: 任务配置
            callback: 完成回调
            
        Returns:
            执行器ID，如果没有可用执行器则返回None
        """
        manager = self.get_available_executor()
        
        if not manager:
            logger.warning("没有可用的训练执行器")
            return None
        
        success = await manager.execute_training_task(task_id, task_config, callback)
        
        if success:
            return manager.config.executor_id
        else:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取池统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total_executors = len(self._executors)
            healthy_executors = sum(
                1 for m in self._executors.values() if m.is_healthy()
            )
            
            return {
                "total_executors": total_executors,
                "healthy_executors": healthy_executors,
                "unhealthy_executors": total_executors - healthy_executors,
                "executor_stats": {
                    executor_id: manager.get_stats()
                    for executor_id, manager in self._executors.items()
                }
            }
    
    async def shutdown_all(self):
        """关闭所有执行器"""
        with self._lock:
            for executor_id, manager in list(self._executors.items()):
                await manager.stop()
            
            self._executors.clear()
            logger.info("所有训练执行器已关闭")


# 全局训练执行器池实例
_global_pool: Optional[TrainingExecutorPool] = None


def get_training_executor_pool() -> TrainingExecutorPool:
    """
    获取全局训练执行器池实例
    
    Returns:
        训练执行器池实例
    """
    global _global_pool
    
    if _global_pool is None:
        _global_pool = TrainingExecutorPool()
    
    return _global_pool
