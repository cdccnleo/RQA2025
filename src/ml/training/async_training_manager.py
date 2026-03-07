"""
async_training_manager.py

异步训练管理器模块

提供异步模型训练功能，支持：
- 异步训练任务提交
- 训练任务队列管理
- 训练进度实时监控
- 训练资源动态调度
- 训练结果回调通知
- 集成训练执行器生命周期管理

适用于大规模模型训练场景，提升训练效率和资源利用率。

作者: RQA2025 Team
日期: 2026-02-13
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

# 导入训练执行器生命周期管理
from .training_executor_manager import (
    TrainingExecutorLifecycleManager,
    TrainingExecutorConfig,
    get_training_executor_pool
)

# 配置日志
logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """训练状态枚举"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingTask:
    """训练任务数据类"""
    task_id: str
    model_type: str
    config: Dict[str, Any]
    status: TrainingStatus = TrainingStatus.PENDING
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    callback: Optional[Callable] = None


@dataclass
class AsyncTrainingConfig:
    """异步训练配置"""
    max_concurrent_tasks: int = 3
    task_queue_size: int = 100
    enable_priority_queue: bool = True
    auto_retry_failed: bool = True
    max_retry_count: int = 3
    progress_update_interval: int = 5  # 秒


class AsyncTrainingManager:
    """
    异步训练管理器
    
    管理异步模型训练任务，支持任务队列、并发控制和进度监控。
    
    Attributes:
        config: 异步训练配置
        _task_queue: 任务队列
        _running_tasks: 运行中的任务
        _completed_tasks: 已完成的任务
        
    Example:
        >>> manager = AsyncTrainingManager()
        >>> 
        >>> # 提交异步训练任务
        >>> task = await manager.submit_training_task(
        ...     model_type="LSTM",
        ...     config={...},
        ...     callback=on_training_complete
        ... )
        >>> 
        >>> # 监控训练进度
        >>> progress = manager.get_task_progress(task.task_id)
    """
    
    def __init__(self, config: Optional[AsyncTrainingConfig] = None):
        """
        初始化异步训练管理器
        
        Args:
            config: 异步训练配置
        """
        self.config = config or AsyncTrainingConfig()
        self._task_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.task_queue_size
        )
        self._running_tasks: Dict[str, TrainingTask] = {}
        self._completed_tasks: Dict[str, TrainingTask] = {}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        self._shutdown = False
        self._lock = threading.RLock()
        
        # 训练执行器生命周期管理器
        self._executor_manager: Optional[TrainingExecutorLifecycleManager] = None
        self._executor_id: Optional[str] = None
        
        # 启动任务处理循环
        self._worker_task = asyncio.create_task(self._process_task_queue())
        
        logger.info(f"AsyncTrainingManager 初始化完成: "
                   f"max_concurrent={self.config.max_concurrent_tasks}")
        
        # 初始化训练执行器
        asyncio.create_task(self._init_training_executor())
    
    async def _init_training_executor(self):
        """初始化训练执行器"""
        try:
            import os
            import threading
            
            # 生成唯一的执行器ID
            self._executor_id = f"training_executor_{threading.current_thread().ident}_{os.getpid()}"
            
            # 创建训练执行器配置
            executor_config = TrainingExecutorConfig(
                executor_id=self._executor_id,
                max_concurrent_tasks=self.config.max_concurrent_tasks,
                heartbeat_interval=10,
                heartbeat_timeout=30,
                auto_recovery=True,
                metadata={
                    "manager_type": "AsyncTrainingManager",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            # 创建并启动训练执行器生命周期管理器
            self._executor_manager = TrainingExecutorLifecycleManager(executor_config)
            success = await self._executor_manager.start()
            
            if success:
                logger.info(f"训练执行器 {self._executor_id} 初始化并注册成功")
            else:
                logger.warning(f"训练执行器 {self._executor_id} 初始化失败")
                
        except Exception as e:
            logger.error(f"初始化训练执行器失败: {e}")
    
    async def _shutdown_training_executor(self):
        """关闭训练执行器"""
        if self._executor_manager:
            try:
                await self._executor_manager.stop()
                logger.info(f"训练执行器 {self._executor_id} 已关闭")
            except Exception as e:
                logger.error(f"关闭训练执行器失败: {e}")
    
    async def submit_training_task(
        self,
        model_type: str,
        config: Dict[str, Any],
        callback: Optional[Callable] = None
    ) -> TrainingTask:
        """
        提交异步训练任务
        
        Args:
            model_type: 模型类型
            config: 训练配置
            callback: 完成回调函数
            
        Returns:
            训练任务对象
        """
        task_id = f"async_train_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(asyncio.current_task())}"
        
        task = TrainingTask(
            task_id=task_id,
            model_type=model_type,
            config=config,
            status=TrainingStatus.QUEUED,
            callback=callback
        )
        
        # 添加到队列
        await self._task_queue.put(task)
        
        logger.info(f"训练任务 {task_id} 已提交到队列")
        return task
    
    async def _process_task_queue(self):
        """处理任务队列"""
        while not self._shutdown:
            try:
                # 获取任务
                task = await self._task_queue.get()
                
                if task.status == TrainingStatus.CANCELLED:
                    continue
                
                # 使用信号量限制并发
                async with self._semaphore:
                    await self._execute_training_task(task)
                    
            except Exception as e:
                logger.error(f"处理训练任务时出错: {e}")
    
    async def _execute_training_task(self, task: TrainingTask):
        """执行训练任务"""
        task.status = TrainingStatus.RUNNING
        task.started_at = datetime.now()
        self._running_tasks[task.task_id] = task
        
        logger.info(f"开始执行训练任务 {task.task_id}")
        
        try:
            # 获取训练配置
            model_type = task.model_type
            config = task.config
            
            # 检查是否使用特征数据
            data_source = config.get("data_source", "historical")
            feature_task_id = config.get("feature_task_id")
            
            if data_source == "features" and feature_task_id:
                logger.info(f"任务 {task.task_id} 使用特征工程任务 {feature_task_id} 的数据")
                
                # 从缓存获取特征数据
                from src.ml.engine.feature_cache_manager import get_feature_cache_manager
                cache_manager = get_feature_cache_manager()
                
                cached_data = cache_manager.get_cached_features(feature_task_id)
                
                if cached_data:
                    logger.info(f"任务 {task.task_id} 使用缓存的特征数据")
                    features = cached_data["features"]
                    target = cached_data.get("target")
                else:
                    # 从特征工程服务获取数据
                    from src.gateway.web.feature_engineering_service import get_feature_data_for_training
                    feature_result = get_feature_data_for_training(feature_task_id)
                    
                    if "error" in feature_result:
                        raise Exception(f"获取特征数据失败: {feature_result['error']}")
                    
                    features = feature_result["features"]
                    target = feature_result.get("target")
                    
                    # 缓存特征数据
                    cache_manager.cache_features(
                        feature_task_id,
                        features,
                        feature_result.get("metadata")
                    )
                
                # 更新配置，添加特征数据
                config["X"] = features
                config["y"] = target
            
            # 执行训练
            from src.ml.core.ml_core import get_ml_core
            ml_core = get_ml_core()
            
            # 模拟训练进度更新
            for progress in range(0, 101, 10):
                task.progress = progress
                await asyncio.sleep(1)  # 模拟训练时间
                
                # 每10%进度输出日志
                if progress % 20 == 0:
                    logger.info(f"任务 {task.task_id} 训练进度: {progress}%")
            
            # 实际训练
            result = ml_core.train_model(
                X=config.get("X"),
                y=config.get("y"),
                model_type=model_type,
                **{k: v for k, v in config.items() if k not in ["X", "y", "data_source", "feature_task_id"]}
            )
            
            task.result = result
            task.status = TrainingStatus.COMPLETED
            task.progress = 100.0
            task.completed_at = datetime.now()
            
            logger.info(f"训练任务 {task.task_id} 完成")
            
            # 调用回调函数
            if task.callback:
                try:
                    if asyncio.iscoroutinefunction(task.callback):
                        await task.callback(task)
                    else:
                        task.callback(task)
                except Exception as e:
                    logger.error(f"回调函数执行失败: {e}")
            
        except Exception as e:
            logger.error(f"训练任务 {task.task_id} 失败: {e}")
            task.status = TrainingStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            # 自动重试
            if self.config.auto_retry_failed and config.get("retry_count", 0) < self.config.max_retry_count:
                config["retry_count"] = config.get("retry_count", 0) + 1
                logger.info(f"任务 {task.task_id} 将在 {config['retry_count']} 秒后重试")
                await asyncio.sleep(config["retry_count"])
                await self.submit_training_task(model_type, config, callback)
        
        finally:
            # 从运行中任务移除
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]
            
            # 添加到已完成任务
            self._completed_tasks[task.task_id] = task
    
    def get_task_status(self, task_id: str) -> Optional[TrainingStatus]:
        """获取任务状态"""
        if task_id in self._running_tasks:
            return self._running_tasks[task_id].status
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id].status
        return None
    
    def get_task_progress(self, task_id: str) -> Optional[float]:
        """获取任务进度"""
        if task_id in self._running_tasks:
            return self._running_tasks[task_id].progress
        if task_id in self._completed_tasks:
            return 100.0
        return None
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id].result
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        # 检查是否在队列中
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            task.status = TrainingStatus.CANCELLED
            logger.info(f"任务 {task_id} 已取消")
            return True
        return False
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        return {
            "queue_size": self._task_queue.qsize(),
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            "max_concurrent": self.config.max_concurrent_tasks
        }
    
    async def shutdown(self):
        """关闭管理器"""
        self._shutdown = True
        
        # 取消所有运行中的任务
        for task in list(self._running_tasks.values()):
            task.status = TrainingStatus.CANCELLED
        
        # 等待工作线程结束
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AsyncTrainingManager 已关闭")


# 全局异步训练管理器实例（单例模式）
_global_async_manager: Optional[AsyncTrainingManager] = None


async def get_async_training_manager(config: Optional[AsyncTrainingConfig] = None) -> AsyncTrainingManager:
    """
    获取全局异步训练管理器实例
    
    Args:
        config: 异步训练配置
        
    Returns:
        异步训练管理器实例
    """
    global _global_async_manager
    
    if _global_async_manager is None:
        _global_async_manager = AsyncTrainingManager(config)
    
    return _global_async_manager


async def close_async_training_manager():
    """关闭全局异步训练管理器实例"""
    global _global_async_manager
    
    if _global_async_manager:
        await _global_async_manager.shutdown()
        _global_async_manager = None
