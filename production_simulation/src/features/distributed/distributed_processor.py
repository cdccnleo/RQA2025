import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式特征处理器

提供分布式特征处理的核心功能，包括任务分发、负载均衡、结果聚合等。
"""

import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time
# 使用统一基础设施集成层
try:
    from src.core.integration import get_features_layer_adapter
    _features_adapter = get_features_layer_adapter()
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    # 降级到直接导入
    logger = logging.getLogger(__name__)

from src.features.core.feature_engineer import FeatureEngineer
from src.features.core.config import FeatureProcessingConfig
from .task_scheduler import FeatureTaskScheduler, FeatureTask, TaskStatus, TaskPriority
from .worker_manager import FeatureWorkerManager, WorkerInfo


class ProcessingStrategy(Enum):

    """处理策略枚举"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_WORKER = "fastest_worker"
    RANDOM = "random"
    WEIGHTED = "weighted"


class LoadBalancingStrategy(Enum):

    """负载均衡策略枚举"""
    SIMPLE = "simple"
    ADAPTIVE = "adaptive"
    INTELLIGENT = "intelligent"


@dataclass
class ProcessingResult:

    """处理结果数据类"""
    task_id: str
    worker_id: str
    result: Any
    processing_time: float
    memory_usage: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadBalancerConfig:

    """负载均衡器配置"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.SIMPLE
    max_retries: int = 3
    timeout: float = 30.0
    health_check_interval: float = 5.0
    worker_selection_timeout: float = 10.0


class FeatureLoadBalancer:

    """特征负载均衡器"""

    def __init__(self, config: LoadBalancerConfig):

        self.config = config
        self._worker_stats: Dict[str, Dict[str, Any]] = {}
        self._last_health_check = 0.0

    def select_worker(self, available_workers: List[WorkerInfo],


                      task_priority: TaskPriority = TaskPriority.NORMAL) -> Optional[str]:
        """选择最佳工作节点"""
        if not available_workers:
            return None

        if self.config.strategy == LoadBalancingStrategy.SIMPLE:
            return self._simple_selection(available_workers)
        elif self.config.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_selection(available_workers, task_priority)
        elif self.config.strategy == LoadBalancingStrategy.INTELLIGENT:
            return self._intelligent_selection(available_workers, task_priority)
        else:
            return self._simple_selection(available_workers)

    def _simple_selection(self, workers: List[WorkerInfo]) -> Optional[str]:
        """简单轮询选择"""
        if not workers:
            return None
        # 选择负载最低的工作节点
        return min(workers, key=lambda w: w.current_load).worker_id

    def _adaptive_selection(self, workers: List[WorkerInfo],


                            task_priority: TaskPriority) -> Optional[str]:
        """自适应选择"""
        if not workers:
            return None

        # 根据任务优先级选择策略
        if task_priority == TaskPriority.HIGH:
            # 高优先级任务选择性能最好的节点
            return max(workers, key=lambda w: w.performance_score).worker_id
        elif task_priority == TaskPriority.LOW:
            # 低优先级任务选择负载最低的节点
            return min(workers, key=lambda w: w.current_load).worker_id
        else:
            # 中等优先级任务平衡性能和负载
            return max(workers, key=lambda w: w.performance_score / (w.current_load + 1)).worker_id

    def _intelligent_selection(self, workers: List[WorkerInfo],


                               task_priority: TaskPriority) -> Optional[str]:
        """智能选择"""
        if not workers:
            return None

        # 综合考虑多个因素
        scores = []
        for worker in workers:
            score = self._calculate_worker_score(worker, task_priority)
            scores.append((worker.worker_id, score))

        # 返回得分最高的节点
        return max(scores, key=lambda x: x[1])[0]

    def _calculate_worker_score(self, worker: WorkerInfo,


                                task_priority: TaskPriority) -> float:
        """计算工作节点得分"""
        base_score = worker.performance_score

        # 根据优先级调整权重
        if task_priority == TaskPriority.HIGH:
            performance_weight = 0.7
            load_weight = 0.3
        elif task_priority == TaskPriority.LOW:
            performance_weight = 0.3
            load_weight = 0.7
        else:
            performance_weight = 0.5
            load_weight = 0.5

        # 计算综合得分
        load_factor = 1.0 / (worker.current_load + 1)
        score = (performance_weight * base_score
                 + load_weight * load_factor)

        return score

    def update_worker_stats(self, worker_id: str, stats: Dict[str, Any]):
        """更新工作节点统计信息"""
        self._worker_stats[worker_id] = stats

    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取工作节点统计信息"""
        return self._worker_stats.copy()


class DistributedFeatureProcessor:

    """分布式特征处理器"""

    def __init__(self,


                 scheduler: FeatureTaskScheduler,
                 worker_manager: FeatureWorkerManager,
                 load_balancer: FeatureLoadBalancer,
                 max_workers: int = 4,
                 executor_type: str = "thread"):
        self.scheduler = scheduler
        self.worker_manager = worker_manager
        self.load_balancer = load_balancer
        self.max_workers = max_workers
        self.executor_type = executor_type

        # 初始化logger
        self.logger = logger

        # 创建执行器
        if executor_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        elif executor_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            raise ValueError(f"Unsupported executor type: {executor_type}")

        self._processing_results: List[ProcessingResult] = []
        self._is_running = False
        self._processing_thread = None

    def start(self):
        """启动分布式处理器"""
        if self._is_running:
            self.logger.warning("Distributed processor is already running")
            return

        self._is_running = True
        self.scheduler.start()
        self.worker_manager.start_monitoring()

        # 启动任务处理线程
        self._processing_thread = threading.Thread(target=self._process_tasks_loop, daemon=True)
        self._processing_thread.start()

        self.logger.info("Distributed feature processor started")

    def stop(self):
        """停止分布式处理器"""
        if not self._is_running:
            return

        self._is_running = False
        self.scheduler.stop()
        self.worker_manager.stop_monitoring()
        self.executor.shutdown(wait=True)
        self.logger.info("Distributed feature processor stopped")

    def process_features(self,


                         data: Any,
                         config: Any,  # FeatureConfig is removed, so Any is used for now
                         processing_config: FeatureProcessingConfig,
                         priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """提交特征处理任务"""
        # 提交到调度器，使用调度器返回的任务ID
        task_id = self.scheduler.submit_task(
            task_type="feature_processing",
            data=data,
            priority=priority,
            metadata={
                "config": config.to_dict() if hasattr(config, 'to_dict') else str(config),
                "processing_config": processing_config.to_dict() if hasattr(processing_config, 'to_dict') else str(processing_config)
            }
        )
        self.logger.info(f"Submitted feature processing task: {task_id}")

        return task_id

    def process_features_batch(self,


                               # FeatureConfig is removed, so Any is used for now
                               data_batch: List[Tuple[Any, Any, FeatureProcessingConfig]],
                               priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
        """批量提交特征处理任务"""
        task_ids = []

        for data, config, processing_config in data_batch:
            task_id = self.process_features(data, config, processing_config, priority)
            task_ids.append(task_id)

        self.logger.info(f"Submitted {len(task_ids)} batch processing tasks")
        return task_ids

    def get_task_result(self, task_id: str) -> Optional[ProcessingResult]:
        """获取任务结果"""
        task_status = self.scheduler.get_task_status(task_id)
        if not task_status or task_status != TaskStatus.COMPLETED:
            return None

        # 查找对应的处理结果
        for result in self._processing_results:
            if result.task_id == task_id:
                return result

        return None

    def get_batch_results(self, task_ids: List[str]) -> List[Optional[ProcessingResult]]:
        """获取批量任务结果"""
        results = []
        for task_id in task_ids:
            result = self.get_task_result(task_id)
            results.append(result)
        return results

    def wait_for_task(self, task_id: str, timeout: float = 30.0) -> Optional[ProcessingResult]:
        """等待任务完成"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            result = self.get_task_result(task_id)
            if result is not None:
                return result
            time.sleep(0.1)

        self.logger.warning(f"Task {task_id} timeout after {timeout}s")
        return None

    def wait_for_batch(self, task_ids: List[str], timeout: float = 60.0) -> List[Optional[ProcessingResult]]:
        """等待批量任务完成"""
        start_time = time.time()
        results = [None] * len(task_ids)
        completed = set()

        while time.time() - start_time < timeout and len(completed) < len(task_ids):
            for i, task_id in enumerate(task_ids):
                if task_id not in completed:
                    result = self.get_task_result(task_id)
                    if result is not None:
                        results[i] = result
                        completed.add(task_id)

            if len(completed) < len(task_ids):
                time.sleep(0.1)

        if len(completed) < len(task_ids):
            self.logger.warning(f"Batch timeout: {len(completed)}/{len(task_ids)} tasks completed")

        return results

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return self.scheduler.cancel_task(task_id)

    def cancel_batch(self, task_ids: List[str]) -> List[bool]:
        """取消批量任务"""
        results = []
        for task_id in task_ids:
            result = self.cancel_task(task_id)
            results.append(result)
        return results

    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        total_tasks = len(self._processing_results)
        successful_tasks = sum(1 for r in self._processing_results if r.success)
        failed_tasks = total_tasks - successful_tasks

        if total_tasks > 0:
            avg_processing_time = sum(
                r.processing_time for r in self._processing_results) / total_tasks
            avg_memory_usage = sum(r.memory_usage for r in self._processing_results) / total_tasks
        else:
            avg_processing_time = 0.0
            avg_memory_usage = 0.0

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "avg_processing_time": avg_processing_time,
            "avg_memory_usage": avg_memory_usage,
            "scheduler_stats": self.scheduler.get_scheduler_stats(),
            "worker_stats": self.worker_manager.get_worker_stats(),
            "load_balancer_stats": self.load_balancer.get_worker_stats()
        }

    def clear_history(self):
        """清除历史记录"""
        self._processing_results.clear()
        self.scheduler.clear_completed_tasks()
        self.logger.info("Cleared processing history")

    def _process_task_worker(self, task: FeatureTask):
        """工作节点处理任务"""
        start_time = time.time()

        # 安全处理task_id，避免Mock对象问题
        try:
            if hasattr(task.task_id, '__getitem__') and callable(getattr(task.task_id, '__getitem__', None)):
                # 如果是字符串或类似字符串的对象，可以切片
                worker_id = f"worker_{str(task.task_id)[:8]}"
            else:
                # 如果是Mock对象或其他类型，使用字符串转换
                worker_id = f"worker_{str(task.task_id)}"
        except (TypeError, IndexError):
            # 如果切片失败，使用完整字符串
            worker_id = f"worker_{str(task.task_id)}"

        try:
            # 从元数据中提取配置
            config_dict = task.metadata.get("config", {})
            processing_config_dict = task.metadata.get("processing_config", {})

            # 创建特征引擎，使用默认的technical_processor
            from src.features.processors.technical.technical_processor import TechnicalProcessor
            technical_processor = TechnicalProcessor()
            engine = FeatureEngineer(technical_processor=technical_processor)

            # 处理特征
            if isinstance(config_dict, dict) and "features" in config_dict:
                # 使用配置中的特征列表
                features = config_dict.get("features", ["ma", "rsi"])
                params = config_dict.get("parameters", {})

                # 生成技术指标特征
                result = engine.generate_technical_features(
                    stock_data=task.data,
                    indicators=features,
                    params=params
                )
            else:
                # 默认处理
                result = engine.generate_technical_features(
                    stock_data=task.data,
                    indicators=["ma", "rsi"]
                )

            processing_time = time.time() - start_time

            # 模拟内存使用
            memory_usage = processing_time * 100  # MB

            return ProcessingResult(
                task_id=task.task_id,
                worker_id=worker_id,
                result=result,
                processing_time=processing_time,
                memory_usage=memory_usage,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Task {task.task_id} failed: {str(e)}")

            return ProcessingResult(
                task_id=task.task_id,
                worker_id=worker_id,
                result=None,
                processing_time=processing_time,
                memory_usage=0.0,
                success=False,
                error_message=str(e)
            )

    def _submit_to_executor(self, task: FeatureTask):
        """提交任务到执行器"""
        future = self.executor.submit(self._process_task_worker, task)

        def callback(fut):

            try:
                result = fut.result()
                self._processing_results.append(result)

                # 更新任务状态
                if result.success:
                    self.scheduler.complete_task(task.task_id, result.result)
                else:
                    # 标记任务失败
                    task.status = TaskStatus.FAILED

            except Exception as e:
                self.logger.error(f"Executor callback error: {str(e)}")
                task.status = TaskStatus.FAILED

        future.add_done_callback(callback)

    def _process_tasks_loop(self):
        """任务处理循环"""
        while self._is_running:
            try:
                # 获取可用工作节点
                available_workers = self.worker_manager.get_available_workers()

                if not available_workers:
                    # 如果没有可用工作节点，创建一个虚拟工作节点
                    worker_id = f"virtual_worker_{int(time.time())}"
                    self.worker_manager.register_worker(
                        worker_id=worker_id,
                        capabilities={"cpu": 2, "memory": 4096}
                    )
                    available_workers = [worker_id]

                # 为每个可用工作节点分配任务
                for worker_id in available_workers:
                    if not self._is_running:
                        break

                    # 从调度器获取任务
                    task = self.scheduler.get_task(worker_id)
                    if task:
                        # 提交任务到执行器
                        self._submit_to_executor(task)

                # 使用可中断的睡眠机制
                if self._is_running:
                    time.sleep(0.1)  # 短暂休眠避免CPU占用过高

            except Exception as e:
                self.logger.error(f"任务处理循环错误: {e}")
                if self._is_running:
                    time.sleep(1)  # 出错后等待1秒再重试

            # 短暂休眠
            time.sleep(0.1)

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


# 便利函数

def create_distributed_processor(max_workers: int = 4,


                                 executor_type: str = "thread",
                                 load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.SIMPLE) -> DistributedFeatureProcessor:
    """创建分布式特征处理器"""
    # 创建组件
    scheduler = FeatureTaskScheduler()
    worker_manager = FeatureWorkerManager()
    load_balancer_config = LoadBalancerConfig(strategy=load_balancing_strategy)
    load_balancer = FeatureLoadBalancer(load_balancer_config)

    # 创建处理器
    processor = DistributedFeatureProcessor(
        scheduler=scheduler,
        worker_manager=worker_manager,
        load_balancer=load_balancer,
        max_workers=max_workers,
        executor_type=executor_type
    )

    return processor
