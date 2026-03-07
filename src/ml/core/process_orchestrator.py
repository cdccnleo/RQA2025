#!/usr/bin/env python3
"""
RQA2025 ML业务流程编排器

提供机器学习业务流程的编排和管理能力，
支持模型训练、预测、评估等流程的自动化执行和管理。
基于业务流程驱动架构，实现ML流程的状态管理和监控。
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Protocol
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue

try:  # pragma: no cover
    from src.core.integration import get_models_adapter as _get_models_adapter
except ImportError:  # pragma: no cover
    import logging

    class _FallbackModelsAdapter:
        def get_models_logger(self):
            return logging.getLogger(__name__)

    def _get_models_adapter():
        return _FallbackModelsAdapter()

get_models_adapter = _get_models_adapter

from .error_handling import handle_ml_error

# 获取统一基础设施集成层的模型层适配器
try:
    models_adapter = _get_models_adapter()
    logger = models_adapter.get_models_logger()
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)


class MLProcessType(Enum):

    """ML业务流程类型"""
    MODEL_TRAINING = "model_training"
    MODEL_PREDICTION = "model_prediction"
    MODEL_EVALUATION = "model_evaluation"
    FEATURE_ENGINEERING = "feature_engineering"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    MODEL_DEPLOYMENT = "model_deployment"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_INFERENCE = "real_time_inference"
    A_B_TESTING = "ab_testing"
    MODEL_MONITORING = "model_monitoring"


class ProcessPriority(Enum):

    """流程优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ProcessStatus(Enum):

    """流程状态"""
    CREATED = "created"
    QUEUED = "queued"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ProcessStep:

    """流程步骤"""
    step_id: str
    step_name: str
    step_type: str
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: ProcessStatus = ProcessStatus.CREATED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLProcess:

    """ML业务流程"""
    process_id: str
    process_type: MLProcessType
    process_name: str
    priority: ProcessPriority = ProcessPriority.NORMAL
    status: ProcessStatus = ProcessStatus.CREATED
    steps: Dict[str, ProcessStep] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: Optional[int] = None
    progress: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    callbacks: Dict[str, List[Callable]] = field(default_factory=dict)


class StepExecutor(Protocol):

    """步骤执行器协议"""

    def execute(self, step: ProcessStep, context: Dict[str, Any]) -> Any:
        """执行步骤"""
        ...  # pragma: no cover

    def validate(self, step: ProcessStep) -> bool:
        """验证步骤配置"""
        ...  # pragma: no cover

    def get_dependencies(self, step: ProcessStep) -> List[str]:
        """获取步骤依赖"""
        ...  # pragma: no cover


class MLProcessOrchestrator:

    """ML业务流程编排器"""

    def __init__(self, max_workers: int = 4, queue_size: int = 100):

        self.max_workers = max_workers
        self.queue_size = queue_size

        # 流程管理
        self.active_processes: Dict[str, MLProcess] = {}
        self.completed_processes: Dict[str, MLProcess] = {}
        self.process_queue: PriorityQueue = PriorityQueue(maxsize=queue_size)

        # 执行器管理
        self.step_executors: Dict[str, StepExecutor] = {}
        self.executor_pool = ThreadPoolExecutor(max_workers=max_workers)

        # 监控和统计
        self.stats = {
            'total_processes': 0,
            'completed_processes': 0,
            'failed_processes': 0,
            'avg_process_time': 0.0,
            'active_workers': 0
        }

        # 控制标志
        self.running = False
        self.shutdown_event = threading.Event()

        # 注册默认步骤执行器
        self._register_default_executors()

        logger.info(f"ML业务流程编排器初始化完成，最大工作线程数: {max_workers}")

    def _register_default_executors(self):
        """注册默认步骤执行器"""
        # 这里将在后续实现中添加具体的执行器

    def start(self):
        """启动编排器"""
        if self.running:
            logger.warning("ML业务流程编排器已在运行中")
            return

        self.running = True
        self.shutdown_event.clear()

        for i in range(self.max_workers):
            thread = threading.Thread(
                target=self._process_worker,
                name=f"MLProcessWorker-{i + 1}",
                daemon=True
            )
            thread.start()

        logger.info(f"ML业务流程编排器已启动，工作线程数: {self.max_workers}")

    def stop(self, timeout: int = 30):
        """停止编排器"""
        if not self.running:
            logger.info("ML业务流程编排器未运行")
            return

        logger.info("正在停止ML业务流程编排器...")
        self.running = False
        self.shutdown_event.set()

        self.shutdown_event.wait(timeout)
        self.executor_pool.shutdown(wait=True)
        logger.info("ML业务流程编排器已停止")

    def submit_process(self, process: MLProcess) -> str:
        """提交业务流程"""
        if not self.running:
            raise RuntimeError("ML业务流程编排器未启动")

        if not process.process_id:
            process.process_id = f"{process.process_type.value}_{int(time.time() * 1000)}"

        process.status = ProcessStatus.QUEUED
        self.active_processes[process.process_id] = process
        self.stats['total_processes'] += 1
        self.process_queue.put((-process.priority.value, process.process_id))

        logger.info(f"已提交ML业务流程: {process.process_name} (ID: {process.process_id})")
        return process.process_id

    def _process_worker(self):
        """流程处理工作线程"""
        logger.info(f"ML业务流程处理线程 {threading.current_thread().name} 已启动")

        while not self.shutdown_event.is_set():
            try:
                # 获取待处理流程
                priority, process_id = self.process_queue.get(timeout=1)
                if process_id in self.active_processes:
                    process = self.active_processes[process_id]

                    # 执行流程
                    self._execute_process(process)

                    # 标记队列任务完成
                    self.process_queue.task_done()

            except Exception as e:
                if not self.shutdown_event.is_set():
                    logger.error(f"流程处理线程异常: {e}")

                    logger.info(f"ML业务流程处理线程 {threading.current_thread().name} 已停止")

    def _execute_process(self, process: MLProcess):
        """执行业务流程"""
        try:
            # 更新流程状态
            process.status = ProcessStatus.INITIALIZING
            process.started_at = datetime.now()

            # 调用开始回调
            self._trigger_callbacks(process, 'on_start')

            # 执行流程步骤
            success = self._execute_process_steps(process)

            # 更新流程状态
            if success:
                process.status = ProcessStatus.COMPLETED
                process.completed_at = datetime.now()
                self.stats['completed_processes'] += 1
                self._trigger_callbacks(process, 'on_complete')
            else:
                process.status = ProcessStatus.FAILED
                self.stats['failed_processes'] += 1
                self._trigger_callbacks(process, 'on_fail')

                # 移动到完成队列
                self.completed_processes[process.process_id] = process
                del self.active_processes[process.process_id]

                # 计算执行时间
            if process.started_at and process.completed_at:
                execution_time = (process.completed_at - process.started_at).total_seconds()
                # 更新平均执行时间
                total_completed = self.stats['completed_processes']
                self.stats['avg_process_time'] = (
                    (self.stats['avg_process_time'] * (total_completed - 1)) + execution_time
                ) / total_completed

                logger.info(f"ML业务流程执行完成: {process.process_name} (状态: {process.status.value})")

        except Exception as e:
            # 使用统一的错误处理机制
            error_context = {
                'process_id': process.process_id,
                'process_name': process.process_name,
                'process_type': process.process_type.value
            }
            handle_ml_error(e, error_context)

            logger.error(f"ML业务流程执行异常: {process.process_name}, 错误: {e}")
            process.status = ProcessStatus.FAILED
            process.metadata['error'] = str(e)

        self.completed_processes[process.process_id] = process
        self.active_processes.pop(process.process_id, None)

    def _execute_process_steps(self, process: MLProcess) -> bool:
        """执行流程步骤"""
        step_graph = self._build_step_graph(process.steps)

        executed_steps = set()
        remaining_steps = set(process.steps.keys())

        while remaining_steps:
            progressed = False
            for step_id in list(remaining_steps):
                step = process.steps[step_id]
                if not self._can_execute_step(step, executed_steps):
                    continue
                try:
                    result = self._execute_step(step, process)
                    executed_steps.add(step_id)
                    remaining_steps.remove(step_id)
                    step.result = result
                    step.status = ProcessStatus.COMPLETED
                    process.progress = len(executed_steps) / len(process.steps)
                    progressed = True
                except Exception as e:
                    logger.error(f"ML业务流程步骤执行失败: {step_id}, 错误: {e}")
                    step.status = ProcessStatus.FAILED
                    step.error = str(e)
                    return False

            if not progressed:
                logger.error(f"ML业务流程存在无法执行的步骤: {process.process_name}")
                return False

        return True

    def _build_step_graph(self, steps: Dict[str, ProcessStep]) -> Dict[str, List[str]]:
        """构建步骤依赖图"""
        graph = {}

        for step_id, step in steps.items():
            graph[step_id] = step.dependencies.copy()

        return graph

    def _can_execute_step(self, step: ProcessStep, executed_steps: set) -> bool:
        """检查步骤是否可以执行"""
        return all(dep in executed_steps for dep in step.dependencies)

    def _execute_step(self, step: ProcessStep, process: MLProcess) -> Any:
        """执行单个步骤"""
        step.status = ProcessStatus.RUNNING
        step.start_time = datetime.now()

        executor = self.step_executors.get(step.step_type)
        if executor is None:
            raise ValueError(f"未找到步骤执行器: {step.step_type}")

        if hasattr(executor, "validate") and not executor.validate(step):
            raise ValueError(f"步骤配置验证失败: {step.step_id}")

        context = {
            'process': process,
            'step': step,
            'config': step.config,
            'metadata': process.metadata
        }

        result = executor.execute(step, context)

        step.end_time = datetime.now()
        step.status = ProcessStatus.COMPLETED

        if step.start_time and step.end_time:
            execution_time = (step.end_time - step.start_time).total_seconds()
            step.metrics['execution_time'] = execution_time

        return result

    def register_step_executor(self, step_type: str, executor: StepExecutor):
        """注册步骤执行器"""
        self.step_executors[step_type] = executor
        logger.info(f"已注册ML业务流程步骤执行器: {step_type}")

    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """获取流程状态"""
        process = (
            self.active_processes.get(process_id)
            or self.completed_processes.get(process_id)
        )

        if not process:
            return None

        return {
            'process_id': process.process_id,
            'process_name': process.process_name,
            'process_type': process.process_type.value,
            'status': process.status.value,
            'priority': process.priority.value,
            'progress': process.progress,
            'created_at': process.created_at.isoformat(),
            'started_at': process.started_at.isoformat() if process.started_at else None,
            'completed_at': process.completed_at.isoformat() if process.completed_at else None,
            'step_count': len(process.steps),
            'completed_steps': sum(1 for s in process.steps.values() if s.status == ProcessStatus.COMPLETED),
            'failed_steps': sum(1 for s in process.steps.values() if s.status == ProcessStatus.FAILED),
            'metrics': process.metrics
        }

    def cancel_process(self, process_id: str) -> bool:
        """取消流程"""
        if process_id in self.active_processes:
            process = self.active_processes[process_id]
            process.status = ProcessStatus.CANCELLED
            self.completed_processes[process_id] = process
            del self.active_processes[process_id]
            logger.info(f"已取消ML业务流程: {process.process_name}")
            return True

        return False

    def pause_process(self, process_id: str) -> bool:
        """暂停流程"""
        if process_id in self.active_processes:
            process = self.active_processes[process_id]
            if process.status == ProcessStatus.RUNNING:
                process.status = ProcessStatus.PAUSED
                logger.info(f"已暂停ML业务流程: {process.process_name}")
                return True
        return False

    def resume_process(self, process_id: str) -> bool:
        """恢复流程"""
        if process_id in self.active_processes:
            process = self.active_processes[process_id]
            if process.status == ProcessStatus.PAUSED:
                process.status = ProcessStatus.RUNNING
                logger.info(f"已恢复ML业务流程: {process.process_name}")
                return True
        return False

    def add_process_callback(self, process_id: str, event: str, callback: Callable):
        """添加流程回调"""
        if process_id in self.active_processes:
            process = self.active_processes[process_id]
            if event not in process.callbacks:
                process.callbacks[event] = []
            process.callbacks[event].append(callback)

    def _trigger_callbacks(self, process: MLProcess, event: str):
        """触发回调"""
        if event in process.callbacks:
            for callback in process.callbacks[event]:
                try:
                    callback(process)
                except Exception as e:
                    logger.error(f"ML业务流程回调执行失败: {event}, 错误: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取编排器统计信息"""
        return {
            'running': self.running,
            'active_processes': len(self.active_processes),
            'completed_processes': len(self.completed_processes),
            'queue_size': self.process_queue.qsize(),
            'max_workers': self.max_workers,
            'stats': self.stats.copy(),
            'executor_pool_info': {
                'active_threads': len([t for t in threading.enumerate()
                                       if t.name.startswith('MLProcessWorker') and t.is_alive()]),
                'total_threads': self.max_workers
            }
        }


# 全局ML业务流程编排器实例
_ml_orchestrator = MLProcessOrchestrator()


def get_ml_process_orchestrator() -> MLProcessOrchestrator:
    """获取ML业务流程编排器实例"""
    return _ml_orchestrator


def create_ml_process(process_type: MLProcessType,


                      process_name: str,
                      steps: Dict[str, ProcessStep],
                      config: Optional[Dict[str, Any]] = None,
                      priority: ProcessPriority = ProcessPriority.NORMAL,
                      timeout: Optional[int] = None) -> MLProcess:
    """创建ML业务流程"""
    return MLProcess(
        process_id="",
        process_type=process_type,
        process_name=process_name,
        steps=steps,
        config=config or {},
        priority=priority,
        timeout=timeout
    )


def submit_ml_process(process: MLProcess) -> str:
    """提交ML业务流程"""
    return _ml_orchestrator.submit_process(process)


def get_ml_process_status(process_id: str) -> Optional[Dict[str, Any]]:
    """获取ML业务流程状态"""
    return _ml_orchestrator.get_process_status(process_id)


def cancel_ml_process(process_id: str) -> bool:
    """取消ML业务流程"""
    return _ml_orchestrator.cancel_process(process_id)


def get_ml_orchestrator_stats() -> Dict[str, Any]:
    """获取ML编排器统计信息"""
    return _ml_orchestrator.get_statistics()


# 便捷函数
__all__ = [
    # 核心类
    'MLProcessOrchestrator',
    'MLProcess',
    'ProcessStep',
    'StepExecutor',

    # 枚举
    'MLProcessType',
    'ProcessPriority',
    'ProcessStatus',

    # 全局函数
    'get_ml_process_orchestrator',
    'create_ml_process',
    'submit_ml_process',
    'get_ml_process_status',
    'cancel_ml_process',
    'get_ml_orchestrator_stats'
]
