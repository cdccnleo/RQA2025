"""
流程执行器组件

职责:
- 执行业务流程各阶段
- 协调阶段间的数据流转
- 处理流程异常和重试
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessStatus(Enum):
    """流程状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """执行结果"""
    process_id: str
    status: ProcessStatus
    stages_completed: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ProcessExecutor:
    """
    流程执行器组件

    负责执行和协调业务流程的各个阶段
    支持重试、超时控制和并发执行
    """

    def __init__(self, config: 'ExecutionConfig'):
        """
        初始化流程执行器

        Args:
            config: 执行配置对象
        """
        self.config = config
        self._active_processes: Dict[str, Any] = {}
        self._execution_queue: List[Dict[str, Any]] = []
        self._completed_count = 0
        self._failed_count = 0
        self._circuit_breaker_active = False
        self._consecutive_failures = 0

        logger.info(f"流程执行器初始化完成 (并发: {config.max_concurrent_processes})")

    async def execute_process(self, context: Any,
                             decision_engine: Any) -> ExecutionResult:
        """
        执行完整流程

        Args:
            context: 流程上下文
            decision_engine: 决策引擎实例

        Returns:
            ExecutionResult: 执行结果
        """
        process_id = getattr(context, 'process_id', 'unknown')
        start_time = datetime.now()

        # 检查断路器状态
        if self._circuit_breaker_active:
            logger.warning(f"断路器激活，拒绝执行流程 {process_id}")
            return ExecutionResult(
                process_id=process_id,
                status=ProcessStatus.FAILED,
                errors=["Circuit breaker is active"]
            )

        # 检查并发限制
        if len(self._active_processes) >= self.config.max_concurrent_processes:
            logger.warning(f"达到最大并发数，流程 {process_id} 进入队列")
            return await self._queue_process(context, decision_engine)

        # 注册活跃流程
        self._active_processes[process_id] = {
            'context': context,
            'start_time': start_time,
            'status': ProcessStatus.RUNNING
        }

        try:
            # 执行流程（带超时控制）
            result = await asyncio.wait_for(
                self._execute_with_retry(context, decision_engine),
                timeout=self.config.execution_timeout
            )

            # 更新状态
            result.status = ProcessStatus.COMPLETED
            result.execution_time = (datetime.now() - start_time).total_seconds()

            # 记录成功
            self._completed_count += 1
            self._consecutive_failures = 0  # 重置失败计数

            logger.info(f"流程执行成功 {process_id}, 耗时: {result.execution_time:.2f}秒")

        except asyncio.TimeoutError:
            logger.error(f"流程执行超时 {process_id}")
            result = ExecutionResult(
                process_id=process_id,
                status=ProcessStatus.TIMEOUT,
                errors=["Execution timeout"]
            )
            self._handle_failure()

        except Exception as e:
            logger.error(f"流程执行异常 {process_id}: {e}")
            result = ExecutionResult(
                process_id=process_id,
                status=ProcessStatus.FAILED,
                errors=[str(e)]
            )
            self._handle_failure()

        finally:
            # 清理活跃流程
            if process_id in self._active_processes:
                del self._active_processes[process_id]

        return result

    async def execute_stage(self, context: Any, stage: str,
                           decision_engine: Any) -> Dict[str, Any]:
        """
        执行单个阶段

        Args:
            context: 流程上下文
            stage: 阶段名称
            decision_engine: 决策引擎

        Returns:
            Dict: 阶段执行结果
        """
        logger.debug(f"执行流程阶段: {stage}")

        try:
            # 调用决策引擎
            decision = await decision_engine.make_market_decision({}, context)

            return {
                'stage': stage,
                'status': 'completed',
                'decision': decision,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"阶段执行失败 {stage}: {e}")
            return {
                'stage': stage,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_active_processes(self) -> Dict[str, Any]:
        """获取活跃流程"""
        return {
            'count': len(self._active_processes),
            'processes': list(self._active_processes.keys()),
            'max_concurrent': self.config.max_concurrent_processes
        }

    async def cancel_process(self, process_id: str) -> bool:
        """取消流程执行"""
        if process_id in self._active_processes:
            del self._active_processes[process_id]
            logger.info(f"流程已取消: {process_id}")
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """获取执行器状态"""
        return {
            'active_processes': len(self._active_processes),
            'completed_count': self._completed_count,
            'failed_count': self._failed_count,
            'circuit_breaker_active': self._circuit_breaker_active,
            'queue_size': len(self._execution_queue),
            'config': {
                'max_concurrent': self.config.max_concurrent_processes,
                'timeout': self.config.execution_timeout,
                'retry_enabled': self.config.enable_retry
            }
        }

    # 私有辅助方法
    async def _execute_with_retry(self, context: Any,
                                  decision_engine: Any) -> ExecutionResult:
        """带重试的执行"""
        process_id = getattr(context, 'process_id', 'unknown')
        last_error = None

        for attempt in range(self.config.max_retries if self.config.enable_retry else 1):
            try:
                result = await self._execute_internal(context, decision_engine)
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"流程执行失败 {process_id}, 尝试 {attempt + 1}/{self.config.max_retries}: {e}")

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)

        # 所有重试都失败
        raise last_error if last_error else Exception("Unknown error")

    async def _execute_internal(self, context: Any,
                               decision_engine: Any) -> ExecutionResult:
        """内部执行逻辑"""
        process_id = getattr(context, 'process_id', 'unknown')

        # 简化实现：执行各阶段
        stages_completed = []

        for stage in ['stage1', 'stage2', 'stage3']:
            stage_result = await self.execute_stage(context, stage, decision_engine)
            if stage_result.get('status') == 'completed':
                stages_completed.append(stage)

        return ExecutionResult(
            process_id=process_id,
            status=ProcessStatus.RUNNING,  # 将在外层更新
            stages_completed=stages_completed,
            metrics={'stages_count': len(stages_completed)}
        )

    async def _queue_process(self, context: Any, decision_engine: Any) -> ExecutionResult:
        """将流程加入队列"""
        process_id = getattr(context, 'process_id', 'unknown')

        self._execution_queue.append({
            'context': context,
            'decision_engine': decision_engine,
            'queued_at': datetime.now()
        })

        logger.info(f"流程已加入队列: {process_id}")

        return ExecutionResult(
            process_id=process_id,
            status=ProcessStatus.PENDING,
            metadata={'queue_position': len(self._execution_queue)}
        )

    def _handle_failure(self):
        """处理失败"""
        self._failed_count += 1
        self._consecutive_failures += 1

        # 检查是否需要触发断路器
        if (self.config.enable_circuit_breaker and
            self._consecutive_failures >= self.config.circuit_breaker_threshold):
            self._circuit_breaker_active = True
            logger.error(f"断路器已激活 (连续失败: {self._consecutive_failures})")


# 配置类会通过参数传入，无需导入
