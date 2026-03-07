"""实时执行器模块"""

from typing import Dict, Any, Optional, Callable
from enum import Enum
import time
import threading
import queue
import asyncio

from .order_executor import Order, OrderSide, OrderExecutor
from enum import Enum
from typing import Optional


class ExecutionMode(Enum):
    """执行模式枚举"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"


class ExecutionPriority(Enum):

    """执行优先级枚举"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ExecutionState(Enum):

    """执行状态枚举"""
    IDLE = "idle"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RealTimeExecutor:

    """实时执行器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化实时执行器

        Args:
            config: 执行器配置
        """
        self.config = config or {}
        self.order_executor: Optional[OrderExecutor] = None
        self.execution_engine = None

        # 执行队列
        self._execution_queue = queue.PriorityQueue()
        self._running = False
        self._thread = None

        # 执行统计
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_volume': 0.0,
            'total_value': 0.0
        }

        # 回调函数
        self.on_execution_start: Optional[Callable[[str], None]] = None
        self.on_execution_complete: Optional[Callable[[str, bool], None]] = None
        self.on_order_submitted: Optional[Callable[[Order], None]] = None

    def set_order_executor(self, executor: OrderExecutor) -> None:
        """设置订单执行器

        Args:
            executor: 订单执行器
        """
        self.order_executor = executor

    def set_execution_engine(self, engine) -> None:
        """设置执行引擎

        Args:
            engine: 执行引擎
        """
        self.execution_engine = engine

    def start(self) -> bool:
        """启动执行器

        Returns:
            是否成功启动
        """
        if self._running:
            return False

        if not self.order_executor or not self.execution_engine:
            return False

        self._running = True
        self._thread = threading.Thread(target=self._execution_loop)
        self._thread.daemon = True
        self._thread.start()

        return True

    def stop(self) -> bool:
        """停止执行器

        Returns:
            是否成功停止
        """
        if not self._running:
            return False

        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        return True

    def submit_execution(self, symbol: str, side: OrderSide, quantity: float,


                         price: Optional[float] = None, mode: ExecutionMode = ExecutionMode.MARKET,
                         priority: ExecutionPriority = ExecutionPriority.NORMAL,
                         **kwargs) -> str:
        """提交执行任务

        Args:
            symbol: 交易标的
            side: 订单方向
            quantity: 数量
            price: 价格
            mode: 执行模式
            priority: 执行优先级
            **kwargs: 其他参数

        Returns:
            执行ID
        """
        if not self.execution_engine:
            return ""

        execution_id = self.execution_engine.create_execution(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            mode=mode,
            **kwargs
        )

        # 添加到执行队列
        priority_value = {
            ExecutionPriority.LOW: 4,
            ExecutionPriority.NORMAL: 3,
            ExecutionPriority.HIGH: 2,
            ExecutionPriority.URGENT: 1
        }[priority]

        self._execution_queue.put((priority_value, execution_id))

        return execution_id

    def _execution_loop(self) -> None:
        """执行主循环"""
        while self._running:
            try:
                # 从队列获取执行任务
                if not self._execution_queue.empty():
                    priority, execution_id = self._execution_queue.get_nowait()
                    self._execute_task(execution_id)

                time.sleep(0.01)  # 10ms间隔

            except queue.Empty:
                time.sleep(0.1)  # 队列为空时等待更长时间
            except Exception as e:
                print(f"Execution loop error: {e}")
                time.sleep(1.0)  # 错误时等待更长时间

    def _execute_task(self, execution_id: str) -> None:
        """执行单个任务

        Args:
            execution_id: 执行ID
        """
        if not self.execution_engine:
            return

        # 开始执行
        if self.on_execution_start:
            self.on_execution_start(execution_id)

        success = self.execution_engine.start_execution(execution_id)

        if success:
            self.execution_stats['total_executions'] += 1
            self.execution_stats['successful_executions'] += 1

            # 获取执行摘要
            summary = self.execution_engine.get_execution_summary(execution_id)
        if summary:
            self.execution_stats['total_volume'] += summary.get('filled_quantity', 0)
            self.execution_stats['total_value'] += summary.get(
                'filled_quantity', 0) * summary.get('avg_price', 0)
        else:
            self.execution_stats['total_executions'] += 1
            self.execution_stats['failed_executions'] += 1

        # 执行完成回调
        if self.on_execution_complete:
            self.on_execution_complete(execution_id, success)

    def cancel_execution(self, execution_id: str) -> bool:
        """取消执行任务

        Args:
            execution_id: 执行ID

        Returns:
            是否成功取消
        """
        if not self.execution_engine:
            return False

        return self.execution_engine.cancel_execution(execution_id)

    def get_execution_status(self, execution_id: str) -> Optional[ExecutionState]:
        """获取执行状态

        Args:
            execution_id: 执行ID

        Returns:
            执行状态
        """
        if not self.execution_engine:
            return None

        status = self.execution_engine.get_execution_status(execution_id)
        if status is None:
            return None

        # 转换状态
        status_map = {
            'pending': ExecutionState.IDLE,
            'running': ExecutionState.EXECUTING,
            'completed': ExecutionState.COMPLETED,
            'cancelled': ExecutionState.CANCELLED,
            'failed': ExecutionState.FAILED
        }

        return status_map.get(status.value, ExecutionState.IDLE)

    def get_execution_summary(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取执行摘要

        Args:
            execution_id: 执行ID

        Returns:
            执行摘要
        """
        if not self.execution_engine:
            return None

        return self.execution_engine.get_execution_summary(execution_id)

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计

        Returns:
            执行统计
        """
        return self.execution_stats.copy()

    def get_queue_size(self) -> int:
        """获取队列大小

        Returns:
            队列中的任务数量
        """
        return self._execution_queue.qsize()

    def clear_queue(self) -> None:
        """清空执行队列"""
        while not self._execution_queue.empty():
            try:
                self._execution_queue.get_nowait()
            except queue.Empty:
                break


class AsyncRealTimeExecutor(RealTimeExecutor):

    """异步实时执行器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化异步实时执行器

        Args:
            config: 执行器配置
        """
        super().__init__(config)
        self._loop = None
        self._async_running = False

    async def start_async(self) -> bool:
        """异步启动执行器

        Returns:
            是否成功启动
        """
        if self._async_running:
            return False

        if not self.order_executor or not self.execution_engine:
            return False

        self._async_running = True
        self._loop = asyncio.get_event_loop()

        # 启动异步执行循环
        asyncio.create_task(self._async_execution_loop())

        return True

    async def stop_async(self) -> bool:
        """异步停止执行器

        Returns:
            是否成功停止
        """
        if not self._async_running:
            return False

        self._async_running = False
        return True

    async def _async_execution_loop(self) -> None:
        """异步执行主循环"""
        while self._async_running:
            try:
                # 从队列获取执行任务
                if not self._execution_queue.empty():
                    priority, execution_id = self._execution_queue.get_nowait()
                    await self._async_execute_task(execution_id)

                await asyncio.sleep(0.01)  # 10ms间隔

            except queue.Empty:
                await asyncio.sleep(0.1)  # 队列为空时等待更长时间
            except Exception as e:
                print(f"Async execution loop error: {e}")
                await asyncio.sleep(1.0)  # 错误时等待更长时间

    async def _async_execute_task(self, execution_id: str) -> None:
        """异步执行单个任务

        Args:
            execution_id: 执行ID
        """
        if not self.execution_engine:
            return

        # 开始执行
        if self.on_execution_start:
            self.on_execution_start(execution_id)

        success = self.execution_engine.start_execution(execution_id)

        if success:
            self.execution_stats['total_executions'] += 1
            self.execution_stats['successful_executions'] += 1

            # 获取执行摘要
            summary = self.execution_engine.get_execution_summary(execution_id)
        if summary:
            self.execution_stats['total_volume'] += summary.get('filled_quantity', 0)
            self.execution_stats['total_value'] += summary.get(
                'filled_quantity', 0) * summary.get('avg_price', 0)
        else:
            self.execution_stats['total_executions'] += 1
            self.execution_stats['failed_executions'] += 1

        # 执行完成回调
        if self.on_execution_complete:
            self.on_execution_complete(execution_id, success)
