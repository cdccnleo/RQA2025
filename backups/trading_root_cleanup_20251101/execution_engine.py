# -*- coding: utf-8 -*-
"""
交易层 - 执行引擎
负责订单执行、风险控制和交易流程管理
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from queue import Queue


class ExecutionMode(Enum):
    """执行模式枚举"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    REAL_TIME = "real_time"


class ExecutionStatus(Enum):
    """执行状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ExecutionEngine:
    """执行引擎"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化执行引擎

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.status = ExecutionStatus.IDLE
        self.mode = ExecutionMode.SYNCHRONOUS
        self.logger = logging.getLogger(__name__)

        # 执行队列和线程池
        self.execution_queue: Queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 4))

        # 订单管理
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.completed_orders: Dict[str, Dict[str, Any]] = {}

        # 回调函数
        self.order_callbacks: List[Callable] = []
        self.execution_callbacks: List[Callable] = []

        # 控制标志
        self._running = False
        self._shutdown_event = threading.Event()

    def start(self) -> bool:
        """启动执行引擎

        Returns:
            是否成功启动
        """
        if self.status == ExecutionStatus.RUNNING:
            self.logger.warning("Execution engine is already running")
            return True

        try:
            self._running = True
            self.status = ExecutionStatus.RUNNING
            self._shutdown_event.clear()

            # 启动处理线程
            threading.Thread(target=self._process_queue, daemon=True).start()

            self.logger.info("Execution engine started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start execution engine: {e}")
            self.status = ExecutionStatus.ERROR
            return False

    def stop(self) -> bool:
        """停止执行引擎

        Returns:
            是否成功停止
        """
        if self.status == ExecutionStatus.STOPPED:
            return True

        try:
            self._running = False
            self._shutdown_event.set()
            self.executor.shutdown(wait=True)
            self.status = ExecutionStatus.STOPPED

            self.logger.info("Execution engine stopped")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop execution engine: {e}")
            return False

    def pause(self) -> bool:
        """暂停执行引擎

        Returns:
            是否成功暂停
        """
        if self.status != ExecutionStatus.RUNNING:
            return False

        self.status = ExecutionStatus.PAUSED
        self.logger.info("Execution engine paused")
        return True

    def resume(self) -> bool:
        """恢复执行引擎

        Returns:
            是否成功恢复
        """
        if self.status != ExecutionStatus.PAUSED:
            return False

        self.status = ExecutionStatus.RUNNING
        self.logger.info("Execution engine resumed")
        return True

    def submit_order(self, order: Dict[str, Any]) -> str:
        """提交订单

        Args:
            order: 订单信息

        Returns:
            订单ID
        """
        order_id = f"order_{int(time.time() * 1000)}"

        order_info = {
            "id": order_id,
            "status": "submitted",
            "submitted_at": datetime.now(),
            "execution_mode": self.mode.value,
            **order
        }

        if self.mode == ExecutionMode.SYNCHRONOUS:
            return self._execute_order_synchronously(order_info)
        else:
            self.execution_queue.put(order_info)
            return order_id

    def cancel_order(self, order_id: str) -> bool:
        """取消订单

        Args:
            order_id: 订单ID

        Returns:
            是否成功取消
        """
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            order["status"] = "cancelled"
            order["cancelled_at"] = datetime.now()
            del self.active_orders[order_id]
            self.completed_orders[order_id] = order

            self.logger.info(f"Order cancelled: {order_id}")
            return True

        return False

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单状态信息
        """
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id]
        return None

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """获取活跃订单

        Returns:
            活跃订单列表
        """
        return list(self.active_orders.values())

    def get_completed_orders(self) -> List[Dict[str, Any]]:
        """获取已完成订单

        Returns:
            已完成订单列表
        """
        return list(self.completed_orders.values())

    def set_execution_mode(self, mode: ExecutionMode) -> bool:
        """设置执行模式

        Args:
            mode: 执行模式

        Returns:
            是否成功设置
        """
        if self.status == ExecutionStatus.RUNNING:
            self.logger.warning("Cannot change mode while engine is running")
            return False

        self.mode = mode
        self.logger.info(f"Execution mode set to: {mode.value}")
        return True

    def add_order_callback(self, callback: Callable) -> None:
        """添加订单回调

        Args:
            callback: 回调函数
        """
        self.order_callbacks.append(callback)

    def add_execution_callback(self, callback: Callable) -> None:
        """添加执行回调

        Args:
            callback: 回调函数
        """
        self.execution_callbacks.append(callback)

    def _process_queue(self) -> None:
        """处理执行队列"""
        while not self._shutdown_event.is_set():
            if self.status != ExecutionStatus.RUNNING:
                time.sleep(0.1)
                continue

            try:
                order = self.execution_queue.get(timeout=1.0)
                self.executor.submit(self._execute_order_async, order)
            except:
                continue

    def _execute_order_synchronously(self, order: Dict[str, Any]) -> str:
        """同步执行订单

        Args:
            order: 订单信息

        Returns:
            订单ID
        """
        order_id = order["id"]
        self.active_orders[order_id] = order

        try:
            # 模拟执行逻辑
            time.sleep(0.1)  # 模拟执行时间

            order["status"] = "completed"
            order["completed_at"] = datetime.now()
            order["result"] = {"executed_quantity": order.get("quantity", 0)}

            self.completed_orders[order_id] = order
            del self.active_orders[order_id]

            # 调用回调
            for callback in self.execution_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    self.logger.error(f"Execution callback error: {e}")

        except Exception as e:
            order["status"] = "error"
            order["error"] = str(e)
            self.completed_orders[order_id] = order
            del self.active_orders[order_id]

        return order_id

    def _execute_order_async(self, order: Dict[str, Any]) -> None:
        """异步执行订单

        Args:
            order: 订单信息
        """
        self._execute_order_synchronously(order)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息
        """
        return {
            "status": self.status.value,
            "mode": self.mode.value,
            "active_orders": len(self.active_orders),
            "completed_orders": len(self.completed_orders),
            "queue_size": self.execution_queue.qsize(),
            "total_orders": len(self.active_orders) + len(self.completed_orders)
        }
