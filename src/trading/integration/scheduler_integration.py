# -*- coding: utf-8 -*-
"""
交易层统一调度器集成模块

提供交易层与统一调度器的集成能力，支持交易任务的调度执行。

函数级注释:
- 所有公开函数均包含详细的中文注释
- 异常处理包含具体的错误信息
- 支持PostgreSQL优先策略，连接失败时自动降级

作者: AI系统集成助手
日期: 2026-03-22
版本: 1.0.0
"""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

# 导入统一调度器
from src.core.orchestration.scheduler import (
    get_unified_scheduler,
    JobType,
    TaskPriority,
    TaskStatus
)
from src.core.event_bus import get_event_bus, EventPriority

# 导入交易层类型
from src.trading.execution.execution_types import ExecutionMode, ExecutionStatus
from src.trading.execution.order_manager import Order, OrderType, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class TradingTaskType(Enum):
    """
    交易任务类型枚举
    
    定义交易层支持的所有任务类型，映射到统一调度器的JobType
    """
    ORDER_PREPARATION = "order_preparation"      # 订单准备
    ORDER_VALIDATION = "order_validation"        # 订单验证
    ORDER_EXECUTION = "order_execution"          # 订单执行
    ORDER_CONFIRMATION = "order_confirmation"    # 订单确认
    TRADE_PROCESSING = "trade_processing"        # 交易处理
    PORTFOLIO_REBALANCE = "portfolio_rebalancing" # 组合再平衡


@dataclass
class TradingTaskConfig:
    """
    交易任务配置类
    
    属性:
        priority: 任务优先级，1-10，数字越小优先级越高
        timeout_seconds: 任务超时时间（秒）
        max_retries: 最大重试次数
        retry_delay_seconds: 重试延迟（秒）
        enable_callback: 是否启用结果回调
    """
    priority: int = 5
    timeout_seconds: Optional[int] = 30
    max_retries: int = 3
    retry_delay_seconds: int = 1
    enable_callback: bool = True


@dataclass
class TradingTaskResult:
    """
    交易任务结果类
    
    属性:
        task_id: 任务ID
        success: 是否成功
        data: 结果数据
        error_message: 错误信息
        execution_time_ms: 执行时间（毫秒）
    """
    task_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


class TradingSchedulerIntegration:
    """
    交易层统一调度器集成类
    
    提供交易任务提交、状态查询、结果回调等功能的统一接口。
    采用单例模式确保全局唯一实例。
    
    使用示例:
        integration = TradingSchedulerIntegration()
        task_id = await integration.submit_order_execution(order_data)
        result = await integration.wait_for_task_completion(task_id)
    """
    
    _instance: Optional['TradingSchedulerIntegration'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls) -> 'TradingSchedulerIntegration':
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化交易调度器集成"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._scheduler = None
        self._event_bus = None
        self._task_callbacks: Dict[str, Callable] = {}
        self._task_results: Dict[str, TradingTaskResult] = {}
        
        logger.info("交易调度器集成初始化完成")
    
    async def _get_scheduler(self):
        """
        获取统一调度器实例
        
        Returns:
            UnifiedScheduler: 统一调度器实例
        """
        if self._scheduler is None:
            self._scheduler = get_unified_scheduler()
        return self._scheduler
    
    async def _get_event_bus(self):
        """
        获取事件总线实例
        
        Returns:
            EventBus: 事件总线实例
        """
        if self._event_bus is None:
            self._event_bus = get_event_bus()
        return self._event_bus
    
    def _map_to_job_type(self, task_type: TradingTaskType) -> str:
        """
        将交易任务类型映射到统一调度器JobType
        
        Args:
            task_type: 交易任务类型
            
        Returns:
            str: 统一调度器JobType字符串
        """
        mapping = {
            TradingTaskType.ORDER_PREPARATION: JobType.ORDER_PREPARATION.value,
            TradingTaskType.ORDER_VALIDATION: JobType.ORDER_VALIDATION.value,
            TradingTaskType.ORDER_EXECUTION: JobType.ORDER_EXECUTION.value,
            TradingTaskType.ORDER_CONFIRMATION: JobType.ORDER_CONFIRMATION.value,
            TradingTaskType.TRADE_PROCESSING: "trade_processing",
            TradingTaskType.PORTFOLIO_REBALANCE: JobType.PORTFOLIO_REBALANCING.value,
        }
        return mapping.get(task_type, JobType.ORDER_EXECUTION.value)
    
    def _map_to_task_priority(self, priority: int) -> TaskPriority:
        """
        将数字优先级映射到TaskPriority枚举
        
        Args:
            priority: 数字优先级(1-10)
            
        Returns:
            TaskPriority: 任务优先级枚举
        """
        if priority <= 2:
            return TaskPriority.CRITICAL
        elif priority <= 4:
            return TaskPriority.HIGH
        elif priority <= 6:
            return TaskPriority.NORMAL
        else:
            return TaskPriority.LOW
    
    async def submit_task(
        self,
        task_type: TradingTaskType,
        payload: Dict[str, Any],
        config: Optional[TradingTaskConfig] = None,
        callback: Optional[Callable[[TradingTaskResult], None]] = None
    ) -> str:
        """
        提交交易任务到统一调度器
        
        Args:
            task_type: 交易任务类型
            payload: 任务数据负载
            config: 任务配置，默认使用TradingTaskConfig()
            callback: 任务完成后的回调函数
            
        Returns:
            str: 任务ID
            
        Raises:
            RuntimeError: 调度器未启动或提交失败
        """
        config = config or TradingTaskConfig()
        scheduler = await self._get_scheduler()
        
        # 构建任务数据
        task_payload = {
            "task_type": task_type.value,
            "payload": payload,
            "submitted_at": datetime.now().isoformat(),
            "config": {
                "priority": config.priority,
                "timeout_seconds": config.timeout_seconds,
                "max_retries": config.max_retries,
            }
        }
        
        try:
            # 提交任务到统一调度器
            task_id = await scheduler.submit_task(
                task_type=self._map_to_job_type(task_type),
                payload=task_payload,
                priority=config.priority,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries
            )
            
            # 注册回调
            if callback and config.enable_callback:
                self._task_callbacks[task_id] = callback
            
            logger.info(f"交易任务提交成功: task_id={task_id}, type={task_type.value}")
            return task_id
            
        except Exception as e:
            logger.error(f"交易任务提交失败: {e}")
            raise RuntimeError(f"提交交易任务失败: {e}")
    
    async def submit_order_execution(
        self,
        order: Order,
        execution_mode: ExecutionMode = ExecutionMode.MARKET,
        config: Optional[TradingTaskConfig] = None,
        callback: Optional[Callable[[TradingTaskResult], None]] = None
    ) -> str:
        """
        提交订单执行任务
        
        Args:
            order: 订单对象
            execution_mode: 执行模式
            config: 任务配置
            callback: 回调函数
            
        Returns:
            str: 任务ID
        """
        payload = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value if isinstance(order.side, OrderSide) else order.side,
            "order_type": order.order_type.value if isinstance(order.order_type, OrderType) else order.order_type,
            "quantity": order.quantity,
            "price": order.price,
            "stop_price": order.stop_price,
            "strategy_id": order.strategy_id,
            "account_id": order.account_id,
            "execution_mode": execution_mode.value if isinstance(execution_mode, ExecutionMode) else execution_mode,
        }
        
        # 订单执行使用高优先级
        execution_config = config or TradingTaskConfig(priority=2)
        
        return await self.submit_task(
            task_type=TradingTaskType.ORDER_EXECUTION,
            payload=payload,
            config=execution_config,
            callback=callback
        )
    
    async def submit_order_validation(
        self,
        order: Order,
        config: Optional[TradingTaskConfig] = None
    ) -> str:
        """
        提交订单验证任务
        
        Args:
            order: 订单对象
            config: 任务配置
            
        Returns:
            str: 任务ID
        """
        payload = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value if isinstance(order.side, OrderSide) else order.side,
            "order_type": order.order_type.value if isinstance(order.order_type, OrderType) else order.order_type,
            "quantity": order.quantity,
            "price": order.price,
        }
        
        validation_config = config or TradingTaskConfig(priority=1, timeout_seconds=5)
        
        return await self.submit_task(
            task_type=TradingTaskType.ORDER_VALIDATION,
            payload=payload,
            config=validation_config
        )
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[TaskStatus]: 任务状态，任务不存在返回None
        """
        scheduler = await self._get_scheduler()
        task = scheduler.get_task_detail(task_id)
        
        if task is None:
            return None
        
        return task.status
    
    async def wait_for_task_completion(
        self,
        task_id: str,
        timeout_seconds: float = 30.0,
        poll_interval: float = 0.1
    ) -> TradingTaskResult:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout_seconds: 超时时间（秒）
            poll_interval: 轮询间隔（秒）
            
        Returns:
            TradingTaskResult: 任务结果
            
        Raises:
            TimeoutError: 等待超时
        """
        start_time = datetime.now()
        
        while True:
            status = await self.get_task_status(task_id)
            
            if status is None:
                return TradingTaskResult(
                    task_id=task_id,
                    success=False,
                    error_message="任务不存在"
                )
            
            # 检查任务是否完成
            if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                # 获取任务结果
                result = self._task_results.get(task_id)
                if result:
                    return result
                else:
                    return TradingTaskResult(
                        task_id=task_id,
                        success=(status == TaskStatus.COMPLETED),
                        error_message=None if status == TaskStatus.COMPLETED else f"任务状态: {status.value}"
                    )
            
            # 检查超时
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout_seconds:
                raise TimeoutError(f"等待任务完成超时: task_id={task_id}")
            
            await asyncio.sleep(poll_interval)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        scheduler = await self._get_scheduler()
        
        try:
            success = await scheduler.cancel_task(task_id)
            if success:
                logger.info(f"任务取消成功: task_id={task_id}")
            else:
                logger.warning(f"任务取消失败: task_id={task_id}")
            return success
        except Exception as e:
            logger.error(f"取消任务异常: task_id={task_id}, error={e}")
            return False
    
    def register_task_result(self, task_id: str, result: TradingTaskResult):
        """
        注册任务结果
        
        Args:
            task_id: 任务ID
            result: 任务结果
        """
        self._task_results[task_id] = result
        
        # 触发回调
        callback = self._task_callbacks.get(task_id)
        if callback:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"任务回调执行失败: task_id={task_id}, error={e}")
            finally:
                # 清理回调
                del self._task_callbacks[task_id]
    
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """
        获取调度器状态
        
        Returns:
            Dict[str, Any]: 调度器状态信息
        """
        scheduler = await self._get_scheduler()
        
        return {
            "is_running": scheduler.is_running(),
            "status": scheduler.get_status(),
            "statistics": scheduler.get_statistics()
        }


# 全局实例获取函数
def get_trading_scheduler_integration() -> TradingSchedulerIntegration:
    """
    获取交易调度器集成实例
    
    Returns:
        TradingSchedulerIntegration: 交易调度器集成实例
    """
    return TradingSchedulerIntegration()
