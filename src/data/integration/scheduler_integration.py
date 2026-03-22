# -*- coding: utf-8 -*-
"""
数据管理层统一调度器集成模块

提供数据管理层与统一调度器的集成能力，支持数据采集、处理、验证任务的调度执行。

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
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field

# 导入统一调度器
from src.core.orchestration.scheduler import (
    get_unified_scheduler,
    JobType,
    TaskPriority,
    TaskStatus,
    TriggerType
)
from src.core.event_bus import get_event_bus, EventPriority

logger = logging.getLogger(__name__)


class DataTaskType(Enum):
    """
    数据任务类型枚举
    
    定义数据管理层支持的所有任务类型，映射到统一调度器的JobType
    """
    DATA_COLLECTION = "data_collection"          # 数据采集
    DATA_CLEANING = "data_cleaning"              # 数据清洗
    DATA_VALIDATION = "data_validation"          # 数据验证
    DATA_PROCESSING = "data_processing"          # 数据处理
    DATA_QUALITY_CHECK = "data_quality_check"    # 数据质量检查
    DATA_BACKUP = "data_backup"                  # 数据备份
    DATA_SYNC = "data_sync"                      # 数据同步


@dataclass
class DataTaskConfig:
    """
    数据任务配置类
    
    属性:
        priority: 任务优先级，1-10，数字越小优先级越高
        timeout_seconds: 任务超时时间（秒）
        max_retries: 最大重试次数
        retry_delay_seconds: 重试延迟（秒）
        enable_callback: 是否启用结果回调
        trigger_type: 触发类型（interval、cron、date、once）
        trigger_config: 触发器配置
    """
    priority: int = 5
    timeout_seconds: Optional[int] = 300  # 数据任务默认5分钟超时
    max_retries: int = 3
    retry_delay_seconds: int = 5
    enable_callback: bool = True
    trigger_type: Optional[str] = None
    trigger_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataTaskResult:
    """
    数据任务结果类
    
    属性:
        task_id: 任务ID
        success: 是否成功
        data: 结果数据
        error_message: 错误信息
        execution_time_ms: 执行时间（毫秒）
        records_processed: 处理记录数
    """
    task_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    records_processed: int = 0


class DataSchedulerIntegration:
    """
    数据管理层统一调度器集成类
    
    提供数据任务提交、状态查询、结果回调等功能的统一接口。
    采用单例模式确保全局唯一实例。
    
    使用示例:
        integration = DataSchedulerIntegration()
        
        # 提交采集任务
        task_id = await integration.submit_collection_task(
            symbols=["000001.SZ", "000002.SZ"],
            data_source="tushare"
        )
        
        # 创建定时采集任务
        job_id = await integration.create_scheduled_collection(
            symbols=["000001.SZ"],
            cron="0 9 * * *"  # 每天9点执行
        )
    """
    
    _instance: Optional['DataSchedulerIntegration'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls) -> 'DataSchedulerIntegration':
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化数据调度器集成"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._scheduler = None
        self._event_bus = None
        self._task_callbacks: Dict[str, Callable] = {}
        self._task_results: Dict[str, DataTaskResult] = {}
        self._active_collectors: Dict[str, Any] = {}
        
        logger.info("数据调度器集成初始化完成")
    
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
    
    def _map_to_job_type(self, task_type: DataTaskType) -> str:
        """
        将数据任务类型映射到统一调度器JobType
        
        Args:
            task_type: 数据任务类型
            
        Returns:
            str: 统一调度器JobType字符串
        """
        mapping = {
            DataTaskType.DATA_COLLECTION: JobType.DATA_COLLECTION.value,
            DataTaskType.DATA_CLEANING: JobType.DATA_CLEANING.value,
            DataTaskType.DATA_VALIDATION: JobType.DATA_VALIDATION.value,
            DataTaskType.DATA_PROCESSING: "data_processing",
            DataTaskType.DATA_QUALITY_CHECK: "data_quality_check",
            DataTaskType.DATA_BACKUP: "data_backup",
            DataTaskType.DATA_SYNC: "data_sync",
        }
        return mapping.get(task_type, JobType.DATA_COLLECTION.value)
    
    async def submit_task(
        self,
        task_type: DataTaskType,
        payload: Dict[str, Any],
        config: Optional[DataTaskConfig] = None,
        callback: Optional[Callable[[DataTaskResult], None]] = None
    ) -> str:
        """
        提交数据任务到统一调度器
        
        Args:
            task_type: 数据任务类型
            payload: 任务数据负载
            config: 任务配置，默认使用DataTaskConfig()
            callback: 任务完成后的回调函数
            
        Returns:
            str: 任务ID
            
        Raises:
            RuntimeError: 调度器未启动或提交失败
        """
        config = config or DataTaskConfig()
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
            
            logger.info(f"数据任务提交成功: task_id={task_id}, type={task_type.value}")
            return task_id
            
        except Exception as e:
            logger.error(f"数据任务提交失败: {e}")
            raise RuntimeError(f"提交数据任务失败: {e}")
    
    async def submit_collection_task(
        self,
        symbols: List[str],
        data_source: str = "tushare",
        data_type: str = "daily_price",
        config: Optional[DataTaskConfig] = None,
        callback: Optional[Callable[[DataTaskResult], None]] = None
    ) -> str:
        """
        提交数据采集任务
        
        Args:
            symbols: 标的代码列表
            data_source: 数据源名称
            data_type: 数据类型
            config: 任务配置
            callback: 回调函数
            
        Returns:
            str: 任务ID
        """
        payload = {
            "symbols": symbols,
            "data_source": data_source,
            "data_type": data_type,
            "symbol_count": len(symbols),
        }
        
        # 采集任务使用普通优先级
        collection_config = config or DataTaskConfig(priority=5)
        
        return await self.submit_task(
            task_type=DataTaskType.DATA_COLLECTION,
            payload=payload,
            config=collection_config,
            callback=callback
        )
    
    async def submit_validation_task(
        self,
        data_source: str,
        validation_rules: List[str],
        config: Optional[DataTaskConfig] = None
    ) -> str:
        """
        提交数据验证任务
        
        Args:
            data_source: 数据源名称
            validation_rules: 验证规则列表
            config: 任务配置
            
        Returns:
            str: 任务ID
        """
        payload = {
            "data_source": data_source,
            "validation_rules": validation_rules,
        }
        
        # 验证任务使用较高优先级
        validation_config = config or DataTaskConfig(priority=3)
        
        return await self.submit_task(
            task_type=DataTaskType.DATA_VALIDATION,
            payload=payload,
            config=validation_config
        )
    
    async def create_scheduled_collection(
        self,
        symbols: List[str],
        cron: str,
        data_source: str = "tushare",
        name: Optional[str] = None
    ) -> str:
        """
        创建定时采集任务
        
        Args:
            symbols: 标的代码列表
            cron: Cron表达式
            data_source: 数据源名称
            name: 任务名称
            
        Returns:
            str: 定时任务ID
        """
        scheduler = await self._get_scheduler()
        
        job_name = name or f"collection_{data_source}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        payload = {
            "symbols": symbols,
            "data_source": data_source,
            "symbol_count": len(symbols),
        }
        
        try:
            job_id = await scheduler.create_job(
                name=job_name,
                job_type=JobType.DATA_COLLECTION.value,
                trigger_type=TriggerType.CRON.value,
                trigger_config={"cron": cron},
                config=payload
            )
            
            logger.info(f"定时采集任务创建成功: job_id={job_id}, name={job_name}")
            return job_id
            
        except Exception as e:
            logger.error(f"定时采集任务创建失败: {e}")
            raise RuntimeError(f"创建定时采集任务失败: {e}")
    
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
        timeout_seconds: float = 300.0,
        poll_interval: float = 1.0
    ) -> DataTaskResult:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout_seconds: 超时时间（秒）
            poll_interval: 轮询间隔（秒）
            
        Returns:
            DataTaskResult: 任务结果
            
        Raises:
            TimeoutError: 等待超时
        """
        start_time = datetime.now()
        
        while True:
            status = await self.get_task_status(task_id)
            
            if status is None:
                return DataTaskResult(
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
                    return DataTaskResult(
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
    
    def register_task_result(self, task_id: str, result: DataTaskResult):
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
    
    async def publish_data_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ):
        """
        发布数据事件到事件总线
        
        Args:
            event_type: 事件类型
            data: 事件数据
            correlation_id: 关联ID
        """
        event_bus = await self._get_event_bus()
        
        event_data = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "source": "data.integration",
            "version": "1.0",
            "payload": data
        }
        
        await event_bus.publish(
            event_type=event_type,
            data=event_data,
            priority=EventPriority.NORMAL
        )
        
        logger.debug(f"数据事件已发布: event_type={event_type}")
    
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
def get_data_scheduler_integration() -> DataSchedulerIntegration:
    """
    获取数据调度器集成实例
    
    Returns:
        DataSchedulerIntegration: 数据调度器集成实例
    """
    return DataSchedulerIntegration()
