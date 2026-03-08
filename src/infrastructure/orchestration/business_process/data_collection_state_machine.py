#!/usr/bin/env python3
"""
数据采集状态机管理器

提供复杂的状态机管理能力，支持：
1. 多分支状态转换
2. 条件判断和分支逻辑
3. 状态持久化和恢复
4. 事件驱动的状态转换
5. 超时和重试机制
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
import json
import threading

# 简单状态机实现，避免复杂的依赖


logger = logging.getLogger(__name__)

class SimpleStateMachine:
    """简化的状态机实现"""

    def __init__(self, initial_state):
        self.current_state = initial_state
        self.state_history = []
        self.state_enter_time = datetime.now()

    def transition_to(self, new_state, context=None):
        """状态转换"""
        old_state = self.current_state
        self.current_state = new_state
        self.state_enter_time = datetime.now()
        self.state_history.append({
            'from': old_state,
            'to': new_state,
            'timestamp': self.state_enter_time,
            'context': context
        })
        logger.info(f"状态转换: {old_state} -> {new_state}")
        return True

    def get_current_state(self):
        return self.current_state

    def get_state_history(self):
        return self.state_history
from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class DataCollectionState(Enum):
    """数据采集流程状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    SCHEDULING = "scheduling"
    COLLECTING = "collecting"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    STANDARDIZING = "standardizing"
    STORING = "storing"
    CATALOGING = "cataloging"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    ROLLING_BACK = "rolling_back"
    CANCELLED = "cancelled"


class DataCollectionEvent(Enum):
    """数据采集事件"""
    START_COLLECTION = "start_collection"
    SCHEDULE_NEXT = "schedule_next"
    COLLECTION_SUCCESS = "collection_success"
    COLLECTION_FAILED = "collection_failed"
    VALIDATION_SUCCESS = "validation_success"
    VALIDATION_FAILED = "validation_failed"
    PREPROCESS_SUCCESS = "preprocess_success"
    PREPROCESS_FAILED = "preprocess_failed"
    STANDARDIZE_SUCCESS = "standardize_success"
    STANDARDIZE_FAILED = "standardize_failed"
    STORAGE_SUCCESS = "storage_success"
    STORAGE_FAILED = "storage_failed"
    CATALOG_SUCCESS = "catalog_success"
    CATALOG_FAILED = "catalog_failed"
    RETRY_EXHAUSTED = "retry_exhausted"
    TIMEOUT = "timeout"
    CANCEL = "cancel"
    FORCE_COMPLETE = "force_complete"


class StateMachineManager:
    """状态机管理器"""

    def __init__(self, workflow_id: str, config: Optional[Dict[str, Any]] = None):
        self.workflow_id = workflow_id
        self.config = config or {}

        # 初始化状态机
        self.state_machine = SimpleStateMachine(DataCollectionState.IDLE)
        self._setup_state_transitions()

        # 状态历史
        self.state_history: List[Dict[str, Any]] = []
        self.state_entry_times: Dict[DataCollectionState, datetime] = {}

        # 配置参数
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout_seconds = self.config.get('timeout_seconds', 300)
        self.retry_delay = self.config.get('retry_delay', 30)

        # 统计信息
        self.retry_count = 0
        self.start_time = None
        self.end_time = None

        # 锁
        self._lock = threading.RLock()

        # 回调函数
        self.state_change_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        self.timeout_callbacks: List[Callable] = []

        # 超时任务
        self.timeout_task: Optional[asyncio.Task] = None

    def _setup_state_transitions(self):
        """设置状态转换规则"""
        # 简化实现，不设置复杂的转换规则
        pass

    def can_trigger_event(self, event: DataCollectionEvent) -> bool:
        """检查是否可以触发指定事件"""
        return self.state_machine.can_trigger(event)

    def get_available_events(self) -> List[DataCollectionEvent]:
        """获取当前可用的状态转换事件"""
        return self.state_machine.get_available_events()

    def get_current_state(self) -> DataCollectionState:
        """获取当前状态"""
        return self.state_machine.current_state

    def is_completed(self) -> bool:
        """检查是否已完成"""
        return self.state_machine.current_state in [DataCollectionState.COMPLETED, DataCollectionState.FAILED, DataCollectionState.CANCELLED]

    def is_successful(self) -> bool:
        """检查是否成功完成"""
        return self.state_machine.current_state == DataCollectionState.COMPLETED

    def get_workflow_stats(self) -> Dict[str, Any]:
        """获取工作流程统计信息"""
        with self._lock:
            if not self.start_time:
                return {}

            end_time = self.end_time or datetime.now()
            duration = (end_time - self.start_time).total_seconds()

            return {
                'workflow_id': self.workflow_id,
                'current_state': self.state_machine.current_state.value,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': duration,
                'retry_count': self.retry_count,
                'state_changes': len(self.state_history),
                'state_history': self.state_history[-10:]  # 只返回最近10个状态变化
            }

    def add_state_change_callback(self, callback: Callable):
        """添加状态变化回调函数"""
        self.state_change_callbacks.append(callback)

    def add_error_callback(self, callback: Callable):
        """添加错误回调函数"""
        self.error_callbacks.append(callback)

    def add_timeout_callback(self, callback: Callable):
        """添加超时回调函数"""
        self.timeout_callbacks.append(callback)

    async def cancel_workflow(self, reason: str = "用户取消") -> bool:
        """取消工作流程"""
        return await self.trigger_event(DataCollectionEvent.CANCEL, {'reason': reason})

    async def force_complete_workflow(self, reason: str = "强制完成") -> bool:
        """强制完成工作流程"""
        return await self.trigger_event(DataCollectionEvent.FORCE_COMPLETE, {'reason': reason})

    def _record_state_change(self, old_state: DataCollectionState,
                           event: DataCollectionEvent,
                           event_data: Optional[Dict[str, Any]] = None):
        """记录状态变化"""
        timestamp = datetime.now()

        # 记录离开时间
        if old_state in self.state_entry_times:
            exit_time = self.state_entry_times[old_state]
            duration_in_state = (timestamp - exit_time).total_seconds()
        else:
            duration_in_state = 0

        # 记录进入时间
        self.state_entry_times[self.state_machine.current_state] = timestamp

        # 添加到历史记录
        state_record = {
            'timestamp': timestamp.isoformat(),
            'old_state': old_state.value,
            'new_state': self.state_machine.current_state.value,
            'event': event.value,
            'event_data': event_data,
            'duration_in_old_state': duration_in_state
        }

        self.state_history.append(state_record)

    async def _notify_state_change(self, old_state: DataCollectionState,
                                 new_state: DataCollectionState,
                                 event: DataCollectionEvent,
                                 event_data: Optional[Dict[str, Any]] = None):
        """通知状态变化"""
        for callback in self.state_change_callbacks:
            try:
                await callback(self.workflow_id, old_state, new_state, event, event_data)
            except Exception as e:
                logger.error(f"状态变化回调函数执行失败: {e}")

    async def _handle_failure(self, event_data: Optional[Dict[str, Any]] = None):
        """处理失败情况"""
        self.end_time = datetime.now()

        # 调用错误回调
        for callback in self.error_callbacks:
            try:
                await callback(self.workflow_id, event_data)
            except Exception as e:
                logger.error(f"错误回调函数执行失败: {e}")

    async def _handle_completion(self, event_data: Optional[Dict[str, Any]] = None):
        """处理完成情况"""
        self.end_time = datetime.now()

    async def _handle_error(self, error: Exception, event_data: Optional[Dict[str, Any]] = None):
        """处理错误"""
        logger.error(f"工作流程 {self.workflow_id} 发生错误: {error}")

        # 调用错误回调
        for callback in self.error_callbacks:
            try:
                await callback(self.workflow_id, error, event_data)
            except Exception as e:
                logger.error(f"错误回调函数执行失败: {e}")

    async def _monitor_timeout(self):
        """监控超时"""
        try:
            await asyncio.sleep(self.timeout_seconds)

            # 如果还没有完成，触发超时事件
            if not self.is_completed():
                logger.warning(f"工作流程 {self.workflow_id} 超时")
                await self.trigger_event(DataCollectionEvent.TIMEOUT)

                # 调用超时回调
                for callback in self.timeout_callbacks:
                    try:
                        await callback(self.workflow_id)
                    except Exception as e:
                        logger.error(f"超时回调函数执行失败: {e}")

        except asyncio.CancelledError:
            # 任务被取消，正常退出
            pass

    def cleanup(self):
        """清理资源"""
        if self.timeout_task and not self.timeout_task.done():
            self.timeout_task.cancel()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'workflow_id': self.workflow_id,
            'current_state': self.state_machine.current_state.value,
            'config': self.config,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'retry_count': self.retry_count,
            'state_history': self.state_history[-50:]  # 只保存最近50个状态变化
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateMachineManager':
        """从字典反序列化"""
        manager = cls(data['workflow_id'], data.get('config', {}))

        # 恢复状态
        if 'current_state' in data:
            manager.state_machine.current_state = DataCollectionState(data['current_state'])

        # 恢复时间信息
        if data.get('start_time'):
            manager.start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            manager.end_time = datetime.fromisoformat(data['end_time'])

        # 恢复统计信息
        manager.retry_count = data.get('retry_count', 0)
        manager.state_history = data.get('state_history', [])

        return manager
