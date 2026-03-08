# -*- coding: utf-8 -*-
"""
RQA2025 流处理层状态管理器
Stream Processing Layer State Manager

负责维护和管理流处理的状态信息。
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .base_processor import StreamProcessorBase, StreamProcessingResult, ProcessingStatus
from .stream_models import StreamEvent

# 获取统一基础设施集成层的日志适配器
try:
    from src.infrastructure.integration import get_models_adapter
    models_adapter = get_models_adapter()
    from src.infrastructure.logging.core.interfaces import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class StreamState:

    """流处理状态"""
    state_id: str
    state_data: Dict[str, Any] = field(default_factory=dict)
    version: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 3600
    is_persistent: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        return self.state_data.get(key, default)

    def set(self, key: str, value: Any):
        """设置状态值"""
        self.state_data[key] = value
        self.updated_at = datetime.now()
        self.version += 1

    def delete(self, key: str) -> bool:
        """删除状态值"""
        if key in self.state_data:
            del self.state_data[key]
            self.updated_at = datetime.now()
            self.version += 1
            return True
        return False

    def is_expired(self) -> bool:
        """检查状态是否过期"""
        if self.ttl_seconds <= 0:
            return False

        return (datetime.now() - self.updated_at).total_seconds() > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'state_id': self.state_id,
            'state_data': self.state_data,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'is_persistent': self.is_persistent,
            'is_expired': self.is_expired()
        }


class StateManager(StreamProcessorBase):

    """
    状态管理器
    负责维护和管理流处理的状态信息
    """

    def __init__(self, processor_id: str, config: Optional[Dict[str, Any]] = None):

        super().__init__(processor_id, config)

        # 状态存储配置
        self.state_backend = self.config.get('state_backend', 'memory')  # memory, redis, etc.
        self.state_ttl = self.config.get('state_ttl_seconds', 3600)

        # 状态存储
        self.states: Dict[str, StreamState] = {}

        # 状态持久化
        self.enable_persistence = self.config.get('enable_persistence', False)
        self.persistence_interval = self.config.get('persistence_interval', 300)  # 5分钟

        # 启动持久化任务
        if self.enable_persistence:
            asyncio.create_task(self._start_persistence_task())

        logger.info(f"状态管理器 {processor_id} 已初始化")

    async def process_event(self, event: StreamEvent) -> StreamProcessingResult:
        """处理事件并更新状态"""
        start_time = datetime.now()

        try:
            # 获取状态
            state = await self._get_or_create_state(event)

            # 更新状态
            await self._update_state_from_event(state, event)

            # 检查状态一致性
            await self._validate_state_consistency(state)

            # 持久化状态
            if self.enable_persistence:
                await self._persist_state(state)

            return StreamProcessingResult(
                event_id=event.event_id,
                processing_status=ProcessingStatus.COMPLETED,
                processed_data={
                    'state_id': state.state_id,
                    'state_version': state.version,
                    'updated_fields': list(event.data.keys()) if event.data else []
                },
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

        except Exception as e:
            return StreamProcessingResult(
                event_id=event.event_id,
                processing_status=ProcessingStatus.FAILED,
                processed_data={},
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )

    async def get_state(self, state_id: str) -> Optional[StreamState]:
        """获取状态"""
        return self.states.get(state_id)

    async def set_state(self, state: StreamState):
        """设置状态"""
        self.states[state.state_id] = state

        if self.enable_persistence:
            await self._persist_state(state)

    async def delete_state(self, state_id: str) -> bool:
        """删除状态"""
        if state_id in self.states:
            del self.states[state_id]

            if self.enable_persistence:
                await self._delete_persisted_state(state_id)

            return True
        return False

    async def list_states(self, pattern: Optional[str] = None) -> List[str]:
        """列出状态ID"""
        state_ids = list(self.states.keys())

        if pattern:
            # 简单的模式匹配
            state_ids = [sid for sid in state_ids if pattern in sid]

        return state_ids

    async def cleanup_expired_states(self) -> int:
        """清理过期状态"""
        expired_states = []

        for state_id, state in self.states.items():
            if state.is_expired():
                expired_states.append(state_id)

        # 删除过期状态
        for state_id in expired_states:
            del self.states[state_id]
            if self.enable_persistence:
                await self._delete_persisted_state(state_id)

        logger.info(f"清理了 {len(expired_states)} 个过期状态")
        return len(expired_states)

    async def _get_or_create_state(self, event: StreamEvent) -> StreamState:
        """获取或创建状态"""
        # 基于事件数据生成状态ID
        state_id = self._generate_state_id(event)

        if state_id not in self.states:
            self.states[state_id] = StreamState(
                state_id=state_id,
                ttl_seconds=self.state_ttl
            )

        state = self.states[state_id]

        # 检查状态是否过期
        if state.is_expired():
            # 重置过期状态
            state = StreamState(
                state_id=state_id,
                ttl_seconds=self.state_ttl
            )
            self.states[state_id] = state

        return state

    async def _update_state_from_event(self, state: StreamState, event: StreamEvent):
        """从事件更新状态"""
        # 根据事件类型更新状态
        if event.event_type.value == 'market_data':
            await self._update_market_data_state(state, event)
        elif event.event_type.value == 'order_update':
            await self._update_order_state(state, event)
        elif event.event_type.value == 'trade_execution':
            await self._update_trade_state(state, event)

    async def _update_market_data_state(self, state: StreamState, event: StreamEvent):
        """更新市场数据状态"""
        # 更新最新价格和成交量
        state.set('last_price', event.data.get('price') if event.data else None)
        state.set('last_volume', event.data.get('volume') if event.data else None)
        state.set('last_update', event.timestamp.isoformat())

        # 更新价格历史（保留最近100个价格）
        price_history = state.get('price_history', [])
        if event.data and 'price' in event.data:
            price_history.append(event.data.get('price'))
        if len(price_history) > 100:
            price_history = price_history[-100:]
        state.set('price_history', price_history)

    async def _update_order_state(self, state: StreamState, event: StreamEvent):
        """更新订单状态"""
        order_id = event.data.get('order_id') if event.data else None
        if order_id:
            state.set(f'order_{order_id}_status', event.data)

    async def _update_trade_state(self, state: StreamState, event: StreamEvent):
        """更新交易状态"""
        # 更新持仓信息
        quantity = event.data.get('quantity') if event.data else 0
        price = event.data.get('price') if event.data else 0

        current_position = state.get('position', 0)
        state.set('position', current_position + quantity)

        # 更新平均成本
        total_cost = state.get('total_cost', 0)
        total_quantity = state.get('total_quantity', 0)

        new_total_cost = total_cost + (quantity * price)
        new_total_quantity = total_quantity + abs(quantity)

        state.set('total_cost', new_total_cost)
        state.set('total_quantity', new_total_quantity)

        if new_total_quantity > 0:
            avg_cost = new_total_cost / new_total_quantity
            state.set('avg_cost', avg_cost)

    async def _validate_state_consistency(self, state: StreamState):
        """验证状态一致性"""
        # 检查必需的字段
        required_fields = ['last_update']
        for req_field in required_fields:
            if req_field not in state.state_data:
                logger.warning(f"状态 {state.state_id} 缺少必需字段: {req_field}")

    async def _start_persistence_task(self):
        """启动持久化任务"""
        while self.is_running:
            try:
                await asyncio.sleep(self.persistence_interval)
                await self._persist_all_states()
            except Exception as e:
                logger.error(f"状态持久化任务异常: {str(e)}")

    async def _persist_state(self, state: StreamState):
        """持久化状态"""
        # 这里可以实现状态的持久化逻辑
        # 例如：保存到Redis、数据库等
        if state.is_persistent:
            logger.debug(f"持久化状态: {state.state_id}")

    async def _persist_all_states(self):
        """持久化所有状态"""
        persistent_states = [s for s in self.states.values() if s.is_persistent]
        logger.info(f"持久化 {len(persistent_states)} 个状态")

        for state in persistent_states:
            await self._persist_state(state)

    async def _delete_persisted_state(self, state_id: str):
        """删除持久化状态"""
        # 删除持久化存储中的状态
        logger.debug(f"删除持久化状态: {state_id}")

    def _generate_state_id(self, event: StreamEvent) -> str:
        """生成状态ID"""
        # 基于事件类型和关键字段生成状态ID
        if event.event_type.value == 'market_data':
            symbol = event.data.get('symbol', 'unknown') if event.data else 'unknown'
            return f"market_{symbol}"
        elif event.event_type.value in ['order_update', 'trade_execution']:
            symbol = event.data.get('symbol', 'unknown') if event.data else 'unknown'
            return f"position_{symbol}"
        else:
            return f"general_{event.source}"


__all__ = [
    'StateManager',
    'StreamState'
]
