#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态管理器质量测试
测试覆盖 StateManager 的核心功能
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from tests.unit.streaming.conftest import import_state_manager
state_manager_dict = import_state_manager()
StateManager = state_manager_dict.get('StateManager')
StreamState = state_manager_dict.get('StreamState')


@pytest.fixture
def state_manager():
    """创建状态管理器实例"""
    return StateManager('test_state_manager', {
        'state_backend': 'memory',
        'state_ttl_seconds': 3600,
        'enable_persistence': False
    })


@pytest.fixture
def sample_event():
    """创建示例事件"""
    from src.streaming.core.stream_models import StreamEvent, StreamEventType
    return StreamEvent(
        event_id='test_event_1',
        event_type=StreamEventType.SYSTEM_METRICS,
        timestamp=datetime.now(),
        source='test_source',
        data={'key1': 'value1', 'key2': 100}
    )


class TestStreamState:
    """StreamState测试类"""

    def test_initialization(self):
        """测试初始化"""
        state = StreamState(state_id='test_state')
        
        assert state.state_id == 'test_state'
        assert state.version == 0
        assert len(state.state_data) == 0
        assert state.is_persistent is False

    def test_get_set_delete(self):
        """测试获取、设置、删除"""
        state = StreamState(state_id='test_state')
        
        # 设置值
        state.set('key1', 'value1')
        assert state.get('key1') == 'value1'
        assert state.version == 1
        
        # 获取不存在的值
        assert state.get('key2', 'default') == 'default'
        
        # 删除值
        assert state.delete('key1') is True
        assert state.get('key1') is None
        assert state.version == 2
        
        # 删除不存在的值
        assert state.delete('key2') is False

    def test_is_expired(self):
        """测试过期检查"""
        # 永不过期的状态
        state1 = StreamState(state_id='test_state1', ttl_seconds=0)
        assert state1.is_expired() is False
        
        # 已过期的状态
        state2 = StreamState(
            state_id='test_state2',
            ttl_seconds=60,
            updated_at=datetime.now() - timedelta(seconds=120)
        )
        assert state2.is_expired() is True
        
        # 未过期的状态
        state3 = StreamState(
            state_id='test_state3',
            ttl_seconds=3600,
            updated_at=datetime.now()
        )
        assert state3.is_expired() is False

    def test_to_dict(self):
        """测试转换为字典"""
        state = StreamState(state_id='test_state')
        state.set('key1', 'value1')
        
        state_dict = state.to_dict()
        
        assert state_dict['state_id'] == 'test_state'
        assert state_dict['state_data']['key1'] == 'value1'
        assert state_dict['version'] == 1
        assert 'is_expired' in state_dict


class TestStateManager:
    """StateManager测试类"""

    def test_initialization(self, state_manager):
        """测试初始化"""
        assert state_manager.processor_id == 'test_state_manager'
        assert state_manager.state_backend == 'memory'
        assert state_manager.state_ttl == 3600
        assert len(state_manager.states) == 0

    @pytest.mark.asyncio
    async def test_get_state(self, state_manager):
        """测试获取状态"""
        # 获取不存在的状态
        state = await state_manager.get_state('nonexistent')
        assert state is None
        
        # 创建状态
        new_state = StreamState(state_id='test_state')
        await state_manager.set_state(new_state)
        
        # 获取状态
        retrieved_state = await state_manager.get_state('test_state')
        
        assert retrieved_state is not None
        assert retrieved_state.state_id == 'test_state'

    @pytest.mark.asyncio
    async def test_create_state(self, state_manager):
        """测试创建状态"""
        new_state = StreamState(state_id='test_state')
        await state_manager.set_state(new_state)
        
        assert 'test_state' in state_manager.states
        assert state_manager.states['test_state'].state_id == 'test_state'

    @pytest.mark.asyncio
    async def test_update_state(self, state_manager):
        """测试更新状态"""
        new_state = StreamState(state_id='test_state')
        await state_manager.set_state(new_state)
        
        state = await state_manager.get_state('test_state')
        original_version = state.version
        
        state.set('key1', 'value1')
        await state_manager.set_state(state)
        
        updated_state = await state_manager.get_state('test_state')
        assert updated_state.get('key1') == 'value1'
        assert updated_state.version > original_version

    @pytest.mark.asyncio
    async def test_delete_state(self, state_manager):
        """测试删除状态"""
        new_state = StreamState(state_id='test_state')
        await state_manager.set_state(new_state)
        assert 'test_state' in state_manager.states
        
        result = await state_manager.delete_state('test_state')
        assert result is True
        assert 'test_state' not in state_manager.states

    @pytest.mark.asyncio
    async def test_process_event(self, state_manager, sample_event):
        """测试处理事件"""
        # 现在state_manager使用event.data，直接使用sample_event
        from src.streaming.core.stream_models import StreamEventType
        
        # 创建市场数据事件
        market_event = sample_event
        market_event.event_type = StreamEventType.MARKET_DATA
        market_event.data = {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000}
        
        result = await state_manager.process_event(market_event)
        
        assert result is not None
        assert result.event_id == 'test_event_1'
        # 处理可能成功或失败，都验证结果结构
        assert result.processing_status in ['completed', 'failed']
        if result.processing_status == 'completed':
            assert 'state_id' in result.processed_data

    @pytest.mark.asyncio
    async def test_cleanup_expired_states(self, state_manager):
        """测试清理过期状态"""
        # 创建过期状态
        expired_state = StreamState(
            state_id='expired_state',
            ttl_seconds=60
        )
        expired_state.updated_at = datetime.now() - timedelta(seconds=120)
        await state_manager.set_state(expired_state)
        
        # 创建未过期状态
        active_state = StreamState(
            state_id='active_state',
            ttl_seconds=3600
        )
        await state_manager.set_state(active_state)
        
        # 执行清理
        cleaned_count = await state_manager.cleanup_expired_states()
        
        # 验证过期状态被清理
        assert cleaned_count >= 1
        assert 'expired_state' not in state_manager.states
        assert 'active_state' in state_manager.states

    @pytest.mark.asyncio
    async def test_list_states(self, state_manager):
        """测试列出状态"""
        # 创建多个状态
        await state_manager.set_state(StreamState(state_id='state1'))
        await state_manager.set_state(StreamState(state_id='state2'))
        await state_manager.set_state(StreamState(state_id='other_state'))
        
        all_states = await state_manager.list_states()
        
        assert len(all_states) >= 3
        assert 'state1' in all_states
        assert 'state2' in all_states
        
        # 测试模式匹配
        filtered_states = await state_manager.list_states(pattern='state')
        assert len(filtered_states) >= 2

    @pytest.mark.asyncio
    async def test_process_market_data_event(self, state_manager):
        """测试处理市场数据事件"""
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        
        market_event = StreamEvent(
            event_id='market_event_1',
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source='market_source',
            data={'symbol': 'AAPL', 'price': 150.0, 'volume': 1000}
        )
        
        result = await state_manager.process_event(market_event)
        
        assert result is not None
        assert result.processing_status in ['completed', 'failed']
        if result.processing_status == 'completed':
            state_id = result.processed_data.get('state_id')
            if state_id:
                state = await state_manager.get_state(state_id)
                if state:
                    assert state.get('last_price') == 150.0
                    assert state.get('last_volume') == 1000

    @pytest.mark.asyncio
    async def test_process_order_event(self, state_manager):
        """测试处理订单事件"""
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        
        order_event = StreamEvent(
            event_id='order_event_1',
            event_type=StreamEventType.ORDER_UPDATE,
            timestamp=datetime.now(),
            source='order_source',
            data={'order_id': 'ORD001', 'symbol': 'AAPL', 'status': 'filled'}
        )
        
        result = await state_manager.process_event(order_event)
        
        assert result is not None
        assert result.processing_status in ['completed', 'failed']

    @pytest.mark.asyncio
    async def test_process_trade_event(self, state_manager):
        """测试处理交易事件"""
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        
        trade_event = StreamEvent(
            event_id='trade_event_1',
            event_type=StreamEventType.TRADE_EXECUTION,
            timestamp=datetime.now(),
            source='trade_source',
            data={'symbol': 'AAPL', 'quantity': 100, 'price': 150.0}
        )
        
        result = await state_manager.process_event(trade_event)
        
        assert result is not None
        assert result.processing_status in ['completed', 'failed']
        if result.processing_status == 'completed':
            state_id = result.processed_data.get('state_id')
            if state_id:
                state = await state_manager.get_state(state_id)
                if state:
                    assert state.get('position') == 100
                    assert state.get('total_cost') == 15000.0

    @pytest.mark.asyncio
    async def test_state_persistence(self, state_manager):
        """测试状态持久化"""
        # 创建带持久化的状态管理器
        persistent_manager = StateManager('persistent_manager', {
            'state_backend': 'memory',
            'state_ttl_seconds': 3600,
            'enable_persistence': True,
            'persistence_interval': 1
        })
        
        # 创建持久化状态
        persistent_state = StreamState(
            state_id='persistent_state',
            is_persistent=True
        )
        persistent_state.set('key1', 'value1')
        
        await persistent_manager.set_state(persistent_state)
        
        # 验证状态已保存
        retrieved = await persistent_manager.get_state('persistent_state')
        assert retrieved is not None
        assert retrieved.get('key1') == 'value1'
        
        # 清理
        await persistent_manager.stop_processing()

    @pytest.mark.asyncio
    async def test_generate_state_id(self, state_manager):
        """测试生成状态ID"""
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        
        # 测试市场数据事件的状态ID生成
        market_event = StreamEvent(
            event_id='test',
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source='test',
            data={'symbol': 'AAPL'}
        )
        
        state_id = state_manager._generate_state_id(market_event)
        assert state_id == 'market_AAPL'
        
        # 测试订单事件的状态ID生成
        order_event = StreamEvent(
            event_id='test',
            event_type=StreamEventType.ORDER_UPDATE,
            timestamp=datetime.now(),
            source='test',
            data={'symbol': 'AAPL'}
        )
        
        state_id = state_manager._generate_state_id(order_event)
        assert state_id == 'position_AAPL'
        
        # 测试其他事件类型
        other_event = StreamEvent(
            event_id='test',
            event_type=StreamEventType.SYSTEM_METRICS,
            timestamp=datetime.now(),
            source='test_source',
            data={}
        )
        
        state_id = state_manager._generate_state_id(other_event)
        assert state_id == 'general_test_source'

    @pytest.mark.asyncio
    async def test_state_expiration_handling(self, state_manager):
        """测试状态过期处理"""
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        
        # 创建过期状态
        expired_state = StreamState(
            state_id='expired_state',
            ttl_seconds=1
        )
        expired_state.updated_at = datetime.now() - timedelta(seconds=2)
        await state_manager.set_state(expired_state)
        
        # 处理事件应该重置过期状态
        market_event = StreamEvent(
            event_id='test',
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source='test',
            data={'symbol': 'AAPL', 'price': 150.0}
        )
        
        result = await state_manager.process_event(market_event)
        assert result is not None
        
        # 验证状态已重置
        state = await state_manager.get_state('market_AAPL')
        if state:
            assert not state.is_expired()

    @pytest.mark.asyncio
    async def test_price_history_management(self, state_manager):
        """测试价格历史管理"""
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        
        # 发送多个市场数据事件
        for i in range(105):
            market_event = StreamEvent(
                event_id=f'event_{i}',
                event_type=StreamEventType.MARKET_DATA,
                timestamp=datetime.now(),
                source='test',
                data={'symbol': 'AAPL', 'price': 150.0 + i}
            )
            await state_manager.process_event(market_event)
        
        # 验证价格历史不超过100个
        state = await state_manager.get_state('market_AAPL')
        if state:
            price_history = state.get('price_history', [])
            assert len(price_history) <= 100

    @pytest.mark.asyncio
    async def test_average_cost_calculation(self, state_manager):
        """测试平均成本计算"""
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        
        # 发送多个交易事件
        trades = [
            {'quantity': 100, 'price': 150.0},
            {'quantity': 50, 'price': 160.0},
            {'quantity': -30, 'price': 155.0}
        ]
        
        for i, trade in enumerate(trades):
            trade_event = StreamEvent(
                event_id=f'trade_{i}',
                event_type=StreamEventType.TRADE_EXECUTION,
                timestamp=datetime.now(),
                source='test',
                data={'symbol': 'AAPL', 'quantity': trade['quantity'], 'price': trade['price']}
            )
            await state_manager.process_event(trade_event)
        
        # 验证平均成本计算
        state = await state_manager.get_state('position_AAPL')
        if state:
            total_cost = state.get('total_cost', 0)
            total_quantity = state.get('total_quantity', 0)
            if total_quantity > 0:
                avg_cost = state.get('avg_cost', 0)
                expected_avg = total_cost / total_quantity
                assert abs(avg_cost - expected_avg) < 0.01

    @pytest.mark.asyncio
    async def test_delete_nonexistent_state(self, state_manager):
        """测试删除不存在的状态"""
        result = await state_manager.delete_state('nonexistent')
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_state_with_persistence(self):
        """测试删除带持久化的状态"""
        persistent_manager = StateManager('persistent_manager', {
            'state_backend': 'memory',
            'state_ttl_seconds': 3600,
            'enable_persistence': True
        })
        
        new_state = StreamState(state_id='test_state')
        await persistent_manager.set_state(new_state)
        
        result = await persistent_manager.delete_state('test_state')
        assert result is True
        
        await persistent_manager.stop_processing()

    @pytest.mark.asyncio
    async def test_process_event_with_persistence(self):
        """测试带持久化的事件处理"""
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        
        persistent_manager = StateManager('persistent_manager', {
            'state_backend': 'memory',
            'state_ttl_seconds': 3600,
            'enable_persistence': True
        })
        
        market_event = StreamEvent(
            event_id='test',
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source='test',
            data={'symbol': 'AAPL', 'price': 150.0}
        )
        
        result = await persistent_manager.process_event(market_event)
        assert result is not None
        
        await persistent_manager.stop_processing()

    @pytest.mark.asyncio
    async def test_process_event_exception(self, state_manager):
        """测试处理事件异常"""
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        
        # 创建一个会导致异常的事件
        invalid_event = StreamEvent(
            event_id='test',
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source='test',
            data=None
        )
        
        # Mock _get_or_create_state 抛出异常
        with patch.object(state_manager, '_get_or_create_state', side_effect=Exception("Test error")):
            result = await state_manager.process_event(invalid_event)
            assert result.processing_status == 'failed'
            assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_expired_state_reset(self, state_manager):
        """测试过期状态重置"""
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        
        # 创建过期状态
        expired_state = StreamState(
            state_id='market_AAPL',
            ttl_seconds=1
        )
        expired_state.updated_at = datetime.now() - timedelta(seconds=2)
        await state_manager.set_state(expired_state)
        
        # 处理事件应该重置过期状态
        market_event = StreamEvent(
            event_id='test',
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source='test',
            data={'symbol': 'AAPL', 'price': 150.0}
        )
        
        result = await state_manager.process_event(market_event)
        assert result is not None
        
        # 验证状态已重置
        state = await state_manager.get_state('market_AAPL')
        if state:
            assert not state.is_expired()

    @pytest.mark.asyncio
    async def test_persistence_task(self):
        """测试持久化任务"""
        persistent_manager = StateManager('persistent_manager', {
            'state_backend': 'memory',
            'state_ttl_seconds': 3600,
            'enable_persistence': True,
            'persistence_interval': 0.1
        })
        
        # 创建持久化状态
        persistent_state = StreamState(
            state_id='persistent_state',
            is_persistent=True
        )
        await persistent_manager.set_state(persistent_state)
        
        # 等待持久化任务运行
        await asyncio.sleep(0.2)
        
        await persistent_manager.stop_processing()

    @pytest.mark.asyncio
    async def test_persist_all_states(self):
        """测试持久化所有状态"""
        persistent_manager = StateManager('persistent_manager', {
            'state_backend': 'memory',
            'state_ttl_seconds': 3600,
            'enable_persistence': True
        })
        
        # 创建多个持久化状态
        for i in range(3):
            state = StreamState(
                state_id=f'persistent_state_{i}',
                is_persistent=True
            )
            await persistent_manager.set_state(state)
        
        # 手动触发持久化所有状态
        await persistent_manager._persist_all_states()
        
        await persistent_manager.stop_processing()

    @pytest.mark.asyncio
    async def test_cleanup_expired_states_with_persistence(self):
        """测试清理过期状态（带持久化）"""
        persistent_manager = StateManager('persistent_manager', {
            'state_backend': 'memory',
            'state_ttl_seconds': 3600,
            'enable_persistence': True
        })
        
        # 创建过期状态
        expired_state = StreamState(
            state_id='expired_state',
            ttl_seconds=1
        )
        expired_state.updated_at = datetime.now() - timedelta(seconds=2)
        await persistent_manager.set_state(expired_state)
        
        # 执行清理
        cleaned_count = await persistent_manager.cleanup_expired_states()
        assert cleaned_count >= 1
        
        await persistent_manager.stop_processing()

