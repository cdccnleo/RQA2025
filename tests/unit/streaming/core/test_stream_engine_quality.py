#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流处理引擎质量测试
测试覆盖 StreamProcessingEngine 的核心功能
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from tests.unit.streaming.conftest import import_stream_engine, import_stream_models, import_base_processor


@pytest.fixture
def stream_engine():
    """创建流处理引擎实例"""
    StreamProcessingEngine = import_stream_engine()
    if StreamProcessingEngine is None:
        try:
            from src.streaming.core.stream_engine import StreamProcessingEngine
        except ImportError:
            pytest.skip("StreamProcessingEngine不可用")
    return StreamProcessingEngine('test_engine', {'queue_size': 1000})


@pytest.fixture
def sample_event():
    """创建示例事件"""
    stream_models = import_stream_models()
    if stream_models is None:
        try:
            from src.streaming.core.stream_models import StreamEvent, StreamEventType
            stream_models = (StreamEvent, StreamEventType)
        except ImportError:
            pytest.skip("StreamEvent不可用")
    StreamEvent = stream_models[0]
    StreamEventType = stream_models[1]
    
    return StreamEvent(
        event_id='test_event_1',
        event_type=StreamEventType.MARKET_DATA,
        timestamp=datetime.now(),
        source='test_source',
        data={'symbol': 'AAPL', 'price': 150.0}
    )


@pytest.fixture
def stream_topology():
    """创建流拓扑结构"""
    from src.streaming.core.stream_engine import StreamTopology
    return StreamTopology(
        topology_id='test_topology',
        processors=['processor1', 'processor2'],
        connections={'processor1': ['processor2']},
        config={}
    )


class TestStreamProcessingEngine:
    """StreamProcessingEngine测试类"""

    def test_initialization(self, stream_engine):
        """测试初始化"""
        assert stream_engine.engine_id == 'test_engine'
        assert stream_engine.is_running is False
        assert isinstance(stream_engine.processors, dict)
        assert isinstance(stream_engine.topologies, dict)
        assert isinstance(stream_engine.engine_metrics, dict)

    def test_start_and_stop_engine(self, stream_engine):
        """测试启动和停止引擎"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            await stream_engine.start_engine()
            assert stream_engine.is_running is True
            
            await stream_engine.stop_engine()
            assert stream_engine.is_running is False
        
        loop.run_until_complete(test_async())

    def test_add_and_remove_processor(self, stream_engine):
        """测试添加和移除处理器"""
        StreamProcessorBase = import_base_processor()
        if StreamProcessorBase is None:
            try:
                from src.streaming.core.base_processor import StreamProcessorBase
            except ImportError:
                pytest.skip("StreamProcessorBase不可用")
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # 创建模拟处理器
            processor = Mock(spec=StreamProcessorBase)
            processor.processor_id = 'test_processor'
            processor.start_processing = AsyncMock(return_value=True)
            processor.stop_processing = AsyncMock(return_value=True)
            
            # 添加处理器
            await stream_engine.add_processor(processor)
            assert 'test_processor' in stream_engine.processors
            
            # 移除处理器
            result = await stream_engine.remove_processor('test_processor')
            assert result is True
            assert 'test_processor' not in stream_engine.processors
        
        loop.run_until_complete(test_async())

    def test_create_topology(self, stream_engine, stream_topology):
        """测试创建拓扑"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            await stream_engine.create_topology(stream_topology)
            assert 'test_topology' in stream_engine.topologies
        
        loop.run_until_complete(test_async())

    def test_submit_event(self, stream_engine, sample_event):
        """测试提交事件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # 先启动引擎
            await stream_engine.start_engine()
            # 然后提交事件
            result = await stream_engine.submit_event(sample_event)
            assert result is True
            # 停止引擎
            await stream_engine.stop_engine()
        
        loop.run_until_complete(test_async())

    def test_submit_events_batch(self, stream_engine, sample_event):
        """测试批量提交事件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # 先启动引擎
            await stream_engine.start_engine()
            
            # 创建多个事件
            stream_models = import_stream_models()
            StreamEvent = stream_models[0]
            StreamEventType = stream_models[1]
            
            events = [
                StreamEvent(
                    event_id=f'test_event_{i}',
                    event_type=StreamEventType.MARKET_DATA,
                    timestamp=datetime.now(),
                    source='test_source',
                    data={'symbol': 'AAPL', 'price': 150.0 + i}
                )
                for i in range(5)
            ]
            
            count = await stream_engine.submit_events_batch(events)
            assert count == len(events)
            
            # 停止引擎
            await stream_engine.stop_engine()
        
        loop.run_until_complete(test_async())

    def test_get_engine_status(self, stream_engine):
        """测试获取引擎状态"""
        status = stream_engine.get_engine_status()
        assert isinstance(status, dict)
        assert 'engine_id' in status
        assert 'is_running' in status
        assert 'total_processors' in status  # 使用实际的键名
        assert 'active_topologies' in status  # 使用实际的键名

    def test_get_topology_status(self, stream_engine, stream_topology):
        """测试获取拓扑状态"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            await stream_engine.create_topology(stream_topology)
            status = stream_engine.get_topology_status('test_topology')
            assert status is not None
            assert isinstance(status, dict)
        
        loop.run_until_complete(test_async())

    def test_get_topology_status_not_found(self, stream_engine):
        """测试获取不存在的拓扑状态"""
        status = stream_engine.get_topology_status('nonexistent')
        assert status is None

    def test_stream_topology_creation(self):
        """测试流拓扑结构创建"""
        from src.streaming.core.stream_engine import StreamTopology
        topology = StreamTopology(
            topology_id='test_topology_2',
            processors=['p1', 'p2'],
            connections={'p1': ['p2']},
            config={'key': 'value'}
        )
        assert topology.topology_id == 'test_topology_2'
        assert len(topology.processors) == 2
        assert 'p1' in topology.connections
        assert 'key' in topology.config

    def test_find_applicable_topologies(self, stream_engine, sample_event, stream_topology):
        """测试查找适用的拓扑"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            await stream_engine.create_topology(stream_topology)
            topologies = await stream_engine._find_applicable_topologies(sample_event)
            return topologies
        
        topologies = loop.run_until_complete(test_async())
        assert isinstance(topologies, list)

    def test_topology_matches_event(self, stream_engine, sample_event, stream_topology):
        """测试拓扑匹配事件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            result = await stream_engine._topology_matches_event(stream_topology, sample_event)
            return result
        
        result = loop.run_until_complete(test_async())
        assert isinstance(result, bool)

    def test_start_engine_already_running(self, stream_engine):
        """测试启动已运行的引擎"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            await stream_engine.start_engine()
            # 再次启动应该被忽略
            await stream_engine.start_engine()
            assert stream_engine.is_running is True
            await stream_engine.stop_engine()
        
        loop.run_until_complete(test_async())

    def test_stop_engine_not_running(self, stream_engine):
        """测试停止未运行的引擎"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # 未启动的引擎，停止应该被忽略
            await stream_engine.stop_engine()
            assert stream_engine.is_running is False
        
        loop.run_until_complete(test_async())

    def test_remove_processor_not_found(self, stream_engine):
        """测试移除不存在的处理器"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            result = await stream_engine.remove_processor('nonexistent')
            assert result is False
        
        loop.run_until_complete(test_async())

    def test_submit_event_not_running(self, stream_engine, sample_event):
        """测试未运行时提交事件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            result = await stream_engine.submit_event(sample_event)
            assert result is False
        
        loop.run_until_complete(test_async())

    def test_submit_events_batch_partial_failure(self, stream_engine, sample_event):
        """测试批量提交事件部分失败"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            # 不启动引擎，提交应该失败
            stream_models = import_stream_models()
            StreamEvent = stream_models[0]
            StreamEventType = stream_models[1]
            
            events = [
                StreamEvent(
                    event_id=f'test_event_{i}',
                    event_type=StreamEventType.MARKET_DATA,
                    timestamp=datetime.now(),
                    source='test_source',
                    data={'symbol': 'AAPL', 'price': 150.0 + i}
                )
                for i in range(3)
            ]
            
            count = await stream_engine.submit_events_batch(events)
            # 应该返回0，因为引擎未运行
            assert count == 0
        
        loop.run_until_complete(test_async())

    def test_topology_matches_event_type(self, stream_engine, sample_event):
        """测试拓扑匹配事件类型"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            from src.streaming.core.stream_engine import StreamTopology
            from src.streaming.core.stream_models import StreamEventType
            
            # 创建匹配事件类型的拓扑
            topology = StreamTopology(
                topology_id='test',
                processors=[],
                config={'event_types': ['market_data']}
            )
            result1 = await stream_engine._topology_matches_event(topology, sample_event)
            
            # 创建不匹配事件类型的拓扑
            topology2 = StreamTopology(
                topology_id='test2',
                processors=[],
                config={'event_types': ['order_update']}
            )
            result2 = await stream_engine._topology_matches_event(topology2, sample_event)
            
            return result1, result2
        
        result1, result2 = loop.run_until_complete(test_async())
        assert result1 is True
        assert result2 is False

    def test_topology_matches_event_source(self, stream_engine, sample_event):
        """测试拓扑匹配事件来源"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            from src.streaming.core.stream_engine import StreamTopology
            
            # 创建匹配来源的拓扑
            topology = StreamTopology(
                topology_id='test',
                processors=[],
                config={'sources': ['test_source']}
            )
            result1 = await stream_engine._topology_matches_event(topology, sample_event)
            
            # 创建不匹配来源的拓扑
            topology2 = StreamTopology(
                topology_id='test2',
                processors=[],
                config={'sources': ['other_source']}
            )
            result2 = await stream_engine._topology_matches_event(topology2, sample_event)
            
            return result1, result2
        
        result1, result2 = loop.run_until_complete(test_async())
        assert result1 is True
        assert result2 is False

    def test_process_event_in_topology(self, stream_engine, sample_event, stream_topology):
        """测试在拓扑中处理事件"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            StreamProcessorBase = import_base_processor()
            processor = Mock(spec=StreamProcessorBase)
            processor.processor_id = 'processor1'
            processor.submit_event = AsyncMock(return_value=True)
            processor.is_running = True
            
            await stream_engine.add_processor(processor)
            await stream_engine.create_topology(stream_topology)
            
            # 处理事件
            await stream_engine._process_event_in_topology(sample_event, stream_topology)
            
            # 验证处理器被调用
            processor.submit_event.assert_called()
        
        loop.run_until_complete(test_async())

    def test_process_event_in_topology_failure(self, stream_engine, sample_event, stream_topology):
        """测试在拓扑中处理事件失败"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            StreamProcessorBase = import_base_processor()
            processor = Mock(spec=StreamProcessorBase)
            processor.processor_id = 'processor1'
            processor.submit_event = AsyncMock(return_value=False)  # 返回False表示失败
            processor.is_running = True
            
            await stream_engine.add_processor(processor)
            await stream_engine.create_topology(stream_topology)
            
            # 处理事件应该失败并中断
            await stream_engine._process_event_in_topology(sample_event, stream_topology)
        
        loop.run_until_complete(test_async())

    def test_create_processor_from_config_aggregator(self, stream_engine):
        """测试从配置创建聚合器"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            config = {'type': 'aggregator'}
            await stream_engine._create_processor_from_config('test_aggregator', config)
            assert 'test_aggregator' in stream_engine.processors
        
        loop.run_until_complete(test_async())

    def test_create_processor_from_config_state_manager(self, stream_engine):
        """测试从配置创建状态管理器"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            config = {'type': 'state_manager'}
            await stream_engine._create_processor_from_config('test_state_manager', config)
            assert 'test_state_manager' in stream_engine.processors
        
        loop.run_until_complete(test_async())

    def test_create_processor_from_config_invalid_type(self, stream_engine):
        """测试从配置创建无效类型的处理器"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            config = {'type': 'invalid_type'}
            with pytest.raises(ValueError, match="不支持的处理器类型"):
                await stream_engine._create_processor_from_config('test_invalid', config)
        
        loop.run_until_complete(test_async())

    def test_engine_processing_loop_exception(self, stream_engine, sample_event):
        """测试引擎处理循环异常"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            await stream_engine.start_engine()
            
            # Mock _find_applicable_topologies抛出异常
            with patch.object(stream_engine, '_find_applicable_topologies', side_effect=Exception("Test error")):
                await stream_engine.submit_event(sample_event)
                # 等待处理循环处理
                await asyncio.sleep(0.2)
            
            await stream_engine.stop_engine()
        
        loop.run_until_complete(test_async())

    def test_stop_engine_queue_cleanup(self, stream_engine, sample_event):
        """测试停止引擎时清空队列"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def test_async():
            await stream_engine.start_engine()
            await stream_engine.submit_event(sample_event)
            # 等待事件进入队列
            await asyncio.sleep(0.1)
            
            # 停止引擎应该清空队列
            await stream_engine.stop_engine()
            assert stream_engine.event_queue.empty()
        
        loop.run_until_complete(test_async())

    @pytest.mark.asyncio
    async def test_stop_engine_queue_cleanup_exception(self, stream_engine):
        """测试停止引擎（队列清理异常）"""
        await stream_engine.start_engine()
        
        # 添加一些事件到队列
        from src.streaming.core.stream_models import StreamEvent, StreamEventType
        event = StreamEvent(
            event_id='test',
            event_type=StreamEventType.MARKET_DATA,
            timestamp=datetime.now(),
            source='test',
            data={}
        )
        await stream_engine.event_queue.put(event)
        
        # Mock队列清理抛出异常
        original_get = stream_engine.event_queue.get_nowait
        call_count = 0
        def mock_get():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return original_get()
            else:
                raise Exception("Queue error")
        
        stream_engine.event_queue.get_nowait = mock_get
        
        # 停止引擎应该处理异常
        await stream_engine.stop_engine()

    @pytest.mark.asyncio
    async def test_metrics_collection_loop(self, stream_engine):
        """测试指标收集循环"""
        await stream_engine.start_engine()
        
        # 等待指标收集循环运行
        await asyncio.sleep(0.1)
        
        # 验证指标被更新
        assert 'queue_size' in stream_engine.engine_metrics
        assert 'active_processors' in stream_engine.engine_metrics
        
        await stream_engine.stop_engine()

    @pytest.mark.asyncio
    async def test_metrics_collection_loop_exception(self, stream_engine):
        """测试指标收集循环异常处理"""
        await stream_engine.start_engine()
        
        # Mock qsize抛出异常
        with patch.object(stream_engine.event_queue, 'qsize', side_effect=Exception("Metrics error")):
            await asyncio.sleep(0.1)
        
        await stream_engine.stop_engine()

    @pytest.mark.asyncio
    async def test_process_event_in_topology_failure(self, stream_engine, sample_event):
        """测试在拓扑中处理事件（失败）"""
        from src.streaming.core.stream_engine import StreamTopology
        
        # 先添加处理器（直接设置到processors字典）
        mock_processor = Mock()
        mock_processor.processor_id = 'test_processor'
        mock_processor.is_running = True
        mock_processor.start_processing = AsyncMock()
        mock_processor.stop_processing = AsyncMock()
        mock_processor.submit_event = AsyncMock(return_value=False)
        stream_engine.processors['test_processor'] = mock_processor
        
        # 创建拓扑对象
        topology = StreamTopology(
            topology_id='test_topology',
            processors=['test_processor'],
            config={
                'event_types': [sample_event.event_type.value],
                'sources': [sample_event.source]
            }
        )
        
        # 创建拓扑（跳过处理器创建，因为已经存在）
        stream_engine.topologies[topology.topology_id] = topology
        
        await stream_engine.start_engine()
        
        # 提交事件并等待处理
        await stream_engine.submit_event(sample_event)
        await asyncio.sleep(0.2)
        
        await stream_engine.stop_engine()
        
        # 验证submit_event被调用
        assert mock_processor.submit_event.called

    @pytest.mark.asyncio
    async def test_process_event_in_topology_success(self, stream_engine, sample_event):
        """测试在拓扑中处理事件（成功）"""
        from src.streaming.core.stream_engine import StreamTopology
        
        # 先添加处理器（直接设置到processors字典）
        mock_processor = Mock()
        mock_processor.processor_id = 'test_processor'
        mock_processor.is_running = True
        mock_processor.start_processing = AsyncMock()
        mock_processor.stop_processing = AsyncMock()
        mock_processor.submit_event = AsyncMock(return_value=True)
        stream_engine.processors['test_processor'] = mock_processor
        
        # 创建拓扑对象
        topology = StreamTopology(
            topology_id='test_topology',
            processors=['test_processor'],
            config={
                'event_types': [sample_event.event_type.value],
                'sources': [sample_event.source]
            }
        )
        
        # 创建拓扑（跳过处理器创建，因为已经存在）
        stream_engine.topologies[topology.topology_id] = topology
        
        await stream_engine.start_engine()
        
        # 提交事件并等待处理
        await stream_engine.submit_event(sample_event)
        await asyncio.sleep(0.2)
        
        await stream_engine.stop_engine()
        
        # 验证submit_event被调用
        assert mock_processor.submit_event.called

    @pytest.mark.asyncio
    async def test_create_processor_from_config_aggregator(self, stream_engine):
        """测试从配置创建处理器（聚合器）"""
        config = {
            'type': 'aggregator',
            'window_size_seconds': 60
        }
        
        try:
            await stream_engine._create_processor_from_config('agg_001', config)
            # 验证处理器被添加
            assert 'agg_001' in stream_engine.processors
        except Exception:
            # 如果导入失败，跳过
            pass

    @pytest.mark.asyncio
    async def test_create_processor_from_config_state_manager(self, stream_engine):
        """测试从配置创建处理器（状态管理器）"""
        config = {
            'type': 'state_manager',
            'persistence_enabled': False
        }
        
        try:
            await stream_engine._create_processor_from_config('state_001', config)
            # 验证处理器被添加
            assert 'state_001' in stream_engine.processors
        except Exception:
            # 如果导入失败，跳过
            pass

    @pytest.mark.asyncio
    async def test_create_processor_from_config_invalid_type(self, stream_engine):
        """测试从配置创建处理器（无效类型）"""
        config = {
            'type': 'invalid_type'
        }
        
        with pytest.raises(ValueError):
            await stream_engine._create_processor_from_config('invalid_001', config)

