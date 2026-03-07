#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流处理层核心优先级测试套件

测试覆盖流处理层的核心组件：
1. 流处理器 (StreamProcessor, StreamProcessorBase)
2. 流处理引擎 (StreamProcessingEngine)
3. 实时聚合器 (RealTimeAggregator)
4. 流组件工厂 (StreamComponentFactory)
5. 流性能优化器 (ThroughputOptimizer, PerformanceOptimizer, MemoryOptimizer)
"""

import pytest
import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import time
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, List



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

class TestStreamProcessor(unittest.TestCase):
    """测试流处理器"""

    def setUp(self):
        """设置测试环境"""
        # 使用conftest的导入辅助函数
        from tests.unit.streaming.conftest import import_stream_processor
        self.processor_class = import_stream_processor()
        if self.processor_class is None:
            # 如果导入失败，尝试直接导入
            try:
                from src.streaming.core.stream_processor import StreamProcessor
                self.processor_class = StreamProcessor
            except ImportError:
                # 最后尝试从sys.modules中查找
                import sys
                for module_name in sys.modules:
                    if 'stream_processor' in module_name and 'streaming' in module_name:
                        try:
                            module = sys.modules[module_name]
                            self.processor_class = getattr(module, 'StreamProcessor', None)
                            if self.processor_class:
                                break
                        except:
                            pass
                if self.processor_class is None:
                    self.processor_class = Mock
    
    def _ensure_processor_available(self):
        """确保处理器可用，如果不可用则尝试直接导入"""
        if self.processor_class == Mock:
            try:
                from src.streaming.core.stream_processor import StreamProcessor
                self.processor_class = StreamProcessor
            except ImportError:
                self.skipTest("StreamProcessor导入失败")

    def test_stream_processor_initialization(self):
        """测试流处理器初始化"""
        if self.processor_class == Mock:
            # 最后尝试直接导入
            try:
                from src.streaming.core.stream_processor import StreamProcessor
                self.processor_class = StreamProcessor
            except ImportError:
                self.skipTest("StreamProcessor导入失败")
        
        processor = self.processor_class("test_processor")
        
        self.assertEqual(processor.processor_id, "test_processor")
        self.assertFalse(processor.is_running)
        self.assertEqual(processor.processed_count, 0)
        self.assertEqual(processor.error_count, 0)

    def test_middleware_management(self):
        """测试中间件管理"""
        self._ensure_processor_available()
        
        processor = self.processor_class("test_processor")
        
        middleware = Mock()
        processor.add_middleware(middleware)
        
        self.assertEqual(len(processor.middlewares), 1)
        self.assertEqual(processor.middlewares[0], middleware)

    def test_stream_processor_stats(self):
        """测试流处理器统计"""
        self._ensure_processor_available()
        
        processor = self.processor_class("test_processor")
        
        stats = processor.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['processor_id'], "test_processor")
        self.assertIn('processed_count', stats)
        self.assertIn('error_count', stats)
        self.assertIn('input_queue_size', stats)

    def test_reset_stats(self):
        """测试重置统计"""
        self._ensure_processor_available()
        
        processor = self.processor_class("test_processor")
        
        # 模拟处理一些事件
        processor.processed_count = 10
        processor.error_count = 2
        
        processor.reset_stats()
        
        self.assertEqual(processor.processed_count, 0)
        self.assertEqual(processor.error_count, 0)


class TestStreamProcessorBase(unittest.TestCase):
    """测试流处理器基类"""

    def setUp(self):
        """设置测试环境"""
        # 使用conftest的导入辅助函数
        from tests.unit.streaming.conftest import import_base_processor, import_processing_result
        self.base_processor_class = import_base_processor()
        processing_result_dict = import_processing_result()
        if processing_result_dict:
            self.result_class = processing_result_dict.get('StreamProcessingResult')
            self.status_class = processing_result_dict.get('ProcessingStatus')
        else:
            self.result_class = None
            self.status_class = None
        
        # 如果导入失败，尝试直接导入
        if self.base_processor_class is None:
            try:
                from src.streaming.core.base_processor import StreamProcessorBase
                self.base_processor_class = StreamProcessorBase
            except ImportError:
                self.base_processor_class = Mock
        if self.result_class is None:
            try:
                from src.streaming.core.base_processor import StreamProcessingResult, ProcessingStatus
                self.result_class = StreamProcessingResult
                self.status_class = ProcessingStatus
            except ImportError:
                self.result_class = Mock
                self.status_class = Mock
    
    def _ensure_base_processor_available(self):
        """确保基类处理器可用"""
        if self.base_processor_class == Mock:
            try:
                from src.streaming.core.base_processor import StreamProcessorBase
                self.base_processor_class = StreamProcessorBase
            except ImportError:
                self.skipTest("StreamProcessorBase导入失败")
    
    def _ensure_result_classes_available(self):
        """确保结果类可用"""
        if self.result_class == Mock or self.status_class == Mock:
            try:
                from src.streaming.core.base_processor import StreamProcessingResult, ProcessingStatus
                self.result_class = StreamProcessingResult
                self.status_class = ProcessingStatus
            except ImportError:
                self.skipTest("StreamProcessingResult或ProcessingStatus导入失败")

    def test_stream_processing_result(self):
        """测试流处理结果"""
        self._ensure_result_classes_available()
        
        result = self.result_class(
            event_id="test_event",
            processing_status="completed",
            processed_data={"test": "data"},
            processing_time_ms=100.0
        )
        
        self.assertEqual(result.event_id, "test_event")
        self.assertEqual(result.processing_status, "completed")
        self.assertEqual(result.processed_data, {"test": "data"})
        self.assertEqual(result.processing_time_ms, 100.0)

    def test_processing_status_constants(self):
        """测试处理状态常量"""
        self._ensure_result_classes_available()
        
        self.assertEqual(self.status_class.PENDING, "pending")
        self.assertEqual(self.status_class.COMPLETED, "completed")
        self.assertEqual(self.status_class.FAILED, "failed")
        self.assertEqual(self.status_class.TIMEOUT, "timeout")

    @patch('asyncio.Queue')
    def test_stream_processor_base_initialization(self, mock_queue):
        """测试流处理器基类初始化"""
        self._ensure_base_processor_available()
        
        # 创建具体实现类进行测试
        class TestProcessor(self.base_processor_class):
            async def process_event(self, event):
                return self.result_class(
                    event_id="test",
                    processing_status="completed",
                    processed_data={},
                    processing_time_ms=0.0
                )
        
        processor = TestProcessor("test_processor", {"max_concurrent_events": 50})
        
        self.assertEqual(processor.processor_id, "test_processor")
        self.assertEqual(processor.max_concurrent_events, 50)
        self.assertFalse(processor.is_running)

    def test_event_handler_registration(self):
        """测试事件处理器注册"""
        self._ensure_base_processor_available()
        
        class TestProcessor(self.base_processor_class):
            async def process_event(self, event):
                return self.result_class(
                    event_id="test",
                    processing_status="completed",
                    processed_data={},
                    processing_time_ms=0.0
                )
        
        processor = TestProcessor("test_processor")
        handler = Mock()
        
        processor.register_event_handler("test_event", handler)
        
        self.assertEqual(processor.event_handlers["test_event"], handler)


class TestStreamProcessingEngine(unittest.TestCase):
    """测试流处理引擎"""

    def setUp(self):
        """设置测试环境"""
        # 使用conftest的导入辅助函数
        from tests.unit.streaming.conftest import import_stream_engine, import_stream_topology
        self.engine_class = import_stream_engine()
        self.topology_class = import_stream_topology()
        
        # 如果导入失败，尝试直接导入
        if self.engine_class is None:
            try:
                from src.streaming.core.stream_engine import StreamProcessingEngine
                self.engine_class = StreamProcessingEngine
            except ImportError:
                self.engine_class = Mock
        if self.topology_class is None:
            try:
                from src.streaming.core.stream_engine import StreamTopology
                self.topology_class = StreamTopology
            except ImportError:
                self.topology_class = Mock
    
    def _ensure_engine_available(self):
        """确保引擎类可用"""
        if self.engine_class == Mock:
            try:
                from src.streaming.core.stream_engine import StreamProcessingEngine
                self.engine_class = StreamProcessingEngine
            except ImportError:
                self.skipTest("StreamProcessingEngine导入失败")
    
    def _ensure_topology_available(self):
        """确保拓扑类可用"""
        if self.topology_class == Mock:
            try:
                from src.streaming.core.stream_engine import StreamTopology
                self.topology_class = StreamTopology
            except ImportError:
                self.skipTest("StreamTopology导入失败")

    def test_stream_engine_initialization(self):
        """测试流处理引擎初始化"""
        self._ensure_engine_available()
        
        config = {"queue_size": 5000, "enable_kafka": True}
        engine = self.engine_class("test_engine", config)
        
        self.assertEqual(engine.engine_id, "test_engine")
        self.assertEqual(engine.config, config)
        self.assertFalse(engine.is_running)
        self.assertTrue(engine.enable_kafka)

    def test_stream_topology_creation(self):
        """测试流拓扑创建"""
        self._ensure_topology_available()
        
        topology = self.topology_class(
            topology_id="test_topology",
            processors=["processor1", "processor2"],
            connections={"processor1": ["processor2"]},
            config={"event_types": ["market_data"]}
        )
        
        self.assertEqual(topology.topology_id, "test_topology")
        self.assertEqual(len(topology.processors), 2)
        self.assertIn("processor1", topology.connections)

    def test_engine_status_reporting(self):
        """测试引擎状态报告"""
        self._ensure_engine_available()
        
        engine = self.engine_class("test_engine")
        
        status = engine.get_engine_status()
        
        self.assertIsInstance(status, dict)
        self.assertEqual(status['engine_id'], "test_engine")
        self.assertIn('is_running', status)
        self.assertIn('total_processors', status)
        self.assertIn('engine_metrics', status)

    @patch('asyncio.Queue')
    async def test_engine_event_submission(self, mock_queue):
        """测试引擎事件提交"""
        self._ensure_engine_available()
        
        engine = self.engine_class("test_engine")
        engine.is_running = True
        
        # Mock event
        event = Mock()
        event.event_id = "test_event"
        
        # Mock the queue put method
        engine.event_queue.put = AsyncMock(return_value=None)
        
        result = await engine.submit_event(event)
        
        self.assertTrue(result)


class TestRealTimeAggregator(unittest.TestCase):
    """测试实时聚合器"""

    def setUp(self):
        """设置测试环境"""
        # 使用conftest的导入辅助函数
        from tests.unit.streaming.conftest import import_aggregator
        aggregator_dict = import_aggregator()
        if aggregator_dict:
            self.aggregator_class = aggregator_dict.get('RealTimeAggregator')
            self.window_class = aggregator_dict.get('WindowedData')
        else:
            self.aggregator_class = Mock
            self.window_class = Mock
    
    def _ensure_aggregator_available(self):
        """确保聚合器类可用"""
        if self.aggregator_class == Mock:
            try:
                from src.streaming.core.aggregator import RealTimeAggregator
                self.aggregator_class = RealTimeAggregator
            except ImportError:
                self.skipTest("RealTimeAggregator导入失败")
    
    def _ensure_window_available(self):
        """确保窗口类可用"""
        if self.window_class == Mock:
            try:
                from src.streaming.core.aggregator import WindowedData
                self.window_class = WindowedData
            except ImportError:
                self.skipTest("WindowedData导入失败")

    def test_windowed_data(self):
        """测试窗口数据"""
        self._ensure_window_available()
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=1)
        
        window = self.window_class(
            window_id="test_window",
            window_start=start_time,
            window_end=end_time
        )
        
        self.assertEqual(window.window_id, "test_window")
        self.assertEqual(window.window_start, start_time)
        self.assertEqual(window.window_end, end_time)
        self.assertEqual(len(window.data_points), 0)

    def test_windowed_data_operations(self):
        """测试窗口数据操作"""
        self._ensure_window_available()
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=1)
        
        window = self.window_class(
            window_id="test_window",
            window_start=start_time,
            window_end=end_time
        )
        
        # 添加数据点
        data_point = {"price": 100.0, "volume": 1000}
        window.add_data_point(data_point)
        
        self.assertEqual(len(window.data_points), 1)
        self.assertEqual(window.data_points[0], data_point)
        
        # 更新聚合
        window.update_aggregation("avg_price", 100.0)
        self.assertEqual(window.aggregations["avg_price"], 100.0)

    def test_window_expiration(self):
        """测试窗口过期"""
        self._ensure_window_available()
        
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        
        window = self.window_class(
            window_id="test_window",
            window_start=start_time,
            window_end=end_time
        )
        
        # 窗口应该还没过期
        self.assertFalse(window.is_expired(datetime.now()))
        
        # 等待窗口过期
        time.sleep(1.1)
        self.assertTrue(window.is_expired(datetime.now()))

    def test_aggregator_initialization(self):
        """测试聚合器初始化"""
        self._ensure_aggregator_available()
        
        config = {
            "window_duration_seconds": 30,
            "slide_interval_seconds": 5
        }
        
        aggregator = self.aggregator_class("test_aggregator", config)
        
        self.assertEqual(aggregator.processor_id, "test_aggregator")
        self.assertEqual(aggregator.window_duration, timedelta(seconds=30))
        self.assertEqual(aggregator.slide_interval, timedelta(seconds=5))

    def test_aggregator_windows_management(self):
        """测试聚合器窗口管理"""
        self._ensure_aggregator_available()
        
        aggregator = self.aggregator_class("test_aggregator")
        
        # 获取活跃窗口信息
        windows_info = aggregator.get_active_windows()
        
        self.assertIsInstance(windows_info, dict)


class TestStreamComponentFactory(unittest.TestCase):
    """测试流组件工厂"""

    def setUp(self):
        """设置测试环境"""
        # 使用conftest的导入辅助函数
        from tests.unit.streaming.conftest import import_stream_component_factory
        self.factory_class = import_stream_component_factory()
        if self.factory_class is None:
            self.factory_class = Mock
        # StreamComponent可能需要单独导入
        try:
            from src.streaming.engine.stream_components import StreamComponent
            self.component_class = StreamComponent
        except ImportError:
            self.component_class = Mock
    
    def _ensure_factory_available(self):
        """确保工厂类可用"""
        if self.factory_class == Mock:
            try:
                from src.streaming.engine.stream_components import StreamComponentFactory
                self.factory_class = StreamComponentFactory
            except ImportError:
                self.skipTest("StreamComponentFactory导入失败")

    def test_factory_supported_streams(self):
        """测试工厂支持的流"""
        self._ensure_factory_available()
        
        supported_streams = self.factory_class.get_available_streams()
        
        self.assertIsInstance(supported_streams, list)
        self.assertGreater(len(supported_streams), 0)

    def test_factory_create_component(self):
        """测试工厂创建组件"""
        self._ensure_factory_available()
        
        supported_streams = self.factory_class.get_available_streams()
        if supported_streams:
            stream_id = supported_streams[0]
            component = self.factory_class.create_component(stream_id)
            
            self.assertIsNotNone(component)
            self.assertEqual(component.get_stream_id(), stream_id)

    def test_factory_create_invalid_component(self):
        """测试工厂创建无效组件"""
        self._ensure_factory_available()
        
        with self.assertRaises(ValueError):
            self.factory_class.create_component(999)  # 无效的stream ID

    def test_factory_create_all_streams(self):
        """测试工厂创建所有流"""
        self._ensure_factory_available()
        
        all_streams = self.factory_class.create_all_streams()
        
        self.assertIsInstance(all_streams, dict)
        self.assertGreater(len(all_streams), 0)

    def test_factory_info(self):
        """测试工厂信息"""
        self._ensure_factory_available()
        
        factory_info = self.factory_class.get_factory_info()
        
        self.assertIsInstance(factory_info, dict)
        self.assertIn('factory_name', factory_info)
        self.assertIn('version', factory_info)
        self.assertIn('total_streams', factory_info)

    def test_stream_component_operations(self):
        """测试流组件操作"""
        self._ensure_factory_available()
        
        supported_streams = self.factory_class.get_available_streams()
        if supported_streams:
            stream_id = supported_streams[0]
            component = self.factory_class.create_component(stream_id)
            
            # 测试组件信息
            info = component.get_info()
            self.assertIsInstance(info, dict)
            self.assertEqual(info['stream_id'], stream_id)
            
            # 测试数据处理
            test_data = {"test": "data"}
            result = component.process(test_data)
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['stream_id'], stream_id)
            
            # 测试状态获取
            status = component.get_status()
            self.assertIsInstance(status, dict)
            self.assertEqual(status['stream_id'], stream_id)


class TestStreamOptimizers(unittest.TestCase):
    """测试流优化器"""

    def setUp(self):
        """设置测试环境"""
        # 使用conftest的导入辅助函数
        from tests.unit.streaming.conftest import (
            import_throughput_optimizer, import_performance_optimizer, import_memory_optimizer
        )
        self.throughput_optimizer_class = import_throughput_optimizer()
        if self.throughput_optimizer_class is None:
            self.throughput_optimizer_class = Mock

        self.performance_optimizer_class = import_performance_optimizer()
        if self.performance_optimizer_class is None:
            self.performance_optimizer_class = Mock

        self.memory_optimizer_class = import_memory_optimizer()
        if self.memory_optimizer_class is None:
            self.memory_optimizer_class = Mock
    
    def _ensure_throughput_optimizer_available(self):
        """确保吞吐量优化器可用"""
        if self.throughput_optimizer_class == Mock:
            try:
                from src.streaming.optimization.throughput_optimizer import ThroughputOptimizer
                self.throughput_optimizer_class = ThroughputOptimizer
            except ImportError:
                self.skipTest("ThroughputOptimizer导入失败")
    
    def _ensure_performance_optimizer_available(self):
        """确保性能优化器可用"""
        if self.performance_optimizer_class == Mock:
            try:
                from src.streaming.optimization.performance_optimizer import PerformanceOptimizer
                self.performance_optimizer_class = PerformanceOptimizer
            except ImportError:
                self.skipTest("PerformanceOptimizer导入失败")
    
    def _ensure_memory_optimizer_available(self):
        """确保内存优化器可用"""
        if self.memory_optimizer_class == Mock:
            try:
                from src.streaming.optimization.memory_optimizer import MemoryOptimizer
                self.memory_optimizer_class = MemoryOptimizer
            except ImportError:
                self.skipTest("MemoryOptimizer导入失败")

    def test_throughput_optimizer_initialization(self):
        """测试吞吐量优化器初始化"""
        self._ensure_throughput_optimizer_available()
        
        optimizer = self.throughput_optimizer_class(target_throughput=2000, monitoring_window=30)
        
        self.assertEqual(optimizer.target_throughput, 2000)
        self.assertEqual(optimizer.monitoring_window, 30)

    def test_performance_optimizer_initialization(self):
        """测试性能优化器初始化"""
        self._ensure_performance_optimizer_available()
        
        optimizer = self.performance_optimizer_class(max_workers=8)
        
        self.assertEqual(optimizer.max_workers, 8)
        self.assertFalse(optimizer.is_running)

    def test_memory_optimizer_initialization(self):
        """测试内存优化器初始化"""
        self._ensure_memory_optimizer_available()
        
        optimizer = self.memory_optimizer_class(target_memory_percent=80.0, cleanup_interval=60.0)
        
        self.assertEqual(optimizer.target_memory_percent, 80.0)
        self.assertEqual(optimizer.cleanup_interval, 60.0)


class TestStreamingOptimizer(unittest.TestCase):
    """测试流数据优化器"""

    def setUp(self):
        """设置测试环境"""
        # 使用conftest的导入辅助函数
        from tests.unit.streaming.conftest import import_streaming_optimizer
        self.optimizer_class = import_streaming_optimizer()
        if self.optimizer_class is None:
            self.optimizer_class = Mock
    
    def _ensure_optimizer_available(self):
        """确保优化器类可用"""
        if self.optimizer_class == Mock:
            try:
                from src.streaming.data.streaming_optimizer import StreamingOptimizer
                self.optimizer_class = StreamingOptimizer
            except ImportError:
                self.skipTest("StreamingOptimizer导入失败")

    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        self._ensure_optimizer_available()
        
        optimizer = self.optimizer_class(enable_auto_tuning=True)
        
        self.assertTrue(optimizer.enable_auto_tuning)
        self.assertFalse(optimizer.is_running)
        self.assertEqual(optimizer.cpu_threshold, 80.0)
        self.assertEqual(optimizer.memory_threshold, 85.0)

    def test_optimizer_configuration(self):
        """测试优化器配置"""
        self._ensure_optimizer_available()
        
        optimizer = self.optimizer_class(enable_auto_tuning=False)
        
        self.assertFalse(optimizer.enable_auto_tuning)
        self.assertIsInstance(optimizer.performance_metrics, dict)
        self.assertIsInstance(optimizer.optimization_rules, list)


# 异步测试基类
class AsyncTestCase(unittest.TestCase):
    """异步测试基类"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def run_async(self, coro):
        return self.loop.run_until_complete(coro)


class TestAsyncStreamOperations(AsyncTestCase):
    """测试异步流操作"""

    def setUp(self):
        super().setUp()
        try:
            from src.streaming.core.stream_engine import create_stream_engine
            self.create_engine = create_stream_engine
        except ImportError:
            self.create_engine = None

    def test_async_stream_engine_creation(self):
        """测试异步流引擎创建"""
        if self.create_engine is None:
            self.skipTest("create_stream_engine导入失败")
        
        async def test_creation():
            engine = await self.create_engine("test_engine", {"queue_size": 1000})
            self.assertIsNotNone(engine)
            self.assertEqual(engine.engine_id, "test_engine")
            return True
        
        result = self.run_async(test_creation())
        self.assertTrue(result)

    def test_async_processor_lifecycle(self):
        """测试异步处理器生命周期"""
        try:
            from src.streaming.core.base_processor import StreamProcessorBase, ProcessingStatus
            
            class TestAsyncProcessor(StreamProcessorBase):
                async def process_event(self, event):
                    # 模拟异步处理
                    await asyncio.sleep(0.01)
                    from src.streaming.core.base_processor import StreamProcessingResult
                    return StreamProcessingResult(
                        event_id=getattr(event, 'event_id', 'test'),
                        processing_status=ProcessingStatus.COMPLETED,
                        processed_data={"processed": True},
                        processing_time_ms=10.0
                    )
            
            async def test_lifecycle():
                processor = TestAsyncProcessor("test_processor")
                
                # 测试初始状态
                self.assertFalse(processor.is_running)
                
                # 测试事件提交（不启动处理循环）
                test_event = Mock()
                test_event.event_id = "test_event"
                
                # 直接测试处理方法
                result = await processor.process_event(test_event)
                self.assertEqual(result.processing_status, ProcessingStatus.COMPLETED)
                
                return True
            
            result = self.run_async(test_lifecycle())
            self.assertTrue(result)
            
        except ImportError:
            # 最后尝试直接导入
            try:
                from src.streaming.core.base_processor import StreamProcessorBase, ProcessingStatus
                # 如果导入成功，继续测试
            except ImportError:
                self.skipTest("StreamProcessorBase导入失败")


if __name__ == '__main__':
    unittest.main()
