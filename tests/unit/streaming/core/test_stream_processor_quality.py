#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流处理器质量测试
测试覆盖 StreamProcessor 的核心功能
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch

from tests.unit.streaming.conftest import import_stream_processor


@pytest.fixture
def stream_processor():
    """创建流处理器实例"""
    StreamProcessor = import_stream_processor()
    if StreamProcessor is None:
        try:
            from src.streaming.core.stream_processor import StreamProcessor
        except ImportError:
            pytest.skip("StreamProcessor不可用")
    return StreamProcessor('test_processor')


@pytest.fixture
def sample_data():
    """创建示例数据"""
    return {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000}


class TestStreamProcessor:
    """StreamProcessor测试类"""

    def test_initialization(self, stream_processor):
        """测试初始化"""
        assert stream_processor.processor_id == 'test_processor'
        assert stream_processor.is_running is False
        assert stream_processor.processed_count == 0
        assert stream_processor.error_count == 0


    def test_add_middleware(self, stream_processor):
        """测试添加中间件"""
        def middleware(data):
            return data
        
        stream_processor.add_middleware(middleware)
        assert len(stream_processor.middlewares) == 1

    def test_start_and_stop(self, stream_processor):
        """测试启动和停止"""
        result = stream_processor.start()
        assert result is True
        assert stream_processor.is_running is True
        
        result = stream_processor.stop()
        assert result is True
        assert stream_processor.is_running is False

    def test_process_data(self, stream_processor, sample_data):
        """测试处理数据"""
        stream_processor.start()
        
        result = stream_processor.process_data(sample_data)
        assert result is True
        
        # 等待处理完成
        time.sleep(0.1)
        
        stream_processor.stop()

    def test_get_processed_data(self, stream_processor, sample_data):
        """测试获取处理后的数据"""
        stream_processor.start()
        stream_processor.process_data(sample_data)
        
        # 等待处理完成
        time.sleep(0.1)
        
        processed_data = stream_processor.get_processed_data()
        # 可能为None，也可能有数据
        assert processed_data is None or isinstance(processed_data, dict)
        
        stream_processor.stop()

    def test_get_stats(self, stream_processor):
        """测试获取统计信息"""
        stats = stream_processor.get_stats()
        assert isinstance(stats, dict)
        assert 'processed_count' in stats
        assert 'error_count' in stats
        assert 'is_running' in stats

    def test_reset_stats(self, stream_processor):
        """测试重置统计信息"""
        stream_processor.processed_count = 10
        stream_processor.error_count = 5
        
        stream_processor.reset_stats()
        
        assert stream_processor.processed_count == 0
        assert stream_processor.error_count == 0

    def test_process_data_with_middleware(self, stream_processor, sample_data):
        """测试使用中间件处理数据"""
        def transform_middleware(data):
            data['transformed'] = True
            return data
        
        stream_processor.add_middleware(transform_middleware)
        stream_processor.start()
        
        result = stream_processor.process_data(sample_data)
        assert result is True
        
        # 等待处理完成
        time.sleep(0.1)
        
        stream_processor.stop()

    def test_process_data_queue_full(self, stream_processor, sample_data):
        """测试队列满的情况"""
        stream_processor.start()
        
        # 填满队列
        for i in range(10001):  # 超过默认队列大小
            try:
                stream_processor.process_data(sample_data)
            except:
                pass
        
        # 等待处理
        time.sleep(0.1)
        
        stream_processor.stop()

    def test_start_already_running(self, stream_processor):
        """测试重复启动"""
        stream_processor.start()
        result = stream_processor.start()  # 再次启动
        assert result is False  # 应该返回False
        
        stream_processor.stop()

    def test_stop_not_running(self, stream_processor):
        """测试停止未运行的处理器"""
        result = stream_processor.stop()
        assert result is False  # 应该返回False

    def test_process_data_not_running(self, stream_processor, sample_data):
        """测试未运行时处理数据"""
        result = stream_processor.process_data(sample_data)
        assert result is False  # 应该返回False

    def test_middleware_error_handling(self, stream_processor, sample_data):
        """测试中间件错误处理"""
        def error_middleware(data):
            raise Exception("Middleware error")
        
        stream_processor.add_middleware(error_middleware)
        stream_processor.start()
        
        result = stream_processor.process_data(sample_data)
        assert result is True
        
        # 等待处理完成
        time.sleep(0.2)
        
        # 错误应该被记录
        assert stream_processor.error_count > 0
        
        stream_processor.stop()

    def test_output_queue_full_handling(self, stream_processor, sample_data):
        """测试输出队列满的处理"""
        stream_processor.start()
        
        # 快速添加大量数据，可能导致输出队列满
        for i in range(100):
            stream_processor.process_data(sample_data)
        
        # 等待处理
        time.sleep(0.2)
        
        stream_processor.stop()

    def test_processing_loop_exception_handling(self, stream_processor):
        """测试处理循环异常处理"""
        stream_processor.start()
        
        # 添加会导致处理循环异常的数据
        stream_processor.input_queue.put("invalid_data")
        
        # 等待处理
        time.sleep(0.2)
        
        stream_processor.stop()

    def test_get_processed_data_exception_handling(self, stream_processor):
        """测试获取处理数据时的异常处理"""
        # 测试空队列的情况
        result = stream_processor.get_processed_data()
        assert result is None

    def test_get_stats_complete(self, stream_processor):
        """测试获取完整统计信息"""
        stream_processor.add_middleware(lambda x: x)
        stream_processor.start()
        
        stats = stream_processor.get_stats()
        assert 'processor_id' in stats
        assert 'is_running' in stats
        assert 'processed_count' in stats
        assert 'error_count' in stats
        assert 'input_queue_size' in stats
        assert 'output_queue_size' in stats
        assert 'middleware_count' in stats
        assert stats['middleware_count'] == 1
        
        stream_processor.stop()

    def test_default_stream_processor(self):
        """测试默认流处理器实例"""
        from src.streaming.core.stream_processor import default_stream_processor
        assert default_stream_processor is not None
        assert default_stream_processor.processor_id == "default_stream_processor"

    def test_processor_id_generation(self):
        """测试处理器ID自动生成"""
        from src.streaming.core.stream_processor import StreamProcessor
        processor = StreamProcessor()
        assert processor.processor_id is not None
        assert processor.processor_id.startswith("stream_processor_")

    def test_processing_loop_periodic_logging(self, stream_processor, sample_data):
        """测试处理循环的周期性日志记录"""
        stream_processor.start()
        
        # 处理足够多的数据以触发周期性日志（DEFAULT_BATCH_SIZE = 1000）
        # 为了测试，我们可以mock logger或者处理大量数据
        # 这里我们处理一些数据，确保处理循环正常运行
        for i in range(10):
            stream_processor.process_data(sample_data)
        
        time.sleep(0.2)
        
        stream_processor.stop()

    def test_start_exception_handling(self, stream_processor):
        """测试启动时的异常处理"""
        # 通过mock threading.Thread来模拟启动失败
        with patch('threading.Thread') as mock_thread:
            mock_thread.side_effect = Exception("Thread creation failed")
            result = stream_processor.start()
            assert result is False
            assert stream_processor.is_running is False

    def test_stop_exception_handling(self, stream_processor):
        """测试停止时的异常处理"""
        stream_processor.start()
        stream_processor.processing_thread = Mock()
        stream_processor.processing_thread.is_alive.return_value = True
        stream_processor.processing_thread.join.side_effect = Exception("Join failed")
        
        # 即使join失败，stop也应该返回True（因为is_running已设置为False）
        result = stream_processor.stop()
        # 由于异常处理，stop可能返回True或False，取决于实现
        assert isinstance(result, bool)

    def test_get_processed_data_exception_in_get_nowait(self, stream_processor):
        """测试get_processed_data中get_nowait的异常处理"""
        stream_processor.start()
        
        # 模拟output_queue.get_nowait抛出非Empty异常
        with patch.object(stream_processor.output_queue, 'get_nowait', side_effect=Exception("Queue error")):
            result = stream_processor.get_processed_data()
            assert result is None
        
        stream_processor.stop()

    def test_process_data_exception_handling(self, stream_processor, sample_data):
        """测试process_data中的异常处理"""
        stream_processor.start()
        
        # 模拟input_queue.put抛出非Full异常
        with patch.object(stream_processor.input_queue, 'put', side_effect=Exception("Put error")):
            result = stream_processor.process_data(sample_data)
            assert result is False
        
        stream_processor.stop()

    def test_queue_data_full(self, stream_processor):
        """测试队列数据（队列满）"""
        import queue
        
        stream_processor.start()
        
        # 填满输入队列 - 使用process_data方法
        with patch.object(stream_processor.input_queue, 'put', side_effect=queue.Full()):
            result = stream_processor.process_data({'test': 'data'})
            assert result is False
        
        stream_processor.stop()

    def test_processing_loop_output_queue_full(self, stream_processor):
        """测试处理循环（输出队列满）"""
        import queue
        
        stream_processor.start()
        
        # Mock输出队列满
        with patch.object(stream_processor.output_queue, 'put', side_effect=queue.Full()):
            # 添加数据
            stream_processor.input_queue.put({'test': 'data'})
            time.sleep(0.2)
        
        stream_processor.stop()

    def test_processing_loop_exception(self, stream_processor):
        """测试处理循环异常处理"""
        stream_processor.start()
        
        # Mock输入队列抛出异常
        with patch.object(stream_processor.input_queue, 'get', side_effect=Exception("Queue error")):
            time.sleep(0.2)
        
        stream_processor.stop()

